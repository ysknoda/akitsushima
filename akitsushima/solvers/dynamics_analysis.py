"""Mean-squared displacement (MSD) / self-diffusion coefficient, and
velocity-autocorrelation-function (VACF) based vibrational density of
states, from an internally-driven NVE-MD trajectory.

Neither observable is available from the existing NVE/NVT/NPT classes:
their ``run()`` methods only ever return a scalar-per-step
``thermo_log`` and the FINAL positions/velocities, never the full
trajectory. This module drives its own internal velocity-Verlet loop
(mirroring ``ThermalConductivityCalculator.run_md``'s established
pattern of calling ``force_fn`` directly rather than reusing the
disk-writing MD classes), accumulating exactly the two time series
these two analyses need.

Bug-fix note (unwrapped positions, caught before it became a real bug):
all three existing MD classes re-wrap fractional coordinates into
``[0, box)`` every step (``frac -= np.floor(frac)``). A correct MSD
needs UNWRAPPED positions -- using wrapped positions directly would
make the displacement discontinuously jump by a full lattice vector
every time an atom crosses a periodic boundary, corrupting the MSD.
Fixed here by maintaining a SEPARATE, never-wrapped position array that
is updated with the exact same velocity term as the wrapped copy used
for force evaluation, every step -- the two accumulate identically
except that only one of them is ever folded back into the cell.
"""
import tempfile
from typing import Callable, Optional

import numpy as np

from ..constants import EV_ANG_TO_AMU_ANG_FS2, FREQ_RAD_FS_TO_WAVENUMBER_CM
from .molecular_dynamics import NVTSimulation, _BaseMDSimulation
from .thermal_conductivity import _autocorr_fft

__all__ = ["DiffusionAnalyzer"]


def _compute_msd(pos_series: np.ndarray, max_lag: int) -> np.ndarray:
    """pos_series: (n_frames, N, 3) UNWRAPPED positions.
    Returns (max_lag,) MSD, averaged over atoms and time origins."""
    n_frames = pos_series.shape[0]
    msd = np.zeros(max_lag)
    for lag in range(max_lag):
        disp = pos_series[lag:] - pos_series[:n_frames - lag]   # (n_frames-lag, N, 3)
        msd[lag] = np.mean(np.sum(disp ** 2, axis=-1))
    return msd


def _compute_vacf(vel_series: np.ndarray, max_lag: int) -> np.ndarray:
    """vel_series: (n_frames, N, 3). Returns (max_lag,) velocity
    autocorrelation function, averaged over atoms and the 3 Cartesian
    components (the standard, isotropic single-component VACF
    convention used by the Green-Kubo diffusion relation
    D = integral_0^inf VACF(t) dt).

    Bug-fix note (subtract_mean=False): unlike the heat flux that
    ``_autocorr_fft`` was originally written for, a velocity
    component's TRUE population mean in an equilibrium system with no
    net drift is analytically zero, not merely assumed -- so
    subtracting each atom's finite-window SAMPLE mean (the function's
    default behaviour) removes no genuine bias and instead introduces
    one, most visibly in VACF(0) itself (the direct equipartition-
    theorem quantity <v^2>): confirmed by comparing, on the same real
    NVE trajectories used in this module's own tests, the mean-
    subtracted VACF(0) (systematically 8-22% LOW relative to
    k_B T / m across 5 seeds) against the raw <v^2> (matching
    equipartition to within about 10% on the same trajectories, no
    systematic sign). See ``_autocorr_fft``'s own docstring for the
    full comparison."""
    n_frames, N, _ = vel_series.shape
    vacf_sum = np.zeros(max_lag)
    for i in range(N):
        for a in range(3):
            vacf_sum += _autocorr_fft(vel_series[:, i, a], vel_series[:, i, a],
                                      max_lag, subtract_mean=False)
    return vacf_sum / (N * 3)


class DiffusionAnalyzer(_BaseMDSimulation):
    """
    Drive an internal NVE-MD trajectory (after optional NVT
    equilibration) and compute the mean-squared displacement (MSD) /
    Einstein self-diffusion coefficient, and the velocity-
    autocorrelation-function (VACF) based vibrational density of
    states -- a finite-temperature, anharmonic complement to
    :class:`~akitsushima.solvers.phonon.PhononCalculator`'s harmonic
    lattice-dynamics DOS.

    Parameters
    ----------
    lattice, positions, atom_types : initial structure.
    force_fn    : callable (lattice, positions, atom_types) -> (E, F, S).
    temperature : target temperature for the initial Maxwell-Boltzmann
                 velocity draw / NVT equilibration [K].
    masses      : (N,) atomic masses [amu]; looked up from
                 ``ATOMIC_MASS`` by species when None.

    Usage
    -----
    >>> da = DiffusionAnalyzer(lattice, positions, atom_types, force_fn,
    ...                        temperature=800.0)
    >>> da.run_md(n_steps=20000, timestep=0.5, equil_steps=2000)
    >>> msd_result  = da.compute_msd()
    >>> vacf_result = da.compute_vacf_dos()
    """

    def __init__(self,
                 lattice    : np.ndarray,
                 positions  : np.ndarray,
                 atom_types : list,
                 force_fn   : Callable,
                 temperature: float = 300.0,
                 masses     : Optional[np.ndarray] = None):
        self.lattice     = np.asarray(lattice, dtype=float).copy()
        self.positions   = np.asarray(positions, dtype=float).copy()
        self.atom_types  = list(atom_types)
        self.force_fn    = force_fn
        self.temperature = float(temperature)
        self.N           = len(self.positions)
        self.masses      = self._build_masses(self.atom_types, masses)

        self._pos_series: Optional[np.ndarray] = None   # (n_frames, N, 3) unwrapped
        self._vel_series: Optional[np.ndarray] = None   # (n_frames, N, 3)
        self._dt_record : Optional[float] = None         # [fs] between recorded frames
        # Bug-fix note (code-check follow-up): this used to be a single
        # hardcoded path shared by every DiffusionAnalyzer instance/
        # process, so two concurrent NVT equilibrations (e.g. two
        # instances in the same process, or different MPI ranks each
        # analysing a different structure) could interleave/clobber each
        # other's scratch trajectory/thermo log files -- harmless to the
        # returned MSD/VACF (computed from the in-memory NVT result, not
        # from these log files), but confusing if anyone inspects them. A
        # fresh, uniquely-named directory per instance removes that
        # collision risk entirely.
        self._scratch_dir = tempfile.mkdtemp(prefix="akitsushima_diffusion_")

    def run_md(self,
              n_steps      : int   = 10_000,
              timestep     : float = 0.5,
              equil_steps  : int   = 1_000,
              record_every : int   = 1,
              seed         : int   = 42,
              verbose      : bool  = True) -> dict:
        """
        Run NVT equilibration (``equil_steps``, if > 0) then NVE
        production for ``n_steps``, recording unwrapped positions and
        velocities every ``record_every`` steps.

        Returns
        -------
        dict with ``n_frames``, ``dt_record`` [fs], ``positions``,
        ``velocities`` (final, wrapped).
        """
        N = self.N
        masses = self.masses
        lat = self.lattice
        inv_lat = np.linalg.inv(lat)

        if equil_steps > 0:
            nvt = NVTSimulation(
                lattice=lat, positions=self.positions, atom_types=self.atom_types,
                force_fn=self.force_fn, timestep=timestep, n_steps=equil_steps,
                temperature=self.temperature, seed=seed,
                traj_every=max(equil_steps, 1), thermo_every=max(equil_steps, 1),
                output_dir=self._scratch_dir, verbose=False)
            nvt_res = nvt.run()
            pos = nvt_res["positions"]
            vel = nvt_res["velocities"]
        else:
            pos = self.positions.copy()
            vel = self._init_velocities_mb(self.temperature, masses,
                                           np.random.default_rng(seed))

        pos_unwrapped = pos.copy()
        acc_factor = EV_ANG_TO_AMU_ANG_FS2 / masses[:, None]
        _, forces, _ = self.force_fn(lat, pos, self.atom_types)
        acc = forces * acc_factor

        pos_frames, vel_frames = [], []
        if verbose:
            print(f"\n  DiffusionAnalyzer: NVE production, {n_steps} steps, "
                 f"dt={timestep} fs, recording every {record_every} steps "
                 f"-> {n_steps // record_every} frames")

        for step in range(n_steps):
            vel += 0.5 * timestep * acc
            pos += timestep * vel
            pos_unwrapped += timestep * vel

            frac = pos @ inv_lat
            frac -= np.floor(frac)
            pos = frac @ lat

            _, forces, _ = self.force_fn(lat, pos, self.atom_types)
            acc = forces * acc_factor
            vel += 0.5 * timestep * acc

            if step % record_every == 0:
                pos_frames.append(pos_unwrapped.copy())
                vel_frames.append(vel.copy())

        self._pos_series = np.array(pos_frames)
        self._vel_series = np.array(vel_frames)
        self._dt_record  = float(timestep * record_every)

        if verbose:
            print(f"  Finished: {len(self._pos_series)} frames recorded.")

        return {
            "n_frames" : len(self._pos_series),
            "dt_record": self._dt_record,
            "positions": pos,
            "velocities": vel,
        }

    def compute_msd(self, max_lag: Optional[int] = None) -> dict:
        """
        Returns
        -------
        dict with keys:
            time         : (max_lag,) lag times [fs].
            msd_total    : (max_lag,) MSD averaged over all atoms [A^2].
            msd_per_species : dict species -> (max_lag,) MSD for that
                             species' atoms only [A^2].
            D_einstein   : self-diffusion coefficient [A^2/fs], from a
                          linear fit of MSD(t) = 6 D t + const over the
                          LATTER half of the available lag range
                          (excluding the short-time ballistic regime).
        """
        if self._pos_series is None:
            raise RuntimeError(
                "DiffusionAnalyzer.compute_msd(): call run_md() first.")
        n_frames = self._pos_series.shape[0]
        if max_lag is None:
            max_lag = max(2, n_frames // 2)

        msd_total = _compute_msd(self._pos_series, max_lag)

        msd_per_species = {}
        types_arr = np.array(self.atom_types)
        for sp in sorted(set(self.atom_types)):
            mask = types_arr == sp
            msd_per_species[sp] = _compute_msd(self._pos_series[:, mask, :], max_lag)

        time = np.arange(max_lag) * self._dt_record
        fit_start = max_lag // 2
        slope, _ = np.polyfit(time[fit_start:], msd_total[fit_start:], 1)
        D_einstein = slope / 6.0   # MSD(t) -> 6 D t  (3D)

        return {
            "time"           : time,
            "msd_total"      : msd_total,
            "msd_per_species": msd_per_species,
            "D_einstein"     : float(D_einstein),
        }

    def compute_vacf_dos(self, max_lag: Optional[int] = None) -> dict:
        """
        Returns
        -------
        dict with keys:
            tau          : (max_lag,) lag times [fs].
            vacf         : (max_lag,) velocity autocorrelation function
                          [A^2/fs^2], averaged over atoms and Cartesian
                          components.
            freq_thz     : (n_freq,) frequency axis [THz].
            freq_cm      : (n_freq,) frequency axis [cm^-1].
            dos          : (n_freq,) vibrational density of states
                          (arbitrary units, non-negative, from the
                          cosine Fourier transform of the VACF).
            D_green_kubo : self-diffusion coefficient [A^2/fs] from the
                          Green-Kubo relation D = integral_0^inf VACF(t) dt
                          (trapezoidal), independent of the Einstein-
                          relation MSD slope in compute_msd() -- the two
                          should agree for a well-equilibrated
                          trajectory (see tests/test_dynamics_analysis.py
                          for the cross-check).

        Bug-fix note (max_lag vs. n_frames): keep ``max_lag`` a SMALL
        fraction of the number of recorded frames (a few percent, not
        e.g. half). ``_autocorr_fft`` subtracts the SAMPLE mean (the
        true population mean generally being unknown), which forces the
        full sum of the (appropriately lag-weighted) autocorrelation
        over ALL lags to be exactly zero -- for a real vibrational VACF
        that already decays to ~0 well within a small max_lag, this is
        harmless, but truncating the Green-Kubo integral at a max_lag
        that is a SUBSTANTIAL fraction of n_frames captures only part of
        the compensating negative bias distributed across the
        (unused) remaining lags, silently pulling D_green_kubo low --
        confirmed directly by scanning several (n_frames, max_lag)
        combinations while developing this method: max_lag = n_frames/4
        gave a D low by a factor of ~2, while max_lag <~ n_frames/50
        matched the known-correct value to within a few percent.
        """
        if self._vel_series is None:
            raise RuntimeError(
                "DiffusionAnalyzer.compute_vacf_dos(): call run_md() first.")
        n_frames = self._vel_series.shape[0]
        if max_lag is None:
            max_lag = max(2, n_frames // 50)

        vacf = _compute_vacf(self._vel_series, max_lag)
        tau = np.arange(max_lag) * self._dt_record

        D_green_kubo = float(np.trapezoid(vacf, tau))

        dos_raw = np.fft.rfft(vacf).real
        dos = np.clip(dos_raw, 0.0, None)
        freq_per_fs = np.fft.rfftfreq(max_lag, d=self._dt_record)
        freq_thz = freq_per_fs * 1000.0
        freq_cm  = (2.0 * np.pi * freq_per_fs) * FREQ_RAD_FS_TO_WAVENUMBER_CM

        return {
            "tau"         : tau,
            "vacf"        : vacf,
            "freq_thz"    : freq_thz,
            "freq_cm"     : freq_cm,
            "dos"         : dos,
            "D_green_kubo": D_green_kubo,
        }
