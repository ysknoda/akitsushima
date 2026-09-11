"""Molecular dynamics: NVT (Nose-Hoover chain), NVE (velocity Verlet), and
NPT (Nose-Hoover chain + isotropic Andersen-Hoover barostat).

Time units : fs  (1 fs = 1e-15 s)
Energy     : eV
Length     : A
Mass       : amu
Force      : eV/A  ->  amu.A/fs^2  via EV_ANG_TO_AMU_ANG_FS2
"""
import os
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..constants import ATOMIC_MASS, EV_ANG3_TO_GPA, EV_ANG_TO_AMU_ANG_FS2, K_B
from ..engine.neighborlist import NeighbourList
from ..fitting.force_fn import _resolve_force_fn
from ..io.poscar import write_poscar
from ..io.xdatcar import _write_xdatcar_frame, _write_xdatcar_header
from ..utils.mpi import _mpi_rank
from ..utils.optional_matplotlib import HAS_MATPLOTLIB, plt
from ..utils.timing import TimingCollector

__all__ = ["NVTSimulation", "NVESimulation", "NPTSimulation"]


class _BaseMDSimulation:
    """
    Shared utilities for all MD solver classes.

    This mixin is not intended to be instantiated directly.  It provides
    methods that were previously duplicated across :class:`NVTSimulation`,
    :class:`NVESimulation`, :class:`NPTSimulation`, and
    :class:`GCMCMDSimulation`.

    Methods
    -------
    _build_masses(atom_types, masses)
        Return a (N,) float array of atomic masses [amu].
        If *masses* is not None it is used as-is (after shape validation).
        Otherwise, masses are looked up from :data:`ATOMIC_MASS` with a
        fallback of 1.0 amu for unknown symbols.

    _init_velocities_mb(T, masses, rng)
        Draw velocities from the Maxwell-Boltzmann distribution at
        temperature *T* [K], remove the centre-of-mass momentum, and
        rescale to the exact target temperature.  Returns (N,3) [A/fs].

    _kinetic_energy(velocities, masses)
        E_kin [eV] = 0.5 sum m_i |v_i|^2 / unit_factor.

    _build_fix_mask(N, fix_atoms)
        Validate *fix_atoms* and return a (N,3) bool array.
        Raises ValueError when the shape is wrong.

    _compute_n_dof(N, fix_mask)
        Number of thermodynamic degrees of freedom, correctly accounting
        for frozen atoms (bug-fix note S8, see docstring below).

    _nhc_step()
        Shared 2-chain Nose-Hoover-chain half-step, used identically by
        :class:`NVTSimulation` and :class:`NPTSimulation` (bug-fix note
        S10, see docstring below).
    """

    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_output_dir(path: str) -> None:
        """Create *path* (and parents) if they do not already exist."""
        os.makedirs(path, exist_ok=True)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _flush_outputs(file_handles: list, fsync: bool = False) -> None:
        """
        Push buffered writes in *file_handles* to the OS (flush), and
        optionally force a physical write to the storage device (fsync).

        See the flush_every_step / fsync_every_step parameter docs on
        each MD simulation class's __init__ for the tradeoffs between
        the two: flush() is cheap and makes the file's current content
        visible to other processes (e.g. `tail -f`) almost immediately;
        fsync() additionally guarantees the data survives a power loss
        or OS crash, but is much slower -- particularly on network
        filesystems (NFS, Lustre, GPFS) common on HPC clusters, where
        it can take tens to hundreds of milliseconds per call.
        """
        for fh in file_handles:
            fh.flush()
            if fsync:
                os.fsync(fh.fileno())

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_masses(atom_types: list,
                      masses    : Optional[np.ndarray]) -> np.ndarray:
        """
        Return per-atom mass array [amu].

        Parameters
        ----------
        atom_types : list[str]
        masses     : (N,) array or None.
                     When None, ATOMIC_MASS is queried for each symbol;
                     unknown symbols fall back to 1.0 amu.
        """
        if masses is not None:
            m = np.asarray(masses, dtype=float)
            if m.shape != (len(atom_types),):
                raise ValueError(
                    f"_BaseMDSimulation: masses must have shape "
                    f"({len(atom_types)},), got {m.shape}.")
            return m
        return np.array([ATOMIC_MASS.get(a, 1.0) for a in atom_types],
                        dtype=float)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _init_velocities_mb(T      : float,
                            masses : np.ndarray,
                            rng    : np.random.Generator,
                            ) -> np.ndarray:
        """
        Maxwell-Boltzmann velocities [A/fs] at temperature T [K].

        Steps
        -----
        1. Draw v ~ N(0, sqrt(k_B T / m_i)) for each component.
        2. Remove net linear momentum (v -= p_COM / M_total).
        3. Rescale to exact temperature via kinetic-energy ratio.
        """
        N = len(masses)
        if N == 0:
            return np.empty((0, 3), dtype=float)
        sigma = np.sqrt(K_B * T * EV_ANG_TO_AMU_ANG_FS2 / masses)  # (N,)
        v     = rng.standard_normal((N, 3)) * sigma[:, None]
        # Remove COM momentum
        v -= (np.sum(masses[:, None] * v, axis=0) / masses.sum())
        # Rescale to exact T
        e_kin = 0.5 * float(np.sum(masses[:, None] * v**2)) / EV_ANG_TO_AMU_ANG_FS2
        n_dof = max(3 * N - 3, 1)
        T_act = 2.0 * e_kin / (n_dof * K_B)
        if T_act > 1e-12:
            v *= np.sqrt(T / T_act)
        return v

    # ------------------------------------------------------------------ #
    @staticmethod
    def _kinetic_energy(velocities : np.ndarray,
                        masses     : np.ndarray,
                        ) -> float:
        """E_kin [eV] = 0.5 sum m_i |v_i|^2 / unit_factor."""
        return (0.5 * float(np.sum(masses[:, None] * velocities**2))
                / EV_ANG_TO_AMU_ANG_FS2)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_fix_mask(N         : int,
                        fix_atoms : Optional[np.ndarray],
                        cls_name  : str = '',
                        ) -> np.ndarray:
        """
        Validate *fix_atoms* and return a (N, 3) bool array.

        Parameters
        ----------
        N         : number of atoms.
        fix_atoms : (N, 3) bool array or None.
        cls_name  : class name for error messages.
        """
        if fix_atoms is not None:
            fa = np.asarray(fix_atoms, dtype=bool)
            if fa.shape != (N, 3):
                raise ValueError(
                    f"{cls_name}: fix_atoms must have shape ({N}, 3), "
                    f"got {fa.shape}.")
            return fa
        return np.zeros((N, 3), dtype=bool)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_n_dof(N: int, fix_mask: np.ndarray) -> int:
        """
        Number of thermodynamic degrees of freedom for the NHC
        thermostat / instantaneous-temperature formula.

        Bug-fix note (S8): every caller of this (NVTSimulation,
        NPTSimulation, GCMCMDSimulation, OSMMCSimulation) used to hard-code
        ``n_dof = 3*N - 3`` (removing exactly the 3 centre-of-mass
        translational modes) regardless of ``fix_atoms``. Once any
        degree of freedom is frozen, the system can no longer translate
        freely as a whole -- a single pinned atom already breaks
        translational invariance and the total momentum is no longer a
        conserved quantity of the (constrained) equations of motion --
        so the "-3" correction is only valid when nothing is fixed.
        With one or more frozen components, the correct DOF count is
        simply 3*N minus however many Cartesian components are frozen.
        """
        n_fixed = int(np.sum(fix_mask))
        if n_fixed == 0:
            return max(3 * N - 3, 1)
        return max(3 * N - n_fixed, 1)

    # ------------------------------------------------------------------ #
    def _nhc_step(self) -> None:
        """
        2-chain Nose-Hoover-chain half-step -- VASP-style MTTK propagator.

        Bug-fix note (S10): this exact method body used to be duplicated
        verbatim in both NVTSimulation and NPTSimulation (NVTSimulation's
        copy took the current kinetic energy as a parameter computed by
        the caller; NPTSimulation's copy recomputed it internally --
        otherwise byte-for-byte identical). Hoisted here so both classes
        share one implementation; it operates purely through instance
        attributes so no signature change was needed at the call site
        beyond dropping the now-redundant externally-computed kinetic
        energy argument.

        Requires ``self.dt``, ``self.n_dof``, ``self.temperature``,
        ``self.Q1``, ``self.Q2``, ``self.vxi1``, ``self.vxi2``,
        ``self.xi1``, ``self.xi2``, ``self.velocities``,
        ``self._fix_mask``, and ``self._kinetic_energy()``.

        All exponential arguments are clamped to [-350, 350] to prevent
        float64 overflow (the negative Suzuki-Yoshida weight in a
        3rd-order scheme is avoided altogether by using only positive
        sub-step intervals, per Martyna, Klein & Tuckerman, J. Chem.
        Phys. 97, 2635 (1992); see https://vasp.at/wiki/Nose-Hoover_thermostat).
        """
        dt = self.dt
        Q1 = self.Q1; Q2 = self.Q2
        kT = K_B * self.temperature
        e_kin = self._kinetic_energy()

        G1 = (2.0 * e_kin - self.n_dof * kT) / Q1
        G2 = (Q1 * self.vxi1 ** 2 - kT) / Q2

        # vxi2 half-kick
        self.vxi2 += G2 * (dt * 0.25)

        # vxi1: damp + kick + damp
        s_arg = float(np.clip(-self.vxi2 * dt * 0.125, -350.0, 350.0))
        s     = np.exp(s_arg)
        self.vxi1 = s * (s * self.vxi1 + G1 * dt * 0.25)

        # Scale velocities
        sc_arg = float(np.clip(-self.vxi1 * dt * 0.5, -350.0, 350.0))
        self.velocities *= np.exp(sc_arg)
        self.velocities *= ~self._fix_mask

        # Advance thermostat positions
        self.xi1 += self.vxi1 * dt * 0.5
        self.xi2 += self.vxi2 * dt * 0.5

        # Recompute E_kin and G1
        e_kin = self._kinetic_energy()
        G1    = (2.0 * e_kin - self.n_dof * kT) / Q1

        # Reverse: damp + kick + damp
        self.vxi1 = s * (s * self.vxi1 + G1 * dt * 0.25)

        # vxi2 half-kick (reverse)
        G2         = (Q1 * self.vxi1 ** 2 - kT) / Q2
        self.vxi2 += G2 * (dt * 0.25)


#  SECTION 6.1: NVT molecular dynamics -- Nose-Hoover chain thermostat
# ============================================================================
#
#  Integrator : velocity Verlet with Nose-Hoover chain (NHC) of length 2
#  Reference  : Martyna, Tobias & Klein, J. Chem. Phys. 97, 2635 (1992).
# ============================================================================

class NVTSimulation(_BaseMDSimulation):
    """
    NVT molecular dynamics with Nose-Hoover chain (NHC) thermostat.

    Integrator  : velocity Verlet + NHC of length 2 (Suzuki-Yoshida, 4 sub-steps)
    Reference   : Martyna, Tobias & Klein, J. Chem. Phys. 97, 2635 (1992)

    Parameters
    ----------
    lattice / positions / atom_types : initial structure.
    buck_params  : Buckingham parameters dict.
    charges      : dict  symbol -> formal charge [e].
    temperature  : target temperature [K].
    timestep     : integration time step [fs]  (0.5--1.0 fs typical).
    n_steps      : production MD steps.
    equil_steps  : equilibration steps (excluded from trajectory / averages).
    tau_nhc      : NHC thermostat time constant [fs]  (50--200 fs typical).
    traj_every   : XYZ trajectory output interval [steps].
    thermo_every : thermo log output interval [steps].
    skin_width   : Verlet skin width [A] for neighbour-list lazy rebuild.
    seed         : RNG seed for initial velocity generation.

    Output files
    ------------
    trajectory.xyz   -- multi-frame XYZ with energy and temperature in comment
    thermo.log       -- step / time[ps] / E_pot / E_kin / E_tot / T / H_NHC
    POSCAR_final_MD  -- last frame as VASP 5 POSCAR
    """

    def __init__(self,
                 lattice         : np.ndarray,
                 positions       : np.ndarray,
                 atom_types      : list,
                 # -- universal interface (A) ---------------------------
                 force_fn        : Optional[Callable] = None,
                 r_cut           : Optional[float] = None,
                 # -- legacy Buckingham kwargs (B, backward-compatible) ----
                 buck_params     : Optional[dict]  = None,
                 charges         : Optional[dict]  = None,
                 ewald_alpha     : Optional[float] = None,
                 coulomb_method  : str   = "ewald",
                 pppm_accuracy   : float = 1e-5,
                 pppm_mesh       : Optional[int] = None,
                 pppm_order      : int   = 6,
                 # -- NHC / simulation parameters -----------------------
                 temperature     : float = 300.0,
                 timestep        : float = 1.0,
                 n_steps         : int   = 10000,
                 tau_nhc         : float = 100.0,
                 output_dir      : str   = ".",
                 traj_every      : int   = 10,
                 thermo_every    : int   = 100,
                 equil_steps     : int   = 0,
                 skin_width      : float = 0.5,
                 seed            : int   = 42,
                 verbose         : bool  = True,
                 masses          : Optional[np.ndarray] = None,
                 fix_atoms       : Optional[np.ndarray] = None,
                 timing          : bool  = False,
                 velocity_rescale_steps : int  = 0,
                 flush_every_step : bool = True,
                 fsync_every_step : bool = False):
        self.lattice        = lattice.copy()
        self.positions      = positions.copy()
        self.atom_types     = list(atom_types)
        self.r_cut          = r_cut
        self.temperature    = temperature
        self.dt             = timestep
        self.n_steps        = n_steps
        self.tau_nhc        = tau_nhc
        self.output_dir     = output_dir
        self.traj_every     = traj_every
        self.thermo_every   = thermo_every
        self.equil_steps    = equil_steps
        self.verbose        = verbose
        self.N              = len(positions)
        self.rng            = np.random.default_rng(seed)
        # See _BaseMDSimulation._flush_outputs for the flush/fsync tradeoffs.
        self.flush_every_step = bool(flush_every_step)
        self.fsync_every_step = bool(fsync_every_step)

        # -- resolve force function + r_cut --------------------
        self._force_fn, r_cut = _resolve_force_fn(
            force_fn, r_cut, buck_params, charges, ewald_alpha,
            coulomb_method, pppm_accuracy, pppm_mesh, pppm_order,
            "NVTSimulation")
        self.r_cut          = float(r_cut)

        self.buck_params    = buck_params
        self.charges        = charges
        self.ewald_alpha    = ewald_alpha
        self.coulomb_method = coulomb_method
        self.pppm_accuracy  = pppm_accuracy
        self.pppm_mesh      = pppm_mesh
        self.pppm_order     = pppm_order

        self.masses = self._build_masses(atom_types, masses)

        self.velocities     = self._init_velocities()
        self._fix_mask = self._build_fix_mask(
            self.N, fix_atoms, self.__class__.__name__)
        self.velocities[self._fix_mask] = 0.0
        # Bug-fix note (S8): n_dof now accounts for fix_atoms -- see
        # _BaseMDSimulation._compute_n_dof.
        self.n_dof  = self._compute_n_dof(self.N, self._fix_mask)

        # 2-chain NHC state variables
        self.Q1   = self.n_dof * K_B * temperature * tau_nhc**2
        self.Q2   = K_B * temperature * tau_nhc**2
        self.xi1  = 0.0;  self.xi2  = 0.0
        self.vxi1 = 0.0;  self.vxi2 = 0.0

        self._tc = TimingCollector(enabled=timing)
        self.velocity_rescale_steps = int(velocity_rescale_steps)
        self._nl            = NeighbourList(r_cut=self.r_cut, skin_width=skin_width)
        self._thermo_log: List[dict] = []

    def _init_velocities(self) -> np.ndarray:
        """Maxwell-Boltzmann velocities at self.temperature; delegates to base."""
        return _BaseMDSimulation._init_velocities_mb(
            self.temperature, self.masses, self.rng)

    def _kinetic_energy(self, v: Optional[np.ndarray] = None) -> float:
        """E_kin [eV]; delegates to base."""
        if v is None:
            v = self.velocities
        return _BaseMDSimulation._kinetic_energy(v, self.masses)

    def _temperature_inst(self, v: Optional[np.ndarray] = None) -> float:
        """Instantaneous temperature [K]."""
        return 2.0 * self._kinetic_energy(v) / (self.n_dof * K_B)

    def _instantaneous_temperature(self) -> float:
        """T_inst = 2 E_kin / (n_dof * k_B)  [K]."""
        return 2.0 * self._kinetic_energy() / (self.n_dof * K_B)

    def _nhc_conserved(self, e_pot: float) -> float:
        """
        Conserved quantity for the 2-chain Nose-Hoover thermostat:
          H_NHC = E_kin + E_pot
                + (Q1*vxi1^2 + Q2*vxi2^2) / (2 * EV_ANG)
                + n_dof*kT*xi1 + kT*xi2
        """
        ek    = self._kinetic_energy()
        t_nhc = (0.5 * self.Q1 * self.vxi1**2
                 + 0.5 * self.Q2 * self.vxi2**2) / EV_ANG_TO_AMU_ANG_FS2
        v_nhc = (self.n_dof * K_B * self.temperature * self.xi1
                 + K_B * self.temperature * self.xi2)
        return ek + e_pot + t_nhc + v_nhc

    def _compute_forces(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluate energy, forces, and stress via self._force_fn."""
        self._nl.check_and_rebuild(self.lattice, self.positions)
        return self._force_fn(
            self.lattice, self.positions, self.atom_types,
            neigh_list=self._nl)

    def _write_xyz_frame(self, fh, step: int, e_pot: float) -> None:
        """
        Append one configuration to the open XDATCAR trajectory file.

        NVT uses a fixed cell, so only the "Direct configuration="
        line and fractional coordinates are written per frame; the
        header (comment, scale, lattice, species, counts) is written
        once at the start of the file, before the MD loop begins.
        """
        _write_xdatcar_frame(fh, self.positions, self.lattice,
                             self.atom_types, self._xdatcar_frame_idx,
                             variable_header=False)
        self._xdatcar_frame_idx += 1

    # ------------------------------------------------------------------ #
    #  Main MD loop                                                        #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """
        Execute the NVT-MD simulation.

        Algorithm (velocity Verlet + NHC):
            1. NHC half-step  (thermostat propagation)
            2. Velocity half-step:  v += 0.5 dt * f/m
            3. Position full-step:  r += dt * v
            4. Force evaluation at new positions
            5. Velocity half-step:  v += 0.5 dt * f/m
            6. NHC half-step  (thermostat propagation)

        Returns
        -------
        dict: final state + time-averaged properties.
        """
        self._tc.start("total")  # timing
        self._ensure_output_dir(self.output_dir)
        traj_path   = os.path.join(self.output_dir, "XDATCAR_nvt")
        thermo_path = os.path.join(self.output_dir, "thermo.log")

        # Initial force evaluation
        e_pot, forces, stress = self._compute_forces()

        # Accelerations  [A/fs^2] = F [eV/A] * EV_ANG_TO_AMU_ANG_FS2 / m [amu]
        acc = (forces * EV_ANG_TO_AMU_ANG_FS2
               / self.masses[:, None])   # (N,3) A/fs^2

        total_steps = self.equil_steps + self.n_steps
        header = (f"{'Step':>8}  {'Time(ps)':>10}  {'E_pot(eV)':>14}  "
                  f"{'E_kin(eV)':>12}  {'E_tot(eV)':>12}  "
                  f"{'T_inst(K)':>10}  {'H_NHC(eV)':>14}  "
                  f"{'NL_builds':>10}")
        if self.verbose:
            print(f"\nNVT-MD: T={self.temperature} K  "
                  f"dt={self.dt} fs  "
                  f"steps={total_steps}  "
                  f"(equil={self.equil_steps})")
            print(header)
            print("  " + "-" * len(header))

        t_avg_list, e_pot_list, e_kin_list = [], [], []

        # Only rank 0 opens (and writes to) the real trajectory/thermo
        # files -- every other MPI rank writes to os.devnull instead.
        # Without this, ALL ranks would independently open() the SAME
        # file path in "w" mode and write IDENTICAL content to it
        # concurrently, which is not just wasteful but a genuine data
        # corruption risk (interleaved/truncated writes from multiple
        # processes to one file). The rest of the loop body is
        # unchanged: every rank still gets a valid, writable file
        # handle, so no other code needs to know which rank it is.
        _traj_path_use   = traj_path   if _mpi_rank == 0 else os.devnull
        _thermo_path_use = thermo_path if _mpi_rank == 0 else os.devnull
        with (open(_traj_path_use,   "w") as traj_fh,
              open(_thermo_path_use, "w") as thermo_fh):

            thermo_fh.write(header + "\n")

            _write_xdatcar_header(traj_fh, self.lattice, self.atom_types,
                                 comment=f"akitsushima.py NVT-MD  T={self.temperature}K")
            self._xdatcar_frame_idx = 1

            for step in range(total_steps):
                self._tc.start("step")  # timing
                t_sim  = step * self.dt
                t_ps   = t_sim * 1e-3

                # --- NHC half-step (before velocity half-step) ------------
                self._nhc_step()   # MTTK propagator; also rescales velocities

                # --- Velocity half-step (v += 0.5 dt a) ------------------
                self.velocities += 0.5 * self.dt * acc
                self.velocities *= ~self._fix_mask   # freeze constrained DoF

                # --- Position full-step  (r += dt v) ---------------------
                self.positions  += self.dt * self.velocities

                # Apply periodic boundary conditions
                inv_lat         = np.linalg.inv(self.lattice)
                frac            = self.positions @ inv_lat
                frac           -= np.floor(frac)
                self.positions  = frac @ self.lattice

                # --- Force evaluation at new positions -------------------
                e_pot, forces, stress = self._compute_forces()
                acc = (forces * EV_ANG_TO_AMU_ANG_FS2
                       / self.masses[:, None])

                # --- Velocity half-step  (v += 0.5 dt a) -----------------
                self.velocities += 0.5 * self.dt * acc
                self.velocities *= ~self._fix_mask   # freeze constrained DoF

                # --- NHC half-step (after velocity half-step) ------------
                self._nhc_step()   # MTTK propagator; also rescales velocities

                # --- Optional velocity rescaling (early steps only) ------
                # When velocity_rescale_steps > 0, apply direct velocity
                # rescaling for the first N steps in addition to NHC.
                # This prevents runaway kinetic energy in the initial
                # transient that occurs for perfect-crystal structures with
                # zero forces at t=0 (SW / Tersoff potentials).
                if self.velocity_rescale_steps > 0 and step < self.velocity_rescale_steps:
                    ek_now = self._kinetic_energy()
                    T_now  = 2.0 * ek_now / (self.n_dof * K_B) if ek_now > 0 else 0.0
                    if T_now > 1e-12:
                        scale_vr = np.sqrt(self.temperature / T_now)
                        self.velocities *= scale_vr
                        self.velocities *= ~self._fix_mask

                # --- Thermodynamic quantities ----------------------------
                e_kin    = self._kinetic_energy()
                e_tot    = e_pot + e_kin
                T_inst   = self._instantaneous_temperature()
                H_nhc    = self._nhc_conserved(e_pot)

                # Accumulate averages only after equilibration
                if step >= self.equil_steps:
                    t_avg_list.append(T_inst)
                    e_pot_list.append(e_pot)
                    e_kin_list.append(e_kin)

                # --- Output ----------------------------------------------
                is_prod = step >= self.equil_steps

                self._tc.lap("step")  # timing
                if step % self.thermo_every == 0:
                    row = (f"{step:8d}  {t_ps:10.4f}  {e_pot:14.6f}  "
                           f"{e_kin:12.6f}  {e_tot:12.6f}  "
                           f"{T_inst:10.2f}  {H_nhc:14.6f}  "
                           f"{self._nl.rebuild_count:10d}")
                    thermo_fh.write(row + "\n")
                    if self.verbose:
                        print("  " + row)
                    self._thermo_log.append({
                        "step"      : step, "time_ps": t_ps,
                        "e_pot"     : e_pot, "e_kin": e_kin, "e_tot": e_tot,
                        "T_inst"    : T_inst, "H_nhc": H_nhc,
                        "nl_builds" : self._nl.rebuild_count,
                    })

                if is_prod and step % self.traj_every == 0:
                    self._write_xyz_frame(traj_fh, step, e_pot)

                # Push this step's output to disk for real-time
                # monitoring during a long-running MD calculation.
                if self.flush_every_step and (
                        step % self.thermo_every == 0
                        or (is_prod and step % self.traj_every == 0)):
                    self._flush_outputs([thermo_fh, traj_fh],
                                        fsync=self.fsync_every_step)

        # ---- Final structure in POSCAR format ---------------------------
        out_poscar = os.path.join(self.output_dir, "POSCAR_final_MD")
        if _mpi_rank == 0:
            write_poscar(out_poscar, self.lattice, self.positions,
                         self.atom_types,
                         title=f"MD final frame  T={self.temperature}K")

        # ---- Summary statistics ----------------------------------------
        t_avg_arr   = np.array(t_avg_list)
        e_pot_arr   = np.array(e_pot_list)
        e_kin_arr   = np.array(e_kin_list)

        T_mean  = t_avg_arr.mean()   if len(t_avg_arr)  else 0.0
        T_std   = t_avg_arr.std()    if len(t_avg_arr)  else 0.0
        Ep_mean = e_pot_arr.mean()   if len(e_pot_arr)  else 0.0

        if self.verbose:
            print(f"\n  NVT-MD finished.")
            print(f"  <T>      = {T_mean:.2f} +/- {T_std:.2f} K  "
                  f"(target {self.temperature} K)")
            print(f"  <E_pot>  = {Ep_mean:.6f} eV")
            print(f"  NL rebuilds: {self._nl.rebuild_count}  "
                  f"(ratio: {self._nl.rebuild_count}/{self._nl.step_count} = "
                  f"{self._nl.rebuild_count/max(self._nl.step_count,1):.3f})")
            print(f"  Trajectory : {traj_path}")
            print(f"  Log        : {thermo_path}")
            print(f"  Final POSCAR: {out_poscar}")

        # ---- Optional plots --------------------------------------------
        if HAS_MATPLOTLIB and len(self._thermo_log) > 1 and _mpi_rank == 0:
            self._save_md_plots()

        self._tc.stop("total")  # timing
        _timing = self._tc.summary()
        if _timing.get("enabled") and _mpi_rank == 0:
            self._tc.print_summary(label="NVTSimulation")

        return {
            "positions"  : self.positions.copy(),
            "velocities" : self.velocities.copy(),
            "lattice"    : self.lattice.copy(),
            "T_mean"     : T_mean,
            "T_std"      : T_std,
            "e_pot_mean" : Ep_mean,
            "thermo_log" : self._thermo_log,
            "timing"     : _timing,
        }

    def _save_md_plots(self) -> None:
        """Save temperature and energy time-series plots."""
        steps  = [r["step"]   for r in self._thermo_log]
        times  = [r["time_ps"] for r in self._thermo_log]
        T_list = [r["T_inst"] for r in self._thermo_log]
        E_list = [r["e_tot"]  for r in self._thermo_log]
        H_list = [r["H_nhc"]  for r in self._thermo_log]

        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        axes[0].plot(times, T_list, lw=0.8)
        axes[0].axhline(self.temperature, color="r", ls="--", lw=1,
                        label=f"Target {self.temperature} K")
        axes[0].set_ylabel("Temperature (K)")
        axes[0].legend(fontsize=8)

        axes[1].plot(times, E_list, lw=0.8, color="steelblue")
        axes[1].set_ylabel("Total energy (eV)")

        axes[2].plot(times, H_list, lw=0.8, color="darkorange")
        axes[2].set_ylabel("NHC conserved qty (eV)")
        axes[2].set_xlabel("Time (ps)")

        for ax in axes:
            ax.grid(True, alpha=0.3)

        # Shade equilibration region
        if self.equil_steps > 0:
            t_equil = self.equil_steps * self.dt * 1e-3
            for ax in axes:
                ax.axvspan(0, t_equil, alpha=0.1, color="grey",
                           label="Equilibration")

        fig.suptitle(f"NVT MD  --  T = {self.temperature} K", fontsize=12)
        fig.tight_layout()
        path = os.path.join(self.output_dir, "md_timeseries.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  MD time-series plot saved: {path}")


# ============================================================================
#  SECTION 6.2: NVE molecular dynamics (microcanonical ensemble)
# ============================================================================
#
#  Integrator : standard velocity Verlet (symplectic, time-reversible)
#  Conserved  : E_tot = E_kin + E_pot  (exact in absence of round-off)
#  Usage      : energy-conservation benchmarks after NVT equilibration;
#               short-time dynamics; diffusion coefficients.
#
#  Quality criterion: |delta_E_tot| / N / sum_ns  < 1 meV/atom/ns
# ============================================================================

class NVESimulation(_BaseMDSimulation):
    """
    NVE (microcanonical) molecular dynamics -- plain velocity Verlet.

    No thermostat or barostat.  Total energy E_tot = E_kin + E_pot is
    conserved up to integration error.  Time-step choice (0.5--1 fs for
    ionic systems) is the primary lever for controlling drift.

    Typical workflow
    ----------------
    1. Equilibrate with NVTSimulation.
    2. Pass the final velocities here via ``velocities=``.
    3. Run production; monitor drift in thermo.log.

    Parameters
    ----------
    lattice / positions / atom_types
        Initial structure.
    buck_params / charges / r_cut
        Potential definition (Buckingham + Coulomb via compute_all).
    timestep     : integration time step [fs].  Default 1.0 fs.
    n_steps      : production steps.
    equil_steps  : warm-up steps (not written to trajectory / averages).
    velocities   : (N,3) [A/fs].  If None, drawn from Maxwell-Boltzmann
                   at ``temperature``.
    temperature  : used only when ``velocities`` is None [K].
    traj_every   : XYZ frame output interval [steps].
    thermo_every : thermo log output interval [steps].
    skin_width   : Verlet skin [A].
    seed         : RNG seed for velocity generation.

    Output files
    ------------
    trajectory.xyz   -- multi-frame XYZ
    thermo.log       -- Step / Time(ps) / E_pot / E_kin / E_tot / T / dE_drift
    POSCAR_final_NVE -- final structure in VASP 5 format
    """

    def __init__(self,
                 lattice        : np.ndarray,
                 positions      : np.ndarray,
                 atom_types     : list,
                 # -- universal interface (A) ---------------------------
                 force_fn       : Optional[Callable] = None,
                 r_cut          : Optional[float] = None,
                 # -- legacy Buckingham kwargs (B) ----------------------
                 buck_params    : Optional[dict]  = None,
                 charges        : Optional[dict]  = None,
                 ewald_alpha    : Optional[float] = None,
                 coulomb_method : str   = "ewald",
                 pppm_accuracy  : float = 1e-5,
                 pppm_mesh      : Optional[int] = None,
                 pppm_order     : int   = 6,
                 # -- simulation parameters -----------------------------
                 timestep       : float = 1.0,
                 n_steps        : int   = 10000,
                 equil_steps    : int   = 0,
                 velocities     : Optional[np.ndarray] = None,
                 temperature    : float = 300.0,
                 traj_every     : int   = 10,
                 thermo_every   : int   = 100,
                 output_dir     : str   = ".",
                 skin_width     : float = 0.5,
                 seed           : int   = 42,
                 verbose        : bool  = True,
                 masses         : Optional[np.ndarray] = None,
                 fix_atoms      : Optional[np.ndarray] = None,
                 timing         : bool  = False,
                 flush_every_step : bool = True,
                 fsync_every_step : bool = False):

        self.flush_every_step = bool(flush_every_step)
        self.fsync_every_step = bool(fsync_every_step)
        self.lattice        = lattice.copy()
        self.positions      = positions.copy()
        self.atom_types     = list(atom_types)
        self.r_cut          = r_cut
        self.dt             = timestep
        self.n_steps        = n_steps
        self.equil_steps    = equil_steps
        self.temperature    = temperature
        self.traj_every     = traj_every
        self.thermo_every   = thermo_every
        self.output_dir     = output_dir
        self.verbose        = verbose
        self.N              = len(positions)
        self.rng            = np.random.default_rng(seed)

        # -- resolve force function + r_cut --------------------
        self._force_fn, r_cut = _resolve_force_fn(
            force_fn, r_cut, buck_params, charges, ewald_alpha,
            coulomb_method, pppm_accuracy, pppm_mesh, pppm_order,
            "NVESimulation")
        self.r_cut          = float(r_cut)

        self.buck_params    = buck_params
        self.charges        = charges
        self.ewald_alpha    = ewald_alpha
        self.coulomb_method = coulomb_method
        self.pppm_accuracy  = pppm_accuracy
        self.pppm_mesh      = pppm_mesh
        self.pppm_order     = pppm_order

        self.masses = self._build_masses(atom_types, masses)

        self.velocities = velocities.copy() if velocities is not None \
                          else self._init_velocities(temperature)
        self._nl = NeighbourList(r_cut=self.r_cut, skin_width=skin_width)
        # -- fix_atoms constraint mask -------------------------
        # fix_atoms : (N, 3) bool array.  True = frozen (never displaced).
        # Equivalent to VASP Selective Dynamics 'F T T' etc.
        self._fix_mask = self._build_fix_mask(
            self.N, fix_atoms, self.__class__.__name__)
        # Zero initial velocities for frozen DoF
        self.velocities[self._fix_mask] = 0.0
        # Bug-fix note (S8): n_dof now accounts for fix_atoms -- see
        # _BaseMDSimulation._compute_n_dof.
        self.n_dof = self._compute_n_dof(self.N, self._fix_mask)
        self._tc = TimingCollector(enabled=timing)

    # ------------------------------------------------------------------
    def _init_velocities(self, T: float) -> np.ndarray:
        """Maxwell-Boltzmann velocities at T; delegates to base."""
        return _BaseMDSimulation._init_velocities_mb(T, self.masses, self.rng)

    def _kinetic_energy(self, v: Optional[np.ndarray] = None) -> float:
        """E_kin [eV] = 0.5 sum m_i v_i^2 / unit_factor."""
        if v is None:
            v = self.velocities
        return _BaseMDSimulation._kinetic_energy(v, self.masses)

    def _temperature_inst(self) -> float:
        return 2.0 * self._kinetic_energy() / (self.n_dof * K_B)

    def _compute_forces(self) -> Tuple[float, np.ndarray, np.ndarray]:
        self._nl.check_and_rebuild(self.lattice, self.positions)
        return self._force_fn(
            self.lattice, self.positions, self.atom_types,
            neigh_list=self._nl)

    def _write_xyz_frame(self, fh, step: int, e_pot: float) -> None:
        """Append one configuration to the open XDATCAR file (fixed cell)."""
        _write_xdatcar_frame(fh, self.positions, self.lattice,
                             self.atom_types, self._xdatcar_frame_idx,
                             variable_header=False)
        self._xdatcar_frame_idx += 1

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """
        Execute the NVE-MD simulation.

        Algorithm (standard velocity Verlet)
        -------------------------------------
        Given F(r_t) at start of step:
            v_{t+1/2} = v_t + 0.5 dt . F(r_t)/m
            r_{t+1} = r_t + dt . v_{t+1/2}
            F_{t+1} = F(r_{t+1})
            v_{t+1} = v_{t+1/2} + 0.5 dt . F_{t+1}/m

        Energy drift per step delta_E ~= O(dt^3).

        Returns
        -------
        dict with final positions, velocities, lattice,
        time-averaged T and E_pot, and total E_tot drift.
        """
        self._tc.start("total")  # timing
        self._ensure_output_dir(self.output_dir)
        traj_path   = os.path.join(self.output_dir, "XDATCAR_nve")
        thermo_path = os.path.join(self.output_dir, "thermo.log")

        # ---- Initial force evaluation -----------------------------------
        e_pot, forces, stress = self._compute_forces()
        acc = forces * EV_ANG_TO_AMU_ANG_FS2 / self.masses[:, None]
        # Reference total energy (set after first force call so drift = 0 at step 0)
        e_tot_ref = e_pot + self._kinetic_energy()

        total_steps = self.equil_steps + self.n_steps
        GPa_fac = EV_ANG3_TO_GPA
        header = (f"{'Step':>8}  {'Time(ps)':>10}  {'E_pot(eV)':>14}  "
                  f"{'E_kin(eV)':>12}  {'E_tot(eV)':>14}  "
                  f"{'T_inst(K)':>10}  {'P_inst(GPa)':>12}  {'dE_drift(eV)':>14}")
        if self.verbose:
            print(f"\nNVE-MD: dt={self.dt} fs  steps={total_steps}  "
                  f"(equil={self.equil_steps})")
            print(f"  E_tot(ref) = {e_tot_ref:.6f} eV")
            print(header)
            print("  " + "-" * len(header))

        t_list, ep_list, ek_list = [], [], []

        # Only rank 0 opens (and writes to) the real trajectory/thermo
        # files -- every other MPI rank writes to os.devnull instead.
        # See NVTSimulation.run() for the full rationale.
        _traj_path_use   = traj_path   if _mpi_rank == 0 else os.devnull
        _thermo_path_use = thermo_path if _mpi_rank == 0 else os.devnull
        with (open(_traj_path_use,   "w") as traj_fh,
              open(_thermo_path_use, "w") as thermo_fh):
            thermo_fh.write(header + "\n")
            _write_xdatcar_header(traj_fh, self.lattice, self.atom_types,
                                 comment="akitsushima.py NVE-MD")
            self._xdatcar_frame_idx = 1

            for step in range(total_steps):
                self._tc.start("step")  # timing
                t_ps = step * self.dt * 1e-3

                # ---- Velocity Verlet ------------------------------------
                # v half-step
                self.velocities += 0.5 * self.dt * acc
                self.velocities *= ~self._fix_mask   # freeze constrained DoF
                # position full-step
                self.positions  += self.dt * self.velocities
                # PBC wrap
                inv_lat          = np.linalg.inv(self.lattice)
                frac             = self.positions @ inv_lat
                frac            -= np.floor(frac)
                self.positions   = frac @ self.lattice
                # new forces
                e_pot, forces, stress = self._compute_forces()
                acc = forces * EV_ANG_TO_AMU_ANG_FS2 / self.masses[:, None]
                # v half-step
                self.velocities += 0.5 * self.dt * acc
                self.velocities *= ~self._fix_mask   # freeze constrained DoF
                # --------------------------------------------------------

                e_kin  = self._kinetic_energy()
                e_tot  = e_pot + e_kin
                T_inst = self._temperature_inst()
                drift  = e_tot - e_tot_ref

                is_prod = step >= self.equil_steps
                if is_prod:
                    t_list.append(T_inst)
                    ep_list.append(e_pot)
                    ek_list.append(e_kin)

                self._tc.lap("step")  # timing
                if step % self.thermo_every == 0:
                    P_gpa = -np.trace(stress) / 3.0 * GPa_fac
                    row = (f"{step:8d}  {t_ps:10.4f}  {e_pot:14.6f}  "
                           f"{e_kin:12.6f}  {e_tot:14.6f}  "
                           f"{T_inst:10.2f}  {P_gpa:12.4f}  {drift:14.6f}")
                    thermo_fh.write(row + "\n")
                    if self.verbose:
                        print("  " + row)

                if is_prod and step % self.traj_every == 0:
                    self._write_xyz_frame(traj_fh, step, e_pot)

                # Push this step's output to disk for real-time
                # monitoring during a long-running MD calculation.
                if self.flush_every_step and (
                        step % self.thermo_every == 0
                        or (is_prod and step % self.traj_every == 0)):
                    self._flush_outputs([thermo_fh, traj_fh],
                                        fsync=self.fsync_every_step)

        # ---- Summary ---------------------------------------------------
        T_mean  = float(np.mean(t_list))  if t_list else 0.0
        T_std   = float(np.std(t_list))   if t_list else 0.0
        ep_mean = float(np.mean(ep_list)) if ep_list else 0.0
        final_drift = e_pot + self._kinetic_energy() - e_tot_ref

        out_poscar = os.path.join(self.output_dir, "POSCAR_final_NVE")
        if _mpi_rank == 0:
            write_poscar(out_poscar, self.lattice, self.positions,
                         self.atom_types, title="NVE final frame")

        if self.verbose:
            print(f"\n  NVE-MD finished.")
            print(f"  <T>          = {T_mean:.2f} +/- {T_std:.2f} K")
            print(f"  <E_pot>      = {ep_mean:.6f} eV")
            print(f"  E_tot drift  = {final_drift:+.6f} eV  "
                  f"({final_drift / self.N * 1e3:.4f} meV/atom)")
            print(f"  Trajectory   : {traj_path}")
            print(f"  Final POSCAR : {out_poscar}")

        self._tc.stop("total")  # timing
        _timing = self._tc.summary()
        if _timing.get("enabled") and _mpi_rank == 0:
            self._tc.print_summary(label="NVESimulation")

        return {
            "positions"  : self.positions.copy(),
            "velocities" : self.velocities.copy(),
            "lattice"    : self.lattice.copy(),
            "T_mean"     : T_mean,
            "T_std"      : T_std,
            "e_pot_mean" : ep_mean,
            "e_tot_drift": final_drift,
            "timing"     : _timing,
        }


# ============================================================================
#  SECTION 6.3: NPT molecular dynamics (isothermal-isobaric ensemble)
# ============================================================================
#
#  Algorithm   : Martyna-Tuckerman-Tobias-Klein (MTTK) with isotropic
#                volume scaling.  Ref: Martyna et al., Mol. Phys. 87, 1117 (1996)
#
#  Thermostat  : Nose-Hoover chain (NHC, length 2) -- same as NVT.
#  Barostat    : Andersen-Hoover isotropic barostat (scalar volume DOF).
#                Cell shape is preserved; only volume V varies.
#
#  Equations of motion (simplified MTTK, isotropic)
#  --------------------------------------------------
#  sum_i = v_i + epsilon_dot r_i          (position)
#  vsum_i = F_i/m_i - (epsilon_dot + xi1) v_i   (velocity, coupling to barostat + NHC)
#  V_dot = 3 V epsilon_dot                (volume)
#  xi_V = 3V (P_inst - P_ext) - xi1 p_V   (barostat momentum)
#  xi1 NHC coupling  (standard)
#
#  where epsilon_dot = p_V / W  [fs^-1], W = barostat mass [eV.fs^2].
#
#  Integration order (Trotter factorisation, per step)
#  ----------------------------------------------------
#  1. NHC half-step
#  2. Barostat half-step   (p_V update)
#  3. Velocity half-step   (v += 0.5dt [F/m - epsilon_dot v - xi1 v] <- via NHC scalinging)
#     -> in practice: v += 0.5dt F/m; apply epsilon_dot and NHC scaling separately
#  4. Position + volume full-step
#  5. Force evaluation
#  6. Velocity half-step (symmetric)
#  7. Barostat half-step
#  8. NHC half-step
#
#  Conserved quantity
#  ------------------
#  H_NPT = E_kin + E_pot + P_ext V + p_V^2/(2W) + NHC terms
# ============================================================================

class NPTSimulation(_BaseMDSimulation):
    """
    NPT (isothermal-isobaric) MD: NHC thermostat + isotropic Andersen barostat.

    Reference: Martyna, Tuckerman, Tobias & Klein, Mol. Phys. 87, 1117 (1996).

    Cell shape (angles) is preserved; only the volume fluctuates.
    For full anisotropic cell dynamics the Parrinello-Rahman barostat
    is needed (not implemented here).

    Unit notes
    ----------
    p_V  [amu.A^3/fs]  --- conjugate momentum to volume V [A^3]
    W    [amu.A^6]     --- barostat mass; epsilon_dot = p_V / W  [A^3/fs / A^3] = [fs^-1]
    Pressure          --- input in GPa; stored internally in eV/A^3
                        (1 eV/A^3 = 160.2176621 GPa)

    Parameters
    ----------
    lattice / positions / atom_types
        Initial structure.
    buck_params / charges / r_cut
        Potential definition.
    temperature  : target temperature [K].
    pressure     : target pressure [GPa].  Default 0.
    timestep     : integration time step [fs].
    n_steps      : production steps.
    equil_steps  : equilibration steps (not written to trajectory / averages).
    tau_nhc      : NHC thermostat time constant [fs].  Default 100 fs.
    tau_baro     : barostat time constant [fs].  Default 1000 fs.
                   Should be >= 5--10 x tau_nhc for stability.
    traj_every   : XYZ frame interval [steps].
    thermo_every : log interval [steps].
    skin_width   : Verlet skin [A].
    seed         : RNG seed.

    Output files
    ------------
    trajectory.xyz   -- multi-frame XYZ (cell vectors in comment line)
    thermo.log       -- Step / Time / E_pot / E_kin / T / P_inst / V / H_NPT
    POSCAR_final_NPT -- final structure with updated cell
    md_npt_timeseries.png (if matplotlib available)
    """

    #: eV/A^3 -> GPa conversion
    _EV_ANG3_TO_GPA = EV_ANG3_TO_GPA

    def __init__(self,
                 lattice        : np.ndarray,
                 positions      : np.ndarray,
                 atom_types     : list,
                 # -- universal interface (A) ---------------------------
                 force_fn       : Optional[Callable] = None,
                 r_cut          : Optional[float] = None,
                 # -- legacy Buckingham kwargs (B) ----------------------
                 buck_params    : Optional[dict]  = None,
                 charges        : Optional[dict]  = None,
                 ewald_alpha    : Optional[float] = None,
                 coulomb_method : str   = "ewald",
                 pppm_accuracy  : float = 1e-5,
                 pppm_mesh      : Optional[int] = None,
                 pppm_order     : int   = 6,
                 # -- NPT parameters ------------------------------------
                 temperature    : float = 300.0,
                 pressure       : float = 0.0,     # GPa
                 timestep       : float = 1.0,
                 n_steps        : int   = 10000,
                 equil_steps    : int   = 0,
                 tau_nhc        : float = 100.0,
                 tau_baro       : float = 1000.0,
                 traj_every     : int   = 10,
                 thermo_every   : int   = 100,
                 output_dir     : str   = ".",
                 skin_width     : float = 0.5,
                 seed           : int   = 42,
                 verbose        : bool  = True,
                 masses         : Optional[np.ndarray] = None,
                 fix_atoms      : Optional[np.ndarray] = None,
                 timing         : bool  = False,
                 flush_every_step : bool = True,
                 fsync_every_step : bool = False):

        self.flush_every_step = bool(flush_every_step)
        self.fsync_every_step = bool(fsync_every_step)
        self.lattice      = lattice.copy()
        self.positions    = positions.copy()
        self.atom_types   = list(atom_types)
        self.r_cut        = r_cut
        self.temperature  = temperature
        self.pressure_gpa = pressure
        self.P_ext        = pressure / self._EV_ANG3_TO_GPA
        self.dt           = timestep
        self.n_steps      = n_steps
        self.equil_steps  = equil_steps
        self.tau_nhc      = tau_nhc
        self.tau_baro     = tau_baro
        self.traj_every   = traj_every
        self.thermo_every = thermo_every
        self.output_dir   = output_dir
        self.verbose      = verbose
        self.N            = len(positions)
        self.rng          = np.random.default_rng(seed)

        # -- resolve force function + r_cut --------------------
        self._force_fn, r_cut = _resolve_force_fn(
            force_fn, r_cut, buck_params, charges, ewald_alpha,
            coulomb_method, pppm_accuracy, pppm_mesh, pppm_order,
            "NPTSimulation")
        self.r_cut          = float(r_cut)

        self.buck_params    = buck_params
        self.charges        = charges
        self.ewald_alpha    = ewald_alpha
        self.coulomb_method = coulomb_method
        self.pppm_accuracy  = pppm_accuracy
        self.pppm_mesh      = pppm_mesh
        self.pppm_order     = pppm_order

        self.masses = self._build_masses(atom_types, masses)

        self.velocities = self._init_velocities(temperature)
        # -- fix_atoms constraint mask -------------------------
        # fix_atoms : (N, 3) bool array.  True = frozen (never displaced).
        # Equivalent to VASP Selective Dynamics 'F T T' etc.
        self._fix_mask = self._build_fix_mask(
            self.N, fix_atoms, self.__class__.__name__)
        # Bug-fix note (S8): n_dof now accounts for fix_atoms -- see
        # _BaseMDSimulation._compute_n_dof.
        self.n_dof = self._compute_n_dof(self.N, self._fix_mask)

        self.Q1   = self.n_dof * K_B * temperature * tau_nhc**2
        self.Q2   =              K_B * temperature * tau_nhc**2
        self.xi1  = 0.0;  self.xi2  = 0.0
        self.vxi1 = 0.0;  self.vxi2 = 0.0

        self.W     = (self.n_dof + 3) * K_B * temperature * tau_baro**2
        self.p_eps = 0.0

        self._tc = TimingCollector(enabled=timing)
        self._nl        = NeighbourList(r_cut=self.r_cut, skin_width=skin_width)

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------

    def _kinetic_energy(self, v: Optional[np.ndarray] = None) -> float:
        """E_kin [eV]; delegates to _BaseMDSimulation."""
        if v is None:
            v = self.velocities
        return _BaseMDSimulation._kinetic_energy(v, self.masses)

    def _temperature_inst(self, v: Optional[np.ndarray] = None) -> float:
        """Instantaneous temperature [K]."""
        return 2.0 * self._kinetic_energy(v) / (self.n_dof * K_B)

    def _init_velocities(self, T: float) -> np.ndarray:
        """Maxwell-Boltzmann velocities at T; delegates to base."""
        return _BaseMDSimulation._init_velocities_mb(T, self.masses, self.rng)

    def _volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    def _pressure_inst(self, stress: np.ndarray) -> float:
        """
        Instantaneous pressure [eV/A^3] from kinetic + virial contributions.

        P_inst = 2 E_kin / (3V) + Tr(stress) / 3

        Bug-fix note (NPT barostat sign): this used to compute
        ``Pv = -np.trace(stress) / 3.0``. In THIS codebase's own
        already-verified stress convention (tensile-positive, confirmed
        via finite-difference cross-checks throughout
        ``test_engine_forces.py`` and independently re-derived during
        the Bug C investigation: sigma_ab = -(1/V) dE/dT_ab under
        h'=h@T), a cell compressed BELOW its mechanical equilibrium
        volume has a POSITIVE stress trace (energy increases moving
        away from the minimum in either direction, so dE/ds < 0 for
        scale s < 1, giving sigma = -(1/V)(dE/dT) > 0) -- the opposite
        of the textbook engineering-mechanics convention where P =
        -Tr(sigma)/3, which assumes a different sign convention for
        "tensile-positive" than the one this codebase's own dE/dT
        derivation produces.

        With the old ``Pv = -Tr(stress)/3``, a compressed cell (positive
        stress trace) was reported as having LOWER pressure than the
        target, and the MTTK barostat (``_barostat_half_step``'s
        ``G_eps = 3V(P_inst - P_ext)``, correctly implementing the
        standard restoring dynamics) then drove it to shrink FURTHER
        rather than relieve the compression -- i.e. the feedback loop
        pushed the cell away from equilibrium instead of toward it, in
        BOTH directions (a compressed cell shrank further, an expanded
        cell grew further), confirmed directly: monkey-patching this
        sign and re-running showed the old formula moving volume in the
        wrong direction in both the compressed and expanded case, while
        this fixed formula (``Pv = +Tr(stress)/3``) moves volume in the
        physically correct direction (compressed -> expands, expanded ->
        contracts) with a small, bounded H_NPT drift in both cases (see
        the corresponding test in test_solvers.py).
        """
        V  = self._volume()
        Pk = 2.0 * self._kinetic_energy() / (3.0 * V)
        Pv = np.trace(stress) / 3.0   # stress already divided by V in compute_all
        return Pk + Pv

    def _npt_conserved(self, e_pot: float) -> float:
        """
        NPT conserved quantity [eV]:
        H_NPT = E_kin + E_pot + P_ext V
              + 0.5 W epsilon_dot^2
              + NHC kinetic (0.5 Q1 vxi1^2 + 0.5 Q2 vxi2^2) / EV_ANG_TO_AMU_ANG_FS2
              + NHC potential (n_dof k_B T xi1 + k_B T xi2)
        """
        ek   = self._kinetic_energy()
        V    = self._volume()
        # barostat KE  [eV]
        eps_dot = self.p_eps / self.W   # [fs^-1]
        h_baro  = 0.5 * self.W * eps_dot**2

        # NHC KE  [eV]  (same formula as NVT._nhc_conserved)
        h_nhc_k = ((0.5 * self.Q1 * self.vxi1**2
                    + 0.5 * self.Q2 * self.vxi2**2)
                   / EV_ANG_TO_AMU_ANG_FS2)
        # NHC PE  [eV]
        h_nhc_p = (self.n_dof * K_B * self.temperature * self.xi1
                   + K_B * self.temperature * self.xi2)
        return ek + e_pot + self.P_ext * V + h_baro + h_nhc_k + h_nhc_p

    # ------------------------------------------------------------------
    #  Barostat integration  (isotropic Andersen-Hoover half-step)
    # ------------------------------------------------------------------

    def _barostat_half_step(self, P_inst: float) -> None:
        """
        Half-step update of the barostat momentum p_eps.

        Equation of motion:
            xi_eps = 3 V (P_inst - P_ext) - xi1 p_eps
        Symplectic approximation (xi1 coupling treated via NHC):
            p_eps += 0.5 dt . 3 V (P_inst - P_ext)
        """
        V = self._volume()
        G_eps = 3.0 * V * (P_inst - self.P_ext)   # [eV]
        self.p_eps += 0.5 * self.dt * G_eps

    def _scale_step(self) -> None:
        """
        Full-step cell + position update due to barostat (isotropic).

        epsilon_dot = p_eps / W  [fs^-1]
        dV/dt = 3 V epsilon_dot  ->  V(t+dt) = V(t) exp(3 epsilon_dot dt)
        Linear scale factor per dimension: s = exp(epsilon_dot dt)
        Fractional coordinates are invariant under uniform scaling.
        """
        eps_dot = self.p_eps / self.W       # [fs^-1]
        s       = np.exp(eps_dot * self.dt) # linear scale per axis

        inv_lat       = np.linalg.inv(self.lattice)
        frac          = self.positions @ inv_lat
        frac         -= np.floor(frac)      # wrap fractional coords

        self.lattice  = self.lattice * s    # scale all three axes equally
        self.positions = frac @ self.lattice

        # Velocity rescaling from epsilon_dot coupling (MTTK):
        # v_i -> v_i exp(-epsilon_dot dt)  (volume expansion slows particles)
        self.velocities *= np.exp(-eps_dot * self.dt)

    # ------------------------------------------------------------------
    #  Force evaluation
    # ------------------------------------------------------------------

    def _compute_forces(self) -> Tuple[float, np.ndarray, np.ndarray]:
        self._nl.check_and_rebuild(self.lattice, self.positions)
        return self._force_fn(
            self.lattice, self.positions, self.atom_types,
            neigh_list=self._nl)

    # ------------------------------------------------------------------
    #  Trajectory output
    # ------------------------------------------------------------------

    def _write_xyz_frame(self, fh, step: int,
                          e_pot: float, P_gpa: float) -> None:
        """
        Append one configuration to the open XDATCAR file.

        NPT has a VARIABLE cell, so a full header (comment, scale,
        lattice, species, counts) is written before every frame,
        matching how VASP itself writes XDATCAR for variable-cell
        (NpT) ab-initio MD runs -- a single fixed header (as used for
        NVT/NVE) cannot represent a changing lattice.
        """
        comment = (f"akitsushima.py NPT-MD  step={step}  E_pot={e_pot:.4f}eV  "
                  f"T={self._temperature_inst():.2f}K  P={P_gpa:.4f}GPa  "
                  f"V={self._volume():.4f}Ang3")
        _write_xdatcar_frame(fh, self.positions, self.lattice,
                             self.atom_types, self._xdatcar_frame_idx,
                             variable_header=True, comment=comment)
        self._xdatcar_frame_idx += 1

    # ------------------------------------------------------------------
    #  Main MD loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute the NPT-MD simulation.

        Integration order per step (MTTK Trotter factorisation)
        ---------------------------------------------------------
        1.  NHC half-step
        2.  Barostat half-step  (p_eps update with P at start of step)
        3.  Velocity half-step  (v += 0.5 dt . F/m)
        4.  Scale step          (cell + positions + velocity rescaling)
        5.  Force evaluation at r_{t+1}
        6.  Velocity half-step  (v += 0.5 dt . F_new/m)
        7.  Barostat half-step  (p_eps update with new P)
        8.  NHC half-step

        Returns
        -------
        dict with final positions, velocities, lattice,
        time-averaged T, P, V, and H_NPT drift.
        """
        self._tc.start("total")  # timing
        self._ensure_output_dir(self.output_dir)
        traj_path   = os.path.join(self.output_dir, "XDATCAR_npt")
        thermo_path = os.path.join(self.output_dir, "thermo.log")

        # Initial force evaluation
        e_pot, forces, stress = self._compute_forces()
        acc   = forces * EV_ANG_TO_AMU_ANG_FS2 / self.masses[:, None]
        H_ref = self._npt_conserved(e_pot)

        total_steps = self.equil_steps + self.n_steps
        header = (f"{'Step':>8}  {'Time(ps)':>10}  {'E_pot(eV)':>14}  "
                  f"{'E_kin(eV)':>12}  {'T_inst(K)':>10}  "
                  f"{'P_inst(GPa)':>12}  {'V(Ang3)':>12}  "
                  f"{'H_NPT(eV)':>14}")
        if self.verbose:
            print(f"\nNPT-MD: T={self.temperature} K  "
                  f"P={self.pressure_gpa} GPa  "
                  f"dt={self.dt} fs  steps={total_steps}  "
                  f"(equil={self.equil_steps})")
            print(f"  V0 = {self._volume():.4f} A^3  "
                  f"H_NPT(ref) = {H_ref:.4f} eV")
            print(header)
            print("  " + "-" * len(header))

        T_list, P_list, V_list = [], [], []

        # Only rank 0 opens (and writes to) the real trajectory/thermo
        # files -- every other MPI rank writes to os.devnull instead.
        # See NVTSimulation.run() for the full rationale.
        _traj_path_use   = traj_path   if _mpi_rank == 0 else os.devnull
        _thermo_path_use = thermo_path if _mpi_rank == 0 else os.devnull
        with (open(_traj_path_use,   "w") as traj_fh,
              open(_thermo_path_use, "w") as thermo_fh):
            thermo_fh.write(header + "\n")
            self._xdatcar_frame_idx = 1

            for step in range(total_steps):
                self._tc.start("step")  # timing
                t_ps    = step * self.dt * 1e-3
                P_inst  = self._pressure_inst(stress)   # [eV/A^3]
                P_gpa   = P_inst * self._EV_ANG3_TO_GPA

                # ---- 1. NHC half-step -----------------------------------
                self._nhc_step()

                # ---- 2. Barostat half-step (using current pressure) -----
                self._barostat_half_step(P_inst)

                # ---- 3. Velocity half-step: v += 0.5 dt F/m ------------
                self.velocities += 0.5 * self.dt * acc
                self.velocities *= ~self._fix_mask   # freeze constrained DoF

                # ---- 4. Scale: cell + positions + velocity rescaling ----
                self._scale_step()

                # ---- 5. Force evaluation at new positions ---------------
                e_pot, forces, stress = self._compute_forces()
                acc = forces * EV_ANG_TO_AMU_ANG_FS2 / self.masses[:, None]

                # ---- 6. Velocity half-step (symmetric) -----------------
                self.velocities += 0.5 * self.dt * acc
                self.velocities *= ~self._fix_mask   # freeze constrained DoF

                # ---- 7. Barostat half-step (using new pressure) --------
                P_inst_new = self._pressure_inst(stress)
                self._barostat_half_step(P_inst_new)

                # ---- 8. NHC half-step ----------------------------------
                self._nhc_step()

                # ---- Output --------------------------------------------
                e_kin  = self._kinetic_energy()
                T_inst = self._temperature_inst()
                V_inst = self._volume()
                H_NPT  = self._npt_conserved(e_pot)
                P_gpa  = P_inst_new * self._EV_ANG3_TO_GPA

                is_prod = step >= self.equil_steps
                if is_prod:
                    T_list.append(T_inst)
                    P_list.append(P_gpa)
                    V_list.append(V_inst)

                self._tc.lap("step")  # timing
                if step % self.thermo_every == 0:
                    row = (f"{step:8d}  {t_ps:10.4f}  {e_pot:14.6f}  "
                           f"{e_kin:12.6f}  {T_inst:10.2f}  "
                           f"{P_gpa:12.4f}  {V_inst:12.4f}  "
                           f"{H_NPT:14.6f}")
                    thermo_fh.write(row + "\n")
                    if self.verbose:
                        print("  " + row)

                if is_prod and step % self.traj_every == 0:
                    self._write_xyz_frame(traj_fh, step, e_pot, P_gpa)

                # Push this step's output to disk for real-time
                # monitoring during a long-running MD calculation.
                if self.flush_every_step and (
                        step % self.thermo_every == 0
                        or (is_prod and step % self.traj_every == 0)):
                    self._flush_outputs([thermo_fh, traj_fh],
                                        fsync=self.fsync_every_step)

        # ---- Summary ---------------------------------------------------
        T_mean  = float(np.mean(T_list)) if T_list else 0.0
        T_std   = float(np.std(T_list))  if T_list else 0.0
        P_mean  = float(np.mean(P_list)) if P_list else 0.0
        P_std   = float(np.std(P_list))  if P_list else 0.0
        V_mean  = float(np.mean(V_list)) if V_list else 0.0
        H_drift = self._npt_conserved(e_pot) - H_ref

        out_poscar = os.path.join(self.output_dir, "POSCAR_final_NPT")
        if _mpi_rank == 0:
            write_poscar(out_poscar, self.lattice, self.positions,
                         self.atom_types, title="NPT final frame")

        if self.verbose:
            print(f"\n  NPT-MD finished.")
            print(f"  <T>         = {T_mean:.2f} +/- {T_std:.2f} K  "
                  f"(target {self.temperature} K)")
            print(f"  <P>         = {P_mean:.4f} +/- {P_std:.4f} GPa  "
                  f"(target {self.pressure_gpa} GPa)")
            print(f"  <V>         = {V_mean:.4f} A^3")
            print(f"  H_NPT drift = {H_drift:+.6f} eV  "
                  f"({H_drift / self.N * 1e3:.4f} meV/atom)")
            print(f"  Trajectory  : {traj_path}")
            print(f"  Final POSCAR: {out_poscar}")

        if HAS_MATPLOTLIB and len(T_list) > 1 and _mpi_rank == 0:
            self._save_plots(T_list, P_list, V_list)

        self._tc.stop("total")  # timing
        _timing = self._tc.summary()
        if _timing.get("enabled") and _mpi_rank == 0:
            self._tc.print_summary(label="NPTSimulation")

        return {
            "positions"   : self.positions.copy(),
            "velocities"  : self.velocities.copy(),
            "lattice"     : self.lattice.copy(),
            "T_mean"      : T_mean, "T_std"    : T_std,
            "P_mean_gpa"  : P_mean, "P_std_gpa": P_std,
            "V_mean"      : V_mean,
            "H_npt_drift" : H_drift,
            "timing"     : _timing,
        }

    def _save_plots(self, T_list, P_list, V_list) -> None:
        """Save T(t), P(t), V(t) time-series plots."""
        times = np.arange(len(T_list)) * self.thermo_every * self.dt * 1e-3

        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        axes[0].plot(times, T_list, lw=0.8)
        axes[0].axhline(self.temperature, color="r", ls="--", lw=1,
                        label=f"Target {self.temperature} K")
        axes[0].set_ylabel("Temperature (K)")
        axes[0].legend(fontsize=8)

        axes[1].plot(times, P_list, lw=0.8, color="steelblue")
        axes[1].axhline(self.pressure_gpa, color="r", ls="--", lw=1,
                        label=f"Target {self.pressure_gpa} GPa")
        axes[1].set_ylabel("Pressure (GPa)")
        axes[1].legend(fontsize=8)

        axes[2].plot(times, V_list, lw=0.8, color="darkorange")
        axes[2].set_ylabel("Volume (A^3)")
        axes[2].set_xlabel("Time (ps)")

        if self.equil_steps > 0:
            t_eq = self.equil_steps * self.dt * 1e-3
            for ax in axes:
                ax.axvspan(0, t_eq, alpha=0.1, color="grey",
                           label="Equilibration")

        for ax in axes:
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"NPT MD  --  T = {self.temperature} K  "
            f"P = {self.pressure_gpa} GPa", fontsize=12)
        fig.tight_layout()
        path = os.path.join(self.output_dir, "md_npt_timeseries.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  NPT time-series plot : {path}")
