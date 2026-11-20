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

import warnings

import numpy as np

from ..constants import ATOMIC_MASS, EV_ANG3_TO_GPA, EV_ANG_TO_AMU_ANG_FS2, K_B
from ..engine.neighborlist import NeighbourList, neighbour_list_for, wrap_positions_into_cell
from ..fitting.force_fn import _resolve_force_fn
from ..io.poscar import write_poscar
from ..io.structure import _sort_species
from ..io.trajectory import TrajectoryWriter, check_traj_format, traj_path as _traj_path
from ..io.xdatcar import _write_xdatcar_frame, _write_xdatcar_header
from ..utils.mpi import _mpi_rank
from ..utils.optional_matplotlib import have_matplotlib, plt
from ..utils.timing import TimingCollector
from ._checkpoint import (check_settings, load_checkpoint, output_sizes,
                          save_checkpoint, truncate_outputs)

__all__ = ["NVTSimulation", "LangevinSimulation", "NVESimulation", "NPTSimulation"]


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
    def _initial_or_mb(self, initial_velocities, draw) -> np.ndarray:
        """Velocities to start from: a copy of ``initial_velocities``
        (A/fs, shape (N, 3)) when given, so a run can continue another
        one's trajectory (the ``anneal`` command chains segments this way,
        ADR-0022); otherwise ``draw()``, the Maxwell-Boltzmann draw."""
        if initial_velocities is None:
            return draw()
        v = np.array(initial_velocities, dtype=float)
        if v.shape != (self.N, 3) or not np.all(np.isfinite(v)):
            raise ValueError(
                f"{self.__class__.__name__}: initial_velocities must be a finite "
                f"({self.N}, 3) array, got shape {v.shape}")
        return v

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
    def _validate_md_inputs(self, *, timestep, n_steps, equil_steps,
                            traj_every, thermo_every, temperature=None,
                            tau_nhc=None, tau_baro=None) -> None:
        """Reject settings that cannot produce a meaningful trajectory.

        None of these were checked.  ``thermo_every = 0`` or
        ``traj_every = 0`` raised ZeroDivisionError from ``step %
        self.thermo_every`` partway into the run; ``temperature = 0`` or
        ``tau_nhc = 0`` made ``Q1 = 0``, so G1 became nan and the run
        finished silently producing NaN; and a negative ``timestep`` ran
        to completion with time flowing backwards and reported a
        plausible-looking mean temperature.
        """
        cls = self.__class__.__name__
        if not (timestep > 0.0):
            raise ValueError(f"{cls}: timestep must be > 0, got {timestep}.")
        vrs = getattr(self, "velocity_rescale_steps", 0)
        if vrs > equil_steps:
            # Direct velocity rescaling is a hard projection onto the
            # target temperature, not canonical sampling.  Applied past
            # equil_steps it lands on PRODUCTION frames and biases every
            # average taken from them.  Nothing tied the two together;
            # the shipped templates happen to set them equal.
            warnings.warn(
                f"{cls}: velocity_rescale_steps ({vrs}) exceeds equil_steps "
                f"({equil_steps}), so direct velocity rescaling will be "
                "applied to production frames. That is not canonical "
                "sampling and will bias the averages; keep it at or below "
                "equil_steps.", UserWarning, stacklevel=3)
        if n_steps < 0 or equil_steps < 0:
            raise ValueError(
                f"{cls}: n_steps and equil_steps must be >= 0, got "
                f"{n_steps} and {equil_steps}.")
        if traj_every <= 0 or thermo_every <= 0:
            raise ValueError(
                f"{cls}: traj_every and thermo_every must be >= 1 (they are "
                f"used as `step % every`), got {traj_every} and "
                f"{thermo_every}. Use a large value to disable output.")
        if temperature is not None and not (temperature > 0.0):
            raise ValueError(
                f"{cls}: temperature must be > 0, got {temperature}. The "
                "thermostat mass is proportional to it, so 0 gives a "
                "division by zero and then NaN.")
        for name, tau in (("tau_nhc", tau_nhc), ("tau_baro", tau_baro)):
            if tau is not None and not (tau > 0.0):
                raise ValueError(
                    f"{cls}: {name} must be > 0, got {tau}. The associated "
                    "mass is proportional to its square.")

    def _warn_if_timestep_too_large_for_masses(self) -> None:
        """Warn when the timestep is large relative to the fastest mode.

        The stability limit is set by the shortest vibrational period in
        the system, which scales as sqrt(m).  The default of 1.0 fs is
        comfortable for the metals and for silicon, but an X-H stretch
        has a period of roughly 10 fs, so 1 fs is only a tenth of it --
        marginal, and nothing warned about it.  Mass is the signal
        available here without a phonon calculation.
        """
        m_min = float(np.min(self.masses))
        if m_min < 4.0 and self.dt > 0.5:
            warnings.warn(
                f"{self.__class__.__name__}: timestep {self.dt} fs with a "
                f"lightest mass of {m_min:.3f} amu. An X-H stretch has a "
                "period near 10 fs, so this is at most a tenth of the "
                "fastest period and energy conservation will suffer; 0.25-0.5 "
                "fs is the usual range for hydrogen-containing systems.",
                UserWarning, stacklevel=3)

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
    #  Ensemble uncertainty column (ADR-0033)
    # ------------------------------------------------------------------ #
    def _unc_fn(self):
        fn = getattr(self, "_force_fn", None)
        if getattr(fn, "is_ensemble", False):
            return fn.uncertainty
        return None

    def _unc_header(self) -> str:
        return f"  {'sigF_max(eV/A)':>14}" if self._unc_fn() else ""

    def _unc_cell(self) -> str:
        """sigma_F,max of the ensemble at the last force call (the step
        being logged), or nothing for a single model."""
        fn = self._unc_fn()
        if fn is None:
            return ""
        u = fn()
        return f"  {u['sigma_F_max']:14.6f}" if u else f"  {'nan':>14}"

    # ------------------------------------------------------------------ #
    #  Checkpoint and restart (ADR-0031)
    # ------------------------------------------------------------------ #
    # Thermostat / barostat scalars a checkpoint carries; set per class.
    _CK_SCALARS: Tuple[str, ...] = ()
    _CK_ENSEMBLE = ""

    def _ck_arrays(self) -> dict:
        """Array-valued state a checkpoint carries (name -> array); the
        anisotropic barostat momenta of ADR-0036.  Empty by default."""
        return {}

    # ------------------------------------------------------------------ #
    #  Trajectory output (ADR-0034: xdatcar | extxyz | lammps-dump)
    # ------------------------------------------------------------------ #
    traj_format = "xdatcar"

    def _traj_header(self, fh, lattice, atom_types, comment="") -> None:
        """The XDATCAR header of a fixed cell; the other formats have none."""
        if self.traj_format == "xdatcar":
            _write_xdatcar_header(fh, lattice, atom_types, comment=comment)

    def _traj_frame(self, fh, step, e_pot, forces, stress, variable_cell,
                    comment="") -> None:
        if not hasattr(self, "_traj_species"):
            self._traj_species = _sort_species(self.atom_types)
        w = TrajectoryWriter(fh, self.traj_format, variable_cell=variable_cell,
                             species=self._traj_species)
        w.frame_idx = self._xdatcar_frame_idx
        w.frame(step, self.lattice, self.positions, self.atom_types,
                energy=e_pot, forces=forces, stress=stress,
                comment=comment or "akitsushima MD trajectory")
        self._xdatcar_frame_idx += 1

    def _ck_settings(self) -> dict:
        """Settings a restart must match (ADR-0031 decision 9)."""
        out = {
            "ensemble": self._CK_ENSEMBLE,
            "n_atoms": int(self.N),
            "atom_types": list(self.atom_types),
            "timestep": float(self.dt),
            "equil_steps": int(self.equil_steps),
            "thermo_every": int(self.thermo_every),
            "traj_every": int(self.traj_every),
            "velocity_rescale_steps": int(getattr(self, "velocity_rescale_steps", 0)),
        }
        if self.traj_format != "xdatcar":  # absent for xdatcar: older checkpoints match
            out["traj_format"] = self.traj_format
        if getattr(self, "cell_dof", "iso") != "iso":  # ADR-0036; absent for iso
            out["cell_dof"] = self.cell_dof
        if getattr(self, "baro_thermostat", "nhc") != "nhc":
            out["baro_thermostat"] = self.baro_thermostat
        for name in ("temperature", "tau_nhc", "tau_baro", "pressure_gpa"):
            if hasattr(self, name):
                out[name] = float(getattr(self, name))
        return out

    def _ck_begin(self, checkpoint, checkpoint_meta, outputs: dict):
        """Restore the state from ``checkpoint`` if that file exists.

        Returns None for a fresh run, else a dict with the next step, the
        per-step lists, the extra scalars and the next XDATCAR frame
        index.  The output files are cut back to their length at the
        checkpoint."""
        if not checkpoint or not os.path.isfile(checkpoint):
            return None
        cls = self.__class__.__name__
        meta, arr = load_checkpoint(checkpoint)
        now = dict(self._ck_settings(), **(checkpoint_meta or {}))
        check_settings(meta["settings"], now, cls, checkpoint)
        total = self.equil_steps + self.n_steps
        if meta["step"] > total:
            raise ValueError(
                f"{cls}: the checkpoint {checkpoint} is at step {meta['step']}, "
                f"beyond equil_steps + n_steps = {total}; n_steps can only be "
                "raised to extend a run.")
        self.positions = arr["positions"]
        self.velocities = arr["velocities"]
        self.lattice = arr["lattice"]
        for name in self._CK_SCALARS:
            setattr(self, name, float(meta["scalars"][name]))
        for name in self._ck_arrays():
            setattr(self, name, np.array(arr["state_" + name], dtype=float))
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = meta["rng"]
        truncate_outputs(outputs, meta["sizes"])
        lists = {k[5:]: arr[k].tolist() for k in arr if k.startswith("list_")}
        if "thermo_log_keys" in meta:
            cols = [arr["tlog_" + k].tolist() for k in meta["thermo_log_keys"]]
            self._thermo_log = [dict(zip(meta["thermo_log_keys"], row))
                                for row in zip(*cols)]
        if self.verbose:
            print(f"  {cls}: continuing from {checkpoint} at step {meta['step']}")
        return {"step": int(meta["step"]), "lists": lists,
                "extra": meta["extra"], "frame": int(meta["frame"])}

    def _ck_save(self, checkpoint, checkpoint_meta, step_next, lists: dict,
                 extra: dict, handles: dict) -> None:
        """Write the state after ``step_next`` completed steps."""
        sizes = output_sizes(handles)
        meta = {
            "step": int(step_next),
            "settings": dict(self._ck_settings(), **(checkpoint_meta or {})),
            "scalars": {k: float(getattr(self, k)) for k in self._CK_SCALARS},
            "rng": self.rng.bit_generator.state,
            "sizes": sizes,
            "extra": extra,
            "frame": int(self._xdatcar_frame_idx),
        }
        arrays = {"positions": self.positions, "velocities": self.velocities,
                  "lattice": self.lattice}
        for k, v in self._ck_arrays().items():
            arrays["state_" + k] = np.asarray(v, dtype=float)
        for k, v in lists.items():
            arrays["list_" + k] = np.asarray(v, dtype=float)
        tlog = getattr(self, "_thermo_log", None)
        if tlog:
            keys = list(tlog[0])
            meta["thermo_log_keys"] = keys
            for k in keys:
                arrays["tlog_" + k] = np.array([r[k] for r in tlog])
        save_checkpoint(checkpoint, meta, arrays)

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
    # ------------------------------------------------------------------
    #  Volume and instantaneous pressure (shared by all three ensembles)
    # ------------------------------------------------------------------
    #  These used to live on NPTSimulation alone, and NVESimulation's
    #  thermo.log computed its own P_inst as -Tr(stress)/3 with no
    #  kinetic term -- the wrong sign under this codebase's stress
    #  convention AND missing the ideal-gas contribution.  One shared
    #  implementation is the point: the duplicated cell-gradient
    #  formula that diverged from its twin is recorded in
    #  docs/adr/0001, and this was the same shape of mistake.
    def _volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    def _pressure_inst(self, stress: np.ndarray) -> float:
        """
        Instantaneous pressure [eV/A^3] from kinetic + virial contributions.

        P_inst = 2 E_kin / (3V) + Tr(stress) / 3

        Bug-fix note (NPT barostat sign): this used to compute
        ``Pv = -np.trace(stress) / 3.0``. In THIS codebase's own
        already-verified stress convention (compressive-positive, confirmed
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


class NVTSimulation(_BaseMDSimulation):
    """
    NVT molecular dynamics with Nose-Hoover chain (NHC) thermostat.

    Integrator  : velocity Verlet + NHC of length 2.  _nhc_step applies a
                  SINGLE symmetric sub-step, not a 4-sub-step
                  Suzuki-Yoshida decomposition -- its own docstring
                  explains that the negative SY weight is avoided.
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
    traj_every   : trajectory output interval [steps].
    traj_format  : "xdatcar" (default) | "extxyz" | "lammps-dump" (ADR-0034):
                   XDATCAR_<task>, traj_<task>.extxyz or traj_<task>.lammpstrj.
    thermo_every : thermo log output interval [steps].
    skin_width   : Verlet skin width [A] for neighbour-list lazy rebuild.
    seed         : RNG seed for initial velocity generation.
    initial_velocities : (N, 3) A/fs to start from instead of a
                   Maxwell-Boltzmann draw (continuing a trajectory).
    save_plots   : write md_timeseries plots at the end (default True).

    Output files
    ------------
    XDATCAR_nvt      -- every traj_every-th frame, VASP XDATCAR format
    thermo.log       -- step / time[ps] / E_pot / E_kin / E_tot / T / H_NHC
    POSCAR_final_MD  -- last frame as VASP 5 POSCAR
    """

    _CK_ENSEMBLE = "nvt"
    _CK_SCALARS = ("xi1", "xi2", "vxi1", "vxi2")

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
                 traj_format     : str   = "xdatcar",
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
                 fsync_every_step : bool = False,
                 initial_velocities : Optional[np.ndarray] = None,
                 save_plots     : bool = True):
        self.save_plots     = bool(save_plots)
        self.lattice        = lattice.copy()
        self.positions      = positions.copy()
        self.atom_types     = list(atom_types)
        self.r_cut          = r_cut
        self.temperature    = temperature
        self._validate_md_inputs(
            timestep=timestep, n_steps=n_steps,
            equil_steps=equil_steps, traj_every=traj_every,
            thermo_every=thermo_every,
            temperature=temperature,
           tau_nhc=tau_nhc)
        self.dt             = timestep
        self.n_steps        = n_steps
        self.tau_nhc        = tau_nhc
        self.output_dir     = output_dir
        self.traj_every     = traj_every
        self.traj_format    = check_traj_format(traj_format)
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
        self._warn_if_timestep_too_large_for_masses()

        self.velocities     = self._initial_or_mb(initial_velocities,
                                              self._init_velocities)
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
        self._nl            = neighbour_list_for(self._force_fn, self.r_cut, skin_width)
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
                + 0.5*Q1*vxi1^2 + 0.5*Q2*vxi2^2
                + n_dof*kT*xi1 + kT*xi2

        The chain kinetic term is ALREADY in eV and must not be unit
        converted.  Q1 = n_dof*k_B[eV/K]*T[K]*tau[fs]^2 is eV*fs^2 and
        vxi is fs^-1 (G1 = (2E_kin - n_dof kT)/Q1 is fs^-2, integrated
        over dt), so 0.5*Q1*vxi^2 is eV directly.  Unlike the particle
        kinetic energy it never enters the amu basis.  Dividing it by
        EV_ANG_TO_AMU_ANG_FS2, as this did, inflated it by 103.6x and
        destroyed the invariant: over 3000 steps of 32-atom LJ Al at
        tau_nhc = 50 fs the quantity ranged over 27.13 eV and drifted
        +9.46 eV, against 1.33e-05 eV and +9.75e-06 eV once the division
        is removed -- six orders of magnitude, with conservation itself
        as the oracle.  The dynamics were never affected; the factor
        appears only in this diagnostic.
        """
        ek    = self._kinetic_energy()
        t_nhc = (0.5 * self.Q1 * self.vxi1**2
                 + 0.5 * self.Q2 * self.vxi2**2)
        v_nhc = (self.n_dof * K_B * self.temperature * self.xi1
                 + K_B * self.temperature * self.xi2)
        return ek + e_pot + t_nhc + v_nhc

    def _compute_forces(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluate energy, forces, and stress via self._force_fn."""
        self._nl.check_and_rebuild(self.lattice, self.positions)
        return self._force_fn(
            self.lattice, self.positions, self.atom_types,
            neigh_list=self._nl)

    def _write_trajectory_frame(self, fh, step, e_pot, forces, stress) -> None:
        """Append one configuration to the open trajectory file (fixed
        cell: an XDATCAR carries its header once, written before the
        loop)."""
        self._traj_frame(fh, step, e_pot, forces, stress, variable_cell=False)

    # ------------------------------------------------------------------ #
    #  Main MD loop                                                        #
    # ------------------------------------------------------------------ #

    def run(self, checkpoint: Optional[str] = None, checkpoint_every: int = 0,
            checkpoint_meta: Optional[dict] = None) -> dict:
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

        Checkpoint (ADR-0031)
        ---------------------
        ``checkpoint`` : path of an ``.npz``.  If the file exists, the run
            continues from it: the state, the per-step averages and the
            random-number state are restored, the output files are cut
            back to their length at the checkpoint, and the run goes on
            to ``equil_steps + n_steps`` -- bit for bit the run that was
            not interrupted.  Raising ``n_steps`` extends a finished run.
            The settings must match (a ValueError names any that differ).
        ``checkpoint_every`` : write the checkpoint every this many steps
            (0: only at the end).
        ``checkpoint_meta`` : extra items that must match on a restart
            (the CLI passes a hash of the potential).
        """
        self._tc.start("total")  # timing
        self._ensure_output_dir(self.output_dir)
        traj_path   = _traj_path(self.output_dir, "nvt", self.traj_format)
        thermo_path = os.path.join(self.output_dir, "thermo.log")
        ck = self._ck_begin(checkpoint, checkpoint_meta,
                            {"thermo": thermo_path, "traj": traj_path})

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
            print(header + self._unc_header())
            print("  " + "-" * len(header))

        lists = {"T": [], "e_pot": [], "e_kin": []}
        if ck is not None:
            lists.update(ck["lists"])
        t_avg_list, e_pot_list, e_kin_list = lists["T"], lists["e_pot"], lists["e_kin"]
        start = ck["step"] if ck is not None else 0
        mode = "a" if ck is not None else "w"

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
        with (open(_traj_path_use,   mode) as traj_fh,
              open(_thermo_path_use, mode) as thermo_fh):

            if ck is None:
                thermo_fh.write(header + self._unc_header() + "\n")
                self._traj_header(traj_fh, self.lattice, self.atom_types,
                                     comment=f"akitsushima.py NVT-MD  T={self.temperature}K")
                self._xdatcar_frame_idx = 1
            else:
                self._xdatcar_frame_idx = ck["frame"]
            handles = {"thermo": thermo_fh, "traj": traj_fh}

            for step in range(start, total_steps):
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
                # `frac -= np.floor(frac)` alone maps a tiny negative frac to
                # exactly 1.0 in float64, which scipy's cKDTree rejects; see
                # wrap_positions_into_cell.
                frac -= np.floor(frac)
                frac[frac >= 1.0] = 0.0
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
                    row += self._unc_cell()
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
                    self._write_trajectory_frame(traj_fh, step, e_pot, forces, stress)

                # Push this step's output to disk for real-time
                # monitoring during a long-running MD calculation.
                if self.flush_every_step and (
                        step % self.thermo_every == 0
                        or (is_prod and step % self.traj_every == 0)):
                    self._flush_outputs([thermo_fh, traj_fh],
                                        fsync=self.fsync_every_step)

                if (checkpoint and checkpoint_every
                        and (step + 1) % checkpoint_every == 0):
                    self._ck_save(checkpoint, checkpoint_meta, step + 1,
                                  lists, {}, handles)

            if checkpoint:
                self._ck_save(checkpoint, checkpoint_meta, total_steps,
                              lists, {}, handles)

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
        if (self.save_plots and have_matplotlib() and len(self._thermo_log) > 1
                and _mpi_rank == 0):
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
        # NHC: the chain's conserved quantity; Langevin: E_tot - Q
        h_key = "H_nhc" if "H_nhc" in self._thermo_log[0] else "e_minus_q"
        H_list = [r[h_key] for r in self._thermo_log]

        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        axes[0].plot(times, T_list, lw=0.8)
        axes[0].axhline(self.temperature, color="r", ls="--", lw=1,
                        label=f"Target {self.temperature} K")
        axes[0].set_ylabel("Temperature (K)")
        axes[0].legend(fontsize=8)

        axes[1].plot(times, E_list, lw=0.8, color="steelblue")
        axes[1].set_ylabel("Total energy (eV)")

        axes[2].plot(times, H_list, lw=0.8, color="darkorange")
        axes[2].set_ylabel("NHC conserved qty (eV)" if h_key == "H_nhc"
                           else "E_tot - Q (eV)")
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
#  SECTION 6.1b: NVT molecular dynamics -- Langevin thermostat (BAOAB)
# ============================================================================
#
#  Integrator : BAOAB splitting of Langevin dynamics.
#  Reference  : B. Leimkuhler and C. Matthews, Appl. Math. Res. Express
#               2013, 34; J. Chem. Phys. 138, 174102 (2013).  ADR-0032.
# ============================================================================

class LangevinSimulation(NVTSimulation):
    """
    NVT molecular dynamics with a Langevin thermostat on every atom,
    integrated with BAOAB (ADR-0032).

    One step:  B  v += dt/2 F/m
               A  r += dt/2 v
               O  v  = c v + sqrt(1 - c^2) sqrt(kT/m) xi,  c = exp(-dt/damping)
               A  r += dt/2 v
                  F = F(r)
               B  v += dt/2 F/m

    BAOAB samples configurations with a small time-step error; for a
    harmonic potential <x^2> = kT/k exactly at any stable dt.  After
    each O step the centre-of-mass velocity is removed (unless atoms are
    fixed), so the crystal does not random-walk; the number of degrees
    of freedom is that of NVTSimulation (3N - 3 without fixed atoms).

    Parameters
    ----------
    As :class:`NVTSimulation`, with
    damping : Langevin relaxation time 1/gamma [fs] (default 100).
    ``tau_nhc`` is not used, and ``velocity_rescale_steps`` must be 0:
    the thermostat itself relaxes to the target in a few damping times.

    Output files
    ------------
    As NVTSimulation (XDATCAR_nvt, thermo.log, POSCAR_final_MD).  The
    thermo.log column ``E-Q(eV)`` is E_tot minus the heat Q the
    thermostat has added (O steps and centre-of-mass removal); its drift
    measures the integration error, as H_NHC does for the chain.
    """

    _CK_ENSEMBLE = "langevin"
    _CK_SCALARS: Tuple[str, ...] = ()

    def __init__(self, *args, damping: float = 100.0,
                 velocity_rescale_steps: int = 0, **kwargs):
        if int(velocity_rescale_steps) != 0:
            raise ValueError(
                "LangevinSimulation: velocity_rescale_steps must be 0, got "
                f"{velocity_rescale_steps}.  The Langevin thermostat relaxes "
                "to the target temperature within a few damping times by "
                "itself; rescaling is a Nose-Hoover-chain aid (ADR-0032).")
        if not (damping > 0.0):
            raise ValueError(
                f"LangevinSimulation: damping must be > 0 fs, got {damping}.")
        super().__init__(*args, velocity_rescale_steps=0, **kwargs)
        self.damping = float(damping)

    def _ck_settings(self) -> dict:
        out = super()._ck_settings()
        out.pop("tau_nhc", None)
        out["damping"] = float(self.damping)
        return out

    def _o_step(self, c: float) -> float:
        """Ornstein-Uhlenbeck step over dt, then removal of the
        centre-of-mass velocity; returns the kinetic energy added [eV]."""
        before = self._kinetic_energy()
        sig = np.sqrt((1.0 - c * c) * K_B * self.temperature
                      * EV_ANG_TO_AMU_ANG_FS2 / self.masses)
        self.velocities = (c * self.velocities
                           + sig[:, None] * self.rng.standard_normal((self.N, 3)))
        if np.any(self._fix_mask):
            self.velocities *= ~self._fix_mask
        else:
            self.velocities -= (np.sum(self.masses[:, None] * self.velocities, axis=0)
                                / self.masses.sum())
        return self._kinetic_energy() - before

    def _wrap(self) -> None:
        frac = self.positions @ np.linalg.inv(self.lattice)
        frac -= np.floor(frac)
        frac[frac >= 1.0] = 0.0
        self.positions = frac @ self.lattice

    def run(self, checkpoint: Optional[str] = None, checkpoint_every: int = 0,
            checkpoint_meta: Optional[dict] = None) -> dict:
        """
        Execute the Langevin (BAOAB) NVT run.  ``checkpoint``,
        ``checkpoint_every`` and ``checkpoint_meta`` as in
        :meth:`NVTSimulation.run`; the checkpoint also holds the heat Q
        and the random-number state, so a restart continues bit for bit.
        """
        self._tc.start("total")  # timing
        self._ensure_output_dir(self.output_dir)
        traj_path   = _traj_path(self.output_dir, "nvt", self.traj_format)
        thermo_path = os.path.join(self.output_dir, "thermo.log")
        ck = self._ck_begin(checkpoint, checkpoint_meta,
                            {"thermo": thermo_path, "traj": traj_path})

        e_pot, forces, stress = self._compute_forces()
        acc = forces * EV_ANG_TO_AMU_ANG_FS2 / self.masses[:, None]
        Q = float(ck["extra"]["Q"]) if ck is not None else 0.0
        c = float(np.exp(-self.dt / self.damping))

        total_steps = self.equil_steps + self.n_steps
        header = (f"{'Step':>8}  {'Time(ps)':>10}  {'E_pot(eV)':>14}  "
                  f"{'E_kin(eV)':>12}  {'E_tot(eV)':>12}  "
                  f"{'T_inst(K)':>10}  {'E-Q(eV)':>14}  "
                  f"{'NL_builds':>10}")
        if self.verbose:
            print(f"\nLangevin NVT-MD (BAOAB): T={self.temperature} K  "
                  f"damping={self.damping} fs  dt={self.dt} fs  "
                  f"steps={total_steps}  (equil={self.equil_steps})")
            print(header + self._unc_header())
            print("  " + "-" * len(header))

        lists = {"T": [], "e_pot": [], "e_kin": []}
        if ck is not None:
            lists.update(ck["lists"])
        start = ck["step"] if ck is not None else 0
        mode = "a" if ck is not None else "w"
        _traj_path_use   = traj_path   if _mpi_rank == 0 else os.devnull
        _thermo_path_use = thermo_path if _mpi_rank == 0 else os.devnull
        with (open(_traj_path_use,   mode) as traj_fh,
              open(_thermo_path_use, mode) as thermo_fh):
            if ck is None:
                thermo_fh.write(header + self._unc_header() + "\n")
                self._traj_header(
                    traj_fh, self.lattice, self.atom_types,
                    comment=f"akitsushima.py Langevin NVT-MD  T={self.temperature}K")
                self._xdatcar_frame_idx = 1
            else:
                self._xdatcar_frame_idx = ck["frame"]
            handles = {"thermo": thermo_fh, "traj": traj_fh}

            for step in range(start, total_steps):
                self._tc.start("step")  # timing
                t_ps = step * self.dt * 1e-3
                # B
                self.velocities += 0.5 * self.dt * acc
                self.velocities *= ~self._fix_mask
                # A
                self.positions += 0.5 * self.dt * self.velocities
                # O
                Q += self._o_step(c)
                # A
                self.positions += 0.5 * self.dt * self.velocities
                self._wrap()
                # force, B
                e_pot, forces, stress = self._compute_forces()
                acc = forces * EV_ANG_TO_AMU_ANG_FS2 / self.masses[:, None]
                self.velocities += 0.5 * self.dt * acc
                self.velocities *= ~self._fix_mask

                e_kin  = self._kinetic_energy()
                e_tot  = e_pot + e_kin
                T_inst = self._instantaneous_temperature()
                is_prod = step >= self.equil_steps
                if is_prod:
                    lists["T"].append(T_inst)
                    lists["e_pot"].append(e_pot)
                    lists["e_kin"].append(e_kin)

                self._tc.lap("step")  # timing
                if step % self.thermo_every == 0:
                    row = (f"{step:8d}  {t_ps:10.4f}  {e_pot:14.6f}  "
                           f"{e_kin:12.6f}  {e_tot:12.6f}  "
                           f"{T_inst:10.2f}  {e_tot - Q:14.6f}  "
                           f"{self._nl.rebuild_count:10d}")
                    row += self._unc_cell()
                    thermo_fh.write(row + "\n")
                    if self.verbose:
                        print("  " + row)
                    self._thermo_log.append({
                        "step": step, "time_ps": t_ps, "e_pot": e_pot,
                        "e_kin": e_kin, "e_tot": e_tot, "T_inst": T_inst,
                        "e_minus_q": e_tot - Q,
                        "nl_builds": self._nl.rebuild_count,
                    })
                if is_prod and step % self.traj_every == 0:
                    self._write_trajectory_frame(traj_fh, step, e_pot, forces, stress)
                if self.flush_every_step and (
                        step % self.thermo_every == 0
                        or (is_prod and step % self.traj_every == 0)):
                    self._flush_outputs([thermo_fh, traj_fh],
                                        fsync=self.fsync_every_step)
                if (checkpoint and checkpoint_every
                        and (step + 1) % checkpoint_every == 0):
                    self._ck_save(checkpoint, checkpoint_meta, step + 1,
                                  lists, {"Q": Q}, handles)
            if checkpoint:
                self._ck_save(checkpoint, checkpoint_meta, total_steps,
                              lists, {"Q": Q}, handles)

        out_poscar = os.path.join(self.output_dir, "POSCAR_final_MD")
        if _mpi_rank == 0:
            write_poscar(out_poscar, self.lattice, self.positions,
                         self.atom_types,
                         title=f"Langevin MD final frame  T={self.temperature}K")
        T_arr = np.array(lists["T"])
        T_mean = float(T_arr.mean()) if len(T_arr) else 0.0
        T_std = float(T_arr.std()) if len(T_arr) else 0.0
        Ep_mean = float(np.mean(lists["e_pot"])) if lists["e_pot"] else 0.0
        if self.verbose:
            print(f"\n  Langevin NVT-MD finished.")
            print(f"  <T>      = {T_mean:.2f} +/- {T_std:.2f} K  "
                  f"(target {self.temperature} K)")
            print(f"  <E_pot>  = {Ep_mean:.6f} eV")
            print(f"  Heat added by the thermostat: {Q:.6f} eV")
            print(f"  Trajectory : {traj_path}")
            print(f"  Log        : {thermo_path}")
            print(f"  Final POSCAR: {out_poscar}")
        if (self.save_plots and have_matplotlib() and len(self._thermo_log) > 1
                and _mpi_rank == 0):
            self._save_md_plots()
        self._tc.stop("total")  # timing
        _timing = self._tc.summary()
        if _timing.get("enabled") and _mpi_rank == 0:
            self._tc.print_summary(label="LangevinSimulation")
        return {
            "positions"  : self.positions.copy(),
            "velocities" : self.velocities.copy(),
            "lattice"    : self.lattice.copy(),
            "T_mean"     : T_mean,
            "T_std"      : T_std,
            "e_pot_mean" : Ep_mean,
            "heat_added" : Q,
            "thermo_log" : self._thermo_log,
            "timing"     : _timing,
        }


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
    traj_every   : trajectory output interval [steps].
    traj_format  : "xdatcar" (default) | "extxyz" | "lammps-dump" (ADR-0034):
                   XDATCAR_<task>, traj_<task>.extxyz or traj_<task>.lammpstrj.
    thermo_every : thermo log output interval [steps].
    skin_width   : Verlet skin [A].
    seed         : RNG seed for velocity generation.

    Output files
    ------------
    XDATCAR_nve      -- every traj_every-th frame, VASP XDATCAR format
    thermo.log       -- Step / Time(ps) / E_pot / E_kin / E_tot / T / dE_drift
    POSCAR_final_NVE -- final structure in VASP 5 format
    """

    _CK_ENSEMBLE = "nve"

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
                 traj_format    : str   = "xdatcar",
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
        self._validate_md_inputs(
            timestep=timestep, n_steps=n_steps,
            equil_steps=equil_steps, traj_every=traj_every,
            thermo_every=thermo_every,
            temperature=None)   # NVE: 0 K start is legal
        self.dt             = timestep
        self.n_steps        = n_steps
        self.equil_steps    = equil_steps
        self.temperature    = temperature
        self.traj_every     = traj_every
        self.traj_format    = check_traj_format(traj_format)
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
        self._warn_if_timestep_too_large_for_masses()

        self.velocities = velocities.copy() if velocities is not None \
                          else self._init_velocities(temperature)
        self._nl = neighbour_list_for(self._force_fn, self.r_cut, skin_width)
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

    def _write_trajectory_frame(self, fh, step, e_pot, forces, stress) -> None:
        """Append one configuration to the open trajectory file (fixed cell)."""
        self._traj_frame(fh, step, e_pot, forces, stress, variable_cell=False)

    # ------------------------------------------------------------------
    def run(self, checkpoint: Optional[str] = None, checkpoint_every: int = 0,
            checkpoint_meta: Optional[dict] = None) -> dict:
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

        Checkpoint (ADR-0031)
        ---------------------
        ``checkpoint`` : path of an ``.npz``.  If the file exists, the run
            continues from it: the state, the per-step averages and the
            random-number state are restored, the output files are cut
            back to their length at the checkpoint, and the run goes on
            to ``equil_steps + n_steps`` -- bit for bit the run that was
            not interrupted.  Raising ``n_steps`` extends a finished run.
            The settings must match (a ValueError names any that differ).
        ``checkpoint_every`` : write the checkpoint every this many steps
            (0: only at the end).
        ``checkpoint_meta`` : extra items that must match on a restart
            (the CLI passes a hash of the potential).
        """
        self._tc.start("total")  # timing
        self._ensure_output_dir(self.output_dir)
        traj_path   = _traj_path(self.output_dir, "nve", self.traj_format)
        thermo_path = os.path.join(self.output_dir, "thermo.log")
        ck = self._ck_begin(checkpoint, checkpoint_meta,
                            {"thermo": thermo_path, "traj": traj_path})

        # ---- Initial force evaluation -----------------------------------
        e_pot, forces, stress = self._compute_forces()
        acc = forces * EV_ANG_TO_AMU_ANG_FS2 / self.masses[:, None]
        # Reference total energy (set after first force call so drift = 0 at step 0)
        e_tot_ref = e_pot + self._kinetic_energy()
        if ck is not None:
            e_tot_ref = ck["extra"]["e_tot_ref"]

        total_steps = self.equil_steps + self.n_steps
        GPa_fac = EV_ANG3_TO_GPA
        header = (f"{'Step':>8}  {'Time(ps)':>10}  {'E_pot(eV)':>14}  "
                  f"{'E_kin(eV)':>12}  {'E_tot(eV)':>14}  "
                  f"{'T_inst(K)':>10}  {'P_inst(GPa)':>12}  {'dE_drift(eV)':>14}")
        if self.verbose:
            print(f"\nNVE-MD: dt={self.dt} fs  steps={total_steps}  "
                  f"(equil={self.equil_steps})")
            print(f"  E_tot(ref) = {e_tot_ref:.6f} eV")
            print(header + self._unc_header())
            print("  " + "-" * len(header))

        lists = {"T": [], "e_pot": [], "e_kin": []}
        if ck is not None:
            lists.update(ck["lists"])
        t_list, ep_list, ek_list = lists["T"], lists["e_pot"], lists["e_kin"]
        start = ck["step"] if ck is not None else 0
        mode = "a" if ck is not None else "w"

        # Only rank 0 opens (and writes to) the real trajectory/thermo
        # files -- every other MPI rank writes to os.devnull instead.
        # See NVTSimulation.run() for the full rationale.
        _traj_path_use   = traj_path   if _mpi_rank == 0 else os.devnull
        _thermo_path_use = thermo_path if _mpi_rank == 0 else os.devnull
        with (open(_traj_path_use,   mode) as traj_fh,
              open(_thermo_path_use, mode) as thermo_fh):
            if ck is None:
                thermo_fh.write(header + self._unc_header() + "\n")
                self._traj_header(traj_fh, self.lattice, self.atom_types,
                                     comment="akitsushima.py NVE-MD")
                self._xdatcar_frame_idx = 1
            else:
                self._xdatcar_frame_idx = ck["frame"]
            handles = {"thermo": thermo_fh, "traj": traj_fh}

            for step in range(start, total_steps):
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
                # `frac -= np.floor(frac)` alone maps a tiny negative frac to
                # exactly 1.0 in float64, which scipy's cKDTree rejects; see
                # wrap_positions_into_cell.
                frac -= np.floor(frac)
                frac[frac >= 1.0] = 0.0
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
                if is_prod and not t_list:
                    # Same re-referencing as NPT, for the same reason:
                    # the summary's averages are over the production
                    # window, so its drift should be too.  NVE does not
                    # rescale velocities, so here this only excludes the
                    # initial transient in which the Maxwell-Boltzmann
                    # draw equipartitions into potential energy -- which
                    # conserves E_tot but is not what a reader asking
                    # about integrator quality wants included.  With
                    # equil_steps = 0 this changes nothing.
                    e_tot_ref = e_tot
                    drift = 0.0
                if is_prod:
                    t_list.append(T_inst)
                    ep_list.append(e_pot)
                    ek_list.append(e_kin)

                self._tc.lap("step")  # timing
                if step % self.thermo_every == 0:
                    # Shared helper: kinetic + virial, with the sign
                    # this codebase's stress convention actually
                    # implies.  This line used to read
                    # -np.trace(stress)/3.0, which flipped the sign
                    # and dropped 2*E_kin/(3V) entirely.
                    P_gpa = self._pressure_inst(stress) * GPa_fac
                    row = (f"{step:8d}  {t_ps:10.4f}  {e_pot:14.6f}  "
                           f"{e_kin:12.6f}  {e_tot:14.6f}  "
                           f"{T_inst:10.2f}  {P_gpa:12.4f}  {drift:14.6f}")
                    row += self._unc_cell()
                    thermo_fh.write(row + "\n")
                    if self.verbose:
                        print("  " + row)

                if is_prod and step % self.traj_every == 0:
                    self._write_trajectory_frame(traj_fh, step, e_pot, forces, stress)

                # Push this step's output to disk for real-time
                # monitoring during a long-running MD calculation.
                if self.flush_every_step and (
                        step % self.thermo_every == 0
                        or (is_prod and step % self.traj_every == 0)):
                    self._flush_outputs([thermo_fh, traj_fh],
                                        fsync=self.fsync_every_step)

                if (checkpoint and checkpoint_every
                        and (step + 1) % checkpoint_every == 0):
                    self._ck_save(checkpoint, checkpoint_meta, step + 1, lists,
                                  {"e_tot_ref": float(e_tot_ref)}, handles)

            if checkpoint:
                self._ck_save(checkpoint, checkpoint_meta, total_steps, lists,
                              {"e_tot_ref": float(e_tot_ref)}, handles)

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

#: Cell degrees of freedom of NPTSimulation (ADR-0036).
CELL_DOF_MODES = ("iso", "aniso", "tri", "xy", "xy_iso", "z")


def _cell_generators(mode: str) -> np.ndarray:
    """Strain generators E_k (symmetric 3x3) of a ``cell_dof`` mode.

    The cell moves as h_dot = G h with G = sum_k (p_k / W) E_k (Cartesian,
    rows of the lattice transform as L -> L @ expm(G dt)).  A diagonal
    unit lets one axis breathe, e_ab + e_ba is a shear, and the sum of
    two diagonal units couples those axes (``xy_iso``)."""
    def unit(a, b):
        e = np.zeros((3, 3))
        e[a, b] = 1.0
        e[b, a] = 1.0
        return e
    xx, yy, zz = unit(0, 0), unit(1, 1), unit(2, 2)
    table = {
        "aniso": [xx, yy, zz],
        "tri": [xx, yy, zz, unit(1, 2), unit(0, 2), unit(0, 1)],
        "xy": [xx, yy],
        "xy_iso": [xx + yy],
        "z": [zz],
    }
    if mode not in table:
        raise ValueError(f"cell_dof must be one of {' | '.join(CELL_DOF_MODES)}, "
                         f"got {mode!r}.")
    return np.array(table[mode])


class NPTSimulation(_BaseMDSimulation):
    """
    NPT (isothermal-isobaric) MD: NHC thermostat + isotropic Andersen barostat.

    Reference: Martyna, Tuckerman, Tobias & Klein, Mol. Phys. 87, 1117 (1996).

    With ``cell_dof="iso"`` (default) the cell shape (angles) is
    preserved and only the volume fluctuates.  The other modes (ADR-0036)
    use the MTTK fully flexible cell restricted to a set of strain
    generators E_k (see :func:`_cell_generators`):

        r_dot   = v + G r,                 G = sum_k (p_k / W) E_k
        v_dot   = F/m - (G + Tr(G)/N_f) v - chain
        p_k_dot = V Tr(E_k (P_int - P I)) + Tr(E_k) sum m v^2 / N_f - chain
        H       = E_kin + E_pot + P V + sum_k p_k^2 / 2W + chains

    with P_int = sum m v v^T / V + stress.  For E = I this is the
    isotropic integrator.

    Unit notes
    ----------
    p_eps [eV.fs]      --- barostat momentum, conjugate to the strain
                           variable epsilon (NOT to the volume directly)
    W     [eV.fs^2]    --- barostat mass, W = (n_dof + 3) k_B T tau_baro^2,
                           so epsilon_dot = p_eps / W is [eV.fs]/[eV.fs^2]
                           = [fs^-1]

    (These lines used to read p_V [amu.A^3/fs] and W [amu.A^6], with the
    bracket arithmetic on the second line derived from those.  The
    implementation was always dimensionally consistent; only the
    annotation was wrong -- the barostat never enters the amu basis that
    the particle kinetic energy uses.)
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
    cell_dof     : "iso" (default, volume only) | "aniso" (Lx, Ly, Lz) |
                   "tri" (all six strain components) | "xy" (Lx, Ly;
                   Lz and shear fixed) | "xy_iso" (Lx, Ly together) |
                   "z" (Lz only).  x, y, z are Cartesian.  With ESM only
                   xy and xy_iso at pressure 0 (ADR-0036).
    baro_thermostat : "nhc" (default) -- one 2-link Nose-Hoover chain on
                   the barostat momenta, deterministic, H_NPT conserved --
                   or "langevin" -- an Ornstein-Uhlenbeck step with
                   friction time tau_baro on every barostat momentum
                   (Langevin piston).  In a crystal the strain degrees of
                   freedom barely exchange energy, and one chain sets only
                   their total; use "langevin" when strain or volume
                   FLUCTUATIONS matter (compressibility, elastic constants).
                   H_NPT then subtracts the heat the noise put in, so it
                   stays a drift diagnostic (ADR-0036).
    traj_every   : trajectory output interval [steps].
    traj_format  : "xdatcar" (default) | "extxyz" | "lammps-dump" (ADR-0034):
                   XDATCAR_<task>, traj_<task>.extxyz or traj_<task>.lammpstrj.
    thermo_every : log interval [steps].
    skin_width   : Verlet skin [A].
    seed         : RNG seed.
    initial_velocities : (N, 3) A/fs to start from instead of a
                   Maxwell-Boltzmann draw (continuing a trajectory).
    save_plots   : write the time-series plot at the end (default True).

    Output files
    ------------
    XDATCAR_npt      -- every traj_every-th frame, VASP XDATCAR format
                        (full header per frame, since the cell changes)
    thermo.log       -- Step / Time / E_pot / E_kin / T / P_inst / V / H_NPT
    POSCAR_final_NPT -- final structure with updated cell
    md_npt_timeseries.png (if matplotlib available)
    """

    _CK_ENSEMBLE = "npt"
    _CK_SCALARS = ("xi1", "xi2", "vxi1", "vxi2", "p_eps",
                   "xi1_b", "xi2_b", "vxi1_b", "vxi2_b")

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
                 traj_format    : str   = "xdatcar",
                 thermo_every   : int   = 100,
                 output_dir     : str   = ".",
                 skin_width     : float = 0.5,
                 seed           : int   = 42,
                 verbose        : bool  = True,
                 masses         : Optional[np.ndarray] = None,
                 fix_atoms      : Optional[np.ndarray] = None,
                 timing         : bool  = False,
                 velocity_rescale_steps : int = 0,
                 flush_every_step : bool = True,
                 fsync_every_step : bool = False,
                 initial_velocities : Optional[np.ndarray] = None,
                 save_plots     : bool = True,
                 cell_dof       : str  = "iso",
                 baro_thermostat: str  = "nhc"):
        from ..fitting.force_fn import check_esm_for_solver, uses_esm
        cell_dof = str(cell_dof).lower()
        if cell_dof not in CELL_DOF_MODES:
            raise ValueError(f"NPTSimulation: cell_dof must be one of "
                             f"{' | '.join(CELL_DOF_MODES)}, got {cell_dof!r}.")
        if cell_dof in ("xy", "xy_iso") and uses_esm(force_fn):
            # The in-plane barostat keeps Lz (the ESM boundaries) and the
            # rectangular in-plane cell ESM needs (ADR-0036 decision 5).
            if float(pressure) != 0.0:
                raise ValueError(
                    "NPTSimulation: with ESM electrostatics only pressure 0 is "
                    "allowed -- the pressure of a cell that includes vacuum "
                    f"is ambiguous otherwise (got {pressure} GPa; ADR-0036).")
        else:
            check_esm_for_solver(
                force_fn, "NPTSimulation",
                refuse=f"cell_dof '{cell_dof}' would move the ESM boundaries; "
                       "use cell_dof xy or xy_iso with pressure 0 (ADR-0036)")
        self.cell_dof = cell_dof
        self._iso = cell_dof == "iso"
        self.baro_thermostat = str(baro_thermostat).lower()
        if self.baro_thermostat not in ("nhc", "langevin"):
            raise ValueError("NPTSimulation: baro_thermostat must be nhc | "
                             f"langevin, got {baro_thermostat!r}.")
        self._baro_lv = self.baro_thermostat == "langevin"
        # Heat [eV] the Langevin piston put into the barostat momenta;
        # subtracted from H_NPT (an array so a checkpoint carries it).
        self.baro_heat = np.zeros(1)

        self.flush_every_step = bool(flush_every_step)
        self.fsync_every_step = bool(fsync_every_step)
        self.velocity_rescale_steps = int(velocity_rescale_steps)
        self.save_plots   = bool(save_plots)
        self.lattice      = lattice.copy()
        self.positions    = positions.copy()
        self.atom_types   = list(atom_types)
        self.r_cut        = r_cut
        self.temperature  = temperature
        self.pressure_gpa = pressure
        self.P_ext        = pressure / self._EV_ANG3_TO_GPA
        self._validate_md_inputs(
            timestep=timestep, n_steps=n_steps,
            equil_steps=equil_steps, traj_every=traj_every,
            thermo_every=thermo_every,
            temperature=temperature,
           tau_nhc=tau_nhc, tau_baro=tau_baro)
        self.dt           = timestep
        self.n_steps      = n_steps
        self.equil_steps  = equil_steps
        self.tau_nhc      = tau_nhc
        self.tau_baro     = tau_baro
        self.traj_every   = traj_every
        self.traj_format  = check_traj_format(traj_format)
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
        self._warn_if_timestep_too_large_for_masses()

        self.velocities = self._initial_or_mb(
            initial_velocities, lambda: self._init_velocities(temperature))
        # -- fix_atoms constraint mask -------------------------
        # fix_atoms : (N, 3) bool array.  True = frozen.  Under NPT that
        # means the FRACTIONAL coordinate is frozen: the barostat's
        # affine rescale still moves the atom in Cartesian space with
        # the cell, which is the conventional variable-cell meaning but
        # is not "never displaced" as the NVT/NVE note says.
        # Equivalent to VASP Selective Dynamics 'F T T' etc.
        self._fix_mask = self._build_fix_mask(
            self.N, fix_atoms, self.__class__.__name__)
        # Bug-fix note (S8): n_dof now accounts for fix_atoms -- see
        # _BaseMDSimulation._compute_n_dof.
        self.n_dof = self._compute_n_dof(self.N, self._fix_mask)
        # Zero the frozen components, the way NVTSimulation and
        # NVESimulation both do at this point.  NPT did not, so until
        # _nhc_step happened to zero them on its first call the frozen
        # DoF carried Maxwell-Boltzmann kinetic energy: it inflated the
        # reported T_inst, biased the step-0 barostat kick through the
        # kinetic pressure, and -- because H_ref is taken before that
        # first _nhc_step -- biased the reported H_npt_drift by almost
        # exactly the spurious kinetic energy.
        self.velocities[self._fix_mask] = 0.0

        self.Q1   = self.n_dof * K_B * temperature * tau_nhc**2
        self.Q2   =              K_B * temperature * tau_nhc**2
        self.xi1  = 0.0;  self.xi2  = 0.0
        self.vxi1 = 0.0;  self.vxi2 = 0.0

        self.W     = (self.n_dof + 3) * K_B * temperature * tau_baro**2
        self.p_eps = 0.0
        # Nose-Hoover chain on the BAROSTAT momentum (MTTK requires one;
        # without it p_eps is an undamped Andersen-Hoover degree of
        # freedom and the volume performs large, essentially undamped
        # oscillations instead of sampling the isothermal-isobaric
        # distribution -- measured growing from ~700 to 134825 A^3 over
        # 1 ps on 216-atom LJ starting at its own 0 K equilibrium).
        # One degree of freedom, hence kT rather than n_dof*kT.
        self.Q1_b   = K_B * temperature * tau_baro**2
        self.Q2_b   = K_B * temperature * tau_baro**2
        self.xi1_b  = 0.0;  self.xi2_b  = 0.0
        self.vxi1_b = 0.0;  self.vxi2_b = 0.0

        # Anisotropic barostat (ADR-0036): one momentum per generator,
        # each with the iso mass W; one chain over all of them with the
        # target n_free kT (Q1_b scaled by n_free, as Q1 is by n_dof).
        if not self._iso:
            self._gens = _cell_generators(cell_dof)
            self._gen_tr = np.trace(self._gens, axis1=1, axis2=2)
            self.n_baro = len(self._gens)
            self.p_g = np.zeros(self.n_baro)
            self.Q1_b = self.n_baro * K_B * temperature * tau_baro**2
            self._angle_warned = False

        self._tc = TimingCollector(enabled=timing)
        self._nl        = neighbour_list_for(self._force_fn, self.r_cut, skin_width)

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

    def _npt_conserved(self, e_pot: float) -> float:
        """
        NPT conserved quantity [eV]:
        H_NPT = E_kin + E_pot + P_ext V
              + 0.5 W epsilon_dot^2
              + NHC kinetic (0.5 Q1 vxi1^2 + 0.5 Q2 vxi2^2)   [already eV]
              + NHC potential (n_dof k_B T xi1 + k_B T xi2)
        """
        if not self._iso:
            return self._npt_conserved_g(e_pot)
        ek   = self._kinetic_energy()
        V    = self._volume()
        # barostat KE  [eV]
        eps_dot = self.p_eps / self.W   # [fs^-1]
        h_baro  = 0.5 * self.W * eps_dot**2

        # NHC KE  [eV]  (same formula as NVT._nhc_conserved)
        h_nhc_k = (0.5 * self.Q1 * self.vxi1**2
                   + 0.5 * self.Q2 * self.vxi2**2)   # already eV
        # NHC PE  [eV]
        h_nhc_p = (self.n_dof * K_B * self.temperature * self.xi1
                   + K_B * self.temperature * self.xi2)
        # Barostat chain (1 DoF, so kT rather than n_dof kT)
        kT      = K_B * self.temperature
        h_bnhc  = (0.5 * self.Q1_b * self.vxi1_b**2
                   + 0.5 * self.Q2_b * self.vxi2_b**2      # already eV
                   + kT * self.xi1_b + kT * self.xi2_b)

        return (ek + e_pot + self.P_ext * V + h_baro
                + h_nhc_k + h_nhc_p + h_bnhc) - self.baro_heat[0]

    # ------------------------------------------------------------------
    #  Barostat integration  (isotropic Andersen-Hoover half-step)
    # ------------------------------------------------------------------

    def _nhc_baro_step(self) -> None:
        """2-chain Nose-Hoover half-step acting on the barostat momentum.

        Same propagator structure as :meth:`_BaseMDSimulation._nhc_step`,
        but the thermostatted "kinetic energy" is the barostat's own,
        ``p_eps**2 / W``, and it is a SINGLE degree of freedom -- so the
        target is ``kT``, not ``n_dof kT``.

        MTTK requires this chain; without it p_eps is undamped and the
        volume oscillates without ever settling (see the note in
        __init__).  It is also what makes the ``- (p_xi1_b/Q1_b) p_eps``
        term of the p_eps equation of motion actually present.
        """
        dt = self.dt
        Q1, Q2 = self.Q1_b, self.Q2_b
        kT = K_B * self.temperature

        e_baro = 0.5 * self.p_eps ** 2 / self.W
        G1 = (2.0 * e_baro - kT) / Q1
        G2 = (Q1 * self.vxi1_b ** 2 - kT) / Q2

        self.vxi2_b += G2 * (dt * 0.25)

        s_arg = float(np.clip(-self.vxi2_b * dt * 0.125, -350.0, 350.0))
        sc    = np.exp(s_arg)
        self.vxi1_b = sc * (sc * self.vxi1_b + G1 * dt * 0.25)

        p_arg = float(np.clip(-self.vxi1_b * dt * 0.5, -350.0, 350.0))
        self.p_eps *= np.exp(p_arg)

        self.xi1_b += self.vxi1_b * dt * 0.5
        self.xi2_b += self.vxi2_b * dt * 0.5

        e_baro = 0.5 * self.p_eps ** 2 / self.W
        G1 = (2.0 * e_baro - kT) / Q1
        self.vxi1_b = sc * (sc * self.vxi1_b + G1 * dt * 0.25)

        G2 = (Q1 * self.vxi1_b ** 2 - kT) / Q2
        self.vxi2_b += G2 * (dt * 0.25)

    def _barostat_half_step(self, P_inst: float) -> None:
        """
        Half-step update of the barostat momentum p_eps.

        Equation of motion (Martyna, Tuckerman, Tobias & Klein 1996,
        isotropic cell, d = 3 dimensions, N_f = n_dof):

            p_eps_dot = d V (P_inst - P_ext) + (d/N_f) 2 E_kin
                        - (p_xi1_b / Q1_b) p_eps

        The ``(d/N_f) 2 E_kin`` term is what makes the isotropic
        barostat sample the correct distribution in a FINITE system; it
        was missing, along with the matching ``(1 + d/N_f)`` factor in
        _scale_step's velocity coupling and any thermostat on p_eps.
        The chain coupling is applied separately by _nhc_baro_step.
        """
        V = self._volume()
        G_eps = (3.0 * V * (P_inst - self.P_ext)
                 + (3.0 / self.n_dof) * 2.0 * self._kinetic_energy())  # [eV]
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

        # Free-flight drift.  Without this the atoms never move relative
        # to the cell: the affine rescale below is a similarity transform
        # that leaves every fractional coordinate exactly unchanged, so a
        # trajectory would be one frozen structure with a breathing
        # lattice, evaluated at a force that never changes.  NVTSimulation
        # and NVESimulation apply the identical update; NPT was missing
        # it.  With this line the tau_nhc = tau_baro -> infinity limit
        # reproduces NVE bit-for-bit, which is how the omission was found.
        self.positions += self.dt * self.velocities
        self.positions = wrap_positions_into_cell(self.positions,
                                                  self.lattice)

        frac          = self.positions @ np.linalg.inv(self.lattice)
        self.lattice  = self.lattice * s    # scale all three axes equally
        self.positions = frac @ self.lattice

        # Velocity rescaling from the epsilon_dot coupling.  The MTTK
        # equation of motion carries a factor (1 + d/N_f), not 1 -- it is
        # the partner of the (d/N_f) 2 E_kin term in _barostat_half_step,
        # and the pair is what makes the finite-system distribution come
        # out right.  Only the bare exponential was applied before.
        self.velocities *= np.exp(-(1.0 + 3.0 / self.n_dof)
                                  * eps_dot * self.dt)

    # ------------------------------------------------------------------
    #  Anisotropic barostat (ADR-0036)
    # ------------------------------------------------------------------

    def _ck_arrays(self) -> dict:
        out = {} if self._iso else {"p_g": self.p_g}
        if self._baro_lv:
            out["baro_heat"] = self.baro_heat
        return out

    def _baro_nhc(self) -> None:
        if self._baro_lv:
            self._baro_langevin()
        elif self._iso:
            self._nhc_baro_step()
        else:
            self._nhc_baro_step_g()

    def _baro_langevin(self) -> None:
        """Half-step Ornstein-Uhlenbeck update of every barostat momentum
        (Langevin piston; Feller, Zhang, Pastor & Brooks, J. Chem. Phys.
        103, 4613 (1995); Quigley & Probert, J. Chem. Phys. 120, 11432
        (2004) for a flexible cell):

            p <- c p + sqrt((1 - c^2) W kT) xi,   c = exp(-dt / (2 tau_baro))

        exact for the Gaussian marginal of p, so each strain degree of
        freedom is thermalised on its own.  The kinetic energy it adds is
        accumulated in ``baro_heat``."""
        kT = K_B * self.temperature
        c = np.exp(-0.5 * self.dt / self.tau_baro)
        p = np.atleast_1d(self.p_eps if self._iso else self.p_g).astype(float)
        new = c * p + np.sqrt((1.0 - c * c) * self.W * kT) * \
            self.rng.standard_normal(p.shape)
        self.baro_heat[0] += 0.5 * float(new @ new - p @ p) / self.W
        if self._iso:
            self.p_eps = float(new[0])
        else:
            self.p_g = new

    def _baro_half(self, P_inst: float, stress: np.ndarray) -> None:
        if self._iso:
            self._barostat_half_step(P_inst)
        else:
            self._barostat_half_step_g(stress)

    def _baro_scale(self) -> None:
        if self._iso:
            self._scale_step()
        else:
            self._scale_step_g()

    def _pressure_tensor(self, stress: np.ndarray) -> np.ndarray:
        """Instantaneous pressure tensor [eV/A^3]: sum m v v^T / V + stress
        (symmetrised).  Its trace / 3 is :meth:`_pressure_inst`."""
        v = self.velocities
        kin = (v.T * self.masses) @ v / EV_ANG_TO_AMU_ANG_FS2   # [eV]
        s = np.asarray(stress, dtype=float)
        return kin / self._volume() + 0.5 * (s + s.T)

    def _G(self) -> np.ndarray:
        """Cell strain rate G = sum_k (p_k / W) E_k  [fs^-1]."""
        return np.einsum("k,kab->ab", self.p_g / self.W, self._gens)

    def _npt_conserved_g(self, e_pot: float) -> float:
        kT = K_B * self.temperature
        h_baro = 0.5 * float(np.sum(self.p_g ** 2)) / self.W
        h_nhc = (0.5 * self.Q1 * self.vxi1**2 + 0.5 * self.Q2 * self.vxi2**2
                 + self.n_dof * kT * self.xi1 + kT * self.xi2)
        h_bnhc = (0.5 * self.Q1_b * self.vxi1_b**2
                  + 0.5 * self.Q2_b * self.vxi2_b**2
                  + self.n_baro * kT * self.xi1_b + kT * self.xi2_b)
        return (self._kinetic_energy() + e_pot + self.P_ext * self._volume()
                + h_baro + h_nhc + h_bnhc) - self.baro_heat[0]

    def _nhc_baro_step_g(self) -> None:
        """:meth:`_nhc_baro_step` for n_free barostat momenta: the chain
        target is n_free kT and every p_k is scaled by the same factor."""
        dt = self.dt
        Q1, Q2 = self.Q1_b, self.Q2_b
        kT = K_B * self.temperature
        nkT = self.n_baro * kT

        e_baro = 0.5 * float(np.sum(self.p_g ** 2)) / self.W
        G1 = (2.0 * e_baro - nkT) / Q1
        G2 = (Q1 * self.vxi1_b ** 2 - kT) / Q2

        self.vxi2_b += G2 * (dt * 0.25)

        s_arg = float(np.clip(-self.vxi2_b * dt * 0.125, -350.0, 350.0))
        sc    = np.exp(s_arg)
        self.vxi1_b = sc * (sc * self.vxi1_b + G1 * dt * 0.25)

        p_arg = float(np.clip(-self.vxi1_b * dt * 0.5, -350.0, 350.0))
        self.p_g = self.p_g * np.exp(p_arg)

        self.xi1_b += self.vxi1_b * dt * 0.5
        self.xi2_b += self.vxi2_b * dt * 0.5

        e_baro = 0.5 * float(np.sum(self.p_g ** 2)) / self.W
        G1 = (2.0 * e_baro - nkT) / Q1
        self.vxi1_b = sc * (sc * self.vxi1_b + G1 * dt * 0.25)

        G2 = (Q1 * self.vxi1_b ** 2 - kT) / Q2
        self.vxi2_b += G2 * (dt * 0.25)

    def _barostat_half_step_g(self, stress: np.ndarray) -> None:
        """p_k += dt/2 [V Tr(E_k (P_int - P I)) + Tr(E_k) sum m v^2 / N_f]."""
        V = self._volume()
        D = self._pressure_tensor(stress) - self.P_ext * np.eye(3)
        mv2 = 2.0 * self._kinetic_energy()
        force = (V * np.einsum("kab,ab->k", self._gens, D)
                 + self._gen_tr * mv2 / self.n_dof)                  # [eV]
        self.p_g = self.p_g + 0.5 * self.dt * force

    def _scale_step_g(self) -> None:
        """Drift, then L -> L expm(G dt), r -> r expm(G dt) and
        v -> v expm(-(G + Tr(G)/N_f) dt).

        A diagonal G is exponentiated element-wise, so an axis whose
        generator is absent is multiplied by exactly 1.0 and stays
        bit-identical (Lz in xy mode)."""
        G = self._G()
        dt = self.dt
        self.positions += dt * self.velocities
        self.positions = wrap_positions_into_cell(self.positions, self.lattice)
        a = np.trace(G) / self.n_dof
        if not np.any(G - np.diag(np.diag(G))):
            g = np.diag(G)
            M = np.diag(np.exp(g * dt))
            Mv = np.diag(np.exp(-(g + a) * dt))
        else:
            w, U = np.linalg.eigh(G)
            M = (U * np.exp(w * dt)) @ U.T
            Mv = (U * np.exp(-(w + a) * dt)) @ U.T
        self.lattice = self.lattice @ M
        self.positions = self.positions @ M
        self.velocities = self.velocities @ Mv
        if self.cell_dof == "tri" and not self._angle_warned:
            ang = self._cell_angles()
            if np.any(ang < 45.0) or np.any(ang > 135.0):
                self._angle_warned = True
                warnings.warn(
                    "NPTSimulation (cell_dof tri): a cell angle left [45, 135] "
                    f"degrees (alpha, beta, gamma = {ang.round(2).tolist()}); "
                    "cells are not flipped (ADR-0036).", stacklevel=2)

    def _cell_angles(self) -> np.ndarray:
        """alpha, beta, gamma [deg] of the current lattice."""
        a, b, c = self.lattice
        def ang(u, w):
            return np.degrees(np.arccos(np.clip(
                u @ w / (np.linalg.norm(u) * np.linalg.norm(w)), -1.0, 1.0)))
        return np.array([ang(b, c), ang(a, c), ang(a, b)])

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

    def _write_trajectory_frame(self, fh, step: int, e_pot: float,
                                P_gpa: float, forces=None, stress=None) -> None:
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
        self._traj_frame(fh, step, e_pot, forces, stress, variable_cell=True,
                         comment=comment)

    # ------------------------------------------------------------------
    #  Main MD loop
    # ------------------------------------------------------------------

    def run(self, checkpoint: Optional[str] = None, checkpoint_every: int = 0,
            checkpoint_meta: Optional[dict] = None) -> dict:
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

        Checkpoint (ADR-0031)
        ---------------------
        ``checkpoint`` : path of an ``.npz``.  If the file exists, the run
            continues from it: the state, the per-step averages and the
            random-number state are restored, the output files are cut
            back to their length at the checkpoint, and the run goes on
            to ``equil_steps + n_steps`` -- bit for bit the run that was
            not interrupted.  Raising ``n_steps`` extends a finished run.
            The settings must match (a ValueError names any that differ).
        ``checkpoint_every`` : write the checkpoint every this many steps
            (0: only at the end).
        ``checkpoint_meta`` : extra items that must match on a restart
            (the CLI passes a hash of the potential).
        """
        self._tc.start("total")  # timing
        self._ensure_output_dir(self.output_dir)
        traj_path   = _traj_path(self.output_dir, "npt", self.traj_format)
        thermo_path = os.path.join(self.output_dir, "thermo.log")
        ck = self._ck_begin(checkpoint, checkpoint_meta,
                            {"thermo": thermo_path, "traj": traj_path})

        # Initial force evaluation
        e_pot, forces, stress = self._compute_forces()
        acc   = forces * EV_ANG_TO_AMU_ANG_FS2 / self.masses[:, None]
        H_ref = self._npt_conserved(e_pot)
        if ck is not None:
            H_ref = ck["extra"]["H_ref"]

        total_steps = self.equil_steps + self.n_steps
        header = (f"{'Step':>8}  {'Time(ps)':>10}  {'E_pot(eV)':>14}  "
                  f"{'E_kin(eV)':>12}  {'T_inst(K)':>10}  "
                  f"{'P_inst(GPa)':>12}  {'V(Ang3)':>12}  "
                  f"{'H_NPT(eV)':>14}")
        if not self._iso:
            header += "".join(f"  {c:>10}" for c in self._extra_cols())
        if self.verbose:
            print(f"\nNPT-MD: T={self.temperature} K  "
                  f"P={self.pressure_gpa} GPa  "
                  f"dt={self.dt} fs  steps={total_steps}  "
                  f"(equil={self.equil_steps})"
                  + ("" if self._iso else f"  cell_dof={self.cell_dof}")
                  + ("  barostat: Langevin piston" if self._baro_lv else ""))
            print(f"  V0 = {self._volume():.4f} A^3  "
                  f"H_NPT(ref) = {H_ref:.4f} eV")
            print(header + self._unc_header())
            print("  " + "-" * len(header))

        lists = {"T": [], "P": [], "V": []}
        if not self._iso:
            for k in self._extra_lists():
                lists[k] = []
        if ck is not None:
            lists.update(ck["lists"])
        T_list, P_list, V_list = lists["T"], lists["P"], lists["V"]
        start = ck["step"] if ck is not None else 0
        mode = "a" if ck is not None else "w"

        # Only rank 0 opens (and writes to) the real trajectory/thermo
        # files -- every other MPI rank writes to os.devnull instead.
        # See NVTSimulation.run() for the full rationale.
        _traj_path_use   = traj_path   if _mpi_rank == 0 else os.devnull
        _thermo_path_use = thermo_path if _mpi_rank == 0 else os.devnull
        with (open(_traj_path_use,   mode) as traj_fh,
              open(_thermo_path_use, mode) as thermo_fh):
            if ck is None:
                thermo_fh.write(header + self._unc_header() + "\n")
                self._xdatcar_frame_idx = 1
            else:
                self._xdatcar_frame_idx = ck["frame"]
            handles = {"thermo": thermo_fh, "traj": traj_fh}

            for step in range(start, total_steps):
                self._tc.start("step")  # timing
                t_ps    = step * self.dt * 1e-3
                P_inst  = self._pressure_inst(stress)   # [eV/A^3]
                P_gpa   = P_inst * self._EV_ANG3_TO_GPA

                # ---- 1. NHC half-steps (particles, then barostat) -------
                self._nhc_step()
                self._baro_nhc()

                # ---- 2. Barostat half-step (using current pressure) -----
                self._baro_half(P_inst, stress)

                # ---- 3. Velocity half-step: v += 0.5 dt F/m ------------
                self.velocities += 0.5 * self.dt * acc
                self.velocities *= ~self._fix_mask   # freeze constrained DoF

                # ---- 4. Scale: cell + positions + velocity rescaling ----
                self._baro_scale()

                # ---- 5. Force evaluation at new positions ---------------
                e_pot, forces, stress = self._compute_forces()
                acc = forces * EV_ANG_TO_AMU_ANG_FS2 / self.masses[:, None]

                # ---- 6. Velocity half-step (symmetric) -----------------
                self.velocities += 0.5 * self.dt * acc
                self.velocities *= ~self._fix_mask   # freeze constrained DoF

                # ---- 7. Barostat half-step (using new pressure) --------
                P_inst_new = self._pressure_inst(stress)
                self._baro_half(P_inst_new, stress)

                # ---- 8. NHC half-steps (barostat, then particles) ------
                self._baro_nhc()
                self._nhc_step()

                # ---- Optional velocity rescaling (early steps only) -----
                # Same equilibration-transient guard NVTSimulation applies;
                # see the comment there.  cmd_md and the shipped NPT
                # template both already passed this parameter, which
                # NPTSimulation did not accept -- `akitsushima md` with
                # ensemble: npt raised TypeError before reaching step 0.
                if (self.velocity_rescale_steps > 0
                        and step < self.velocity_rescale_steps):
                    ek_now = self._kinetic_energy()
                    T_now  = (2.0 * ek_now / (self.n_dof * K_B)
                              if ek_now > 0 else 0.0)
                    if T_now > 1e-12:
                        self.velocities *= np.sqrt(self.temperature / T_now)
                        self.velocities *= ~self._fix_mask

                # ---- Output --------------------------------------------
                e_kin  = self._kinetic_energy()
                T_inst = self._temperature_inst()
                V_inst = self._volume()
                H_NPT  = self._npt_conserved(e_pot)
                P_gpa  = P_inst_new * self._EV_ANG3_TO_GPA

                is_prod = step >= self.equil_steps
                if is_prod:
                    # Re-reference the conserved quantity at the first
                    # production step.  H_ref was taken before the loop,
                    # so the reported drift spanned the equilibration
                    # phase -- during which velocity_rescale_steps
                    # DELIBERATELY rescales velocities and therefore
                    # breaks conservation.  The number that came out was
                    # not an integrator diagnostic at all: a healthy
                    # 128-atom GeSn run reported 3.2053 eV of "H_NPT
                    # drift" against 0.0007 eV across its actual
                    # production window, a factor of 4600.  With
                    # equil_steps = 0 this changes nothing.
                    if not T_list:
                        H_ref = H_NPT
                    T_list.append(T_inst)
                    P_list.append(P_gpa)
                    V_list.append(V_inst)
                    if not self._iso:
                        xv = self._extra_values(stress)
                        for k in self._extra_lists():
                            lists[k].append(xv[k])

                self._tc.lap("step")  # timing
                if step % self.thermo_every == 0:
                    row = (f"{step:8d}  {t_ps:10.4f}  {e_pot:14.6f}  "
                           f"{e_kin:12.6f}  {T_inst:10.2f}  "
                           f"{P_gpa:12.4f}  {V_inst:12.4f}  "
                           f"{H_NPT:14.6f}")
                    if not self._iso:
                        xv = self._extra_values(stress)
                        row += "".join(f"  {xv[k]:10.4f}"
                                       for k in self._extra_cols())
                    row += self._unc_cell()
                    thermo_fh.write(row + "\n")
                    if self.verbose:
                        print("  " + row)

                if is_prod and step % self.traj_every == 0:
                    self._write_trajectory_frame(traj_fh, step, e_pot, P_gpa, forces, stress)

                # Push this step's output to disk for real-time
                # monitoring during a long-running MD calculation.
                if self.flush_every_step and (
                        step % self.thermo_every == 0
                        or (is_prod and step % self.traj_every == 0)):
                    self._flush_outputs([thermo_fh, traj_fh],
                                        fsync=self.fsync_every_step)

                if (checkpoint and checkpoint_every
                        and (step + 1) % checkpoint_every == 0):
                    self._ck_save(checkpoint, checkpoint_meta, step + 1, lists,
                                  {"H_ref": float(H_ref)}, handles)

            if checkpoint:
                self._ck_save(checkpoint, checkpoint_meta, total_steps, lists,
                              {"H_ref": float(H_ref)}, handles)

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

        if (self.save_plots and have_matplotlib() and len(T_list) > 1
                and _mpi_rank == 0):
            self._save_plots(T_list, P_list, V_list,
                             extra=None if self._iso else lists)

        self._tc.stop("total")  # timing
        _timing = self._tc.summary()
        if _timing.get("enabled") and _mpi_rank == 0:
            self._tc.print_summary(label="NPTSimulation")

        out = {
            "positions"   : self.positions.copy(),
            "velocities"  : self.velocities.copy(),
            "lattice"     : self.lattice.copy(),
            "T_mean"      : T_mean, "T_std"    : T_std,
            "P_mean_gpa"  : P_mean, "P_std_gpa": P_std,
            "V_mean"      : V_mean,
            "H_npt_drift" : H_drift,
            "timing"     : _timing,
        }
        if not self._iso:
            out.update(self._extra_summary(lists))
        return out

    # ---- anisotropic output (ADR-0036) ---------------------------------
    _P_KEYS = ("Pxx", "Pyy", "Pzz", "Pyz", "Pxz", "Pxy")
    _P_IDX = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))

    def _extra_cols(self):
        """thermo.log columns added for cell_dof != iso."""
        if self.cell_dof == "tri":
            return (*self._P_KEYS, "a", "b", "c", "alpha", "beta", "gamma")
        return ("Pxx", "Pyy", "Pzz", "a", "b", "c")

    def _extra_lists(self):
        return (*self._P_KEYS, "a", "b", "c", *(f"L{i}" for i in range(9)))

    def _extra_values(self, stress) -> dict:
        Pt = self._pressure_tensor(stress) * self._EV_ANG3_TO_GPA
        out = {k: float(Pt[i, j]) for k, (i, j) in zip(self._P_KEYS, self._P_IDX)}
        for k, x in zip("abc", np.linalg.norm(self.lattice, axis=1)):
            out[k] = float(x)
        for k, x in zip(("alpha", "beta", "gamma"), self._cell_angles()):
            out[k] = float(x)
        for i, x in enumerate(self.lattice.ravel()):
            out[f"L{i}"] = float(x)
        return out

    def _extra_summary(self, lists) -> dict:
        if not lists["a"]:
            return {}
        Pm = np.zeros((3, 3))
        for k, (i, j) in zip(self._P_KEYS, self._P_IDX):
            Pm[i, j] = Pm[j, i] = float(np.mean(lists[k]))
        abc = np.array([lists["a"], lists["b"], lists["c"]])
        return {
            "cell_dof": self.cell_dof,
            "lattice_mean": np.array(
                [np.mean(lists[f"L{i}"]) for i in range(9)]).reshape(3, 3),
            "P_tensor_mean_gpa": Pm,
            "abc_mean": abc.mean(axis=1),
            "abc_std": abc.std(axis=1),
        }

    def _save_plots(self, T_list, P_list, V_list, extra=None) -> None:
        """Save T(t), P(t), V(t) time-series plots (plus Pxx/Pyy/Pzz and
        a/b/c panels for cell_dof != iso).

        The lists hold one entry per production step, so entry i is at
        step equil_steps + i (this used thermo_every * dt per entry, which
        stretched the time axis by thermo_every)."""
        times = (self.equil_steps + np.arange(len(T_list))) * self.dt * 1e-3

        n_ax = 3 if extra is None else 5
        fig, axes = plt.subplots(n_ax, 1, figsize=(8, 3 * n_ax), sharex=True)
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
        axes[-1].set_xlabel("Time (ps)")
        if extra is not None:
            t_x = times
            for k in ("Pxx", "Pyy", "Pzz"):
                axes[3].plot(t_x, extra[k], lw=0.6, label=k)
            axes[3].axhline(self.pressure_gpa, color="r", ls="--", lw=1)
            axes[3].set_ylabel("P_aa (GPa)")
            axes[3].legend(fontsize=8)
            for k in "abc":
                axes[4].plot(t_x, extra[k], lw=0.8, label=k)
            axes[4].set_ylabel("Length (A)")
            axes[4].legend(fontsize=8)

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
