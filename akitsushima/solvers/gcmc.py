"""Grand-canonical Monte Carlo + MD hybrid simulations (muVT ensemble):
GCMCMDSimulation (discrete insertion/deletion/displacement moves) and
CFCMCSimulation (a continuous fractional atom, ADR-0009).
"""
import os
import warnings
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..constants import ATOMIC_MASS, EV_ANG_TO_AMU_ANG_FS2, K_B
from ..engine.neighborlist import NeighbourList
from ..fitting.mlip_engine import check_mlip_for_solver
from ..io.poscar import write_poscar
from ..io.xdatcar import _write_xdatcar_frame
from ..utils.mpi import _mpi_rank
from ..utils.optional_matplotlib import HAS_MATPLOTLIB, plt
from ..utils.timing import TimingCollector
from .molecular_dynamics import _BaseMDSimulation

__all__ = ["GCMCMDSimulation", "CFCMCSimulation"]


class GCMCMDSimulation(_BaseMDSimulation):
    """
    Grand-canonical molecular dynamics via GCMC+MD hybridisation (muVT).

    Alternates between short NVT-MD segments (Nose-Hoover thermostat) and
    Grand-Canonical Monte Carlo cycles (insertion / deletion / displacement).

    Parameters
    ----------
    lattice / positions / atom_types : initial structure.
    force_fn    : unified force callable from ``make_force_fn()``.
    temperature : target temperature T [K].
    mu          : FULL chemical potential mu [eV].  Controls the average
                  particle number: more negative -> fewer particles,
                  more positive -> more.  The acceptance rule below is
                  the textbook grand-canonical form (Frenkel & Smit,
                  *Understanding Molecular Simulation*, eq. 5.6.8),

                      acc(N->N+1) = min(1, V/(lambda^3 (N+1))
                                           * exp(beta (mu - dU)))

                  whose ``mu`` is the full chemical potential: writing
                  ``mu = mu_id + mu_ex`` (``mu_ex`` the EXCESS chemical
                  potential) collapses the prefactor to ``N/(N+1)``, so
                  keeping ``V/(lambda^3 (N+1))`` while being handed an
                  excess mu double-counts the ideal-gas term.

                  Rename note (bug-fix, documentation only -- not
                  behaviour).  This parameter was originally named
                  ``mu_ex`` and documented as the excess chemical
                  potential, which was wrong and would make a caller
                  supply a value too large by ``|mu_ideal| =
                  |kT ln(lambda^3 rho)|`` -- 9.5 kT for supercritical Ar
                  at 200 K and rho* = 0.4, i.e. larger than the whole
                  range a user typically tunes over.  Settled by direct
                  calculation on an IDEAL GAS, where
                  ``<N> = V exp(beta mu) / lambda^3`` exactly and no
                  reference data or fitting is involved: passing
                  ``mu = kT ln(N_target lambda^3 / V)`` reproduced
                  N_target = 30, 60 and 120 to within 1-3% (see
                  test_gcmc_mu_is_the_full_chemical_potential).  The
                  excess reading also cannot be right on its own terms:
                  for an ideal gas ``mu_ex == 0`` at EVERY density, so it
                  cannot select one, whereas the full mu can.  The
                  formula was always correct; only the old name and its
                  documentation were wrong, so the parameter is renamed
                  to ``mu`` here.  ``mu_ex=`` is kept as a deprecated
                  constructor alias (mutually exclusive with ``mu=``,
                  warns via :class:`UserWarning`), and reading
                  ``.mu_ex`` on a constructed instance is likewise kept
                  as a deprecated, warning property returning ``.mu``.
    species     : element symbol of the species whose number fluctuates.
    mass        : atomic mass of the fluctuating species [amu].
                  Defaults to ``ATOMIC_MASS[species]``.
    timestep    : MD integration time step [fs].
    n_cycles    : total number of GCMC+MD cycles.
    md_steps_per_cycle     : NVT-MD steps between GCMC batches.
    gcmc_attempts_per_cycle: trial moves (insert / delete / displace) per cycle.
    ins_del_ratio  : fraction of moves that are insert-or-delete
                     (remainder are displacements).  Default 0.5.
    max_disp    : maximum displacement amplitude for displacement moves [A].
    tau_nhc     : Nose-Hoover chain time constant [fs].
    traj_every  : XYZ trajectory write interval [cycles].
    thermo_every: thermo log write interval [cycles].
    output_dir  : directory for output files.
    seed        : RNG seed.
    verbose     : print progress.

    Output files
    ------------
    gcmc_md_trajectory.xyz  -- particle positions at each ``traj_every`` cycle.
    gcmc_md_thermo.log      -- cycle / N / E_pot / T_inst / acc_ins / acc_del.
    N_vs_cycle.dat          -- particle-number time series.
    POSCAR_gcmc_final       -- final structure.

    Notes
    -----
    *Cavity-biased insertion* is enabled automatically when
    ``cavity_bias=True`` (default False).  It pre-screens random positions
    against a coarse grid and preferentially places trial particles in low-
    energy voids.  The required bias correction factor is applied
    automatically so that detailed balance is preserved.

    *Energy-biased deletion* (``energy_bias_del=True``, default False) picks
    the deletion candidate with probability proportional to
    ``exp(+beta u_i)`` (local energy) rather than uniformly.  The bias
    correction  ``exp(-beta u_i) / Z``  is applied to the acceptance criterion.

    Both extensions can be combined independently.

    Example
    -------
    >>> engine = PotentialEngine().add(LennardJonesPotential(lj, r_cut=10.))
    >>> fn = make_force_fn(engine)
    >>> gc = GCMCMDSimulation(
    ...         lattice, positions, ['Ar']*N, fn,
    ...         temperature=120., mu=-0.05,
    ...         species='Ar', mass=39.948,
    ...         n_cycles=2000, md_steps_per_cycle=20,
    ...         gcmc_attempts_per_cycle=N,
    ...         output_dir='gcmd_run')
    >>> result = gc.run()
    >>> print(f"<N> = {result['N_mean']:.1f}")
    """

    # Task name used to build the XDATCAR output filename
    # (XDATCAR_<_xdatcar_task_name>).  Subclasses (e.g. CFCMCSimulation)
    # override this to get their own distinct filename while still
    # sharing this class's run() method via super().run().
    _xdatcar_task_name = "gcmc"

    # A reason this solver takes no MLIP at all (None: it does, if
    # atom-centred).  See check_mlip_for_solver.
    _MLIP_REFUSAL = None

    def __init__(self,
                 lattice         : np.ndarray,
                 positions       : np.ndarray,
                 atom_types      : list,
                 force_fn        : Callable,
                 temperature     : float = 300.0,
                 mu              : Optional[float] = None,
                 species         : str   = "Ar",
                 mass            : Optional[float] = None,
                 timestep        : float = 1.0,
                 n_cycles        : int   = 1000,
                 md_steps_per_cycle      : int   = 20,
                 gcmc_attempts_per_cycle : int   = 50,
                 ins_del_ratio   : float = 0.5,
                 max_disp        : float = 0.2,
                 tau_nhc         : float = 100.0,
                 traj_every      : int   = 10,
                 thermo_every    : int   = 1,
                 output_dir      : str   = ".",
                 skin_width      : float = 0.5,
                 seed            : int   = 42,
                 verbose         : bool  = True,
                 # Extension hooks (biased insertion / deletion)
                 cavity_bias     : bool  = False,
                 energy_bias_del : bool  = False,
                 fix_atoms       : Optional[np.ndarray] = None,
                 timing         : bool  = False,
                 flush_every_step : bool = True,
                 fsync_every_step : bool = False,
                 # Deprecated alias for `mu`, kept for backward
                 # compatibility -- see the `mu` parameter's rename note
                 # above.  Always keyword-only in practice (nothing in
                 # this codebase or its history calls this constructor
                 # positionally past `force_fn`), so appending it here
                 # rather than at `mu`'s original position does not
                 # disturb any positional call.
                 mu_ex            : Optional[float] = None):

        if mu_ex is not None:
            if mu is not None:
                raise TypeError(
                    "GCMCMDSimulation: pass either 'mu' or the "
                    "deprecated 'mu_ex' alias, not both.")
            warnings.warn(
                "GCMCMDSimulation: the 'mu_ex' parameter is deprecated; "
                "use 'mu' instead.  Note 'mu_ex' was ALWAYS the full "
                "chemical potential despite its name, not the excess "
                "chemical potential -- see the class docstring's "
                "rename note.", UserWarning, stacklevel=2)
            mu = mu_ex
        if mu is None:
            mu = -0.1   # preserve the historical default value

        self.flush_every_step = bool(flush_every_step)
        self.fsync_every_step = bool(fsync_every_step)
        self.lattice      = np.asarray(lattice,    dtype=float).copy()
        self.positions    = np.asarray(positions,  dtype=float).copy()
        self.atom_types   = list(atom_types)
        self._force_fn    = force_fn
        self.temperature  = float(temperature)
        self.mu           = float(mu)
        self.species      = species
        from ..fitting.external import check_external_for_solver
        check_external_for_solver(force_fn, type(self).__name__, slow=True)
        check_mlip_for_solver(force_fn, type(self).__name__,
                              need_site_energies=True,
                              species=[species] if species else [],
                              refuse=self._MLIP_REFUSAL)
        # ADR-0008: ESM bc1 (vacuum on both sides) needs a neutral cell;
        # inserting a charged species breaks it.  bc2 / bc3 have an
        # electrode that takes up the net charge.
        from ..fitting.force_fn import esm_bc, uses_esm
        if uses_esm(force_fn) and esm_bc(force_fn) == "bc1":
            q_ins = (getattr(force_fn, "charges", None) or {}).get(species, 0.0)
            if abs(q_ins) > 1e-10:
                raise ValueError(
                    f"{type(self).__name__}: ESM bc1 (vacuum | slab | vacuum) "
                    f"needs a neutral cell, and inserting '{species}' (charge "
                    f"{q_ins:+g}) changes the net charge; use bc2 or bc3, whose "
                    f"electrode compensates it.")
        self.mass         = float(mass) if mass is not None \
                            else float(ATOMIC_MASS.get(species, 1.0))
        self.dt           = float(timestep)
        self.n_cycles     = int(n_cycles)
        self.md_steps     = int(md_steps_per_cycle)
        self.gcmc_tries   = int(gcmc_attempts_per_cycle)
        self.ins_del_ratio= float(ins_del_ratio)
        self.max_disp     = float(max_disp)
        self.tau_nhc      = float(tau_nhc)
        self.traj_every   = int(traj_every)
        self.thermo_every = int(thermo_every)
        self.output_dir   = output_dir
        self.verbose      = verbose
        self.cavity_bias      = cavity_bias
        self.energy_bias_del  = energy_bias_del
        self.rng          = np.random.default_rng(seed)

        self.N = len(atom_types)
        # -- fix_atoms constraint mask -------------------------
        # fix_atoms : (N, 3) bool array.  True = frozen (never displaced).
        # Equivalent to VASP Selective Dynamics 'F T T' etc.
        self._fix_mask = self._build_fix_mask(
            self.N, fix_atoms, self.__class__.__name__)
        self._tc = TimingCollector(enabled=timing)
        r_cut = float(getattr(force_fn, 'r_cut', 10.0))
        self._nl = NeighbourList(r_cut=r_cut, skin_width=skin_width)

        # Thermal de Broglie wavelength lambda [A]
        # lambda = h / sqrt(2pi m k_B T)
        # h  = 6.62607015e-34 J.s = 6.62607015e-34 / 1.60218e-19 eV.s
        #    = 4.13566769e-15 eV.s
        # m [amu] -> [kg]: m * 1.66054e-27
        # k_B [eV/K] * T [K] -> [eV]
        # lambda [m] = h_J / sqrt(2pi m_kg k_J T) -> convert to A
        h_J   = 6.62607015e-34      # J.s
        kB_J  = 1.380649e-23        # J/K
        m_kg  = self.mass * 1.66053906660e-27
        self._lambda_dB = (h_J / np.sqrt(2.0 * np.pi * m_kg * kB_J * temperature)
                           ) * 1e10  # -> A
        self._beta  = 1.0 / (K_B * temperature)   # [eV^-1]

        # NHC state (shared across cycles for continuity)
        masses_arr = np.array([ATOMIC_MASS.get(a, 1.0) for a in self.atom_types])
        self._masses = masses_arr
        n_dof  = self._compute_n_dof(self.N, self._fix_mask)
        self._nhc = {
            "Q1"  : n_dof  * K_B * temperature * tau_nhc**2,
            "Q2"  : K_B * temperature * tau_nhc**2,
            "xi1" : 0.0, "xi2"  : 0.0,
            "vxi1": 0.0, "vxi2" : 0.0,
            "n_dof": n_dof,
        }

        # Velocities (Maxwell-Boltzmann at T)
        self.velocities = self._init_velocities()

        # Statistics
        self._N_history     : List[int]   = []
        self._thermo_log    : List[dict]  = []
        self._n_ins_try  = 0;  self._n_ins_acc  = 0
        self._n_del_try  = 0;  self._n_del_acc  = 0
        self._n_disp_try = 0;  self._n_disp_acc = 0

    @property
    def mu_ex(self) -> float:
        """Deprecated alias for :attr:`mu`, kept for any code that reads
        the attribute directly after construction (as opposed to the
        deprecated ``mu_ex=`` constructor kwarg, handled separately in
        ``__init__``).  See the ``mu`` parameter's rename note in the
        class docstring: this was always the full chemical potential,
        never the excess."""
        warnings.warn(
            "GCMCMDSimulation.mu_ex is deprecated; use .mu instead (it "
            "is the full chemical potential, not the excess).",
            UserWarning, stacklevel=2)
        return self.mu

    # ------------------------------------------------------------------ #
    #  Initialisation helpers                                              #
    # ------------------------------------------------------------------ #

    def _init_velocities(self) -> np.ndarray:
        """Maxwell-Boltzmann velocities at self.temperature; delegates to base."""
        if self.N == 0:
            return np.empty((0, 3), dtype=float)
        return _BaseMDSimulation._init_velocities_mb(
            self.temperature, self._masses, self.rng)

    def _new_particle_velocity(self) -> np.ndarray:
        """
        Draw a single-particle velocity from Maxwell-Boltzmann [A/fs].

        Uses the **active species'** mass (``_active_sp`` / ``_sp_table``
        when set by a multi-species subclass, otherwise ``self.mass``).
        """
        active_sp = getattr(self, '_active_sp', self.species)
        sp_tbl    = getattr(self, '_sp_table', {})
        m         = sp_tbl[active_sp]['mass'] if active_sp in sp_tbl \
                    else self.mass
        sigma = np.sqrt(K_B * self.temperature * EV_ANG_TO_AMU_ANG_FS2 / m)
        return self.rng.standard_normal(3) * sigma

    # ------------------------------------------------------------------ #
    #  Thermodynamic quantities                                            #
    # ------------------------------------------------------------------ #

    def _kinetic_energy(self, v: Optional[np.ndarray] = None) -> float:
        """E_kin [eV] = 0.5 m_i v_i^2 / unit_factor."""
        if v is None:
            v = self.velocities
        return _BaseMDSimulation._kinetic_energy(v, self._masses)

    def _temperature_inst(self, v: Optional[np.ndarray] = None) -> float:
        """Instantaneous temperature [K] for GCMC (uses _masses)."""
        # Bug-fix note (S8, consistency): the original recomputed
        # `n_dof = max(3*self.N - 3, 1)` inline here instead of reading
        # the already-tracked `self._nhc["n_dof"]` (kept up to date by
        # _update_nhc_dof() on every insertion/deletion, and now fixed to
        # account for fix_atoms -- see _BaseMDSimulation._compute_n_dof).
        # Reading the tracked value keeps this method consistent with
        # the rest of the class instead of silently reintroducing the
        # un-fixed formula.
        n_dof = self._nhc["n_dof"]
        return (2.0 * _BaseMDSimulation._kinetic_energy(
                    v if v is not None else self.velocities, self._masses)
                / (n_dof * K_B) if n_dof > 0 else 0.0)

    def _get_energy(self) -> float:
        """Return current potential energy [eV]."""
        if self.N == 0:
            return 0.0
        self._nl.check_and_rebuild(self.lattice, self.positions)
        e, _, _ = self._force_fn(self.lattice, self.positions, self.atom_types)
        return float(e)

    def _get_forces(self) -> Tuple[float, np.ndarray, np.ndarray]:
        if self.N == 0:
            return 0.0, np.empty((0, 3)), np.zeros((3, 3))
        self._nl.check_and_rebuild(self.lattice, self.positions)
        return self._force_fn(self.lattice, self.positions, self.atom_types)

    # ------------------------------------------------------------------ #
    #  NHC thermostat step (dict-based state; see _BaseMDSimulation for   #
    #  the flat-attribute variant shared by NVT/NPT -- see S10 note)      #
    # ------------------------------------------------------------------ #

    def _nhc_step(self, e_kin_curr: float) -> None:
        """
        2-chain NHC half-step for GCMCMDSimulation -- VASP-style MTTK propagator.

        Uses a dict ``self._nhc`` for thermostat state (vxi1, vxi2, xi1, xi2,
        Q1, Q2, n_dof).  All exp() arguments clamped to [-350, 350].

        Kept as its own dict-based implementation rather than sharing
        ``_BaseMDSimulation._nhc_step`` (used by NVT/NPT, which store the
        same state as flat ``self.Q1``/``self.vxi1``/etc. attributes):
        GCMC's particle count -- and hence ``n_dof``, ``Q1``, ``Q2`` --
        changes on every accepted insertion/deletion, so bundling the
        thermostat state in one dict (updated in one place by
        ``_update_nhc_dof``) is clearer here than it would be spread
        across several separately-mutated flat attributes.
        """
        nhc  = self._nhc
        T    = self.temperature
        kT   = K_B * T
        dt   = self.dt
        Q1   = nhc["Q1"]; Q2 = nhc["Q2"]

        G1 = (2.0 * e_kin_curr - nhc["n_dof"] * kT) / Q1
        G2 = (Q1 * nhc["vxi1"] ** 2 - kT) / Q2

        nhc["vxi2"] += G2 * (dt * 0.25)

        s_arg = float(np.clip(-nhc["vxi2"] * dt * 0.125, -350.0, 350.0))
        s     = np.exp(s_arg)
        nhc["vxi1"] = s * (s * nhc["vxi1"] + G1 * dt * 0.25)

        sc_arg = float(np.clip(-nhc["vxi1"] * dt * 0.5, -350.0, 350.0))
        self.velocities *= np.exp(sc_arg)

        nhc["xi1"] += nhc["vxi1"] * dt * 0.5
        nhc["xi2"] += nhc["vxi2"] * dt * 0.5

        e_kin_curr = self._kinetic_energy()
        G1 = (2.0 * e_kin_curr - nhc["n_dof"] * kT) / Q1
        nhc["vxi1"] = s * (s * nhc["vxi1"] + G1 * dt * 0.25)

        G2 = (Q1 * nhc["vxi1"] ** 2 - kT) / Q2
        nhc["vxi2"] += G2 * (dt * 0.25)

    # ------------------------------------------------------------------ #
    #  MD segment (velocity Verlet + NHC)                                 #
    # ------------------------------------------------------------------ #

    def _md_segment(self) -> float:
        """
        Run ``self.md_steps`` NVT-MD steps.

        Returns the potential energy at the end of the segment.
        Uses the shared NHC state so thermostat continuity is preserved
        across GCMC cycles.
        """
        if self.N == 0:
            return 0.0

        e_pot, forces, _ = self._get_forces()
        acc = forces * EV_ANG_TO_AMU_ANG_FS2 / self._masses[:, None]
        inv_lat = np.linalg.inv(self.lattice)

        for _ in range(self.md_steps):
            ek = self._kinetic_energy()
            self._nhc_step(ek)

            self.velocities  += 0.5 * self.dt * acc
            self.velocities  *= ~self._fix_mask[:self.N]  # freeze constrained DoF
            self.positions   += self.dt * self.velocities

            # PBC wrap
            frac = self.positions @ inv_lat
            # `frac -= np.floor(frac)` alone maps a tiny negative frac to
            # exactly 1.0 in float64, which scipy's cKDTree rejects; see
            # wrap_positions_into_cell.
            frac -= np.floor(frac)
            frac[frac >= 1.0] = 0.0
            self.positions = frac @ self.lattice

            e_pot, forces, _ = self._get_forces()
            acc = forces * EV_ANG_TO_AMU_ANG_FS2 / self._masses[:, None]

            self.velocities  += 0.5 * self.dt * acc
            self.velocities  *= ~self._fix_mask[:self.N]  # freeze constrained DoF
            ek = self._kinetic_energy()
            self._nhc_step(ek)

        return float(e_pot)

    # ------------------------------------------------------------------ #
    #  GCMC moves                                                          #
    # ------------------------------------------------------------------ #

    def _trial_insertion_position(self) -> np.ndarray:
        """
        Generate a trial insertion position in Cartesian coordinates.

        Standard mode (cavity_bias=False):
            Uniform random position in the unit cell.

        Cavity-biased mode (cavity_bias=True):
            Sample ``n_probe`` random positions and return the one with
            the lowest local energy.  The bias correction factor is stored
            in ``self._last_ins_bias_log``.

        The probe energy is computed by inserting a particle of the
        **active species** (``_active_sp`` when set, else ``self.species``).
        """
        frac     = self.rng.uniform(0, 1, 3)
        pos_trial = frac @ self.lattice

        if not self.cavity_bias:
            self._last_ins_bias_log = 0.0
            return pos_trial

        n_probe = max(10, self.N // 2)
        probes  = self.rng.uniform(0, 1, (n_probe, 3)) @ self.lattice
        energies = np.array([self._probe_energy(p) for p in probes])
        idx_best = int(np.argmin(energies))
        pos_trial = probes[idx_best]

        log_w  = -self._beta * energies
        log_w -= log_w.max()
        log_Z  = np.log(np.sum(np.exp(log_w)))
        self._last_ins_bias_log = (log_w[idx_best] - log_Z
                                   + np.log(float(n_probe)))
        return pos_trial

    def _probe_energy(self, r_trial: np.ndarray) -> float:
        """
        Compute the interaction energy of a single trial atom at r_trial
        with all current particles (no self-interaction).

        The trial atom's species is the **active species**
        (``_active_sp`` when set by a multi-species subclass,
        otherwise ``self.species``).

        This is correct for INSERTION (r_trial is a genuinely new
        position, not already occupied by any real particle) but must
        NOT be reused for deletion -- see _local_energy_of_atom below.
        """
        active_sp = getattr(self, '_active_sp', self.species)
        pos_ext   = np.vstack([self.positions, r_trial[None, :]])
        types_ext = self.atom_types + [active_sp]
        e, _, _   = self._force_fn(self.lattice, pos_ext, types_ext)
        return float(e) - self._get_energy()

    def _local_energy_of_atom(self, idx: int) -> float:
        """
        E(full system) - E(system with particle idx removed) [eV].

        The physically correct "local energy" contribution of particle
        idx, used for deletion moves and energy-biased deletion-candidate
        selection.

        Bug-fix note (S3): the original computed deletion's delta_U (and,
        identically, OSMMCSimulation's per-atom local energy used to bias
        deletion-candidate selection) by calling
        ``_probe_energy(self.positions[idx])`` -- but ``_probe_energy`` is
        designed for INSERTION: it appends a *new* ghost particle at the
        given position to the *existing* N-particle configuration. Since
        particle ``idx`` is still present in ``self.positions`` at that
        point, this placed the ghost at the exact same coordinates as a
        real, already-there particle -- an r=0 self-overlap that diverges
        to +inf (purely repulsive terms) or -inf (e.g. the Buckingham
        C/r^6 dispersion term) depending on the potential, corrupting
        every deletion move's acceptance probability and any
        energy-biased deletion-candidate selection. Instead of reusing
        the insertion helper, this evaluates the energy of the system
        with idx genuinely removed and compares it to the current total.
        """
        e_full = self._get_energy()
        if self.N <= 1:
            return e_full   # removing the only atom leaves an empty (E=0) system
        pos_wo   = np.delete(self.positions, idx, axis=0)
        types_wo = self.atom_types[:idx] + self.atom_types[idx + 1:]
        e_wo, _, _ = self._force_fn(self.lattice, pos_wo, types_wo)
        return e_full - float(e_wo)

    def _delta_U_insertion(self, r_new: np.ndarray) -> float:
        """delta_U for inserting one particle at r_new [eV]."""
        return self._probe_energy(r_new)

    def _delta_U_deletion(self, idx: int) -> float:
        """delta_U for removing particle idx [eV] (= -local energy of idx)."""
        return -self._local_energy_of_atom(idx)

    def _select_deletion_candidate(self) -> int:
        """
        Select the index of the particle to attempt deletion.

        Filters candidates to the **active species** (``_active_sp`` when set
        by a multi-species subclass, otherwise ``self.species``).

        Standard mode (energy_bias_del=False):
            Uniform random among atoms of the active species.

        Energy-biased mode (energy_bias_del=True):
            Select with probability proportional to exp(+beta u_i).  The reciprocal bias
            correction is stored in ``self._last_del_bias_log``.
        """
        active_sp   = getattr(self, '_active_sp', self.species)
        species_idx = [i for i, s in enumerate(self.atom_types)
                       if s == active_sp]
        if not species_idx:
            return -1

        if not self.energy_bias_del:
            self._last_del_bias_log = 0.0
            return int(self.rng.choice(species_idx))

        # S3 fix applies here too: local energy via _local_energy_of_atom,
        # not the insertion-style _probe_energy self-overlap.
        e_local = np.array([self._local_energy_of_atom(i) for i in species_idx])
        log_w  = self._beta * e_local
        log_w -= log_w.max()
        w      = np.exp(log_w)
        w     /= w.sum()
        chosen_local = int(self.rng.choice(len(species_idx), p=w))
        idx    = species_idx[chosen_local]
        self._last_del_bias_log = float(np.log(w[chosen_local] + 1e-300))
        return idx

    # ------------------------------------------------------------------ #
    #  GCMC cycle: one batch of trial moves                               #
    # ------------------------------------------------------------------ #

    def _gcmc_cycle(self) -> None:
        """
        Perform ``self.gcmc_tries`` Metropolis trial moves.

        Move types (decided per trial independently):
        - Insert  : probability  ins_del_ratio / 2
        - Delete  : probability  ins_del_ratio / 2
        - Displace: probability  1 - ins_del_ratio

        Multi-species support
        ----------------------
        When the subclass sets ``_sp_table`` (OSMMCSimulation), each trial
        move first calls ``_choose_active_species()`` to select the element
        for this trial, then reads mu and lambda from the active species entry.
        In single-species mode (plain GCMCMDSimulation) the behaviour is
        identical to the previous implementation.
        """
        V    = abs(np.linalg.det(self.lattice))
        beta = self._beta
        p_id = self.ins_del_ratio * 0.5

        for _ in range(self.gcmc_tries):
            # -- Per-trial species selection (multi-species subclass) ----
            # GCMCMDSimulation does not define _sp_table; the attribute check
            # lets OSMMCSimulation's species-switching logic work transparently.
            if hasattr(self, '_sp_table') and len(self._sp_table) > 1:
                self._choose_active_species()
            active_sp = getattr(self, '_active_sp', self.species)

            # Element-specific lambda^3 and mu for the Metropolis criterion
            Lam3 = self._lambda_dB ** 3
            mu   = self.mu

            u = self.rng.random()

            if u < p_id:
                # -- Insertion -----------------------------------------
                self._n_ins_try += 1
                r_new = self._trial_insertion_position()
                dU    = self._delta_U_insertion(r_new)

                ln_acc = (np.log(V / (Lam3 * (self.N + 1)))
                          + beta * (mu - dU)
                          - getattr(self, '_last_ins_bias_log', 0.0))

                if ln_acc >= 0.0 or np.log(self.rng.random()) < ln_acc:
                    self._accept_insertion(r_new)
                    self._n_ins_acc += 1

            elif u < 2.0 * p_id:
                # -- Deletion ------------------------------------------
                if self.N == 0:
                    continue
                self._n_del_try += 1
                idx = self._select_deletion_candidate()
                if idx < 0:
                    continue
                dU = self._delta_U_deletion(idx)

                # Bug-fix note: dU = U(N-1) - U(N) here, and Frenkel & Smit
                # eq. 5.6.9 is acc(N->N-1) = min(1, lambda^3 N / V
                # exp(-beta (mu + dU))).  This read beta (-mu + dU), which
                # favoured deleting the most strongly bound atoms: LJ Ar
                # vapour at T* = 0.90 (NIST TMMC, rho* = 0.0818, N = 41.9 in
                # 512 sigma^3) settled at N = 18.8, below even the ideal gas
                # at the same activity (22.5).  An ideal gas (dU = 0) could
                # not see it, which is how the earlier check passed.
                ln_acc = (np.log(Lam3 * self.N / V)
                          + beta * (-mu - dU)
                          + getattr(self, '_last_del_bias_log', 0.0))

                if ln_acc >= 0.0 or np.log(self.rng.random()) < ln_acc:
                    self._accept_deletion(idx)
                    self._n_del_acc += 1

            else:
                # -- Displacement --------------------------------------
                if self.N == 0:
                    continue
                self._n_disp_try += 1
                # Displace an atom of the active species
                species_idx = [i for i, s in enumerate(self.atom_types)
                               if s == active_sp]
                if not species_idx:
                    continue
                idx   = int(self.rng.choice(species_idx))
                r_old = self.positions[idx].copy()
                disp  = (self.rng.random(3) - 0.5) * 2.0 * self.max_disp

                e_old = self._get_energy()
                self.positions[idx] += disp
                self._nl.build(self.lattice, self.positions)
                e_new = self._get_energy()
                dU    = e_new - e_old

                ln_acc = -beta * dU
                if ln_acc >= 0.0 or np.log(self.rng.random()) < ln_acc:
                    self._n_disp_acc += 1
                else:
                    self.positions[idx] = r_old
                    self._nl.build(self.lattice, self.positions)

    # ------------------------------------------------------------------ #
    #  Insertion / deletion state updates                                  #
    # ------------------------------------------------------------------ #

    def _accept_insertion(self, r_new: np.ndarray) -> None:
        """
        Commit an accepted insertion.

        Appends the new particle to positions, atom_types, masses, and
        NHC state.  A new velocity is drawn from Maxwell-Boltzmann.

        The inserted atom's element is the **active species**
        (``_active_sp`` when set, otherwise ``self.species``).
        Its mass is taken from the active species' entry in ``_sp_table``
        when available, otherwise from ``self.mass``.
        """
        active_sp = getattr(self, '_active_sp', self.species)
        sp_tbl    = getattr(self, '_sp_table', {})
        mass_new  = sp_tbl[active_sp]['mass'] if active_sp in sp_tbl \
                    else self.mass

        self.positions  = np.vstack([self.positions, r_new[None, :]])
        self.atom_types.append(active_sp)
        self._masses    = np.append(self._masses, mass_new)

        v_new = self._new_particle_velocity()
        self.velocities = np.vstack([self.velocities, v_new[None, :]])

        self.N += 1
        self._update_nhc_dof()
        self._nl.build(self.lattice, self.positions)
        self._nl.mark_dirty()   # force rebuild on next force evaluation
        # Extend fix_mask: newly inserted atoms have no constraint by default
        self._fix_mask = np.vstack(
            [self._fix_mask, np.zeros((1, 3), dtype=bool)])
        # Bug-fix note (S12): the original used
        # `isinstance(self._force_fn, ESMCalculator)`, coupling this
        # solver to the engine's internal ESMCalculator class. force_fn
        # objects that need geometry updates now declare it via the
        # `.kind` attribute (set by make_force_fn / ESMCalculator itself),
        # so this check no longer needs to import or know about
        # ESMCalculator at all.
        if getattr(self._force_fn, 'kind', None) == 'esm':
            self._force_fn.update_geometry(
                self.lattice, self.positions, self.atom_types)

    def _accept_deletion(self, idx: int) -> None:
        """
        Commit an accepted deletion.

        Removes particle ``idx`` from all arrays.
        """
        keep = [i for i in range(self.N) if i != idx]
        self.positions  = self.positions[keep]
        self.velocities = self.velocities[keep]
        self._masses    = self._masses[keep]
        self.atom_types = [self.atom_types[i] for i in keep]

        self.N -= 1
        self._update_nhc_dof()
        self._nl.build(self.lattice, self.positions)
        self._nl.mark_dirty()   # force rebuild on next force evaluation
        # Remove the corresponding row from fix_mask
        self._fix_mask = np.delete(self._fix_mask, idx, axis=0)
        # Bug-fix note (S12): see _accept_insertion above.
        if getattr(self._force_fn, 'kind', None) == 'esm':
            self._force_fn.update_geometry(
                self.lattice, self.positions, self.atom_types)

    def _update_nhc_dof(self) -> None:
        """Recompute NHC parameters after N changes."""
        T    = self.temperature
        tau  = self.tau_nhc
        # Bug-fix note (S8): now accounts for fix_atoms -- see
        # _BaseMDSimulation._compute_n_dof.
        n_dof = self._compute_n_dof(self.N, self._fix_mask)
        # Update mass (Q) only; preserve thermostat velocities for continuity
        self._nhc["Q1"]   = n_dof * K_B * T * tau**2
        self._nhc["Q2"]   = K_B * T * tau**2
        self._nhc["n_dof"] = n_dof

    # ------------------------------------------------------------------ #
    #  Output helpers                                                      #
    # ------------------------------------------------------------------ #

    def _write_xyz_frame(self, fh, cycle: int, e_pot: float) -> None:
        """
        Append one configuration to the open XDATCAR file.

        GCMC allows particle insertion/deletion, so the atom count
        (and possibly species composition) can change between frames.
        Standard VASP XDATCAR format assumes a fixed atom count for the
        whole file, so -- as an extension for this variable-N case -- a
        full header (comment, scale, lattice, species, counts) is
        written before every frame, reflecting the CURRENT particle
        count at that cycle.
        """
        comment = (f"akitsushima.py GCMC  cycle={cycle}  N={self.N}  "
                  f"E_pot={e_pot:.6f}eV  T={self._temperature_inst():.2f}K")
        _write_xdatcar_frame(fh, self.positions, self.lattice,
                             self.atom_types, self._xdatcar_frame_idx,
                             variable_header=True, comment=comment)
        self._xdatcar_frame_idx += 1

    def _acceptance_rates(self) -> Tuple[float, float, float]:
        acc_i = (self._n_ins_acc  / self._n_ins_try
                 if self._n_ins_try  > 0 else float('nan'))
        acc_d = (self._n_del_acc  / self._n_del_try
                 if self._n_del_try  > 0 else float('nan'))
        acc_m = (self._n_disp_acc / self._n_disp_try
                 if self._n_disp_try > 0 else float('nan'))
        return acc_i, acc_d, acc_m

    # ------------------------------------------------------------------ #
    #  Main simulation loop                                               #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """
        Execute the GCMC+MD simulation.

        Each cycle:
            1. ``md_steps_per_cycle`` NVT-MD steps.
            2. ``gcmc_attempts_per_cycle`` Metropolis GCMC moves.

        Returns
        -------
        dict with keys:
            positions, velocities, lattice, atom_types,
            N_history, N_mean, N_std,
            e_pot_mean,
            acc_ins, acc_del, acc_disp,
            thermo_log.
        """
        self._tc.start("total")  # timing
        self._ensure_output_dir(self.output_dir)
        traj_path   = os.path.join(self.output_dir,
                                    f"XDATCAR_{self._xdatcar_task_name}")
        thermo_path = os.path.join(self.output_dir, "gcmc_md_thermo.log")
        n_path      = os.path.join(self.output_dir, "N_vs_cycle.dat")

        V = abs(np.linalg.det(self.lattice))
        header = (f"{'Cycle':>8}  {'N':>6}  {'rho(1/A3)':>10}  "
                  f"{'E_pot(eV)':>14}  {'T_inst(K)':>10}  "
                  f"{'acc_ins':>8}  {'acc_del':>8}  {'acc_disp':>8}")
        if self.verbose:
            print(f"\n  GCMC+MD  (muVT)  T={self.temperature} K  "
                  f"mu={self.mu:.4f} eV  species={self.species}")
            print(f"  cycles={self.n_cycles}  md/cycle={self.md_steps}  "
                  f"gcmc_tries/cycle={self.gcmc_tries}")
            # Show lambda_dB per species if multi-species table is available
            sp_tbl = getattr(self, '_sp_table', {})
            if sp_tbl:
                for sp_k, sp_v in sp_tbl.items():
                    print(f"  [{sp_k}] mu={sp_v['mu']:.4f} eV  "
                          f"lambda_dB={sp_v['lambda_dB']:.4f} A  "
                          f"m_lam={sp_v['m_lam']:.2f}  "
                          f"sc_alpha={sp_v['sc_alpha']:.2f}  "
                          f"n={sp_v['sc_n']}")
            else:
                print(f"  lambda_dB = {self._lambda_dB:.4f} A")
            print()
            print(header); print("  " + "-" * len(header))

        e_pot_history  : List[float] = []

        # See the 2-file-handle variant of this comment elsewhere in this
        # file: only rank 0 writes to the real paths; other ranks
        # write to os.devnull to avoid concurrent-write corruption.
        _traj_path_use   = traj_path   if _mpi_rank == 0 else os.devnull
        _thermo_path_use = thermo_path if _mpi_rank == 0 else os.devnull
        _n_path_use      = n_path      if _mpi_rank == 0 else os.devnull
        with (open(_traj_path_use,   "w") as traj_fh,
              open(_thermo_path_use, "w") as thermo_fh,
              open(_n_path_use,      "w") as n_fh):

            thermo_fh.write(header + "\n")
            n_fh.write("# cycle  N\n")
            self._xdatcar_frame_idx = 1

            for cycle in range(self.n_cycles):
                self._tc.start("step")  # timing
                # 1. MD segment
                e_pot = self._md_segment()

                # 2. GCMC cycle
                self._gcmc_cycle()
                e_pot = self._get_energy()   # refresh after GCMC moves

                # Record
                self._N_history.append(self.N)
                e_pot_history.append(float(e_pot))
                T_inst = self._temperature_inst()
                rho    = self.N / V
                acc_i, acc_d, acc_m = self._acceptance_rates()

                n_fh.write(f"{cycle}  {self.N}\n")
                self._thermo_log.append({
                    "cycle": cycle, "N": self.N, "rho": rho,
                    "e_pot": e_pot, "T_inst": T_inst,
                    "acc_ins": acc_i, "acc_del": acc_d, "acc_disp": acc_m,
                })

                self._tc.lap("step")  # timing
                if cycle % self.thermo_every == 0:
                    ai = f"{acc_i:.4f}" if np.isfinite(acc_i) else "  n/a "
                    ad = f"{acc_d:.4f}" if np.isfinite(acc_d) else "  n/a "
                    am = f"{acc_m:.4f}" if np.isfinite(acc_m) else "  n/a "
                    row = (f"{cycle:8d}  {self.N:6d}  {rho:10.6f}  "
                           f"{e_pot:14.6f}  {T_inst:10.2f}  "
                           f"{ai:>8}  {ad:>8}  {am:>8}")
                    thermo_fh.write(row + "\n")
                    if self.verbose and cycle % max(1, self.n_cycles // 20) == 0:
                        print("  " + row)

                if cycle % self.traj_every == 0:
                    self._write_xyz_frame(traj_fh, cycle, e_pot)

                # Push this cycle's output to disk for real-time
                # monitoring (n_fh is written every cycle already, so
                # this also covers it) during a long-running GCMC run.
                if self.flush_every_step:
                    self._flush_outputs([n_fh, thermo_fh, traj_fh],
                                        fsync=self.fsync_every_step)

        # Final structure
        out_poscar = os.path.join(self.output_dir, "POSCAR_gcmc_final")
        if _mpi_rank == 0:
            write_poscar(out_poscar, self.lattice, self.positions,
                         self.atom_types, title="GCMC+MD final frame")

        # Summary statistics
        N_arr   = np.array(self._N_history)
        e_arr   = np.array(e_pot_history)
        N_mean  = float(N_arr.mean())  if len(N_arr)  else 0.0
        N_std   = float(N_arr.std())   if len(N_arr)  else 0.0
        e_mean  = float(e_arr.mean())  if len(e_arr)  else 0.0
        acc_i, acc_d, acc_m = self._acceptance_rates()

        if self.verbose:
            print(f"\n  GCMC+MD finished.")
            print(f"  <N>      = {N_mean:.2f} +/- {N_std:.2f}")
            print(f"  <rho>      = {N_mean/V:.6f} A^-^3")
            print(f"  <E_pot>  = {e_mean:.6f} eV")
            print(f"  Acc_ins  = {acc_i:.4f}" if np.isfinite(acc_i) else "  Acc_ins = n/a")
            print(f"  Acc_del  = {acc_d:.4f}" if np.isfinite(acc_d) else "  Acc_del = n/a")
            print(f"  Acc_disp = {acc_m:.4f}" if np.isfinite(acc_m) else "  Acc_disp = n/a")
            print(f"  Files    : {traj_path}, {thermo_path}, {n_path}")

        if HAS_MATPLOTLIB and _mpi_rank == 0:
            self._save_plots()

        self._tc.stop("total")  # timing
        _timing = self._tc.summary()
        if _timing.get("enabled") and _mpi_rank == 0:
            self._tc.print_summary(label="GCMCMDSimulation")

        return {
            "positions"  : self.positions.copy(),
            "velocities" : self.velocities.copy(),
            "lattice"    : self.lattice.copy(),
            "atom_types" : list(self.atom_types),
            "N_history"  : list(self._N_history),
            "N_mean"     : N_mean,
            "N_std"      : N_std,
            "e_pot_mean" : e_mean,
            "acc_ins"    : acc_i,
            "acc_del"    : acc_d,
            "acc_disp"   : acc_m,
            "thermo_log" : self._thermo_log,
            "timing"     : _timing,
        }

    # ------------------------------------------------------------------ #
    #  Plotting                                                            #
    # ------------------------------------------------------------------ #

    def _save_plots(self) -> None:
        """Save N(t) and E_pot(t) time-series plots."""
        if not self._thermo_log:
            return
        cycles = [r["cycle"]  for r in self._thermo_log]
        N_list = [r["N"]      for r in self._thermo_log]
        E_list = [r["e_pot"]  for r in self._thermo_log]
        T_list = [r["T_inst"] for r in self._thermo_log]

        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        axes[0].plot(cycles, N_list, lw=0.8, color="steelblue")
        axes[0].axhline(np.mean(N_list), color="r", ls="--", lw=1,
                        label=f"<N> = {np.mean(N_list):.1f}")
        axes[0].set_ylabel("N"); axes[0].legend(fontsize=8)

        axes[1].plot(cycles, E_list, lw=0.8, color="darkorange")
        axes[1].set_ylabel("E_pot (eV)")

        axes[2].plot(cycles, T_list, lw=0.8, color="green")
        axes[2].axhline(self.temperature, color="r", ls="--", lw=1)
        axes[2].set_ylabel("T_inst (K)")
        axes[2].set_xlabel("Cycle")

        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"GCMC+MD  --  T={self.temperature} K  "
                     f"mu={self.mu:.3f} eV", fontsize=12)
        fig.tight_layout()
        path = os.path.join(self.output_dir, "gcmc_md_timeseries.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  Plot: {path}")


#  SECTION 6.5: Continuous fractional component MC (CFCMC, muVT) -- ADR-0009
# ============================================================================


class CFCMCSimulation(GCMCMDSimulation):
    """
    Grand-canonical MC/MD with a continuous fractional atom (CFCMC, muVT).

    One atom of the exchanged species is *fractional*: its coupling
    ``lambda`` in [0, 1] enters the energy through a soft core in energy
    space (ADR-0009, Q18),

        U(lambda) = U_N + g(lambda, dU),   dU = U_{N+1} - U_N,
        g = lambda dU / (1 + (1 - lambda) max(dU, 0) / E_c),

    U_N without the fractional atom, U_{N+1} with it as a full atom, so any
    force function works unchanged (engines, Ewald / PPPM / ESM, an
    atom-centred MLIP).  g is exact at both ends; an overlap costs at most
    lambda E_c / (1 - lambda), which keeps the lambda = 0 end regular
    (linear coupling, E_c -> infinity, biases mu_ex there).  Each cycle runs an NVT-MD segment at fixed N and
    lambda (the fractional atom included) and then
    ``gcmc_attempts_per_cycle`` MC moves: lambda changes (probability
    ``lambda_move_ratio``) or displacements.  A lambda change past 1 makes
    the fractional atom whole and places a new one at random (insertion);
    past 0 it removes the fractional atom and makes a random whole atom
    fractional (deletion).  The acceptance rules and the estimators are
    derived in docs/adr/0009-osmc-as-cfcmc.md:

        insertion: min(1, V e^{beta mu} / (Lambda^3 (N+1)) e^{-beta dU + dw})
        deletion : min(1, Lambda^3 N e^{-beta mu} / V     e^{-beta dU + dw})

    Method: Shi & Maginn, J. Chem. Theory Comput. 3, 1451 (2007); the rules
    here were derived independently and verified against the ideal gas,
    the NIST SRSW LJ vapour and ``GCMCMDSimulation``
    (example/gcmc/check_cfcmc.py).

    Parameters (beyond :class:`GCMCMDSimulation`'s)
    ----------
    lambda_move_ratio : fraction of MC moves that change lambda (rest are
                        displacements).  Default 0.5.
    max_dlambda       : lambda step d is uniform in [-max_dlambda,
                        max_dlambda].  Default 0.2.
    n_lambda_bins     : bins of the bias w(lambda) and of the estimators.
                        Default 10.
    equil_cycles      : cycles discarded from the estimators; the bias is
                        adapted during them when ``adapt_weights``.
                        Default n_cycles // 5.
    adapt_weights     : Wang-Landau adaptation of w(lambda) during the
                        equilibration.  Default True.
    softcore_kT       : E_c in units of k_B T.  Default 10.  0 gives
                        linear coupling (not recommended, ADR-0009).

    ``mu`` is the FULL chemical potential, as in GCMCMDSimulation.  Atoms of
    other species are a fixed framework; atoms of ``species`` may not be
    fixed.  ``N_history`` / ``N_vs_cycle.dat`` count whole atoms; the muVT
    averages are the lambda-end estimators ``N_mean`` (lambda = 0) and
    ``N_mean_lambda1`` (lambda = 1), which must agree, and ``mu_ex``.
    """

    _xdatcar_task_name = "osmc"
    _MLIP_REFUSAL = None

    def __init__(
        self,
        *args,
        lambda_move_ratio: float = 0.5,
        max_dlambda: float = 0.2,
        n_lambda_bins: int = 10,
        equil_cycles: Optional[int] = None,
        adapt_weights: bool = True,
        softcore_kT: float = 10.0,
        **kwargs,
    ):
        kwargs.setdefault("cavity_bias", False)
        kwargs.setdefault("energy_bias_del", False)
        if kwargs["cavity_bias"] or kwargs["energy_bias_del"]:
            raise ValueError(
                "CFCMCSimulation: cavity_bias / energy_bias_del "
                "belong to gcmc's discrete moves."
            )
        super().__init__(*args, **kwargs)
        if getattr(self._force_fn, "kind", None) == "esm":
            raise ValueError(
                "CFCMCSimulation: a bare ESMCalculator holds per-geometry "
                "state; use potential.coulomb_method: esm (ADR-0008)."
            )
        if not 0.0 <= lambda_move_ratio <= 1.0:
            raise ValueError("CFCMCSimulation: lambda_move_ratio must be in [0, 1].")
        if not 0.0 < max_dlambda <= 1.0:
            raise ValueError("CFCMCSimulation: max_dlambda must be in (0, 1].")
        if int(n_lambda_bins) < 2:
            raise ValueError("CFCMCSimulation: n_lambda_bins must be >= 2.")
        sp_rows = [i for i, s in enumerate(self.atom_types) if s == self.species]
        if self._fix_mask[sp_rows].any():
            raise ValueError(
                f"CFCMCSimulation: atoms of the exchanged species "
                f"'{self.species}' may not be fixed (they are inserted and "
                f"deleted); fix only the framework."
            )
        self.lambda_move_ratio = float(lambda_move_ratio)
        self.max_dlambda = float(max_dlambda)
        self.n_lambda_bins = int(n_lambda_bins)
        self.equil_cycles = (
            self.n_cycles // 5 if equil_cycles is None else int(equil_cycles)
        )
        self.adapt_weights = bool(adapt_weights)
        if softcore_kT < 0:
            raise ValueError("CFCMCSimulation: softcore_kT must be >= 0.")
        self.softcore_kT = float(softcore_kT)
        # 1 / E_c [1/eV]; 0 = linear coupling
        self._inv_Ec = self._beta / self.softcore_kT if self.softcore_kT > 0 else 0.0
        self._w = np.zeros(self.n_lambda_bins)
        self._wl_f = 1.0
        self._wl_hist = np.zeros(self.n_lambda_bins)

        # The fractional atom is the LAST row of every per-atom array;
        # self.N counts whole atoms only.  It starts decoupled.
        self._lam = 0.0
        x_f = self.rng.uniform(0.0, 1.0, 3) @ self.lattice
        self.positions = np.vstack([self.positions, x_f[None, :]])
        self.atom_types.append(self.species)
        self._masses = np.append(self._masses, self.mass)
        self.velocities = np.vstack(
            [self.velocities, self._new_particle_velocity()[None, :]]
        )
        self._fix_mask = np.vstack([self._fix_mask, np.zeros((1, 3), bool)])
        self._update_nhc_dof()
        self._refresh_energies()

        self._cycle = 0
        self._samples: List[Tuple[int, float, float]] = []  # (N_sp, lam, dU_f)
        self._lam_log: List[float] = []
        self._n_lam_try = 0
        self._n_lam_acc = 0

    # ------------------------------------------------------------------ #
    #  Energies of the two end states                                     #
    # ------------------------------------------------------------------ #

    def _eval(self, pos: np.ndarray, types: list, forces: bool = False):
        if len(types) == 0:
            return (0.0, np.empty((0, 3))) if forces else 0.0
        e, f, _ = self._force_fn(self.lattice, pos, types)
        return (float(e), np.asarray(f)) if forces else float(e)

    def _refresh_energies(self, forces: bool = False):
        """Cache U_N (whole atoms) and U_{N+1} (fractional atom as whole)."""
        r0 = self._eval(self.positions[:-1], self.atom_types[:-1], forces)
        r1 = self._eval(self.positions, self.atom_types, forces)
        if forces:
            (self._E0, F0), (self._E1, F1) = r0, r1
            gam = self._dg_ddU(self._lam, self._E1 - self._E0)
            F = gam * F1
            F[:-1] += (1.0 - gam) * F0
            return F
        self._E0, self._E1 = r0, r1
        return None

    def _g(self, lam, dU):
        """Soft-core coupling g(lambda, dU) (scalar or array)."""
        a = (1.0 - lam) * self._inv_Ec
        return lam * dU / (1.0 + a * np.maximum(dU, 0.0))

    def _dg_ddU(self, lam: float, dU: float) -> float:
        a = (1.0 - lam) * self._inv_Ec
        return lam / (1.0 + a * max(dU, 0.0)) ** 2

    def _U_at(self, lam: float, E0: float, E1: float) -> float:
        return E0 + self._g(lam, E1 - E0)

    def _U(self) -> float:
        return self._U_at(self._lam, self._E0, self._E1)

    def _get_energy(self) -> float:
        return self._U()

    def _get_forces(self):
        F = self._refresh_energies(forces=True)
        return self._U(), F, np.zeros((3, 3))

    def _update_nhc_dof(self) -> None:
        # whole atoms plus the fractional one
        n_dof = self._compute_n_dof(self.N + 1, self._fix_mask)
        self._nhc["Q1"] = n_dof * K_B * self.temperature * self.tau_nhc**2
        self._nhc["Q2"] = K_B * self.temperature * self.tau_nhc**2
        self._nhc["n_dof"] = n_dof

    def _n_species(self) -> int:
        return sum(1 for s in self.atom_types[:-1] if s == self.species)

    def _bin(self, lam: float) -> int:
        return min(int(lam * self.n_lambda_bins), self.n_lambda_bins - 1)

    # ------------------------------------------------------------------ #
    #  MD segment: all N + 1 rows, fixed lambda                            #
    # ------------------------------------------------------------------ #

    def _md_segment(self) -> float:
        if self.md_steps == 0:
            return self._U()
        free = ~self._fix_mask
        _, F, _ = self._get_forces()
        acc = F * EV_ANG_TO_AMU_ANG_FS2 / self._masses[:, None]
        inv_lat = np.linalg.inv(self.lattice)
        for _ in range(self.md_steps):
            self._nhc_step(self._kinetic_energy())
            self.velocities += 0.5 * self.dt * acc
            self.velocities *= free
            self.positions += self.dt * self.velocities
            frac = self.positions @ inv_lat
            frac -= np.floor(frac)
            frac[frac >= 1.0] = 0.0
            self.positions = frac @ self.lattice
            _, F, _ = self._get_forces()
            acc = F * EV_ANG_TO_AMU_ANG_FS2 / self._masses[:, None]
            self.velocities += 0.5 * self.dt * acc
            self.velocities *= free
            self._nhc_step(self._kinetic_energy())
        return self._U()

    # ------------------------------------------------------------------ #
    #  MC moves                                                            #
    # ------------------------------------------------------------------ #

    def _accept(self, ln_acc: float) -> bool:
        return ln_acc >= 0.0 or np.log(self.rng.random()) < ln_acc

    def _lambda_move(self) -> None:
        beta = self._beta
        V = abs(np.linalg.det(self.lattice))
        Lam3 = self._lambda_dB**3
        lam = self._lam
        new = lam + self.rng.uniform(-self.max_dlambda, self.max_dlambda)
        U_old, w_old = self._U(), self._w[self._bin(lam)]
        self._n_lam_try += 1

        if 0.0 <= new <= 1.0:
            dU = self._U_at(new, self._E0, self._E1) - U_old
            if self._accept(-beta * dU + self._w[self._bin(new)] - w_old):
                self._lam = new
                self._n_lam_acc += 1

        elif new > 1.0:
            # insertion: the fractional atom becomes whole, a new one appears
            self._n_ins_try += 1
            lam2 = new - 1.0
            n_sp = self._n_species()
            x_new = self.rng.uniform(0.0, 1.0, 3) @ self.lattice
            pos2 = np.vstack([self.positions, x_new[None, :]])
            types2 = self.atom_types + [self.species]
            E0n, E1n = self._E1, self._eval(pos2, types2)
            dU = self._U_at(lam2, E0n, E1n) - U_old
            ln_acc = (
                np.log(V / (Lam3 * (n_sp + 1)))
                + beta * self.mu
                - beta * dU
                + self._w[self._bin(lam2)]
                - w_old
            )
            if self._accept(ln_acc):
                self._n_ins_acc += 1
                self._n_lam_acc += 1
                self.positions, self.atom_types = pos2, types2
                self._masses = np.append(self._masses, self.mass)
                self.velocities = np.vstack(
                    [self.velocities, self._new_particle_velocity()[None, :]]
                )
                self._fix_mask = np.vstack([self._fix_mask, np.zeros((1, 3), bool)])
                self.N += 1
                self._lam, self._E0, self._E1 = lam2, E0n, E1n
                self._update_nhc_dof()

        else:
            # deletion: the fractional atom goes, a random whole one takes over
            n_sp = self._n_species()
            if n_sp == 0:
                return
            self._n_del_try += 1
            lam2 = new + 1.0
            cand = [i for i, s in enumerate(self.atom_types[:-1]) if s == self.species]
            j = int(self.rng.choice(cand))
            keep = [i for i in range(self.N) if i != j]
            order = keep + [j]  # old fractional row (index N) dropped
            E1n = self._E0  # same atoms, j now fractional
            E0n = self._eval(self.positions[keep], [self.atom_types[i] for i in keep])
            dU = self._U_at(lam2, E0n, E1n) - U_old
            ln_acc = (
                np.log(Lam3 * n_sp / V)
                - beta * self.mu
                - beta * dU
                + self._w[self._bin(lam2)]
                - w_old
            )
            if self._accept(ln_acc):
                self._n_del_acc += 1
                self._n_lam_acc += 1
                self.positions = self.positions[order]
                self.velocities = self.velocities[order]
                self._masses = self._masses[order]
                self._fix_mask = self._fix_mask[order]
                self.atom_types = [self.atom_types[i] for i in order]
                self.N -= 1
                self._lam, self._E0, self._E1 = lam2, E0n, E1n
                self._update_nhc_dof()

    def _displacement_move(self) -> None:
        rows = [i for i, s in enumerate(self.atom_types[:-1]) if s == self.species] + [
            self.N
        ]
        k = int(self.rng.choice(rows))
        self._n_disp_try += 1
        U_old, E0_old, E1_old = self._U(), self._E0, self._E1
        r_old = self.positions[k].copy()
        self.positions[k] += (self.rng.random(3) - 0.5) * 2.0 * self.max_disp
        if k == self.N:  # only U_{N+1} sees the fractional atom
            self._E1 = self._eval(self.positions, self.atom_types)
        else:
            self._refresh_energies()
        if self._accept(-self._beta * (self._U() - U_old)):
            self._n_disp_acc += 1
        else:
            self.positions[k] = r_old
            self._E0, self._E1 = E0_old, E1_old

    def _wang_landau(self) -> None:
        b = self._bin(self._lam)
        self._w[b] -= self._wl_f
        self._wl_hist[b] += 1
        h = self._wl_hist
        if h.min() > 0.8 * h.mean():
            self._wl_f *= 0.5
            self._wl_hist[:] = 0.0

    def _gcmc_cycle(self) -> None:
        equil = self._cycle < self.equil_cycles
        if not equil and self._cycle == self.equil_cycles:
            self._w -= self._w[0]  # freeze (the offset is irrelevant)
        for _ in range(self.gcmc_tries):
            if self.rng.random() < self.lambda_move_ratio:
                self._lambda_move()
                if equil and self.adapt_weights:
                    self._wang_landau()
            else:
                self._displacement_move()
            if not equil:
                self._samples.append(
                    (self._n_species(), self._lam, self._E1 - self._E0)
                )
        self._lam_log.append(self._lam)
        self._cycle += 1

    # ------------------------------------------------------------------ #
    #  Estimators (ADR-0009)                                               #
    # ------------------------------------------------------------------ #

    def _end_estimates(self, s: np.ndarray) -> dict:
        """<N>_0, <N>_1 and beta mu_ex from samples s = (N_sp, lam, dU_f)."""
        n = self.n_lambda_bins
        N, lam, dU = s[:, 0], s[:, 1], s[:, 2]
        bot, top = lam < 1.0 / n, lam >= 1.0 - 1.0 / n
        out = {
            "n_bottom": int(bot.sum()),
            "n_top": int(top.sum()),
            "N0": np.nan,
            "N1": np.nan,
            "beta_mu_ex": np.nan,
        }
        # pi(0) / pi(lam) = exp(beta g), pi(1) / pi(lam) = exp(-beta (dU - g))
        la0 = self._beta * self._g(lam[bot], dU[bot])
        la1 = -self._beta * (dU[top] - self._g(lam[top], dU[top]))
        c = max(
            la0.max() if bot.any() else -np.inf, la1.max() if top.any() else -np.inf
        )
        a0, a1 = np.exp(la0 - c), np.exp(la1 - c)
        if bot.any():
            out["N0"] = float(np.sum(a0 * N[bot]) / a0.sum())
        if top.any():
            out["N1"] = float(a1.sum() / np.sum(a1 / (N[top] + 1.0)))
        if bot.any() and top.any():
            ratio = (np.exp(-self._w[n - 1]) * a1.sum()) / (
                np.exp(-self._w[0]) * a0.sum()
            )
            out["beta_mu_ex"] = float(-np.log(ratio))
        return out

    def estimates(self, n_blocks: int = 10) -> dict:
        s = np.asarray(self._samples, dtype=float).reshape(-1, 3)
        n_frame = self.N - self._n_species()
        est = self._end_estimates(s) if len(s) else {}
        blocks = [self._end_estimates(b) for b in np.array_split(s, n_blocks) if len(b)]
        for key in ("N0", "N1", "beta_mu_ex"):
            v = np.array([b[key] for b in blocks], dtype=float)
            v = v[np.isfinite(v)]
            est[key + "_se"] = (
                float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
            )
        hist = (
            np.bincount([self._bin(x) for x in s[:, 1]], minlength=self.n_lambda_bins)
            if len(s)
            else []
        )
        est.update(
            N_framework=n_frame,
            n_samples=len(s),
            lambda_hist=[int(h) for h in hist],
            weights=[float(x) for x in self._w],
        )
        return est

    # ------------------------------------------------------------------ #
    #  Output                                                              #
    # ------------------------------------------------------------------ #

    def _write_xyz_frame(self, fh, cycle: int, e_pot: float) -> None:
        comment = (
            f"akitsushima.py CFCMC  cycle={cycle}  N={self.N}  "
            f"lambda={self._lam:.4f}  U={e_pot:.6f}eV"
        )
        _write_xdatcar_frame(
            fh,
            self.positions[:-1],
            self.lattice,
            self.atom_types[:-1],
            self._xdatcar_frame_idx,
            variable_header=True,
            comment=comment,
        )
        self._xdatcar_frame_idx += 1

    def run(self) -> dict:
        res = super().run()
        est = self.estimates()
        V = abs(np.linalg.det(self.lattice))
        kT = K_B * self.temperature
        est["mu_ex"] = est.get("beta_mu_ex", np.nan) * kT
        est["mu_ex_se"] = est.get("beta_mu_ex_se", np.nan) * kT
        n0 = est.get("N0", np.nan)
        # write the whole atoms only; the fractional one is reported apart
        res.update(
            positions=self.positions[:-1].copy(),
            velocities=self.velocities[:-1].copy(),
            atom_types=list(self.atom_types[:-1]),
            N_whole_mean=res["N_mean"],
            N_mean=est.get("N_framework", 0) + n0,
            N_mean_se=est.get("N0_se", np.nan),
            N_mean_lambda1=est.get("N_framework", 0) + est.get("N1", np.nan),
            N_mean_lambda1_se=est.get("N1_se", np.nan),
            mu_ex=est["mu_ex"],
            mu_ex_se=est["mu_ex_se"],
            rho_mean=(est.get("N_framework", 0) + n0) / V,
            fractional={"position": self.positions[-1].copy(), "lambda": self._lam},
            lambda_history=list(self._lam_log),
            acc_lambda=(
                self._n_lam_acc / self._n_lam_try if self._n_lam_try else float("nan")
            ),
            cfcmc=est,
        )
        if _mpi_rank == 0:
            write_poscar(
                os.path.join(self.output_dir, "POSCAR_gcmc_final"),
                self.lattice,
                res["positions"],
                res["atom_types"],
                title="CFCMC final frame (whole atoms)",
            )
            with open(os.path.join(self.output_dir, "lambda_vs_cycle.dat"), "w") as fh:
                fh.write("# cycle  lambda\n")
                fh.writelines(f"{c}  {x:.6f}\n" for c, x in enumerate(self._lam_log))
            import json

            with open(os.path.join(self.output_dir, "cfcmc_estimates.json"), "w") as fh:
                json.dump({k: v for k, v in est.items()}, fh, indent=1, default=float)
        if self.verbose:
            print(
                f"  CFCMC (lambda-end estimators, equil_cycles="
                f"{self.equil_cycles} discarded)"
            )
            print(f"  <N> (lambda=0) = {res['N_mean']:.3f} +/- {res['N_mean_se']:.3f}")
            print(
                f"  <N> (lambda=1) = {res['N_mean_lambda1']:.3f} +/- "
                f"{res['N_mean_lambda1_se']:.3f}"
            )
            print(f"  mu_ex          = {res['mu_ex']:.5f} +/- {res['mu_ex_se']:.5f} eV")
            print(f"  lambda hist    = {est.get('lambda_hist')}")
        return res
