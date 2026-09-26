"""Grand-canonical Monte Carlo + MD hybrid simulations (muVT ensemble):
GCMCMDSimulation (discrete insertion/deletion/displacement moves) and
OSMMCSimulation (continuous lambda-dynamics / soft-core insertion,
single- or multi-species).
"""
import os
import warnings
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

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

__all__ = ["SpeciesParams", "GCMCMDSimulation", "OSMMCSimulation"]


class _SpeciesParamsFields(NamedTuple):
    """Plain field/default declaration for :class:`SpeciesParams`, split
    into its own base class because ``typing.NamedTuple`` refuses to let
    a subclass override ``__new__`` in the SAME class body that declares
    the fields (``AttributeError: Cannot overwrite NamedTuple attribute
    __new__``, confirmed directly against this Python version).
    Subclassing a step further, as :class:`SpeciesParams` below does, is
    unaffected by that restriction and is the standard workaround."""
    mu        : float
    mass      : Optional[float] = None
    m_lam     : Optional[float] = None
    sc_alpha  : float = 0.5
    sc_n      : int = 4
    lambda_dB : Optional[float] = None
    weight    : Optional[float] = None


class SpeciesParams(_SpeciesParamsFields):
    """
    Per-species lambda-dynamics parameters for :class:`OSMMCSimulation`'s
    multi-species mode (``species_params={symbol: SpeciesParams(...)}``).

    Bug-fix note (S14): this type was referenced throughout the original
    ``OSMMCSimulation.__init__`` (``isinstance(sp, SpeciesParams)``, the
    class docstring's own multi-species example, error messages) but was
    never actually defined anywhere in the file -- so passing
    ``species_params=`` to construct a multi-species OSMMC simulation
    raised ``NameError: name 'SpeciesParams' is not defined`` on the very
    first ``isinstance`` check, making the documented multi-species
    feature entirely unusable. Field names, the one mandatory field
    (``mu``), and the defaults for ``sc_alpha``/``sc_n`` (matching
    the single-species shorthand constructor defaults) are all inferred
    directly from how ``sp.<field>`` is actually accessed in
    ``OSMMCSimulation.__init__``.

    Rename note (``mu_ex`` -> ``mu``): the field was originally named
    ``mu_ex`` and documented as the *excess* chemical potential, which
    was wrong -- it was always the FULL chemical potential fed straight
    into the standard grand-canonical acceptance rule (see
    :class:`GCMCMDSimulation`'s own ``mu`` parameter note for the
    derivation and the ideal-gas calculation that settled it). The field
    is renamed to ``mu`` so the name states what the value actually is.
    ``mu_ex=`` is kept as a deprecated constructor alias (mutually
    exclusive with ``mu=``, warns via :class:`UserWarning`), and
    ``.mu_ex`` is kept as a deprecated read-only property returning
    ``.mu`` (also warns), so existing code keeps working unchanged.

    Parameters
    ----------
    mu        : FULL chemical potential mu [eV].  Required.
    mass      : atomic mass [amu].  None -> ATOMIC_MASS lookup.
    m_lam     : lambda virtual mass [amu.A^2].  None -> scaled from Ar=39.948 reference.
    sc_alpha  : soft-core alpha (void size at lambda=0).
    sc_n      : soft-core exponent n.
    lambda_dB : thermal de Broglie wavelength [A].  None -> computed from mass/T.
    weight    : relative sampling weight for species selection.  None -> 1.0.
    """
    __slots__ = ()

    def __new__(cls, mu=None, mass=None, m_lam=None, sc_alpha=0.5, sc_n=4,
                lambda_dB=None, weight=None, mu_ex=None):
        if mu_ex is not None:
            if mu is not None:
                raise TypeError(
                    "SpeciesParams: pass either 'mu' or the deprecated "
                    "'mu_ex' alias, not both.")
            warnings.warn(
                "SpeciesParams: the 'mu_ex' argument is deprecated; use "
                "'mu' instead (it was always the full chemical "
                "potential, not the excess -- see the class docstring).",
                UserWarning, stacklevel=2)
            mu = mu_ex
        if mu is None:
            raise TypeError(
                "SpeciesParams: 'mu' (the full chemical potential [eV]) "
                "is required.")
        return super().__new__(cls, float(mu), mass, m_lam, sc_alpha,
                               sc_n, lambda_dB, weight)

    @property
    def mu_ex(self):
        """Deprecated alias for :attr:`mu`.  See the class docstring's
        rename note."""
        warnings.warn(
            "SpeciesParams.mu_ex is deprecated; use .mu instead (it is "
            "the full chemical potential, not the excess).",
            UserWarning, stacklevel=2)
        return self.mu


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

    Both extensions are placeholders for the planned OSMMC implementation
    and can be combined independently.

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
    # (XDATCAR_<_xdatcar_task_name>).  Subclasses (e.g. OSMMCSimulation)
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
                 # Extension hooks (used by OSMMCSimulation subclass)
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


#  SECTION 6.5: Open System Molecular Dynamics (OSMMC, lambda-dynamics, muVT)
# ============================================================================

class OSMMCSimulation(GCMCMDSimulation):
    """
    Open System Molecular Dynamics via continuous lambda-dynamics (muVT ensemble).

    Supports **single-species** (backward-compatible) and **multi-species**
    grand-canonical MD via the ``species_params`` dict interface.

    Parameters
    ----------
    lattice, positions, atom_types, force_fn, temperature, timestep,
    n_cycles, md_steps_per_cycle, gcmc_attempts_per_cycle, tau_nhc,
    traj_every, thermo_every, output_dir, skin_width, seed, verbose
        Inherited from :class:`GCMCMDSimulation`.

    species_params : dict {symbol: SpeciesParams} -- **multi-species mode**.
        When supplied, the single-species shorthand parameters below are
        ignored.  Each key is an element symbol; the value is a
        :class:`SpeciesParams` namedtuple.

    species  : str   -- single-species symbol.  Ignored when species_params given.
    mu       : float -- single-species FULL chemical potential mu [eV]
               (see GCMCMDSimulation's ``mu`` parameter note).  The
               deprecated ``mu_ex=`` alias is also accepted here (it is
               forwarded to :class:`GCMCMDSimulation` via ``**kwargs``).
    mass     : float -- single-species atomic mass [amu].  Default: ATOMIC_MASS.
    m_lam    : float -- single-species lambda virtual mass [amu.A^2].  Default 50.
    sc_alpha : float -- single-species soft-core alpha.  Default 0.5.
    sc_n     : int   -- single-species soft-core exponent n.  Default 4.

    lam_dt_frac    : float -- lambda integrator timestep as fraction of dt. Default 1.0.
    insertion_mode : str   -- ``'random'`` or ``'biased'``.  Default ``'random'``.
    n_probe        : int   -- candidate positions for biased insertion.  Default 20.
    lambda_probe   : float -- lambda used to score probe positions.  Default 0.1.

    Examples
    --------
    **Single-species (Ar, backward-compatible):**

    >>> sim = OSMMCSimulation(
    ...     lat, pos, ['Ar']*N, fn,
    ...     temperature=120., mu=-0.05, species='Ar',
    ...     m_lam=50., sc_alpha=0.5, sc_n=4,
    ...     insertion_mode='biased', n_probe=20)

    **Multi-species (Li2O, stoichiometric weighting):**

    >>> sim = OSMMCSimulation(
    ...     lat, pos, types, fn,
    ...     temperature=1000.,
    ...     species_params={
    ...         'Li': SpeciesParams(mu=-0.12, mass=6.941,
    ...                             m_lam=7.0, sc_alpha=0.55,
    ...                             sc_n=4, weight=2.0),
    ...         'O' : SpeciesParams(mu=-0.35, mass=15.999,
    ...                             m_lam=16.0, sc_alpha=0.40,
    ...                             sc_n=4, weight=1.0),
    ...     },
    ...     n_cycles=5000, md_steps_per_cycle=20,
    ...     insertion_mode='biased')
    """

    # Overrides GCMCMDSimulation._xdatcar_task_name -> XDATCAR_osmc
    _xdatcar_task_name = "osmc"

    # ADR-0007: the ghost particle couples through a soft-core LJ taken
    # from the engine's pair parameters (_get_lj_params), which an MLIP
    # does not have, and osmc fails its own muVT check (ADR-0003 T7,
    # findings O1-O2).  Refused until those are settled.
    _MLIP_REFUSAL = ("its ghost particle couples through soft-core LJ "
                     "parameters an MLIP does not have, and osmc fails the "
                     "muVT check (ADR-0003 T7, O1-O2)")

    def __init__(self,
                 *args,
                 # -- multi-species -------------------------------------
                 species_params  : Optional[Dict[str, object]] = None,
                 # -- single-species shorthand (backward-compatible) ----
                 m_lam          : float = 50.0,
                 sc_alpha       : float = 0.5,
                 sc_n           : int   = 4,
                 # -- lambda integrator ---------------------------------
                 lam_dt_frac    : float = 1.0,
                 # -- placement strategy --------------------------------
                 insertion_mode : str   = 'random',
                 n_probe        : int   = 20,
                 lambda_probe   : float = 0.1,
                 _allow_unverified: bool = False,
                 **kwargs):

        # Refused until redesigned (ADR-0003 T7, O2; decided 2026-09-26):
        # the lambda dynamics does not sample the muVT ensemble -- a
        # deletion removes a random atom with no energy criterion and the
        # lambda force has no translational term, so with O1 fixed N grows
        # without bound (ideal gas: 30 -> 710 in 713 cycles, exact 40).
        # The planned replacement is CFCMC (Shi & Maginn, J. Chem. Theory
        # Comput. 3, 1451 (2007)).  _allow_unverified is for tests of the
        # present code only.
        if not _allow_unverified:
            raise ValueError(
                "OSMMCSimulation (osmc) is disabled: its lambda dynamics does "
                "not sample the muVT ensemble (ADR-0003 T7, finding O2; N "
                "grows without bound).  Use gcmc, which passes the NIST check.")

        kwargs.setdefault('cavity_bias',     False)
        kwargs.setdefault('energy_bias_del', False)
        super().__init__(*args, **kwargs)

        if insertion_mode not in ('random', 'biased'):
            raise ValueError(
                f"OSMMCSimulation: insertion_mode must be 'random' or "
                f"'biased', got '{insertion_mode}'.")

        self.lam_dt_frac    = float(lam_dt_frac)
        self.insertion_mode = insertion_mode
        self.n_probe        = int(n_probe)
        self.lambda_probe   = float(lambda_probe)

        # -- Build internal species parameter table ------------
        T = self.temperature
        h_J = 6.62607015e-34; kB_J = 1.380649e-23; amu_kg = 1.66053906660e-27

        def _ldb(m_amu: float) -> float:
            return (h_J / np.sqrt(2.0 * np.pi * m_amu * amu_kg * kB_J * T)
                    ) * 1e10   # -> A

        def _mlam_default(m_amu: float) -> float:
            """Virtual mass scales with atomic mass (reference: Ar = 39.948)."""
            return 50.0 * m_amu / 39.948

        if species_params is not None:
            # -- Multi-species mode --------------------------------
            if not isinstance(species_params, dict) or not species_params:
                raise ValueError(
                    "OSMMCSimulation: species_params must be a non-empty "
                    "dict {symbol: SpeciesParams(mu=..., ...)}.")
            for sym, sp in species_params.items():
                if not isinstance(sp, SpeciesParams):
                    raise TypeError(
                        f"species_params['{sym}'] must be a SpeciesParams "
                        f"namedtuple, got {type(sp).__name__}.")
                # sp.mu is never None here: SpeciesParams.__new__ already
                # requires it (or the deprecated mu_ex= alias) and raises
                # TypeError otherwise, so this is a defensive re-check,
                # not the primary validation path.
                if sp.mu is None:
                    raise ValueError(
                        f"species_params['{sym}'].mu is required "
                        "(the full chemical potential mu [eV]).")

            self._multi_species = True
            self._sp_table: Dict[str, dict] = {}

            for sym, sp in species_params.items():
                mass  = float(sp.mass)    if sp.mass    is not None \
                        else float(ATOMIC_MASS.get(sym, 1.0))
                m_l   = float(sp.m_lam)   if sp.m_lam   is not None \
                        else _mlam_default(mass)
                ldb   = float(sp.lambda_dB) if sp.lambda_dB is not None \
                        else _ldb(mass)
                self._sp_table[sym] = {
                    'mu'       : float(sp.mu),
                    'mass'     : mass,
                    'm_lam'    : m_l,
                    'sc_alpha' : float(sp.sc_alpha),
                    'sc_n'     : int(sp.sc_n),
                    'lambda_dB': ldb,
                    'weight'   : float(sp.weight) if sp.weight is not None
                                 else 1.0,
                }

            syms    = list(self._sp_table.keys())
            wts     = np.array([self._sp_table[s]['weight'] for s in syms],
                               dtype=float)
            wts    /= wts.sum()
            self._sp_symbols = syms
            self._sp_weights = wts

            # Seed parent attrs with first species (overwritten each cycle)
            first = syms[0]
            self.species      = first
            self.mu           = self._sp_table[first]['mu']
            self.mass         = self._sp_table[first]['mass']
            self._lambda_dB   = self._sp_table[first]['lambda_dB']

        else:
            # -- Single-species mode (backward-compatible) ---------
            self._multi_species = False
            self._sp_table = {}
            self._sp_symbols = [self.species]
            self._sp_weights = np.array([1.0])
            self._sp_table[self.species] = {
                'mu'       : self.mu,
                'mass'     : self.mass,
                'm_lam'    : float(m_lam),
                'sc_alpha' : float(sc_alpha),
                'sc_n'     : int(sc_n),
                'lambda_dB': self._lambda_dB,
                'weight'   : 1.0,
            }

        # Active species for the current cycle (set by _choose_active_species)
        self._active_sp : str = self.species
        self._apply_species_params(self._active_sp)   # load live attrs

        # Ghost-particle state
        self._lam        : float      = 0.0
        self._lam_dot    : float      = 0.0
        self._ghost_pos  : np.ndarray = self._new_ghost_position()
        self._ghost_vel  : np.ndarray = self._new_particle_velocity()
        self._del_candidate : int     = -1

        # Bias corrections (per-cycle)
        self._ins_bias_dmu : float = 0.0
        self._del_bias_dmu : float = 0.0

        # Diagnostics
        self._n_ins_complete = 0
        self._n_del_complete = 0
        self._n_lam_reflect  = 0
        # Per-species insertion/deletion counters
        self._sp_ins_count : Dict[str, int] = {s: 0 for s in self._sp_symbols}
        self._sp_del_count : Dict[str, int] = {s: 0 for s in self._sp_symbols}

    # ------------------------------------------------------------------ #
    #  Active species management                                           #
    # ------------------------------------------------------------------ #

    def _choose_active_species(self) -> str:
        """
        Select which species to use for the current lambda-dynamics cycle.

        Single-species: always returns the one registered species.
        Multi-species : samples proportionally to ``_sp_weights``.

        Calls :meth:`_apply_species_params` to propagate the chosen
        species' parameters into all live instance attributes.
        """
        if len(self._sp_symbols) == 1:
            sp = self._sp_symbols[0]
        else:
            sp = str(self.rng.choice(self._sp_symbols, p=self._sp_weights))
        self._active_sp = sp
        self._apply_species_params(sp)
        return sp

    def _apply_species_params(self, sp: str) -> None:
        """
        Load the per-species lambda-dynamics parameters for species ``sp``
        into the live instance attributes used by all downstream methods.

        Updated attributes
        ------------------
        ``species``     -- element symbol of the active ghost
        ``mass``        -- atomic mass [amu]  -> used for ghost Verlet
        ``mu``          -- full chemical potential mu [eV]  -> lambda
                           driving force
        ``_lambda_dB``  -- thermal de Broglie wavelength [A]  -> lambda^3 in acceptance
        ``_m_lam``      -- virtual mass [amu.A^2]  -> lambda equation of motion
        ``_sc_alpha``   -- soft-core alpha  -> void size at lambda = 0
        ``_sc_n``       -- soft-core exponent n

        This is the single place where element-dependent parameters are
        committed; all other methods read from these attributes.
        """
        p = self._sp_table[sp]
        self._active_sp  = sp      # <- must be set before any method reads it
        self.species     = sp
        self.mass        = p['mass']
        self.mu          = p['mu']
        self._lambda_dB  = p['lambda_dB']
        self._m_lam      = p['m_lam']
        self._sc_alpha   = p['sc_alpha']
        self._sc_n       = p['sc_n']

    # ------------------------------------------------------------------ #
    #  Ghost-particle placement                                            #
    # ------------------------------------------------------------------ #

    def _new_ghost_position(self) -> np.ndarray:
        """
        Sample the initial Cartesian position for a new ghost particle.

        Uses the **active species'** soft-core parameters when scoring
        candidate positions in ``'biased'`` mode.

        Random mode
        -----------
        Uniform random in the unit cell.  At lambda = 0 the soft-core potential
        is exactly zero, so any position is equally valid energetically.

        Biased mode (cavity-biased)
        ---------------------------
        ``n_probe`` candidates are scored with the soft-core at
        ``lambda_probe``.  The lowest-energy position (deepest void) is
        selected.  A Boltzmann-weight bias correction is stored in
        ``_ins_bias_dmu`` and subtracted from the effective mu in the lambda
        equation of motion to preserve detailed balance.
        """
        if self.insertion_mode == 'random' or self.N == 0:
            self._ins_bias_dmu = 0.0
            return self.rng.uniform(0, 1, 3) @ self.lattice

        probes  = self.rng.uniform(0, 1, (self.n_probe, 3)) @ self.lattice
        e_probe = np.array([
            self._softcore_energy_at(p, self.lambda_probe)
            for p in probes
        ])
        log_w  = -self._beta * e_probe
        log_w -= log_w.max()
        w      = np.exp(log_w); w_norm = w / w.sum()
        idx_b  = int(np.argmin(e_probe))
        # Bias correction: deltamu = -kT ln(w_best / (1/n_probe))
        self._ins_bias_dmu = -(1.0 / self._beta) * (
            float(np.log(w_norm[idx_b] + 1e-300)) + np.log(self.n_probe))
        return probes[idx_b]

    def _select_deletion_candidate(self) -> int:
        """
        Choose which real atom begins a lambda: 1 -> 0 deletion transition
        for the **active species**.

        Random mode
        -----------
        Uniform random selection among atoms of the active species.

        Biased mode (energy-biased)
        ---------------------------
        Candidate selected with probability proportional to exp(+beta u_i), where u_i is
        the local energy of atom i.  High-energy (unstable) atoms are
        preferred.  The log-bias correction is stored in ``_del_bias_dmu``
        and added to the effective mu to preserve detailed balance.
        """
        sp_idx = [i for i, s in enumerate(self.atom_types)
                  if s == self._active_sp]
        if not sp_idx:
            return -1

        self._del_bias_dmu = 0.0

        if self.insertion_mode == 'random' or len(sp_idx) == 1:
            return int(self.rng.choice(sp_idx))

        u_local = np.array([self._local_energy(i) for i in sp_idx])
        log_w   = self._beta * u_local
        log_w  -= log_w.max()
        w       = np.exp(log_w); w_sum = w.sum()
        w_norm  = w / w_sum if w_sum > 0 else np.ones(len(w)) / len(w)
        cl      = int(self.rng.choice(len(sp_idx), p=w_norm))
        idx     = sp_idx[cl]
        # Bias correction: deltamu = +kT ln(w_i x n)
        self._del_bias_dmu = (1.0 / self._beta) * (
            float(np.log(w_norm[cl] + 1e-300)) + np.log(len(sp_idx)))
        return idx

    # ------------------------------------------------------------------ #
    #  Soft-core potential  (element-dependent epsilon, sigma, alpha, n)               #
    # ------------------------------------------------------------------ #

    def _softcore_energy_at(self,
                             r_ghost : np.ndarray,
                             lam     : float,
                             ) -> float:
        """
        Soft-core interaction energy of the ghost at ``r_ghost`` with all
        real atoms of the **active species**, using the active species'
        epsilon, sigma, alpha, and n.

        At lambda = 0 returns exactly 0.0.
        """
        if self.N == 0 or lam < 1e-12:
            return 0.0
        lj = self._get_lj_params()
        if lj is None:
            return self._gaussian_repulsion(r_ghost, lam)
        eps, sig  = lj
        inv_lat   = np.linalg.inv(self.lattice)
        da        = self._sc_alpha * (1.0 - lam) ** 2
        lam_n     = lam ** self._sc_n
        E = 0.0
        for i in range(self.N):
            if self.atom_types[i] != self._active_sp:
                continue
            rij = self.positions[i] - r_ghost
            s   = rij @ inv_lat;  s -= np.round(s)
            rij = s @ self.lattice
            r6  = (np.dot(rij, rij) / sig**2) ** 3
            d   = da + r6
            if d < 1e-30:
                continue
            E += 4.0 * eps * lam_n * (1.0 / d**2 - 1.0 / d)
        return float(E)

    def _softcore_force_dlam(self,
                              r_ghost : np.ndarray,
                              lam     : float,
                              ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Soft-core forces on ghost + real atoms and dV_sc/dlambda, using the
        **active species'** element-dependent parameters (epsilon, sigma, alpha, n).

        Returns
        -------
        f_ghost : (3,)    force on ghost [eV/A]
        f_real  : (N, 3)  forces on real atoms [eV/A]
        dVdlam  : float   dV_sc/dlambda [eV]
        """
        f_ghost = np.zeros(3)
        f_real  = np.zeros((self.N, 3))
        dVdlam  = 0.0
        if self.N == 0 or lam < 1e-12:
            return f_ghost, f_real, dVdlam
        lj = self._get_lj_params()
        if lj is None:
            return f_ghost, f_real, dVdlam
        eps, sig  = lj
        inv_lat   = np.linalg.inv(self.lattice)
        n_sc      = self._sc_n
        lam_n     = lam ** n_sc
        lam_n1    = lam ** max(n_sc - 1, 0)
        da_dlam   = -2.0 * self._sc_alpha * (1.0 - lam)
        for i in range(self.N):
            if self.atom_types[i] != self._active_sp:
                continue
            rij  = self.positions[i] - r_ghost
            s    = rij @ inv_lat;  s -= np.round(s)
            rij  = s @ self.lattice
            r2   = np.dot(rij, rij)
            r6   = (r2 / sig**2) ** 3
            d    = self._sc_alpha * (1.0 - lam)**2 + r6
            if d < 1e-30:
                continue
            d2, d3 = d**2, d**3
            # dV/dlambda = 4epsilon { n lambda^{n-1} [1/d^2-1/d] + lambda^n (dalpha/dlambda) [-2/d^3+1/d^2] }
            dVdlam += 4.0 * eps * (
                n_sc * lam_n1 * (1.0/d2 - 1.0/d)
                + lam_n * da_dlam * (-2.0/d3 + 1.0/d2))
            dVdr6   = 4.0 * eps * lam_n * (-2.0/d3 + 1.0/d2)
            fg_i    = -dVdr6 * (-6.0 * r6 / (r2 + 1e-30) * rij)
            f_ghost += fg_i
            f_real[i] -= fg_i
        return f_ghost, f_real, float(dVdlam)

    def _gaussian_repulsion(self, r_ghost: np.ndarray, lam: float) -> float:
        """Fallback soft-core when LJ parameters are unavailable."""
        if self.N == 0:
            return 0.0
        inv_lat = np.linalg.inv(self.lattice)
        E = 0.0
        for i in range(self.N):
            if self.atom_types[i] != self._active_sp:
                continue
            rij = self.positions[i] - r_ghost
            s   = rij @ inv_lat;  s -= np.round(s)
            rij = s @ self.lattice
            E  += lam * np.exp(-np.dot(rij, rij))
        return float(E)

    def _get_lj_params(self) -> Optional[Tuple[float, float]]:
        """
        Extract (epsilon, sigma) for the **active species** self-pair from force_fn.

        Bug-fix note (S12): the original inspected ``force_fn.__closure__``
        cell-by-cell looking for a captured ``PotentialEngine`` instance --
        a brittle, implementation-dependent trick that silently stops
        working the moment the closure's captured-variable order or count
        changes (which is exactly what happened when make_force_fn was
        rewritten to always build a real PotentialEngine rather than
        inlining a bare closure -- see fitting/force_fn.py). force_fn
        objects now carry an explicit ``.engine`` attribute when one
        exists, so this can read it directly with no introspection and
        no dependency on PotentialEngine's import at all.

        Returns None when not found (triggers Gaussian fallback).
        """
        sp = self._active_sp
        engine = getattr(self._force_fn, 'engine', None)
        if engine is None:
            return None
        for pot in getattr(engine, '_potentials', []):
            try:
                p = pot._get_params(sp, sp)
            except Exception:
                p = None
            if p is not None:
                eps = p.get('epsilon') or p.get('D_e') or p.get('De', 0.0)
                sig = p.get('sigma')   or p.get('r0',  0.0)
                if eps and eps > 0 and sig and sig > 0:
                    return float(eps), float(sig)
        return None

    def _local_energy(self, atom_idx: int) -> float:
        """
        Local interaction energy of atom ``atom_idx`` [eV].

        Bug-fix note (S3): delegates to the shared, correctly-implemented
        _local_energy_of_atom (see GCMCMDSimulation) instead of the
        original's ``-_probe_energy(self.positions[atom_idx])``, which
        placed a ghost particle exactly on top of the still-present real
        atom (an r=0 self-overlap divergence) -- see
        GCMCMDSimulation._local_energy_of_atom's docstring for the full
        explanation.
        """
        return self._local_energy_of_atom(atom_idx)

    # ------------------------------------------------------------------ #
    #  lambda dynamics                                                          #
    # ------------------------------------------------------------------ #

    def _lambda_step(self) -> None:
        """
        Integrate the ghost particle and lambda for one MD cycle using the
        **active species'** element-dependent m_lam, sc_alpha, sc_n, mu.

        The effective mu driving lambda is:
            mu_eff = mu  +  deltamu_ins  +  deltamu_del
        where deltamu_ins and deltamu_del are bias corrections that preserve
        detailed balance when biased placement is enabled.
        """
        dt    = self.dt * self.lam_dt_frac
        m_l   = self._m_lam
        mu_eff = self.mu + self._ins_bias_dmu + self._del_bias_dmu
        inv_lat = np.linalg.inv(self.lattice)

        for _ in range(self.md_steps):
            f_ghost, f_real_sc, dVdlam = self._softcore_force_dlam(
                self._ghost_pos, self._lam)

            # -- Ghost Velocity Verlet -----------------------------
            a_ghost = f_ghost * EV_ANG_TO_AMU_ANG_FS2 / self.mass
            self._ghost_vel  += 0.5 * dt * a_ghost
            self._ghost_pos  += dt * self._ghost_vel
            frac = self._ghost_pos @ inv_lat
            # `frac -= np.floor(frac)` alone maps a tiny negative frac to
            # exactly 1.0 in float64, which scipy's cKDTree rejects; see
            # wrap_positions_into_cell.
            frac -= np.floor(frac)
            frac[frac >= 1.0] = 0.0
            self._ghost_pos = frac @ self.lattice

            # Soft-core force on real atoms (half-step)
            if self.N > 0:
                self.velocities += (0.5 * dt * f_real_sc
                                    * EV_ANG_TO_AMU_ANG_FS2
                                    / self._masses[:, None])

            # -- lambda Velocity Verlet ----------------------------
            lam_acc = (-dVdlam - mu_eff) * EV_ANG_TO_AMU_ANG_FS2 / m_l
            self._lam_dot += 0.5 * dt * lam_acc
            self._lam     += dt * self._lam_dot

            event = self._handle_lambda_boundary()
            if event in ('inserted', 'deleted'):
                break   # ghost state reset; restart cycle

            f_ghost2, f_real_sc2, dVdlam2 = self._softcore_force_dlam(
                self._ghost_pos, self._lam)

            a_ghost2 = f_ghost2 * EV_ANG_TO_AMU_ANG_FS2 / self.mass
            self._ghost_vel += 0.5 * dt * a_ghost2

            if self.N > 0:
                self.velocities += (0.5 * dt * f_real_sc2
                                    * EV_ANG_TO_AMU_ANG_FS2
                                    / self._masses[:, None])

            lam_acc2 = (-dVdlam2 - mu_eff) * EV_ANG_TO_AMU_ANG_FS2 / m_l
            self._lam_dot += 0.5 * dt * lam_acc2

    def _handle_lambda_boundary(self) -> str:
        """
        Handle lambda boundary events.

        lambda >= 1 -> insertion complete: graduate ghost to real atom,
                choose next active species, place new ghost at lambda = 0.
        lambda <= 0 -> deletion complete (if candidate active): remove candidate,
                choose next active species, place new ghost.

        In both cases the per-species counters are updated and the new
        active species' parameters are loaded via ``_choose_active_species``.
        """
        if self._lam >= 1.0:
            self._lam     = 0.0
            self._lam_dot = abs(self._lam_dot)   # elastic wall
            self._n_lam_reflect += 1

            self._accept_insertion(self._ghost_pos.copy())
            self._n_ins_complete += 1
            self._sp_ins_count[self._active_sp] = (
                self._sp_ins_count.get(self._active_sp, 0) + 1)

            # Next cycle: pick active species, place new ghost
            self._choose_active_species()
            self._ghost_pos    = self._new_ghost_position()
            self._ghost_vel    = self._new_particle_velocity()
            self._ins_bias_dmu = 0.0
            return 'inserted'

        elif self._lam <= 0.0:
            self._lam     = 0.0
            # Elastic wall, upward.  Bug-fix note (ADR-0003 T7, O1): this
            # was -abs(), which kept lambda moving down, so it stayed at 0,
            # hit this branch every step and deleted the candidate each
            # time (N -> 0.74 against the NIST 41.9).
            self._lam_dot = abs(self._lam_dot)
            self._n_lam_reflect += 1

            if self._del_candidate >= 0:
                self._accept_deletion(self._del_candidate)
                self._del_candidate = -1
                self._n_del_complete += 1
                self._sp_del_count[self._active_sp] = (
                    self._sp_del_count.get(self._active_sp, 0) + 1)
                self._del_bias_dmu = 0.0

            self._choose_active_species()
            self._ghost_pos = self._new_ghost_position()
            self._ghost_vel = self._new_particle_velocity()
            return 'deleted'

        return ''

    # ------------------------------------------------------------------ #
    #  Override: _gcmc_cycle -> lambda-dynamics step                           #
    # ------------------------------------------------------------------ #

    def _gcmc_cycle(self) -> None:
        """
        Choose the active species, optionally pre-select a deletion
        candidate, then propagate lambda-dynamics for one MD cycle.
        """
        self._choose_active_species()

        if (self._del_candidate < 0 and self.N > 0
                and self.rng.random() < 0.5):
            self._del_candidate = self._select_deletion_candidate()

        self._lambda_step()

    # ------------------------------------------------------------------ #
    #  Diagnostic properties and helpers                                  #
    # ------------------------------------------------------------------ #

    @property
    def lambda_current(self) -> float:
        """Current lambda of the ghost particle."""
        return self._lam

    @property
    def active_species(self) -> str:
        """Element symbol of the ghost particle in the current cycle."""
        return self._active_sp

    @property
    def n_insertions(self) -> int:
        """Total completed insertions (lambda reached 1)."""
        return self._n_ins_complete

    @property
    def n_deletions(self) -> int:
        """Total completed deletions (lambda reached 0)."""
        return self._n_del_complete

    def species_counts(self) -> Dict[str, int]:
        """
        Return the current number of real atoms per fluctuating species.

        >>> sim.species_counts()
        {'Li': 32, 'O': 16}
        """
        return {s: self.atom_types.count(s) for s in self._sp_symbols}

    def species_stats(self) -> Dict[str, dict]:
        """
        Return per-species insertion/deletion counts and all lambda-dynamics
        parameters as a nested dict.

        Keys per species:
            n_ins, n_del, mu, mass, m_lam, sc_alpha, sc_n,
            lambda_dB, weight, N_current.

        Rename note (``mu_ex`` -> ``mu``): the returned dict also carries
        a deprecated ``mu_ex`` key duplicating ``mu`` (same value, no
        warning -- a plain dict key can't intercept access the way a
        property can), kept for any external code parsing this method's
        return value directly. Use ``mu`` going forward.
        """
        out = {}
        for sp in self._sp_symbols:
            p = self._sp_table[sp]
            out[sp] = {
                'n_ins'    : self._sp_ins_count.get(sp, 0),
                'n_del'    : self._sp_del_count.get(sp, 0),
                'mu'       : p['mu'],
                'mu_ex'    : p['mu'],   # deprecated duplicate, see docstring
                'mass'     : p['mass'],
                'm_lam'    : p['m_lam'],
                'sc_alpha' : p['sc_alpha'],
                'sc_n'     : p['sc_n'],
                'lambda_dB': p['lambda_dB'],
                'weight'   : p['weight'],
                'N_current': self.atom_types.count(sp),
            }
        return out

    def _acceptance_rates(self) -> Tuple[float, float, float]:
        """lambda-completion rates (insertions/cycle, deletions/cycle, nan)."""
        n = max(getattr(self, '_n_calls_total', 1), 1)
        return (self._n_ins_complete / n,
                self._n_del_complete / n,
                float('nan'))

    # ------------------------------------------------------------------ #
    #  Override run() and XYZ writer                                      #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """
        Execute the OSMMC simulation.

        Additional keys in return dict (beyond GCMCMDSimulation.run):
            n_insertions, n_deletions, insertion_mode,
            species_stats, multi_species, timing.
        """
        self._n_calls_total = 0
        result = super().run()
        result['n_insertions']   = self._n_ins_complete
        result['n_deletions']    = self._n_del_complete
        result['insertion_mode'] = self.insertion_mode
        result['species_stats']  = self.species_stats()
        result['multi_species']  = self._multi_species
        # timing is inherited from GCMCMDSimulation.run() via super()
        return result

    def _write_xyz_frame(self, fh, cycle: int, e_pot: float) -> None:
        """
        Append one configuration to the open XDATCAR file.

        OS-MC allows particle insertion/deletion (variable N), so --
        as with GCMCMDSimulation -- a full header is written before
        every frame, reflecting the CURRENT particle count.

        The soft-core "ghost" particle (partially inserted, lambda in
        (0,1)) is NOT a real VASP element and is therefore excluded
        from the XDATCAR output; only fully-real atoms are written.
        The ghost particle's position/species/lambda are still tracked
        separately in the thermo log for internal bookkeeping.
        """
        comment = (f"akitsushima.py OS-MC  cycle={cycle}  N={self.N}  "
                  f"ghost_sp={self._active_sp}  lambda={self._lam:.4f}  "
                  f"E_pot={e_pot:.6f}eV  T={self._temperature_inst():.2f}K")
        _write_xdatcar_frame(fh, self.positions, self.lattice,
                             self.atom_types, self._xdatcar_frame_idx,
                             variable_header=True, comment=comment)
        self._xdatcar_frame_idx += 1
