"""Thermal conductivity via Green-Kubo (equilibrium MD) or the
Slack-Debye approximation.

Two methods
-----------
1. Green-Kubo (equilibrium MD)
   kappa_{alpha,beta} = V/(k_B T^2) int_0^inf <J_alpha(t) J_beta(0)> dt

   The heat flux J is computed from the Hardy (1963) / Schelling (2002)
   formula for pair-potential systems:

     J = (1/V) [ sum_i E_i v_i  +  (1/2) sum_{i<j} (f_ij . (v_i + v_j)) r_ij ]

   where E_i = epsilon_kin,i + epsilon_pot,i is the per-atom total energy.

2. Slack-Debye approximation
   Estimates kappa from the Debye temperature and Gruneisen parameter
   without running MD. Useful as a quick order-of-magnitude check.

Workflow (Green-Kubo)
---------------------
1. Run NVT equilibration (handled internally by run_md()).
2. Run NVE production with heat-flux recording (run_md()).
3. Post-process: compute HCACF and integrate to kappa (compute_kappa()).

References
----------
Green-Kubo  : Green (1954), Kubo (1957)
Heat flux   : Hardy, J. Chem. Phys. 76, 622 (1963)
              Schelling et al., PRB 65, 144306 (2002)
Slack-Debye : Slack, Solid State Phys. 34 (1979)
"""
import os
import warnings
from typing import Callable, List, Optional

import numpy as np

from ..constants import ATOMIC_MASS, EV_ANG3_TO_GPA, EV_ANG_TO_AMU_ANG_FS2, K_B
from ..utils.optional_matplotlib import HAS_MATPLOTLIB, plt
from .molecular_dynamics import NVTSimulation
from .phonon import PhononCalculator

__all__ = ["ThermalConductivityCalculator"]

#  Unit conversion: 1 eV/(A.K.fs) = 1.6022x10^6 W/(m.K)
_EV_ANG_K_FS_TO_W_MK = (1.60217663e-19) / (1e-10 * 1e-15)   # ~= 1.6022e6


class ThermalConductivityCalculator:
    """
    Thermal conductivity tensor kappa_{alpha,beta} by Green-Kubo MD or
    Slack-Debye.

    Parameters
    ----------
    lattice    : (3,3) lattice vectors [A].
    positions  : (N,3) equilibrium positions [A].
    atom_types : list of element symbols, length N.
    force_fn   : unified force callable from make_force_fn().
    masses     : (N,) array [amu].  Defaults to ATOMIC_MASS lookup.
    temperature: simulation temperature [K].

    Examples
    --------
    **Green-Kubo**::

        tc = ThermalConductivityCalculator(
            lat, pos, types, make_force_fn(engine), temperature=1000)
        md_result = tc.run_md(
            n_steps=50_000, timestep=0.5,
            equil_steps=10_000, output_dir="tc_run")
        kappa = tc.compute_kappa(tau_max=10.0)   # [W/(m.K)]
        tc.save_results("tc_run")

    **Slack-Debye** (quick estimate from phonon dispersion)::

        kappa_slack = tc.slack_debye(phonon_calc, T=300)
    """

    def __init__(self,
                 lattice    : np.ndarray,
                 positions  : np.ndarray,
                 atom_types : list,
                 force_fn   : Callable,
                 masses     : Optional[np.ndarray] = None,
                 temperature: float = 300.0):

        self.lattice    = np.asarray(lattice,    dtype=float).copy()
        self.positions  = np.asarray(positions,  dtype=float).copy()
        self.atom_types = list(atom_types)
        self.force_fn   = force_fn
        self.temperature = float(temperature)
        self.N = len(atom_types)
        from ..fitting.force_fn import check_esm_for_solver
        check_esm_for_solver(force_fn, "ThermalConductivityCalculator",
                             refuse="ESM has no site-energy decomposition for the "
                                    "heat flux, and slab kappa is out of scope "
                                    "(ADR-0008)")
        from ..fitting.mlip_engine import check_mlip_for_solver
        check_mlip_for_solver(force_fn, "ThermalConductivityCalculator",
                              need_site_energies=True,
                              species=set(self.atom_types))

        if masses is not None:
            self.masses = np.asarray(masses, dtype=float)
        else:
            self.masses = np.array(
                [ATOMIC_MASS.get(a, 1.0) for a in atom_types])

        # Results populated by run_md / compute_kappa
        self._J_series   : Optional[np.ndarray] = None  # (n_frames, 3)
        self._dt_hf      : Optional[float]      = None  # heat-flux timestep [fs]
        self._hcacf      : Optional[np.ndarray] = None  # (n_lags, 3, 3)
        self._tau_axis   : Optional[np.ndarray] = None  # [fs]
        self._kappa      : Optional[np.ndarray] = None  # (3, 3) [W/(m.K)]
        self._kappa_run  : Optional[np.ndarray] = None  # running kappa(t) [W/(m.K)]

        # S4: emit the "falling back to the approximate virial method"
        # warning at most once per instance, not once per MD step.
        self._warned_virial_fallback = False

    # ------------------------------------------------------------------ #
    #  Heat-flux computation (pair-potential / ELM)                       #
    # ------------------------------------------------------------------ #

    def _warn_virial_fallback_once(self, reason: str) -> None:
        if not self._warned_virial_fallback:
            warnings.warn(
                f"ThermalConductivityCalculator: {reason}",
                UserWarning, stacklevel=3)
            self._warned_virial_fallback = True

    def _gather_pair_data(self, positions: np.ndarray):
        """
        Site-energy decomposition (ADR-0007) of every term of ``force_fn``
        that has one, combined: pair potentials (each pair split equally),
        Stillinger-Weber, Tersoff, EAM, and an atom-centred MLIP.  See
        :mod:`akitsushima.engine.site_terms`.

        Returns ``None`` when nothing decomposes (an ``ExternalFileEngine``,
        a tabular MLIP -- which ``run_md`` refuses up front -- or an engine
        of Coulomb terms only).  Whatever is left undecomposed (Coulomb) is
        found by subtraction in :meth:`_pair_virial_flux`.

        Computed once per heat-flux recording step and shared between
        :meth:`_pair_virial_flux` and :meth:`_per_atom_energy`.  Every rank
        holds the whole decomposition (no reduction needed).
        """
        from ..engine.site_terms import SiteTerms
        engine = getattr(self.force_fn, "engine", None)
        if engine is None:
            fn_site = getattr(self.force_fn, "site_terms", None)
            if fn_site is None or not getattr(self.force_fn, "per_atom_energy", True):
                return None
            return fn_site(self.lattice, positions, self.atom_types)
        parts = [pot.site_terms(self.lattice, positions, self.atom_types)
                 for pot in engine._potentials if hasattr(pot, "site_terms")]
        if not parts:
            return None
        return SiteTerms.combine(parts, self.N)

    def _compute_heat_flux(self,
                           positions : np.ndarray,
                           velocities: np.ndarray,
                           forces    : np.ndarray,
                           stress    : np.ndarray,
                           e_pot_i   : np.ndarray,
                           pair_data : Optional[list],
                           ) -> np.ndarray:
        """
        Compute the instantaneous heat-flux vector J [eV/(A^2.fs)].

        Uses the Hardy / Schelling formula:

        .. math::

            \\mathbf{J} = \\frac{1}{V} \\left[
                \\sum_i E_i \\mathbf{v}_i
                + \\frac{1}{2} \\sum_{i<j}
                  (\\mathbf{f}_{ij} \\cdot (\\mathbf{v}_i + \\mathbf{v}_j))
                  \\mathbf{r}_{ij}
            \\right]

        The convective term ``sum_i E_i v_i`` uses the per-atom energies
        from :meth:`_per_atom_energy`. The pair-virial term is computed
        exactly for pair-decomposable potentials by :meth:`_pair_virial_flux`
        (see its docstring for the many-body/Coulomb fallback).

        Parameters
        ----------
        positions  : (N,3) [A].
        velocities : (N,3) [A/fs].
        forces     : (N,3) [eV/A] (unused directly here; kept for
                     interface symmetry with the per-atom energy call).
        stress     : (3,3) [eV/A^3], from the same force_fn call that
                     produced ``forces`` (avoids a redundant force_fn
                     evaluation inside _pair_virial_flux).
        e_pot_i    : (N,)  per-atom potential energy [eV].
        pair_data  : output of :meth:`_gather_pair_data`.

        Returns
        -------
        J : (3,) [eV/(A^2.fs)]
        """
        V   = abs(np.linalg.det(self.lattice))
        # Bug-fix note: masses are in amu and velocities in A/fs, so
        # 0.5 m v^2 is in amu A^2/fs^2 = 103.6 eV; it was used as eV, leaving
        # the kinetic part of the convective term 103.6x too small.
        e_kin_i = (0.5 * self.masses * np.sum(velocities ** 2, axis=1)
                   / EV_ANG_TO_AMU_ANG_FS2)                           # [eV]
        E_i     = e_kin_i + e_pot_i    # total energy per atom [eV]

        J_conv   = np.einsum('i,ix->x', E_i, velocities)  # (3,)
        J_virial = self._pair_virial_flux(positions, velocities, stress, pair_data)

        J = (J_conv + J_virial) / V
        return J

    def _pair_virial_flux(self,
                          positions : np.ndarray,
                          velocities: np.ndarray,
                          stress_full: np.ndarray,
                          pair_data : Optional[list],
                          ) -> np.ndarray:
        """
        Compute the per-pair virial heat-flux contribution.

        .. math::

            J_{\\rm vir,\\alpha} = \\frac{1}{2}\\sum_{i<j}
                (\\mathbf{f}_{ij}\\cdot(\\mathbf{v}_i+\\mathbf{v}_j))\\,
                r_{ij,\\alpha}

        Bug-fix note (S4): the original approximated this ENTIRE term
        from the aggregate virial stress tensor and the mean (centre-of-
        mass) velocity alone::

            J_vir ~= -(V/2) sigma @ v_mean

        In a momentum-conserving NVE run (this class always removes net
        linear momentum from the initial velocities -- see run_md()),
        ``v_mean`` is ~0 throughout the trajectory, so this term -- the
        dominant contribution to kappa in a solid, where atoms barely
        move but strongly interact -- nearly vanished, systematically
        underestimating kappa. The aggregate stress tensor alone cannot
        give the exact per-pair result either way: it has already summed
        every pair's f_ij (x) r_ij contribution into a single 3x3 matrix,
        discarding the per-pair velocity-weighting the exact formula
        needs.

        Fix (S4), generalised to many-body potentials (ADR-0007): the
        virial term is evaluated per neighbour entry from the site-energy
        decomposition of :meth:`_gather_pair_data`,
        ``J_vir = -sum_m v_m (dU_centre/dv_m . vel_nbr)`` (Fan et al., Phys.
        Rev. B 92, 094301 (2015), eq. 21).  For pair potentials it reduces
        to the per-pair formula above; for Stillinger-Weber, Tersoff, EAM
        and an atom-centred MLIP it is exact too, where the S4 version
        handed their whole contribution to the centre-of-mass method.
        What still has no decomposition (a Coulomb term) is isolated by
        subtraction (``stress_full - stress_site``) and keeps the
        approximate centre-of-mass method, with a one-time warning.

        Returns
        -------
        J_vir : (3,) [eV.A/fs]  (not yet divided by V; see caller)
        """
        V = abs(np.linalg.det(self.lattice))

        if pair_data is None:
            self._warn_virial_fallback_once(
                "force_fn has no site-energy decomposition (e.g. an "
                "ExternalFileEngine); the heat-flux virial term uses the "
                "approximate centre-of-mass method for its ENTIRE "
                "contribution, which is ~0 in a momentum-conserving NVE "
                "run and will underestimate kappa for solids.")
            v_mean = velocities.mean(axis=0)
            return -V * stress_full @ v_mean * 0.5

        if isinstance(pair_data, list):     # parts, e.g. [] = none
            from ..engine.site_terms import SiteTerms
            pair_data = SiteTerms.combine(pair_data, self.N)
        J_vir_exact = pair_data.virial_flux(velocities)
        stress_pair = pair_data.stress(V)

        stress_remainder = stress_full - stress_pair
        scale = max(float(np.max(np.abs(stress_full))), 1e-12)
        if float(np.max(np.abs(stress_remainder))) > 1e-6 * scale:
            self._warn_virial_fallback_once(
                "the engine's stress has a component without a site-energy "
                "decomposition (a Coulomb term); that remainder's share of "
                "the heat-flux virial term uses the approximate "
                "centre-of-mass method, which underestimates its "
                "contribution to kappa.")
        v_mean = velocities.mean(axis=0)
        J_vir_remainder = -V * stress_remainder @ v_mean * 0.5

        return J_vir_exact + J_vir_remainder

    # ------------------------------------------------------------------ #
    #  Per-atom potential energy                                           #
    # ------------------------------------------------------------------ #

    def _per_atom_energy(self,
                         positions: np.ndarray,
                         forces   : np.ndarray,
                         E_tot    : float,
                         pair_data: Optional[list],
                         ) -> np.ndarray:
        """
        Estimate per-atom potential energies.

        Bug-fix note (S4): the original always returned a spatially
        uniform ``E_tot / N`` split. Multiplying a spatially-uniform
        per-atom value by v_i and summing over i (the convective term
        ``sum_i E_i v_i``) is proportional to the total momentum alone,
        which is ~0 by construction in the momentum-conserving NVE runs
        this class performs -- so a uniform split silently discards any
        genuine correlation between an atom's local energy environment
        and its velocity, a real physical contribution to the heat flux
        that an accurate per-atom decomposition should retain.

        Fix: the site energies U_i of :meth:`_gather_pair_data` (pair
        energies split equally; the many-body potentials' and an
        atom-centred MLIP's own site energies, ADR-0007).  Energy with no
        decomposition (Coulomb) is still split uniformly -- the same scope
        as :meth:`_pair_virial_flux`.

        Parameters
        ----------
        positions : (N,3) [A].
        forces    : (N,3) [eV/A] (unused; kept for interface symmetry).
        E_tot     : float [eV].
        pair_data : output of :meth:`_gather_pair_data`.

        Returns
        -------
        e_pot_i : (N,) [eV]  per-atom potential energy.
        """
        if pair_data is None:
            return np.full(self.N, E_tot / self.N)

        if isinstance(pair_data, list):
            from ..engine.site_terms import SiteTerms
            pair_data = SiteTerms.combine(pair_data, self.N)
        e_remainder = E_tot - pair_data.energy
        return pair_data.e_site + e_remainder / self.N

    # ------------------------------------------------------------------ #
    #  1. Run MD with heat-flux recording                                  #
    # ------------------------------------------------------------------ #

    def run_md(self,
               n_steps      : int   = 100_000,
               timestep     : float = 0.5,
               equil_steps  : int   = 10_000,
               velocities   : Optional[np.ndarray] = None,
               heat_flux_every : int = 5,
               traj_every   : int   = 100,
               thermo_every : int   = 1000,
               output_dir   : str   = ".",
               seed         : int   = 42,
               verbose      : bool  = True,
               ) -> dict:
        """
        Run NVE-MD and record the heat flux J at every ``heat_flux_every``
        steps.

        Equilibration (``equil_steps``) is done with NVT (Nose-Hoover
        chain, single chain) to reach the target temperature, then
        switched to NVE for the production run.

        Parameters
        ----------
        n_steps         : production MD steps.
        timestep        : [fs].  0.5 fs is recommended for ionic systems.
        equil_steps     : NVT equilibration steps (not recorded).
        velocities      : (N,3) initial velocities [A/fs].
                          If None, sampled from Maxwell-Boltzmann at
                          ``self.temperature``.
        heat_flux_every : record J every this many steps.
        traj_every      : write XYZ trajectory every this many steps.
        thermo_every    : write thermo log every this many steps.
        output_dir      : directory for output files.
        seed            : RNG seed for the initial Maxwell-Boltzmann
                          velocity draw (only used when ``velocities`` is
                          None). Bug-fix note (S11): this was previously
                          hard-coded to 42 with no way to change it
                          (needed e.g. to run independent repeats for
                          error estimation).
        verbose         : print progress.

        Returns
        -------
        dict with ``J_series`` (n_frames, 3), ``dt_hf`` [fs],
        ``positions``, ``velocities``.
        """
        import time as _time
        os.makedirs(output_dir, exist_ok=True)

        N = self.N
        masses = self.masses         # [amu]
        T = self.temperature

        # ---- Initial velocities ----------------------------------------
        if velocities is not None:
            vel = np.asarray(velocities, dtype=float).copy()
        else:
            # Maxwell-Boltzmann velocities at target temperature
            rng_mb = np.random.default_rng(seed)
            sigma  = np.sqrt(K_B * T * EV_ANG_TO_AMU_ANG_FS2 / self.masses)
            vel    = rng_mb.standard_normal((N, 3)) * sigma[:, None]
            # Remove net linear momentum
            vel   -= (np.sum(self.masses[:, None] * vel, axis=0)
                      / np.sum(self.masses))
            # Rescale to exact temperature
            e_kin = 0.5 * np.sum(self.masses * np.sum(vel**2, axis=1))
            dof   = 3 * N - 3
            T_act = 2.0 * e_kin / (dof * K_B * EV_ANG_TO_AMU_ANG_FS2)
            if T_act > 1e-12:
                vel *= np.sqrt(T / T_act)

        pos = self.positions.copy()
        lat = self.lattice.copy()

        # ---- NVT equilibration -----------------------------------------
        if equil_steps > 0 and verbose:
            print(f"\n  ThermalConductivityCalculator: NVT equilibration "
                  f"({equil_steps} steps, T={T:.0f} K) ...")

        if equil_steps > 0:
            # Bug-fix note (S17, newly found during the port):
            # NVTSimulation.__init__ has never accepted a `velocities=`
            # keyword argument (checked against both the original
            # monolithic file and this port) -- passing one here raised
            # TypeError immediately, meaning run_md() crashed on its very
            # first call with the default equil_steps=10_000. Removed;
            # NVTSimulation draws its own Maxwell-Boltzmann velocities
            # internally (at the same target temperature T), which
            # equilibration will converge from regardless of the specific
            # initial draw -- the pre-equilibration `vel` constructed
            # above is only actually used as-is when equil_steps=0.
            nvt = NVTSimulation(
                lattice     = lat,
                positions   = pos,
                atom_types  = self.atom_types,
                force_fn    = self.force_fn,
                timestep    = timestep,
                n_steps     = equil_steps,
                temperature = T,
                # Bug-fix note: without seed= the equilibration drew its
                # velocities from NVTSimulation's default seed (42), so
                # every `seed` gave the same trajectory -- five
                # "independent" runs were one run five times.
                seed        = seed,
                traj_every  = max(equil_steps, 1),   # suppress trajectory
                thermo_every= max(equil_steps // 10, 1),
                output_dir  = os.path.join(output_dir, "nvt_equil"),
                verbose     = False,
            )
            nvt_res = nvt.run()
            pos = nvt_res["positions"]
            vel = nvt_res["velocities"]

        # ---- NVE production --------------------------------------------
        if verbose:
            print(f"  NVE production: {n_steps} steps, dt={timestep} fs, "
                  f"T={T:.0f} K")
            print(f"  Recording J every {heat_flux_every} steps "
                  f"-> {n_steps // heat_flux_every} frames")

        V_cell = abs(np.linalg.det(lat))
        inv_lat = np.linalg.inv(lat)
        dt = timestep
        acc_factor = EV_ANG_TO_AMU_ANG_FS2 / masses[:, None]   # [A/fs^2] per [eV/A]

        # Initial forces
        E_pot, forces, _ = self.force_fn(lat, pos, self.atom_types)
        acc = forces * acc_factor

        J_list      : List[np.ndarray] = []
        thermo_path = os.path.join(output_dir, "thermo_nve.log")
        hf_path     = os.path.join(output_dir, "heat_flux.dat")
        traj_path   = os.path.join(output_dir, "trajectory_tc.xyz")

        GPa = EV_ANG3_TO_GPA
        hdr = (f"{'Step':>8}  {'T(K)':>9}  {'E_pot(eV)':>14}  "
               f"{'E_tot(eV)':>14}  {'P(GPa)':>9}")
        e_kin_ref = (0.5 * np.sum(masses * np.sum(vel**2, axis=1))
                     / EV_ANG_TO_AMU_ANG_FS2)
        e_tot_ref = float(E_pot) + e_kin_ref
        t0 = _time.perf_counter()

        with (open(thermo_path, 'w') as t_fh,
              open(hf_path,    'w') as j_fh,
              open(traj_path,  'w') as x_fh):

            t_fh.write(hdr + "\n")
            j_fh.write("step  Jx_eV_Ang2_fs  Jy  Jz\n")

            for step in range(n_steps):
                # Velocity Verlet
                vel  += 0.5 * dt * acc
                pos  += dt * vel

                # PBC wrap
                frac  = pos @ inv_lat
                # `frac -= np.floor(frac)` alone maps a tiny negative frac to
                # exactly 1.0 in float64, which scipy's cKDTree rejects; see
                # wrap_positions_into_cell.
                frac -= np.floor(frac)
                frac[frac >= 1.0] = 0.0
                pos   = frac @ lat

                E_pot, forces, stress = self.force_fn(lat, pos, self.atom_types)
                acc   = forces * acc_factor
                vel  += 0.5 * dt * acc

                # Heat flux recording
                if step % heat_flux_every == 0:
                    pair_data = self._gather_pair_data(pos)
                    e_pot_i = self._per_atom_energy(pos, forces, float(E_pot), pair_data)
                    J = self._compute_heat_flux(pos, vel, forces, stress, e_pot_i, pair_data)
                    J_list.append(J)
                    j_fh.write(f"{step}  {J[0]:.8e}  {J[1]:.8e}  {J[2]:.8e}\n")

                # Thermo
                if step % thermo_every == 0:
                    # Bug-fix note: e_kin was amu A^2/fs^2 read as eV (the
                    # log showed 0.48 K for a 50 K run), and P carried the
                    # tensile-positive sign; the engine's stress is
                    # compressive-positive (ADR-0001), so P = +tr(sigma)/3.
                    e_kin = (0.5 * np.sum(masses * np.sum(vel**2, axis=1))
                             / EV_ANG_TO_AMU_ANG_FS2)
                    T_inst= e_kin * 2.0 / (3.0 * N * K_B)
                    e_tot = float(E_pot) + e_kin
                    P_gpa = np.trace(stress) / 3.0 * GPa
                    row = (f"{step:8d}  {T_inst:9.2f}  {float(E_pot):14.6f}  "
                           f"{e_tot:14.6f}  {P_gpa:9.4f}")
                    t_fh.write(row + "\n")
                    if verbose and step % (thermo_every * 10) == 0:
                        drift = e_tot - e_tot_ref
                        print(f"  step={step:7d}  T={T_inst:7.1f} K  "
                              f"drift={drift:+.4e} eV")

                # Trajectory
                if step % traj_every == 0:
                    x_fh.write(f"{N}\n")
                    a, b, c = lat[0], lat[1], lat[2]
                    x_fh.write(
                        f"Lattice=\"{a[0]:.6f} {a[1]:.6f} {a[2]:.6f} "
                        f"{b[0]:.6f} {b[1]:.6f} {b[2]:.6f} "
                        f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\" "
                        f"Properties=species:S:1:pos:R:3 "
                        f"step={step}\n")
                    for ia, (sp, p) in enumerate(zip(self.atom_types, pos)):
                        x_fh.write(f"{sp}  {p[0]:.8f}  {p[1]:.8f}  {p[2]:.8f}\n")

        J_series = np.array(J_list)   # (n_frames, 3)
        self._J_series = J_series
        self._dt_hf    = float(timestep * heat_flux_every)

        elapsed = _time.perf_counter() - t0
        if verbose:
            print(f"  NVE finished  ({elapsed:.1f} s)  "
                  f"n_frames={len(J_series)}  "
                  f"J_rms={float(np.sqrt(np.mean(J_series**2))):.4e} eV/(A^2.fs)")
            print(f"  Heat flux  -> {hf_path}")
            print(f"  Thermo     -> {thermo_path}")

        return {
            "J_series"  : J_series,
            "dt_hf"     : self._dt_hf,
            "positions" : pos,
            "velocities": vel,
        }

    # ------------------------------------------------------------------ #
    #  2. Green-Kubo: HCACF -> kappa tensor                                   #
    # ------------------------------------------------------------------ #

    def compute_kappa(self,
                      J_series   : Optional[np.ndarray] = None,
                      dt_hf      : Optional[float]      = None,
                      tau_max    : float  = 20.0,
                      n_blocks   : int    = 5,
                      volume     : Optional[float] = None,
                      verbose    : bool   = True,
                      ) -> np.ndarray:
        """
        Compute the thermal conductivity tensor from the heat-flux ACF.

        Green-Kubo relation:

        .. math::

            \\kappa_{\\alpha\\beta} = \\frac{V}{k_B T^2}
            \\int_0^{\\tau_{\\rm max}} C_{\\alpha\\beta}(\\tau)\\,d\\tau

        where :math:`C_{\\alpha\\beta}(\\tau) = \\langle J_\\alpha(t)
        J_\\beta(t+\\tau) \\rangle`.

        Block averaging is used to estimate the statistical uncertainty.

        Parameters
        ----------
        J_series : (n_frames, 3) [eV/(A^2.fs)].  Defaults to ``self._J_series``.
        dt_hf    : time step between J frames [fs].  Defaults to
                   ``self._dt_hf``.
        tau_max  : upper limit of ACF integration [ps].
        n_blocks : number of blocks for uncertainty estimation.
        volume   : cell volume [A^3].  Defaults to det(self.lattice).
        verbose  : print summary.

        Returns
        -------
        kappa : (3,3) array [W/(m.K)].
        """
        if J_series is None:
            if self._J_series is None:
                raise RuntimeError(
                    "compute_kappa: no J_series available. "
                    "Call run_md() first or supply J_series=.")
            J_series = self._J_series
        if dt_hf is None:
            dt_hf = self._dt_hf if self._dt_hf else 1.0

        J    = np.asarray(J_series, dtype=float)
        n    = len(J)
        dt   = float(dt_hf)          # fs
        T    = self.temperature
        V    = float(volume) if volume is not None \
               else abs(np.linalg.det(self.lattice))

        # Maximum lag index
        n_lag = min(int(tau_max * 1e3 / dt), n // 2)   # tau_max [ps] -> [fs]
        tau_ax = np.arange(n_lag) * dt * 1e-3           # [ps]

        # HCACF via FFT (Wiener-Khinchin theorem)  shape: (n_lag, 3, 3)
        hcacf = np.zeros((n_lag, 3, 3))
        for a in range(3):
            for b in range(3):
                c = _autocorr_fft(J[:, a], J[:, b], n_lag)
                hcacf[:, a, b] = c

        # kappa from cumulative integral (running kappa)
        # kappa(t) = (V / k_B T^2) int_0^tau C(tau') dtau'
        pref = V / (K_B * T**2)           # [A^3 x K / eV]
        # Trapezoid rule: sum in [fs], multiply by pref
        kappa_run = np.zeros((n_lag, 3, 3))
        for tau_i in range(1, n_lag):
            kappa_run[tau_i] = kappa_run[tau_i - 1] + \
                0.5 * dt * (hcacf[tau_i - 1] + hcacf[tau_i])
        kappa_run *= pref  # [eV/(A.K.fs)]

        # Final kappa at tau_max
        kappa_evafs = kappa_run[-1]              # [eV/(A.K.fs)]
        kappa_WmK   = kappa_evafs * _EV_ANG_K_FS_TO_W_MK

        # Block-averaged uncertainty
        #
        # Bug-fix note (S18, newly found during the port): block_size was
        # computed as max(n // n_blocks, n_lag) -- i.e. floored at n_lag so
        # every block would be long enough to support the full lag range
        # -- but the slicing below then used the UNFLOORED n // n_blocks
        # directly, so block_size's computed (and correctly floored) value
        # was never actually applied. Whenever n_lag > n // n_blocks (the
        # common case: it already happens for this class's own docstring
        # example, n_steps=50_000/heat_flux_every=5/tau_max=10.0 with the
        # n_blocks=5 default), each J_blk ended up SHORTER than n_lag, and
        # _autocorr_fft(..., n_lag) silently returns an array only as long
        # as its (shorter) input, one which is then too short to store into
        # the (n_lag, 3, 3)-shaped hb -- raising a ValueError deep inside
        # NumPy's broadcasting on every call with these (in practice, quite
        # ordinary) parameter combinations.
        #
        # Fix: actually use block_size (properly floored at n_lag) for the
        # slicing, and derive n_blocks_eff = n // block_size -- the number
        # of full-sized, non-overlapping blocks that actually fit in a
        # trajectory of length n. If fewer than 2 blocks fit (tau_max is
        # too large relative to the trajectory length and n_blocks), block
        # averaging cannot give a meaningful uncertainty estimate; kappa
        # itself (already computed above from the FULL series) is still
        # returned, but kappa_std is reported as NaN with a warning rather
        # than silently crashing or fabricating a number from too little
        # data.
        block_size = max(n // n_blocks, n_lag)
        n_blocks_eff = n // block_size if block_size > 0 else 0

        if n_blocks_eff < 2:
            warnings.warn(
                "compute_kappa: trajectory too short relative to tau_max "
                f"and n_blocks for block-averaged uncertainty estimation "
                f"(need >= 2 blocks of >= n_lag={n_lag} samples each out "
                f"of n={n} total; got {n_blocks_eff}). kappa itself is "
                f"still computed from the full series; kappa_std is set "
                f"to NaN. Reduce tau_max or n_blocks, or lengthen the MD "
                f"run, for a real uncertainty estimate.",
                UserWarning, stacklevel=2)
            kappa_std = np.full((3, 3), np.nan)
        else:
            kappa_blocks = np.zeros((n_blocks_eff, 3, 3))
            for ib in range(n_blocks_eff):
                start = ib * block_size
                end   = start + block_size
                J_blk = J[start:end]
                hb    = np.zeros((n_lag, 3, 3))
                for a in range(3):
                    for bb in range(3):
                        hb[:, a, bb] = _autocorr_fft(J_blk[:, a], J_blk[:, bb], n_lag)
                kr = np.zeros((3, 3))
                for tau_i in range(1, n_lag):
                    kr += 0.5 * dt * (hb[tau_i - 1] + hb[tau_i])
                kappa_blocks[ib] = kr * pref * _EV_ANG_K_FS_TO_W_MK

            kappa_std = np.std(kappa_blocks, axis=0) / np.sqrt(n_blocks_eff)

        self._hcacf      = hcacf
        self._tau_axis   = tau_ax
        self._kappa      = kappa_WmK
        self._kappa_run  = kappa_run * _EV_ANG_K_FS_TO_W_MK
        self._kappa_std  = kappa_std

        if verbose:
            kxx = kappa_WmK[0, 0]; kyy = kappa_WmK[1, 1]; kzz = kappa_WmK[2, 2]
            kavg = (kxx + kyy + kzz) / 3.0
            print(f"\n  Green-Kubo ... (T={T:.0f} K, tau_max={tau_max:.1f} ps)")
            print(f"    k_xx = {kxx:.2f} +/- {kappa_std[0,0]:.2f} W/(m.K)")
            print(f"    k_yy = {kyy:.2f} +/- {kappa_std[1,1]:.2f} W/(m.K)")
            print(f"    k_zz = {kzz:.2f} +/- {kappa_std[2,2]:.2f} W/(m.K)")
            print(f"    k_avg = {kavg:.2f} W/(m.K)")

        return kappa_WmK

    # ------------------------------------------------------------------ #
    #  3. Slack-Debye approximation                                       #
    # ------------------------------------------------------------------ #

    def slack_debye(self,
                    phonon_calc  : PhononCalculator,
                    T            : Optional[float]  = None,
                    gruneisen    : Optional[float]  = None,
                    use_vsound   : bool = False,
                    verbose      : bool = True,
                    ) -> float:
        """
        Estimate the isotropic thermal conductivity from the Slack-Debye
        formula.

        .. math::

            \\kappa = A \\frac{\\bar{M} \\theta_D^3 V_0^{1/3}}
                           {n^{2/3} \\gamma^2 T}

        where :math:`A = 2.43\\times10^{-8}/(1 - 0.514/\\gamma + 0.228/\\gamma^2)`
        [SI units].

        Parameters
        ----------
        phonon_calc : PhononCalculator with IFC already computed.
        T           : temperature [K].  Defaults to ``self.temperature``.
        gruneisen   : Gruneisen parameter gamma.  Default 1.7.
        use_vsound  : when True, derive theta_D from acoustic sound velocities
                      via
                      :meth:`PhononCalculator.debye_temperature_from_vsound`
                      instead of the DOS cutoff frequency.  More accurate
                      for materials with strongly dispersive acoustic
                      branches.  Default False (backward-compatible).
        verbose     : print result.

        Returns
        -------
        kappa_slack : float [W/(m.K)]
        """
        if T is None:
            T = self.temperature

        if phonon_calc.IFC is None:
            raise RuntimeError(
                "slack_debye: PhononCalculator IFC not computed. "
                "Call phonon_calc.compute_ifc() first.")

        # Debye temperature
        if use_vsound:
            theta_D = phonon_calc.debye_temperature_from_vsound(verbose=verbose)
        else:
            theta_D = self._debye_temperature(phonon_calc)

        # Gruneisen parameter (use 1.7 as default if not supplied)
        gamma = float(gruneisen) if gruneisen is not None else 1.7

        # Mean atomic mass [kg]
        # For multi-species systems the Slack-Debye formula uses the
        # average mass per atom  Mbar = (1/N) sum_i m_i  (arithmetic mean),
        # which is the standard convention in the literature
        # (Morelli & Slack, "High Lattice Thermal Conductivity Solids",
        #  in "High Thermal Conductivity Materials", Springer 2006).
        # The arithmetic mean already accounts for the stoichiometry because
        # self.masses contains one entry per atom in the simulation cell.
        # For a compound A_p B_q the arithmetic mean gives:
        #   Mbar = (p m_A + q m_B) / (p + q)
        # which is correct.  The geometric mean or reduced mass would be
        # appropriate only for a two-atom Einstein model, not the full
        # Slack-Debye formula.
        M_avg_amu = float(np.mean(self.masses))   # arithmetic mean over all atoms
        M_avg_kg  = M_avg_amu * 1.66053906660e-27

        # Volume per atom [m^3]
        V0 = abs(np.linalg.det(self.lattice)) * 1e-30   # A^3 -> m^3
        V0_per_atom = V0 / self.N

        # Number of atoms per primitive cell
        n = phonon_calc.N_basis

        # Slack formula (Slack 1979, Julian 1994)
        A_slack = 2.43e-8 / (1.0 - 0.514 / gamma + 0.228 / gamma**2)
        kappa = (A_slack * M_avg_kg * theta_D**3 * V0_per_atom**(1/3)
                 / (n**(2/3) * gamma**2 * T))

        if verbose:
            print(f"\n  Slack-Debye ... (T={T:.0f} K)")
            print(f"    theta_D      = {theta_D:.1f} K")
            print(f"    gamma        = {gamma:.2f}")
            print(f"    M_avg    = {M_avg_amu:.2f} amu")
            print(f"    k_slack = {kappa:.2f} W/(m.K)")

        return float(kappa)

    def _debye_temperature(self,
                           phonon_calc: PhononCalculator) -> float:
        """
        Estimate the Debye temperature from the phonon DOS.

        The Debye temperature is defined as the temperature corresponding
        to the maximum phonon frequency (acoustic cut-off):

        .. math::

            \\theta_D = \\hbar \\omega_{\\rm max} / k_B
                     = h \\nu_{\\rm max} / k_B

        where ``omega_max`` is the maximum real frequency in the DOS.
        """
        # Use the computed DOS if available; else compute from Gamma-point Hessian
        if phonon_calc._dos_result is not None:
            freqs  = phonon_calc._dos_result["frequencies"]
            dos    = phonon_calc._dos_result["dos"]
            # Maximum frequency with non-negligible DOS
            mask   = dos > 0.01 * dos.max()
            nu_max = float(np.max(freqs[mask])) if mask.any() else float(np.max(freqs))
        elif phonon_calc._band_result is not None:
            nu_max = float(np.max(phonon_calc._band_result["frequencies"]))
        else:
            # Compute Gamma-point frequencies as a rough proxy
            D_G  = phonon_calc.build_dynamical_matrix(np.zeros(3))
            eigs = np.linalg.eigvalsh(D_G.real)
            nu_max = float(np.max(PhononCalculator._eig_to_freq(np.sort(eigs))))

        # Convert nu [cm^-1] -> theta_D [K]
        # nu [cm^-1] x c [cm/s] x h [J.s] / k_B [J/K]
        c_cm = 2.99792458e10   # cm/s
        h_J  = 6.62607015e-34  # J.s
        kB_J = 1.380649e-23    # J/K
        theta_D = abs(nu_max) * c_cm * h_J / kB_J
        return float(theta_D)

    # ------------------------------------------------------------------ #
    #  4. Save results                                                     #
    # ------------------------------------------------------------------ #

    def save_results(self,
                     output_dir : str  = ".",
                     verbose    : bool = True) -> None:
        """
        Save HCACF, running kappa(t), and final kappa tensor to CSV files.

        Files written
        -------------
        hcacf.csv              -- tau [ps], C_xx, C_yy, C_zz, C_xy, ...
        kappa_running.csv      -- tau [ps], kappa_xx(t), kappa_yy(t), kappa_zz(t) [W/(m.K)]
        kappa_tensor.csv       -- 3x3 kappa tensor [W/(m.K)]
        hcacf.png              -- HCACF plot (if matplotlib available)
        kappa_running.png      -- running kappa plot (if matplotlib available)
        """
        os.makedirs(output_dir, exist_ok=True)

        if self._hcacf is not None and self._tau_axis is not None:
            # HCACF CSV
            c = self._hcacf   # (n_lag, 3, 3)
            path = os.path.join(output_dir, "hcacf.csv")
            with open(path, 'w') as fh:
                fh.write("tau_ps,Cxx,Cyy,Czz,Cxy,Cxz,Cyz\n")
                for i, tau in enumerate(self._tau_axis):
                    fh.write(f"{tau:.6f},{c[i,0,0]:.8e},{c[i,1,1]:.8e},"
                              f"{c[i,2,2]:.8e},{c[i,0,1]:.8e},"
                              f"{c[i,0,2]:.8e},{c[i,1,2]:.8e}\n")
            if verbose: print(f"  HCACF      -> {path}")

        if self._kappa_run is not None:
            path = os.path.join(output_dir, "kappa_running.csv")
            with open(path, 'w') as fh:
                fh.write("tau_ps,kappa_xx,kappa_yy,kappa_zz,kappa_avg  # W/(m.K)\n")
                for i, tau in enumerate(self._tau_axis):
                    kr = self._kappa_run[i]
                    kavg = (kr[0,0]+kr[1,1]+kr[2,2])/3
                    fh.write(f"{tau:.6f},{kr[0,0]:.4f},{kr[1,1]:.4f},"
                              f"{kr[2,2]:.4f},{kavg:.4f}\n")
            if verbose: print(f"  Running ... -> {path}")

        if self._kappa is not None:
            path = os.path.join(output_dir, "kappa_tensor.csv")
            with open(path, 'w') as fh:
                fh.write("# Thermal conductivity tensor [W/(m.K)]\n")
                fh.write("alpha,beta,kappa_WmK,kappa_std_WmK\n")
                for a in range(3):
                    for b in range(3):
                        std = self._kappa_std[a, b] if hasattr(self, '_kappa_std') \
                              else 0.0
                        fh.write(f"{a},{b},{self._kappa[a,b]:.4f},{std:.4f}\n")
            if verbose: print(f"  ... tensor -> {path}")

        if HAS_MATPLOTLIB:
            self._plot_hcacf(output_dir)
            self._plot_kappa_running(output_dir)

    def _plot_hcacf(self, output_dir: str) -> None:
        if self._hcacf is None: return
        fig, ax = plt.subplots(figsize=(7, 4))
        tau = self._tau_axis
        for a, label in [(0,'xx'),(1,'yy'),(2,'zz')]:
            c0 = self._hcacf[0, a, a] or 1e-30
            ax.plot(tau, self._hcacf[:, a, a] / c0, label=f'C_{label}', lw=1.5)
        ax.axhline(0, color='grey', lw=0.8, ls='--')
        ax.set_xlabel("tau (ps)"); ax.set_ylabel("C(tau)/C(0)")
        ax.set_title("Heat-flux ACAF"); ax.legend(); fig.tight_layout()
        path = os.path.join(output_dir, "hcacf.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  HCACF plot -> {path}")

    def _plot_kappa_running(self, output_dir: str) -> None:
        if self._kappa_run is None: return
        fig, ax = plt.subplots(figsize=(7, 4))
        tau = self._tau_axis
        for a, label in [(0,'xx'),(1,'yy'),(2,'zz')]:
            ax.plot(tau, self._kappa_run[:, a, a], label=f'sum_{label}', lw=1.5)
        avg = (self._kappa_run[:,0,0]+self._kappa_run[:,1,1]+self._kappa_run[:,2,2])/3
        ax.plot(tau, avg, 'k--', lw=2, label='kappa_avg')
        ax.axhline(0, color='grey', lw=0.8, ls=':')
        ax.set_xlabel("tau (ps)"); ax.set_ylabel("kappa (W/(m.K))")
        ax.set_title(f"Running ... (T={self.temperature:.0f} K)")
        ax.legend(); fig.tight_layout()
        path = os.path.join(output_dir, "kappa_running.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  Running ... plot -> {path}")


# ---------------------------------------------------------------------------
#  Helper: FFT-based normalized autocorrelation
# ---------------------------------------------------------------------------

def _autocorr_fft(a: np.ndarray,
                  b: np.ndarray,
                  n_lag: int,
                  subtract_mean: bool = True) -> np.ndarray:
    """
    Compute the unnormalised cross-correlation <a(0) b(tau)> for lags
    0..n_lag-1 using the Wiener-Khinchin theorem (FFT).

    By default (``subtract_mean=True``), returns the mean-subtracted,
    ensemble-averaged correlation:

        C(tau) = <(a(t) - <a>) (b(t+tau) - <b>)>

    which is the right quantity for a series (e.g. the heat flux, this
    helper's original use in ``compute_kappa``) whose TRUE population
    mean is not analytically known in advance and might carry a small
    residual, non-physical numerical offset that should not leak into
    the reported correlation/its long-time integral.

    Parameters
    ----------
    a, b   : (N,) time series.
    n_lag  : number of lag values to return.
    subtract_mean : whether to subtract each series' own SAMPLE mean
                   before correlating (default True, the original
                   behaviour). Pass False for a series whose population
                   mean is analytically zero by symmetry (e.g. a
                   velocity component in an equilibrium system with no
                   net drift) -- subtracting a merely-finite-sample
                   estimate of that already-known-zero mean does not
                   remove any genuine bias, and instead introduces one:
                   confirmed directly (while adding
                   ``DiffusionAnalyzer.compute_vacf_dos``) that
                   VACF(0), computed with ``subtract_mean=True`` on a
                   short (n_frames ~ 800) real NVE trajectory, reads
                   systematically LOW relative to the equipartition
                   prediction k_B T / m across every one of 5 random
                   seeds tested (ratios 0.78-0.92), while the raw
                   (non-mean-subtracted) <v^2> on the very same
                   trajectories matches equipartition much more closely
                   (ratios 0.93-1.09) -- exactly the sign and magnitude
                   expected from spuriously subtracting a nonzero
                   finite-window SAMPLE mean of a quantity whose true
                   population mean is zero.

    Returns
    -------
    corr : (n_lag,) float  (units = units_a x units_b)
    """
    n   = len(a)
    if subtract_mean:
        a = a - a.mean()
        b = b - b.mean()
    # Zero-pad to power-of-2 length for efficiency
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    fa = np.fft.rfft(a, n=nfft)
    fb = np.fft.rfft(b, n=nfft)
    corr = np.fft.irfft(fa * np.conj(fb), n=nfft)[:n]
    # Normalise by number of samples contributing to each lag
    norm = (n - np.arange(n)).astype(float)
    corr = corr / norm
    return corr[:n_lag]
