"""Embedded Atom Method (EAM) potential for metallic systems.

Reference: Daw & Baskes, Phys. Rev. B 29, 6443 (1984).
"""
from typing import Optional, Tuple

import numpy as np

from ..utils.mpi import (
    _HAS_MPI,
    _mpi_allreduce_sum,
    _mpi_allreduce_sum_scalar,
    _mpi_domain_split,
)
from .neighborlist import NeighbourList, _check_min_image


class EAMPotential:
    """
    Embedded Atom Method (EAM) potential for metallic systems.

    Reference: Daw & Baskes, Phys. Rev. B 29, 6443 (1984).

    Energy
    ------
    E = sum_i F_i(rho_bar_i) + (1/2) sum_{i!=j} phi(r_ij)

    rho_bar_i = sum_{j!=i} rho(r_ij)   (electron density at atom i)

    Functions
    ---------
    F(rho)   : embedding function [eV]  --- energy cost of embedding atom in density rho
    rho(r)   : electron density function --- contribution from atom at distance r
    phi(r)   : pair potential [eV]       --- core-core repulsion

    All three functions are provided as 1-D NumPy arrays (tabulated) or as
    callable Python functions.  If arrays are provided, cubic spline
    interpolation is used automatically.

    Parameters
    ----------
    F_table   : (n_rho,) array or callable -- embedding energy vs rho
    rho_table : (n_r,) array or callable  -- electron density vs r
    phi_table : (n_r,) array or callable  -- pair potential vs r
    rho_grid  : (n_rho,) array -- rho grid for F_table (if array provided)
    r_grid    : (n_r,) array  -- r grid for rho_table and phi_table
    r_cut     : cut-off [A]
    species   : list of species symbols (single-element default)

    Notes
    -----
    The EAM potential is inherently many-body due to the embedding term.
    Forces require two passes over the neighbour list:
      Pass 1: compute rho_bar_i for all atoms and F'(rho_bar_i).
      Pass 2: compute pair forces and embedding-force contributions.
    """

    def __init__(self,
                 F_table: object,
                 rho_table: object,
                 phi_table: object,
                 rho_grid: Optional[np.ndarray] = None,
                 r_grid: Optional[np.ndarray] = None,
                 r_cut: float = 6.0,
                 species: Optional[list] = None):
        """
        Parameters (multi-species alloy EAM format)
        ---------------------------------------------
        F_table   : dict {species: callable/array/(f, df)} -- embedding
                    function per element; a (f, df) tuple of callables
                    supplies the exact derivative (the same holds for
                    rho_table and phi_table).
        rho_table : dict {species: callable/array} -- electron-density
                    function per element (the density AT atom i is the
                    sum of rho_j(r_ij) over neighbours j, using the
                    NEIGHBOUR's species function -- standard DYNAMO/
                    setfl alloy-EAM convention).
        phi_table : dict {frozenset({sp1,sp2}): callable/array} -- pair
                    repulsion function per species pair.

        Legacy single-species format (backward compatible)
        -----------------------------------------------------
        F_table, rho_table, phi_table given directly as callables or
        arrays (not dicts) -- automatically wrapped so every species in
        `species` (or a single dummy species) maps to the same function.
        """
        from scipy.interpolate import CubicSpline

        self.r_cut = r_cut
        self.species = list(species) if species else None

        def _make_interp(tab, grid):
            # (f, df): a callable with its exact derivative (analytic
            # forms such as Zhou et al. 2004, ADR-0015); a bare callable
            # is differentiated by central differences in _eval.
            if (isinstance(tab, tuple) and len(tab) == 2
                    and callable(tab[0]) and callable(tab[1])):
                return tab[0], tab[1]
            if callable(tab):
                return tab, None
            cs = CubicSpline(grid, tab)
            return cs, cs.derivative()

        is_legacy = not isinstance(F_table, dict)
        if is_legacy:
            sp0 = (self.species[0] if self.species else '__X__')
            self.species = [sp0]
            F_i, dF_i = _make_interp(F_table, rho_grid)
            rho_i, drho_i = _make_interp(rho_table, r_grid)
            phi_ij, dphi_ij = _make_interp(phi_table, r_grid)
            self._F = {sp0: (F_i, dF_i)}
            self._rho = {sp0: (rho_i, drho_i)}
            self._phi = {frozenset({sp0, sp0}): (phi_ij, dphi_ij)}
        else:
            if not self.species:
                raise ValueError(
                    "EAMPotential: 'species' is required for the "
                    "multi-species (dict) parameter format.")
            self._F = {sp: _make_interp(F_table[sp], rho_grid)
                       for sp in F_table}
            self._rho = {sp: _make_interp(rho_table[sp], r_grid)
                         for sp in rho_table}
            self._phi = {pair: _make_interp(fn, r_grid)
                         for pair, fn in phi_table.items()}

    def _eval(self, func, dfunc, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate function and its derivative at points x."""
        if dfunc is None:
            # Callable pair: use finite difference for derivative
            h = 1e-5 * np.maximum(np.abs(x), 1e-8)
            return func(x), (func(x + h) - func(x - h)) / (2.0 * h)
        return func(x), dfunc(x)

    def site_terms(self, lattice, positions, atom_types):
        """Per-atom energies and per-neighbour gradients (ADR-0007); see
        akitsushima.engine.site_terms."""
        from .site_terms import eam_site_terms
        return eam_site_terms(self, np.asarray(lattice, float),
                   np.asarray(positions, float), list(atom_types))

    def compute(self,
                lattice: np.ndarray,
                positions: np.ndarray,
                atom_types: list,
                neigh_list: Optional[NeighbourList] = None,
                use_numba: bool = False,
                ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute (multi-species alloy) EAM energy, forces, and virial stress.

        E = sum_i F_i(rho_bar_i) + 0.5 * sum_{i!=j} phi_ij(r_ij)
        rho_bar_i = sum_{j!=i} rho_j(r_ij)   (density contributed by the
                                              NEIGHBOUR's species function
                                              -- standard alloy-EAM form)

        Two-pass algorithm, grouped by species for vectorised evaluation:
          Pass 1: rho_bar_i for all atoms.
          Pass 2: pair (phi) forces + embedding-force cross terms.
        """
        N = len(positions)
        volume = float(np.abs(np.linalg.det(lattice)))
        inv_lat = np.linalg.inv(lattice)
        sp = list(atom_types)
        sp_arr = np.array(sp)

        if neigh_list is not None:
            # Candidates only: get_pairs_arrays' build-time ``<= r_cut``
            # mask permanently dropped skin-resident pairs that later
            # migrated inside r_cut.  This routine already recomputes its
            # own distances below, so it needs the unmasked candidate set.
            # See NeighbourList.get_pairs_arrays' bug-fix note.
            idx_i_raw, idx_j_raw = neigh_list.get_candidate_pairs()
        else:
            # The O(N^2) fallback never goes through NeighbourList.build(),
            # which is where the minimum-image precondition is enforced;
            # past half the cell width `s -= np.round(s)` keeps one image
            # per pair and silently drops the rest (BasePairPotential).
            _check_min_image(lattice, self.r_cut, "EAMPotential.compute")
            ii_all, jj_all = np.triu_indices(N, k=1)
            idx_i_raw = ii_all.astype(np.int32)
            idx_j_raw = jj_all.astype(np.int32)

        if len(idx_i_raw) > 0:
            dr = positions[idx_j_raw] - positions[idx_i_raw]
            s = dr @ inv_lat
            s -= np.round(s)
            rij_all = s @ lattice
            r_all = np.linalg.norm(rij_all, axis=1)
            mask = (r_all < self.r_cut) & (r_all > 1e-12)
            idx_i = idx_i_raw[mask].astype(np.int64)
            idx_j = idx_j_raw[mask].astype(np.int64)
            rij_arr = rij_all[mask]
            r_arr = r_all[mask]
        else:
            idx_i = np.empty(0, np.int64)
            idx_j = np.empty(0, np.int64)
            rij_arr = np.empty((0, 3))
            r_arr = np.empty(0)

        if len(idx_i) == 0:
            return 0.0, np.zeros((N, 3)), np.zeros((3, 3))

        P = len(idx_i)
        p_start, p_end = _mpi_domain_split(P)
        r_loc = r_arr[p_start:p_end]
        rij_loc = rij_arr[p_start:p_end]
        ii_loc = idx_i[p_start:p_end]
        jj_loc = idx_j[p_start:p_end]
        sp_i_loc = sp_arr[ii_loc]
        sp_j_loc = sp_arr[jj_loc]

        # ---- Pass 1: electron density rho_bar_i = sum_j rho_j(r_ij) ----
        # Density AT atom i comes from the NEIGHBOUR's rho function, so
        # rho_i_loc uses sp_j_loc's rho function (contribution TO atom i
        # FROM atom j uses rho_j), and rho_j_loc uses sp_i_loc's function
        # (contribution TO atom j FROM atom i uses rho_i).
        rho_bar = np.zeros(N)
        drho_from_j = np.zeros(P)  # d(rho_j(r))/dr, indexed like r_arr (global)
        drho_from_i = np.zeros(P)
        for spk, (rho_f, drho_f) in self._rho.items():
            mask_j_is_k = (sp_j_loc == spk)
            if np.any(mask_j_is_k):
                rvals, dvals = self._eval(rho_f, drho_f, r_loc[mask_j_is_k])
                np.add.at(rho_bar, ii_loc[mask_j_is_k], rvals)  # rho_j contributes to i
                idxs = np.where(mask_j_is_k)[0]
                drho_from_j[p_start:p_end][idxs] = dvals
            mask_i_is_k = (sp_i_loc == spk)
            if np.any(mask_i_is_k):
                rvals, dvals = self._eval(rho_f, drho_f, r_loc[mask_i_is_k])
                np.add.at(rho_bar, jj_loc[mask_i_is_k], rvals)  # rho_i contributes to j
                idxs = np.where(mask_i_is_k)[0]
                drho_from_i[p_start:p_end][idxs] = dvals

        if _HAS_MPI:
            rho_bar = _mpi_allreduce_sum(rho_bar)

        rho_safe = np.maximum(rho_bar, 0.0)
        F_vals = np.zeros(N)
        dFdp = np.zeros(N)
        for spk, (F_f, dF_f) in self._F.items():
            mask_sp = (sp_arr == spk)
            if np.any(mask_sp):
                fv, dfv = self._eval(F_f, dF_f, rho_safe[mask_sp])
                F_vals[mask_sp] = fv
                dFdp[mask_sp] = dfv

        # ---- Pass 2: pair (phi) energy/forces + embedding cross terms --
        forces = np.zeros((N, 3))
        stress = np.zeros((3, 3))
        phi_r = np.zeros(len(r_loc))
        dphi_r = np.zeros(len(r_loc))
        for pair_key, (phi_f, dphi_f) in self._phi.items():
            pair_mask = np.array([frozenset({a, b}) == pair_key
                                   for a, b in zip(sp_i_loc, sp_j_loc)])
            if np.any(pair_mask):
                pv, dpv = self._eval(phi_f, dphi_f, r_loc[pair_mask])
                phi_r[pair_mask] = pv
                dphi_r[pair_mask] = dpv

        # Bug-fix note (E11, newly found via Phase 6's real mpirun testing
        # -- never exercised by ordinary pytest runs, where _HAS_MPI is
        # always False): energy used to be computed as
        # `F_vals.sum() + phi_r.sum()` BEFORE the Allreduce below. But
        # F_vals (the embedding energy) is derived from rho_bar, which
        # was already Allreduce'd above (line ~210) -- so F_vals.sum()
        # is already the correct, GLOBAL total on every single rank,
        # identically. phi_r, by contrast, is sized to len(r_loc) (this
        # rank's LOCAL share of pairs only) and only becomes a global
        # total once Allreduce'd. Summing an already-global quantity
        # (F_vals.sum()) with a merely-local one (phi_r.sum()) and then
        # Allreduce-summing THAT combined value silently multiplied the
        # embedding energy by the number of MPI ranks (e.g. exactly 2x
        # under -np 2, 4x under -np 4) while leaving the pair term
        # correct -- confirmed by comparing -np 1/2/4 results directly
        # (test_mpi.py). Fixed by keeping the local phi-only sum
        # separate through the Allreduce and adding the (already-global)
        # F_vals.sum() back in exactly once, afterwards.
        energy_phi_local = float(phi_r.sum())
        # coeff = d(phi_ij)/dr + dF_i/drho_i * d(rho_j)/dr + dF_j/drho_j * d(rho_i)/dr
        coeff = (dphi_r
                 + dFdp[ii_loc] * drho_from_j[p_start:p_end]
                 + dFdp[jj_loc] * drho_from_i[p_start:p_end])
        fij = (coeff / r_loc)[:, None] * rij_loc
        np.add.at(forces, ii_loc, fij)
        np.add.at(forces, jj_loc, -fij)
        stress -= np.einsum('ki,kj->ij', fij, rij_loc)

        # Bug-fix note (found in code-check, same class as F16): the
        # embedding-energy reduction below used to be a raw
        # `_mpi_comm.allreduce(energy_phi_local, op=_MPI.SUM)` call, which
        # bypasses `local_only()` (it never checks `_mpi_is_active()`, unlike
        # `forces`/`stress` below, which already go through the
        # `local_only()`-aware `_mpi_allreduce_sum()`). Under real
        # multi-rank `mpirun`, an outer item-splitting caller such as
        # `fitting/loss.py::generic_loss_function` or `solvers/neb.py`
        # wrapped in `with local_only():` could not suppress it, so each
        # rank's own-structure/image energy silently got summed together
        # with every OTHER rank's DIFFERENT structure/image energy. Fixed
        # by routing through the same `_mpi_allreduce_sum_scalar()` helper
        # `pair_potentials.py` and `coulomb/ewald.py` already use for their
        # own scalar energy.
        if _HAS_MPI:
            energy_phi = _mpi_allreduce_sum_scalar(energy_phi_local)
            forces = _mpi_allreduce_sum(forces)
            stress = _mpi_allreduce_sum(stress)
        else:
            energy_phi = energy_phi_local

        energy = float(F_vals.sum()) + energy_phi
        stress /= volume
        return energy, forces, stress
