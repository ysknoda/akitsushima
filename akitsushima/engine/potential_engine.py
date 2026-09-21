"""Unified multi-potential engine and top-level energy/force/stress driver."""
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np

from .coulomb.ewald import ewald_sum
from .coulomb.pppm import PPPMSolver, _pppm_alpha_from_accuracy, _pppm_mesh_from_alpha
from .neighborlist import NeighbourList

# Small LRU cache of PPPMSolver instances, keyed by (mesh, order, alpha, lattice
# bytes). Bug-fix note: an earlier version used a plain dict that called
# .clear() on every cache miss (making it a size-1 cache in practice), which
# thrashed to zero benefit whenever called with varying lattices -- e.g. every
# step of variable-cell relaxation/NPT MD, or every distinct structure in a
# fitting loop over many training structures. This was also, textually, a
# reverse dependency: it lived in the file's fitting section while being
# read/written from this engine-side function. A small LRU (capacity a
# handful of entries, not unbounded -- PPPMSolver holds an M^3 influence
# function array, so unbounded growth would leak memory) fixes both issues.
_PPPM_CACHE_CAPACITY = 8
_pppm_cache: "OrderedDict[tuple, PPPMSolver]" = OrderedDict()


def _pppm_cache_get_or_build(cache_key: tuple, lattice: np.ndarray,
                              alpha: float, mesh: int, order: int,
                              r_cut: float) -> PPPMSolver:
    """Fetch a cached PPPMSolver for cache_key, building (and evicting the
    least-recently-used entry if at capacity) on a miss."""
    if cache_key in _pppm_cache:
        _pppm_cache.move_to_end(cache_key)
        return _pppm_cache[cache_key]
    solver = PPPMSolver(lattice, alpha, mesh=mesh, order=order, r_cut=r_cut)
    _pppm_cache[cache_key] = solver
    if len(_pppm_cache) > _PPPM_CACHE_CAPACITY:
        _pppm_cache.popitem(last=False)
    return solver


class PotentialEngine:
    """
    Unified engine that sums contributions from any number of potentials.

    Usage
    -----
    engine = PotentialEngine()
    engine.add(LennardJonesPotential(params, r_cut=10.0))
    engine.add(StillingerWeberPotential(sw_params))
    engine.add(EAMPotential(...))

    energy, forces, stress = engine.compute(
        lattice, positions, atom_types,
        neigh_list=nl, use_numba=False)

    The Coulomb long-range part is added separately by
    :func:`compute_all_general` when charges are non-zero.

    Parameters
    ----------
    potentials : list of BasePairPotential or many-body potential instances.
    """

    def __init__(self):
        self._potentials: list = []

    def add(self, potential) -> 'PotentialEngine':
        """Register a potential with this engine. Returns self for chaining."""
        self._potentials.append(potential)
        return self

    @property
    def r_cut(self) -> float:
        """Maximum cut-off across all registered potentials."""
        return max((getattr(p, 'r_cut', 0.0) for p in self._potentials), default=0.0)

    def compute(self,
                lattice: np.ndarray,
                positions: np.ndarray,
                atom_types: list,
                neigh_list: Optional[NeighbourList] = None,
                use_numba: bool = False,
                ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Sum energy, forces, and stress from all registered potentials.

        Returns
        -------
        energy : float        [eV]
        forces : (N,3) array  [eV/A]
        stress : (3,3) array  [eV/A^3]
        """
        N = len(positions)
        energy = 0.0
        forces = np.zeros((N, 3))
        stress = np.zeros((3, 3))

        for pot in self._potentials:
            e, f, s = pot.compute(
                lattice, positions, atom_types,
                neigh_list=neigh_list, use_numba=use_numba)
            energy += e
            forces += f
            stress += s

        return energy, forces, stress


def compute_all_general(
        lattice: np.ndarray,
        positions: np.ndarray,
        atom_types: list,
        engine: PotentialEngine,
        charges: Optional[dict] = None,
        r_cut: Optional[float] = None,
        ewald_alpha: Optional[float] = None,
        coulomb_method: str = "ewald",
        pppm_accuracy: float = 1e-5,
        pppm_mesh: Optional[int] = None,
        pppm_order: int = 6,
        neigh_list: Optional[NeighbourList] = None,
        use_numba: bool = False,
        ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Unified energy / force / stress driver for any combination of potentials.

    Parameters
    ----------
    engine     : PotentialEngine holding any combination of potentials.
    charges    : dict  symbol -> charge [e].  None or all-zero -> no Coulomb.
    r_cut      : Ewald/PPPM real-space cutoff (defaults to engine.r_cut).

    Returns
    -------
    energy : float        [eV]
    forces : (N,3) array  [eV/A]
    stress : (3,3) array  [eV/A^3]
    """
    # Short-range: sum over all registered potentials
    e_sr, f_sr, s_sr = engine.compute(
        lattice, positions, atom_types,
        neigh_list=neigh_list, use_numba=use_numba)

    # Long-range Coulomb (only when charges are present)
    rc = r_cut if r_cut is not None else engine.r_cut
    if charges is not None:
        q = np.array([charges.get(a, 0.0) for a in atom_types])
        if np.any(np.abs(q) > 1e-10):
            if coulomb_method == "pppm":
                alpha = (_pppm_alpha_from_accuracy(rc, pppm_accuracy)
                         if ewald_alpha is None else ewald_alpha)
                mesh = (_pppm_mesh_from_alpha(lattice, alpha, pppm_order,
                                              pppm_accuracy)
                        if pppm_mesh is None else pppm_mesh)
                # Bug-fix note (stale r_cut on a cache hit): the key used
                # to omit ``rc``, even though PPPMSolver takes r_cut as a
                # constructor argument and uses it for BOTH its real-space
                # erfc sum and its minimum-image precondition check.  With
                # the default (``ewald_alpha=None``) alpha is derived from
                # rc, so rc changes were tracked indirectly through the
                # alpha component; but with an EXPLICIT ``ewald_alpha`` --
                # e.g. a real-space cutoff convergence study at fixed Ewald
                # splitting -- nothing in the key varied with rc, so a
                # second call at a different r_cut silently got back the
                # solver built with the FIRST one.  Measured on a 1024-atom
                # MgO cell at alpha=0.35: warming the cache at r_cut=5.0
                # and then asking for r_cut=7.0 returned the r_cut=5.0
                # energy bit-for-bit, an error of 730 eV out of 12271 eV.
                # Worse, the cache hit also bypassed update_lattice()'s
                # ``r_cut < min_i(d_i)/2`` validation entirely: an r_cut
                # that a fresh solver correctly REJECTS with ValueError was
                # accepted and silently used.  rc is rounded like alpha so
                # float noise cannot fragment the cache.
                cache_key = (mesh, pppm_order, round(alpha, 8),
                             round(rc, 8), lattice.tobytes())
                solver = _pppm_cache_get_or_build(
                    cache_key, lattice, alpha, mesh, pppm_order, rc)
                if not np.allclose(solver.lattice, lattice, atol=1e-10):
                    solver.update_lattice(lattice)
                e_c, f_c, s_c = solver.compute(positions, q,
                                                neigh_list=neigh_list)
            else:
                alpha = (5.0 / rc if ewald_alpha is None else ewald_alpha)
                e_c, f_c, s_c = ewald_sum(
                    lattice, positions, q, alpha=alpha, r_cut=rc)
            return e_sr + e_c, f_sr + f_c, s_sr + s_c

    return e_sr, f_sr, s_sr
