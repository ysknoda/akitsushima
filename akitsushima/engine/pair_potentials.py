"""Simple 2-body (pair) potentials: Lennard-Jones, Morse, Born-Mayer, Buckingham."""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from ..utils.jit import _HAS_NUMBA, _NUMBA_ENV
from ..utils.mpi import _mpi_allreduce_sum, _mpi_allreduce_sum_scalar, _mpi_domain_split
from .neighborlist import NeighbourList
from ._pair_kernels import _pair_body_jit, _pair_body_numpy


class BasePairPotential(ABC):
    """
    Abstract base class for all 2-body (pair) potentials.

    Subclasses must implement ``pair_energy_force(r)`` which returns
    ``(V(r), dV/dr)`` as scalars or NumPy arrays. Everything else
    (neighbour-list dispatch, Numba JIT, MPI domain split, virial stress)
    is handled by the ``compute()`` method inherited from this class.

    Parameters
    ----------
    r_cut   : interaction cut-off [A].
    species : list of species symbols this potential acts on.
              If None, acts on all species pairs.
    """

    def __init__(self, r_cut: float, species: Optional[list] = None):
        self.r_cut = r_cut
        self.species = species   # None -> apply to every pair

    @abstractmethod
    def pair_energy_force(self,
                           r: np.ndarray,
                           sp_i: Optional[str] = None,
                           sp_j: Optional[str] = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate pair potential and its derivative.

        Parameters
        ----------
        r   : (P,) distances [A].
        sp_i, sp_j : species labels (needed for heterogeneous potentials).

        Returns
        -------
        V    : (P,) energies [eV].
        dVdr : (P,) dV/dr    [eV/A].
        """
        ...

    def _get_params(self, sp_i: str, sp_j: str) -> Optional[dict]:
        """
        Return this potential's parameter dict for the (sp_i, sp_j) pair,
        or None if the pair is not parameterised.

        Bug-fix note: previously only ``pair_energy_force`` was marked
        ``@abstractmethod``, while every concrete subclass (LJ/Morse/
        BornMayer) independently implemented an identical
        ``self.params.get(frozenset({sp_i, sp_j}))`` lookup. A future
        subclass implementing only the documented abstract method would
        have instantiated successfully but raised ``AttributeError`` on
        its first ``compute()`` call. Providing the (identical, in every
        existing subclass) default implementation here removes both the
        duplication and that failure mode; a subclass with a genuinely
        different parameter-storage scheme can still override it.
        """
        return self.params.get(frozenset({sp_i, sp_j}))

    def _applies(self, sp_i: str, sp_j: str) -> bool:
        """Return True if this potential acts on the (sp_i, sp_j) pair."""
        if self.species is None:
            return True
        return sp_i in self.species and sp_j in self.species

    def _build_pair_groups(self,
                           lattice: np.ndarray,
                           positions: np.ndarray,
                           atom_types: list,
                           neigh_list: Optional[NeighbourList] = None):
        """
        Build (or retrieve) the full pair list and group it by species
        type-pair. Shared by :meth:`compute` and :meth:`compute_pairs` so
        the two stay exactly consistent (same pair list, same grouping).

        Returns
        -------
        idx_i, idx_j : (P,) int32   atom indices (i < j within a periodic
                       image; ``idx_i``/``idx_j`` need not be globally
                       sorted).
        rij_all      : (P,3)  r_j - r_i, minimum-image convention [A].
        r_all        : (P,)   |rij| [A].
        groups       : dict {(sp1,sp2) sorted: [pair-index,...]}.
        """
        N = len(positions)

        # Bug-fix note (stale-distance / defeated-skin defect): this used
        # to call ``neigh_list.get_pairs_arrays()`` and consume the rij/r
        # it cached at the last rebuild.  With the default skin_width > 0
        # the list is deliberately NOT rebuilt every step, so those cached
        # values describe the geometry as it was at the last rebuild
        # rather than as it is now; and the ``<= r_cut`` mask baked into
        # that accessor permanently discarded exactly the skin-resident
        # pairs the Verlet skin exists to preserve.  Candidates are now
        # taken as indices only, and the geometry is recomputed from the
        # CURRENT positions -- the same treatment the neigh_list-free
        # branch always applied, and the same idiom eam.py /
        # stillinger_weber.py / tersoff.py already used.  See
        # NeighbourList.get_pairs_arrays' own bug-fix note.
        if neigh_list is not None:
            ii, jj = neigh_list.get_candidate_pairs()
        else:
            ii, jj = np.triu_indices(N, k=1)

        # Minimum-image displacements/distances at the CURRENT positions,
        # then this potential's own cutoff mask.  Shared by both branches
        # so a neighbour-list-backed evaluation and the O(N^2) fallback
        # can never disagree about geometry or cutoff.
        inv_lat = np.linalg.inv(lattice)
        dr = positions[jj] - positions[ii]           # (P, 3)
        s = dr @ inv_lat
        s -= np.round(s)
        dr = s @ lattice                              # (P, 3) MIC
        r2 = np.sum(dr**2, axis=1)
        mask = (r2 < self.r_cut**2) & (r2 > 1e-20)
        idx_i = ii[mask].astype(np.int32)
        idx_j = jj[mask].astype(np.int32)
        rij_all = dr[mask]
        r_all = np.sqrt(r2[mask])

        if len(idx_i) == 0:
            return idx_i, idx_j, rij_all, r_all, {}

        # Build a type-pair key per pair: frozenset({si, sj})
        # We sort type strings so (A,B) and (B,A) map to the same key.
        types_arr = np.array(atom_types)
        si_arr = types_arr[idx_i]
        sj_arr = types_arr[idx_j]

        # Unique type-pair keys: use sorted (si,sj) tuple as dict key
        type_pair_key = [
            (s1, s2) if s1 <= s2 else (s2, s1)
            for s1, s2 in zip(si_arr, sj_arr)
        ]
        # Group indices by key
        groups: dict = {}
        for k, key in enumerate(type_pair_key):
            groups.setdefault(key, []).append(k)

        return idx_i, idx_j, rij_all, r_all, groups

    def compute(self,
                lattice: np.ndarray,
                positions: np.ndarray,
                atom_types: list,
                neigh_list: Optional[NeighbourList] = None,
                use_numba: bool = False,
                ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute energy, forces, and virial stress for this 2-body potential.

        Uses the neighbour list when provided (O(N)); falls back to
        minimum-image O(N^2) otherwise.

        Pairs of the same species type are batched into contiguous NumPy
        arrays so that ``pair_energy_force`` is called once per type-pair
        (passing a vector of distances) instead of once per atom pair.
        Forces and stress are accumulated with ``np.add.at`` to avoid
        temporary Python loops over pairs. When running under MPI (>1
        rank), each type-pair's own pair list is further split across
        ranks before an ``Allreduce`` combines the partial sums -- with a
        single rank this reduces to exactly the serial algorithm.

        Returns
        -------
        energy : float        [eV]
        forces : (N,3) array  [eV/A]
        stress : (3,3) array  [eV/A^3]
        """
        N = len(positions)
        volume = np.abs(np.linalg.det(lattice))
        forces = np.zeros((N, 3))
        stress = np.zeros((3, 3))
        energy = 0.0

        idx_i, idx_j, rij_all, r_all, groups = self._build_pair_groups(
            lattice, positions, atom_types, neigh_list)
        if len(idx_i) == 0:
            return energy, forces, stress

        # ---- Vectorised evaluation per type-pair -------------------------
        use_jit = (use_numba or _NUMBA_ENV) and _HAS_NUMBA
        for (s1, s2), grp_idx in groups.items():
            if not self._applies(s1, s2):
                continue
            if self._get_params(s1, s2) is None:
                continue

            grp = np.array(grp_idx, dtype=np.int64)
            p_start, p_end = _mpi_domain_split(len(grp))
            grp = grp[p_start:p_end]
            if len(grp) == 0:
                continue
            r_g = r_all[grp]                    # (M,) distances
            V_g, dVdr_g = self.pair_energy_force(r_g, s1, s2)  # (M,) each

            if use_jit:
                # Numba JIT scatter: _pair_body_jit uses prange internally
                e_g = _pair_body_jit(
                    idx_i[grp].astype(np.int32),
                    idx_j[grp].astype(np.int32),
                    rij_all[grp], r_g,
                    forces, stress,
                    V_g.astype(np.float64), dVdr_g.astype(np.float64))
                energy += float(e_g)
            else:
                energy += float(V_g.sum())

                # F_i = +dV/dr * rij/r, rij = r_j - r_i (see NeighbourList's
                # own docstring for this convention, validated throughout
                # the engine package against finite differences)
                fij = (dVdr_g / r_g)[:, None] * rij_all[grp]

                # Scatter forces (Newton 3rd law)
                np.add.at(forces, idx_i[grp], fij)
                np.add.at(forces, idx_j[grp], -fij)

                # Virial stress
                stress -= np.einsum('ki,kj->ij', fij, rij_all[grp])

        energy = _mpi_allreduce_sum_scalar(energy)
        forces = _mpi_allreduce_sum(forces)
        stress = _mpi_allreduce_sum(stress)
        stress /= volume
        return energy, forces, stress

    def compute_pairs(self,
                      lattice: np.ndarray,
                      positions: np.ndarray,
                      atom_types: list,
                      neigh_list: Optional[NeighbourList] = None,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray]:
        """
        Return this potential's contribution as raw per-pair arrays,
        rather than the aggregated (energy, forces, stress) of
        :meth:`compute`.

        Needed by :class:`~akitsushima.solvers.thermal_conductivity.
        ThermalConductivityCalculator` for the exact Hardy/Schelling
        pair-virial heat flux (plan item S4) and a per-atom energy
        decomposition finer than a uniform ``E_tot / N`` split -- neither
        is recoverable from the aggregated stress tensor alone, since it
        has already summed over every pair and lost the per-pair
        velocity-weighting information the heat-flux formula needs.

        Uses the exact same pair list and grouping as :meth:`compute`
        (via :meth:`_build_pair_groups`), so the two are always
        consistent. Not JIT-compiled: this is only called at the
        (typically much less frequent) heat-flux recording interval, not
        every MD step, so the existing vectorised-NumPy evaluation used
        by ``compute()``'s own non-JIT path is already fast enough here.

        MPI: like ``compute()``, each type-pair's pairs are further split
        across ranks via :func:`_mpi_domain_split`, but -- unlike
        ``compute()`` -- the raw per-pair arrays returned here are NOT
        Allreduced (concatenating differently-sized per-rank pair lists
        is not a summable quantity). Callers that reduce these pairs down
        to a summed quantity (e.g. a total heat-flux vector) must
        Allreduce-sum that FINAL result themselves once, exactly as
        ``compute()`` does for energy/forces/stress.

        Returns
        -------
        idx_i, idx_j : (P,) int32   atom indices (this rank's share only).
        rij          : (P,3)  r_j - r_i, minimum-image convention [A].
        fij          : (P,3)  force ON atom i FROM atom j [eV/A] (same
                       sign convention as compute()'s F_i accumulation).
        V_pair       : (P,)   pair energy [eV] (undivided; the caller
                       decides how to split it between the two atoms).
        """
        idx_i, idx_j, rij_all, r_all, groups = self._build_pair_groups(
            lattice, positions, atom_types, neigh_list)
        if len(idx_i) == 0:
            empty_i3 = np.zeros((0, 3))
            return (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32),
                    empty_i3, empty_i3, np.zeros(0))

        idx_i_out, idx_j_out, rij_out, fij_out, v_out = [], [], [], [], []
        for (s1, s2), grp_idx in groups.items():
            if not self._applies(s1, s2):
                continue
            if self._get_params(s1, s2) is None:
                continue

            grp = np.array(grp_idx, dtype=np.int64)
            p_start, p_end = _mpi_domain_split(len(grp))
            grp = grp[p_start:p_end]
            if len(grp) == 0:
                continue
            r_g = r_all[grp]
            V_g, dVdr_g = self.pair_energy_force(r_g, s1, s2)
            fij_g = (dVdr_g / r_g)[:, None] * rij_all[grp]

            idx_i_out.append(idx_i[grp])
            idx_j_out.append(idx_j[grp])
            rij_out.append(rij_all[grp])
            fij_out.append(fij_g)
            v_out.append(V_g)

        if not idx_i_out:
            empty_i3 = np.zeros((0, 3))
            return (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32),
                    empty_i3, empty_i3, np.zeros(0))

        return (np.concatenate(idx_i_out), np.concatenate(idx_j_out),
                np.concatenate(rij_out), np.concatenate(fij_out),
                np.concatenate(v_out))


class LennardJonesPotential(BasePairPotential):
    """
    Lennard-Jones 12-6 potential with optional smooth shift to zero at r_cut.

    V(r) = 4epsilon [(sigma/r)^12 - (sigma/r)^6]

    With  shift=True:
        V_shifted(r) = V(r) - V(r_cut) - V'(r_cut)(r - r_cut)
    ensuring both V and dV/dr are continuous at r_cut.

    Parameters
    ----------
    params  : dict  frozenset({sp1, sp2}) -> {'epsilon': epsilon[eV], 'sigma': sigma[A]}
    r_cut   : cut-off [A]
    shift   : apply force-shifted truncation (default True)
    """

    def __init__(self,
                 params: dict,
                 r_cut: float = 10.0,
                 shift: bool = True):
        species = sorted({s for pair in params for s in pair})
        super().__init__(r_cut=r_cut, species=species)
        self.params = params
        self.shift = shift
        self._cache_shift: dict = {}  # (sp1,sp2) -> (V_cut, dV_cut)

    def _shift_values(self, sp_i: str, sp_j: str) -> Tuple[float, float]:
        """Return (V(r_cut), dV/dr(r_cut)) for the force-shifted correction."""
        key = (sp_i, sp_j) if sp_i <= sp_j else (sp_j, sp_i)
        if key not in self._cache_shift:
            p = self._get_params(sp_i, sp_j)
            eps, sig = p['epsilon'], p['sigma']
            sr6 = (sig / self.r_cut)**6
            Vc = 4.0 * eps * (sr6**2 - sr6)
            dVc = 4.0 * eps * (-12.0 * sig**12 / self.r_cut**13
                                + 6.0 * sig**6 / self.r_cut**7)
            self._cache_shift[key] = (Vc, dVc)
        return self._cache_shift[key]

    def pair_energy_force(self,
                           r: np.ndarray,
                           sp_i: Optional[str] = None,
                           sp_j: Optional[str] = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
        p = self._get_params(sp_i, sp_j)
        eps, sig = p['epsilon'], p['sigma']

        sr6 = (sig / r)**6
        V = 4.0 * eps * (sr6**2 - sr6)
        dVdr = 4.0 * eps * (-12.0 * sig**12 / r**13
                             + 6.0 * sig**6 / r**7)

        if self.shift:
            Vc, dVc = self._shift_values(sp_i, sp_j)
            V = V - Vc - dVc * (r - self.r_cut)
            dVdr = dVdr - dVc

        return V, dVdr


class MorsePotential(BasePairPotential):
    """
    Morse potential with smooth force-shifted cut-off.

    V(r) = D_e [1 - exp(-alpha(r - r_0))]^2  - D_e

    The  -D_e  term sets V(r_0) = 0 and V(inf) = 0
    (well depth = D_e at r = r_0).

    Parameters
    ----------
    params : dict  frozenset({sp1, sp2}) ->
             {'D_e': D_e[eV], 'alpha': alpha[A^-1], 'r0': r0[A]}
    r_cut  : cut-off [A]
    shift  : apply force-shifted truncation (default True)
    """

    def __init__(self,
                 params: dict,
                 r_cut: float = 10.0,
                 shift: bool = True):
        species = sorted({s for pair in params for s in pair})
        super().__init__(r_cut=r_cut, species=species)
        self.params = params
        self.shift = shift
        self._cache_shift: dict = {}

    def _shift_values(self, sp_i, sp_j):
        key = (sp_i, sp_j) if sp_i <= sp_j else (sp_j, sp_i)
        if key not in self._cache_shift:
            p = self._get_params(sp_i, sp_j)
            De, alp, r0 = p['D_e'], p['alpha'], p['r0']
            rc = self.r_cut
            ex = np.exp(-alp * (rc - r0))
            Vc = De * (1.0 - ex)**2 - De
            dVc = 2.0 * De * alp * ex * (1.0 - ex)
            self._cache_shift[key] = (Vc, dVc)
        return self._cache_shift[key]

    def pair_energy_force(self, r, sp_i=None, sp_j=None):
        p = self._get_params(sp_i, sp_j)
        De, alp, r0 = p['D_e'], p['alpha'], p['r0']
        ex = np.exp(-alp * (r - r0))
        V = De * (1.0 - ex)**2 - De
        dVdr = 2.0 * De * alp * ex * (1.0 - ex)
        if self.shift:
            Vc, dVc = self._shift_values(sp_i, sp_j)
            V = V - Vc - dVc * (r - self.r_cut)
            dVdr = dVdr - dVc
        return V, dVdr


class BornMayerPotential(BasePairPotential):
    """
    Born-Mayer short-range repulsion.

    V(r) = A . exp(-r / rho)

    Typically combined with Coulomb (Ewald/PPPM) to form the full
    Born-Mayer-Huggins ionic model:
        V_total = V_BM(r) + q_i q_j / r

    Parameters
    ----------
    params : dict  frozenset({sp1, sp2}) -> {'A': A[eV], 'rho': rho[A]}
    r_cut  : cut-off [A]
    """

    def __init__(self, params: dict, r_cut: float = 10.0):
        species = sorted({s for pair in params for s in pair})
        super().__init__(r_cut=r_cut, species=species)
        self.params = params

    def pair_energy_force(self, r, sp_i=None, sp_j=None):
        p = self._get_params(sp_i, sp_j)
        A, rho = p['A'], p['rho']
        ex = np.exp(-r / rho)
        V = A * ex
        dVdr = -A / rho * ex
        return V, dVdr


class BuckinghamPotential(BasePairPotential):
    """
    Buckingham potential: Born-Mayer repulsion + C/r^6 dispersion.

    V(r) = A . exp(-r / rho) - C / r^6

    Newly added engine class: prior to this restructuring, Buckingham was
    only ever evaluated through a bespoke numba pair-loop invoked directly
    by the legacy fitting code, bypassing ``PotentialEngine``/
    ``BasePairPotential`` entirely (unlike LJ/Morse/BornMayer, which were
    already proper ``BasePairPotential`` subclasses). That meant
    ``BuckinghamParameterSet.build_engine()`` (fitting/parameter_sets.py)
    had no engine-side class to construct and silently dropped the
    dispersion term ``C``, forwarding only ``A``/``rho`` into a
    ``BornMayerPotential`` -- fine for the legacy Buckingham-only fitting
    path (which used its own separate, dispersion-complete implementation)
    but a latent bug for the *generic* fitting path
    (``fit_potential``/``generic_loss_function``), which is exactly the
    path the `fit` CLI command's now-unified 6-potential support needs.
    This class closes that gap: it has the same shape as
    ``BornMayerPotential`` (same base class, same parameter-lookup
    convention), so it is a drop-in addition to ``PotentialEngine`` and
    automatically gets numba JIT + MPI domain-split "for free" via the
    inherited ``BasePairPotential.compute()``.

    Parameters
    ----------
    params : dict  frozenset({sp1, sp2}) -> {'A': A[eV], 'rho': rho[A], 'C': C[eV.A^6]}
    r_cut  : cut-off [A]
    """

    def __init__(self, params: dict, r_cut: float = 10.0):
        species = sorted({s for pair in params for s in pair})
        super().__init__(r_cut=r_cut, species=species)
        self.params = params

    def pair_energy_force(self, r, sp_i=None, sp_j=None):
        p = self._get_params(sp_i, sp_j)
        A, rho, C = p['A'], p['rho'], p['C']
        ex = np.exp(-r / rho)
        r6 = r**6
        V = A * ex - C / r6
        dVdr = -A / rho * ex + 6.0 * C / (r6 * r)
        return V, dVdr


class MiePotential(BasePairPotential):
    """
    Mie potential: generalized Lennard-Jones with independently tunable
    repulsive/attractive exponents.

    V(r) = C(n,m) . epsilon . [(sigma/r)^n - (sigma/r)^m],   n > m > 0
    C(n,m) = (n/(n-m)) . (n/m)^(m/(n-m))

    C(n,m) is chosen so that V has its minimum, -epsilon, at
    r_min = sigma.(n/m)^(1/(n-m)) -- the same normalisation convention
    as the standard 12-6 Lennard-Jones potential, which this reduces to
    exactly when n=12, m=6 (C(12,6)=4); see
    test_engine_forces.py::test_mie_reduces_to_lj_at_12_6 for a direct
    cross-check against LennardJonesPotential.

    n and m are fixed hyperparameters set at construction, not fit
    variables -- only epsilon/sigma are exposed to fitting via
    MieParameterSet, the same design choice BuckinghamPotential makes
    for its own functional form (fitting A/rho/C, not the exponent of
    r^-6): jointly optimising the exponents themselves would be a much
    harder, non-convex problem, and is out of scope for what this class
    is meant to demonstrate (Phase 1/2's "add a new pair potential"
    recipe).

    Parameters
    ----------
    params : dict  frozenset({sp1, sp2}) -> {'epsilon': epsilon[eV], 'sigma': sigma[A]}
    n, m   : repulsive / attractive exponents (n > m > 0). Default
             (12, 6) reproduces standard Lennard-Jones.
    r_cut  : cut-off [A]
    shift  : apply force-shifted truncation (default True)
    """

    def __init__(self,
                 params: dict,
                 n: float = 12.0,
                 m: float = 6.0,
                 r_cut: float = 10.0,
                 shift: bool = True):
        if not n > m > 0:
            raise ValueError(f"MiePotential: require n > m > 0, got n={n}, m={m}")
        species = sorted({s for pair in params for s in pair})
        super().__init__(r_cut=r_cut, species=species)
        self.params = params
        self.n = float(n)
        self.m = float(m)
        self.C = ((self.n / (self.n - self.m))
                 * (self.n / self.m) ** (self.m / (self.n - self.m)))
        self.shift = shift
        self._cache_shift: dict = {}

    def _V_dVdr(self, r, eps, sig):
        sr_n = (sig / r) ** self.n
        sr_m = (sig / r) ** self.m
        V = self.C * eps * (sr_n - sr_m)
        dVdr = self.C * eps / r * (-self.n * sr_n + self.m * sr_m)
        return V, dVdr

    def _shift_values(self, sp_i: str, sp_j: str) -> Tuple[float, float]:
        key = (sp_i, sp_j) if sp_i <= sp_j else (sp_j, sp_i)
        if key not in self._cache_shift:
            p = self._get_params(sp_i, sp_j)
            Vc, dVc = self._V_dVdr(self.r_cut, p['epsilon'], p['sigma'])
            self._cache_shift[key] = (float(Vc), float(dVc))
        return self._cache_shift[key]

    def pair_energy_force(self,
                           r: np.ndarray,
                           sp_i: Optional[str] = None,
                           sp_j: Optional[str] = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
        p = self._get_params(sp_i, sp_j)
        V, dVdr = self._V_dVdr(r, p['epsilon'], p['sigma'])
        if self.shift:
            Vc, dVc = self._shift_values(sp_i, sp_j)
            V = V - Vc - dVc * (r - self.r_cut)
            dVdr = dVdr - dVc
        return V, dVdr


class YukawaPotential(BasePairPotential):
    """
    Yukawa (screened Coulomb) potential.

    V(r) = A . exp(-kappa.r) / r

    A single, global screening length 1/kappa is shared by every
    species pair (a screening length is a property of the surrounding
    medium, not of a specific pair of atoms), while the per-pair
    prefactor A folds in the effective q_i.q_j charge product -- the
    same "one fitted prefactor per pair" design BuckinghamPotential uses
    for A/rho/C, rather than separately tracking per-atom charges the
    way the bare/Ewald Coulomb path does.

    Screening is exactly what makes the real-space sum converge, so
    this potential uses direct real-space summation via the existing
    BasePairPotential/NeighbourList machinery, with no reciprocal-space
    (Ewald/PPPM) treatment -- unlike bare 1/r Coulomb, none is needed.
    kappa must be large enough that the screening length 1/kappa stays
    well inside r_cut, or the truncation at r_cut will not be a good
    approximation.

    Parameters
    ----------
    params : dict  frozenset({sp1, sp2}) -> {'A': A[eV.A]}
    kappa  : inverse screening length [A^-1], shared by every pair.
    r_cut  : cut-off [A]
    shift  : apply force-shifted truncation (default True; matters less
             here than for LJ/Morse since V already decays
             exponentially, but kept for consistency and exact energy
             conservation in MD).
    """

    def __init__(self,
                 params: dict,
                 kappa: float,
                 r_cut: float = 10.0,
                 shift: bool = True):
        if kappa <= 0:
            raise ValueError(f"YukawaPotential: kappa must be > 0, got {kappa}")
        species = sorted({s for pair in params for s in pair})
        super().__init__(r_cut=r_cut, species=species)
        self.params = params
        self.kappa = float(kappa)
        self.shift = shift
        self._cache_shift: dict = {}

    def _V_dVdr(self, r, A):
        ex = np.exp(-self.kappa * r)
        V = A * ex / r
        dVdr = -A * ex * (self.kappa / r + 1.0 / r**2)
        return V, dVdr

    def _shift_values(self, sp_i: str, sp_j: str) -> Tuple[float, float]:
        key = (sp_i, sp_j) if sp_i <= sp_j else (sp_j, sp_i)
        if key not in self._cache_shift:
            p = self._get_params(sp_i, sp_j)
            Vc, dVc = self._V_dVdr(self.r_cut, p['A'])
            self._cache_shift[key] = (float(Vc), float(dVc))
        return self._cache_shift[key]

    def pair_energy_force(self,
                           r: np.ndarray,
                           sp_i: Optional[str] = None,
                           sp_j: Optional[str] = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
        p = self._get_params(sp_i, sp_j)
        V, dVdr = self._V_dVdr(r, p['A'])
        if self.shift:
            Vc, dVc = self._shift_values(sp_i, sp_j)
            V = V - Vc - dVc * (r - self.r_cut)
            dVdr = dVdr - dVc
        return V, dVdr
