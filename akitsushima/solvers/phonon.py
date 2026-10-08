"""Phonon dispersion, density of states, and harmonic thermodynamics via
the finite-displacement method.
"""
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from ..constants import ATOMIC_MASS, FREQ_RAD_FS_TO_WAVENUMBER_CM
from ..utils.jit import _HAS_NUMBA, _numba_njit
from ..utils.mpi import (
    _HAS_MPI,
    _mpi_allreduce_sum,
    _mpi_domain_split,
    _mpi_rank,
    _mpi_size,
    local_only,
)
from ..utils.optional_matplotlib import HAS_MATPLOTLIB, plt
from ..utils.timing import TimingCollector

__all__ = ["PhononCalculator"]

# cm^-1 -> THz and cm^-1 -> rad/s (c = 2.99792458e10 cm/s)
_THZ_TO_CM  = 1e12 / 2.99792458e10
# h [eV s] x 1e12: photon energy of 1 THz in eV (h, e exact SI 2019).
_THZ_TO_EV  = 6.62607015e-34 * 1e12 / 1.602176634e-19
_KB_EV      = 1.380649e-23 / 1.602176634e-19
_EV_TO_KJ_MOL = 1.602176634e-19 * 6.02214076e23 / 1e3   # 96.485 kJ/mol per eV
_R_J_K_MOL  = 1.380649e-23 * 6.02214076e23               # k_B N_A [J/K/mol]


def _debye_cv_ratio(y: float) -> float:
    """Debye C_v / (3 N k_B) at Theta_D / T = y:
    (3 / y^3) int_0^y x^4 e^x / (e^x - 1)^2 dx."""
    from scipy.integrate import quad
    if y < 1e-8:
        return 1.0
    g = lambda x: x**4 * np.exp(-x) / (-np.expm1(-x))**2 if x > 0 else 0.0
    return 3.0 / y**3 * quad(g, 0.0, y, limit=200)[0]


def _debye_theta_from_cv(ratio: float, T: float) -> float:
    """Theta_D [K] at which the Debye model gives C_v / (3 N k_B) = ratio
    at temperature T; NaN outside (0, 1) or at T = 0.  D(y) falls
    monotonically from 1 to 0, so the root is unique."""
    from scipy.optimize import brentq
    if T <= 0.0 or not (0.0 < ratio < 1.0 - 1e-10):
        return float("nan")
    hi = 1.0
    while _debye_cv_ratio(hi) > ratio:
        hi *= 2.0
        if hi > 1e4:
            return float("nan")
    return float(T * brentq(lambda y: _debye_cv_ratio(y) - ratio, 0.0, hi,
                            xtol=1e-12, rtol=1e-12))
_CM_TO_RAD_S = 2.0 * np.pi * 2.99792458e10


def _build_D_raw_numpy(IFC, masses, sc_to_prim, img_R, img_n, origin,
                       q_cart, N):
    """
    Build the raw (pre-D-ASR) dynamical matrix at q_cart.

    D_{ia,ib}(q) = sum_j  Phi(ia_0 -> j) / sqrt(M_a M_b) . <exp(i q . R)>_j

    where <.>_j averages over the lattice translations R that bring atom j
    to its minimum-image position relative to ia_0 (img_R[ia, j, :img_n]);
    several when j sits exactly on the supercell's Wigner-Seitz boundary.

    Written as explicit loops (rather than vectorised NumPy slicing) so it
    can be JIT-compiled with numba: this is called once per q-point, and
    band-structure/DOS calculations typically evaluate hundreds to
    thousands of q-points, each doing O(N_basis x N_sc) work.
    """
    N_sc = sc_to_prim.shape[0]
    D = np.zeros((N * 3, N * 3), dtype=np.complex128)
    for ia_loc in range(N):
        ia_glob = origin[ia_loc]
        mi = masses[ia_loc]
        for j in range(N_sc):
            ib = sc_to_prim[j]
            mj = masses[ib]
            n_im = img_n[ia_loc, j]
            ph = complex(0.0, 0.0)
            for k in range(n_im):
                dot = (q_cart[0] * img_R[ia_loc, j, k, 0]
                       + q_cart[1] * img_R[ia_loc, j, k, 1]
                       + q_cart[2] * img_R[ia_loc, j, k, 2])
                ph += complex(np.cos(dot), np.sin(dot))
            ph /= n_im
            inv_sqrt_m = 1.0 / np.sqrt(mi * mj)
            for di in range(3):
                for dj in range(3):
                    D[ia_loc * 3 + di, ib * 3 + dj] += (
                        IFC[ia_glob * 3 + di, j * 3 + dj] * inv_sqrt_m * ph)
    return D


_build_D_raw_jit = (_numba_njit(_build_D_raw_numpy) if _HAS_NUMBA
                    else _build_D_raw_numpy)


def _dos_projection_numpy(freqs_q, evecs_q, freq_axis, half_inv_sig2, norm,
                          N_basis):
    """
    Gaussian-smeared total + atom-projected DOS contribution from a single
    q-point's mode frequencies/eigenvectors.

    Written as explicit loops for the same reason as _build_D_raw_numpy
    above: called once per q-point, avoiding N_modes short-lived
    (n_dos,)-sized NumPy temporaries per call.
    """
    n_modes = freqs_q.shape[0]
    n_dos = freq_axis.shape[0]
    dos = np.zeros(n_dos)
    dos_partial = np.zeros((N_basis, n_dos))
    for m in range(n_modes):
        nu = freqs_q[m]
        weights = np.zeros(N_basis)
        for ia in range(N_basis):
            s = 0.0
            for c in range(3):
                v = evecs_q[ia * 3 + c, m]
                s += v.real * v.real + v.imag * v.imag
            weights[ia] = s
        for k in range(n_dos):
            g = np.exp(-half_inv_sig2 * (freq_axis[k] - nu) ** 2) * norm
            dos[k] += g
            for ia in range(N_basis):
                dos_partial[ia, k] += g * weights[ia]
    return dos, dos_partial


_dos_projection_jit = (_numba_njit(_dos_projection_numpy) if _HAS_NUMBA
                       else _dos_projection_numpy)


class PhononCalculator:
    """
    Phonon dispersion and density of states via the finite-displacement method.

    Parameters
    ----------
    lattice    : (3,3) primitive-cell lattice vectors [A].
    positions  : (N_basis, 3) equilibrium atomic positions [A].
    atom_types : list of element symbols, length N_basis.
    force_fn   : unified force callable from make_force_fn().
    supercell  : int or (3,) int array -- supercell replication along each
                 lattice vector.  Default 2 (= 2x2x2).
    delta      : finite-displacement amplitude [A].  Default 0.01.
    masses     : optional (N_basis,) array [amu]; defaults to ATOMIC_MASS.

    Examples
    --------
    >>> engine, meta = load_potential("lj_Ar.json")
    >>> fn  = make_force_fn(engine)
    >>> pc  = PhononCalculator(lattice, positions, atom_types, fn,
    ...                         supercell=3)
    >>> pc.compute_ifc()
    >>> bands = pc.compute_band_structure(
    ...     [('G',[0,0,0]),('X',[0.5,0,0]),('M',[0.5,0.5,0]),
    ...      ('G',[0,0,0]),('R',[0.5,0.5,0.5])],
    ...     n_points=50)
    >>> pc.compute_dos(q_mesh=[10,10,10])
    >>> pc.save_results("phonon_output")
    """

    def __init__(self,
                 lattice    : np.ndarray,
                 positions  : np.ndarray,
                 atom_types : list,
                 force_fn   : Callable,
                 supercell  : Union[int, List[int]] = 2,
                 delta      : float = 0.01,
                 masses     : Optional[np.ndarray] = None):

        self.lattice_prim = np.asarray(lattice, dtype=float).copy()
        self.positions    = np.asarray(positions, dtype=float).copy()
        self.atom_types   = list(atom_types)
        self.force_fn     = force_fn
        self.delta        = float(delta)
        self.N_basis      = len(atom_types)

        # Supercell replication factors
        if np.isscalar(supercell):
            self.supercell = [int(supercell)] * 3
        else:
            self.supercell = [int(s) for s in supercell]

        # Atomic masses
        if masses is not None:
            self.masses = np.asarray(masses, dtype=float)
        else:
            self.masses = np.array(
                [ATOMIC_MASS.get(a, 1.0) for a in atom_types])

        # Supercell lattice
        self.lattice_sc = (np.diag(self.supercell) @ self.lattice_prim)

        # Build supercell geometry
        (self._pos_sc,
         self._types_sc,
         self._sc_to_prim,
         self._sc_cell) = self._build_supercell()

        self.N_sc  = len(self._pos_sc)

        # Lattice-vector shift of each supercell atom's cell, precomputed
        # once here (rather than recomputed inside _build_D_raw on every
        # one of the potentially thousands of q-point calls that a band
        # structure / DOS mesh entails).
        self._R_sc = np.array([c.astype(float) @ self.lattice_prim
                               for c in self._sc_cell])
        self._img_R, self._img_n = self._minimum_image_translations()

        # r_cut from force_fn attribute (for supercell adequacy check)
        self._r_cut = float(getattr(force_fn, 'r_cut', 10.0))

        # Check supercell adequacy
        self._check_supercell_size()

        # C-4: Warn if an ESM (explicit-surface-model) force_fn is used
        # with z-replication.
        # Bug-fix note (S12): the original checked
        # isinstance(force_fn, ESMCalculator), requiring this module to
        # import ESMCalculator purely to recognise it. ESMCalculator sets
        # a public .kind = "esm" attribute on itself (see engine/coulomb/
        # esm.py); make_force_fn()'s returned closures carry the same
        # convention (see fitting/force_fn.py), so checking
        # getattr(force_fn, "kind", None) works uniformly for both and
        # removes the need for this module to depend on engine internals.
        # ADR-0008: an error, not a warning -- replicating along z compresses
        # the vacuum and moves the ESM boundaries, so the result is not the
        # system asked for.
        from ..fitting.force_fn import esm_bc, uses_esm
        if uses_esm(force_fn) and self.supercell[2] > 1:
            raise ValueError(
                f"PhononCalculator: ESM electrostatics (bc='{esm_bc(force_fn)}') "
                f"with supercell[2] = {self.supercell[2]} > 1: replicating the "
                "cell in z compresses the vacuum layer and corrupts the ESM "
                "boundary condition.  Use supercell = [N, N, 1].")

        # Results (populated after compute_ifc / compute_band_structure / etc.)
        self.IFC          : Optional[np.ndarray] = None   # (3N_sc, 3N_sc) eV/A^2
        self._D_correction: Optional[np.ndarray] = None   # D-ASR correction
        self._band_result : Optional[dict]       = None
        self._dos_result  : Optional[dict]       = None
        self._thermal_result: Optional[dict]     = None
        self._nac         : Optional[dict]       = None   # ADR-0019

    # ------------------------------------------------------------------ #
    #  1. Supercell construction                                           #
    # ------------------------------------------------------------------ #

    def _build_supercell(self):
        """
        Replicate the primitive cell into a S0xS1xS2 supercell.

        Returns pos_sc, types_sc, sc_to_prim, sc_cell where
        sc_cell[j] = (n0, n1, n2) is the lattice-vector index of atom j.
        """
        S0, S1, S2 = self.supercell
        pos_sc_list, types_sc, sc_to_prim, sc_cell = [], [], [], []
        for n0 in range(S0):
            for n1 in range(S1):
                for n2 in range(S2):
                    shift = (n0 * self.lattice_prim[0] +
                             n1 * self.lattice_prim[1] +
                             n2 * self.lattice_prim[2])
                    for ib, (p, sp) in enumerate(
                            zip(self.positions, self.atom_types)):
                        pos_sc_list.append(p + shift)
                        types_sc.append(sp)
                        sc_to_prim.append(ib)
                        sc_cell.append(np.array([n0, n1, n2]))
        return (np.array(pos_sc_list, dtype=float),
                types_sc,
                sc_to_prim,
                sc_cell)

    def _minimum_image_translations(self, tol: float = 1e-5):
        """
        For every (origin-cell atom a, supercell atom j) pair, the lattice
        translations R_j + T (T a supercell vector) that place j at its
        minimum-image distance from a.

        Bug-fix note: D(q) used the raw cell offset R_j of each supercell
        atom, n in [0, S) along each axis, as the Fourier phase.  That is
        exact only for q commensurate with the supercell (q = m/S); at any
        other q it interpolates with the wrong images.  Checked against
        phonopy on Tersoff Si (4x4x4): Gamma, X and L agreed to 3e-4 THz,
        while q = (0.13, 0.13, 0.13) was off by 10 THz.  Ties (j exactly on
        the Wigner-Seitz boundary of the supercell) are averaged, as in
        phonopy.  Returns img_R (N_basis, N_sc, n_max, 3) and img_n.
        """
        shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1)
                           for k in (-1, 0, 1)], dtype=float) @ self.lattice_sc
        origin = self._origin_indices()
        N, N_sc = self.N_basis, self.N_sc
        cand = []
        n_img = np.zeros((N, N_sc), dtype=np.int64)
        for a, ia in enumerate(origin):
            row = []
            d0 = self._pos_sc - self._pos_sc[ia]              # (N_sc, 3)
            for j in range(N_sc):
                dist = np.linalg.norm(d0[j] + shifts, axis=1)
                keep = dist < dist.min() + tol
                row.append(self._R_sc[j] + shifts[keep])
                n_img[a, j] = int(keep.sum())
            cand.append(row)
        img_R = np.zeros((N, N_sc, int(n_img.max()), 3))
        for a in range(N):
            for j in range(N_sc):
                img_R[a, j, :n_img[a, j]] = cand[a][j]
        return img_R, n_img

    def _check_supercell_size(self) -> None:
        """Warn if any supercell direction is smaller than 2 * r_cut."""
        lat  = self.lattice_prim
        S    = self.supercell
        rc   = self._r_cut
        for i in range(3):
            side = S[i] * np.linalg.norm(lat[i])
            if side < 2.0 * rc:
                import warnings as _w
                _w.warn(
                    f"PhononCalculator: supercell side {i} = {side:.2f} A "
                    f"< 2*r_cut = {2*rc:.2f} A.  "
                    f"IFCs may be incomplete; increase supercell or reduce r_cut. "
                    f"D-ASR correction will be applied automatically.",
                    UserWarning, stacklevel=3)

    # ------------------------------------------------------------------ #
    #  2. IFC extraction by central-difference finite displacements        #
    # ------------------------------------------------------------------ #

    def compute_ifc(self,
                    n_jobs : int  = 1,
                    verbose: bool = True,
                    timing : bool = False,
                    ) -> np.ndarray:
        """
        Compute interatomic force constants by central-difference FD.

        Each atom of the ORIGIN cell is displaced by +/-delta along x, y, z;
        the columns of every other supercell atom follow by lattice
        translation, Phi[(a, M), (b, L)] = Phi[(a, M - L), (b, 0)] (indices
        modulo the supercell).  Cost: 6 * N_basis force evaluations (it was
        6 * N_sc, displacing every supercell atom: the same matrix up to
        round-off, N_cells times the work), split across MPI ranks (each
        displacement is a fully independent force_fn call) and, within
        each rank's share, optionally further split across n_jobs threads.

        Parameters
        ----------
        n_jobs  : number of parallel worker THREADS per MPI rank. Each
                  worker calls force_fn independently for one displacement.
                  Default 1 (serial). Set to -1 to use all available CPU
                  cores. Note: force_fn must be thread-safe for n_jobs > 1.
                  ExternalFileEngine and ELMModel.compute are NOT thread-safe;
                  use n_jobs=1 for these (MPI-splitting across ranks
                  remains available regardless of n_jobs).
        verbose : print progress (rank 0 only).

        The raw IFC is then:
          1. Symmetrised:  IFC <- (IFC + IFC.T) / 2
          2. ASR enforced: sum_ii <- -sum_{j!=i} sum_ij

        A dynamical-matrix-level ASR correction (D-ASR) is also precomputed
        and stored for use in build_dynamical_matrix().

        Returns
        -------
        IFC : (3*N_sc, 3*N_sc) ndarray [eV/A^2]
        """
        import os as _os
        N_sc  = self.N_sc
        delta = self.delta

        if n_jobs == -1:
            n_jobs = _os.cpu_count() or 1

        tc_ifc = TimingCollector(enabled=timing)
        tc_ifc.start("total")
        if verbose and _mpi_rank == 0:
            parallel_str = f", {n_jobs} threads/rank" if n_jobs > 1 else ", serial"
            mpi_str = f", {_mpi_size} MPI ranks" if _HAS_MPI else ""
            print(f"\n  Computing IFC  "
                  f"({6 * self.N_basis} force evaluations, delta={delta} A, "
                  f"N_sc={N_sc}{parallel_str}{mpi_str}) ...")

        def _eval_displacement(j: int, b: int) -> np.ndarray:
            """Return -(F_plus - F_minus) / (2*delta) for atom j, axis b.

            Bug-fix note (S2): the force-constant matrix is
            Phi_ij = -dF_i/dr_j (since F = -dV/dr and Phi = d^2V/dr^2 =
            -dF/dr); the original omitted this negative sign here,
            flipping the sign of the entire dynamical matrix and
            misclassifying every stable mode (acoustic and optical alike)
            as "imaginary" -- confirmed by comparing against
            TransitionStateAnalyser.compute_hessian, which computes the
            structurally identical central-difference Hessian correctly
            (with the negative sign) and is used as the regression
            reference. This sign error is also why
            debye_temperature_from_vsound() would previously fail on
            every dynamically-stable structure passed to it.
            """
            pos = self._pos_sc.copy()
            pos[j, b] += delta
            _, F_p, _ = self.force_fn(self.lattice_sc, pos, self._types_sc)
            pos[j, b] -= 2.0 * delta
            _, F_m, _ = self.force_fn(self.lattice_sc, pos, self._types_sc)
            return -(np.asarray(F_p) - np.asarray(F_m)) / (2.0 * delta)

        # MPI: split the flat list of (j, b) displacement tasks across
        # ranks (each is an independent force_fn call); Allreduce-sum the
        # resulting IFC columns afterwards (each column is populated by
        # exactly one rank and left at zero on every other rank, so the
        # sum reconstructs the full matrix exactly, not approximately).
        #
        # Bug-fix note: force_fn's underlying engine potentials ALSO
        # unconditionally split their own internal pair-list work across
        # every rank on the same global communicator and Allreduce the
        # result. Without local_only() here, each rank's force_fn call
        # for its OWN displacement would try to Allreduce with every
        # other rank's call for a DIFFERENT displacement -- silently
        # mixing together partial force sums from unrelated displaced
        # structures (confirmed via a bit-for-bit divergence between a
        # 1-rank and a 3-rank run of exactly this pattern: no crash or
        # hang, just silently wrong IFC entries everywhere). local_only()
        # is entered ONCE around the entire local batch below (rather
        # than once per task) so that, even with n_jobs > 1 spawning
        # worker threads inside it, only the single calling thread ever
        # writes the suppression flag -- worker threads only ever read it.
        origin = self._origin_indices()
        tasks = [(j, b) for j in origin for b in range(3)]
        i_start, i_end = _mpi_domain_split(len(tasks))
        local_tasks = tasks[i_start:i_end]

        IFC_local = np.zeros((N_sc * 3, N_sc * 3))

        with local_only():
            if n_jobs == 1:
                for j, b in local_tasks:
                    IFC_local[:, j * 3 + b] = _eval_displacement(j, b).ravel()
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                    fut_map = {ex.submit(_eval_displacement, j, b): (j, b)
                               for j, b in local_tasks}
                    for fut in as_completed(fut_map):
                        j, b = fut_map[fut]
                        IFC_local[:, j * 3 + b] = fut.result().ravel()

        IFC = self._fill_by_translation(_mpi_allreduce_sum(IFC_local), origin)

        # 1. Symmetrise (Newton's 3rd law: sum_ij = sum_ji^T)
        IFC = 0.5 * (IFC + IFC.T)

        # 2. Acoustic Sum Rule at IFC level
        # A-2: reshape IFC to (N_sc, 3, N_sc, 3) and sum over atom/cart axes
        # instead of a Python loop over rows.
        IFC_4d  = IFC.reshape(N_sc, 3, N_sc, 3)
        row_sum = IFC_4d.sum(axis=2)                # (N_sc, 3, 3)
        for i in range(N_sc):
            IFC_4d[i, :, i, :] -= row_sum[i]        # subtract total and re-add
            # After ASR: sum_ii = -sum_{j!=i} sum_ij
            # row_sum[i] includes sum_ii, so the correct ASR diagonal is:
            # sum_ii_new = sum_ii - row_sum[i] = sum_ii - (Phi_ii + sum_{j!=i} sum_ij)
            #          = -sum_{j!=i} sum_ij  (check)
        IFC = IFC_4d.reshape(N_sc * 3, N_sc * 3)

        self.IFC = IFC

        # 3. Pre-compute D-ASR correction (D_0 after ASR - D_0 before ASR)
        self._precompute_dasr()

        if verbose and _mpi_rank == 0:
            # Symmetry residual as quality indicator
            row_sums = IFC.reshape(N_sc, 3, N_sc, 3).sum(axis=(2, 3))
            print(f"  IFC computed  "
                  f"(ASR residual = {np.max(np.abs(row_sums)):.2e} eV/A^2)")

        return IFC

    def _fill_by_translation(self, IFC: np.ndarray, origin: List[int]
                             ) -> np.ndarray:
        """Complete an IFC whose only filled columns are the origin-cell
        atoms': the column of atom (b, L) is that of (b, 0) with every row
        atom (a, M) read from (a, M - L), cell indices modulo the
        supercell."""
        S = np.asarray(self.supercell, dtype=np.int64)
        cells = np.asarray(self._sc_cell, dtype=np.int64)
        basis = np.asarray(self._sc_to_prim, dtype=np.int64)
        index = {(int(b), *map(int, c)): k
                 for k, (b, c) in enumerate(zip(basis, cells))}
        orig_of = {int(basis[j]): j for j in origin}
        out = np.empty_like(IFC)
        for jp in range(self.N_sc):
            L = cells[jp]
            src_rows = np.array([index[(int(basis[i]), *map(int, (cells[i] - L) % S))]
                                 for i in range(self.N_sc)])
            j0 = orig_of[int(basis[jp])]
            rows3 = (3 * src_rows[:, None] + np.arange(3)).ravel()
            out[:, 3 * jp:3 * jp + 3] = IFC[rows3][:, 3 * j0:3 * j0 + 3]
        return out

    def _precompute_dasr(self) -> None:
        """
        Pre-compute the D-ASR correction matrix.

        D(q) is built from the IFC, then its diagonal blocks are corrected
        to enforce exact acoustic zeros.  The difference
            D_correction = D(0)_ASR - D(0)_raw
        is stored and added to every D(q) call, ensuring all acoustic modes
        are exactly zero at Gamma regardless of supercell completeness.
        """
        D0_raw = self._build_D_raw(np.zeros(3)).real
        D0_asr = D0_raw.copy()
        N = self.N_basis
        # A-2: reshape to (N,3,N,3) and sum over atom/Cartesian axes
        # Bug-fix note: D is mass-weighted, so the acoustic sum rule reads
        # sum_b sqrt(M_b) D_ab(0) = 0, not sum_b D_ab(0) = 0.  The plain
        # row sum agreed only for equal masses (Si); for AlN it moved the
        # Gamma optical modes by up to 1.8 THz even though the IFC already
        # satisfied the sum rule to 5e-15.
        D_4d     = D0_asr.reshape(N, 3, N, 3)
        sqm      = np.sqrt(np.asarray(self.masses, dtype=float))
        row_sum  = np.einsum("aibj,b->aij", D_4d, sqm) / sqm[:, None, None]
        for ia in range(N):
            D_4d[ia, :, ia, :] -= row_sum[ia]
        D0_asr = D_4d.reshape(N * 3, N * 3)
        self._D_correction = D0_asr - D0_raw   # (3N, 3N) real

    # ------------------------------------------------------------------ #
    #  3. Dynamical matrix                                                 #
    # ------------------------------------------------------------------ #

    def _origin_indices(self) -> List[int]:
        """Indices of atoms in the origin cell (0,0,0) within the SC."""
        return [k for k, c in enumerate(self._sc_cell)
                if np.all(c == [0, 0, 0])]

    def _build_D_raw(self, q_cart: np.ndarray,
                     IFC: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Build the 3Nx3N dynamical matrix at q_cart [A^-1] without D-ASR.

        D_{ia,ib}(q) = sum_j  Phi(ia_0 -> j) / sqrt(M_a M_b) . exp(i q . R_j)

        where ia_0 is the origin-cell copy of basis atom ia, and R_j is
        the lattice-vector shift of atom j's cell.
        """
        origin = np.asarray(self._origin_indices(), dtype=np.int64)
        sc_to_prim = np.asarray(self._sc_to_prim, dtype=np.int64)
        D = _build_D_raw_jit(self.IFC if IFC is None else IFC,
                             self.masses, sc_to_prim, self._img_R,
                             self._img_n, origin,
                             np.asarray(q_cart, dtype=np.float64),
                             self.N_basis)
        return 0.5 * (D + D.conj().T)

    def build_dynamical_matrix(self, q_cart: np.ndarray,
                               q_direction: Optional[np.ndarray] = None
                               ) -> np.ndarray:
        """
        Build the mass-weighted dynamical matrix D(q) with D-ASR correction.

        Parameters
        ----------
        q_cart      : (3,) array, q-vector in Cartesian coordinates [A^-1].
        q_direction : with NAC set and q = 0, the direction (Cartesian) of
                      approach, which selects the LO modes; without it
                      q = 0 gives the analytic (TO) matrix.

        Returns
        -------
        D : (3*N_basis, 3*N_basis) complex Hermitian matrix [eV/(A^2 amu)].
        """
        if self.IFC is None:
            raise RuntimeError(
                "PhononCalculator: call compute_ifc() first.")
        if self._nac is None:
            D = self._build_D_raw(q_cart)
            D += self._D_correction    # D-ASR: enforce acoustic zeros at Gamma
            return 0.5 * (D + D.conj().T)
        D = self._build_D_raw(q_cart, self._nac["IFC_SR"])
        D += self._nac["D_correction"]
        D += self._nac_dd(q_cart, q_direction)
        return 0.5 * (D + D.conj().T)

    # ------------------------------------------------------------------ #
    #  NAC: Gonze-Lee dipole-dipole term (ADR-0019)                       #
    # ------------------------------------------------------------------ #

    def set_nac(self, born: np.ndarray, epsilon_inf: np.ndarray,
                n_G: int = 300) -> None:
        """
        Non-analytic correction by the method of Gonze and Lee (PRB 55,
        10355 (1997)), with only the reciprocal-space Ewald term as in
        phonopy's default: the dipole-dipole force constants are removed
        from the supercell IFC at the q-points commensurate with the
        supercell (shortest representatives), leaving short-range IFC;
        every D(q) then adds the dipole-dipole term back.  D is unchanged
        at commensurate q; LO-TO splitting appears at Gamma along a
        q_direction.  Z* and eps_inf are taken as volume independent.

        born        : (N_basis, 3, 3) Born effective charges [e], Z[k, a, b]
                      = dP_a / du_kb (sum rule already imposed).
        epsilon_inf : (3, 3) high-frequency dielectric tensor.
        n_G         : number of G-vectors in the sum (phonopy's 300).
        """
        from .dielectric import dipole_dipole, g_vectors
        if self.IFC is None:
            raise RuntimeError("set_nac: call compute_ifc() first.")
        Z = np.asarray(born, dtype=float).reshape(self.N_basis, 3, 3)
        eps = np.asarray(epsilon_inf, dtype=float).reshape(3, 3)
        G, g_cut = g_vectors(self.lattice_prim, n_G)
        # exp(-G_cut^2 tr(eps)/3 / 4 Lambda^2) = 1e-10, as phonopy.
        lam = np.sqrt(g_cut**2 * np.trace(eps) / 3.0 / 4.0 / np.log(1e10))
        vol = abs(float(np.linalg.det(self.lattice_prim)))
        pos = self.positions
        drift = dipole_dipole(np.zeros(3), G, pos, Z, eps, vol, lam,
                              drift=np.zeros((self.N_basis, 3, 3))).sum(axis=2)
        args = (G, pos, Z, eps, vol, lam)
        # Commensurate points, shortest representatives.
        rec = 2.0 * np.pi * np.linalg.inv(self.lattice_prim).T
        S = self.supercell
        shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1)
                           for k in (-1, 0, 1)], float) @ rec
        comm = []
        for i in range(S[0]):
            for j in range(S[1]):
                for k in range(S[2]):
                    q = np.array([i / S[0], j / S[1], k / S[2]]) @ rec
                    cand = q + shifts
                    comm.append(cand[np.argmin(np.linalg.norm(cand, axis=1))])
        n_c = len(comm)
        N, N_sc = self.N_basis, self.N_sc
        phi = np.zeros((N, 3, N_sc, 3))
        for q in comm:
            C = dipole_dipole(q, *args, drift=drift)          # (N,3,N,3)
            ph = np.exp(-1j * self._R_sc @ q)                  # (N_sc,)
            for j in range(N_sc):
                b = self._sc_to_prim[j]
                phi[:, :, j, :] += np.real(C[:, :, b, :] * ph[j])
        phi /= n_c
        IFC_SR = self.IFC.copy()
        for a, ia in enumerate(self._origin_indices()):
            IFC_SR[3 * ia:3 * ia + 3, :] -= phi[a].reshape(3, 3 * N_sc)
        self._nac = {"Z": Z, "epsilon_inf": eps, "args": args, "drift": drift,
                     "lambda": lam, "n_G": len(G), "IFC_SR": IFC_SR}
        # D-ASR of the short-range part (the dipole term obeys the sum
        # rule by construction of the drift).
        D0 = self._build_D_raw(np.zeros(3), IFC_SR).real
        sqm = np.sqrt(self.masses)
        D4 = D0.copy().reshape(N, 3, N, 3)
        rs = np.einsum("aibj,b->aij", D4, sqm) / sqm[:, None, None]
        for a in range(N):
            D4[a, :, a, :] -= rs[a]
        self._nac["D_correction"] = D4.reshape(3 * N, 3 * N) - D0

    def _nac_dd(self, q_cart, q_direction=None) -> np.ndarray:
        """Mass-weighted dipole-dipole matrix at q (cell gauge)."""
        from .dielectric import dipole_dipole
        C = dipole_dipole(q_cart, *self._nac["args"], q_direction=q_direction,
                          drift=self._nac["drift"])
        inv = 1.0 / np.sqrt(self.masses)
        C = C * inv[:, None, None, None] * inv[None, None, :, None]
        n = 3 * self.N_basis
        return C.reshape(n, n)

    # ------------------------------------------------------------------ #
    #  4. Eigenvalue -> frequency conversion                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _eig_to_freq(eigenvalues: np.ndarray) -> np.ndarray:
        """
        Convert dynamical-matrix eigenvalues [eV/(A^2 amu)] to
        frequencies [cm^-1].

        Negative eigenvalues (imaginary frequencies) are returned as
        negative numbers with a physical magnitude.
        """
        from ..constants import EV_ANG_TO_AMU_ANG_FS2
        freqs = np.zeros(len(eigenvalues))
        for i, lam in enumerate(eigenvalues):
            omega2 = lam * EV_ANG_TO_AMU_ANG_FS2   # (rad/fs)^2
            if omega2 >= 0.0:
                freqs[i] =  np.sqrt( omega2) * FREQ_RAD_FS_TO_WAVENUMBER_CM
            else:
                freqs[i] = -np.sqrt(-omega2) * FREQ_RAD_FS_TO_WAVENUMBER_CM
        return freqs

    # ------------------------------------------------------------------ #
    #  5. Phonon band structure along high-symmetry path                  #
    # ------------------------------------------------------------------ #

    def compute_band_structure(
            self,
            q_path      : List[Tuple[str, List[float]]],
            n_points    : int  = 50,
            coordinates : str  = 'reduced',
            verbose     : bool = True,
    ) -> dict:
        """
        Compute phonon dispersion along a high-symmetry path.

        Parameters
        ----------
        q_path : list of (label, q) tuples defining the path.
                 Example::

                     [('G', [0, 0, 0]),
                      ('X', [0.5, 0, 0]),
                      ('M', [0.5, 0.5, 0]),
                      ('G', [0, 0, 0]),
                      ('R', [0.5, 0.5, 0.5])]

        n_points    : number of q-points per segment (default 50).
        coordinates : ``'reduced'`` (fractional of reciprocal lattice,
                      default) or ``'cartesian'`` [A^-1].

        Returns
        -------
        dict with keys:

        ``q_distances``  : (N_q,) arc length in reciprocal space [A^-1]
        ``frequencies``  : (N_q, 3*N_basis) frequencies [cm^-1]
        ``q_points``     : (N_q, 3) q-vectors in Cartesian [A^-1]
        ``labels``       : list of (position, label) for high-sym points
        ``segment_ends`` : list of arc-length values at segment endpoints
        """
        if self.IFC is None:
            raise RuntimeError(
                "PhononCalculator: call compute_ifc() before "
                "compute_band_structure().")

        # Reciprocal lattice vectors (primitive)
        rec_lat = 2.0 * np.pi * np.linalg.inv(self.lattice_prim).T  # (3,3)

        # Bug-fix note: the rows of rec_lat are the reciprocal vectors, so
        # q_cart = q_frac @ rec_lat.  rec_lat @ q_frac used the columns,
        # which coincides only for a symmetric lattice matrix (the FCC
        # primitive cell) and sent hexagonal paths to the wrong q.
        def to_cartesian(q_frac):
            return np.asarray(q_frac) @ rec_lat

        # Build list of q-points and segment metadata
        q_cart_list : List[np.ndarray] = []
        label_positions: List[Tuple[int, str]] = []
        segment_ends: List[float] = []

        for seg_idx in range(len(q_path) - 1):
            label_s, q_s_red = q_path[seg_idx]
            label_e, q_e_red = q_path[seg_idx + 1]

            if coordinates == 'reduced':
                q_s = to_cartesian(q_s_red)
                q_e = to_cartesian(q_e_red)
            else:
                q_s = np.asarray(q_s_red)
                q_e = np.asarray(q_e_red)

            # Add start label (skip if repeated at same position)
            if seg_idx == 0:
                label_positions.append((len(q_cart_list), label_s))

            # Interpolate n_points between q_s and q_e
            # Exclude endpoint (added by next segment start)
            for k in range(n_points):
                t = k / n_points
                q_cart_list.append(q_s + t * (q_e - q_s))

        # Add the final endpoint
        _, q_e_red = q_path[-1]
        label_e = q_path[-1][0]
        q_e = to_cartesian(q_e_red) if coordinates == 'reduced' else np.asarray(q_e_red)
        q_cart_list.append(q_e)
        label_positions.append((len(q_cart_list) - 1, label_e))

        q_cart_arr = np.array(q_cart_list)   # (N_q, 3)
        N_q = len(q_cart_arr)

        # Arc lengths (distances in reciprocal space)
        q_dist = np.zeros(N_q)
        for i in range(1, N_q):
            q_dist[i] = q_dist[i - 1] + np.linalg.norm(
                q_cart_arr[i] - q_cart_arr[i - 1])

        # Segment end positions for x-axis ticks
        seg_end_idx = [0] + [
            int(round((s + 1) * n_points - 1))
            for s in range(len(q_path) - 2)
        ] + [N_q - 1]
        seg_end_dist = [float(q_dist[i]) for i in seg_end_idx]

        # Evaluate phonon frequencies at each q-point
        n_modes = 3 * self.N_basis
        freq_arr = np.zeros((N_q, n_modes))

        if verbose and _mpi_rank == 0:
            print(f"\n  Phonon band structure: "
                  f"{N_q} q-points x {n_modes} branches ...")

        for i, q in enumerate(q_cart_arr):
            # With NAC, Gamma takes the direction of the path through it
            # (towards the next point; from the previous one at the end),
            # so the LO-TO splitting shows.
            q_dir = None
            if self._nac is not None and np.linalg.norm(q) < 1e-10:
                q_dir = (q_cart_arr[i + 1] - q if i + 1 < N_q
                         else q - q_cart_arr[i - 1])
            D    = self.build_dynamical_matrix(q, q_dir)
            # Bug-fix note: D(q) is complex Hermitian away from
            # time-reversal-invariant q; eigvalsh(D.real) dropped its
            # imaginary part (1.2-4 THz errors on Si at q = (1/4, 0, 0)).
            eigs = np.sort(np.linalg.eigvalsh(D))
            freq_arr[i] = self._eig_to_freq(eigs)

        result = {
            "q_distances"  : q_dist,
            "frequencies"  : freq_arr,
            "q_points"     : q_cart_arr,
            "labels"       : label_positions,
            "segment_ends" : seg_end_dist,
        }
        self._band_result = result

        if verbose and _mpi_rank == 0:
            n_neg = int(np.sum(freq_arr < -1.0))
            print(f"  Done.  "
                  f"freq range: [{freq_arr.min():.1f}, {freq_arr.max():.1f}] cm^-1"
                  + (f"  ({n_neg} imaginary)" if n_neg else ""))

        return result

    # ------------------------------------------------------------------ #
    #  6. Phonon DOS on a uniform q-mesh                                  #
    # ------------------------------------------------------------------ #

    def compute_dos(self,
                    q_mesh     : Union[int, List[int]] = 10,
                    freq_min   : Optional[float] = None,
                    freq_max   : Optional[float] = None,
                    n_dos      : int   = 200,
                    smearing   : float = 5.0,
                    verbose    : bool  = True,
                    ) -> dict:
        """
        Compute the phonon DOS on a Gamma-centred uniform q-mesh.

        Parameters
        ----------
        q_mesh   : int or (3,) int array -- grid divisions along each
                   reciprocal axis.  Default 10 (= 10x10x10).
        freq_min, freq_max : frequency window [cm^-1].  Auto if None.
        n_dos    : number of frequency bins for the histogram.
        smearing : Gaussian smearing width [cm^-1].  Default 5.0.

        Returns
        -------
        dict with keys:

        ``frequencies`` : (n_dos,) frequency axis [cm^-1]
        ``dos``         : (n_dos,) density of states [states/cm^-1]
        ``dos_partial`` : (N_basis, n_dos) atom-projected DOS
        ``n_q_total``   : total number of q-points sampled
        ``eigenvectors``: None (eigenvectors not stored for memory efficiency)
        """
        if self.IFC is None:
            raise RuntimeError(
                "PhononCalculator: call compute_ifc() before compute_dos().")

        if np.isscalar(q_mesh):
            q_mesh = [int(q_mesh)] * 3
        q_mesh = [int(m) for m in q_mesh]

        # Gamma-centred Monkhorst-Pack-like grid
        rec_lat = 2.0 * np.pi * np.linalg.inv(self.lattice_prim).T
        q_points = []
        for i0 in range(q_mesh[0]):
            for i1 in range(q_mesh[1]):
                for i2 in range(q_mesh[2]):
                    q_frac = np.array([
                        (i0 + 0.5) / q_mesh[0] - 0.5,
                        (i1 + 0.5) / q_mesh[1] - 0.5,
                        (i2 + 0.5) / q_mesh[2] - 0.5,
                    ])
                    q_points.append(q_frac @ rec_lat)

        N_q_total = len(q_points)
        n_modes   = 3 * self.N_basis

        if verbose and _mpi_rank == 0:
            print(f"\n  Phonon DOS: {N_q_total} q-points "
                  f"({q_mesh[0]}x{q_mesh[1]}x{q_mesh[2]}) ...")

        # First pass: collect all frequencies to determine the axis range
        # (only frequencies needed for axis; eigenvectors NOT stored -- B-1)
        all_freqs = np.zeros((N_q_total, n_modes))
        for i, q in enumerate(q_points):
            D    = self.build_dynamical_matrix(q)
            eigs = np.linalg.eigvalsh(D)
            all_freqs[i] = self._eig_to_freq(np.sort(eigs))

        # Frequency axis
        fmin = float(freq_min) if freq_min is not None \
               else float(np.min(all_freqs)) - 3.0 * smearing
        fmax = float(freq_max) if freq_max is not None \
               else float(np.max(all_freqs)) + 3.0 * smearing
        freq_axis = np.linspace(fmin, fmax, n_dos)

        # Second pass: accumulate DOS and partial DOS on-the-fly
        # B-1: eigenvectors are used immediately and discarded each q-point;
        # no (N_q, n_modes, n_modes) array is allocated.
        sigma       = smearing
        _norm       = 1.0 / (sigma * np.sqrt(2.0 * np.pi) * N_q_total)
        dos         = np.zeros(n_dos)
        dos_partial = np.zeros((self.N_basis, n_dos))
        _half_inv_sig2 = 0.5 / sigma**2

        for i, q in enumerate(q_points):
            D    = self.build_dynamical_matrix(q)
            eigs, evecs = np.linalg.eigh(D)
            idx         = np.argsort(eigs)
            freqs_q     = self._eig_to_freq(eigs[idx])
            evecs_q     = np.ascontiguousarray(evecs[:, idx])  # (3N, n_modes)

            dos_q, dos_partial_q = _dos_projection_jit(
                freqs_q, evecs_q, freq_axis, _half_inv_sig2, _norm,
                self.N_basis)
            dos         += dos_q
            dos_partial += dos_partial_q

        result = {
            "frequencies" : freq_axis,
            "dos"         : dos,
            "dos_partial" : dos_partial,
            "n_q_total"   : N_q_total,
        }
        self._dos_result = result

        if verbose and _mpi_rank == 0:
            print(f"  Done.  Peak DOS at "
                  f"{freq_axis[np.argmax(dos)]:.1f} cm^-1")

        return result

    # ------------------------------------------------------------------ #
    #  C-1: Debye temperature from acoustic sound velocity               #
    # ------------------------------------------------------------------ #

    def debye_temperature_from_vsound(self,
                                       n_q     : int   = 20,
                                       verbose : bool  = True,
                                       ) -> float:
        """
        Compute the Debye temperature from the slope of acoustic branches
        near Gamma (sound velocity method).

        More accurate than the DOS-cutoff estimate in
        :meth:`ThermalConductivityCalculator._debye_temperature` because
        it uses the actual acoustic group velocity rather than the highest
        phonon frequency.

        The Debye average velocity and temperature are

        .. math::

            v_D = \\left[\\frac{1}{3}\\left(\\frac{1}{v_L^3}
                  + \\frac{2}{v_T^3}\\right)\\right]^{-1/3}

            \\theta_D = \\frac{\\hbar v_D}{k_B}
                        \\left(\\frac{6\\pi^2 n}{V_0}\\right)^{1/3}

        Parameters
        ----------
        n_q     : number of q-points per direction for slope fitting.
        verbose : print results.

        Returns
        -------
        theta_D : float [K]
        """
        if self.IFC is None:
            raise RuntimeError(
                "debye_temperature_from_vsound: call compute_ifc() first.")

        h_J  = 6.62607015e-34
        kB_J = 1.380649e-23
        ang_to_m = 1e-10

        # Bug-fix note (S16, newly found during the port): the original
        # referenced self.lattice here and below, an attribute this class
        # never sets (only self.lattice_prim is ever assigned, in
        # __init__) -- every call to this method raised AttributeError
        # immediately. self.lattice_prim is the correct attribute, as
        # used identically for the reciprocal lattice everywhere else in
        # this class (compute_band_structure, compute_dos).
        b_vecs  = 2.0 * np.pi * np.linalg.inv(self.lattice_prim).T
        b_norms = np.linalg.norm(b_vecs, axis=1)

        sound_velocities = []
        for b_vec, b_norm in zip(b_vecs, b_norms):
            direction = b_vec / b_norm
            q_mags    = np.linspace(0.005, 0.05, n_q) * b_norm
            freqs_along = []
            for qmag in q_mags:
                D = self.build_dynamical_matrix(direction * qmag)
                eigs = np.linalg.eigvalsh(D)
                freqs_along.append(self._eig_to_freq(np.sort(eigs))[:3])

            freqs_along = np.array(freqs_along)
            q_m = q_mags * 1e10                     # A^-1 -> m^-1
            v_dir = []
            for branch in range(3):
                # Bug-fix note: freqs_along is in cm^-1 (_eig_to_freq), not
                # THz; treating it as THz made v and theta_D 33.4x too large.
                omega = freqs_along[:, branch] * _CM_TO_RAD_S
                v_dir.append(abs(np.polyfit(q_m, omega, 1)[0]))
            if min(v_dir) > 1.0:
                sound_velocities.append(sorted(v_dir))

        if not sound_velocities:
            raise RuntimeError("No valid acoustic branches found.")

        # Bug-fix note: the three slopes of ALL directions used to be pooled
        # and sorted, with v_T taken as the mean of the 2nd and 3rd largest
        # -- which are the longitudinal slopes of the other directions, so
        # v_T came out equal to v_L (Tersoff Si: 9011 m/s for both).  Within
        # each direction the largest slope is L and the other two are T.
        sv   = np.array(sound_velocities)          # (n_dir, 3), ascending
        v_L  = float(np.mean(sv[:, 2]))
        v_T  = float(np.mean(sv[:, :2]))
        # Bug-fix note (ADR-0018): the 1/3 was missing, giving v_D and
        # theta_D 3^(-1/3) = 0.69 times the Debye value (Anderson, J. Phys.
        # Chem. Solids 24, 909 (1963), eq. (5): 3/v_D^3 = 1/v_L^3 + 2/v_T^3).
        v_D  = ((1.0 / v_L**3 + 2.0 / v_T**3) / 3.0) ** (-1.0 / 3.0)

        V0_m3   = abs(np.linalg.det(self.lattice_prim)) * ang_to_m**3
        omega_D = v_D * (6.0 * np.pi**2 * self.N_basis / V0_m3) ** (1.0 / 3.0)
        theta_D = h_J / (2.0 * np.pi) * omega_D / kB_J

        if verbose:
            print(f"\n  Debye temperature (v_sound method):")
            print(f"    v_L = {v_L:.0f} m/s,  v_T = {v_T:.0f} m/s")
            print(f"    v_D = {v_D:.0f} m/s,  theta_D = {theta_D:.1f} K")

        return float(theta_D)

    # ------------------------------------------------------------------ #
    #  C-2: Harmonic thermodynamics by a q-mesh mode sum (ADR-0018)       #
    # ------------------------------------------------------------------ #

    def mesh_frequencies(self, q_mesh: Union[int, List[int]] = 20) -> np.ndarray:
        """
        Frequencies [THz] on a Gamma-centred q-mesh (q_frac = i / n_i,
        Gamma included for every n), shape (n_q, 3 N_basis).  Imaginary
        modes are returned negative.  phonopy's is_gamma_center=True grid
        is the same set of points.
        """
        if self.IFC is None:
            raise RuntimeError("PhononCalculator: call compute_ifc() first.")
        if np.isscalar(q_mesh):
            q_mesh = [int(q_mesh)] * 3
        n0, n1, n2 = (int(m) for m in q_mesh)
        rec_lat = 2.0 * np.pi * np.linalg.inv(self.lattice_prim).T
        grid = np.array([[i / n0, j / n1, k / n2] for i in range(n0)
                         for j in range(n1) for k in range(n2)])
        grid -= np.round(grid)
        out = np.empty((len(grid), 3 * self.N_basis))
        for i, qf in enumerate(grid):
            eigs = np.linalg.eigvalsh(self.build_dynamical_matrix(qf @ rec_lat))
            out[i] = self._eig_to_freq(np.sort(eigs)) / _THZ_TO_CM
        return out

    def thermal_properties(self,
                           temperatures : Union[float, List[float]],
                           q_mesh       : Union[int, List[int]] = 20,
                           cutoff_THz   : float = 1e-3,
                           verbose      : bool  = False,
                           ) -> dict:
        """
        Harmonic Helmholtz free energy, entropy, heat capacity and internal
        energy by a direct sum over the modes of a Gamma-centred q-mesh
        (ADR-0018 decision 2; the method of phonopy):

            F = sum_qv [hw/2 + k_B T ln(1 - e^{-hw/k_BT})] / n_q
            U = sum_qv hw [1/2 + n(w)] / n_q,        S = (U - F) / T
            C_v = sum_qv k_B x^2 e^x / (e^x - 1)^2 / n_q,  x = hw / k_BT

        Modes with frequency <= ``cutoff_THz`` (the three acoustic modes
        at Gamma, and imaginary modes) are left out; imaginary ones are
        counted in ``n_imaginary`` and warned about.

        Returns
        -------
        dict: ``temperatures`` [K]; per primitive cell ``F_eV``, ``U_eV``,
        ``S_kB``, ``Cv_kB`` and ``ZPE_eV``; the same per atom (``*_per_atom``);
        phonopy's units per mole of cells ``F_kJ_mol``, ``S_J_K_mol``,
        ``Cv_J_K_mol``; ``theta_D_K`` (Debye temperature that reproduces
        C_v at each T; NaN where C_v is outside (0, 3 N k_B)); ``q_mesh``,
        ``n_imaginary``, ``min_frequency_THz``.
        """
        T = np.atleast_1d(np.asarray(temperatures, dtype=float))
        f = self.mesh_frequencies(q_mesh)
        n_q = f.shape[0]
        n_imag = int(np.sum(f < -cutoff_THz))
        if n_imag and verbose and _mpi_rank == 0:
            print(f"  WARNING: {n_imag} imaginary modes on the mesh "
                  f"(lowest {f.min():.3f} THz) left out of the sums")
        e = f[f > cutoff_THz] * _THZ_TO_EV                 # hw [eV], flat
        zpe = 0.5 * e.sum() / n_q
        F, U, S, Cv = (np.zeros(len(T)) for _ in range(4))
        for k, t in enumerate(T):
            if t <= 0.0:
                F[k] = U[k] = zpe
                continue
            x = e / (_KB_EV * t)
            em = np.expm1(-x)                               # e^{-x} - 1 < 0
            F[k] = zpe + _KB_EV * t * np.sum(np.log(-em)) / n_q
            nb = -np.exp(-x) / em                           # Bose 1/(e^x - 1)
            U[k] = zpe + np.sum(e * nb) / n_q
            S[k] = (U[k] - F[k]) / (_KB_EV * t)
            Cv[k] = np.sum(x * x * nb * (nb + 1.0)) / n_q
        n = self.N_basis
        theta = np.array([_debye_theta_from_cv(c / (3 * n), t)
                          for c, t in zip(Cv, T)])
        res = {
            "temperatures": T, "F_eV": F, "U_eV": U, "S_kB": S, "Cv_kB": Cv,
            "ZPE_eV": zpe,
            "F_eV_per_atom": F / n, "U_eV_per_atom": U / n,
            "S_kB_per_atom": S / n, "Cv_kB_per_atom": Cv / n,
            "F_kJ_mol": F * _EV_TO_KJ_MOL, "S_J_K_mol": S * _R_J_K_MOL,
            "Cv_J_K_mol": Cv * _R_J_K_MOL,
            "theta_D_K": theta,
            "q_mesh": list(np.broadcast_to(q_mesh, 3).astype(int).tolist()
                           if not np.isscalar(q_mesh) else [int(q_mesh)] * 3),
            "n_imaginary": n_imag, "min_frequency_THz": float(f.min()),
            "cutoff_THz": float(cutoff_THz),
        }
        self._thermal_result = res
        return res

    def phonon_free_energy(self,
                            T       : float,
                            n_q     : int   = 20,
                            verbose : bool  = True,
                            ) -> float:
        """
        Helmholtz vibrational free energy F_vib(T) [eV per primitive cell],
        the mode sum of :meth:`thermal_properties` on an n_q^3
        Gamma-centred mesh.  T = 0 gives the zero-point energy.

        Since ADR-0018 this no longer integrates the smeared DOS
        histogram (bin width and smearing errors, worst at low T).

        This is the foundation of the quasi-harmonic approximation:
        F(V, T) = E(V) + F_vib(V, T) at several volumes (the ``qha``
        command, ADR-0018).
        """
        r = self.thermal_properties([T], q_mesh=n_q)
        F_vib_eV = float(r["F_eV"][0])
        if verbose and _mpi_rank == 0:
            print(f"  F_vib(T={T:.0f} K) = {F_vib_eV:.4f} eV "
                  f"(ZPE = {r['ZPE_eV']:.4f} eV)")
        return F_vib_eV

    # ------------------------------------------------------------------ #
    #  7. Output: save results and plot                                    #
    # ------------------------------------------------------------------ #

    def save_results(self,
                     output_dir : str  = ".",
                     verbose    : bool = True) -> None:
        """
        Save IFC, dispersion, and DOS results to files.

        Files written
        -------------
        ifc.npz                   -- raw IFC + supercell geometry
        phonon_dispersion.csv     -- q-distance, frequencies [cm^-1]
        phonon_dos.csv            -- frequency [cm^-1], total DOS, partial DOS
        phonon_dispersion.png     -- dispersion plot (if matplotlib available)
        phonon_dos.png            -- DOS plot (if matplotlib available)

        Only MPI rank 0 writes files; other ranks return immediately.
        """
        if _mpi_rank != 0:
            return

        import os as _os
        _os.makedirs(output_dir, exist_ok=True)

        # IFC
        if self.IFC is not None:
            ifc_path = _os.path.join(output_dir, "ifc.npz")
            np.savez(ifc_path,
                     IFC         = self.IFC,
                     lattice_prim= self.lattice_prim,
                     lattice_sc  = self.lattice_sc,
                     positions   = self.positions,
                     atom_types  = np.array(self.atom_types),
                     pos_sc      = self._pos_sc,
                     sc_to_prim  = np.array(self._sc_to_prim),
                     sc_cell     = np.array(self._sc_cell),
                     masses      = self.masses,
                     supercell   = np.array(self.supercell))
            if verbose:
                print(f"  IFC          -> {ifc_path}")

        # Dispersion CSV
        if self._band_result is not None:
            br   = self._band_result
            csv_path = _os.path.join(output_dir, "phonon_dispersion.csv")
            n_modes  = br["frequencies"].shape[1]
            header   = "q_distance_invAng," + ",".join(
                f"freq_{n}_cm-1" for n in range(n_modes))
            rows = np.column_stack([br["q_distances"], br["frequencies"]])
            np.savetxt(csv_path, rows, delimiter=",", header=header,
                       comments='')
            if verbose:
                print(f"  Dispersion   -> {csv_path}")

        # DOS CSV
        if self._dos_result is not None:
            dr = self._dos_result
            dos_path = _os.path.join(output_dir, "phonon_dos.csv")
            header = "freq_cm-1,dos_states_per_cm-1," + ",".join(
                f"pdos_{sp}" for sp in self.atom_types[:self.N_basis])
            rows = np.column_stack(
                [dr["frequencies"], dr["dos"], dr["dos_partial"].T])
            np.savetxt(dos_path, rows, delimiter=",", header=header,
                       comments='')
            if verbose:
                print(f"  DOS          -> {dos_path}")

        # Plots
        if HAS_MATPLOTLIB:
            if self._band_result is not None:
                self._plot_dispersion(output_dir)
            if self._dos_result is not None:
                self._plot_dos(output_dir)

    def _plot_dispersion(self, output_dir: str) -> None:
        """Plot phonon dispersion with high-symmetry labels."""
        br = self._band_result
        fig, ax = plt.subplots(figsize=(7, 5))

        freqs = br["frequencies"]   # (N_q, n_modes)
        qdist = br["q_distances"]

        # Colour: acoustic branches blue, optical red
        n_modes = freqs.shape[1]
        n_ac    = 3   # acoustic branches (lowest 3)
        for n in range(n_modes):
            colour = 'steelblue' if n < n_ac else 'tomato'
            ax.plot(qdist, freqs[:, n], color=colour, lw=1.2, alpha=0.85)

        # High-symmetry labels and vertical lines
        for pos_idx, label in br["labels"]:
            x = float(qdist[pos_idx])
            ax.axvline(x, color='grey', lw=0.8, ls='--')
            ax.text(x, ax.get_ylim()[0] - 15, label,
                    ha='center', va='top', fontsize=10)

        ax.axhline(0.0, color='grey', lw=0.6, ls=':')
        ax.set_xlim(qdist[0], qdist[-1])
        ax.set_xlabel("Wave vector")
        ax.set_ylabel("Frequency (cm^-1)")
        ax.set_title("Phonon dispersion")
        ax.set_xticks([float(qdist[p]) for p, _ in br["labels"]])
        ax.set_xticklabels([l for _, l in br["labels"]])
        ax.tick_params(axis='x', which='both', bottom=False)
        fig.tight_layout()
        path = os.path.join(output_dir, "phonon_dispersion.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  Dispersion plot -> {path}")

    def _plot_dos(self, output_dir: str) -> None:
        """Plot total and partial phonon DOS."""
        dr  = self._dos_result
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.fill_between(dr["frequencies"], dr["dos"], alpha=0.4,
                        color='steelblue', label='Total')
        ax.plot(dr["frequencies"], dr["dos"], color='steelblue', lw=1.5)
        ax.set_xlabel("Frequency (cm^-1)")
        ax.set_ylabel("DOS (states / cm^-1)")
        ax.set_title("Phonon DOS")
        ax.axvline(0.0, color='grey', lw=0.8, ls=':')
        ax.legend(); fig.tight_layout()
        path = os.path.join(output_dir, "phonon_dos.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  DOS plot        -> {path}")

    # ------------------------------------------------------------------ #
    #  8. Load IFC from saved npz                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def load_ifc(cls, npz_path: str,
                 force_fn: Callable,
                 **kwargs) -> 'PhononCalculator':
        """
        Restore a PhononCalculator from a saved ifc.npz file.

        Allows recomputing the dispersion or DOS without re-running IFC.

        Parameters
        ----------
        npz_path : path to the .npz file saved by save_results().
        force_fn : force callable (needed for the class constructor;
                   not used for IFC---the stored IFC is loaded directly).

        Returns
        -------
        PhononCalculator with IFC already populated.

        Bug-fix note (S11): save_results() did not save atom_types at
        all, so the .get('atom_types', ...) fallback below always used
        its placeholder ['X', ...] default in practice -- the real
        element symbols were silently lost on every save/load round
        trip. save_results() now writes atom_types explicitly; the
        placeholder default here is kept only for loading older/foreign
        .npz files that predate this fix. allow_pickle=True has also
        been dropped: every array saved by save_results() (including the
        now-added atom_types, a plain string array) has a native
        (non-object) dtype and needs no pickling.
        """
        d   = np.load(npz_path)
        obj = cls(
            lattice    = d['lattice_prim'],
            positions  = d['positions'],
            atom_types = list(d['atom_types']) if 'atom_types' in d
                        else ['X'] * len(d['positions']),
            force_fn   = force_fn,
            supercell  = d['supercell'].tolist(),
            **kwargs)
        obj.IFC = d['IFC']
        obj._precompute_dasr()
        return obj
