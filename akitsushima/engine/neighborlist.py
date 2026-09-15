"""Neighbour list: Verlet-skin cell-linked list (CLL) or scipy cKDTree back-end."""
import contextlib
from typing import Optional, Tuple

import numpy as np

from ..utils.jit import _HAS_NUMBA, _numba_njit

try:
    from scipy.spatial import cKDTree as _cKDTree
    _HAS_KDTREE = True
except ImportError:
    _cKDTree = None
    _HAS_KDTREE = False


def _minimum_image(rij: np.ndarray,
                    lattice: np.ndarray,
                    inv_lat: np.ndarray) -> np.ndarray:
    """Apply minimum-image convention to displacement vector rij.

    Uses the row-vector convention consistent with parse_poscar
    (cart = frac @ lattice, where `lattice` has lattice vectors as
    ROWS): frac = cart @ inv_lat, cart = frac @ lattice.
    """
    s = rij @ inv_lat
    s -= np.round(s)
    return s @ lattice


# Largest possible minimum-image distance between two points of the unit
# cube in FRACTIONAL space: every component of `s - round(s)` lies in
# [-1/2, 1/2], so |s| <= sqrt(3 * (1/2)^2) = sqrt(3)/2.  A fractional
# KD-tree query with a radius at or above this therefore already returns
# every pair, and capping at it is exactly lossless (verified against a
# brute-force periodic reference: radii 0.866 and 0.9 both return all
# C(N,2) pairs).
_MAX_FRACTIONAL_MIN_IMAGE_DIST = np.sqrt(3.0) / 2.0


_min_image_suppress_depth = 0


@contextlib.contextmanager
def allow_minimum_image_violation():
    """Suppress the minimum-image precondition for the duration of the block.

    ``BasePairPotential.compute()``'s neighbour-list-free O(N^2) path
    normally refuses to run when ``r_cut`` reaches past half the smallest
    perpendicular cell width, because beyond that limit
    ``s -= np.round(s)`` keeps exactly one image per pair and the other
    images inside the cutoff are never counted -- silently wrong numbers.

    A context manager rather than a parameter, for the same reason
    :func:`akitsushima.utils.mpi.local_only` is one: the potential objects
    are constructed deep inside ``make_force_fn``, ``build_engine``,
    ``load_potential``, ``_build_force_fn_from_yaml`` and the fitting
    registry -- 26 sites across six modules -- so a keyword argument
    would have to be threaded through most of the fitting API to reach
    them.

    Use it only where the absolute physics genuinely does not matter: a
    self-consistency check between two code paths that are affected
    identically, a finite-difference oracle that differentiates the same
    (truncated) energy it validates, or a deliberate comparison against
    the unguarded behaviour.  It is NOT a way to run real physics in a
    cell that cannot host its cutoff.

    Scope: this affects only the O(N^2) fallback.  ``NeighbourList``
    still refuses such a cell outright, because a Verlet or cell-linked
    list cannot even enumerate the missing images -- there is no
    "degraded but deterministic" result to fall back on there.

    Re-entrant: nesting increments and decrements a depth counter, so a
    block that calls code opening another block behaves correctly
    (suppressed until the OUTERMOST block exits).
    """
    global _min_image_suppress_depth
    _min_image_suppress_depth += 1
    try:
        yield
    finally:
        _min_image_suppress_depth -= 1


def _check_min_image(lattice: np.ndarray, r_list: float,
                     caller: str) -> None:
    """Raise unless ``r_list`` is inside the minimum-image validity limit.

    ``r_list`` is whatever radius the caller will actually reach out to --
    ``r_cut + skin_width`` for a Verlet list, plain ``r_cut`` for a direct
    O(N^2) evaluation that keeps no skin.

    The bound is half the smallest PERPENDICULAR cell width
    ``d_i = V / |a_j x a_k|``, not half the shortest lattice vector: for a
    sheared cell the width is smaller than the vector length, so a cell
    can look large enough and still be too thin.  See ``_cell_metrics``.

    Beyond that bound ``s -= np.round(s)`` still returns exactly one image
    per pair, so the other images that are also inside the cutoff are
    simply never counted -- no error, no warning, just wrong numbers.
    """
    if _min_image_suppress_depth > 0:
        return
    perp_widths, _ = _cell_metrics(lattice)
    min_width = float(np.min(perp_widths))
    if r_list >= min_width / 2.0:
        cell_lengths = np.linalg.norm(np.asarray(lattice, dtype=float),
                                      axis=1)
        raise ValueError(
            f"{caller}: cutoff radius ({r_list:.4f} A) must be < half the "
            f"smallest perpendicular cell width ({min_width / 2.0:.4f} A) "
            f"for the minimum-image convention to be valid. Perpendicular "
            f"widths are [{perp_widths[0]:.4f}, {perp_widths[1]:.4f}, "
            f"{perp_widths[2]:.4f}] A for lattice vector lengths "
            f"[{cell_lengths[0]:.4f}, {cell_lengths[1]:.4f}, "
            f"{cell_lengths[2]:.4f}] A -- for a sheared cell the width is "
            "SMALLER than the vector length, so a cell can look large "
            "enough and still be too thin. Use a larger supercell or a "
            "smaller r_cut.")


def wrap_positions_into_cell(positions: np.ndarray,
                            lattice: np.ndarray) -> np.ndarray:
    """Wrap Cartesian ``positions`` into the ``[0, 1)`` fractional cell.

    ``frac -= np.floor(frac)`` on its own is not sufficient.  For a
    fractional coordinate just below zero -- say ``-1e-17``, which is what
    an atom sitting exactly at ``frac = 0`` produces after any downward
    displacement -- the subtraction evaluates ``1.0 + (-1e-17)``, which
    rounds to exactly ``1.0`` in float64.  The atom lands on the far cell
    face rather than inside the cell.

    That matters because ``NeighbourList``'s periodic KD-tree backend hands
    the coordinates to ``scipy.spatial.cKDTree(..., boxsize=...)``, which
    requires every coordinate to be strictly below the box size and
    otherwise raises "Some input data are greater than the size of the
    periodic box."  The first atom of almost every POSCAR sits at
    ``frac = 0``, so this is common rather than exotic.

    Parameters
    ----------
    positions : (N, 3) Cartesian coordinates [A].
    lattice   : (3, 3) row-vector lattice [A].

    Returns
    -------
    (N, 3) Cartesian coordinates, every fractional component in ``[0, 1)``.
    """
    frac = positions @ np.linalg.inv(lattice)
    frac -= np.floor(frac)
    # Catch the components that rounded up to exactly 1.0 (and any that
    # floating-point error pushed marginally past it).
    frac[frac >= 1.0] = 0.0
    return frac @ lattice


def _cell_metrics(lattice: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Geometric reach limits of a periodic cell [A].

    Every site in this module that must decide "how far does a periodic
    search have to reach?" needs one of these two numbers, so they are
    derived once here rather than re-invented per call site.

    Parameters
    ----------
    lattice : (3, 3) array
        Lattice vectors as ROWS ``a_0, a_1, a_2`` -- the row-vector
        convention used throughout this package (``cart = frac @ lattice``,
        ``frac = cart @ inv(lattice)``).

    Returns
    -------
    perp_widths : (3,) float64
        ``d_i``, the perpendicular width (interplanar spacing) of the cell
        along axis ``i`` [A].
    sigma_min : float
        Smallest singular value of ``lattice`` [A].

    Perpendicular width -- derivation
    ---------------------------------
    The family of lattice planes normal to axis ``i`` is spanned by the
    other two lattice vectors ``a_j`` and ``a_k``, so its normal direction
    is ``n_i = a_j x a_k``.  Successive planes are separated by the
    component of ``a_i`` along that normal:

        d_i = |a_i . n_i| / |n_i| = V / |a_j x a_k|,    V = |det(lattice)|

    using the scalar triple product ``a_i . (a_j x a_k) = +/- det(lattice)``.
    Equivalently ``d_i = 1 / |b_i|``, where ``b_i`` is the i-th COLUMN of
    ``inv(lattice)`` (the reciprocal vector without the 2*pi factor),
    because ``a_i . b_i = 1`` while ``b_i`` is perpendicular to both
    ``a_j`` and ``a_k``.

    Writing ``theta`` for the angle between ``a_i`` and ``n_i``,

        d_i = ||a_i|| * |cos(theta)| <= ||a_i||

    with equality if and only if ``a_i`` is parallel to ``n_i``, i.e. only
    when ``a_i`` is perpendicular to both other lattice vectors.  So the
    perpendicular width coincides with the vector length exactly for a cell
    with mutually perpendicular vectors, and is STRICTLY SMALLER for any
    sheared cell.  ``d_i`` -- never ``||a_i||`` -- is therefore the quantity
    that limits how far a periodic search may reach; see the bug-fix note
    on :meth:`NeighbourList._check_min_cell`.

    sigma_min -- derivation
    -----------------------
    Converting a Cartesian displacement to fractional coordinates,
    ``s = d @ inv(lattice)``, stretches its length by at most the spectral
    norm of the inverse:

        |s| <= |d| * ||inv(lattice)||_2 = |d| / sigma_min(lattice)

    and that bound is attained (by the corresponding singular vector), so
    ``sigma_min`` is the TIGHT denominator for converting a Cartesian
    search radius into a fractional one -- see
    :meth:`NeighbourList._build_kdtree`.

    Ordering
    --------
    For every cell ``sigma_min <= min_i d_i <= min_i ||a_i||`` (the second
    inequality from the ``|cos(theta)| <= 1`` argument above; the first
    because ``sigma_min = min_{|x|=1} |x @ lattice|``, and choosing ``x``
    along the normal ``n_i`` gives exactly ``d_i``).  Verified numerically
    on 2000 random lattices: 0 violations.  Each criterion built on a term
    further to the RIGHT of this chain is therefore too permissive.
    """
    lat = np.asarray(lattice, dtype=float)
    if lat.shape != (3, 3):
        raise ValueError(
            f"NeighbourList: lattice must have shape (3, 3), got {lat.shape}.")

    volume = abs(float(np.linalg.det(lat)))
    cross_norms = np.array(
        [np.linalg.norm(np.cross(lat[(i + 1) % 3], lat[(i + 2) % 3]))
         for i in range(3)])

    # A degenerate (zero / collinear / coplanar) cell has no well-defined
    # width.  Guard explicitly rather than dividing: V / 0 gives inf, and
    # for a zero lattice 0 / 0 gives NaN -- and EVERY comparison against
    # NaN is False, so a NaN width would make the minimum-image check in
    # _check_min_cell silently ACCEPT the cell.  That is precisely the
    # silent-wrong-answer mode this validation exists to prevent, so fail
    # loudly instead.
    if volume <= 0.0 or not np.all(cross_norms > 0.0):
        raise ValueError(
            f"NeighbourList: degenerate (singular) lattice with volume "
            f"{volume:.6g} A^3 -- the three lattice vectors must be linearly "
            "independent for a minimum-image convention to exist.")

    perp_widths = volume / cross_norms
    sigma_min = float(np.linalg.svd(lat, compute_uv=False)[-1])
    return perp_widths, sigma_min


def _axis_offsets(nc_axis: int) -> tuple:
    """
    Canonical, non-redundant neighbour-cell offsets for one axis.

    A naive 27-cell stencil tries da in {-1, 0, +1} on every axis, but when
    a cell dimension is small (``nc_axis`` cells along that axis), two or
    more of those offsets wrap (modulo nc_axis) onto the *same* physical
    cell:

    - nc_axis == 1: da in {-1, 0, +1} all map to the single cell 0.
    - nc_axis == 2: da in {-1, +1} both map to the *other* cell (only da=0
      and one nonzero offset are needed to reach both of the 2 cells).
    - nc_axis >= 3: da in {-1, 0, +1} map to 3 distinct cells -- no
      collision.

    Since re-visiting the same physical cell via a different offset always
    finds the exact same minimum-image displacement for a given pair (the
    minimum-image reduction only depends on the raw positions, not on
    which offset guess led there), visiting it more than once would only
    ever produce an exact duplicate -- never a distinct, legitimate
    periodic image. Returning the minimal offset set per axis avoids that
    duplication *by construction*, so no runtime deduplication (e.g. a
    Python ``set``) is needed downstream -- which in turn makes the pair
    search loop compatible with ``numba`` nopython mode.
    """
    if nc_axis == 1:
        return (0,)
    if nc_axis == 2:
        return (0, 1)
    return (-1, 0, 1)


def _build_offset_table(nc0: int, nc1: int, nc2: int) -> np.ndarray:
    """Cartesian product of the three per-axis canonical offsets, as an
    (K, 3) int64 array (K <= 27)."""
    offs = [(da, db, dc)
            for da in _axis_offsets(nc0)
            for db in _axis_offsets(nc1)
            for dc in _axis_offsets(nc2)]
    return np.array(offs, dtype=np.int64)


def _cell_linked_pairs_numpy(N: int, nc0: int, nc1: int, nc2: int,
                              cell_idx: np.ndarray,
                              offsets: np.ndarray,
                              head: np.ndarray, nxt: np.ndarray,
                              positions: np.ndarray,
                              lattice: np.ndarray, inv_lat: np.ndarray,
                              rl2: float,
                              out_i: np.ndarray, out_j: np.ndarray,
                              out_rij: np.ndarray, out_r: np.ndarray,
                              count_only: bool) -> int:
    """
    Cell-linked-list pair search.

    Called twice by :meth:`NeighbourList._build_cell_linked`: first with
    ``count_only=True`` (``out_*`` are size-0 placeholder arrays, just used
    to get an exact pair count so the real output arrays can be allocated
    with no wasted/overflowing space), then with ``count_only=False`` and
    correctly-sized ``out_*`` arrays to fill in.

    Because ``offsets`` already contains each physically-distinct
    neighbour cell exactly once (see :func:`_axis_offsets`), and pairs are
    only recorded when ``j > i`` (each unordered pair is generated once,
    from the lower-index atom), no deduplication set is needed here.
    """
    n_offsets = offsets.shape[0]
    n_found = 0
    for i in range(N):
        ci0 = cell_idx[i, 0]
        ci1 = cell_idx[i, 1]
        ci2 = cell_idx[i, 2]
        for o in range(n_offsets):
            cj0 = (ci0 + offsets[o, 0]) % nc0
            cj1 = (ci1 + offsets[o, 1]) % nc1
            cj2 = (ci2 + offsets[o, 2]) % nc2
            cj_flat = cj0 * nc1 * nc2 + cj1 * nc2 + cj2
            j = head[cj_flat]
            while j >= 0:
                if j > i:
                    rx = positions[j, 0] - positions[i, 0]
                    ry = positions[j, 1] - positions[i, 1]
                    rz = positions[j, 2] - positions[i, 2]
                    sx = rx * inv_lat[0, 0] + ry * inv_lat[1, 0] + rz * inv_lat[2, 0]
                    sy = rx * inv_lat[0, 1] + ry * inv_lat[1, 1] + rz * inv_lat[2, 1]
                    sz = rx * inv_lat[0, 2] + ry * inv_lat[1, 2] + rz * inv_lat[2, 2]
                    sx -= round(sx)
                    sy -= round(sy)
                    sz -= round(sz)
                    dx = sx * lattice[0, 0] + sy * lattice[1, 0] + sz * lattice[2, 0]
                    dy = sx * lattice[0, 1] + sy * lattice[1, 1] + sz * lattice[2, 1]
                    dz = sx * lattice[0, 2] + sy * lattice[1, 2] + sz * lattice[2, 2]
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 < rl2 and r2 > 1e-20:
                        if not count_only:
                            out_i[n_found] = i
                            out_j[n_found] = j
                            out_rij[n_found, 0] = dx
                            out_rij[n_found, 1] = dy
                            out_rij[n_found, 2] = dz
                            out_r[n_found] = np.sqrt(r2)
                        n_found += 1
                j = nxt[j]
    return n_found


_cell_linked_pairs_jit = _numba_njit(_cell_linked_pairs_numpy)


class NeighbourList:
    """
    Combined Verlet skin neighbour list with two build back-ends.

    Back-end selection ('backend' parameter)
    -----------------------------------------
    'auto'    -- Use scipy.spatial.cKDTree when available (typically 5--20x
                faster than CLL for N > 500); fall back to CLL otherwise.
    'kdtree'  -- Force cKDTree.  For orthogonal cells the native boxsize PBC
                is used.  For non-orthogonal cells, atoms are transformed to
                fractional coordinates where the KD-tree runs in a unit cube,
                and results are mapped back to Cartesian space.
    'cll'     -- Force the cell-linked list (always available; numba-JIT
                accelerated when numba is installed).

    Parameters
    ----------
    r_cut       : interaction cutoff [A].
    skin_width  : Verlet skin [A]; list rebuilt when any atom moves > skin/2.
    backend     : 'auto' | 'kdtree' | 'cll'
    use_numba   : JIT-compile the cell-linked-list pair search (ignored for
                  the kdtree back-end, which is already scipy-compiled).

    Internal storage
    ----------------
    Pairs are stored in four parallel NumPy arrays instead of a Python list
    of tuples, eliminating per-tuple object overhead:

        _pair_i   : (P,)   int32   -- first atom index
        _pair_j   : (P,)   int32   -- second atom index
        _pair_rij : (P, 3) float64 -- Cartesian displacement r_j - r_i [A]
        _pair_r   : (P,)   float64 -- distance |r_j - r_i| [A]
        _n_pairs  : int            -- number of valid pairs in the arrays

    get_pairs_arrays() returns these arrays directly for vectorised consumers.
    get_pairs() still works and returns a list for backwards compatibility.

    Public interface
    ----------------
    build(lattice, positions)               -- force rebuild.
    check_and_rebuild(lattice, positions)   -- rebuild if needed; return bool.
    get_pairs() -> list[(i,j,dr,r)]         -- pairs within r_cut (compat).
    get_pairs_arrays() -> (i, j, rij, r)    -- structured NumPy arrays.
    rebuild_count / step_count             -- diagnostic counters.
    """

    _HAS_KDTREE = _HAS_KDTREE

    def __init__(self, r_cut: float, skin_width: float = 0.5,
                 backend: str = 'auto', use_numba: bool = False):
        if r_cut <= 0.0:
            raise ValueError(f"NeighbourList: r_cut must be positive, got {r_cut}.")
        self.r_cut = r_cut
        self.skin_width = skin_width
        self.r_list = r_cut + skin_width
        self.backend = backend
        self.use_numba = use_numba

        self._lattice: Optional[np.ndarray] = None
        self._ref_pos: Optional[np.ndarray] = None

        self._pair_i: np.ndarray = np.empty(0, dtype=np.int32)
        self._pair_j: np.ndarray = np.empty(0, dtype=np.int32)
        self._pair_rij: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self._pair_r: np.ndarray = np.empty(0, dtype=np.float64)
        self._n_pairs: int = 0

        self._n_builds: int = 0
        self._n_steps: int = 0

        # Lazy-build flag: set True when geometry changes externally
        # (GCMC insert/delete, manual cell change) so the next
        # check_and_rebuild call forces a full rebuild unconditionally.
        self._dirty: bool = False

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def _check_min_cell(self, lattice: np.ndarray) -> None:
        """
        Validate the minimum-image assumption ``r_list < min_i(d_i) / 2``,
        where ``d_i`` is the PERPENDICULAR WIDTH of the cell along axis
        ``i`` (see :func:`_cell_metrics`).

        Why the perpendicular width is the right bound
        ----------------------------------------------
        :func:`_minimum_image` reduces a displacement with
        ``s -= np.round(s)``, i.e. it returns the periodic image whose
        FRACTIONAL components all lie in ``[-1/2, 1/2)``.  For a sheared
        cell that fractional box is not the Wigner-Seitz cell, so the
        reduction is not unconditionally the true nearest image.  It is
        the true nearest image exactly when the nearest image is
        guaranteed to fall inside that box, and the clean sufficient
        condition for that is ``r < min_i(d_i) / 2``:

        writing ``b_i`` for the i-th column of ``inv(lattice)``, a
        displacement ``d`` has fractional components
        ``s_i = d . b_i``, so

            |s_i| <= |d| * |b_i| = |d| / d_i <= |d| / min_i(d_i).

        Hence any pair with ``|d| < min_i(d_i) / 2`` satisfies
        ``|s_i| < 1/2`` for every ``i`` -- it lies inside the reduction box
        and ``round`` finds it.  Conversely, once ``r_list`` reaches
        ``min_i(d_i) / 2`` a second image can enter range and the
        convention breaks down.

        Bug-fix note (lattice length vs perpendicular width)
        ---------------------------------------------------
        This check previously used the LATTICE VECTOR LENGTHS,

            OLD:  r_list  <  min_i ||a_i|| / 2
            NEW:  r_list  <  min_i   d_i   / 2

        Because ``d_i = ||a_i|| * |cos(theta_i)| <= ||a_i||`` (derivation in
        :func:`_cell_metrics`), the old criterion is never stricter and is
        usually LOOSER, so it ACCEPTED cells for which minimum image was
        already invalid.  Nothing raised and nothing warned: the neighbour
        list simply returned wrong pairs, and every energy, force and
        stress built on it was silently wrong.  This is the same
        silent-wrong-answer failure mode as the already-fixed "Bug G" and
        "Bug L".

        The two criteria COINCIDE exactly for a cell whose lattice vectors
        are mutually perpendicular (``cos(theta_i) = 1``), which is why an
        orthogonal-cell-only test suite could not see the difference --
        and, at the time of this fix, all 725 distinct lattices reaching
        this method across the entire test suite were orthogonal, so the
        defect had zero test coverage.

        Worked example -- hexagonal (120 deg) cell
        ------------------------------------------
        ``a_0 = (a, 0, 0)``, ``a_1 = (-a/2, a*sqrt(3)/2, 0)``,
        ``a_2 = (0, 0, c)``.  Both in-plane vectors have length ``a``, but

            d_0 = d_1 = a * sqrt(3) / 2 ~= 0.866 * a,

        so the old check was ~15% too lenient there.  For an HCP Mg
        4x4x3 supercell (a = 3.21 A, c = 5.21 A): ``min ||a_i|| / 2 =
        6.42 A`` but ``min d_i / 2 = 5.56 A``.  A run with
        ``r_cut + skin_width = 6.0 A`` therefore passed the old check while
        minimum image was already invalid.  The shear can be made
        arbitrarily large, so the gap widens without bound; the hexagonal
        case is merely the most common one in materials science.
        """
        perp_widths, _ = _cell_metrics(lattice)
        min_width = float(np.min(perp_widths))
        if self.r_list >= min_width / 2.0:
            cell_lengths = np.linalg.norm(np.asarray(lattice, dtype=float),
                                           axis=1)
            raise ValueError(
                f"NeighbourList: r_cut + skin_width ({self.r_list:.4f} A) must be "
                f"< half the smallest perpendicular cell width "
                f"({min_width / 2.0:.4f} A) for the minimum-image convention to "
                f"be valid. Perpendicular widths are "
                f"[{perp_widths[0]:.4f}, {perp_widths[1]:.4f}, "
                f"{perp_widths[2]:.4f}] A for lattice vector lengths "
                f"[{cell_lengths[0]:.4f}, {cell_lengths[1]:.4f}, "
                f"{cell_lengths[2]:.4f}] A -- for a sheared cell the width is "
                "SMALLER than the vector length, so a cell can look large "
                "enough and still be too thin. Use a larger supercell or a "
                "smaller r_cut/skin_width.")

    def build(self, lattice: np.ndarray, positions: np.ndarray) -> None:
        """Force-rebuild the neighbour list."""
        self._check_min_cell(lattice)
        self._lattice = lattice.copy()
        self._ref_pos = positions.copy()
        use_kd = (self._HAS_KDTREE and self.backend in ('auto', 'kdtree'))
        if use_kd:
            self._build_kdtree(lattice, positions)
        else:
            self._build_cell_linked(lattice, positions)
        self._n_builds += 1

    def mark_dirty(self) -> None:
        """
        Mark the neighbour list as stale.

        Call whenever the geometry changes outside the normal MD / GO
        step cycle (e.g. GCMC insertion / deletion, manual position
        reset, or cell rescaling).  The next ``check_and_rebuild`` call
        will force a full rebuild regardless of the displacement criterion.
        """
        self._dirty = True

    def check_and_rebuild(self,
                           lattice: np.ndarray,
                           positions: np.ndarray) -> bool:
        """Rebuild if needed; return True when rebuilt.

        The stored list holds every pair within ``r_list = r_cut + skin``
        as measured at the last build.  It stays valid as long as no pair
        that was OUTSIDE ``r_list`` then can be inside ``r_cut`` now.

        Two things move a pair between builds.  The cell itself may
        deform: with the row-vector convention a point transforms as
        ``r' = r @ M`` with ``M = inv(h_build) @ h_now``, so any
        separation is scaled by at least ``sigma_min(M)``.  And the atoms
        may move non-affinely, by at most ``d_max`` each, which can close
        a gap by at most ``2 d_max``.  So the list is still complete while

            sigma_min(M) * r_list - 2 * d_max > r_cut

        Two earlier criteria are replaced by that one.

        Any lattice change at all used to force a rebuild, which under
        NPT -- where the cell changes every single step -- meant
        rebuilding on 99.4% of steps and buying nothing from the skin.

        And ``d_max`` was measured as ``max|positions - ref|`` in raw
        Cartesian coordinates, while the MD and relax loops PBC-wrap the
        positions every step.  One atom oscillating around a cell face
        therefore appeared to jump a whole lattice vector each step: a
        single such atom took an NVE run from 43 rebuilds in 501 steps
        to 501, an 11.6x slowdown in list construction.  The non-affine
        displacement is now minimum-imaged, which is exact here because
        a genuine between-build displacement is far below half the cell
        width -- that is precisely what the skin bounds.
        """
        self._n_steps += 1
        if self._dirty or self._ref_pos is None:
            self.build(lattice, positions)
            self._dirty = False
            return True

        # Affine part: how much the cell deformation can shrink a
        # separation.  sigma_min is the smallest singular value of M.
        if np.allclose(lattice, self._lattice, atol=1e-12):
            sigma_min = 1.0
        else:
            M = np.linalg.inv(self._lattice) @ lattice
            sigma_min = float(np.linalg.svd(M, compute_uv=False)[-1])

        # Non-affine part: displacement relative to where the affine map
        # alone would have put each atom, minimum-imaged in the current
        # cell so a PBC wrap does not register as motion.
        ref_affine = self._ref_pos @ (self._lattice_map(lattice))
        dr = positions - ref_affine
        frac = dr @ np.linalg.inv(lattice)
        frac -= np.round(frac)
        d_max = float(np.max(np.linalg.norm(frac @ lattice, axis=1)))

        if sigma_min * self.r_list - 2.0 * d_max <= self.r_cut:
            self.build(lattice, positions)
            return True
        return False

    def _lattice_map(self, lattice: np.ndarray) -> np.ndarray:
        """``M`` such that a build-frame point maps to ``r @ M`` now."""
        return np.linalg.inv(self._lattice) @ lattice

    def get_candidate_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the CANDIDATE pair indices -- every pair within ``r_list``
        (= ``r_cut + skin_width``) as of the last :meth:`build`.

        Returns
        -------
        idx_i : (P,) int32 -- first atom indices
        idx_j : (P,) int32 -- second atom indices

        This is the correct accessor for any energy/force/stress evaluator.
        It deliberately returns ONLY indices, never distances: the caller
        MUST recompute ``rij``/``r`` from the positions it is currently
        evaluating, and apply its OWN cutoff mask to those fresh distances.

        Why this is the whole point of the Verlet skin
        ----------------------------------------------
        :meth:`check_and_rebuild` declines to rebuild while every atom has
        moved at most ``skin_width / 2`` from its reference position. Two
        atoms can each move that far in opposite directions, so a pair's
        separation can change by at most ``2 * (skin_width / 2) =
        skin_width`` between rebuilds. Therefore any pair that is within
        ``r_cut`` NOW was within ``r_cut + skin_width = r_list`` at build
        time, and so is guaranteed to appear in this candidate list. That
        guarantee is what makes it safe to skip the O(N^2) rescan -- but it
        only holds for the FULL ``r_list``-range candidate set. Pre-masking
        the candidates down to ``r_cut`` destroys it (see the bug-fix note
        on :meth:`get_pairs_arrays`).

        Cost: O(P) for the caller's distance recomputation, where P is the
        number of listed candidate pairs -- not O(N^2). The neighbour
        list's performance benefit is fully preserved; only the cached
        distances, which were never safe to reuse, are given up.
        """
        return (self._pair_i[:self._n_pairs],
                self._pair_j[:self._n_pairs])

    def get_pairs_arrays(self) -> Tuple[np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray]:
        """
        Return the structured pair arrays for the pairs that were within
        ``r_cut`` AT THE TIME OF THE LAST :meth:`build`.

        Returns
        -------
        idx_i  : (P,)   int32   -- first atom indices
        idx_j  : (P,)   int32   -- second atom indices
        rij    : (P, 3) float64 -- Cartesian displacements r_j - r_i [A]
        r      : (P,)   float64 -- distances [A]

        WARNING -- build-time snapshot, not current geometry
        ---------------------------------------------------
        ``rij``/``r`` are the values cached by :meth:`build`, and the
        ``<= r_cut`` mask is applied to those same cached values. Both are
        therefore only valid for the exact positions that were passed to
        :meth:`build`. This accessor is safe ONLY for a consumer that
        rebuilds immediately beforehand and never advances the geometry in
        between (``solvers/rdf.py`` is the one such caller in this package;
        it also uses ``skin_width=0.0``, so for it ``r_list == r_cut`` and
        the mask below is a no-op).

        Any consumer that evaluates a potential at positions which may have
        moved since the last build MUST use :meth:`get_candidate_pairs`
        instead and recompute distances itself.

        Bug-fix note (stale-distance / defeated-skin defect)
        ---------------------------------------------------
        Every interatomic potential in this package used to call this
        method, which produced silently wrong energies, forces and stresses
        during normal MD. The failure had two compounding halves:

        1. ``BasePairPotential._build_pair_groups`` and
           ``coulomb.ewald.real_space_coulomb`` consumed the cached
           ``rij``/``r`` directly, so with ``skin_width > 0`` -- the
           DEFAULT, and thus the normal path -- they evaluated the geometry
           as it was at the last rebuild rather than as it is now. Measured
           on a 108-atom LJ cell after displacing atoms by 0.4 A (< skin/2,
           so no rebuild): energy off by 34.6 eV and forces off by
           13.9 eV/A against a max force of 13.8 eV/A, i.e. an error the
           size of the signal, with the result reproducing the OLD
           positions bit-for-bit.
        2. The ``<= r_cut`` mask below was applied to those build-time
           distances. ``_store_pairs`` deliberately retains candidates out
           to ``r_list = r_cut + skin_width``, but masking them down to
           ``r_cut`` with build-time values discards exactly the pairs the
           skin exists to preserve: a pair sitting between ``r_cut`` and
           ``r_cut + skin`` at build time was dropped permanently, so when
           it later migrated INSIDE ``r_cut`` its interaction was never
           counted. This half also affected the potentials that already
           recomputed their own distances (EAM, Stillinger-Weber, Tersoff),
           because the pair was missing from the index list they started
           from. Net effect: the Verlet skin provided no benefit whatever
           and actively injected error, and larger skins were WORSE.

        End-to-end, this broke energy conservation outright: an NVE run
        drifted 2.8e-01 eV at the default ``skin_width=0.5`` versus
        5.1e-06 eV at ``skin_width=0.0`` (where the list is rebuilt every
        step and distances are always fresh) -- a factor of 5.5e4 -- and an
        NVT run at 600 K diverged to a mean temperature of 5.8e+51 K at
        ``skin_width=0.5`` while remaining stable at 548.9 K with an
        otherwise identical ``skin_width=0.0`` setup.

        The fix keeps this method's contract exactly as it always was, for
        ``rdf.py`` and any external caller, and adds
        :meth:`get_candidate_pairs` as the accessor every potential now
        uses. Returning only indices from that method makes it
        structurally impossible to reintroduce half 1: an accessor that
        never hands out a cached distance cannot have a cached distance
        consumed by mistake.
        """
        rc = self.r_cut
        mask = self._pair_r[:self._n_pairs] <= rc
        return (self._pair_i[:self._n_pairs][mask],
                self._pair_j[:self._n_pairs][mask],
                self._pair_rij[:self._n_pairs][mask],
                self._pair_r[:self._n_pairs][mask])

    def get_pairs(self) -> list:
        """Return pairs within r_cut as a list of (i, j, rij, r) tuples.

        Kept for backward compatibility with existing callers.
        Prefer get_pairs_arrays() for performance-critical paths.
        """
        idx_i, idx_j, rij, r = self.get_pairs_arrays()
        return list(zip(idx_i.tolist(), idx_j.tolist(),
                         [rij[k] for k in range(len(r))],
                         r.tolist()))

    @property
    def rebuild_count(self) -> int:
        return self._n_builds

    @property
    def step_count(self) -> int:
        return self._n_steps

    # ------------------------------------------------------------------
    # Internal: store pairs into structured arrays
    # ------------------------------------------------------------------

    def _store_pairs(self, raw_pairs: list) -> None:
        """Convert a list of (i, j, rij, r) into structured arrays."""
        P = len(raw_pairs)
        self._pair_i = np.empty(P, dtype=np.int32)
        self._pair_j = np.empty(P, dtype=np.int32)
        self._pair_rij = np.empty((P, 3), dtype=np.float64)
        self._pair_r = np.empty(P, dtype=np.float64)
        for k, (i, j, rij, r) in enumerate(raw_pairs):
            self._pair_i[k] = i
            self._pair_j[k] = j
            self._pair_rij[k] = rij
            self._pair_r[k] = r
        self._n_pairs = P

    # ------------------------------------------------------------------
    # Back-end A: scipy cKDTree  (fast; O(N log N) build, O(N) query)
    # ------------------------------------------------------------------

    def _build_kdtree(self,
                       lattice: np.ndarray,
                       positions: np.ndarray) -> None:
        """
        Build neighbour pairs using scipy.spatial.cKDTree.

        Orthogonal cells  : use cKDTree(boxsize=L) for native PBC.
        Non-orthogonal    : transform to fractional coords, run KD-tree in
                            the unit cube [0,1)^3 (boxsize=1.0), then convert
                            found pairs back to Cartesian minimum-image.
        """
        r_list = self.r_list
        inv_lat = np.linalg.inv(lattice)

        # Detect orthogonality (off-diagonal elements < 1e-8)
        off = lattice - np.diag(np.diag(lattice))
        is_ortho = np.max(np.abs(off)) < 1e-8

        if is_ortho:
            box = np.diag(lattice)              # [Lx, Ly, Lz]
            pos_w = positions % box               # wrap to [0, box)
            # `%` alone does not guarantee the half-open interval: for a
            # coordinate just below zero (an atom at exactly 0 after any
            # downward displacement gives ~-1e-38) the sum `x + box`
            # rounds to exactly `box` in float64, and cKDTree rejects a
            # coordinate equal to boxsize with "Some input data are
            # greater than the size of the periodic box."
            pos_w[pos_w >= box] = 0.0
            tree = _cKDTree(pos_w, boxsize=box)
            raw = tree.query_pairs(r_list, output_type='ndarray')  # (P,2)

            result = []
            for row in raw:
                i, j = int(row[0]), int(row[1])
                rij = pos_w[j] - pos_w[i]
                rij -= box * np.round(rij / box)   # minimum image
                r = float(np.linalg.norm(rij))
                if r > 1e-12:
                    result.append((i, j, rij, r))
        else:
            # Fractional-space KD-tree
            frac = positions @ inv_lat
            frac -= np.floor(frac)                 # wrap to [0,1)
            frac[frac >= 1.0] = 0.0                # see the note above
            tree = _cKDTree(frac, boxsize=1.0)

            # Fractional query radius.
            #
            # `s = d @ inv_lat` stretches a length by at most
            # 1 / sigma_min(lattice) and that bound is tight (see
            # _cell_metrics), so every pair within `r_list` in Cartesian
            # space lies within `r_list / sigma_min` in fractional space.
            #
            # Bug-fix note (lattice length vs sigma_min, and a lossy clamp)
            # ------------------------------------------------------------
            # This used `r_list / min_i ||a_i|| + 0.01` and then clamped
            # the result with `min(..., 0.5)`.  Both halves silently LOST
            # pairs on a sheared cell:
            #   1. sigma_min <= min_i ||a_i|| always, so the old
            #      denominator was too LARGE and the query radius too
            #      SMALL -- the tree was simply not asked for far enough.
            #      Measured on an HCP Mg 4x4x3 cell (a=3.21, c=5.21) at
            #      r_cut=5.0, skin=0: 186 of 864 true pairs missing
            #      (21.5%), giving an LJ energy error of 2.9 meV/atom and
            #      max|dF| = 1.2e-2 eV/A against the brute-force
            #      minimum-image reference.
            #   2. the `0.5` cap truncated the radius further with no
            #      error and no warning.  It was not protecting against a
            #      scipy limitation: cKDTree(boxsize=1.0).query_pairs was
            #      verified correct against a brute-force periodic
            #      reference at radii 0.55, 0.6, 0.866 and 0.9.  The only
            #      lossless cap is sqrt(3)/2, the largest minimum-image
            #      distance the unit cube admits at all.
            # The tiny relative epsilon guards float round-off exactly at
            # the boundary; over-searching is harmless because every
            # candidate is re-checked against `r_list` in Cartesian space
            # below, whereas under-searching loses pairs irrecoverably.
            _, sigma_min = _cell_metrics(lattice)
            r_frac = min(r_list / sigma_min * (1.0 + 1e-9),
                          _MAX_FRACTIONAL_MIN_IMAGE_DIST)
            raw = tree.query_pairs(r_frac, output_type='ndarray')

            result = []
            for row in raw:
                i, j = int(row[0]), int(row[1])
                rij = positions[j] - positions[i]
                s = rij @ inv_lat
                s -= np.round(s)
                rij = s @ lattice
                r = float(np.linalg.norm(rij))
                if r <= r_list and r > 1e-12:
                    result.append((i, j, rij, r))

        self._store_pairs(result)

    # ------------------------------------------------------------------
    # Back-end B: cell-linked list (always available; numba-accelerated)
    # ------------------------------------------------------------------

    def _build_cell_linked(self,
                            lattice: np.ndarray,
                            positions: np.ndarray) -> None:
        """O(N) neighbour list via cell-linked lists."""
        N = len(positions)
        r_list = self.r_list
        inv_lat = np.linalg.inv(lattice)

        # Sub-cell counts: each sub-cell must be >= r_list wide
        # PERPENDICULAR to its own faces.
        #
        # Sub-cell k along axis i spans fractional s_i in
        # [k/nc_i, (k+1)/nc_i).  With the canonical +/-1 stencil (see
        # _axis_offsets) two atoms whose sub-cell indices differ by 2 or
        # more (mod nc_i) are NEVER examined, and the smallest fractional
        # separation such a pair can have is 1/nc_i.  Since a displacement
        # d = sum_l ds_l a_l projects onto the axis-i plane normal as
        # |d| >= |ds_i| * d_i, the closest an unexamined pair can be is
        # exactly one sub-cell width w_i = d_i / nc_i.  Full coverage out
        # to r_list therefore requires w_i >= r_list, i.e.
        # nc_i <= d_i / r_list.
        #
        # Bug-fix note (lattice length vs perpendicular width)
        # ----------------------------------------------------
        # This used `floor(||a_i|| / r_list)`.  Since d_i <= ||a_i||, that
        # OVER-counts the sub-cells and makes them thinner than r_list, so
        # the stencil no longer reaches r_list and true pairs are dropped
        # with no error or warning.  The defect only bites for nc_i >= 4,
        # because for nc_i <= 3 the offsets {-1, 0, +1} already cover every
        # sub-cell along that axis -- which is why the small-cell tests
        # could not see it.  Measured on a hexagonal Mg 6x6x3 cell
        # (||a|| = 19.26 A, d = 16.68 A) with 250 atoms at r_list = 4.8 A,
        # where minimum image is perfectly VALID (4.8 < d/2 = 7.82):
        # nc_old = [4, 4, 3] gives w_old = 4.17 A < r_list and drops 6 of
        # 2891 true pairs, all at separations in [4.44, 4.79] A -- exactly
        # the predicted [w_old, r_list] band.  nc_new = [3, 3, 3] gives
        # w_new = 5.56 A >= r_list and drops none.
        #
        # Note the `maximum(1, ...)` is now defensive only: _check_min_cell
        # has already guaranteed r_list < d_i / 2, hence d_i / r_list > 2
        # and nc_i >= 2 for every axis.
        perp_widths, _ = _cell_metrics(lattice)
        nc = np.maximum(1, np.floor(perp_widths / r_list).astype(int))
        nc0, nc1, nc2 = int(nc[0]), int(nc[1]), int(nc[2])
        n_cells_total = nc0 * nc1 * nc2

        # Fractional coordinates -> sub-cell index per atom
        frac = positions @ inv_lat
        frac = frac - np.floor(frac)
        cell_idx = np.floor(frac * nc).astype(np.int64)
        cell_idx = np.clip(cell_idx, 0, nc - 1)

        flat_idx = (cell_idx[:, 0] * nc1 * nc2
                    + cell_idx[:, 1] * nc2
                    + cell_idx[:, 2])

        # Build linked list
        head = -np.ones(n_cells_total, dtype=np.int64)
        nxt = -np.ones(N, dtype=np.int64)
        for atom in range(N - 1, -1, -1):
            c = flat_idx[atom]
            nxt[atom] = head[c]
            head[c] = atom

        offsets = _build_offset_table(nc0, nc1, nc2)
        rl2 = r_list ** 2

        kernel = _cell_linked_pairs_jit if (self.use_numba and _HAS_NUMBA) \
            else _cell_linked_pairs_numpy

        dummy_i = np.empty(0, dtype=np.int32)
        dummy_rij = np.empty((0, 3), dtype=np.float64)
        dummy_r = np.empty(0, dtype=np.float64)
        n_pairs = kernel(N, nc0, nc1, nc2, cell_idx, offsets, head, nxt,
                          positions, lattice, inv_lat, rl2,
                          dummy_i, dummy_i, dummy_rij, dummy_r, True)

        out_i = np.empty(n_pairs, dtype=np.int32)
        out_j = np.empty(n_pairs, dtype=np.int32)
        out_rij = np.empty((n_pairs, 3), dtype=np.float64)
        out_r = np.empty(n_pairs, dtype=np.float64)
        kernel(N, nc0, nc1, nc2, cell_idx, offsets, head, nxt,
               positions, lattice, inv_lat, rl2,
               out_i, out_j, out_rij, out_r, False)

        self._pair_i = out_i
        self._pair_j = out_j
        self._pair_rij = out_rij
        self._pair_r = out_r
        self._n_pairs = n_pairs
