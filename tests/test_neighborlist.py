"""Correctness tests for the neighbour list -- especially the cell-linked-list
back-end's small-cell (n_cells[axis] in {1, 2}) edge cases, since that is
exactly where the numba-compatibility redesign (replacing a runtime dedup
``set`` with canonical per-axis offsets) could silently drop or duplicate
pairs if the offset logic were wrong.
"""
import numpy as np
import pytest

from akitsushima.engine import LennardJonesPotential
from akitsushima.engine.neighborlist import (
    NeighbourList,
    _HAS_KDTREE,
    _cell_metrics,
)
from akitsushima.utils.jit import _HAS_NUMBA

from conftest import (
    diamond_si_dict,
    fcc_al_dict,
    randomly_perturbed,
    rocksalt_nacl_dict,
    toy_dimer_dict,
)


def _brute_force_pairs(lattice, positions, r_cut):
    """Reference O(N^2) minimum-image pair list, independent of NeighbourList."""
    N = len(positions)
    inv_lat = np.linalg.inv(lattice)
    pairs = {}
    for i in range(N):
        for j in range(i + 1, N):
            rij = positions[j] - positions[i]
            s = rij @ inv_lat
            s -= np.round(s)
            rij = s @ lattice
            r = float(np.linalg.norm(rij))
            if r < r_cut and r > 1e-12:
                pairs[(i, j)] = (rij, r)
    return pairs


def _nl_pairs_as_dict(nl: NeighbourList):
    idx_i, idx_j, rij, r = nl.get_pairs_arrays()
    out = {}
    for k in range(len(idx_i)):
        out[(int(idx_i[k]), int(idx_j[k]))] = (rij[k].copy(), float(r[k]))
    return out


def _assert_pairs_match(pairs_a, pairs_b, atol=1e-8):
    assert set(pairs_a.keys()) == set(pairs_b.keys()), (
        f"pair sets differ: only in A={set(pairs_a) - set(pairs_b)}, "
        f"only in B={set(pairs_b) - set(pairs_a)}")
    for key in pairs_a:
        rij_a, r_a = pairs_a[key]
        rij_b, r_b = pairs_b[key]
        assert abs(r_a - r_b) < atol
        assert np.allclose(rij_a, rij_b, atol=atol)


# nc==1 requires r_list > L/2, which is exactly the region the minimum-image
# validity check (`_check_min_cell`, bug E5's fix) rejects with ValueError --
# so nc==1 can never legitimately reach the pair-search loop; it is covered
# separately by test_rcut_too_large_for_cell_raises() below, not here.
# nc==2 (r_list a bit under L/2) IS a valid, allowed configuration and is the
# smallest cell-count case the pair-search loop can actually see in practice,
# so that's what the correctness parametrization below exercises.
STRUCTS_AND_RCUT = [
    ("fcc_al_nc2", fcc_al_dict, 1.6),         # r_list=1.9 -> nc=2 on a 4.05 A cell
    ("nacl_nc2", rocksalt_nacl_dict, 2.0),    # r_list=2.3 -> nc=2 on a 5.64 A cell
    ("si_nc2", diamond_si_dict, 2.2),         # r_list=2.5 -> nc=2 on a 5.43 A cell
    ("dimer_large_nc", toy_dimer_dict, 5.0),  # 20 A box -> nc large, ordinary case
]


@pytest.mark.parametrize("name,fixture_fn,r_cut", STRUCTS_AND_RCUT,
                          ids=[s[0] for s in STRUCTS_AND_RCUT])
def test_cll_matches_bruteforce_numpy(name, fixture_fn, r_cut):
    struct = randomly_perturbed(fixture_fn(), magnitude=0.03, seed=1)
    nl = NeighbourList(r_cut=r_cut, skin_width=0.3, backend="cll", use_numba=False)
    nl.build(struct["lattice"], struct["positions"])
    ref = _brute_force_pairs(struct["lattice"], struct["positions"], r_cut)
    _assert_pairs_match(_nl_pairs_as_dict(nl), ref)


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("name,fixture_fn,r_cut", STRUCTS_AND_RCUT,
                          ids=[s[0] for s in STRUCTS_AND_RCUT])
def test_cll_matches_bruteforce_numba(name, fixture_fn, r_cut):
    struct = randomly_perturbed(fixture_fn(), magnitude=0.03, seed=1)
    nl = NeighbourList(r_cut=r_cut, skin_width=0.3, backend="cll", use_numba=True)
    nl.build(struct["lattice"], struct["positions"])
    ref = _brute_force_pairs(struct["lattice"], struct["positions"], r_cut)
    _assert_pairs_match(_nl_pairs_as_dict(nl), ref)


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("name,fixture_fn,r_cut", STRUCTS_AND_RCUT,
                          ids=[s[0] for s in STRUCTS_AND_RCUT])
def test_cll_numba_matches_numpy(name, fixture_fn, r_cut):
    """The JIT and pure-NumPy code paths must agree exactly (same kernel,
    same inputs -- fastmath=True should not matter for integer/exact
    distance-squared comparisons at these tolerances)."""
    struct = randomly_perturbed(fixture_fn(), magnitude=0.03, seed=2)
    nl_py = NeighbourList(r_cut=r_cut, skin_width=0.3, backend="cll", use_numba=False)
    nl_py.build(struct["lattice"], struct["positions"])
    nl_jit = NeighbourList(r_cut=r_cut, skin_width=0.3, backend="cll", use_numba=True)
    nl_jit.build(struct["lattice"], struct["positions"])
    _assert_pairs_match(_nl_pairs_as_dict(nl_py), _nl_pairs_as_dict(nl_jit))


@pytest.mark.skipif(not _HAS_KDTREE, reason="scipy cKDTree not installed")
@pytest.mark.parametrize("name,fixture_fn,r_cut", STRUCTS_AND_RCUT,
                          ids=[s[0] for s in STRUCTS_AND_RCUT])
def test_kdtree_matches_bruteforce(name, fixture_fn, r_cut):
    struct = randomly_perturbed(fixture_fn(), magnitude=0.03, seed=1)
    nl = NeighbourList(r_cut=r_cut, skin_width=0.3, backend="kdtree")
    nl.build(struct["lattice"], struct["positions"])
    ref = _brute_force_pairs(struct["lattice"], struct["positions"], r_cut)
    _assert_pairs_match(_nl_pairs_as_dict(nl), ref)


@pytest.mark.parametrize("name,fixture_fn", [
    ("fcc_al", fcc_al_dict), ("nacl", rocksalt_nacl_dict), ("si", diamond_si_dict),
])
def test_rcut_too_large_for_cell_raises(name, fixture_fn):
    """r_list > L/2 (i.e. what would otherwise produce nc==1) must be
    rejected outright (bug E5's fix), not silently miss/duplicate pairs."""
    struct = fixture_fn()
    min_l = float(np.min(np.linalg.norm(struct["lattice"], axis=1)))
    nl = NeighbourList(r_cut=min_l, skin_width=0.5, backend="cll")  # r_list > L/2
    with pytest.raises(ValueError):
        nl.build(struct["lattice"], struct["positions"])


def test_check_and_rebuild_skips_when_displacement_small():
    struct = fcc_al_dict()
    nl = NeighbourList(r_cut=1.0, skin_width=0.3, backend="cll")
    nl.build(struct["lattice"], struct["positions"])
    assert nl.rebuild_count == 1
    rebuilt = nl.check_and_rebuild(struct["lattice"], struct["positions"] + 1e-6)
    assert rebuilt is False
    assert nl.rebuild_count == 1


def test_mark_dirty_forces_rebuild():
    struct = fcc_al_dict()
    nl = NeighbourList(r_cut=1.0, skin_width=0.3, backend="cll")
    nl.build(struct["lattice"], struct["positions"])
    nl.mark_dirty()
    rebuilt = nl.check_and_rebuild(struct["lattice"], struct["positions"])
    assert rebuilt is True
    assert nl.rebuild_count == 2


# ===========================================================================
# Stale-distance / defeated-skin defect
# ===========================================================================

def _fcc_al_supercell(n=4, a=4.05):
    """n^3 conventional FCC cells.  n=4 keeps r_cut + skin_width safely
    below half the shortest cell vector, which NeighbourList enforces."""
    base = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * a
    pos = np.concatenate([base + np.array([i, j, k]) * a
                          for i in range(n) for j in range(n)
                          for k in range(n)])
    return np.eye(3) * (n * a), pos


def test_candidate_pairs_retain_skin_shell_unlike_get_pairs_arrays():
    """
    Half 2 of the stale-distance defect, at the accessor level:
    ``_store_pairs`` deliberately keeps candidates out to
    ``r_list = r_cut + skin_width``, but :meth:`get_pairs_arrays` masks
    them down to ``r_cut`` using BUILD-TIME distances.  A consumer that
    started from that masked list had permanently lost every pair sitting
    in the skin shell, so a pair which later migrated inside ``r_cut``
    was never counted -- the Verlet skin actively removed interactions
    instead of preserving them, and a LARGER skin was WORSE.

    :meth:`get_candidate_pairs` must hand back the full ``r_list`` set.
    """
    lat, pos = _fcc_al_supercell(n=4)
    r_cut, skin = 6.0, 1.0
    nl = NeighbourList(r_cut=r_cut, skin_width=skin)
    nl.build(lat, pos)

    cand_i, _ = nl.get_candidate_pairs()
    masked_i, _, _, _ = nl.get_pairs_arrays()

    assert len(cand_i) > len(masked_i), (
        "get_candidate_pairs must retain the skin shell that "
        "get_pairs_arrays masks away")

    # Every candidate must genuinely lie within r_list, and the candidate
    # set must be a superset of the r_cut-masked set.
    d = pos[nl.get_candidate_pairs()[1]] - pos[cand_i]
    s = d @ np.linalg.inv(lat)
    r_cand = np.linalg.norm((s - np.round(s)) @ lat, axis=1)
    assert np.all(r_cand <= r_cut + skin + 1e-9)
    assert int(np.sum(r_cand <= r_cut)) == len(masked_i)


@pytest.mark.parametrize("skin", [0.0, 0.5, 1.0])
def test_potentials_see_current_geometry_not_the_last_rebuild(skin):
    """
    The decisive end-to-end invariant, and the regression test for both
    halves of the stale-distance defect: evaluating a potential WITH a
    neighbour list must give exactly the same energy/forces/stress as
    evaluating it WITHOUT one at the same positions, for any displacement
    small enough that ``check_and_rebuild`` declines to rebuild.

    Before the fix every potential consumed :meth:`get_pairs_arrays`, so
    with the default ``skin_width > 0`` it evaluated the geometry as of
    the last rebuild (half 1) starting from a pair list that had already
    dropped the skin shell (half 2).  Measured on this very system, the
    energy was wrong by ~122 eV out of ~587 eV and the forces by an
    amount comparable to the largest force present -- an error the size
    of the signal, which broke energy conservation outright in NVE.

    ``skin=0.0`` is included as a control: there the list rebuilds every
    step, so the test must pass both before and after the fix, which
    confirms the other two parametrisations are failing for the intended
    reason rather than from an unrelated setup mistake.
    """
    lat, pos = _fcc_al_supercell(n=4)
    r_cut = 6.0
    lj = LennardJonesPotential(
        {frozenset({"Al"}): {"epsilon": 0.4, "sigma": 2.5}},
        r_cut=r_cut, shift=True)
    types = ["Al"] * len(pos)

    nl = NeighbourList(r_cut=r_cut, skin_width=skin)
    nl.build(lat, pos)

    # Move EVERY atom by the same magnitude, just under the skin/2
    # rebuild threshold, in an independent random direction.  Displacing
    # every atom (rather than normalising so that only the largest moves
    # that far) is what actually drives pairs across the r_cut boundary:
    # two atoms moving oppositely change their separation by up to
    # 2 * 0.45 * skin, which is what half 2 of the defect mishandles.
    rng = np.random.default_rng(7)
    step = rng.normal(0.0, 1.0, pos.shape)
    step /= np.linalg.norm(step, axis=1)[:, None]
    pos_new = pos + step * (0.45 * skin if skin > 0.0 else 0.4)

    rebuilt = nl.check_and_rebuild(lat, pos_new)
    if skin > 0.0:
        assert not rebuilt, (
            "test setup: displacement must stay inside the skin so the "
            "list is NOT rebuilt, or this cannot detect stale data")

    e_exact, f_exact, s_exact = lj.compute(lat, pos_new, types,
                                            neigh_list=None)
    e_nl, f_nl, s_nl = lj.compute(lat, pos_new, types, neigh_list=nl)

    assert e_nl == pytest.approx(e_exact, abs=1e-10), (
        f"neighbour-list-backed energy must match the exact O(N^2) "
        f"evaluation at the same positions: {e_nl} vs {e_exact}")
    assert np.max(np.abs(f_nl - f_exact)) < 1e-10
    assert np.max(np.abs(s_nl - s_exact)) < 1e-12


def test_real_space_coulomb_sees_current_geometry_not_the_last_rebuild():
    """Same invariant for the real-space Ewald sum, which consumed the
    same cached rij/r (PPPMSolver is the caller that passes it a
    neighbour list)."""
    from akitsushima.engine.coulomb.ewald import real_space_coulomb

    lat, pos = _fcc_al_supercell(n=4)
    r_cut, skin = 6.0, 1.0
    q = np.where(np.arange(len(pos)) % 2 == 0, 1.0, -1.0)

    nl = NeighbourList(r_cut=r_cut, skin_width=skin)
    nl.build(lat, pos)

    rng = np.random.default_rng(11)
    step = rng.normal(0.0, 1.0, pos.shape)
    step /= np.max(np.linalg.norm(step, axis=1))
    pos_new = pos + step * (0.45 * skin)
    assert not nl.check_and_rebuild(lat, pos_new)

    e_exact, f_exact, s_exact = real_space_coulomb(
        lat, pos_new, q, 0.3, r_cut, neigh_list=None)
    e_nl, f_nl, s_nl = real_space_coulomb(
        lat, pos_new, q, 0.3, r_cut, neigh_list=nl)

    assert e_nl == pytest.approx(e_exact, abs=1e-9)
    assert np.max(np.abs(f_nl - f_exact)) < 1e-9
    assert np.max(np.abs(s_nl - s_exact)) < 1e-11


# ===========================================================================
# Non-orthogonal cells: lattice-vector LENGTH vs PERPENDICULAR WIDTH
#
# Until this block was added, every lattice in this entire test suite was
# orthogonal (measured: all 725 distinct lattices reaching
# ``_check_min_cell`` across the full suite), and for an orthogonal cell the
# perpendicular width equals the vector length exactly.  That is the coverage
# hole that hid three separate length-instead-of-width defects.
#
# A MILD shear does not discriminate.  The ``magnitude=0.02-0.03``
# off-diagonal idiom used elsewhere in this suite moves the two measures
# apart by ~0.1%, which is far inside every tolerance here and passes
# happily against the buggy code.  These tests therefore use genuinely
# skewed cells: hexagonal (120 deg, the most common non-orthogonal case in
# materials science) and triclinic.
# ===========================================================================

# HCP Mg lattice parameters: a = 3.21 A, c = 5.21 A (c/a = 1.623, close to
# the ideal 1.633).  Used only as a realistic, reproducible source of a
# 120-degree cell -- no Mg energetics are computed from them.
MG_A = 3.21
MG_C = 5.21


def _hex_lattice(nx: int, ny: int, nz: int,
                  a: float = MG_A, c: float = MG_C) -> np.ndarray:
    """An ``nx x ny x nz`` supercell of a hexagonal (gamma = 120 deg) cell.

    ``a_0`` and ``a_1`` both have length ``nx*a`` / ``ny*a`` but their
    perpendicular widths are only ``sqrt(3)/2 ~= 0.866`` of that.
    """
    prim = np.array([[a, 0.0, 0.0],
                     [-a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0],
                     [0.0, 0.0, c]])
    return np.diag([float(nx), float(ny), float(nz)]) @ prim


def _triclinic_lattice() -> np.ndarray:
    """A genuinely triclinic cell (all three angles far from 90 deg)."""
    return np.array([[12.0, 0.0, 0.0],
                     [4.5, 11.0, 0.0],
                     [3.5, 4.0, 10.5]])


def _random_positions_in_cell(lattice: np.ndarray, n: int,
                               seed: int) -> np.ndarray:
    """Uniformly random Cartesian positions filling ``lattice``.

    Random (rather than ideal-crystal) positions matter here: a perfect HCP
    lattice has EMPTY distance shells, and the sub-cell defect only drops
    pairs whose separation falls in a narrow band.  Probed on ideal HCP the
    buggy code looked clean purely because that band happened to contain no
    pairs.
    """
    rng = np.random.default_rng(seed)
    return rng.random((n, 3)) @ lattice


def _hcp_mg_structure(nx: int, ny: int, nz: int,
                       magnitude: float = 0.05, seed: int = 0):
    """A real HCP Mg supercell (basis at (0,0,0) and (1/3, 2/3, 1/2)),
    randomly perturbed per this suite's convention.

    Used for the ENERGY/FORCE comparison rather than uniformly random
    positions: random filling puts some atoms ~0.44 A apart, where the LJ
    r^-13 term reaches ~3e10 eV/A and a one-ULP relative difference shows up
    as a ~1e-5 absolute one.  That is float round-off, not a neighbour-list
    discrepancy, and a physically sensible structure keeps the tight
    absolute tolerances below meaningful.
    """
    lat = _hex_lattice(nx, ny, nz)
    basis = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.5]])
    counts = np.array([nx, ny, nz], dtype=float)
    frac = np.concatenate([(basis + np.array([i, j, k])) / counts
                           for i in range(nx) for j in range(ny)
                           for k in range(nz)])
    rng = np.random.default_rng(seed)
    pos = frac @ lat + rng.uniform(-magnitude, magnitude, (len(frac), 3))
    return lat, pos


def _perp_widths_via_reciprocal(lattice: np.ndarray) -> np.ndarray:
    """Independent oracle for the perpendicular widths.

    Structurally different route from :func:`_cell_metrics`, which uses
    ``V / |a_j x a_k|``: here ``d_i = 1 / |b_i|`` with ``b_i`` the i-th
    COLUMN of ``inv(lattice)``.  No determinant and no cross product, so a
    sign/normalisation error in one cannot be reproduced by the other.
    """
    inv = np.linalg.inv(lattice)
    return 1.0 / np.linalg.norm(inv, axis=0)


def test_cell_metrics_widths_match_independent_reciprocal_oracle():
    """The helper's ``V / |a_j x a_k|`` must agree with ``1 / |b_i|``."""
    for lat in (_hex_lattice(4, 4, 3), _triclinic_lattice(),
                np.diag([7.0, 9.0, 11.0])):
        widths, _ = _cell_metrics(lat)
        assert np.allclose(widths, _perp_widths_via_reciprocal(lat), rtol=1e-12)


def test_cell_metrics_ordering_sigma_min_le_width_le_length():
    """``sigma_min <= min_i d_i <= min_i ||a_i||`` -- the chain that makes a
    length-based criterion strictly too permissive.  Checked on random
    lattices, not just the fixtures, so it is a property test rather than a
    spot check."""
    rng = np.random.default_rng(20250817)
    for _ in range(200):
        lat = rng.normal(0.0, 1.0, (3, 3))
        if abs(np.linalg.det(lat)) < 1e-3:
            continue
        widths, sigma_min = _cell_metrics(lat)
        lengths = np.linalg.norm(lat, axis=1)
        assert sigma_min <= widths.min() + 1e-9
        assert widths.min() <= lengths.min() + 1e-9


def test_cell_metrics_widths_equal_lengths_for_orthogonal_cell():
    """The two criteria coincide exactly when the vectors are mutually
    perpendicular -- including a ROTATED orthogonal cell, which the module's
    cheap ``is_ortho`` diagonality test does not recognise as orthogonal."""
    lat = np.diag([7.0, 9.0, 11.0])
    widths, _ = _cell_metrics(lat)
    assert np.allclose(widths, np.linalg.norm(lat, axis=1), rtol=1e-12)

    theta = 0.37
    rot = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0],
                    [0.0, 0.0, 1.0]])
    lat_rot = lat @ rot.T
    widths_rot, _ = _cell_metrics(lat_rot)
    assert np.allclose(widths_rot, np.linalg.norm(lat_rot, axis=1), rtol=1e-12)


def test_cell_metrics_rejects_degenerate_lattice():
    """A singular cell has no perpendicular width.  It must raise, not
    return inf/NaN: a NaN width would compare False in every ``>=`` test and
    so silently ACCEPT the cell."""
    coplanar = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [3.0, 3.0, 0.0]])
    with pytest.raises(ValueError, match="degenerate"):
        _cell_metrics(coplanar)
    with pytest.raises(ValueError, match="degenerate"):
        _cell_metrics(np.zeros((3, 3)))


@pytest.mark.parametrize("backend", ["cll", "kdtree"])
def test_sheared_cell_with_adequate_vector_lengths_is_rejected(backend):
    """
    CASE (a): the headline regression.

    On a hexagonal 4x4x3 Mg supercell the vector lengths are
    [12.84, 12.84, 15.63] A, so ``min ||a_i|| / 2 = 6.42 A`` -- but the
    perpendicular widths are only [11.12, 11.12, 15.63] A, so
    ``min d_i / 2 = 5.56 A``.

    An ``r_list`` chosen strictly between those two numbers therefore looks
    perfectly safe by vector length while minimum image is ALREADY invalid.
    The old length-based check accepted it and the neighbour list then
    returned wrong pairs silently; it must now raise.
    """
    lat = _hex_lattice(4, 4, 3)
    pos = _random_positions_in_cell(lat, 60, seed=3)

    lengths = np.linalg.norm(lat, axis=1)
    widths, _ = _cell_metrics(lat)
    # The band exists at all only because the cell is sheared.
    assert widths.min() / 2.0 < lengths.min() / 2.0

    r_list = 0.5 * (widths.min() / 2.0 + lengths.min() / 2.0)  # ~5.99 A
    assert widths.min() / 2.0 < r_list < lengths.min() / 2.0

    nl = NeighbourList(r_cut=r_list - 0.1, skin_width=0.1, backend=backend)
    with pytest.raises(ValueError, match="perpendicular cell width"):
        nl.build(lat, pos)


def test_orthogonal_cell_acceptance_boundary_is_unchanged():
    """
    CASE (b): for an orthogonal cell the old and new criteria are the SAME
    number, so the tightening must not move the accept/reject boundary by
    even a float.  This is the control that shows case (a) fails for the
    intended reason rather than from a blanket tightening.
    """
    lat = np.diag([10.0, 12.0, 14.0])
    pos = _random_positions_in_cell(lat, 40, seed=4)
    half = 10.0 / 2.0  # both criteria give exactly this

    ok = NeighbourList(r_cut=half - 0.01, skin_width=0.0, backend="cll")
    ok.build(lat, pos)              # just inside -- must be accepted
    assert ok.rebuild_count == 1

    bad = NeighbourList(r_cut=half + 0.01, skin_width=0.0, backend="cll")
    with pytest.raises(ValueError):
        bad.build(lat, pos)         # just outside -- must be rejected


# (name, lattice-factory, n_atoms, r_cut, why this case matters)
NONORTHO_CASES = [
    # nc_old = floor(19.26/4.8) = 4 sub-cells but only floor(16.68/4.8) = 3
    # are safe, so the buggy CLL binning makes sub-cells 4.17 A wide for a
    # 4.8 A reach.  nc >= 4 is REQUIRED to expose it: for nc <= 3 the
    # {-1,0,+1} stencil already covers every sub-cell along that axis.
    ("hex_nc4", lambda: _hex_lattice(6, 6, 3), 250, 4.8),
    # sigma_min = 9.08 A << min||a_i|| = 12.84 A, so the buggy kdtree
    # fractional radius (0.399) badly under-reaches the correct one (0.551).
    ("hex_sigma", lambda: _hex_lattice(4, 4, 3), 120, 5.0),
    ("triclinic", _triclinic_lattice, 150, 4.0),
]


@pytest.mark.parametrize("name,lat_fn,n,r_cut", NONORTHO_CASES,
                          ids=[c[0] for c in NONORTHO_CASES])
@pytest.mark.parametrize("backend", ["cll", "kdtree"])
def test_nonorthogonal_pairs_match_bruteforce(name, lat_fn, n, r_cut, backend):
    """
    CASE (c): both back-ends must reproduce the brute-force O(N^2)
    minimum-image pair list on a genuinely skewed cell.

    The reference is deliberately NOT the other back-end: ``cll`` and
    ``kdtree`` shared this same root cause and could agree with each other
    while both were wrong.  ``_brute_force_pairs`` is an independent
    O(N^2) minimum-image implementation.
    """
    if backend == "kdtree" and not _HAS_KDTREE:
        pytest.skip("scipy cKDTree not installed")
    lat = lat_fn()
    pos = _random_positions_in_cell(lat, n, seed=17)

    widths, _ = _cell_metrics(lat)
    assert r_cut < widths.min() / 2.0, (
        "test setup: this case must be minimum-image VALID, so that any "
        "discrepancy is a search-reach bug and not an invalid cell")

    nl = NeighbourList(r_cut=r_cut, skin_width=0.0, backend=backend,
                        use_numba=False)
    nl.build(lat, pos)
    ref = _brute_force_pairs(lat, pos, r_cut)
    _assert_pairs_match(_nl_pairs_as_dict(nl), ref)


@pytest.mark.parametrize("backend", ["cll", "kdtree"])
def test_lj_on_hexagonal_cell_matches_neighbourless_evaluation(backend):
    """End-to-end consequence: a neighbour-list-backed LJ evaluation on a
    hexagonal cell must equal the exact O(N^2) evaluation.  Before the fix
    the kdtree back-end lost 21.5% of pairs here, which showed up as a
    2.9 meV/atom energy error and max|dF| ~ 1.2e-2 eV/A -- silently."""
    if backend == "kdtree" and not _HAS_KDTREE:
        pytest.skip("scipy cKDTree not installed")
    lat, pos = _hcp_mg_structure(4, 4, 3, magnitude=0.05, seed=23)
    types = ["Mg"] * len(pos)
    r_cut = 5.0

    lj = LennardJonesPotential(
        {frozenset({"Mg"}): {"epsilon": 0.05, "sigma": 2.9}},
        r_cut=r_cut, shift=True)

    nl = NeighbourList(r_cut=r_cut, skin_width=0.0, backend=backend)
    nl.build(lat, pos)

    e_exact, f_exact, s_exact = lj.compute(lat, pos, types, neigh_list=None)
    e_nl, f_nl, s_nl = lj.compute(lat, pos, types, neigh_list=nl)

    assert e_nl == pytest.approx(e_exact, abs=1e-10)
    assert np.max(np.abs(f_nl - f_exact)) < 1e-10
    assert np.max(np.abs(s_nl - s_exact)) < 1e-12


def test_rebuild_criterion_survives_a_pbc_wrap_and_a_deforming_cell():
    """The skin must still buy something for an atom on a cell face, and
    under a cell that changes every step.

    Two separate defects met in check_and_rebuild.  The non-affine
    displacement was measured as max|positions - ref| in raw Cartesian
    coordinates, but the MD and relax loops PBC-wrap every step, so one
    atom oscillating around a face appeared to jump a whole lattice
    vector each step -- taking an NVE run from 43 rebuilds in 501 steps
    to 501.  And any lattice change at all forced a rebuild, so NPT,
    which changes the cell every step, rebuilt on 99.4% of them.

    The replacement keeps the identical safety margin when the cell does
    not deform: sigma_min = 1 reduces the test to d_max < skin/2.  It is
    a correction to how d_max is measured, not a loosening.
    """
    # r_cut + skin = 4.2, so the cell must be wider than 8.4 A.
    a = 12.0
    lattice = np.eye(3) * a
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, a, size=(40, 3))
    pos[0] = [0.0, 0.3 * a, 0.3 * a]          # exactly on a face

    nl = NeighbourList(r_cut=3.2, skin_width=1.0)
    nl.build(lattice, pos)

    # An atom stepping just across the face and back must not look like
    # motion of a whole lattice vector.
    rebuilds = 0
    p = pos.copy()
    for k in range(20):
        p[0, 0] = -1e-9 if k % 2 else 1e-9
        p[0, 0] %= a
        rebuilds += int(nl.check_and_rebuild(lattice, p))
    assert rebuilds == 0, (
        f"a PBC wrap forced {rebuilds} rebuilds; the skin is being wasted")

    # Cell deformation.  Expansion alone can never invalidate the list:
    # it only pushes separations apart, so a pair that was outside
    # r_list cannot reach r_cut.  Compression can, and must trigger a
    # rebuild once sigma_min * r_list drops to r_cut -- here at
    # sigma_min = 3.2 / 4.2 = 0.762.
    nl2 = NeighbourList(r_cut=3.2, skin_width=1.0)
    nl2.build(lattice, pos)
    assert not nl2.check_and_rebuild(lattice * 1.5, pos * 1.5), (
        "an expansion forced a rebuild it cannot need")
    nl3 = NeighbourList(r_cut=3.2, skin_width=1.0)
    nl3.build(lattice, pos)
    assert not nl3.check_and_rebuild(lattice * 0.99, pos * 0.99)
    # 0.75 * 4.2 = 3.15 < 3.2 triggers, and the compressed cell is still
    # 9.0 A wide, so the rebuild it asks for is itself legal.  (0.70
    # also triggers, but the 8.4 A cell it lands in cannot host
    # r_list = 4.2, so build() rightly raises instead of returning.)
    nl4 = NeighbourList(r_cut=3.2, skin_width=1.0)
    nl4.build(lattice, pos)
    assert nl4.check_and_rebuild(lattice * 0.75, pos * 0.75), (
        "a compression past sigma_min = 0.762 did not trigger a rebuild")
