"""Force functions called without a neighbour list (ADR-0028, tier A step 3).

A force function that is given no list takes one from
``owned_neighbour_list`` -- built once, reused while still complete -- in
place of enumerating all N(N-1)/2 pairs.  These tests pin down that

1. every potential gives the all-pairs answer (to rounding: the pair order,
   and so the summation order, differs) for energy, forces and stress,
   and for the per-site terms of the Green-Kubo heat flux;
2. a reused list stays correct through displacements and strains that do
   not need a rebuild, and is rebuilt when they do;
3. small systems and cells too thin for ``r_cut + skin`` keep the
   all-pairs path, with its error message and the
   ``allow_minimum_image_violation`` escape hatch;
4. phonon/elastic-style concurrent calls from a thread pool are correct;
5. the per-thread cache is bounded and keyed by (r_cut, N).
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from conftest import diamond_si_dict, fcc_al_dict
from test_site_terms_heat_flux import SW_SI, T3_SI, T_XY, _eam, _gesn_sw

import akitsushima.engine.neighborlist as nlm
from akitsushima.engine import (
    LennardJonesPotential,
    StillingerWeberPotential,
    TersoffPotential,
)
from akitsushima.engine.coulomb.ewald import real_space_coulomb
from akitsushima.engine.neighborlist import (
    allow_minimum_image_violation,
    clear_owned_neighbour_lists,
    owned_neighbour_list,
)

RTOL = 1e-10
SHEAR = np.array([[1.0, 0.0, 0.0], [0.21, 0.97, 0.0], [0.12, -0.09, 1.03]])


def _struct(d):
    return (
        np.asarray(d["lattice"], float),
        np.asarray(d["positions"], float),
        list(d["atom_types"]),
    )


def _perturbed(lat, pos, types, seed, shear=False, noise=0.05):
    rng = np.random.default_rng(seed)
    if shear:
        lat2 = lat @ SHEAR
        pos = (pos @ np.linalg.inv(lat)) @ lat2
        lat = lat2
    return lat, pos + rng.normal(0.0, noise, pos.shape), types


def _si3():
    return _struct(diamond_si_dict(3))  # 216 atoms, half width 8.1 A


def _al4():
    return _struct(fcc_al_dict(4))  # 256 atoms, half width 8.1 A


def _gesn3():
    lat, pos, _ = _si3()
    f = 5.9 / 5.43
    types = list(np.random.default_rng(3).choice(["Ge", "Sn"], len(pos)))
    return lat * f, pos * f, types


def _xy3():
    lat, pos, _ = _si3()
    return lat, pos, (["X"] * 4 + ["Y"] * 4) * 27


POTENTIALS = {
    "sw": (lambda: StillingerWeberPotential(dict(SW_SI), species=["Si"]), _si3),
    "sw_multi_gesn": (_gesn_sw, _gesn3),
    "tersoff": (lambda: TersoffPotential(dict(T3_SI), species=["Si"]), _si3),
    "tersoff_xy": (lambda: TersoffPotential(T_XY, species=["X", "Y"]), _xy3),
    "eam": (_eam, _al4),
    "lj": (
        lambda: LennardJonesPotential(
            {frozenset({"Al"}): {"epsilon": 0.4, "sigma": 2.6}}, r_cut=6.0
        ),
        _al4,
    ),
}


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_owned_neighbour_lists()
    yield
    clear_owned_neighbour_lists()


def _all_pairs(monkeypatch, fn):
    """``fn()`` with owned lists switched off: the all-pairs reference."""
    with monkeypatch.context() as m:
        m.setattr(nlm, "_OWNED_MIN_ATOMS", 10**9)
        return fn()


def _assert_close(a, b, what):
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(float(np.max(np.abs(b))), 1e-30)
    rel = float(np.max(np.abs(a - b))) / scale
    assert rel < RTOL, f"{what}: relative difference {rel:.2e}"


def _cache():
    return getattr(nlm._owned, "lists", {})


@pytest.mark.parametrize("shear", [False, True], ids=["cubic", "sheared"])
@pytest.mark.parametrize("name", sorted(POTENTIALS))
def test_matches_all_pairs(monkeypatch, name, shear):
    make, structure = POTENTIALS[name]
    pot = make()
    lat, pos, types = _perturbed(*structure(), seed=1, shear=shear)
    ref = _all_pairs(monkeypatch, lambda: pot.compute(lat, pos, types))
    assert not _cache()
    got = pot.compute(lat, pos, types)
    assert len(_cache()) == 1, "the owned list was not used"
    for a, b, what in zip(got, ref, ("E", "F", "S")):
        _assert_close(a, b, f"{name} {what}")


@pytest.mark.parametrize("name", ["sw", "sw_multi_gesn", "tersoff_xy", "eam"])
def test_site_terms_match_all_pairs(monkeypatch, name):
    make, structure = POTENTIALS[name]
    pot = make()
    lat, pos, types = _perturbed(*structure(), seed=2, shear=True)
    V = abs(np.linalg.det(lat))
    ref = _all_pairs(monkeypatch, lambda: pot.site_terms(lat, pos, types))
    got = pot.site_terms(lat, pos, types)
    assert _cache()
    _assert_close(got.e_site, ref.e_site, "site energies")
    _assert_close(got.forces(), ref.forces(), "site forces")
    _assert_close(got.stress(V), ref.stress(V), "site stress")


def test_real_space_coulomb_matches_all_pairs(monkeypatch):
    lat, pos, _ = _perturbed(*_si3(), seed=3, shear=True)
    q = np.where(np.arange(len(pos)) % 2 == 0, 2.0, -2.0)

    def run():
        return real_space_coulomb(lat, pos, q, 0.6, 6.0)

    ref = _all_pairs(monkeypatch, run)
    got = run()
    assert _cache()
    for a, b, what in zip(got, ref, ("E", "F", "S")):
        _assert_close(a, b, f"real-space Coulomb {what}")


def test_reuse_and_rebuild_stay_exact(monkeypatch):
    """A sequence of geometries an owned list meets in phonon / elastic /
    one-off use: each result equals the all-pairs one, the list is reused
    for the small changes and rebuilt for the large ones."""
    pot = StillingerWeberPotential(dict(SW_SI), species=["Si"])
    lat0, pos0, types = _perturbed(*_si3(), seed=4)
    rng = np.random.default_rng(5)
    disp = np.zeros_like(pos0)
    disp[7, 0] = 0.02
    strain = np.eye(3) + np.diag([0.01, -0.01, 0.0])
    # compression shrinks separations (sigma_min 0.9) and needs a
    # rebuild; a 10 % stretch would not
    big_strain = np.eye(3) + np.diag([-0.1, 0.0, 0.0])
    steps = [
        ("same", lat0, pos0, False),
        ("phonon +", lat0, pos0 + disp, False),
        ("phonon -", lat0, pos0 - disp, False),
        ("1% strain", lat0 @ strain, pos0 @ strain, False),
        ("-10% strain", lat0 @ big_strain, pos0 @ big_strain, True),
        ("0.3 A noise", lat0, pos0 + rng.normal(0, 0.3, pos0.shape), True),
        ("back", lat0, pos0, True),
    ]
    pot.compute(lat0, pos0, types)
    (nl,) = _cache().values()
    for what, lat, pos, rebuilds in steps:
        before = nl.rebuild_count
        got = pot.compute(lat, pos, types)
        assert (nl.rebuild_count > before) == rebuilds, what
        ref = _all_pairs(monkeypatch, lambda: pot.compute(lat, pos, types))
        for a, b, q in zip(got, ref, ("E", "F", "S")):
            _assert_close(a, b, f"{what} {q}")


def test_small_systems_keep_the_all_pairs_path():
    lat, pos, _ = _struct(diamond_si_dict(2))  # 64 atoms < _OWNED_MIN_ATOMS
    assert len(pos) < nlm._OWNED_MIN_ATOMS
    assert owned_neighbour_list(lat, pos, 3.77) is None
    StillingerWeberPotential(dict(SW_SI), species=["Si"]).compute(
        lat, pos, ["Si"] * len(pos)
    )
    assert not _cache()


def test_thin_cells_keep_the_all_pairs_path_and_its_checks():
    # 3 x 3 x 1 diamond Si: 72 atoms in plane, half the thin width 2.7 A
    lat, pos, types = _struct(diamond_si_dict(1))
    rep = np.array([[i, j, 0] for i in range(4) for j in range(4)])
    pos = np.concatenate([pos + r @ lat for r in rep])
    lat = lat * np.array([4, 4, 1])[:, None]
    types = ["Si"] * len(pos)
    assert len(pos) >= nlm._OWNED_MIN_ATOMS
    assert owned_neighbour_list(lat, pos, 3.77) is None
    pot = StillingerWeberPotential(dict(SW_SI), species=["Si"])
    with pytest.raises(ValueError, match="minimum-image"):
        pot.compute(lat, pos, types)
    with allow_minimum_image_violation():
        e, _, _ = pot.compute(lat, pos, types)
    assert np.isfinite(e)
    assert not _cache()


def test_thread_pool_calls_are_correct(monkeypatch):
    """Phonon and elastic evaluate displaced cells in a thread pool."""
    pot = TersoffPotential(dict(T3_SI), species=["Si"])
    lat, pos0, types = _perturbed(*_si3(), seed=6)
    rng = np.random.default_rng(7)
    cells = [(lat, pos0 + rng.normal(0, 0.2, pos0.shape)) for _ in range(12)]
    ref = [
        _all_pairs(monkeypatch, lambda c=c: pot.compute(c[0], c[1], types))
        for c in cells
    ]
    with ThreadPoolExecutor(max_workers=4) as ex:
        got = list(ex.map(lambda c: pot.compute(c[0], c[1], types), cells))
    for k, (g, r) in enumerate(zip(got, ref)):
        for a, b, q in zip(g, r, ("E", "F", "S")):
            _assert_close(a, b, f"cell {k} {q}")


def test_cache_is_bounded_and_keyed_by_cutoff_and_size():
    lat, pos, _ = _si3()
    for k in range(nlm._OWNED_CACHE_SIZE + 2):
        assert owned_neighbour_list(lat, pos, 3.0 + 0.1 * k) is not None
    assert len(_cache()) == nlm._OWNED_CACHE_SIZE
    clear_owned_neighbour_lists()
    a = owned_neighbour_list(lat, pos, 4.0)
    b = owned_neighbour_list(lat, pos[:-1], 4.0)  # one atom fewer
    assert a is not b
    assert owned_neighbour_list(lat, pos, 4.0) is a


@pytest.mark.parametrize("name", ["sw", "tersoff_xy", "eam", "lj"])
def test_result_does_not_depend_on_when_the_list_was_built(monkeypatch, name):
    """Candidates are kept in np.triu_indices order after every build, so
    a call through a reused list, a freshly built one and the all-pairs
    path give the same bits (a restarted run must continue exactly)."""
    make, structure = POTENTIALS[name]
    pot = make()
    lat, pos, types = _perturbed(*structure(), seed=8, shear=True)
    moved = pos + np.random.default_rng(9).normal(0, 0.03, pos.shape)
    pot.compute(lat, moved, types)  # the list is built at `moved`
    reused = pot.compute(lat, pos, types)  # ... and reused here
    assert next(iter(_cache().values())).rebuild_count == 1
    clear_owned_neighbour_lists()
    fresh = pot.compute(lat, pos, types)
    ref = _all_pairs(monkeypatch, lambda: pot.compute(lat, pos, types))
    for r, f, a in zip(reused, fresh, ref):
        assert np.array_equal(r, f)
        assert np.array_equal(f, a)
