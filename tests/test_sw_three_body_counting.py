"""Stillinger-Weber three-body counting.

Stillinger & Weber, PRB 31, 5262 (1985), eq. (4)-(5):

    Phi3 = sum_{i<j<k} [h(r_ij, r_ik, theta_jik) + h(r_ji, r_jk, theta_ijk)
                        + h(r_ki, r_kj, theta_ikj)],

i.e. every (centre i, UNORDERED neighbour pair {j, k}) contributes h once.
The references below are brute-force sums written from that definition,
independent of the engine's neighbour list and kernels.  Multi-species:
each unordered pair contributes (h_ijk + h_ikj) / 2 with the (i, j, k)
record's own sigma, a, gamma, lambda (this engine's record convention),
which is the definition above whenever the two records agree and does
not depend on neighbour order when they do not.
"""

import contextlib
import io
import itertools

import numpy as np
import pytest

from akitsushima import (
    ElasticConstantsCalculator,
    PotentialEngine,
    StillingerWeberPotential,
)
from akitsushima.fitting import make_force_fn

SW_SI = {
    "epsilon": 2.1683,
    "sigma": 2.0951,
    "a": 1.80,
    "lambda": 21.0,
    "gamma": 1.20,
    "A": 7.049556277,
    "B": 0.6022245584,
    "p": 4.0,
    "q": 0.0,
    "costheta0": -1.0 / 3.0,
}
A_SI = 5.430950
BASIS = np.array(
    [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ]
)


def _diamond(a, n=2):
    frac = np.concatenate(
        [(BASIS + [i, j, k]) / n for i in range(n) for j in range(n) for k in range(n)]
    )
    lat = np.eye(3) * n * a
    return lat, frac @ lat


def _neighbours(lat, pos, rc):
    sh = np.array(list(itertools.product((-1, 0, 1), repeat=3))) @ lat
    nb = [[] for _ in pos]
    for i in range(len(pos)):
        for j in range(len(pos)):
            for s in sh:
                v = pos[j] + s - pos[i]
                r = np.linalg.norm(v)
                if 1e-8 < r < rc:
                    nb[i].append((j, v))
    return nb


def _h(v1, v2, rec):
    eps, sig, a = rec.get("epsilon", 1.0), rec["sigma"], rec["a"]
    x1, x2 = np.linalg.norm(v1) / sig, np.linalg.norm(v2) / sig
    if x1 >= a or x2 >= a:
        return 0.0
    c = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return (
        eps
        * rec["lambda"]
        * np.exp(rec["gamma"] / (x1 - a) + rec["gamma"] / (x2 - a))
        * (c - rec.get("costheta0", -1.0 / 3.0)) ** 2
    )


def _v2(r, rec):
    eps, sig, a = rec.get("epsilon", 1.0), rec["sigma"], rec["a"]
    x = r / sig
    if x >= a:
        return 0.0
    return (
        eps
        * rec["A"]
        * (rec["B"] * x ** -rec["p"] - x ** -rec["q"])
        * np.exp(1 / (x - a))
    )


def brute_sw(lat, pos, types, records):
    """records[(i, j, k)] -> dict; two-body from (i, j, j)."""
    rc = max(r["a"] * r["sigma"] for r in records.values())
    nb = _neighbours(lat, pos, rc + 1e-9)
    E = 0.0
    for i, lst in enumerate(nb):
        for j, v in lst:
            E += 0.5 * _v2(np.linalg.norm(v), records[(types[i], types[j], types[j])])
        for (j, vj), (k, vk) in itertools.combinations(lst, 2):
            ti, tj, tk = types[i], types[j], types[k]
            E += 0.5 * (
                _h(vj, vk, records[(ti, tj, tk)]) + _h(vk, vj, records[(ti, tk, tj)])
            )
    return E


def _rattled(seed=0, n=2, a=A_SI):
    lat, pos = _diamond(a, n)
    return lat, pos + np.random.default_rng(seed).normal(0, 0.1, pos.shape)


@pytest.mark.parametrize("use_numba", [False, True])
def test_single_species_energy_matches_the_paper(use_numba):
    lat, pos = _rattled()
    types = ["Si"] * len(pos)
    eng = PotentialEngine().add(StillingerWeberPotential(SW_SI, species=["Si"]))
    E = eng.compute(lat, pos, types, use_numba=use_numba)[0]
    ref = brute_sw(lat, pos, types, {("Si", "Si", "Si"): SW_SI})
    assert E == pytest.approx(ref, rel=1e-10)


@pytest.mark.parametrize("use_numba", [False, True])
def test_multi_species_with_identical_records_equals_single_species(use_numba):
    """Two labels carrying SW Si's parameters in every record are Si."""
    lat, pos = _rattled(1)
    types = ["Ge" if k % 3 else "Sn" for k in range(len(pos))]
    recs = {t: dict(SW_SI) for t in itertools.product(["Ge", "Sn"], repeat=3)}
    eng = PotentialEngine().add(StillingerWeberPotential(recs, species=["Ge", "Sn"]))
    single = PotentialEngine().add(StillingerWeberPotential(SW_SI, species=["Si"]))
    E, F, S = eng.compute(lat, pos, types, use_numba=use_numba)
    E1, F1, S1 = single.compute(lat, pos, ["Si"] * len(pos), use_numba=use_numba)
    assert E == pytest.approx(E1, rel=1e-10)
    np.testing.assert_allclose(F, F1, atol=1e-6)
    np.testing.assert_allclose(S, S1, atol=1e-8)


def _tomita():
    import yaml

    from akitsushima.fitting import MultiSpeciesStillingerWeberParameterSet as MS

    recs = yaml.safe_load(open("example/GeSn/relax/gesn_sw_relax.yaml"))["potential"][
        "params"
    ]
    return MS._to_nested(recs)


@pytest.mark.parametrize("use_numba", [False, True])
def test_multi_species_tomita_energy_matches_brute_force(use_numba):
    """Tomita et al. (2018) GeSn, whose (i,j,k) and (i,k,j) records
    differ in gamma: each unordered pair counts (h_ijk + h_ikj) / 2."""
    nested = _tomita()
    lat, pos = _rattled(2, a=5.9)
    rng = np.random.default_rng(3)
    types = list(rng.choice(["Ge", "Sn"], len(pos)))
    eng = PotentialEngine().add(StillingerWeberPotential(nested, species=["Ge", "Sn"]))
    E = eng.compute(lat, pos, types, use_numba=use_numba)[0]
    assert E == pytest.approx(brute_sw(lat, pos, types, nested), rel=1e-10)


@pytest.mark.parametrize("use_numba", [False, True])
def test_multi_species_forces_and_stress_are_derivatives(use_numba):
    nested = _tomita()
    lat, pos = _rattled(4, a=5.9)
    types = list(np.random.default_rng(5).choice(["Ge", "Sn"], len(pos)))
    eng = PotentialEngine().add(StillingerWeberPotential(nested, species=["Ge", "Sn"]))
    E, F, S = eng.compute(lat, pos, types, use_numba=use_numba)
    h = 1e-5
    for j, a in ((0, 0), (17, 2)):
        p = pos.copy()
        p[j, a] += h
        Ep = eng.compute(lat, p, types, use_numba=use_numba)[0]
        p[j, a] -= 2 * h
        Em = eng.compute(lat, p, types, use_numba=use_numba)[0]
        assert -(Ep - Em) / (2 * h) == pytest.approx(F[j, a], abs=1e-5)
    V = abs(np.linalg.det(lat))
    frac = pos @ np.linalg.inv(lat)
    for a, b in ((0, 0), (1, 2)):
        T = np.eye(3)
        T[a, b] += h
        Ep = eng.compute(lat @ T, frac @ lat @ T, types, use_numba=use_numba)[0]
        T = np.eye(3)
        T[a, b] -= h
        Em = eng.compute(lat @ T, frac @ lat @ T, types, use_numba=use_numba)[0]
        assert -(Ep - Em) / (2 * h) / V == pytest.approx(S[a, b], abs=1e-6)


def test_sw_si_elastic_constants_are_stillinger_webers():
    """SW Si at a = 5.430950 A: C11 151.4, C12 76.4, C44 (ions relaxed)
    56.4 GPa -- the values published for this parameter set (Cowley,
    PRL 60, 2379 (1988); LAMMPS examples/ELASTIC with Si.sw).  The bulk
    modulus alone (101.4 GPa) cannot catch a three-body scale error:
    hydrostatic strain keeps every angle tetrahedral."""
    lat, pos = _diamond(A_SI)
    fn = make_force_fn(
        PotentialEngine().add(StillingerWeberPotential(SW_SI, species=["Si"]))
    )
    with contextlib.redirect_stdout(io.StringIO()):
        C = ElasticConstantsCalculator(
            lat, pos, ["Si"] * 64, fn, relax_ions=True, fmax=1e-5, max_steps=5000
        ).compute()["C"]
    assert C[0, 0] == pytest.approx(151.4, abs=0.5)
    assert C[0, 1] == pytest.approx(76.4, abs=0.5)
    assert C[3, 3] == pytest.approx(56.4, abs=0.5)
