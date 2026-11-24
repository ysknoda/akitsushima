"""Analytic three-body forces of the Stillinger-Weber kernels (ADR-0047).

The numba kernels used central finite differences of the three-body term
h3 until ADR-0047 (12 extra evaluations per triple); they now use its
analytic gradient (_sw_h3_grad).  A hand-derived formula used before the
finite differences was wrong in a way a perfect crystal hides (every
force vanishes by symmetry), so everything here runs on rattled, strained
cells with distinct species and, for the multi-species path, records
whose (i, j, k) and (i, k, j) entries differ.

Checks
- h of _sw_h3_grad equals _sw_h3_scalar / _sw_h3_scalar_multi bit for bit
  (the energy of the kernels is unchanged), and its gradient matches
  central differences of h3 on random triples, including legs close to
  their cutoff.
- Forces and stress of the numba path match central differences of the
  total energy to a relative 1e-7; forces sum to zero.
- The NumPy fallback (still finite differences) agrees within 1e-8 eV/A
  (forces) and a relative 1e-8 (stress): the former kernels' accuracy.
"""

import copy

import numpy as np
import pytest
from test_engine_forces import finite_diff_stress
from test_sw_three_body_counting import A_SI, SW_SI, _rattled, _tomita

from akitsushima import PotentialEngine, StillingerWeberPotential
from akitsushima.engine import stillinger_weber as swm


def _asymmetric_tomita():
    """Tomita et al. (2018) GeSn with the Ge-Ge-Sn and Sn-Ge-Ge triples
    made different from their mirror records (lambda and cos theta0), so
    that the two ordered visits of an unordered pair carry different
    records.  The changes are test values, not physical parameters."""
    nested = copy.deepcopy(_tomita())
    for key, f_lam, d_cos in (
        (("Ge", "Ge", "Sn"), 1.3, 0.05),
        (("Sn", "Ge", "Ge"), 0.8, -0.04),
    ):
        nested[key]["lambda"] = float(nested[key]["lambda"]) * f_lam
        nested[key]["costheta0"] = float(nested[key].get("costheta0", -1 / 3)) + d_cos
    return nested


def _strained(lat, pos, seed, amp=0.02):
    rng = np.random.default_rng(seed)
    e = rng.uniform(-amp, amp, (3, 3))
    e = 0.5 * (e + e.T)
    lat2 = lat @ (np.eye(3) + e)
    return lat2, pos @ np.linalg.inv(lat) @ lat2


def _cases():
    lat, pos = _rattled(seed=11, n=2, a=A_SI)
    lat, pos = _strained(lat, pos, 1)
    si = (
        PotentialEngine().add(StillingerWeberPotential(SW_SI)),
        lat,
        pos,
        ["Si"] * len(pos),
    )
    lat, pos = _rattled(seed=12, n=2, a=5.9)
    lat, pos = _strained(lat, pos, 2)
    types = list(np.random.default_rng(13).choice(["Ge", "Sn"], len(pos)))
    gesn = (
        PotentialEngine().add(
            StillingerWeberPotential(_asymmetric_tomita(), species=["Ge", "Sn"])
        ),
        lat,
        pos,
        types,
    )
    return {"si": si, "gesn": gesn}


CASES = _cases()


def _py(f):
    return getattr(f, "py_func", f)


def test_h3_value_is_the_former_one_bit_for_bit():
    """Same operations in the same order: bit for bit in plain Python.  The
    numba kernels are compiled with fastmath, which may reorder them, so
    the compiled functions are compared to a relative 1e-15 (the kernel
    energies were also checked bit for bit against 0.3.0a9, ADR-0047)."""
    grad, h1, hm = _py(swm._sw_h3_grad), _py(swm._sw_h3_scalar), _py(swm._sw_h3_scalar_multi)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        v = rng.normal(0.0, 1.6, 6)
        sj, sk = rng.uniform(1.9, 2.6, 2)
        aj, ak = rng.uniform(1.7, 1.9, 2)
        gj, gk = rng.uniform(1.0, 1.4, 2)
        eps, lam, c0 = rng.uniform(1.5, 2.5), rng.uniform(15, 30), rng.uniform(-0.4, -0.3)
        assert grad(*v, sj, aj, gj, sj, aj, gj, lam * eps, c0)[0] == h1(
            *v, sj, aj, gj, lam, eps, c0)
        assert grad(*v, sj, aj, gj, sk, ak, gk, eps * lam, c0)[0] == hm(
            *v, sj, aj, gj, sk, ak, gk, eps, lam, c0)
        a = swm._sw_h3_grad(*v, sj, aj, gj, sk, ak, gk, eps * lam, c0)[0]
        b = swm._sw_h3_scalar_multi(*v, sj, aj, gj, sk, ak, gk, eps, lam, c0)
        assert a == pytest.approx(b, rel=1e-15, abs=0.0)


def test_h3_gradient_matches_central_differences():
    rng = np.random.default_rng(1)
    hh = 1e-6
    n_inside = 0
    for _ in range(500):
        sj, sk = rng.uniform(1.9, 2.6, 2)
        aj, ak = rng.uniform(1.7, 1.9, 2)
        gj, gk = rng.uniform(1.0, 1.4, 2)
        c3, c0 = rng.uniform(20, 60), rng.uniform(-0.4, -0.3)
        # legs from deep inside up to 2 % below the cutoff
        u = rng.normal(size=(2, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        v = np.concatenate(
            [
                u[0] * rng.uniform(0.9, 0.98) * aj * sj,
                u[1] * rng.uniform(0.9, 0.98) * ak * sk,
            ]
        )
        args = (sj, aj, gj, sk, ak, gk, c3, c0)
        out = swm._sw_h3_grad(*v, *args)
        if out[0] == 0.0:
            continue
        n_inside += 1
        g = np.array(out[1:])
        fd = np.empty(6)
        for c in range(6):
            p, m = v.copy(), v.copy()
            p[c] += hh
            m[c] -= hh
            fd[c] = (swm._sw_h3_grad(*p, *args)[0] - swm._sw_h3_grad(*m, *args)[0]) / (
                2 * hh
            )
        np.testing.assert_allclose(
            g, fd, atol=1e-7 * max(1.0, np.abs(g).max()), rtol=1e-6
        )
    assert n_inside > 400


@pytest.mark.parametrize("case", ["si", "gesn"])
def test_forces_and_stress_are_energy_derivatives(case):
    eng, lat, pos, types = CASES[case]
    E, F, S = eng.compute(lat, pos, types, use_numba=True)
    assert np.abs(F.sum(axis=0)).max() < 1e-11
    h = 1e-5
    fd = np.empty_like(F)
    for i in range(len(pos)):
        for a in range(3):
            p = pos.copy()
            p[i, a] += h
            ep = eng.compute(lat, p, types, use_numba=True)[0]
            p[i, a] -= 2 * h
            em = eng.compute(lat, p, types, use_numba=True)[0]
            fd[i, a] = -(ep - em) / (2 * h)
    assert np.abs(F - fd).max() <= 1e-7 * np.abs(F).max()
    sfd = finite_diff_stress(
        lambda la, po: eng.compute(la, po, types, use_numba=True)[0], lat, pos
    )
    assert np.abs(S - sfd).max() <= 1e-7 * np.abs(S).max()


@pytest.mark.parametrize("case", ["si", "gesn"])
def test_analytic_kernel_agrees_with_the_finite_difference_fallback(case):
    eng, lat, pos, types = CASES[case]
    E1, F1, S1 = eng.compute(lat, pos, types, use_numba=True)
    from akitsushima.utils import jit

    env = jit._NUMBA_ENV
    jit._NUMBA_ENV = False  # the NumPy fallback, still central differences
    try:
        E0, F0, S0 = eng.compute(lat, pos, types, use_numba=False)
    finally:
        jit._NUMBA_ENV = env
    assert E1 == pytest.approx(E0, rel=1e-12)
    assert np.abs(F1 - F0).max() <= 1e-8
    assert np.abs(S1 - S0).max() <= 1e-8 * np.abs(S0).max()
