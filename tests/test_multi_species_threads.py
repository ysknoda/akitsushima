"""Threaded multi-species SW / Tersoff kernels (ADR-0047).

- The multi-species SW three-body kernel and both multi-species Tersoff
  kernels have prange versions with per-thread buffers.  One thread runs
  the serial kernels (bit for bit the former path); more threads must
  agree with one to a relative 1e-12 (summation order only).
"""

import numpy as np
import pytest
from test_sw_analytic import _asymmetric_tomita, _strained
from test_sw_three_body_counting import _rattled

from akitsushima import PotentialEngine, StillingerWeberPotential
from akitsushima.engine import TersoffPotential
from akitsushima.utils.jit import _HAS_NUMBA

numba = pytest.importorskip("numba") if _HAS_NUMBA else None
pytestmark = pytest.mark.skipif(
    not _HAS_NUMBA or numba.config.NUMBA_NUM_THREADS < 2,
    reason="needs numba with at least two threads",
)

TERSOFF_XY = {
    ("X", "Y", "Y"): {
        "A": 1830.8,
        "B": 471.18,
        "lambda1": 2.4799,
        "lambda2": 1.7322,
        "R": 2.85,
        "D": 0.15,
        "beta": 1.1e-6,
        "n": 0.78734,
        "c": 100390.0,
        "d": 16.217,
        "h": -0.59825,
        "lambda3": 1.7322,
    },
    ("Y", "X", "X"): {
        "A": 1500.0,
        "B": 400.0,
        "lambda1": 2.2,
        "lambda2": 1.5,
        "R": 2.85,
        "D": 0.15,
        "beta": 0.9e-6,
        "n": 0.7,
        "c": 90000.0,
        "d": 14.0,
        "h": -0.55,
        "lambda3": 1.5,
    },
}


def _threads(eng_compute):
    n0 = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        one = eng_compute()
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
        many = eng_compute()
    finally:
        numba.set_num_threads(n0)
    return one, many


def _assert_close(one, many):
    (e1, f1, s1), (e2, f2, s2) = one, many
    assert e2 == pytest.approx(e1, rel=1e-12)
    assert np.abs(f2 - f1).max() <= 1e-12 * np.abs(f1).max()
    assert np.abs(s2 - s1).max() <= 1e-12 * np.abs(s1).max()


def test_multi_species_sw_threads_agree():
    lat, pos = _strained(*_rattled(seed=21, n=3, a=5.9), 3)
    types = list(np.random.default_rng(22).choice(["Ge", "Sn"], len(pos)))
    eng = PotentialEngine().add(
        StillingerWeberPotential(_asymmetric_tomita(), species=["Ge", "Sn"])
    )
    _assert_close(*_threads(lambda: eng.compute(lat, pos, types, use_numba=True)))


def test_multi_species_tersoff_threads_agree():
    lat, pos = _strained(*_rattled(seed=23, n=3, a=5.431), 4)
    # the two diamond sublattices as species X and Y, see
    # test_engine_forces.test_tersoff_multi_species_numba_matches_numpy
    types = (["X"] * 4 + ["Y"] * 4) * (len(pos) // 8)
    pot = TersoffPotential(TERSOFF_XY, species=["X", "Y"])
    _assert_close(*_threads(lambda: pot.compute(lat, pos, types, use_numba=True)))
