"""SW and Tersoff refuse a cell that cannot host their cutoff.

Without a neighbour list both engines take minimum-image displacements
(`s -= np.round(s)`), which keep one image per pair: once the cutoff
reaches past half the smallest perpendicular cell width the other images
inside it are silently dropped.  BasePairPotential already raised there;
SW and Tersoff did not, and a differential-evolution SW fit
(example/fit/Si_sw) settled on sigma 7.7 A in a 10.5 A cell because of it
(ADR-0006, re-run table).
"""

import numpy as np
import pytest
from conftest import diamond_si_dict, randomly_perturbed

from akitsushima.engine import StillingerWeberPotential, TersoffPotential
from akitsushima.engine.neighborlist import allow_minimum_image_violation

# Stillinger & Weber, PRB 31, 5262 (1985): a * sigma = 3.771 A
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
# Tersoff T3 Si, PRB 38, 9902 (1988): R + D = 3.0 A
T3_SI = dict(
    A=1830.8,
    B=471.18,
    lambda1=2.4799,
    lambda2=1.7322,
    lambda3=1.7322,
    beta=1.1e-6,
    n=0.78734,
    c=100390.0,
    d=16.217,
    h=-0.59825,
    R=2.85,
    D=0.15,
)

POTENTIALS = {
    "sw": lambda: StillingerWeberPotential(dict(SW_SI), species=["Si"]),
    "sw_multi": lambda: StillingerWeberPotential(
        {("Si", "Si", "Si"): dict(SW_SI)}, species=["Si"]
    ),
    "tersoff": lambda: TersoffPotential(dict(T3_SI), species=["Si"]),
    "tersoff_multi": lambda: TersoffPotential(
        {("Si", "Si", "Si"): dict(T3_SI)}, species=["Si"]
    ),
}


def _args(n):
    s = randomly_perturbed(diamond_si_dict(n), magnitude=0.05, seed=0)
    return s["lattice"], s["positions"], s["atom_types"]


@pytest.mark.parametrize("use_numba", [False, True])
@pytest.mark.parametrize("name", list(POTENTIALS))
def test_too_small_cell_raises(name, use_numba):
    # 8-atom diamond cell: half-width 2.715 A, below both cutoffs.
    with pytest.raises(ValueError, match="minimum-image"):
        POTENTIALS[name]().compute(*_args(1), use_numba=use_numba)


@pytest.mark.parametrize("name", list(POTENTIALS))
def test_large_enough_cell_runs(name):
    E, F, S = POTENTIALS[name]().compute(*_args(2))
    assert np.isfinite(E) and np.all(np.isfinite(F)) and np.all(np.isfinite(S))


def test_sheared_cell_uses_the_perpendicular_width():
    """Lattice vectors 8.0 / 9.0 / 8.1 A long, all over twice SW's 3.771 A
    cutoff, but perpendicular widths 2.97 / 9.0 / 3.0 A."""
    lat = np.array([[8.0, 0.0, 0.0], [0.0, 9.0, 0.0], [7.5, 0.0, 3.0]])
    pos = np.random.default_rng(1).uniform(0, 1, (20, 3)) @ lat
    assert min(np.linalg.norm(lat, axis=1)) / 2 > 3.771
    with pytest.raises(ValueError, match="perpendicular"):
        POTENTIALS["sw"]().compute(lat, pos, ["Si"] * 20)


def test_the_violation_can_be_allowed_explicitly():
    with allow_minimum_image_violation():
        E, _, _ = POTENTIALS["sw"]().compute(*_args(1))
    assert np.isfinite(E)


def test_fit_loss_treats_the_refusal_as_a_failed_evaluation():
    """The fitting loss catches the exception, so a global search that
    wanders to a large sigma is penalised instead of rewarded."""
    from akitsushima.fitting import StillingerWeberParameterSet
    from akitsushima.fitting.loss import generic_loss_function

    lat, pos, types = _args(2)
    ps = StillingerWeberParameterSet(initial_sw=dict(SW_SI), species=["Si"])
    structs = [
        {
            "lattice": lat,
            "positions": pos,
            "atom_types": types,
            "natoms": len(types),
            "volume": float(np.linalg.det(lat)),
            "ref_energy": -4.3,
            "ref_forces": np.zeros_like(pos),
            "ref_stress": np.zeros((3, 3)),
        }
    ]
    w = {"energy": 1.0, "forces": 1.0, "stress": 0.1, "r0": 1.0, "eps0": 1.0}
    ok = generic_loss_function(ps.x0, ps, structs, w, r_cut=4.0, ewald_alpha=None)
    big = dict(SW_SI, sigma=7.7)
    ps_big = StillingerWeberParameterSet(initial_sw=big, species=["Si"])
    bad = generic_loss_function(
        ps_big.x0, ps_big, structs, w, r_cut=4.0, ewald_alpha=None
    )
    assert np.isfinite(ok) and bad > 1e3 * ok

