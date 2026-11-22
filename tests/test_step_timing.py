"""Regression: the per-step timer must cover the force evaluation.

FIRE and RFO in GeometryOptimiser and the FIRE loop in NEBOptimiser used
to lap the "step" timer right after writing the log row, before the force
call; the restarted timer was then overwritten by the next start(), so the
force evaluation fell outside every recorded step.  A 512-atom GeSn relax
reported "Mean / step 1.21 ms" while Total time / Steps was 77.7 ms.

Each test wraps an analytic force function in a fixed sleep and checks
that every recorded step lasts at least that long.  Before the fix the
recorded steps were microseconds.
"""

import time

import numpy as np
import pytest

from akitsushima.solvers import GeometryOptimiser, NEBChain, NEBOptimiser

SLEEP_S = 0.01  # emulated cost of one force call [s]


def _slow_double_well(h=1.0, w=1.0, x0=10.0):
    """Single-atom double well V = h (1 - ((x - x0)/w)^2)^2 whose every
    call sleeps SLEEP_S, standing in for an expensive MLIP evaluation."""

    def force_fn(lattice, positions, atom_types, neigh_list=None):
        time.sleep(SLEEP_S)
        x = positions[:, 0] - x0
        xi = x / w
        V = float(np.sum(h * (1.0 - xi**2) ** 2))
        F = np.zeros_like(positions)
        F[:, 0] = 4.0 * h * x / w**2 * (1.0 - xi**2)
        return V, F, np.zeros((3, 3))

    force_fn.r_cut = 5.0
    return force_fn


@pytest.mark.parametrize("method", ["fire", "rfo"])
def test_relax_step_time_includes_force_evaluation(method, tmp_path):
    go = GeometryOptimiser(
        np.eye(3) * 20.0,
        np.array([[9.3, 10.0, 10.0]]),  # off the minimum at x = 9
        ["X"],
        force_fn=_slow_double_well(),
        method=method,
        relax_cell=False,
        fmax=1e-12,  # never reached: every step takes a force call
        max_steps=3,
        output_dir=str(tmp_path),
        verbose=False,
        timing=True,
    )
    t = go.run()["timing"]
    assert t["n_steps"] == 3
    assert min(t["step_times_s"]) >= SLEEP_S, t["step_times_s"]
    # the steps now account for most of the run (the rest is the initial
    # force call and, for RFO, the finite-difference Hessian)
    assert sum(t["step_times_s"]) <= t["total_s"]


def test_neb_step_time_includes_chain_evaluation(tmp_path):
    fn = _slow_double_well()
    chain = NEBChain.linear_interpolation(
        np.array([[9.0, 10.0, 10.0]]),
        np.array([[11.0, 10.0, 10.0]]),
        np.eye(3) * 20.0,
        ["X"],
        fn,
        n_images=2,
        k_spring=5.0,
        climbing=True,
        r_cut=5.0,
    )
    opt = NEBOptimiser(
        chain,
        fmax=1e-12,
        max_steps=3,
        verbose=False,
        output_dir=str(tmp_path),
        timing=True,
    )
    t = opt.run()["timing"]
    assert t["n_steps"] == 3
    # one chain.evaluate() = one force call per intermediate image
    assert min(t["step_times_s"]) >= 2 * SLEEP_S, t["step_times_s"]
