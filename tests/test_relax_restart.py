"""relax restart from a checkpoint (ADR-0031 decisions 1 and 10).

relax continues from the checkpointed structure with the optimiser's
own state started afresh, so the continued path is not the uninterrupted
one.  What must hold: the rows before the checkpoint are kept, the
checkpointed structure is logged again with the same energy, the step
count goes on without gaps or repeats, a finished run is not redone, a
run stopped at max_steps continues when max_steps is raised, and
different settings are refused.
"""

import os

import numpy as np
import pytest
from conftest import diamond_si_dict
from test_md_restart import SW_SI

import akitsushima as ak
from akitsushima.solvers.geometry_optimiser import GeometryOptimiser

METHODS = ["fire", "lbfgs", "diis", "rmmdiis", "rfo", "sd"]
EVERY, STOP_AT = 5, 12


class _Killed(RuntimeError):
    pass


def _structure():
    d = diamond_si_dict(2)  # 64 atoms
    rng = np.random.default_rng(1)
    pos = np.asarray(d["positions"], float) + rng.normal(0, 0.08, (64, 3))
    return np.asarray(d["lattice"], float), pos, list(d["atom_types"])


def _make(method, out, **over):
    lat, pos, types = _structure()
    kw = dict(
        force_fn=ak.force_fn_from_config(SW_SI),
        r_cut=3.77,
        method=method,
        fmax=1e-6,
        max_steps=30,
        output_dir=str(out),
        verbose=False,
        timing=False,
    )
    kw.update(over)
    return GeometryOptimiser(lat, pos, types, **kw)


def _kill_at(opt, gstep):
    real = opt._log_row

    def dying(log_fh, step, *a, **k):
        if step + opt._step_offset >= gstep:
            raise _Killed("simulated end of the job")
        return real(log_fh, step, *a, **k)

    opt._log_row = dying


def _rows(out):
    """The step rows of relax.log (the summary after a blank line is not one)."""
    lines = open(os.path.join(out, "relax.log")).read().splitlines()
    rows = []
    for ln in lines[1:]:
        if not ln.strip():
            break
        rows.append(ln.split())
    return rows


def _frames(out):
    text = open(os.path.join(out, "XDATCAR_relax")).read()
    return text.count("Direct configuration=")


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
def test_interrupted_relax_continues_from_the_checkpoint(tmp_path, method):
    out = tmp_path / method
    ck = str(out / "relax_checkpoint.npz")
    opt = _make(method, out)
    _kill_at(opt, STOP_AT)
    with pytest.raises(_Killed):
        opt.run(checkpoint=ck, checkpoint_every=EVERY)
    before = _rows(out)
    assert len(before) == STOP_AT  # rows 0..11, two past the step-10 checkpoint

    res = _make(method, out).run(checkpoint=ck, checkpoint_every=EVERY)
    after = _rows(out)
    steps = [int(r[0]) for r in after]
    assert steps == list(range(len(after)))  # no gap, no repeat
    assert after[:10] == before[:10]  # rows before the checkpoint kept
    assert after[10][:3] == before[10][:3]  # same structure, logged again
    assert _frames(out) == len(after)
    assert res["n_steps"] == len(after)
    assert res["energy"] <= float(before[10][1]) + 1e-9


@pytest.mark.slow
def test_finished_run_is_not_redone_and_max_steps_extends(tmp_path):
    ck = str(tmp_path / "relax_checkpoint.npz")
    first = _make("fire", tmp_path, max_steps=15).run(
        checkpoint=ck, checkpoint_every=EVERY
    )
    assert not first["converged"] and first["n_steps"] == 15
    log = open(tmp_path / "relax.log").read()
    # same max_steps: nothing to do, files untouched, same result
    again = _make("fire", tmp_path, max_steps=15).run(
        checkpoint=ck, checkpoint_every=EVERY
    )
    assert open(tmp_path / "relax.log").read() == log
    assert again["n_steps"] == 15 and again["energy"] == first["energy"]
    # raised max_steps: continues from the last structure
    more = _make("fire", tmp_path, max_steps=25).run(
        checkpoint=ck, checkpoint_every=EVERY
    )
    steps = [int(r[0]) for r in _rows(tmp_path)]
    assert steps == list(range(25)) and more["n_steps"] == 25
    assert more["energy"] <= first["energy"] + 1e-9

    # a converged run is returned as it is
    ck2 = str(tmp_path / "conv" / "relax_checkpoint.npz")
    done = _make("lbfgs", tmp_path / "conv", fmax=0.05).run(checkpoint=ck2)
    assert done["converged"]
    redo = _make("lbfgs", tmp_path / "conv", fmax=0.05, max_steps=500).run(
        checkpoint=ck2
    )
    assert redo["n_steps"] == done["n_steps"]
    assert np.array_equal(redo["positions"], done["positions"])


def test_relax_restart_with_different_settings_is_refused(tmp_path):
    ck = str(tmp_path / "relax_checkpoint.npz")
    _make("fire", tmp_path, max_steps=3).run(checkpoint=ck)
    with pytest.raises(ValueError, match="fmax"):
        _make("fire", tmp_path, max_steps=6, fmax=1e-3).run(checkpoint=ck)
    with pytest.raises(ValueError, match="method"):
        _make("lbfgs", tmp_path, max_steps=6).run(checkpoint=ck)


def test_checkpointing_does_not_change_the_path(tmp_path):
    a = _make("lbfgs", tmp_path / "a", max_steps=12).run(
        checkpoint=str(tmp_path / "a" / "ck.npz"), checkpoint_every=3
    )
    b = _make("lbfgs", tmp_path / "b", max_steps=12).run()
    assert np.array_equal(a["positions"], b["positions"])
    assert _rows(tmp_path / "a") == _rows(tmp_path / "b")
