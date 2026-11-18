"""md restart from a checkpoint (ADR-0031).

An NVT, NVE or NPT run interrupted after its last checkpoint and
restarted must give the same bits as the run that was not interrupted:
final state, averages and the output files (the NVT thermo column
NL_builds counts the builds of the neighbour list in each process, so it
is left out of that comparison).  A finished run can be extended by
raising n_steps, and a restart with different settings is refused.
"""

import os

import numpy as np
import pytest
from conftest import diamond_si_dict

import akitsushima as ak
from akitsushima.solvers.molecular_dynamics import (
    NPTSimulation,
    NVESimulation,
    LangevinSimulation,
    NVTSimulation,
)

# Stillinger and Weber (1985) silicon, as in benchmarks/bench_force.py
SW_SI = {
    "type": "SW",
    "species": ["Si"],
    "params": {
        "epsilon": 2.1683,
        "sigma": 2.0951,
        "a": 1.80,
        "gamma": 1.20,
        "lambda": 21.0,
        "A": 7.049556277,
        "B": 0.6022245584,
        "p": 4.0,
        "q": 0.0,
        "costheta0": -1 / 3,
    },
}
CLASSES = {
    "nvt": NVTSimulation,
    "nve": NVESimulation,
    "npt": NPTSimulation,
    "langevin": LangevinSimulation,  # ADR-0032: the RNG state is restored
}
EQUIL, PROD, EVERY = 20, 60, 25
STOP_AT = 37  # force call at which the "job" dies: past the step-25 checkpoint


class _Killed(RuntimeError):
    pass


def _structure():
    d = diamond_si_dict(2)  # 64 atoms
    rng = np.random.default_rng(0)
    pos = np.asarray(d["positions"], float) + rng.normal(0, 0.03, (64, 3))
    return np.asarray(d["lattice"], float), pos, list(d["atom_types"])


def _make(ens, out, n_steps=PROD, **over):
    lat, pos, types = _structure()
    kw = dict(
        force_fn=ak.force_fn_from_config(SW_SI),
        r_cut=3.77,
        timestep=1.0,
        n_steps=n_steps,
        equil_steps=EQUIL,
        output_dir=str(out),
        thermo_every=5,
        traj_every=5,
        seed=7,
        verbose=False,
        temperature=900.0,
    )
    if ens in ("nvt", "npt"):
        kw.update(tau_nhc=50.0, velocity_rescale_steps=10, save_plots=False)
    if ens == "langevin":
        kw.update(damping=40.0, save_plots=False)
    if ens == "npt":
        kw.update(pressure=1.0, tau_baro=300.0)
    kw.update(over)
    return CLASSES[ens](lat, pos, types, **kw)


def _kill_after(sim, n_calls):
    real = sim._compute_forces
    count = {"n": 0}

    def dying():
        count["n"] += 1
        if count["n"] > n_calls:
            raise _Killed("simulated end of the job")
        return real()

    sim._compute_forces = dying


def _files(out, ens):
    thermo = open(os.path.join(out, "thermo.log")).read().splitlines()
    if ens in ("nvt", "langevin"):  # drop NL_builds, the last column
        thermo = [line.rsplit(None, 1)[0] for line in thermo]
    name = "nvt" if ens == "langevin" else ens
    traj = open(os.path.join(out, f"XDATCAR_{name}"), "rb").read()
    return thermo, traj


def _same_result(a, b):
    for k in ("positions", "velocities", "lattice"):
        assert np.array_equal(a[k], b[k]), k
    for k, v in a.items():
        if isinstance(v, float):
            assert v == b[k] or (np.isnan(v) and np.isnan(b[k])), k


@pytest.mark.slow
@pytest.mark.parametrize("ens", sorted(CLASSES))
def test_interrupted_run_continues_bit_for_bit(tmp_path, ens):
    ref = _make(ens, tmp_path / "ref").run()

    out = tmp_path / "ck"
    ck = str(out / "md_checkpoint.npz")
    sim = _make(ens, out)
    _kill_after(sim, STOP_AT + 1)  # +1: the force call before the loop
    with pytest.raises(_Killed):
        sim.run(checkpoint=ck, checkpoint_every=EVERY)
    # the files hold lines past the checkpoint, which a restart must drop
    assert len(_files(out, ens)[0]) > 1 + EVERY // 5
    got = _make(ens, out).run(checkpoint=ck, checkpoint_every=EVERY)

    _same_result(got, ref)
    assert _files(out, ens) == _files(tmp_path / "ref", ens)


@pytest.mark.slow
@pytest.mark.parametrize("ens", ["nvt", "npt", "langevin"])
def test_finished_run_is_extended_by_raising_n_steps(tmp_path, ens):
    ref = _make(ens, tmp_path / "ref").run()
    out = tmp_path / "ext"
    ck = str(out / "md_checkpoint.npz")
    _make(ens, out, n_steps=PROD - 25).run(checkpoint=ck)
    got = _make(ens, out).run(checkpoint=ck)
    _same_result(got, ref)
    assert _files(out, ens) == _files(tmp_path / "ref", ens)


def test_restart_with_different_settings_is_refused(tmp_path):
    ck = str(tmp_path / "md_checkpoint.npz")
    meta = {"potential_hash": "a"}
    _make("nvt", tmp_path, n_steps=5).run(checkpoint=ck, checkpoint_meta=meta)
    with pytest.raises(ValueError, match="temperature"):
        _make("nvt", tmp_path, n_steps=10, temperature=600.0).run(
            checkpoint=ck, checkpoint_meta=meta
        )
    with pytest.raises(ValueError, match="potential_hash"):
        _make("nvt", tmp_path, n_steps=10).run(
            checkpoint=ck, checkpoint_meta={"potential_hash": "other"}
        )
    with pytest.raises(ValueError, match="potential_hash"):  # given, then not
        _make("nvt", tmp_path, n_steps=10).run(checkpoint=ck)
    with pytest.raises(ValueError, match="ensemble"):
        _make("npt", tmp_path, n_steps=10).run(checkpoint=ck, checkpoint_meta=meta)
    # n_steps may not fall below the checkpoint's step
    with pytest.raises(ValueError, match="n_steps can only be"):
        _make("nvt", tmp_path, n_steps=2).run(checkpoint=ck, checkpoint_meta=meta)


def test_missing_checkpoint_file_starts_a_fresh_run(tmp_path):
    ck = str(tmp_path / "md_checkpoint.npz")
    a = _make("nve", tmp_path / "a", n_steps=5).run(checkpoint=ck, checkpoint_every=2)
    b = _make("nve", tmp_path / "b", n_steps=5).run()
    _same_result(a, b)
    assert os.path.isfile(ck)
