"""``anneal``: MD-quench annealing and basin hopping (ADR-0022).

* basin hopping finds the LJ13 global minimum, the Mackay icosahedron,
  E = -44.326801 epsilon (Wales & Doye, J. Phys. Chem. A 101, 5111 (1997));
* md_quench heals a defected SW silicon cell to the diamond energy;
* the enthalpy wrapper's stress is -(1/V) dH/d(strain) (finite differences),
  and a cell relaxed with it ends at the requested pressure;
* a restarted run reproduces the uninterrupted one;
* failed cycles are recorded and skipped, and enough in a row stop the run;
* initial_velocities continues an MD trajectory exactly.
"""

import json
import os

import numpy as np
import pytest
import yaml

from akitsushima import LennardJonesPotential
from akitsushima.cli.templates import _YAML_RELAX_DEFAULT
from akitsushima.constants import EV_ANG3_TO_GPA, K_B
from akitsushima.solvers import GeometryOptimiser, NVTSimulation
from akitsushima.solvers.config_bridge import _build_force_fn_from_yaml
from akitsushima.solvers.global_opt import (
    GlobalOptimiser,
    enthalpy_force_fn,
    temperature_schedule,
)

LJ13_GLOBAL_MIN = -44.326801  # epsilon, Wales & Doye 1997 (Cambridge Cluster Database)


def _lj_unshifted():
    pot = LennardJonesPotential(
        {frozenset({"X"}): {"epsilon": 1.0, "sigma": 1.0}}, r_cut=4.0, shift=False
    )

    def fn(lattice, positions, atom_types, **kw):
        return pot.compute(lattice, positions, atom_types)[:3]

    return fn, 4.0


def _lj13_start(seed=0):
    rng = np.random.default_rng(seed)
    pos = []
    while len(pos) < 13:
        p = rng.uniform(-1.3, 1.3, 3)
        if all(np.linalg.norm(p - q) > 0.95 for q in pos):
            pos.append(p)
    # 12 A box: wider than 2 x (r_cut 4 + skin 0.5), so no image interacts.
    return np.eye(3) * 12.0, np.array(pos) + 6.0


def _sw_si():
    pot = yaml.safe_load(_YAML_RELAX_DEFAULT)["potential"]
    fn, rc, *_ = _build_force_fn_from_yaml(pot)
    return fn, rc


def _diamond_si(n=2, a=5.431, jitter=0.0, seed=3):
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    frac = np.vstack([base, base + 0.25])
    cells = np.array(
        [[i, j, k] for i in range(n) for j in range(n) for k in range(n)], float
    )
    frac = (frac[None] + cells[:, None]).reshape(-1, 3) / n
    lat = np.eye(3) * a * n
    pos = frac @ lat
    if jitter:
        pos = pos + np.random.default_rng(seed).uniform(-jitter, jitter, pos.shape)
    return lat, pos


def _ledger(out):
    with open(os.path.join(out, "anneal_log.jsonl")) as fh:
        return [json.loads(line) for line in fh]


def _bh(tmp_path, name, n_trials, restart=False, **kw):
    fn, rc = _lj_unshifted()
    lat, pos = _lj13_start()
    opts = dict(
        method="basin_hopping",
        output_dir=str(tmp_path / name),
        relax={"method": "lbfgs", "fmax": 1e-3, "max_steps": 2000},
        n_trials=n_trials,
        temperature=0.8 / K_B,  # kT = 0.8 epsilon, Wales & Doye's choice
        step=0.36,
        seed=1,
        restart=restart,
        verbose=False,
        plot=False,
    )
    opts.update(kw)
    return GlobalOptimiser(lat, pos, ["X"] * 13, fn, rc, **opts).run()


# ------------------------------------------------------------- schedule
def test_temperature_schedules():
    np.testing.assert_allclose(temperature_schedule("constant", 3, 500.0), [500] * 3)
    np.testing.assert_allclose(
        temperature_schedule("linear", 3, 900.0, 300.0), [900, 600, 300]
    )
    g = temperature_schedule("geometric", 3, 1200.0, 300.0)
    np.testing.assert_allclose(g, [1200, 600, 300])
    np.testing.assert_allclose(
        temperature_schedule("list", 0, temperatures=[800, 400]), [800, 400]
    )
    with pytest.raises(ValueError):
        temperature_schedule("geometric", 3, 1000.0, 0.0)
    with pytest.raises(ValueError):
        temperature_schedule("cosine", 3, 1000.0, 300.0)


# ------------------------------------------------------------- basin hopping
def test_basin_hopping_finds_the_lj13_icosahedron(tmp_path):
    res = _bh(tmp_path, "bh", 60)
    assert res["best"]["E"] == pytest.approx(LJ13_GLOBAL_MIN, abs=1e-4)
    # The start (cycle 0) is a higher local minimum.
    led = _ledger(tmp_path / "bh")
    assert led[0]["kind"] == "initial" and led[0]["E"] > LJ13_GLOBAL_MIN + 0.1
    assert len(led) == 61 and all(r["status"] == "ok" for r in led)
    assert os.path.isfile(tmp_path / "bh" / "POSCAR_best")
    assert os.path.isfile(tmp_path / "bh" / "minima.csv")
    # Distinct minima, sorted, none closer than the dedup tolerance.
    h = [m["H"] / 13 for m in res["minima"]]
    assert h == sorted(h)


def test_basin_hopping_step_adapts_towards_the_target_acceptance(tmp_path):
    # LJ13 at kT = 0.8 accepts almost every move, so the step must grow.
    _bh(tmp_path, "bh", 30)
    steps = [r["step"] for r in _ledger(tmp_path / "bh") if r["kind"] == "trial"]
    assert steps[0] == pytest.approx(0.36)
    assert steps[-1] > steps[0]
    # Updated only every adapt_every = 10 trials.
    assert len(set(steps[:10])) == 1


def test_restart_reproduces_the_uninterrupted_run(tmp_path):
    full = _bh(tmp_path, "full", 16)
    _bh(tmp_path, "split", 7)
    resumed = _bh(tmp_path, "split", 16, restart=True)
    a = _ledger(tmp_path / "full")
    b = _ledger(tmp_path / "split")
    assert [r["cycle"] for r in b] == list(range(17))
    for ra, rb in zip(a, b):
        assert ra.get("accepted") == rb.get("accepted")
        assert ra["H"] == pytest.approx(rb["H"], abs=1e-9)
        assert ra.get("step") == pytest.approx(rb.get("step"))
    assert resumed["best"]["H"] == pytest.approx(full["best"]["H"], abs=1e-9)


def test_restart_drops_ledger_lines_after_the_checkpoint(tmp_path):
    _bh(tmp_path, "bh", 5)
    out = tmp_path / "bh"
    with open(out / "anneal_log.jsonl", "a") as fh:  # a line the checkpoint never saw
        fh.write(json.dumps({"cycle": 6, "kind": "trial", "H": 0.0}) + "\n")
    _bh(tmp_path, "bh", 8, restart=True)
    assert [r["cycle"] for r in _ledger(out)] == list(range(9))


def test_failed_trials_are_logged_and_skipped(tmp_path, monkeypatch):
    real = GlobalOptimiser._quench

    def flaky(self, lat, pos, out_dir):
        if out_dir.endswith(("c00003", "c00004")):
            raise RuntimeError("synthetic solver failure")
        return real(self, lat, pos, out_dir)

    monkeypatch.setattr(GlobalOptimiser, "_quench", flaky)
    res = _bh(tmp_path, "bh", 8)
    led = _ledger(tmp_path / "bh")
    assert [r["status"] for r in led] == ["ok"] * 3 + ["failed"] * 2 + ["ok"] * 4
    assert "synthetic" in led[3]["error"]
    # The walker stayed where it was across the failures.
    assert led[5]["current_H_per_atom"] is not None
    assert res["n_failed"] == 2 and not res["aborted"]


def test_consecutive_failures_stop_the_run(tmp_path, monkeypatch):
    real = GlobalOptimiser._quench

    def broken(self, lat, pos, out_dir):
        if not out_dir.endswith("c00000") and not out_dir.endswith("c00001"):
            raise RuntimeError("synthetic collapse")
        return real(self, lat, pos, out_dir)

    monkeypatch.setattr(GlobalOptimiser, "_quench", broken)
    res = _bh(tmp_path, "bh", 20, max_consecutive_failures=3)
    assert res["aborted"]
    led = _ledger(tmp_path / "bh")
    assert [r["cycle"] for r in led] == [0, 1, 2, 3, 4]
    assert res["best"] is not None  # what was found before is still written
    assert os.path.isfile(tmp_path / "bh" / "POSCAR_best")


def test_min_energy_per_atom_marks_a_collapse_as_failed(tmp_path):
    # The icosahedron (-3.41 eV/atom) is "below" a floor of -3.40: every
    # quench that reaches it fails, as an MLIP collapse would.
    res = _bh(
        tmp_path, "bh", 30, min_energy_per_atom=-3.40, max_consecutive_failures=30
    )
    led = _ledger(tmp_path / "bh")
    assert all(r["E"] / 13 >= -3.40 for r in led if r["status"] == "ok")
    assert any(r["status"] == "failed" and "below" in r["error"] for r in led)
    assert res["best"]["E"] / 13 >= -3.40


# ------------------------------------------------------------- md_quench
def test_md_quench_heals_a_defected_silicon_cell(tmp_path):
    fn, rc = _sw_si()
    lat, pos = _diamond_si(jitter=0.8)
    res = GlobalOptimiser(
        lat,
        pos,
        ["Si"] * 64,
        fn,
        rc,
        method="md_quench",
        output_dir=str(tmp_path / "mq"),
        relax={"method": "fire", "fmax": 0.05, "max_steps": 3000},
        md={
            "ensemble": "nvt",
            "timestep": 1.0,
            "tau_nhc": 20.0,
            "traj_every": 100,
            "thermo_every": 100,
        },
        schedule={"type": "constant", "n_segments": 1, "T_start": 1300.0},
        segment_steps=1000,
        quench_every=1000,
        seed=5,
        verbose=False,
        plot=False,
    ).run()
    led = _ledger(tmp_path / "mq")
    e_relax_only = led[0]["E"] / 64
    # SW diamond Si: -4.3366 eV/atom at a = 5.431 A (fmax 0.05 leaves ~1e-4).
    assert e_relax_only > -4.30
    assert res["best"]["E"] / 64 == pytest.approx(-4.3366, abs=1e-3)
    assert led[1]["T"] == 1300.0


def test_md_quench_continues_the_trajectory_and_redraws_after_an_md_failure(
    tmp_path, monkeypatch
):
    fn, rc = _sw_si()
    lat, pos = _diamond_si(jitter=0.05)
    calls = []
    real = GlobalOptimiser._md_chunk

    def spy(self, lat_, pos_, vel, T, cycle, out_dir):
        calls.append((cycle, None if vel is None else vel.copy(), pos_.copy()))
        if cycle == 3:
            raise RuntimeError("synthetic MD blow-up")
        out = real(self, lat_, pos_, vel, T, cycle, out_dir)
        calls[-1] = calls[-1] + (out[1].copy(), out[2].copy())
        return out

    monkeypatch.setattr(GlobalOptimiser, "_md_chunk", spy)
    res = GlobalOptimiser(
        lat,
        pos,
        ["Si"] * 64,
        fn,
        rc,
        method="md_quench",
        output_dir=str(tmp_path / "mq"),
        relax={"method": "fire", "fmax": 0.05},
        md={"ensemble": "nvt", "traj_every": 10, "thermo_every": 10},
        schedule={"type": "linear", "n_segments": 2, "T_start": 600.0, "T_end": 300.0},
        segment_steps=40,
        quench_every=20,
        seed=2,
        verbose=False,
        plot=False,
    ).run()
    assert [c[0] for c in calls] == [1, 2, 3, 4]
    assert calls[0][1] is None  # first chunk: MB draw
    np.testing.assert_array_equal(calls[1][1], calls[0][4])  # chunk 2 continues chunk 1
    np.testing.assert_array_equal(calls[1][2], calls[0][3])
    assert calls[3][1] is None  # after the failure: fresh draw...
    np.testing.assert_array_equal(
        calls[3][2], calls[1][3]
    )  # ...from the last good state
    led = _ledger(tmp_path / "mq")
    assert [r["status"] for r in led] == ["ok", "ok", "ok", "failed", "ok"]
    assert led[3]["stage"] == "md"
    assert [r.get("T") for r in led[1:]] == [600.0, 600.0, 300.0, 300.0]
    assert res["n_failed"] == 1


def test_npt_md_quench_runs_with_cell_relaxation(tmp_path):
    fn, rc = _sw_si()
    lat, pos = _diamond_si(a=5.50, jitter=0.05)
    res = GlobalOptimiser(
        lat,
        pos,
        ["Si"] * 64,
        fn,
        rc,
        method="md_quench",
        output_dir=str(tmp_path / "mq"),
        relax={
            "method": "fire",
            "fmax": 0.01,
            "pmax": 0.05,
            "relax_cell": True,
            "max_steps": 3000,
        },
        md={"ensemble": "npt", "traj_every": 10, "thermo_every": 10},
        schedule={"type": "constant", "n_segments": 1, "T_start": 300.0},
        segment_steps=20,
        quench_every=20,
        seed=2,
        verbose=False,
        plot=False,
    ).run()
    # SW diamond Si relaxes back to a = 5.431 A: V = 8 x 160.2 A^3 for 64 atoms.
    assert res["best"]["V"] == pytest.approx(8 * 5.431**3, rel=2e-3)


def test_initial_velocities_continue_an_nvt_run(tmp_path):
    fn, rc = _sw_si()
    lat, pos = _diamond_si(jitter=0.05)
    common = dict(
        force_fn=fn,
        r_cut=rc,
        temperature=500.0,
        # A chain this slow exerts no force over 30 fs, so the restart of
        # the chain between the two halves cannot hide a wrong velocity.
        tau_nhc=1e9,
        traj_every=5,
        thermo_every=5,
        verbose=False,
        save_plots=False,
    )
    one = NVTSimulation(
        lat,
        pos,
        ["Si"] * 64,
        n_steps=30,
        seed=4,
        output_dir=str(tmp_path / "a"),
        **common,
    ).run()
    first = NVTSimulation(
        lat,
        pos,
        ["Si"] * 64,
        n_steps=15,
        seed=4,
        output_dir=str(tmp_path / "b"),
        **common,
    ).run()
    second = NVTSimulation(
        first["lattice"],
        first["positions"],
        ["Si"] * 64,
        n_steps=15,
        seed=99,
        initial_velocities=first["velocities"],
        output_dir=str(tmp_path / "c"),
        **common,
    ).run()
    # Continuing with the first half's velocities retraces the 30-step run;
    # the second run's own seed (99) never enters.
    np.testing.assert_allclose(second["positions"], one["positions"], atol=1e-9)
    np.testing.assert_allclose(second["velocities"], one["velocities"], atol=1e-11)
    assert not os.path.isfile(tmp_path / "a" / "md_timeseries.png")
    with pytest.raises(ValueError, match="initial_velocities"):
        NVTSimulation(
            lat,
            pos,
            ["Si"] * 64,
            initial_velocities=np.zeros((3, 3)),
            output_dir=str(tmp_path / "d"),
            **common,
        )


# ------------------------------------------------------------- pressure
def test_enthalpy_wrapper_stress_matches_finite_differences():
    fn, rc = _sw_si()
    lat, pos = _diamond_si(jitter=0.1)
    P = 5.0  # GPa
    h = enthalpy_force_fn(fn, P)
    H0, F0, S0 = h(lat, pos, ["Si"] * 64)
    E0, Fr, Sr = fn(lat, pos, ["Si"] * 64)
    V = abs(np.linalg.det(lat))
    assert H0 == pytest.approx(E0 + P / EV_ANG3_TO_GPA * V)
    np.testing.assert_array_equal(F0, Fr)
    frac = pos @ np.linalg.inv(lat)
    eps = 1e-5
    for a, b in [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]:
        strain = np.zeros((3, 3))
        strain[a, b] = strain[b, a] = eps / (1 if a == b else 2)

        def H_at(sgn):
            L = lat @ (np.eye(3) + sgn * strain)
            return h(L, frac @ L, ["Si"] * 64)[0]

        # sigma = -(1/V) dH/d(eps), compression positive (ADR-0001)
        dHde = (H_at(+1) - H_at(-1)) / (2 * eps)
        # off-diagonal: eps is the engineering shear, dH/d(gamma) = -V sigma_ab
        assert S0[a, b] == pytest.approx(-dHde / V, abs=2e-6), (a, b)


def test_cell_relaxed_under_pressure_ends_at_that_pressure(tmp_path):
    fn, rc = _sw_si()
    lat, pos = _diamond_si()
    P = 5.0
    res = GeometryOptimiser(
        lat,
        pos,
        ["Si"] * 64,
        force_fn=enthalpy_force_fn(fn, P),
        r_cut=rc,
        relax_cell=True,
        fmax=1e-4,
        pmax=1e-3,
        max_steps=5000,
        verbose=False,
        output_dir=str(tmp_path),
    ).run()
    assert res["converged"]
    S = fn(res["lattice"], res["positions"], ["Si"] * 64)[2] * EV_ANG3_TO_GPA
    np.testing.assert_allclose(np.diag(S), [P] * 3, atol=2e-3)
    assert abs(np.linalg.det(res["lattice"])) < 8 * 5.431**3  # compressed


def test_bad_settings_are_refused():
    fn, rc = _lj_unshifted()
    lat, pos = _lj13_start()
    with pytest.raises(ValueError, match="multiple"):
        GlobalOptimiser(
            lat,
            pos,
            ["X"] * 13,
            fn,
            rc,
            method="md_quench",
            segment_steps=100,
            quench_every=30,
        )
    with pytest.raises(ValueError, match="method"):
        GlobalOptimiser(lat, pos, ["X"] * 13, fn, rc, method="simulated")
    with pytest.raises(ValueError, match="ensemble"):
        GlobalOptimiser(lat, pos, ["X"] * 13, fn, rc, md={"ensemble": "nve"})


# ------------------------------------------------------------- CLI
def _cli_cfg(tmp_path, method, **anneal):
    from akitsushima.cli.templates import _YAML_ANNEAL_DEFAULT
    from akitsushima.io.poscar import write_poscar

    cfg = yaml.safe_load(_YAML_ANNEAL_DEFAULT)
    lat, pos = _diamond_si(jitter=0.05)
    write_poscar(str(tmp_path / "POSCAR"), lat, pos, ["Si"] * 64)
    cfg["structure"]["poscar"] = str(tmp_path / "POSCAR")
    cfg["anneal"].update(method=method, output_dir=str(tmp_path / "out"), **anneal)
    path = tmp_path / "anneal.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def test_cli_basin_hopping_from_the_template(tmp_path):
    from test_cli_smoke import _ns

    from akitsushima.cli.parser import _DISPATCH

    path, cfg = _cli_cfg(tmp_path, "basin_hopping")
    cfg["anneal"]["basin_hopping"]["n_trials"] = 3
    path.write_text(yaml.safe_dump(cfg))
    assert _DISPATCH["anneal"](_ns(path)) == 0
    led = _ledger(tmp_path / "out")
    assert [r["kind"] for r in led] == ["initial", "trial", "trial", "trial"]
    for f in ("POSCAR_best", "POSCAR_rank01", "minima.csv", "anneal.png", "checkpoint.npz"):
        assert (tmp_path / "out" / f).is_file(), f


def test_cli_refuses_unknown_keys(tmp_path, capsys):
    from test_cli_smoke import _ns

    from akitsushima.cli.parser import _DISPATCH

    path, cfg = _cli_cfg(tmp_path, "md_quench")
    cfg["anneal"]["relax"]["fmaxx"] = 0.01
    path.write_text(yaml.safe_dump(cfg))
    assert _DISPATCH["anneal"](_ns(path)) == 1
    assert "fmaxx" in capsys.readouterr().out


def test_md_collapse_below_the_energy_floor_rewinds_the_trajectory(tmp_path, monkeypatch):
    """An MLIP that collapses during an MD chunk raises nothing; the chunk's
    last frame is checked against min_energy_per_atom like a quench is."""
    fn, rc = _sw_si()
    collapse = {"on": False}

    def deep(lattice, positions, atom_types, **kw):
        e, f, s = fn(lattice, positions, atom_types, **kw)
        return (e - 1000.0 if collapse["on"] else e), f, s

    real_run = NVTSimulation.run
    n_runs = {"n": 0}

    def run(self):
        n_runs["n"] += 1
        out = real_run(self)
        collapse["on"] = n_runs["n"] == 2   # the second chunk "collapses"
        return out

    monkeypatch.setattr(NVTSimulation, "run", run)
    calls = []
    real_chunk = GlobalOptimiser._md_chunk

    def spy(self, lat_, pos_, vel, T, cycle, out_dir):
        calls.append((cycle, vel is None, pos_.copy()))
        try:
            return real_chunk(self, lat_, pos_, vel, T, cycle, out_dir)
        finally:
            collapse["on"] = False

    monkeypatch.setattr(GlobalOptimiser, "_md_chunk", spy)
    lat, pos = _diamond_si(jitter=0.05)
    GlobalOptimiser(
        lat, pos, ["Si"] * 64, deep, rc, method="md_quench",
        output_dir=str(tmp_path / "mq"),
        relax={"method": "fire", "fmax": 0.05},
        md={"ensemble": "nvt", "traj_every": 10, "thermo_every": 10},
        schedule={"type": "constant", "n_segments": 1, "T_start": 300.0},
        segment_steps=60, quench_every=20, seed=2,
        min_energy_per_atom=-10.0, verbose=False, plot=False,
    ).run()
    led = _ledger(tmp_path / "mq")
    assert [r["status"] for r in led] == ["ok", "ok", "failed", "ok"]
    assert led[2]["stage"] == "md" and "below" in led[2]["error"]
    # Chunk 3 restarts from chunk 1's end state with fresh velocities.
    assert calls[2][1] is True
    np.testing.assert_array_equal(calls[2][2], calls[1][2])
