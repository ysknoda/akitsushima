"""Per-run temperatures and per-run outputs of independent runs (ADR-0011 A2, A3)."""

import json
import warnings

import numpy as np
import pytest

from akitsushima import mdspec as nozuchi
from akitsushima.mdspec import charges as ch
from akitsushima.mdspec import config

from .conftest import write_xdatcar

N_FRAMES = 4000  # every 4th step of the shared Langevin crystal, 16 ps


def _outcar(path, temps):
    path.write_text(
        "".join(f"  kinetic energy EKIN   =  1.0  (temperature  {t:8.2f} K)\n" for t in temps)
    )


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory, crystal_traj):
    d = tmp_path_factory.mktemp("runs")
    pos, lattice, symbols, *_ = crystal_traj
    frac = (pos[: 4 * N_FRAMES : 4] @ np.linalg.inv(lattice)) % 1.0
    n_mg = symbols.count("Mg")
    for name in ("XA", "XB"):  # the same trajectory twice
        write_xdatcar(d / name, ["Mg", "O"], [n_mg, len(symbols) - n_mg], frac, lattice)
    return d


def _cfg(d, sources, prefix, estimator="fsum", skip=0):
    return config.validate(
        {
            "trajectory": {
                "files": ["XA", "XB"],
                "mode": "independent",
                "timestep_fs": 1.0,
                "frame_every": 4,
                "skip_frames": skip,
            },
            "temperature": "auto",
            "temperature_source": sources,
            "masses": {"Mg": 24.305, "O": 15.999},
            "dipole": {"model": "point_charge", "charges": {"Mg": 2.0, "O": -2.0}},
            "epsilon_inf": 1.0,
            "spectrum": {"max_lag_ps": 1.0, "estimator": estimator, "kk_check": False},
            "output": {"dir": "out", "prefix": prefix, "plot": False},
        },
        d,
    )


def test_per_run_temperature(two_runs):
    d = two_runs
    n_steps = 4 * N_FRAMES
    _outcar(d / "OUTCAR_A", np.full(n_steps, 300.0))
    _outcar(d / "OUTCAR_B", np.full(n_steps, 600.0))
    _outcar(d / "OUTCAR_C", np.full(n_steps, 300.0))
    quiet = lambda *a: None
    same = nozuchi.run(_cfg(d, ["OUTCAR_A", "OUTCAR_C"], "same"), log=quiet)
    diff = nozuchi.run(_cfg(d, ["OUTCAR_A", "OUTCAR_B"], "diff"), log=quiet)
    assert diff["temperature_K_per_run"] == [300.0, 600.0]
    assert diff["temperature_K"] == pytest.approx(450.0)
    # raw: Delta eps ~ 1/T per run, the mean of (1 + 1/2) / 2; fsum: T cancels
    d_same = np.diag(same["eps_static_raw"]) - 1
    d_diff = np.diag(diff["eps_static_raw"]) - 1
    assert np.allclose(d_diff, 0.75 * d_same, rtol=1e-10)
    assert np.allclose(diff["eps_static_fsum"], same["eps_static_fsum"], rtol=1e-10)


def test_per_file_skip_of_temperatures(two_runs):
    d = two_runs
    n_steps = 4 * N_FRAMES
    hot = np.full(n_steps, 300.0)
    hot[: 4 * 100] = 1000.0  # the first 100 frames are skipped in each run
    _outcar(d / "OUTCAR_H1", hot)
    _outcar(d / "OUTCAR_H2", hot)
    rep = nozuchi.run(_cfg(d, ["OUTCAR_H1", "OUTCAR_H2"], "skip", skip=100), log=lambda *a: None)
    assert rep["temperature_K_per_run"] == [300.0, 300.0]


def test_source_count_mismatch_falls_back_with_warning(two_runs):
    d = two_runs
    _outcar(d / "OUTCAR_A", np.full(4 * N_FRAMES, 300.0))
    rep = nozuchi.run(_cfg(d, ["OUTCAR_A"], "one"), log=lambda *a: None)
    assert rep["temperature_K_per_run"] == [300.0, 300.0]
    assert any("one mean temperature" in w for w in rep["warnings"])


def test_independent_runs_write_per_run_outputs(two_runs):
    d = two_runs
    _outcar(d / "OUTCAR_A", np.full(4 * N_FRAMES, 300.0))
    _outcar(d / "OUTCAR_C", np.full(4 * N_FRAMES, 300.0))
    cfg = _cfg(d, ["OUTCAR_A", "OUTCAR_C"], "files")
    cfg["output"]["plot"] = True
    nozuchi.run(cfg, log=lambda *a: None)
    out = d / "out"
    assert (out / "files_dipole_run1.dat").exists() and (out / "files_dipole_run2.dat").exists()
    assert not (out / "files_dipole.dat").exists()
    z = np.load(out / "files.npz")
    assert int(z["n_runs"]) == 2 and np.array_equal(z["dipole_run2"], z["dipole_run1"])
    assert (out / "files_dielectric.png").exists()
    res = json.loads((out / "files_results.json").read_text())
    assert len(res["temperature_K_per_run"]) == 2


def test_thermo_log_detected_by_content(tmp_path):
    (tmp_path / "md.log").write_text("Step Time_ps T_inst_K E_tot\n1 0.001 300.0 -5.0\n")
    (tmp_path / "thermo_notes.txt").write_text("not a log\n")
    assert ch.is_thermo_log(tmp_path / "md.log")
    assert not ch.is_thermo_log(tmp_path / "thermo_notes.txt")
    assert not ch.is_thermo_log(tmp_path / "missing")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        steps, t = ch.read_thermo_log(tmp_path / "md.log")
    assert steps.tolist() == [1] and t.tolist() == [300.0]
