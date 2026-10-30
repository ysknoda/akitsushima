"""VASP readers (akitsushima.io.vasp) and the ``dataset`` command (ADR-0021)."""

import csv
import os
import warnings

import numpy as np
import pytest
import yaml
from test_cli_smoke import _ns
from vasp_synth import make_frames, write_outcar, write_run, write_vasprun

from akitsushima.cli.cmd_dataset import cmd_dataset
from akitsushima.io.dataset import _split_runs
from akitsushima.cli.cmd_fit import _structures_from_csv
from akitsushima.constants import VASP_KB_TO_EV_ANG3
from akitsushima.fitting.models.csv_io import load_mlip_csv
from akitsushima.io import (
    build_training_set,
    parse_poscar,
    read_outcar,
    read_vasprun,
)

REAL = os.path.join(os.path.dirname(__file__), "data", "vasp_gesn")
ENERGIES = ("energy_free", "energy_wo_entropy", "energy_sigma0")


# ---------------------------------------------------------------- readers


@pytest.mark.parametrize("which", ["vasprun", "outcar"])
def test_reader_recovers_synthetic_frames(tmp_path, which):
    frames = make_frames([("Ge", 5), ("Sn", 3)], n_steps=3)
    if which == "vasprun":
        write_vasprun(tmp_path / "f", frames)
        got = read_vasprun(str(tmp_path / "f"))
        tol = 1e-7
    else:
        write_outcar(tmp_path / "f", frames)
        got = read_outcar(str(tmp_path / "f"))
        tol = 1e-5  # OUTCAR prints 5-6 decimals
    assert len(got) == 3
    for f, g in zip(frames, got):
        assert g["atom_types"] == f["atom_types"] and g["natoms"] == 8
        for e in ENERGIES:
            assert g[e] == pytest.approx(f[e], abs=1e-7)
        assert np.allclose(g["lattice"], f["lattice"], atol=1e-8)
        assert np.allclose(g["positions"], f["positions"], atol=tol)
        assert np.allclose(g["forces"], f["forces"], atol=tol)
        # every component in its place: no sign flip, kB -> eV/A^3
        assert np.allclose(
            g["stress"],
            f["stress_kB"] * VASP_KB_TO_EV_ANG3,
            atol=tol * VASP_KB_TO_EV_ANG3,
        )


def test_truncated_vasprun_keeps_complete_steps(tmp_path):
    frames = make_frames([("Si", 4)], n_steps=3)
    write_vasprun(tmp_path / "v", frames, truncate=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = read_vasprun(str(tmp_path / "v"))
    assert len(got) == 2 and any("truncated" in str(x.message) for x in w)


def test_real_gesn_excerpt_both_readers_agree():
    """Two ionic steps of a VASP 6 GeSn relaxation (63 atoms, 60 Ge + 3 Sn)."""
    a = read_vasprun(os.path.join(REAL, "vasprun.xml"))
    b = read_outcar(os.path.join(REAL, "OUTCAR"))
    assert len(a) == len(b) == 2
    assert a[0]["atom_types"] == ["Ge"] * 60 + ["Sn"] * 3
    # values as printed in the OUTCAR of step 1
    assert a[0]["energy_wo_entropy"] == pytest.approx(-298.38941090, abs=1e-8)
    assert a[0]["energy_free"] == pytest.approx(-298.38943478, abs=1e-8)
    assert np.allclose(a[0]["forces"][0], [0.540031, -0.394073, -0.229797], atol=1e-6)
    kB = a[0]["stress"] / VASP_KB_TO_EV_ANG3
    # in kB  56.45944 53.98175 40.77224 -4.99684(xy) 79.88234(yz) -28.12972(zx)
    assert np.allclose(
        [kB[0, 0], kB[1, 1], kB[2, 2], kB[0, 1], kB[1, 2], kB[2, 0]],
        [56.45944, 53.98175, 40.77224, -4.99684, 79.88234, -28.12972],
        atol=1e-5,
    )
    for x, y in zip(a, b):
        for e in ENERGIES:
            assert x[e] == pytest.approx(y[e], abs=1e-8)
        assert np.abs(x["lattice"] - y["lattice"]).max() < 1e-8
        assert np.abs(x["positions"] - y["positions"]).max() < 1e-5
        assert np.abs(x["forces"] - y["forces"]).max() < 1e-6
        assert np.abs(x["stress"] - y["stress"]).max() / VASP_KB_TO_EV_ANG3 < 1e-5


# ---------------------------------------------------------------- dataset


def _runs(root, n_runs=5, interleave=False):
    """Runs with Sn content rising with the run index."""
    all_frames = {}
    for r in range(n_runs):
        fr = make_frames(
            [("Ge", 7 - r % 4), ("Sn", 1 + r % 4)],
            n_steps=2 + r % 2,
            seed=r,
            interleave=interleave,
        )
        write_run(root / f"{r + 1:05d}", fr)
        all_frames[f"{r + 1:05d}"] = fr
    return all_frames


def _run_dataset(tmp_path, **sec):
    cfg = {
        "dataset": {
            "sources": [str(tmp_path / "runs")],
            "output_dir": str(tmp_path / "out"),
            **sec,
        }
    }
    p = tmp_path / "dataset.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return cmd_dataset(_ns(p))


def _manifest(tmp_path):
    with open(tmp_path / "out" / "manifest.csv") as fh:
        return list(csv.DictReader(fh))


def test_dataset_rows_follow_poscar_atoms_and_energy_shift(tmp_path):
    frames = _runs(tmp_path / "runs", n_runs=3, interleave=True)
    e_atom = {"Ge": -0.5, "Sn": -0.7}
    assert (
        _run_dataset(
            tmp_path, isolated_atom_energy=e_atom, split={"test_fraction": 0.0}
        )
        == 0
    )
    out = tmp_path / "out" / "train"
    t, lab, _, X = load_mlip_csv(str(out / "structures.csv"), verbose=False)
    structs = {
        s["name"]: s for s in _structures_from_csv(t, lab, X, str(out), verbose=False)
    }
    rows = _manifest(tmp_path)
    assert len(rows) == sum(len(f) for f in frames.values()) == len(structs)
    for row in rows:
        f = frames[os.path.basename(row["run"])][int(row["step"]) - 1]
        s = structs[row["struct_idx"]]
        # POSCAR groups atoms by species; each F row must move with its atom
        for i, (typ, pos) in enumerate(zip(s["atom_types"], s["positions"])):
            j = int(np.argmin(np.linalg.norm(f["positions"] - pos, axis=1)))
            assert f["atom_types"][j] == typ
            assert np.allclose(s["ref_forces"][i], f["forces"][j], atol=1e-8)
        shift = sum(e_atom[x] for x in f["atom_types"])
        assert s["ref_energy"] == pytest.approx(
            (f["energy_wo_entropy"] - shift) / len(f["atom_types"])
        )
        assert np.allclose(
            s["ref_stress"], f["stress_kB"] * VASP_KB_TO_EV_ANG3, atol=1e-12
        )


def test_dataset_splits_whole_runs(tmp_path):
    _runs(tmp_path / "runs", n_runs=5)
    assert _run_dataset(tmp_path, split={"test_fraction": 0.4, "seed": 3}) == 0
    rows = _manifest(tmp_path)
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_run.values())
    assert sum(v == {"test"} for v in by_run.values()) == 2
    assert (tmp_path / "out" / "test" / "structures.csv").is_file()
    assert parse_poscar(str(tmp_path / "out" / "test" / "0" / "POSCAR"))["natoms"] == 8


def test_stratified_split_takes_one_run_per_composition_group():
    comps = [(1 - x, x) for x in np.linspace(0.0, 0.3, 50)]
    test = _split_runs(comps, 0.2, "composition", seed=0)
    idx = np.flatnonzero(test)
    assert len(idx) == 10
    assert sorted(i // 5 for i in idx) == list(range(10))


def test_dataset_refuses_disagreeing_files(tmp_path):
    frames = make_frames([("Si", 4)], n_steps=2)
    write_run(tmp_path / "runs" / "a", frames)
    bad = [dict(f) for f in frames]
    bad[1] = dict(bad[1], forces=bad[1]["forces"] + 1e-3)
    write_outcar(tmp_path / "runs" / "a" / "OUTCAR", bad)
    assert _run_dataset(tmp_path) == 1
    assert _run_dataset(tmp_path, cross_check=False) == 0


def test_dataset_requires_every_isolated_atom_energy(tmp_path):
    _runs(tmp_path / "runs", n_runs=2)
    assert _run_dataset(tmp_path, isolated_atom_energy={"Ge": -0.5}) == 1


# ------------------------------------------------- isolated atoms (ADR-0026)


def _atom_run(d, sp, energy=-0.75):
    fr = make_frames([(sp, 1)], n_steps=2, seed=7)
    fr[-1] = dict(fr[-1], energy_free=energy - 0.02, energy_wo_entropy=energy,
                  energy_sigma0=energy - 0.01)
    write_run(d, fr)


def test_isolated_atom_path_gives_the_last_step_energy_with_the_dataset_key(tmp_path):
    _runs(tmp_path / "runs", n_runs=3, interleave=True)
    _atom_run(tmp_path / "atoms" / "Ge", "Ge", energy=-0.75)
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # no warning about the reference
        res = build_training_set(
            str(tmp_path / "runs"), output_dir=str(tmp_path / "out"),
            isolated_atom_energy={"Ge": str(tmp_path / "atoms" / "Ge"), "Sn": -0.6})
    assert res["isolated_atom_energy"] == {"Ge": -0.75, "Sn": -0.6}
    res = build_training_set(
        str(tmp_path / "runs"), output_dir=str(tmp_path / "out2"), energy="sigma0",
        isolated_atom_energy={"Ge": str(tmp_path / "atoms" / "Ge" / "OUTCAR"), "Sn": -0.6})
    assert res["isolated_atom_energy"]["Ge"] == pytest.approx(-0.76)


def test_dataset_cli_subtracts_an_isolated_atom_given_by_path(tmp_path, capsys):
    frames = _runs(tmp_path / "runs", n_runs=2)
    _atom_run(tmp_path / "atoms" / "Ge", "Ge", energy=-0.75)
    assert _run_dataset(tmp_path, split={"test_fraction": 0.0}, isolated_atom_energy={
        "Ge": str(tmp_path / "atoms" / "Ge"), "Sn": -0.6}) == 0
    assert "Isolated atoms : Ge -0.75000000 eV, Sn -0.60000000 eV" in capsys.readouterr().out
    row = _manifest(tmp_path)[0]
    f = frames[os.path.basename(row["run"])][int(row["step"]) - 1]
    shift = sum(-0.75 if t == "Ge" else -0.6 for t in f["atom_types"])
    assert float(row["target_eV_per_atom"]) == pytest.approx(
        (f["energy_wo_entropy"] - shift) / len(f["atom_types"]))


def test_isolated_atom_path_must_hold_one_atom_of_that_species(tmp_path):
    _runs(tmp_path / "runs", n_runs=2)
    _atom_run(tmp_path / "atoms" / "Sn", "Sn")
    with pytest.raises(ValueError, match="not one Ge atom"):
        build_training_set(str(tmp_path / "runs"), output_dir=str(tmp_path / "out"),
                           isolated_atom_energy={"Ge": str(tmp_path / "atoms" / "Sn"),
                                                 "Sn": -0.6})
    with pytest.raises(FileNotFoundError, match="no OUTCAR"):
        build_training_set(str(tmp_path / "runs"), output_dir=str(tmp_path / "out"),
                           isolated_atom_energy={"Ge": str(tmp_path / "nowhere"), "Sn": -0.6})
