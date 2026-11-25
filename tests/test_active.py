"""Active learning (ADR-0037).

1. Farthest-point sampling picks spread-out points, starting from the
   given one.
2. Structures flagged train_only never enter the validation split; the
   split of the other structures is the one they get alone.
3. stop_sigma stops md on the logged step where sigma_F,max has exceeded
   the threshold on `consecutive` logged steps, writes no final
   checkpoint, and leaves a run below the threshold bit-identical; it
   needs an ensemble.  In anneal it fails the cycle at stage md.
4. `active` with an external labeller (SW through akitsushima_efs): the
   labels equal the oracle, the added structures are train-only, a rerun
   does nothing, and an interrupted round retrains to the same model.
5. `active` with vasp_inputs: the VASP directories (INCAR copy, KPOINTS
   k_i = ceil(K / L_i), make_potcar.sh, no POTCAR), exit code 2 while
   waiting, then vasprun.xml read back; a run whose SCF hit NELM is
   dropped.
"""

import importlib
import json
import math
import os
import sys

import numpy as np
import pytest
import yaml
from test_elm_options import _acsf
from test_mlip_atom_centred import _random_structs

import akitsushima as ak

A = importlib.import_module("akitsushima.active")
from akitsushima import build as B
from akitsushima.active import farthest_point
from akitsushima.fitting import AtomicELMEnsemble, AtomicELMModel
from akitsushima.fitting.mlip_engine import MLIPForceField

# Stillinger & Weber, PRB 31, 5262 (1985): Si parameters (as
# example/external/sw.yaml).
SW_SI = {
    "type": "SW",
    "species": ["Si"],
    "params": {
        "A": 7.049556277,
        "B": 0.6022245584,
        "a": 1.8,
        "costheta0": -1.0 / 3.0,
        "epsilon": 2.1683,
        "gamma": 1.2,
        "lambda": 21.0,
        "p": 4,
        "q": 0,
        "sigma": 2.0951,
    },
}
ACSF_SMALL = {
    "type": "acsf",
    "r_cut": 4.0,
    "g2": {"eta": [4.0], "rs": [2.0, 2.4, 2.8, 3.2, 3.6]},
    "g5": {"eta": [0.0], "zeta": [1, 2], "lambda": [-1, 1]},
}


# --------------------------------------------------------------- unit tests
def test_farthest_point_picks_spread_points():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert farthest_point(X, 0, 3) == [0, 3, 2]
    assert farthest_point(X, 1, 10)[:2] == [1, 3]
    assert len(farthest_point(X, 0, 10)) == 5
    # duplicates are not picked twice
    assert farthest_point(np.zeros((4, 2)), 0, 3) == [0]


@pytest.mark.slow
def test_train_only_structures_stay_out_of_the_validation_split():
    from akitsushima.fitting.models.atomic import _prepare

    desc = _acsf()
    base = _random_structs(10, 0)
    extra = [dict(s, train_only=True) for s in _random_structs(4, 50)]
    m = AtomicELMModel(n_hidden=8, verbose=False)
    a = _prepare(m, base, desc, None, 0.3, 3)
    b = _prepare(m, base + extra, desc, None, 0.3, 3)
    np.testing.assert_array_equal(a[-1], b[-1])  # same validation structures
    assert set(b[-2]) == set(a[-2]) | {10, 11, 12, 13}
    # unflagged additions do change the split: the flag is what holds it
    c = _prepare(
        m, base + [dict(s) for s in _random_structs(4, 50)], desc, None, 0.3, 3
    )
    assert not np.array_equal(a[-1], c[-1]) or set(c[-2]) != set(b[-2])


@pytest.fixture(scope="module")
def ens_ff():
    ens = AtomicELMEnsemble(
        n_members=3, n_hidden=24, r0=0.1, eps0=0.01, reg=1e-6, verbose=False
    )
    return MLIPForceField(ens.fit(_random_structs(10, 0), _acsf()))


def _nvt(ff, out, **kw):
    from akitsushima.solvers.molecular_dynamics import NVTSimulation

    s = _random_structs(3, 5)[0]
    return NVTSimulation(
        s["lattice"],
        s["positions"],
        s["atom_types"],
        force_fn=ff,
        timestep=0.5,
        n_steps=20,
        temperature=300.0,
        output_dir=str(out),
        thermo_every=2,
        traj_every=2,
        verbose=False,
        save_plots=False,
        **kw,
    )


@pytest.mark.slow
def test_md_stop_sigma(ens_ff, tmp_path):
    ref = _nvt(ens_ff, tmp_path / "ref").run(
        checkpoint=str(tmp_path / "ref.npz"), checkpoint_every=1000
    )
    assert os.path.isfile(tmp_path / "ref.npz")

    sim = _nvt(ens_ff, tmp_path / "hi").set_stop_sigma(1e6, 3)
    got = sim.run()
    assert sim.stopped is None
    np.testing.assert_array_equal(ref["positions"], got["positions"])

    sim = _nvt(ens_ff, tmp_path / "lo").set_stop_sigma(1e-12, 3)
    sim.run(checkpoint=str(tmp_path / "lo.npz"), checkpoint_every=1000)
    # logged steps 0, 2, 4: the third in a row is step 4
    assert sim.stopped["step"] == 4 and sim.stopped["sigma_F_max"] > 1e-12
    assert not os.path.isfile(tmp_path / "lo.npz")  # no final checkpoint
    rows = (tmp_path / "lo" / "thermo.log").read_text().splitlines()[1:]
    assert [int(r.split()[0]) for r in rows] == [0, 2, 4]

    single = MLIPForceField(ens_ff.model.members[0])
    with pytest.raises(ValueError, match="ensemble"):
        _nvt(single, tmp_path / "s").set_stop_sigma(0.1)
    with pytest.raises(ValueError, match="threshold"):
        _nvt(ens_ff, tmp_path / "t").set_stop_sigma(0.0)


@pytest.mark.slow
def test_anneal_stop_sigma_fails_the_cycle(ens_ff, tmp_path):
    from akitsushima.solvers.global_opt import GlobalOptimiser

    s = _random_structs(3, 5)[0]
    GlobalOptimiser(
        s["lattice"],
        s["positions"],
        s["atom_types"],
        ens_ff,
        ens_ff.r_cut,
        method="md_quench",
        output_dir=str(tmp_path),
        relax={"method": "fire", "fmax": 0.1, "max_steps": 20},
        md={"timestep": 0.5, "thermo_every": 2, "traj_every": 2},
        schedule={"type": "constant", "n_segments": 1, "T_start": 300.0},
        segment_steps=20,
        quench_every=10,
        seed=1,
        verbose=False,
        plot=False,
        max_consecutive_failures=2,
        stop_sigma={"threshold": 1e-12, "consecutive": 2},
    ).run()
    recs = [json.loads(x) for x in open(tmp_path / "anneal_log.jsonl")]
    failed = [r for r in recs if r.get("status") == "failed"]
    assert failed and all(
        r["stage"] == "md" and "stop_sigma" in r["error"] for r in failed
    )
    with pytest.raises(ValueError, match="ensemble"):
        GlobalOptimiser(
            s["lattice"],
            s["positions"],
            s["atom_types"],
            MLIPForceField(ens_ff.model.members[0]),
            3.0,
            stop_sigma={"threshold": 0.1},
        )


# ------------------------------------------------------------ end to end
@pytest.fixture(scope="module")
def sw_case(tmp_path_factory):
    """64-atom SW Si data (12 train, 6 test), a start structure, the
    oracle's YAML and an initial ensemble."""
    from akitsushima.active import _train
    from akitsushima.io.dataset import _write_split

    d = tmp_path_factory.mktemp("sw_case")
    fn = ak.force_fn_from_config(SW_SI)
    base = B.supercell(B.bulk("diamond", "Si", 5.431), matrix=[2, 2, 2])

    def items(n, seed0, sig):
        out = []
        for k in range(n):
            s = B.rattle(base, sig, seed=seed0 + k)
            E, F, S = fn(s["lattice"], s["positions"], s["atom_types"])
            out.append(
                {
                    "frame": {
                        "lattice": s["lattice"],
                        "positions": s["positions"],
                        "atom_types": s["atom_types"],
                        "forces": F,
                        "stress": S,
                    },
                    "target": E / 64,
                    "run": "init",
                    "step": k,
                }
            )
        return out

    _write_split(str(d / "data" / "train"), items(12, 0, 0.05))
    _write_split(str(d / "data" / "test"), items(6, 100, 0.12))
    os.makedirs(d / "start")
    ak.write_poscar(
        str(d / "start" / "POSCAR_a"),
        base["lattice"],
        base["positions"],
        base["atom_types"],
    )
    with open(d / "sw.yaml", "w") as fh:
        yaml.safe_dump({"potential": SW_SI}, fh)
    cfg = {
        "model": {
            "train_dir": str(d / "data" / "train"),
            "test_dir": str(d / "data" / "test"),
        },
        "train": {
            "model": "elm",
            "form": "atom_centred",
            "ensemble": 2,
            "n_hidden": 20,
            "descriptor": ACSF_SMALL,
        },
    }
    path, _ = _train(cfg, str(d / "initial"), [])
    return d, path


def _cfg(d, model, out, label, rounds=1):
    return {
        "active": {
            "output_dir": str(out),
            "rounds": rounds,
            "model": {
                "file": model,
                "train_dir": str(d / "data" / "train"),
                "test_dir": str(d / "data" / "test"),
            },
            "train": {
                "model": "elm",
                "form": "atom_centred",
                "ensemble": 2,
                "n_hidden": 20,
                "descriptor": ACSF_SMALL,
            },
            "explore": [
                {
                    "name": "md",
                    "kind": "md",
                    "structures": [str(d / "start" / "POSCAR_a")],
                    "md": {
                        "ensemble": "nvt",
                        "temperature": 1200.0,
                        "n_steps": 40,
                        "traj_every": 10,
                        "thermo_every": 10,
                    },
                }
            ],
            "select": {"k": 2},
            "label": label,
            "verbose": False,
        }
    }


@pytest.mark.slow
def test_active_external_labels_resume_and_retrain(sw_case, tmp_path):
    from akitsushima.io.training_set import load_training_set

    d, model = sw_case
    label = {
        "mode": "external",
        "n_jobs": 2,
        "potential": {
            "type": "external",
            "cmd": f"{sys.executable} -m akitsushima.tools.akitsushima_efs {d / 'sw.yaml'}",
        },
    }
    cfg = _cfg(d, model, tmp_path / "out", label)
    res = ak.active(cfg)
    assert res["status"] == "rounds_done" and len(res["rounds"]) == 1
    summ = res["rounds"][0]["summary"]
    assert summ["n_added"] == 2 and summ["n_train_structures"] == 14
    assert set(summ["rmse_test"]) == {"E", "F", "S"}

    # labels = the oracle, through the fit-layout files the model trains on
    fn = ak.force_fn_from_config(SW_SI)
    added = load_training_set(
        str(tmp_path / "out" / "round_000" / "train" / "added" / "structures.csv")
    )
    for s in added:
        E, F, S = fn(s["lattice"], s["positions"], s["atom_types"])
        assert abs(E / s["natoms"] - s["ref_energy"]) < 1e-9
        np.testing.assert_allclose(s["ref_forces"], F, atol=1e-8)
        np.testing.assert_allclose(s["ref_stress"], S, atol=1e-9)

    led = tmp_path / "out" / "active_ledger.jsonl"
    rows = led.read_text().splitlines()
    sel = [json.loads(r) for r in rows if '"stage": "select"' in r][0]
    assert sel["n_selected"] == 2 and sel["n_frames"] >= 2

    # a rerun does nothing
    again = ak.active(cfg)
    assert led.read_text().splitlines() == rows and again["model"] == res["model"]

    # interrupted after labelling: only the retraining runs, to the same model
    s0 = added[0]
    E_ref = MLIPForceField.from_file(res["model"])(
        s0["lattice"], s0["positions"], s0["atom_types"]
    )[0]
    led.write_text("\n".join(r for r in rows if '"stage": "train"' not in r) + "\n")
    os.remove(res["model"])
    redo = ak.active(cfg)
    E_new = MLIPForceField.from_file(redo["model"])(
        s0["lattice"], s0["positions"], s0["atom_types"]
    )[0]
    assert E_new == E_ref
    n_label_rows = sum(
        '"stage": "label_item"' in r for r in led.read_text().splitlines()
    )
    assert n_label_rows == 2


@pytest.mark.slow
def test_active_vasp_inputs_wait_then_read(sw_case, tmp_path):
    from vasp_synth import write_vasprun

    from argparse import Namespace

    from akitsushima.cli.cmd_active import cmd_active

    def cli_main(argv):
        return cmd_active(Namespace(yaml=argv[1], quiet=True))

    d, model = sw_case
    (tmp_path / "INCAR").write_text("ENCUT = 500\nNSW = 0\n")
    label = {
        "mode": "vasp_inputs",
        "incar": str(tmp_path / "INCAR"),
        "potcar": {"Si": "/nonexistent/potpaw/Si/POTCAR"},
        "kpoint_density": 44.0,
    }
    cfg = _cfg(d, model, tmp_path / "out", label)
    yml = tmp_path / "active.yaml"
    yml.write_text(yaml.safe_dump(cfg))
    assert cli_main(["active", str(yml)]) == 2  # waiting for VASP

    ldir = tmp_path / "out" / "round_000" / "label"
    runs = sorted(p for p in ldir.iterdir() if p.is_dir())
    assert len(runs) == 2
    for r in runs:
        assert (r / "INCAR").read_text() == "ENCUT = 500\nNSW = 0\n"
        assert not (r / "POTCAR").exists()
        assert "/nonexistent/potpaw/Si/POTCAR" in (r / "make_potcar.sh").read_text()
        st = ak.parse_poscar(str(r / "POSCAR"))
        k = [int(x) for x in (r / "KPOINTS").read_text().splitlines()[3].split()]
        assert k == [math.ceil(44.0 / np.linalg.norm(v)) for v in st["lattice"]]

    # one converged run, one that hit NELM
    for j, r in enumerate(runs):
        st = ak.parse_poscar(str(r / "POSCAR"))
        n = st["natoms"]
        fr = {
            "lattice": st["lattice"],
            "positions": st["positions"],
            "atom_types": st["atom_types"],
            "forces": np.full((n, 3), 0.01 * (j + 1)),
            "stress_kB": np.diag([1.0, 2.0, 3.0]),
            "energy_free": -4.0 * n,
            "energy_wo_entropy": -4.1 * n,
            "energy_sigma0": -4.05 * n,
        }
        write_vasprun(str(r / "vasprun.xml"), [fr])
        if j == 1:  # NELM = 1 with one SCF step: not converged
            text = (
                (r / "vasprun.xml")
                .read_text()
                .replace(
                    "<modeling>",
                    '<modeling>\n <parameters><i name="NELM"> 1</i></parameters>',
                    1,
                )
            )
            (r / "vasprun.xml").write_text(text)
    assert cli_main(["active", str(yml)]) == 0
    rows = [json.loads(x) for x in open(tmp_path / "out" / "active_ledger.jsonl")]
    items = {r["item"]: r for r in rows if r.get("stage") == "label_item"}
    assert (
        items["000"]["status"] == "ok" and abs(items["000"]["energy"] + 4.1 * 64) < 1e-9
    )
    assert items["001"]["status"] == "dropped" and "NELM" in items["001"]["error"]
    train = [r for r in rows if r.get("stage") == "train"][0]
    assert train["summary"]["n_added"] == 1


def test_dump_default_active_is_valid_yaml():
    from akitsushima.cli.templates import _YAML_ACTIVE_DEFAULT

    cfg = yaml.safe_load(_YAML_ACTIVE_DEFAULT)["active"]
    assert {"model", "train", "explore", "select", "label", "converge"} <= set(cfg)


@pytest.mark.slow
def test_select_fills_from_failed_frames_without_duplicates(sw_case, tmp_path):
    """Every frame above hi (absolute lo / hi far below the spread): the k
    selected come from the failed frames with the smallest sigma, with no
    structure twice although two explorers start from one."""
    d, model = sw_case
    (tmp_path / "INCAR").write_text("NSW = 0\n")
    label = {"mode": "vasp_inputs", "incar": str(tmp_path / "INCAR"),
             "potcar": {"Si": "/x/POTCAR"}, "kpoint_density": 30.0}
    cfg = _cfg(d, model, tmp_path / "out", label)
    a = cfg["active"]
    a["explore"].append(dict(a["explore"][0], name="md2"))
    a["select"].update(k=3, lo=1e-9, hi=2e-9)
    res = ak.active(cfg)
    assert res["status"] == "waiting_for_labels"
    rows = [json.loads(x) for x in open(tmp_path / "out" / "active_ledger.jsonl")]
    sel = next(r for r in rows if r.get("stage") == "select")
    # every frame fails except the two explorers' frame 0, the perfect
    # start crystal (zero forces for every model, sigma 0; written since
    # ADR-0048 labels frames by completed steps)
    assert sel["n_candidate"] == 0 and sel["n_accurate"] == 2
    assert sel["n_failed"] == sel["n_frames"] - 2
    assert sel["n_selected"] == 3 and sel["n_filled_from_failed"] == 3
    assert sel["sigma_selected"] == sorted(sel["sigma_selected"])
    # both explorers' frame 0 is the start structure: only one is taken
    starts = [s for s in sel["selected"] if s.endswith("#0")]
    assert len(starts) <= 1
    import csv

    rows_csv = list(csv.DictReader(open(tmp_path / "out" / "round_000" / "select"
                                        / "frames.csv")))
    assert sum(r["class"] == "failed_filled" for r in rows_csv) == 3


def test_shortest_distance_of_diamond_si():
    """Diamond Si, a = 5.431 A: nearest neighbours at a sqrt(3) / 4 =
    2.3517 A, found across the periodic boundary of the 8-atom cell."""
    a = 5.431
    frac = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                     [.25, .25, .25], [.25, .75, .75], [.75, .25, .75],
                     [.75, .75, .25]])
    s = {"lattice": np.eye(3) * a, "positions": frac * a,
         "atom_types": ["Si"] * 8}
    assert abs(A._shortest_distance(s) - a * np.sqrt(3) / 4) < 1e-9
    # two atoms 0.3 A apart across the boundary of a 10 A box
    s = {"lattice": np.eye(3) * 10.0,
         "positions": np.array([[0.1, 5.0, 5.0], [9.8, 5.0, 5.0]]),
         "atom_types": ["Si", "Si"]}
    assert abs(A._shortest_distance(s) - 0.3) < 1e-9


@pytest.mark.slow
def test_select_distance_guard_excludes_frames(sw_case, tmp_path):
    """The default guard is 0.75 times the training set's shortest
    distance; a guard above every frame's shortest distance leaves
    nothing to select (no fill either), and the loop stalls."""
    d, model = sw_case
    (tmp_path / "INCAR").write_text("NSW = 0\n")
    label = {"mode": "vasp_inputs", "incar": str(tmp_path / "INCAR"),
             "potcar": {"Si": "/x/POTCAR"}, "kpoint_density": 30.0}
    cfg = _cfg(d, model, tmp_path / "out", label)
    cfg["active"]["select"].update(k=3, lo=1e-9, hi=2e-9, min_distance=50.0)
    res = ak.active(cfg)
    assert res["status"] == "stalled"
    rows = [json.loads(x) for x in open(tmp_path / "out" / "active_ledger.jsonl")]
    thr = next(r for r in rows if r.get("stage") == "thresholds")
    assert thr["min_distance"] == 50.0 and 2.0 < thr["train_shortest_distance"] < 2.6
    sel = next(r for r in rows if r.get("stage") == "select")
    assert sel["n_unphysical"] == sel["n_frames"] > 0
    assert sel["n_selected"] == 0 and not sel["converged"]
    import csv

    rows_csv = list(csv.DictReader(open(tmp_path / "out" / "round_000" / "select"
                                        / "frames.csv")))
    assert all(r["class"] == "unphysical" for r in rows_csv)
    assert all(float(r["d_min"]) < 50.0 for r in rows_csv)
    # default: 0.75 times the training set's shortest distance
    g, ref = A._distance_guard({"select": {"min_distance_factor": 0.75}},
                               [{"lattice": np.eye(3) * 5.431,
                                 "positions": np.array([[0, 0, 0],
                                                        [1.0, 1.0, 1.0]]),
                                 "atom_types": ["Si", "Si"]}])
    assert abs(ref - np.sqrt(3)) < 1e-9 and abs(g - 0.75 * np.sqrt(3)) < 1e-9
