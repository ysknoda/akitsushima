"""ADR-0023: the Python API reproduces the CLI.

Seconds-long counterparts of example/api (the GeSn chain and the other
commands, where 78 of 79 comparisons agreed, 74 bit for bit).  Each test
runs a CLI command in-process and the same calculation written against
the public API with every CLI setting passed explicitly (decision 8),
then requires identical numbers: both paths call the same code with the
same inputs, so anything but bitwise agreement is a CLI-API divergence.
"""

import json

import numpy as np
import pytest
import yaml
from test_cli_smoke import _ns
from test_mlip_chain_rule import _sw_si_structures, _write_layout_b

import akitsushima as ak
from akitsushima.cli.parser import _DISPATCH
from akitsushima.fitting import ELMModel, load_mlip_csv
from akitsushima.fitting.descriptors import ACSF, build_table
from akitsushima.fitting.mlip_engine import load_mlip_model
from akitsushima.fitting.models.atomic import AtomicELMModel

# Stillinger & Weber, PRB 31, 5262 (1985), Si.
SW_SI = {"epsilon": 2.1683, "sigma": 2.0951, "a": 1.80, "gamma": 1.20,
         "lambda": 21.0, "A": 7.049556277, "B": 0.6022245584, "p": 4.0,
         "q": 0.0, "costheta0": -1.0 / 3.0}
DESC = {"type": "acsf", "r_cut": 3.771,
        "g2": {"eta": [4.0], "rs": [2.2, 2.6, 3.0]},
        "g4": {"eta": [0.05], "zeta": [1, 2], "lambda": [-1, 1]}}


def _si64(seed):
    """Rattled 2x2x2 diamond Si (64 atoms, width 10.9 A > 2 (r_cut + skin))."""
    s = _sw_si_structures(1, seed)[0]
    shifts = np.array([[i, j, k] for i in range(2) for j in range(2)
                       for k in range(2)]) @ s["lattice"]
    pos = (s["positions"][None] + shifts[:, None]).reshape(-1, 3)
    return 2 * s["lattice"], pos, ["Si"] * len(pos)


def _layout_b(root):
    """The fit-layout loader a user writes (gap G2)."""
    y, lab, _, X = load_mlip_csv(str(root / "structures.csv"), verbose=False)
    sid = X[:, 0].astype(int)
    out = []
    for i in sorted(set(sid)):
        p = ak.parse_poscar(str(root / str(i) / "POSCAR"))
        m = sid == i
        xx, yy, zz, yz, zx, xy = y[m & (lab == "S")]
        out.append({"lattice": p["lattice"], "positions": p["positions"],
                    "atom_types": p["atom_types"], "natoms": p["natoms"],
                    "ref_energy": float(y[m & (lab == "E")][-1]),
                    "ref_forces": y[m & (lab == "F")].reshape(-1, 3),
                    "ref_stress": np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]])})
    return out


def _sw_fn():
    eng = ak.PotentialEngine().add(ak.StillingerWeberPotential(params=SW_SI, species=["Si"]))
    return ak.make_force_fn(eng)


def test_relax_api_matches_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lat, pos, types = _si64(3)
    ak.write_poscar("POSCAR", lat, pos, types)
    rel = {"output_dir": "cli_out", "method": "fire", "relax_cell": True,
           "fmax": 0.05, "pmax": 0.5, "max_steps": 40}
    (tmp_path / "relax.yaml").write_text(yaml.safe_dump(
        {"structure": {"poscar": "POSCAR"},
         "potential": {"type": "SW", "species": ["Si"], "params": SW_SI},
         "relax": rel}))
    assert _DISPATCH["relax"](_ns(tmp_path / "relax.yaml")) == 0
    p = ak.parse_poscar("POSCAR")
    ak.GeometryOptimiser(p["lattice"], p["positions"], p["atom_types"],
                         force_fn=_sw_fn(), method="fire", relax_cell=True,
                         fmax=0.05, pmax=0.5, max_steps=40, output_dir="api_out",
                         verbose=False, timing=True).run()
    assert (tmp_path / "api_out" / "POSCAR_relaxed").read_text() == \
        (tmp_path / "cli_out" / "POSCAR_relaxed").read_text()


def test_nvt_md_api_matches_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lat, pos, types = _si64(4)
    ak.write_poscar("POSCAR", lat, pos, types)
    md = {"output_dir": "cli_out", "ensemble": "nvt", "temperature": 600.0,
          "timestep": 1.0, "n_steps": 30, "equil_steps": 10, "tau_nhc": 50.0,
          "thermo_every": 1, "traj_every": 10, "seed": 7}
    (tmp_path / "md.yaml").write_text(yaml.safe_dump(
        {"structure": {"poscar": "POSCAR"},
         "potential": {"type": "SW", "species": ["Si"], "params": SW_SI}, "md": md}))
    assert _DISPATCH["md"](_ns(tmp_path / "md.yaml")) == 0
    p = ak.parse_poscar("POSCAR")
    fn = _sw_fn()
    ak.NVTSimulation(lattice=p["lattice"], positions=p["positions"],
                     atom_types=p["atom_types"], force_fn=fn, r_cut=fn.r_cut,
                     masses=np.array([ak.ATOMIC_MASS["Si"]] * len(types)),
                     temperature=600.0, timestep=1.0, n_steps=30, equil_steps=10,
                     tau_nhc=50.0, thermo_every=1, traj_every=10, seed=7,
                     output_dir="api_out", verbose=False, timing=True).run()
    assert (tmp_path / "api_out" / "thermo.log").read_text() == \
        (tmp_path / "cli_out" / "thermo.log").read_text()


@pytest.mark.parametrize("form", ["tabular", "atom_centred"])
def test_mlip_elm_api_matches_cli(form, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    structs = _sw_si_structures(16, 1)
    _write_layout_b(tmp_path / "train", structs)
    ml = {"output_dir": "out", "model": "elm", "form": form, "descriptor": DESC,
          "n_hidden": 24, "seed": 5}
    (tmp_path / "mlip.yaml").write_text(yaml.safe_dump(
        {"structure": {"training_csv": "train/structures.csv"},
         "potential": {"species": ["Si"]}, "mlip": ml}))
    assert _DISPATCH["mlip"](_ns(tmp_path / "mlip.yaml")) == 0
    cli_model = load_mlip_model(json.loads(
        (tmp_path / "out" / "mlip_summary.json").read_text())["model"])

    data = _layout_b(tmp_path / "train")
    desc = ACSF.from_grid(["Si"], DESC["r_cut"], DESC["g2"], DESC["g4"])
    common = dict(w_energy=1.0, w_forces=0.1, w_stress=10.0, r0=1.0, eps0=0.01,
                  seed=5, direct_link=True, verbose=False, n_hidden=24,
                  activation="tanh", reg=1e-6, reg_search=None)
    if form == "atom_centred":
        api = AtomicELMModel(**common).fit(data, desc, val_fraction=0.2, val_seed=0)
    else:
        api = ELMModel(row_independent=False, solver="auto", **common)
        y, lab, w, X = build_table(data, desc, r0=1.0, eps0=0.01)
        api.descriptor_config = desc.to_json()      # gap G3: needed by hand
        api.fit(y, lab, X, weights=w, val_fraction=0.2, val_seed=0)
        api.descriptor_config = desc.to_json()
    lat, pos, types = _si64(9)
    for a, b in zip(ak.make_force_fn(api)(lat, pos, types),
                    ak.make_force_fn(cli_model)(lat, pos, types)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("target, param, cli_default", [
    ("akitsushima.ELMModel", "eps0", 0.01),
    ("akitsushima.QuantizedELMModel", "eps0", 0.01),
    ("akitsushima.fitting.descriptors.build_table", "eps0", 0.01),
    ("akitsushima.fit_potential", "de_maxiter", 300),
    ("akitsushima.fit_potential", "de_popsize", 10),
    ("akitsushima.GeometryOptimiser", "timing", True),
    ("akitsushima.GeometryOptimiser", "output_dir", "./relax_output"),
])
def test_api_defaults_are_the_cli_defaults(target, param, cli_default):
    """ADR-0024 Q22: an API call without the keyword computes what the CLI
    computes without the YAML key (the values cmd_*.py fall back to)."""
    import importlib
    import inspect
    mod, name = target.rsplit(".", 1)
    obj = getattr(importlib.import_module(mod), name)
    assert inspect.signature(obj).parameters[param].default == cli_default


def test_relax_without_output_dir_keeps_the_working_directory_clean(tmp_path, monkeypatch):
    """ADR-0023 O3: output_dir defaulted to '.', so an API relax wrote
    relax.log, XDATCAR_relax and POSCAR_relaxed into the working directory."""
    monkeypatch.chdir(tmp_path)
    lat, pos, types = _si64(5)
    ak.GeometryOptimiser(lat, pos, types, force_fn=_sw_fn(), max_steps=2, verbose=False).run()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["relax_output"]


def test_nvt_tau_nhc_default_is_the_same_in_cli_and_api():
    """ADR-0024 Q22: the CLI md fell back to 20 fs, below the 50-200 fs
    NVTSimulation documents as typical; both are now its 100 fs."""
    import inspect
    from akitsushima.cli.templates import _DUMP_MAP

    assert inspect.signature(ak.NVTSimulation).parameters["tau_nhc"].default == 100.0
    src = inspect.getsource(__import__("akitsushima.cli.cmd_md", fromlist=["x"]))
    assert 'md_cfg.get("tau_nhc",    20.0)' not in src
    for key in ("nvt", "npt"):
        assert yaml.safe_load(_DUMP_MAP[key])["md"]["tau_nhc"] == 100.0


# ADR-0024 Q23: the MLIP API without hand-steps.  The baseline reaches
# into the Si data (nearest neighbours ~2.35 A < r_outer 2.6 A), so
# subtracting it twice or not at all would change every number.
ZBL = {"type": "zbl", "r_inner": 1.5, "r_outer": 2.6}


def _cli_mlip(tmp_path, form, extra):
    structs = _sw_si_structures(16, 1)
    _write_layout_b(tmp_path / "train", structs)
    ml = {"output_dir": "out", "model": "elm", "form": form, "descriptor": DESC,
          "n_hidden": 24, "seed": 5, **extra}
    (tmp_path / "mlip.yaml").write_text(yaml.safe_dump(
        {"structure": {"training_csv": "train/structures.csv"},
         "potential": {"species": ["Si"]}, "mlip": ml}))
    assert _DISPATCH["mlip"](_ns(tmp_path / "mlip.yaml")) == 0
    return load_mlip_model(json.loads(
        (tmp_path / "out" / "mlip_summary.json").read_text())["model"])


@pytest.mark.parametrize("form", ["tabular", "atom_centred"])
def test_mlip_with_baseline_api_matches_cli(form, tmp_path, monkeypatch):
    from akitsushima.fitting.physics import PairBaseline
    from akitsushima.io import load_training_set

    monkeypatch.chdir(tmp_path)
    cli_model = _cli_mlip(tmp_path, form, {"baseline": ZBL})
    data = load_training_set("train/structures.csv")
    desc = ACSF.from_grid(["Si"], DESC["r_cut"], DESC["g2"], DESC["g4"])
    kw = dict(n_hidden=24, seed=5, verbose=False, baseline=PairBaseline(["Si"], [ZBL]))
    if form == "atom_centred":
        api = AtomicELMModel(**kw).fit(data, desc)          # raw structures
    else:
        api = ELMModel(**kw).fit_structures(data, desc)
    lat, pos, types = _si64(9)
    for a, b in zip(ak.make_force_fn(api)(lat, pos, types),
                    ak.make_force_fn(cli_model)(lat, pos, types)):
        np.testing.assert_array_equal(a, b)


def test_fit_with_descriptor_makes_a_force_field(tmp_path, monkeypatch):
    """G3: the tabular model needed descriptor_config set by hand."""
    monkeypatch.chdir(tmp_path)
    structs = _sw_si_structures(8, 2)
    desc = ACSF.from_grid(["Si"], DESC["r_cut"], DESC["g2"], DESC["g4"])
    y, lab, w, X = build_table(structs, desc)
    m = ELMModel(n_hidden=16, verbose=False).fit(y, lab, X, weights=w, descriptor=desc)
    assert m.descriptor_config == desc.to_json()
    e, f, s = ak.make_force_fn(m)(*_si64(1))
    assert np.isfinite(e)


def test_physics_arguments_write_the_cli_physics_config():
    from akitsushima.fitting.physics import PairBaseline
    base = PairBaseline(["Si"], [ZBL])
    for cls in (ELMModel, AtomicELMModel, ak.LinearModel):
        m = cls(baseline=base, isolated_atom_energy={"Si": -0.5}, verbose=False)
        assert json.loads(m.physics_config) == {
            "baseline": base.config(), "isolated_atom_energy": {"Si": -0.5}}
        assert m.baseline.config() == base.config()


@pytest.mark.parametrize("form", ["tabular", "atom_centred"])
def test_baseline_is_subtracted_exactly_once(form):
    """Independent of the CLI: trained on raw labels with baseline=, the
    model must equal a baseline-free model trained on labels minus the
    baseline, plus the baseline itself (once, not twice, not never)."""
    from akitsushima.fitting.physics import PairBaseline

    structs = _sw_si_structures(16, 1)
    desc = ACSF.from_grid(["Si"], DESC["r_cut"], DESC["g2"], DESC["g4"])
    base = PairBaseline(["Si"], [ZBL])
    kw = dict(n_hidden=24, seed=5, verbose=False)
    if form == "atom_centred":
        with_b = AtomicELMModel(baseline=base, **kw).fit(structs, desc)
        plain = AtomicELMModel(**kw).fit(base.subtract(structs), desc)
    else:
        with_b = ELMModel(baseline=base, **kw).fit_structures(structs, desc)
        plain = ELMModel(**kw).fit_structures(base.subtract(structs), desc)
    lat, pos, types = _si64(9)
    Eb, Fb, Sb = base.compute(lat, pos, types)
    a = ak.make_force_fn(with_b)(lat, pos, types)
    b = ak.make_force_fn(plain)(lat, pos, types)
    assert abs(Eb) > 1e-3                      # the baseline matters here
    np.testing.assert_allclose(a[0], b[0] + Eb, rtol=0, atol=1e-9)
    np.testing.assert_allclose(a[1], b[1] + Fb, rtol=0, atol=1e-9)


def test_build_training_set_api_matches_cli_dataset(tmp_path):
    """ADR-0024 Q24 (G1): the dataset command's work is a library call."""
    import filecmp

    from test_vasp_dataset import _run_dataset, _runs

    from akitsushima.io import build_training_set, load_training_set

    _runs(tmp_path / "runs", n_runs=5, interleave=True)
    e_atom = {"Ge": -0.5, "Sn": -0.7}
    split = {"test_fraction": 0.4, "stratify": "composition", "seed": 3}
    assert _run_dataset(tmp_path, isolated_atom_energy=e_atom, split=split) == 0
    res = build_training_set(str(tmp_path / "runs"), output_dir=str(tmp_path / "api"),
                             isolated_atom_energy=e_atom, test_fraction=0.4, seed=3)
    assert res["n_test"] > 0
    for rel in ("train/structures.csv", "test/structures.csv", "manifest.csv",
                "train/0/POSCAR", "test/0/POSCAR"):
        assert filecmp.cmp(tmp_path / "out" / rel, tmp_path / "api" / rel, shallow=False), rel
    assert len(load_training_set(str(tmp_path / "api" / "train" / "structures.csv"))) == res["n_train"]


def test_force_fn_from_config_reproduces_cli_relax(tmp_path, monkeypatch):
    """ADR-0024 Q24 (G5): the YAML potential bridge is public."""
    monkeypatch.chdir(tmp_path)
    lat, pos, types = _si64(6)
    ak.write_poscar("POSCAR", lat, pos, types)
    pot = {"type": "SW", "species": ["Si"], "params": SW_SI}
    rel = {"output_dir": "cli_out", "max_steps": 20, "relax_cell": True}
    (tmp_path / "relax.yaml").write_text(yaml.safe_dump(
        {"structure": {"poscar": "POSCAR"}, "potential": pot, "relax": rel}))
    assert _DISPATCH["relax"](_ns(tmp_path / "relax.yaml")) == 0
    fn = ak.force_fn_from_config(pot)
    p = ak.parse_poscar("POSCAR")      # the CLI's input, at POSCAR precision
    ak.GeometryOptimiser(p["lattice"], p["positions"], p["atom_types"], force_fn=fn,
                         relax_cell=True, max_steps=20,
                         output_dir="api_out", verbose=False).run()
    assert (tmp_path / "api_out" / "POSCAR_relaxed").read_text() == \
        (tmp_path / "cli_out" / "POSCAR_relaxed").read_text()


def test_score_table_prints_energy_in_ev_per_atom(capsys):
    """ADR-0024 O8: the E rows are eV/atom; the table said eV."""
    desc = ACSF.from_grid(["Si"], DESC["r_cut"], DESC["g2"], DESC["g4"])
    ELMModel(n_hidden=8, verbose=True).fit_structures(_sw_si_structures(6, 3), desc)
    rows = [l for l in capsys.readouterr().out.splitlines() if l.strip().startswith("E ")]
    assert rows and all(l.rstrip().endswith("eV/atom") for l in rows)


def test_relax_template_fix_cell_example_is_accepted():
    """ADR-0024 O9: the template showed a 6-element 'Voigt mask', which
    GeometryOptimiser rejects (it takes a (3, 3) mask)."""
    import re as _re
    from akitsushima.cli.templates import _DUMP_MAP

    line = next(l for l in _DUMP_MAP["relax"].splitlines() if "fix_cell:" in l)
    mask = yaml.safe_load(_re.sub(r"^\s*#\s*fix_cell:\s*", "", line).split("#")[0])
    lat, pos, types = _si64(1)
    go = ak.GeometryOptimiser(lat, pos, types, force_fn=_sw_fn(), relax_cell=True,
                              fix_cell=mask, max_steps=1, verbose=False)  # constructed only
    assert go._fix_cell_mask.reshape(3, 3)[2].all()
