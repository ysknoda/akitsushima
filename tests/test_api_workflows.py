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
