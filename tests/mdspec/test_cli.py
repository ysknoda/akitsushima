"""Config validation, templates, the akitsushima YAML translation, the
nozuchi converter and an end-to-end run of ``akitsushima mdspec``
(nozuchi ADR-0007; ADR-0051).  nozuchi's own CLI (run / check / template,
exit code 2) is replaced by the akitsushima command."""

import argparse
import json

import numpy as np
import pytest
import yaml
from akitsushima.cli.cmd_mdspec import cmd_mdspec
from akitsushima.mdspec import config, yaml_io

from akitsushima.cli.templates import _DUMP_MAP

from .conftest import write_xdatcar


def _ns(path, **kw):
    base = dict(
        yaml=str(path),
        quiet=False,
        check=False,
        convert_nozuchi=False,
        energy=None,
        target=0.02,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_unknown_and_missing_keys():
    with pytest.raises(config.ConfigError, match="unknown"):
        config.validate(
            {
                "trajectory": {
                    "files": ["x"],
                    "timestep_fs": 1,
                    "frame_every": 1,
                    "bogus": 1,
                },
                "temperature": 300,
                "dipole": {"model": "point_charge", "charges": {"A": 0}},
            }
        )
    with pytest.raises(config.ConfigError, match="timestep_fs"):
        config.validate(
            {
                "trajectory": {"files": ["x"], "frame_every": 1},
                "temperature": 300,
                "dipole": {"model": "point_charge", "charges": {"A": 0}},
            }
        )


@pytest.mark.parametrize("model", sorted(config.TEMPLATE))
def test_templates_validate(model, tmp_path):
    raw = yaml.safe_load(config.TEMPLATE[model])
    cfg = config.validate(raw, tmp_path)
    assert cfg["dipole"]["model"] == model


@pytest.mark.parametrize("model", sorted(config.TEMPLATE))
def test_converter_round_trip(model, tmp_path):
    """nozuchi input -> akitsushima YAML -> internal: the same validated
    configuration as nozuchi's own (paths, defaults, Born block)."""
    raw = yaml.safe_load(config.TEMPLATE[model])
    (tmp_path / "noz.yaml").write_text(config.TEMPLATE[model])
    (tmp_path / "aki.yaml").write_text(yaml_io.convert_file(tmp_path / "noz.yaml"))
    assert config.validate(raw, tmp_path) == yaml_io.load(tmp_path / "aki.yaml")
    aki = yaml.safe_load((tmp_path / "aki.yaml").read_text())
    assert "epsilon_inf" not in aki["mdspec"]
    if model == "born_tensor":
        assert set(aki["born"]["born_charges"]) == {"outcar", "unit_cell", "supercell"}


def test_translation_refusals():
    base = {
        "mdspec": {
            "trajectory": {"files": ["x"], "timestep_fs": 1, "frame_every": 1},
            "temperature": 300,
            "dipole": {"model": "born_tensor"},
        }
    }

    def copy():
        return json.loads(json.dumps(base))

    bad = copy()
    bad["mdspec"]["epsilon_inf"] = 2.0
    with pytest.raises(config.ConfigError, match="born.epsilon_inf"):
        yaml_io.to_internal(bad)
    bad = copy()
    bad["mdspec"]["dipole"]["born"] = {"per_species": {"A": 1}}
    with pytest.raises(config.ConfigError, match="born.born_charges"):
        yaml_io.to_internal(bad)
    bad = copy()
    bad["mdspec"]["dipole"]["map_frames"] = 10
    with pytest.raises(config.ConfigError, match="map_frames"):
        yaml_io.to_internal(bad)
    with pytest.raises(config.ConfigError, match="mdspec:"):
        yaml_io.to_internal({"born": {}})
    ok = copy()
    ok["born"] = {"epsilon_inf": 3.0, "born_charges": {"per_species": {"A": 1.0}}}
    ok["mdspec"]["dipole"]["map_frames"] = 10
    raw = yaml_io.to_internal(ok)
    assert raw["epsilon_inf"] == 3.0
    assert raw["dipole"]["born"] == {"per_species": {"A": 1.0}, "map_frames": 10}
    pc = json.loads(json.dumps(ok))
    pc["mdspec"]["dipole"] = {"model": "point_charge", "charges": {"A": 0.0}}
    with pytest.warns(UserWarning, match="not used"):
        yaml_io.to_internal(pc)


def test_dump_default_template_translates(tmp_path):
    cfg = yaml.safe_load(_DUMP_MAP["mdspec"])
    out = config.validate(yaml_io.to_internal(cfg), tmp_path)
    assert out["dipole"]["model"] == "point_charge"
    assert out["epsilon_inf"] == 1.0


def _write_run(tmp_path, crystal_traj):
    pos, lattice, symbols, _masses, _q, d_eps = crystal_traj
    frac = (pos[::2] @ np.linalg.inv(lattice)) % 1.0
    n_mg = symbols.count("Mg")
    write_xdatcar(
        tmp_path / "XDATCAR",
        ["Mg", "O"],
        [n_mg, len(symbols) - n_mg],
        frac,
        lattice,
        comment="akitsushima test",
    )
    (tmp_path / "in.yaml").write_text(
        "born: {epsilon_inf: 1.0}\n"
        "mdspec:\n"
        "  trajectory: {files: [XDATCAR], timestep_fs: 1.0, frame_every: 2}\n"
        "  temperature: 300.0\n"
        "  dipole: {model: point_charge, charges: {Mg: 2.0, O: -2.0}}\n"
        "  spectrum: {max_lag_ps: 2.0, freq_max: 30}\n"
        "  fit: {model: additive, terms: [{type: lorentz, f0: 9.0}, "
        "{type: lorentz, f0: 11.0}], freq_range: [2, 20]}\n"
        "  output: {dir: out, prefix: t}\n"
    )
    return d_eps


def test_end_to_end_cli(tmp_path, crystal_traj, capsys):
    d_eps = _write_run(tmp_path, crystal_traj)
    assert cmd_mdspec(_ns(tmp_path / "in.yaml")) == 0
    res = json.loads((tmp_path / "out" / "t_results.json").read_text())
    assert abs((res["eps_static_iso"] - 1) / d_eps - 1) < 0.03
    assert res["nozuchi_base"] == "0.2.0a2"
    for suffix in (
        "_eps.dat",
        "_acf.dat",
        "_dipole.dat",
        "_fit.dat",
        ".npz",
        "_dielectric.png",
        ".log",
    ):
        assert (tmp_path / "out" / f"t{suffix}").exists()
    eps = np.loadtxt(tmp_path / "out" / "t_eps.dat")
    assert eps[:, 0].max() <= 30.0 + 1e-9
    # second run uses the dipole cache and gives identical numbers
    assert cmd_mdspec(_ns(tmp_path / "in.yaml")) == 0
    assert "cache hit" in capsys.readouterr().out
    res2 = json.loads((tmp_path / "out" / "t_results.json").read_text())
    assert res2["eps_static_iso"] == res["eps_static_iso"]


def test_cli_errors(tmp_path, monkeypatch, capsys):
    """Input errors are one ERROR line from akitsushima's main (exit 1);
    run-time errors keep their traceback and are written to the log."""
    from akitsushima.cli import parser

    (tmp_path / "in.yaml").write_text(
        "born: {epsilon_inf: 1.0}\n"
        "mdspec:\n"
        "  trajectory: {files: [missing_XDATCAR], timestep_fs: 1.0, frame_every: 1}\n"
        "  temperature: 300.0\n"
        "  dipole: {model: point_charge, charges: {Mg: 2.0, O: -2.0}}\n"
        "  output: {dir: out, prefix: t}\n"
    )
    with pytest.raises(FileNotFoundError):
        cmd_mdspec(_ns(tmp_path / "in.yaml"))
    assert "[error] FileNotFoundError" in (tmp_path / "out" / "t.log").read_text()
    (tmp_path / "bad.yaml").write_text("mdspec: {trajectory: {files: [x]}}\n")
    monkeypatch.setattr(
        "sys.argv", ["akitsushima", "mdspec", str(tmp_path / "bad.yaml")]
    )
    assert parser.main() == 1
    assert "ERROR:" in capsys.readouterr().out
