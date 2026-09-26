"""An MLIP drives every force-field solver of the CLI (ADR-0007).

relax and md were covered by ADR-0004/0005.  Here: neb, tsa, phonon, kappa
and gcmc run end to end through the CLI on ``potential: {type: MLIP}``
with an atom-centred model; the solvers that need per-atom energies (gcmc,
kappa, osmc) refuse a tabular model; esm refuses an MLIP outright; a
solver asked to place an element the model was not trained on refuses it.
The physics of the MLIP-driven results against the reference potential is
checked in example/mlip/adr0007 (ADR-0007), not here.
"""

import numpy as np
import pytest
import yaml
from test_cli_smoke import _ns
from test_mlip_atom_centred import _si_acsf
from test_mlip_chain_rule import _sw_si_structures

from akitsushima.cli.parser import _DISPATCH
from akitsushima.cli.templates import _DUMP_MAP
from akitsushima.fitting.descriptors import build_table
from akitsushima.fitting.mlip_engine import MLIPForceField, check_mlip_for_solver
from akitsushima.fitting.models import ELMModel
from akitsushima.fitting.models.atomic import AtomicELMModel
from akitsushima.io.poscar import write_poscar

A_SI = 5.431


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("mlip")
    structs = _sw_si_structures(16, 1)
    ac = _si_acsf()
    atomic = AtomicELMModel(n_hidden=20, r0=0.1, eps0=0.01, verbose=False)
    atomic.fit(structs, ac)
    atomic.save(str(d / "atomic.npz"))
    y, lab, w, X = build_table(structs, ac, r0=0.1, eps0=0.01)
    tab = ELMModel(n_hidden=20, verbose=False)
    tab.fit(y, lab, X, weights=w)
    tab.descriptor_config = ac.to_json()
    tab.save(str(d / "tabular.npz"))
    return {"atomic": str(d / "atomic.npz"), "tabular": str(d / "tabular.npz")}


def _diamond(n, a=A_SI):
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    frac = np.vstack([base, base + 0.25])
    cells = np.array(
        [[i, j, k] for i in range(n) for j in range(n) for k in range(n)], float
    )
    frac = (frac[None] + cells[:, None]).reshape(-1, 3) / n
    lat = np.eye(3) * a * n
    return lat, frac @ lat


def _run(tmp_path, cmd, cfg):
    path = tmp_path / f"{cmd}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return _DISPATCH[cmd](_ns(path))


def _cfg(cmd, model, structure, section):
    cfg = yaml.safe_load(_DUMP_MAP[cmd])
    cfg["structure"] = structure
    cfg["potential"] = {"type": "MLIP", "model": model}
    cfg[cmd].update(section)
    return cfg


def test_neb_and_tsa_run_on_an_mlip(models, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lat, pos = _diamond(2)
    rest = pos[1:]
    fin = rest.copy()
    fin[3] = pos[0] + 0.4 * (rest[3] - pos[0])  # hop atom toward the vacancy
    ini = rest.copy()
    ini[3] = rest[3] + 0.4 * (pos[0] - rest[3])
    write_poscar("POSCAR_initial", lat, ini, ["Si"] * 63)
    write_poscar("POSCAR_final", lat, fin, ["Si"] * 63)
    cfg = _cfg(
        "neb",
        models["atomic"],
        {"poscar_initial": "POSCAR_initial", "poscar_final": "POSCAR_final"},
        {"output_dir": "neb_out", "n_images": 3, "max_steps": 5, "verbose": False},
    )
    assert _run(tmp_path, "neb", cfg) == 0
    assert (tmp_path / "neb_out" / "POSCAR_TS").is_file()
    cfg = _cfg(
        "tsa",
        models["atomic"],
        {"poscar": "neb_out/POSCAR_TS"},
        {"output_dir": "tsa_out", "irc_max_steps": 3, "verbose": False},
    )
    assert _run(tmp_path, "tsa", cfg) == 0


def test_neb_runs_on_a_tabular_mlip(models, tmp_path, monkeypatch):
    """neb, tsa and phonon need only E and F: a tabular model is fine."""
    monkeypatch.chdir(tmp_path)
    lat, pos = _diamond(2)
    rest = pos[1:]
    fin = rest.copy()
    fin[3] = pos[0]
    write_poscar("POSCAR_initial", lat, rest, ["Si"] * 63)
    write_poscar("POSCAR_final", lat, fin, ["Si"] * 63)
    cfg = _cfg(
        "neb",
        models["tabular"],
        {"poscar_initial": "POSCAR_initial", "poscar_final": "POSCAR_final"},
        {"output_dir": "neb_out", "n_images": 3, "max_steps": 3, "verbose": False},
    )
    assert _run(tmp_path, "neb", cfg) == 0


def test_phonon_runs_on_an_mlip(models, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lat, pos = _diamond(1)
    write_poscar("POSCAR", lat, pos, ["Si"] * 8)
    cfg = _cfg(
        "phonon",
        models["atomic"],
        {"poscar": "POSCAR"},
        {
            "output_dir": "ph_out",
            "supercell": [2, 2, 2],
            "verbose": False,
            "band_structure": {
                "q_path": [["G", [0, 0, 0]], ["X", [0.5, 0, 0.5]]],
                "n_points": 5,
                "coordinates": "reduced",
            },
            "dos": {"q_mesh": 4, "n_dos": 20, "smearing": 5.0},
        },
    )
    assert _run(tmp_path, "phonon", cfg) == 0


def test_kappa_runs_on_an_atom_centred_mlip_without_the_fallback(
    models, tmp_path, monkeypatch, recwarn
):
    monkeypatch.chdir(tmp_path)
    lat, pos = _diamond(2)
    write_poscar("POSCAR", lat, pos, ["Si"] * 64)
    cfg = _cfg(
        "kappa",
        models["atomic"],
        {"poscar": "POSCAR"},
        {
            "output_dir": "k_out",
            "n_steps": 20,
            "equil_steps": 5,
            "heat_flux_every": 2,
            "tau_max": 0.004,
            "n_blocks": 2,
            "slack_debye": None,
            "verbose": False,
        },
    )
    assert _run(tmp_path, "kappa", cfg) == 0
    # the exact site-term heat flux, not the centre-of-mass approximation
    assert not [w for w in recwarn if "centre-of-mass" in str(w.message)]


def test_kappa_refuses_a_tabular_mlip(models, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lat, pos = _diamond(2)
    write_poscar("POSCAR", lat, pos, ["Si"] * 64)
    cfg = _cfg(
        "kappa",
        models["tabular"],
        {"poscar": "POSCAR"},
        {
            "output_dir": "k_out",
            "n_steps": 20,
            "equil_steps": 5,
            "slack_debye": None,
            "verbose": False,
        },
    )
    with pytest.raises(ValueError, match="per-atom energies"):
        _run(tmp_path, "kappa", cfg)


def _gcmc_cfg(model, cmd="gcmc", species="Si"):
    cfg = yaml.safe_load(_DUMP_MAP[cmd])
    cfg["structure"] = {"poscar": "POSCAR"}
    cfg["potential"] = {"type": "MLIP", "model": model}
    cfg[cmd].update(
        {
            "output_dir": f"{cmd}_out",
            "species": species,
            "n_cycles": 2,
            "md_steps_per_cycle": 2,
            "gcmc_attempts_per_cycle": 3,
            "verbose": False,
        }
    )
    cfg[cmd].pop("species_params", None)
    return cfg


def test_gcmc_runs_on_an_atom_centred_mlip(models, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lat, pos = _diamond(2)
    write_poscar("POSCAR", lat, pos[1:], ["Si"] * 63)
    assert _run(tmp_path, "gcmc", _gcmc_cfg(models["atomic"])) == 0


def test_gcmc_refuses_a_tabular_mlip_and_untrained_elements(
    models, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    lat, pos = _diamond(2)
    write_poscar("POSCAR", lat, pos[1:], ["Si"] * 63)
    with pytest.raises(ValueError, match="per-atom energies"):
        _run(tmp_path, "gcmc", _gcmc_cfg(models["tabular"]))
    with pytest.raises(ValueError, match=r"\['Ge'\] are not among"):
        _run(tmp_path, "gcmc", _gcmc_cfg(models["atomic"], species="Ge"))


def test_osmc_takes_an_atom_centred_mlip_not_a_tabular_one(
    models, tmp_path, monkeypatch
):
    """osmc is CFCMC (ADR-0009): the fractional atom couples through total
    energies, so it takes what gcmc takes."""
    monkeypatch.chdir(tmp_path)
    lat, pos = _diamond(2)
    write_poscar("POSCAR", lat, pos[1:], ["Si"] * 63)
    cfg = _gcmc_cfg(models["atomic"], cmd="osmc")
    cfg["osmc"]["equil_cycles"] = 1
    assert _run(tmp_path, "osmc", cfg) == 0
    with pytest.raises(ValueError, match="per-atom energies"):
        _run(tmp_path, "osmc", _gcmc_cfg(models["tabular"], cmd="osmc"))


def test_esm_refuses_an_mlip(models, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    lat, pos = _diamond(1)
    write_poscar("POSCAR_slab", lat, pos, ["Si"] * 8)
    cfg = yaml.safe_load(_DUMP_MAP["esm"])
    cfg["structure"] = {"poscar": "POSCAR_slab"}
    cfg["potential"] = {"type": "MLIP", "model": models["atomic"]}
    assert _run(tmp_path, "esm", cfg) == 1
    assert "electrostatics only" in capsys.readouterr().out


def test_check_mlip_for_solver_ignores_other_force_functions():
    def fn(lat, pos, types, neigh_list=None):
        return 0.0, np.zeros_like(pos), np.zeros((3, 3))

    check_mlip_for_solver(
        fn, "X", need_site_energies=True, species=["Zz"], refuse="never"
    )


def test_cli_main_turns_a_refusal_into_a_message(models, tmp_path, monkeypatch, capsys):
    import sys

    from akitsushima.cli.parser import main

    monkeypatch.chdir(tmp_path)
    lat, pos = _diamond(2)
    write_poscar("POSCAR", lat, pos[1:], ["Si"] * 63)
    (tmp_path / "osmc.yaml").write_text(
        yaml.safe_dump(_gcmc_cfg(models["tabular"], cmd="osmc"))
    )
    monkeypatch.setattr(sys, "argv", ["akitsushima", "osmc", "osmc.yaml"])
    assert main() == 1
    assert "ERROR:" in capsys.readouterr().out
    assert MLIPForceField.from_file(models["atomic"]).per_atom_energy
