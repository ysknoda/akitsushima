"""md and relax restart through the CLI (ADR-0031 decisions 3, 6, 9).

An existing checkpoint is continued unless ``restart: false``; raising
``n_steps`` extends a finished md run to the result of one longer run;
a changed potential (its config section or a file it reads) is refused.
"""

import os

import numpy as np
import pytest
from test_cli_smoke import _LJ_POT_FITTED, _al_supercell_poscar, _ns, _write_yaml

from akitsushima.cli.cmd_md import cmd_md
from akitsushima.cli.cmd_relax import cmd_relax
from akitsushima.io.poscar import parse_poscar, write_poscar


def _rattled_poscar(tmp_path):
    """The 2x2x2 Al supercell with 0.05 A noise: a relax with work to do."""
    path = _al_supercell_poscar(tmp_path, name="POSCAR_rattled")
    d = parse_poscar(str(path))
    rng = np.random.default_rng(5)
    pos = d["positions"] + rng.normal(0, 0.05, d["positions"].shape)
    write_poscar(str(path), d["lattice"], pos, d["atom_types"])
    return path


def _md_cfg(tmp_path, out, n_steps, pot=None, **md):
    sec = {
        "ensemble": "nvt",
        "output_dir": str(out),
        "temperature": 600.0,
        "timestep": 1.0,
        "n_steps": n_steps,
        "equil_steps": 5,
        "tau_nhc": 50.0,
        "traj_every": 5,
        "thermo_every": 5,
        "checkpoint_every": 10,
        "verbose": False,
        "seed": 3,
    }
    sec.update(md)
    cfg = {
        "structure": {"poscar": str(_al_supercell_poscar(tmp_path))},
        "potential": pot or _LJ_POT_FITTED,
        "md": sec,
    }
    path = tmp_path / f"md_{os.path.basename(out)}.yaml"
    _write_yaml(path, cfg)
    return path


def _final(out):
    return parse_poscar(str(out / "POSCAR_final_MD"))["positions"]


@pytest.mark.slow
def test_md_cli_extends_a_finished_run(tmp_path):
    one = tmp_path / "one"
    assert cmd_md(_ns(_md_cfg(tmp_path, one, 40))) == 0
    two = tmp_path / "two"
    assert cmd_md(_ns(_md_cfg(tmp_path, two, 20))) == 0
    assert os.path.isfile(two / "md_checkpoint.npz")
    assert cmd_md(_ns(_md_cfg(tmp_path, two, 40))) == 0  # continued
    assert np.array_equal(_final(one), _final(two))
    assert (one / "thermo.log").read_text().split("\n") != []
    t1 = [ln.rsplit(None, 1)[0] for ln in (one / "thermo.log").read_text().splitlines()]
    t2 = [ln.rsplit(None, 1)[0] for ln in (two / "thermo.log").read_text().splitlines()]
    assert t1 == t2


@pytest.mark.slow
def test_md_cli_restart_false_starts_afresh(tmp_path):
    out = tmp_path / "run"
    assert cmd_md(_ns(_md_cfg(tmp_path, out, 20))) == 0
    first = _final(out)
    # same n_steps, continued: nothing new; restart false: the same run again
    assert cmd_md(_ns(_md_cfg(tmp_path, out, 20, restart=False))) == 0
    assert np.array_equal(_final(out), first)
    rows = [ln for ln in (out / "thermo.log").read_text().splitlines()]
    assert len(rows) == 1 + 25 // 5  # header + steps 0, 5, ..., 20


def test_md_cli_refuses_a_changed_potential(tmp_path):
    out = tmp_path / "run"
    assert cmd_md(_ns(_md_cfg(tmp_path, out, 10))) == 0
    pot = dict(_LJ_POT_FITTED, params={"epsilon": 0.41, "sigma": 2.5})
    with pytest.raises(ValueError, match="potential_hash"):
        cmd_md(_ns(_md_cfg(tmp_path, out, 20, pot=pot)))


def test_relax_cli_writes_and_continues_a_checkpoint(tmp_path):
    out = tmp_path / "relax"
    cfg = {
        "structure": {"poscar": str(_rattled_poscar(tmp_path))},
        "potential": _LJ_POT_FITTED,
        "relax": {
            "output_dir": str(out),
            "method": "fire",
            "fmax": 1e-8,
            "max_steps": 12,
            "checkpoint_every": 5,
            "verbose": False,
        },
    }
    path = tmp_path / "relax.yaml"
    _write_yaml(path, cfg)
    assert cmd_relax(_ns(path)) == 0
    assert os.path.isfile(out / "relax_checkpoint.npz")
    cfg["relax"]["max_steps"] = 20
    _write_yaml(path, cfg)
    assert cmd_relax(_ns(path)) == 0
    lines = (out / "relax.log").read_text().split("\n\n")[0].splitlines()[1:]
    assert [int(ln.split()[0]) for ln in lines] == list(range(20))
