"""End-to-end smoke tests for every CLI sub-command: each is invoked
in-process (argparse.Namespace built directly, matching how parser.py
dispatches) against a small, fast, synthetic structure/config, asserting
a clean (0) return code and, where meaningful, that the expected output
files were written.

These are deliberately NOT physics-accuracy tests (that is
test_engine_forces.py/test_solvers.py/test_fitting.py's job) -- the goal
here is purely "does the CLI wiring (YAML parsing, force_fn construction,
solver kwargs, result-dict access) work end-to-end without crashing",
which is exactly the class of bug (S5/S7/S19-S25/F1-F3/F5/F6/F12) found
and fixed while porting this layer.
"""
import argparse
import os

import numpy as np
import pytest

from akitsushima.cli.cmd_esm import cmd_esm
from akitsushima.cli.cmd_fit import cmd_fit
from akitsushima.cli.cmd_gcmc import cmd_gcmc
from akitsushima.cli.cmd_kappa import cmd_kappa
from akitsushima.cli.cmd_md import cmd_md
from akitsushima.cli.cmd_mlip import cmd_mlip
from akitsushima.cli.cmd_neb import cmd_neb
from akitsushima.cli.cmd_osmc import cmd_osmc
from akitsushima.cli.cmd_phonon import cmd_phonon
from akitsushima.cli.cmd_relax import cmd_relax
from akitsushima.cli.cmd_tsa import cmd_tsa
from akitsushima.cli.parser import _DISPATCH, build_parser
from akitsushima.io.poscar import parse_poscar, write_poscar

from conftest import fcc_al_dict, randomly_perturbed


def _write_yaml(path, cfg: dict) -> None:
    import yaml
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh)


def _ns(yaml_path, **extra) -> argparse.Namespace:
    base = dict(yaml=str(yaml_path), quiet=True)
    base.update(extra)
    return argparse.Namespace(**base)


@pytest.fixture
def al_poscar(tmp_path):
    al = fcc_al_dict()
    path = tmp_path / "POSCAR"
    write_poscar(str(path), al["lattice"], al["positions"], al["atom_types"])
    return path


_LJ_POT = {"type": "LJ", "species": ["Al"],
          "params": {"epsilon": 0.4, "sigma": 2.5}, "r_cut": 5.0}

# Same potential at a cutoff that actually fits a 2x2x2 FCC Al supercell:
# NeighbourList requires r_cut + skin_width < half the smallest
# perpendicular cell width, which is 4.05 A for that supercell.
_LJ_POT_FITTED = {"type": "LJ", "species": ["Al"],
                 "params": {"epsilon": 0.4, "sigma": 2.5}, "r_cut": 3.2}


def _al_supercell_poscar(tmp_path, n=2, name="POSCAR"):
    """Write an n x n x n FCC Al supercell POSCAR and return its path.

    The bare 4.05 A conventional cell is too small to host any useful
    cutoff.  Commands that drive a GeometryOptimiser (relax, and tsa via
    its IRC relaxations) need this just as much as md does -- that only
    became visible once GeometryOptimiser started building a neighbour
    list, which is the one place the minimum-image precondition is
    enforced.
    """
    base = fcc_al_dict()
    lattice = base["lattice"] * n
    positions, atom_types = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                shift = np.array([i, j, k], dtype=float) @ base["lattice"]
                positions.append(base["positions"] + shift)
                atom_types.extend(base["atom_types"])
    positions = np.concatenate(positions, axis=0)
    poscar = tmp_path / name
    write_poscar(str(poscar), lattice, positions, atom_types)
    return poscar


def test_dispatch_table_has_no_fitpot():
    """F1 regression: fitpot must be gone from the dispatch table."""
    assert "fitpot" not in _DISPATCH
    assert "fit" in _DISPATCH


def test_parser_potential_choices_include_all_six():
    """F3 regression: --potential must accept all 6 classical potentials."""
    parser = build_parser()
    ns = parser.parse_args(["fit", "--potential", "sw"])
    assert ns.potential == "sw"
    for pot in ("buckingham", "lj", "morse", "bornmayer", "sw", "tersoff"):
        ns = parser.parse_args(["fit", "--potential", pot])
        assert ns.potential == pot


def test_cmd_md_npt_smoke(tmp_path):
    """C2 regression: `akitsushima md` with ensemble: npt raised
    TypeError before reaching step 0, because cmd_md passed
    velocity_rescale_steps -- an NVTSimulation-only parameter -- to
    NPTSimulation unconditionally.  The shipped NPT template sets that
    key too, so removing it from the YAML did not help.  The NPT CLI
    path had never run.  Exercised with the key present, which is the
    combination that used to fail.
    """
    poscar = _al_supercell_poscar(tmp_path)
    cfg = {
        "structure": {"poscar": str(poscar)},
        "potential": _LJ_POT_FITTED,
        "md": {"ensemble": "npt", "output_dir": str(tmp_path / "npt_out"),
              "temperature": 300.0, "pressure": 0.0, "timestep": 0.5,
              # equil_steps >= velocity_rescale_steps: rescaling past
              # equilibration lands on production frames and warns.
              "n_steps": 20, "equil_steps": 5, "tau_nhc": 100.0,
              "tau_baro": 1000.0, "velocity_rescale_steps": 5,
              "traj_every": 100, "thermo_every": 100, "verbose": False},
    }
    yaml_path = tmp_path / "npt.yaml"
    _write_yaml(yaml_path, cfg)
    assert cmd_md(_ns(yaml_path)) == 0
    assert (tmp_path / "npt_out" / "POSCAR_final_NPT").is_file()


def test_cmd_relax_smoke(tmp_path):
    poscar = _al_supercell_poscar(tmp_path)
    cfg = {
        "structure": {"poscar": str(poscar)},
        "potential": _LJ_POT_FITTED,
        "relax": {"output_dir": str(tmp_path / "relax_out"),
                  "method": "fire", "fmax": 0.05, "max_steps": 100,
                  "verbose": False},
    }
    yaml_path = tmp_path / "relax.yaml"
    _write_yaml(yaml_path, cfg)
    assert cmd_relax(_ns(yaml_path)) == 0
    assert (tmp_path / "relax_out" / "POSCAR_relaxed").is_file()


def test_cmd_md_smoke(tmp_path):
    # NeighbourList enforces r_cut + skin_width < half the shortest cell
    # vector (bug E5); the bare 4.05 A conventional FCC Al cell is too
    # small for any physically meaningful cutoff, so use a 2x2x2
    # supercell instead (same fixture pattern as test_solvers.py).
    n = 2
    base = fcc_al_dict()
    lattice = base["lattice"] * n
    positions, atom_types = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                shift = np.array([i, j, k], dtype=float) @ base["lattice"]
                positions.append(base["positions"] + shift)
                atom_types.extend(base["atom_types"])
    positions = np.concatenate(positions, axis=0)
    poscar = tmp_path / "POSCAR"
    write_poscar(str(poscar), lattice, positions, atom_types)

    cfg = {
        "structure": {"poscar": str(poscar)},
        "potential": {"type": "LJ", "species": ["Al"],
                     "params": {"epsilon": 0.4, "sigma": 2.5}, "r_cut": 3.2},
        "md": {"ensemble": "nve", "output_dir": str(tmp_path / "md_out"),
              "temperature": 300.0, "timestep": 0.2, "n_steps": 20,
              "equil_steps": 0, "traj_every": 100, "thermo_every": 100,
              "verbose": False},
    }
    yaml_path = tmp_path / "md.yaml"
    _write_yaml(yaml_path, cfg)
    assert cmd_md(_ns(yaml_path)) == 0


def test_cmd_gcmc_smoke(tmp_path):
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 15.0, size=(6, 3))
    poscar = tmp_path / "POSCAR"
    write_poscar(str(poscar), np.eye(3) * 15.0, positions, ["Ar"] * 6)

    cfg = {
        "structure": {"poscar": str(poscar)},
        "potential": {"type": "LJ", "species": ["Ar"],
                     "params": {"epsilon": 0.0103, "sigma": 3.405}, "r_cut": 6.0},
        "gcmc": {"output_dir": str(tmp_path / "gcmc_out"),
                "species": "Ar", "mu": -0.05, "temperature": 120.0,
                "n_cycles": 10, "md_steps_per_cycle": 5,
                "gcmc_attempts_per_cycle": 8, "seed": 1, "verbose": False},
    }
    yaml_path = tmp_path / "gcmc.yaml"
    _write_yaml(yaml_path, cfg)
    assert cmd_gcmc(_ns(yaml_path)) == 0


def test_cmd_gcmc_smoke_accepts_deprecated_mu_ex_yaml_key(tmp_path):
    """
    Backward compatibility for config files predating the `mu_ex` -> `mu`
    rename (see GCMCMDSimulation's `mu` parameter note and
    _resolve_mu_from_cfg in cli/common.py): a 'gcmc.mu_ex:' key with no
    'gcmc.mu:' key must still be read and used, with a warning, rather
    than silently falling back to the unrelated hardcoded default.
    """
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 15.0, size=(6, 3))
    poscar = tmp_path / "POSCAR"
    write_poscar(str(poscar), np.eye(3) * 15.0, positions, ["Ar"] * 6)

    cfg = {
        "structure": {"poscar": str(poscar)},
        "potential": {"type": "LJ", "species": ["Ar"],
                     "params": {"epsilon": 0.0103, "sigma": 3.405}, "r_cut": 6.0},
        "gcmc": {"output_dir": str(tmp_path / "gcmc_out"),
                "species": "Ar", "mu_ex": -0.234, "temperature": 120.0,
                "n_cycles": 10, "md_steps_per_cycle": 5,
                "gcmc_attempts_per_cycle": 8, "seed": 1, "verbose": False},
    }
    yaml_path = tmp_path / "gcmc.yaml"
    _write_yaml(yaml_path, cfg)
    with pytest.warns(UserWarning, match="mu_ex"):
        assert cmd_gcmc(_ns(yaml_path)) == 0


def test_cmd_neb_smoke(tmp_path):
    """S5/S20 regression coverage at the CLI level."""
    n = 2
    base = fcc_al_dict()
    lattice = base["lattice"] * n
    positions, atom_types = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                shift = np.array([i, j, k], dtype=float) @ base["lattice"]
                positions.append(base["positions"] + shift)
                atom_types.extend(base["atom_types"])
    positions = np.concatenate(positions, axis=0)
    pos_final = positions.copy()
    pos_final[0] = pos_final[0] + np.array([0.3, 0.0, 0.0])

    write_poscar(str(tmp_path / "POSCAR_initial"), lattice, positions, atom_types)
    write_poscar(str(tmp_path / "POSCAR_final"), lattice, pos_final, atom_types)

    cfg = {
        "structure": {"poscar_initial": str(tmp_path / "POSCAR_initial"),
                     "poscar_final": str(tmp_path / "POSCAR_final")},
        "potential": {"type": "LJ", "species": ["Al"],
                     "params": {"epsilon": 0.4, "sigma": 2.5}, "r_cut": 3.2},
        "neb": {"output_dir": str(tmp_path / "neb_out"), "n_images": 3,
               "k_spring": 5.0, "climbing": True, "fmax": 0.1,
               "max_steps": 100, "verbose": False},
    }
    yaml_path = tmp_path / "neb.yaml"
    _write_yaml(yaml_path, cfg)
    assert cmd_neb(_ns(yaml_path)) == 0


def test_cmd_tsa_smoke(tmp_path):
    """S6/S7/S22 regression coverage at the CLI level."""
    poscar = _al_supercell_poscar(tmp_path, name="POSCAR_saddle")
    st = parse_poscar(str(poscar))
    rng = np.random.default_rng(0)
    pos = st["positions"] + rng.uniform(-0.05, 0.05,
                                        size=st["positions"].shape)
    write_poscar(str(poscar), st["lattice"], pos, st["atom_types"])

    cfg = {
        "structure": {"poscar": str(poscar)},
        "potential": _LJ_POT_FITTED,
        "tsa": {"output_dir": str(tmp_path / "tsa_out"), "delta": 0.01,
               "run_irc": True, "irc_delta": 0.05, "irc_max_steps": 30,
               "temperature": 300.0, "verbose": False},
    }
    yaml_path = tmp_path / "tsa.yaml"
    _write_yaml(yaml_path, cfg)
    assert cmd_tsa(_ns(yaml_path)) == 0


def test_cmd_osmc_smoke(tmp_path):
    """S7 regression coverage at the CLI level."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 15.0, size=(6, 3))
    poscar = tmp_path / "POSCAR"
    write_poscar(str(poscar), np.eye(3) * 15.0, positions, ["Ar"] * 6)

    cfg = {
        "structure": {"poscar": str(poscar)},
        "potential": {"type": "LJ", "species": ["Ar"],
                     "params": {"epsilon": 0.0103, "sigma": 3.405}, "r_cut": 6.0},
        "osmc": {"output_dir": str(tmp_path / "osmc_out"), "species": "Ar",
                "mu": -0.05, "temperature": 120.0, "n_cycles": 10,
                "md_steps_per_cycle": 5, "gcmc_attempts_per_cycle": 8,
                "m_lam": 50.0, "sc_alpha": 0.5, "sc_n": 4, "seed": 1,
                "verbose": False},
    }
    yaml_path = tmp_path / "osmc.yaml"
    _write_yaml(yaml_path, cfg)
    assert cmd_osmc(_ns(yaml_path)) == 0


def test_cmd_osmc_smoke_accepts_deprecated_mu_ex_yaml_key(tmp_path):
    """Backward compatibility counterpart of the gcmc test above, for the
    'osmc' YAML section."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 15.0, size=(6, 3))
    poscar = tmp_path / "POSCAR"
    write_poscar(str(poscar), np.eye(3) * 15.0, positions, ["Ar"] * 6)

    cfg = {
        "structure": {"poscar": str(poscar)},
        "potential": {"type": "LJ", "species": ["Ar"],
                     "params": {"epsilon": 0.0103, "sigma": 3.405}, "r_cut": 6.0},
        "osmc": {"output_dir": str(tmp_path / "osmc_out"), "species": "Ar",
                "mu_ex": -0.234, "temperature": 120.0, "n_cycles": 10,
                "md_steps_per_cycle": 5, "gcmc_attempts_per_cycle": 8,
                "m_lam": 50.0, "sc_alpha": 0.5, "sc_n": 4, "seed": 1,
                "verbose": False},
    }
    yaml_path = tmp_path / "osmc.yaml"
    _write_yaml(yaml_path, cfg)
    with pytest.warns(UserWarning, match="mu_ex"):
        assert cmd_osmc(_ns(yaml_path)) == 0


def test_cmd_esm_smoke(tmp_path):
    """S21 regression coverage at the CLI level (no relax: section, so
    ESMCalculator would previously never be __call__-ed before its
    profiles were read)."""
    lattice = np.diag([10.0, 10.0, 40.0])   # slab + vacuum along z
    positions = np.array([[5.0, 5.0, 15.0], [5.0, 5.0, 17.0]])
    poscar = tmp_path / "POSCAR_slab"
    write_poscar(str(poscar), lattice, positions, ["Mg", "O"])

    cfg = {
        "structure": {"poscar": str(poscar)},
        "potential": {"type": "buckingham",
                     "charges": {"Mg": 2.0, "O": -2.0}},
        "esm": {"output_dir": str(tmp_path / "esm_out"), "bc": "bc1",
                "sigma": 0.5, "n_grid_xy": 8, "n_grid_z": 32,
                "r_cut": 8.0, "verbose": False},
    }
    yaml_path = tmp_path / "esm.yaml"
    _write_yaml(yaml_path, cfg)
    assert cmd_esm(_ns(yaml_path)) == 0
    assert (tmp_path / "esm_out" / "work_function.txt").is_file()


def test_cmd_phonon_smoke(tmp_path, al_poscar, small_cell_ok):
    cfg = {
        "structure": {"poscar": str(al_poscar)},
        "potential": {"type": "LJ", "species": ["Al"],
                     "params": {"epsilon": 0.4, "sigma": 2.5}, "r_cut": 5.0},
        "phonon": {"output_dir": str(tmp_path / "phonon_out"),
                  "supercell": [2, 2, 2], "delta": 0.01, "verbose": False,
                  "dos": {"q_mesh": 2}},
    }
    yaml_path = tmp_path / "phonon.yaml"
    _write_yaml(yaml_path, cfg)
    assert cmd_phonon(_ns(yaml_path)) == 0


def test_cmd_kappa_smoke(tmp_path):
    n = 2
    base = fcc_al_dict()
    lattice = base["lattice"] * n
    positions, atom_types = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                shift = np.array([i, j, k], dtype=float) @ base["lattice"]
                positions.append(base["positions"] + shift)
                atom_types.extend(base["atom_types"])
    positions = np.concatenate(positions, axis=0)
    poscar = tmp_path / "POSCAR"
    write_poscar(str(poscar), lattice, positions, atom_types)

    cfg = {
        "structure": {"poscar": str(poscar)},
        "potential": {"type": "LJ", "species": ["Al"],
                     "params": {"epsilon": 0.4, "sigma": 2.5}, "r_cut": 3.2},
        "kappa": {"output_dir": str(tmp_path / "kappa_out"),
                 "temperature": 300.0, "n_steps": 100, "timestep": 0.2,
                 "equil_steps": 20, "heat_flux_every": 2, "traj_every": 1000,
                 "thermo_every": 1000, "tau_max": 0.01, "n_blocks": 2,
                 "verbose": False},
    }
    yaml_path = tmp_path / "kappa.yaml"
    _write_yaml(yaml_path, cfg)
    assert cmd_kappa(_ns(yaml_path)) == 0


def test_cmd_mlip_smoke(tmp_path):
    """S23/F6 regression coverage at the CLI level."""
    rng = np.random.default_rng(0)
    lines = ["target,label,weight,feat0,feat1,feat2"]
    for _ in range(20):
        x = rng.normal(size=3)
        lines.append(f"{float(np.sum(x))},E,1.0,{x[0]},{x[1]},{x[2]}")
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("\n".join(lines))

    cfg = {
        "structure": {"training_csv": str(csv_path)},
        "potential": {"r_cut": 5.0},
        "mlip": {"output_dir": str(tmp_path / "mlip_out"), "model": "elm",
                "n_hidden": 16, "reg": 1e-3, "verbose": False},
    }
    yaml_path = tmp_path / "mlip.yaml"
    _write_yaml(yaml_path, cfg)
    assert cmd_mlip(_ns(yaml_path)) == 0
    assert (tmp_path / "mlip_out" / "elm_model.npz").is_file()


def test_cmd_mlip_training_dir_gives_clear_error_not_crash(tmp_path):
    """S23 regression: training_dir must not raise a low-level exception."""
    os.makedirs(tmp_path / "some_dir", exist_ok=True)
    cfg = {
        "structure": {"training_dir": str(tmp_path / "some_dir")},
        "potential": {"r_cut": 5.0},
        "mlip": {"output_dir": str(tmp_path / "mlip_out2"), "model": "elm",
                "verbose": False},
    }
    yaml_path = tmp_path / "mlip2.yaml"
    _write_yaml(yaml_path, cfg)
    assert cmd_mlip(_ns(yaml_path)) == 1


def _write_lj_training_set(tmp_path):
    from akitsushima.engine import LennardJonesPotential, PotentialEngine

    pair = frozenset({"Al"})
    ref_engine = PotentialEngine().add(
        LennardJonesPotential({pair: {"epsilon": 0.4, "sigma": 2.5}},
                              r_cut=6.0, shift=True))
    al = fcc_al_dict()
    structures = [al, randomly_perturbed(al, magnitude=0.05, seed=1)]

    rows = ["target,label,weight,struct_idx"]
    for sid, s in enumerate(structures):
        sdir = tmp_path / "training_set" / str(sid)
        os.makedirs(sdir, exist_ok=True)
        write_poscar(str(sdir / "POSCAR"), s["lattice"], s["positions"],
                    s["atom_types"])
        E, F, S = ref_engine.compute(s["lattice"], s["positions"], s["atom_types"])
        N = len(s["atom_types"])
        rows.append(f"{E/N},E,1.0,{sid}")
        for f in F.ravel():
            rows.append(f"{f},F,1.0,{sid}")
        xx, yy, zz = S[0, 0], S[1, 1], S[2, 2]
        yz, xz, xy = S[1, 2], S[0, 2], S[0, 1]
        for v in (xx, yy, zz, yz, xz, xy):
            rows.append(f"{v},S,1.0,{sid}")

    csv_path = tmp_path / "training_set" / "structures.csv"
    csv_path.write_text("\n".join(rows))
    return csv_path


def test_cmd_fit_lj_end_to_end(tmp_path, small_cell_ok):
    """F1/F2/F3/F5/F12 regression coverage: the consolidated `fit` command
    must recover parameters close to the true LJ potential used to
    generate the (synthetic) training data."""
    csv_path = _write_lj_training_set(tmp_path)

    args = argparse.Namespace(
        potential="lj", config=None, write_config=None,
        training_csv=str(csv_path), output_dir=str(tmp_path / "fit_lj_out"),
        r_cut=None, r0=None, eps0=None, use_de=False,
        no_forces=False, no_stress=False, quiet=True)
    assert cmd_fit(args) == 0

    import json
    with open(tmp_path / "fit_lj_out" / "fitted_potential.json") as fh:
        fitted = json.load(fh)
    # "Al-Al" is the actually-trained pair; the untouched default
    # template's own example pair ("Ar-Ar", irrelevant to this Al-only
    # training set) is also present, auto-filled-around rather than
    # replaced -- see the S19/F2 area notes for why _build_param_set's
    # auto-fill only ever ADDS missing pairs, never prunes extras.
    pair_params = fitted["params"]["Al-Al"]
    assert pair_params["epsilon"] == pytest.approx(0.4, rel=0.5)
    assert pair_params["sigma"] == pytest.approx(2.5, rel=0.2)


def test_cmd_fit_lj_huber_loss_type_end_to_end(tmp_path, small_cell_ok):
    """Phase 3 (Huber loss) regression: cfg["loss_type"]="huber" plus
    the required huber_delta_* keys must flow all the way from a JSON
    --config file through cmd_fit.py -> fit_potential ->
    generic_loss_function without crashing, and must still recover
    parameters close to the true LJ potential."""
    import json

    csv_path = _write_lj_training_set(tmp_path)
    cfg = {
        "potential": "lj", "training_csv": str(csv_path),
        "output_dir": str(tmp_path / "fit_lj_huber_out"),
        "r_cut": 6.0, "shift": True,
        "initial_lj": {"Al-Al": {"epsilon": 0.3, "sigma": 2.6}},
        "loss_type": "huber",
        "huber_delta_energy": 0.05, "huber_delta_forces": 0.05,
        "huber_delta_stress": 0.05,
    }
    cfg_path = tmp_path / "fit_lj_huber_cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    args = argparse.Namespace(
        potential="lj", config=str(cfg_path), write_config=None,
        training_csv=None, output_dir=None,
        r_cut=None, r0=None, eps0=None, use_de=False,
        no_forces=False, no_stress=False, quiet=True)
    assert cmd_fit(args) == 0

    with open(tmp_path / "fit_lj_huber_out" / "fitted_potential.json") as fh:
        fitted = json.load(fh)
    pair_params = fitted["params"]["Al-Al"]
    assert pair_params["epsilon"] == pytest.approx(0.4, rel=0.5)
    assert pair_params["sigma"] == pytest.approx(2.5, rel=0.2)


def test_cmd_fit_mie_end_to_end(tmp_path, small_cell_ok):
    """Phase 2 registry regression: `fit --potential mie` must work
    end-to-end through the exact same cmd_fit.py/parser.py/templates.py
    code paths as the original six potentials, having required zero
    edits to any of those three files -- only a new PotentialSpec
    registration in fitting/registry.py (see its module docstring). This
    is the concrete demonstration that Phase 1's registry consolidation
    actually reduces "add a potential" to touching one file."""
    from akitsushima.engine import MiePotential, PotentialEngine

    pair = frozenset({"Al"})
    ref_engine = PotentialEngine().add(
        MiePotential({pair: {"epsilon": 0.4, "sigma": 2.5}},
                    n=12.0, m=6.0, r_cut=6.0, shift=True))
    al = fcc_al_dict()
    structures = [al, randomly_perturbed(al, magnitude=0.05, seed=1)]

    rows = ["target,label,weight,struct_idx"]
    for sid, s in enumerate(structures):
        sdir = tmp_path / "training_set" / str(sid)
        os.makedirs(sdir, exist_ok=True)
        write_poscar(str(sdir / "POSCAR"), s["lattice"], s["positions"],
                    s["atom_types"])
        E, F, S = ref_engine.compute(s["lattice"], s["positions"], s["atom_types"])
        N = len(s["atom_types"])
        rows.append(f"{E/N},E,1.0,{sid}")
        for f in F.ravel():
            rows.append(f"{f},F,1.0,{sid}")
        xx, yy, zz = S[0, 0], S[1, 1], S[2, 2]
        yz, xz, xy = S[1, 2], S[0, 2], S[0, 1]
        for v in (xx, yy, zz, yz, xz, xy):
            rows.append(f"{v},S,1.0,{sid}")
    csv_path = tmp_path / "training_set" / "structures.csv"
    csv_path.write_text("\n".join(rows))

    args = argparse.Namespace(
        potential="mie", config=None, write_config=None,
        training_csv=str(csv_path), output_dir=str(tmp_path / "fit_mie_out"),
        r_cut=None, r0=None, eps0=None, use_de=False,
        no_forces=False, no_stress=False, quiet=True)
    assert cmd_fit(args) == 0

    import json
    with open(tmp_path / "fit_mie_out" / "fitted_potential.json") as fh:
        fitted = json.load(fh)
    pair_params = fitted["params"]["Al-Al"]
    assert pair_params["epsilon"] == pytest.approx(0.4, rel=0.5)
    assert pair_params["sigma"] == pytest.approx(2.5, rel=0.2)


def test_cmd_fit_r_cut_zero_is_not_ignored(tmp_path, small_cell_ok):
    """F12 regression: --r-cut 0 must not be silently dropped by a
    truthiness check (it must actually reach cfg["r_cut"])."""
    csv_path = _write_lj_training_set(tmp_path)

    args = argparse.Namespace(
        potential="lj", config=None, write_config=None,
        training_csv=str(csv_path), output_dir=str(tmp_path / "fit_rcut0_out"),
        r_cut=0.0, r0=None, eps0=None, use_de=False,
        no_forces=False, no_stress=False, quiet=True)
    # r_cut=0 makes every pair distance exceed the cutoff, so every
    # structure contributes zero pair energy/force -- fit_potential()
    # itself may raise or simply converge trivially, but cmd_fit's own
    # CLI-override logic must be exercised (not silently ignored).
    try:
        cmd_fit(args)
    except Exception:
        pass

    import akitsushima.cli.cmd_fit as cmd_fit_mod
    captured = {}
    orig = cmd_fit_mod.fit_potential

    def _spy(*a, **kw):
        captured["r_cut"] = kw.get("r_cut")
        return orig(*a, **kw)

    cmd_fit_mod.fit_potential = _spy
    try:
        args.output_dir = str(tmp_path / "fit_rcut0_out2")
        try:
            cmd_fit(args)
        except Exception:
            pass
    finally:
        cmd_fit_mod.fit_potential = orig

    assert captured.get("r_cut") == 0.0, (
        "F12 regression: --r-cut 0 must reach fit_potential(), not be "
        "dropped by an `if args.r_cut:` truthiness check")


def test_build_param_set_sw_and_tersoff_construct_without_keyerror():
    """
    F2 regression: StillingerWeberParameterSet/TersoffParameterSet's
    initial_sw/initial_t are FLAT parameter dicts ({"epsilon": ...,
    "sigma": ..., ...}), not species-pair-keyed like LJ/Morse/BornMayer/
    Buckingham's initial_* dicts. The original applied the same
    "Sp1-Sp2" -> frozenset-pair splitting used for those four here too,
    turning every flat key (e.g. "epsilon", which contains no "-") into
    the useless 1-tuple ("epsilon",) -- so the ParameterSet's own code,
    which looks up the literal string key "epsilon", always raised
    KeyError. This checks construction only (fast); the full
    fit_potential() optimisation loop for SW/Tersoff is exercised
    end-to-end (slowly, via differential_evolution) in a separate,
    non-pytest smoke run.
    """
    from akitsushima.cli.cmd_fit import _build_param_set
    from akitsushima.cli.templates import (
        _DEFAULT_FIT_CFG_SW,
        _DEFAULT_FIT_CFG_TERSOFF,
    )

    sw_set = _build_param_set("sw", _DEFAULT_FIT_CFG_SW, ["Si"], [], 3.77118)
    assert sw_set is not None
    assert sw_set.x0 is not None and len(sw_set.x0) > 0

    t_set = _build_param_set(
        "tersoff", _DEFAULT_FIT_CFG_TERSOFF, ["C"], [], 2.10)
    assert t_set is not None
    assert t_set.x0 is not None and len(t_set.x0) > 0


def test_fit_cfg_templates_all_have_r_cut():
    """S25 regression: cmd_fit's r_cut = float(cfg["r_cut"]) is
    unconditional for every potential type, but the SW/Tersoff templates
    never defined an "r_cut" key at all, so `fit --potential sw` (and
    `tersoff`) raised KeyError even after F2's fix let it get that far.
    """
    from akitsushima.cli.templates import _FIT_CFG_TEMPLATES

    for name, cfg in _FIT_CFG_TEMPLATES.items():
        assert "r_cut" in cfg, f"{name} template is missing 'r_cut'"
        assert float(cfg["r_cut"]) > 0


def test_esm_template_rcut_matches_the_cell_size_it_documents():
    """
    The shipped ESM template pairs ``r_cut: 15.0`` with a ``relax:``
    section, and ``cmd_esm`` hands that r_cut to ``GeometryOptimiser``,
    which builds a ``NeighbourList`` with the default ``skin_width=0.5``.
    So the template silently requires every PERPENDICULAR cell width to
    exceed ``2 * (r_cut + 0.5)`` -- 31.0 A at the shipped value.  For an
    MgO slab (a = 4.21 A) that is an 8x8 in-plane supercell; the 4x4 and
    6x6 slabs a user would reach for first are rejected outright.

    Nothing caught this because ``test_cmd_esm_smoke`` uses ``r_cut: 8.0``
    AND omits the ``relax:`` section, so it never constructs a
    NeighbourList at all -- the template's own defaults were never
    executed.  This test pins the arithmetic the template's comment now
    states, so the comment cannot drift away from the code.

    It deliberately checks the constraint rather than running a full ESM
    relaxation on a 128-atom slab, which would cost minutes for no extra
    information: the failure mode is NeighbourList construction, and that
    is exactly what is exercised here.
    """
    import re

    from akitsushima.cli.templates import _YAML_ESM_DEFAULT
    from akitsushima.engine.neighborlist import NeighbourList, _cell_metrics

    m = re.search(r"^\s*r_cut:\s*([0-9.]+)", _YAML_ESM_DEFAULT, re.MULTILINE)
    assert m, "ESM template must declare an r_cut"
    r_cut = float(m.group(1))
    assert "relax:" in _YAML_ESM_DEFAULT, (
        "this test's premise is that the template enables relaxation, "
        "which is what pulls in the NeighbourList constraint")

    skin = 0.5                      # GeometryOptimiser's default
    required_width = 2.0 * (r_cut + skin)

    # The comment in the template states 31.0 A for the shipped r_cut.
    assert required_width == pytest.approx(31.0), (
        f"the ESM template's comment documents a 31.0 A minimum width; "
        f"r_cut={r_cut} + skin={skin} now implies {required_width} A -- "
        f"update the comment alongside the value")

    a_mgo = 4.21                    # MgO rocksalt lattice constant [A]
    probe = np.array([[0.0, 0.0, 20.0]])
    for n, should_pass in ((4, False), (6, False), (8, True)):
        lat = np.diag([n * a_mgo, n * a_mgo, 40.0])
        width = float(_cell_metrics(lat)[0].min())
        nl = NeighbourList(r_cut=r_cut, skin_width=skin)
        if should_pass:
            assert width > required_width, "fixture sanity"
            nl.build(lat, probe)    # must not raise
        else:
            assert width < required_width, "fixture sanity"
            with pytest.raises(ValueError, match="perpendicular"):
                nl.build(lat, probe)


def test_relax_template_keys_are_all_read_by_cmd_relax():
    """Every key the relax template ships must actually be consumed.

    This project has hit the template-key/loader-mismatch defect
    repeatedly: a GCMC template using epsilon_Ar_Ar where the code reads
    epsilon, an SW fit template missing the r_cut key cmd_fit reads
    unconditionally, and an MD template shipping `timing:` that cmd_md
    never looked at.  A key that the loader ignores is worse than a
    missing one -- the user sets it, sees no effect, and has no way to
    tell which of the two happened.

    Structural rather than behavioural on purpose: it fails the moment
    someone adds a key to the template without wiring it up, which is
    exactly when the mistake is cheap to fix.
    """
    import inspect
    import yaml as _yaml
    from akitsushima.cli import cmd_relax as _cmd_relax_mod
    from akitsushima.cli.templates import _YAML_RELAX_DEFAULT

    cfg = _yaml.safe_load(_YAML_RELAX_DEFAULT)
    src = inspect.getsource(_cmd_relax_mod)
    unread = [k for k in cfg.get("relax", {})
              if f'rel_cfg.get("{k}"' not in src
              and f"rel_cfg.get('{k}'" not in src
              and f'rel_cfg["{k}"]' not in src]
    assert not unread, (
        f"relax template keys never read by cmd_relax: {unread}")
