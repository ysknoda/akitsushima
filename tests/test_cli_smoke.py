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


def test_cmd_relax_buckingham_coulomb_runs(tmp_path):
    """`relax` on a Buckingham + Coulomb config must actually run.

    The Buckingham branch of _build_force_fn_from_yaml deliberately
    returns force_fn=None plus (buck_params, charges), which the solver
    takes as legacy constructor kwargs.  cmd_relax used to evaluate the
    initial state by calling force_fn() directly, before the guard that
    handles the None, so this whole path died with "'NoneType' object is
    not callable" and the Buckingham + Coulomb relax had never run.  The
    `md` equivalent was fine, which is why nothing else caught it.

    Rocksalt MgO with the Lewis-Catlow parameters: perfect symmetry, so
    the forces are zero and the run converges immediately -- this test
    is about the path executing at all, not about the optimiser.  The
    energy assertion pins the one number that would move if the legacy
    Buckingham kwargs were silently dropped rather than forwarded.
    """
    import json

    a, n = 4.212, 2
    lattice = np.eye(3) * a * n
    fcc = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    base = np.concatenate([fcc, fcc + [.5, 0, 0]])
    frac, types = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                frac.append((base + [i, j, k]) / n)
                types += ["Mg"] * 4 + ["O"] * 4
    positions = np.concatenate(frac) @ lattice

    poscar = tmp_path / "POSCAR_MgO"
    write_poscar(str(poscar), lattice, positions, types)

    pot_json = tmp_path / "mgo_buck.json"
    pot_json.write_text(json.dumps({
        "potential_type": "buckingham",
        "r_cut": 3.5, "species": ["Mg", "O"], "final_loss": 0.0,
        "params": {"Mg-O":  {"A": 1279.69,  "rho": 0.29969, "C": 0.0},
                   "O-O":   {"A": 22764.30, "rho": 0.14900, "C": 27.88},
                   "Mg-Mg": {"A": 0.0,      "rho": 0.1,     "C": 0.0}},
        "charges": {"Mg": 2.0, "O": -2.0},
    }))

    cfg = {
        "structure": {"poscar": str(poscar)},
        "potential": {"type": "Buckingham", "potential_json": str(pot_json),
                      "r_cut": 3.5, "coulomb_method": "ewald"},
        "relax": {"output_dir": str(tmp_path / "buck_out"), "method": "fire",
                  "fmax": 0.05, "max_steps": 5, "verbose": False},
    }
    yaml_path = tmp_path / "relax_buck.yaml"
    _write_yaml(yaml_path, cfg)

    assert cmd_relax(_ns(yaml_path)) == 0
    assert (tmp_path / "buck_out" / "POSCAR_relaxed").is_file()

    # MgO's measured lattice energy is about -41 eV per MgO formula unit,
    # i.e. -20.5 eV/atom; these parameters reproduce it.  A path that
    # lost the charges would come out near the (tiny, purely repulsive)
    # short-range energy instead, and a path that lost the short-range
    # term would come out far too negative.
    log = (tmp_path / "buck_out" / "relax.log").read_text().splitlines()
    energies = []
    for ln in log:                      # skip the header and "Converged:" rows
        f = ln.split()
        if len(f) >= 2:
            try:
                energies.append(float(f[1]))
            except ValueError:
                pass
    assert energies, "relax.log carried no energy rows"
    assert -21.5 < energies[-1] / len(types) < -19.5, (
        f"MgO Buckingham + Ewald should give about -20.6 eV/atom, got "
        f"{energies[-1] / len(types):.3f}")


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


def _osmc_cfg(tmp_path, **sec):
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 15.0, size=(6, 3))
    poscar = tmp_path / "POSCAR"
    write_poscar(str(poscar), np.eye(3) * 15.0, positions, ["Ar"] * 6)
    osmc = {"output_dir": str(tmp_path / "osmc_out"), "species": "Ar",
            "mu": -0.05, "temperature": 120.0, "n_cycles": 10,
            "equil_cycles": 2, "md_steps_per_cycle": 5,
            "gcmc_attempts_per_cycle": 8, "seed": 1, "verbose": False}
    osmc.update(sec)
    return {
        "structure": {"poscar": str(poscar)},
        "potential": {"type": "LJ", "species": ["Ar"],
                     "params": {"epsilon": 0.0103, "sigma": 3.405}, "r_cut": 6.0},
        "osmc": osmc,
    }


def test_cmd_osmc_smoke(tmp_path):
    """osmc runs CFCMC (ADR-0009) and writes its estimators."""
    yaml_path = tmp_path / "osmc.yaml"
    _write_yaml(yaml_path, _osmc_cfg(tmp_path))
    assert cmd_osmc(_ns(yaml_path)) == 0
    out = tmp_path / "osmc_out"
    for f in ("cfcmc_estimates.json", "lambda_vs_cycle.dat",
              "POSCAR_gcmc_final", "N_vs_cycle.dat"):
        assert (out / f).is_file()


def test_cmd_osmc_refuses_old_lambda_dynamics_keys(tmp_path):
    yaml_path = tmp_path / "osmc.yaml"
    _write_yaml(yaml_path, _osmc_cfg(tmp_path, m_lam=50.0, sc_alpha=0.5))
    with pytest.raises(ValueError, match="ADR-0009"):
        cmd_osmc(_ns(yaml_path))


def test_cmd_osmc_smoke_accepts_deprecated_mu_ex_yaml_key(tmp_path):
    """Backward compatibility counterpart of the gcmc test above, for the
    'osmc' YAML section."""
    cfg = _osmc_cfg(tmp_path)
    cfg["osmc"]["mu_ex"] = cfg["osmc"].pop("mu")
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


def test_esm_command_is_analysis_only():
    """ADR-0008: the esm template has no relax section any more (it relaxed
    on electrostatics alone and collapsed the ions), and a config that
    still has one is refused with a pointer to relax + coulomb_method: esm.
    (Replaces the check of the template's r_cut, which only that relax
    section used.)"""
    from akitsushima.cli.templates import _YAML_ESM_DEFAULT
    assert "relax:" not in _YAML_ESM_DEFAULT

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


# ---------------------------------------------------------------------------
# fit --potential sw, end to end
# ---------------------------------------------------------------------------

_SW_SI_TRUE = {"epsilon": 2.1683, "sigma": 2.0951, "a": 1.80, "gamma": 1.20,
               "lambda": 21.0, "A": 7.049556277, "B": 0.6022245584,
               "p": 4.0, "q": 0.0, "costheta0": -1.0 / 3.0}


def _write_sw_si_training_set(root):
    """Three 64-atom Si cells labelled with Stillinger-Weber Si itself.
    Rattled, because in a perfect diamond lattice the three-body term
    vanishes and lambda/gamma leave no trace in the data."""
    from akitsushima.engine import PotentialEngine, StillingerWeberPotential

    eng = PotentialEngine().add(
        StillingerWeberPotential({("Si", "Si", "Si"): _SW_SI_TRUE},
                                 species=["Si"]))
    basis = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                      [.25, .25, .25], [.25, .75, .75], [.75, .25, .75],
                      [.75, .75, .25]], float)
    frac = np.concatenate([(basis + [i, j, k]) / 2
                           for i in range(2) for j in range(2) for k in range(2)])
    rng = np.random.default_rng(7)
    rows = ["target,label,weight,struct_idx"]
    for sid, (scale, rattle) in enumerate([(1.0, 0.06), (0.98, 0.08),
                                           (1.02, 0.08)]):
        lat = np.eye(3) * (5.430950 * scale * 2)
        pos = frac @ lat + rng.normal(0.0, rattle, (64, 3))
        d = root / str(sid)
        os.makedirs(d, exist_ok=True)
        write_poscar(str(d / "POSCAR"), lat, pos, ["Si"] * 64)
        E, F, S = eng.compute(lat, pos, ["Si"] * 64)
        rows.append(f"{E / 64},E,1.0,{sid}")
        rows += [f"{f},F,1.0,{sid}" for f in F.ravel()]
        rows += [f"{v},S,1.0,{sid}" for v in (S[0, 0], S[1, 1], S[2, 2],
                                              S[1, 2], S[0, 2], S[0, 1])]
    (root / "structures.csv").write_text("\n".join(rows) + "\n")
    return root / "structures.csv"


def test_print_fit_result_handles_flat_sw_and_tersoff_params(capsys):
    """Single-species SW and Tersoff sets decode to a FLAT {name: value}
    dict.  print_fit_result used to treat every entry as (triple,
    sub-dict) and raised AttributeError on the first float -- after the
    optimisation had finished and before anything was saved."""
    from akitsushima.fitting.driver import print_fit_result
    print_fit_result({"params": {"sw": dict(_SW_SI_TRUE)}, "loss": 0.0}, "sw")
    print_fit_result({"params": {"tersoff": {"A": 1830.8, "B": 471.18}},
                      "loss": 0.0}, "tersoff")
    out = capsys.readouterr().out
    assert "lambda" in out and "1830.8" in out


def test_print_fit_result_handles_like_species_buckingham_pairs(capsys):
    """ADR-0003 T8: a like-species Buckingham pair is keyed frozenset({'O'}),
    and `s1, s2 = sorted(pair)` raised ValueError on it -- after the
    optimisation, before the result was saved, for every oxide fit."""
    from akitsushima.fitting.driver import print_fit_result
    buck = {frozenset({"Zn", "O"}): {"A": 529.7, "rho": 0.3581, "C": 0.0},
            frozenset({"O"}): {"A": 9547.96, "rho": 0.21916, "C": 32.0}}
    print_fit_result({"params": {"buck": buck, "charges": {"Zn": 2.0,
                                                          "O": -2.0}},
                      "loss": 0.0}, "buckingham")
    out = capsys.readouterr().out
    assert "O-O" in out and "O-Zn" in out


def test_cmd_fit_sw_end_to_end_saves_a_reloadable_potential(tmp_path):
    """The whole `fit --potential sw` path: fit, print, save, reload.

    Before the print_fit_result fix this returned only by raising, and
    no fitted_potential.json was ever written.  Starting from the true
    parameters, L-BFGS-B must stay there, and the reloaded potential
    must reproduce the training energy it was fitted to.
    """
    from akitsushima.fitting.io_utils import load_potential
    from akitsushima.io import parse_poscar

    csv_path = _write_sw_si_training_set(tmp_path / "training_set")
    cfg = {"training_csv": str(csv_path),
           "output_dir": str(tmp_path / "out"), "r_cut": 3.77118,
           "weights": {"energy": 1.0, "forces": 1.0, "stress": 1.0,
                       "r0": 0.1, "eps0": 0.01},
           "use_de": False, "val_fraction": 0.0,
           "species": ["Si"], "initial_sw": dict(_SW_SI_TRUE),
           "fix_params": {"epsilon": 2.1683}, "fit_keys": None}
    cfg_path = tmp_path / "fit_sw.yaml"
    _write_yaml(cfg_path, cfg)
    args = argparse.Namespace(
        potential="sw", config=str(cfg_path), write_config=None,
        training_csv=None, output_dir=None, r_cut=None, r0=None, eps0=None,
        use_de=False, no_forces=False, no_stress=False, quiet=True)
    assert cmd_fit(args) == 0

    out_json = tmp_path / "out" / "fitted_potential.json"
    assert out_json.is_file()
    engine, meta = load_potential(str(out_json))
    s0 = parse_poscar(str(tmp_path / "training_set" / "0" / "POSCAR"))
    e_fit, _, _ = engine.compute(s0["lattice"], s0["positions"],
                                 s0["atom_types"])
    ref = float(csv_path.read_text().splitlines()[1].split(",")[0])
    assert e_fit / 64 == pytest.approx(ref, abs=1e-6)


def test_cmd_fit_multi_species_sw_round_trips_through_the_record_format(tmp_path):
    """`fit --potential sw` with initial_sw given as i/j/k records: the
    fitted potential must be written back in that same record-list form
    (tuple keys are not JSON -- they used to be stringified into keys no
    loader could read) and must reload to the energy it was fitted to."""
    import json
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_fitting import _gesn_cell, _gesn_records, _sw_efs

    from akitsushima.fitting import MultiSpeciesStillingerWeberParameterSet as MS
    from akitsushima.fitting.io_utils import load_potential

    truth = MS._to_nested(_gesn_records())
    root = tmp_path / "training_set"
    rows = ["target,label,weight,struct_idx"]
    for sid, scale in enumerate((0.98, 1.0, 1.02)):
        lat, pos, types = _gesn_cell(seed=sid + 20, n_sn=32, scale=scale)
        order = np.argsort([t != "Ge" for t in types], kind="stable")
        lat, pos, types = lat, pos[order], [types[i] for i in order]
        os.makedirs(root / str(sid), exist_ok=True)
        write_poscar(str(root / str(sid) / "POSCAR"), lat, pos, types)
        E, F, S = _sw_efs(truth, lat, pos, types)
        rows.append(f"{E / 64},E,1.0,{sid}")
        rows += [f"{f},F,1.0,{sid}" for f in F.ravel()]
        rows += [f"{v},S,1.0,{sid}" for v in (S[0, 0], S[1, 1], S[2, 2],
                                              S[1, 2], S[0, 2], S[0, 1])]
    (root / "structures.csv").write_text("\n".join(rows) + "\n")

    cfg = {"training_csv": str(root / "structures.csv"),
           "output_dir": str(tmp_path / "out"), "r_cut": 4.5,
           "weights": {"energy": 1.0, "forces": 1.0, "stress": 1.0,
                       "r0": 0.1, "eps0": 0.01},
           "use_de": False, "val_fraction": 0.0, "species": ["Ge", "Sn"],
           "initial_sw": _gesn_records(), "fix_params": None,
           "fit_keys": ["A", "B"]}
    _write_yaml(tmp_path / "fit.yaml", cfg)
    args = argparse.Namespace(
        potential="sw", config=str(tmp_path / "fit.yaml"), write_config=None,
        training_csv=None, output_dir=None, r_cut=None, r0=None, eps0=None,
        use_de=False, no_forces=False, no_stress=False, quiet=True)
    assert cmd_fit(args) == 0

    out = tmp_path / "out" / "fitted_potential.json"
    saved = json.loads(out.read_text())["params"]
    assert isinstance(saved, list) and len(saved) == 8
    assert {(r["i"], r["j"], r["k"]) for r in saved} == set(truth)

    engine, _ = load_potential(str(out))
    s0 = parse_poscar(str(root / "1" / "POSCAR"))
    e_fit, _, _ = engine.compute(s0["lattice"], s0["positions"], s0["atom_types"])
    e_ref = _sw_efs(truth, s0["lattice"], s0["positions"], s0["atom_types"])[0]
    assert e_fit == pytest.approx(e_ref, abs=1e-6)


@pytest.mark.parametrize("method", ["BFGS", "Powell", "nonsense"])
def test_cmd_fit_refuses_a_bad_local_method_before_reading_any_data(tmp_path, capsys, method):
    """The training CSV named here does not exist.  If the method check ran
    after data loading, the error would be about the CSV instead."""
    cfg = {"training_csv": str(tmp_path / "missing.csv"),
           "output_dir": str(tmp_path / "out"), "r_cut": 6.0,
           "local_method": method}
    _write_yaml(tmp_path / "fit.yaml", cfg)
    args = argparse.Namespace(
        potential="lj", config=str(tmp_path / "fit.yaml"), write_config=None,
        training_csv=None, output_dir=None, r_cut=None, r0=None, eps0=None,
        use_de=False, no_forces=False, no_stress=False, quiet=True)
    assert cmd_fit(args) == 1
    out = capsys.readouterr().out
    assert "local_method" in out and "Training CSV not found" not in out


def test_esm_command_refuses_a_relax_section(tmp_path, capsys):
    """ADR-0008: esm.relax is refused with a pointer to relax + ESM."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 10.0, size=(2, 3))
    pos[:, 2] = [-1.0, 1.0]
    poscar = tmp_path / "POSCAR_slab"
    write_poscar(str(poscar), np.diag([12.0, 12.0, 24.0]), pos, ["Na", "Cl"])
    cfg = {"structure": {"poscar": str(poscar)},
           "potential": {"charges": {"Na": 1.0, "Cl": -1.0}},
           "esm": {"output_dir": str(tmp_path / "out"), "relax": {"max_steps": 2}}}
    yaml_path = tmp_path / "esm.yaml"
    _write_yaml(yaml_path, cfg)
    assert cmd_esm(_ns(yaml_path)) == 1
    assert "coulomb_method: esm" in capsys.readouterr().out


def test_cmd_fit_takes_the_potential_from_its_config(tmp_path, small_cell_ok):
    """ADR-0024 O4: `fit --config cfg` without --potential used
    --potential's default (buckingham) and failed with "Cannot build
    BUCKINGHAM ParameterSet"; the config's own `potential:` now applies."""
    import json

    from akitsushima.cli.parser import build_parser

    csv_path = _write_lj_training_set(tmp_path)
    cfg = {"potential": "lj", "training_csv": str(csv_path),
           "output_dir": str(tmp_path / "out"), "r_cut": 6.0, "shift": True,
           "initial_lj": {"Al-Al": {"epsilon": 0.3, "sigma": 2.6}}}
    cfg_path = tmp_path / "fit.json"
    cfg_path.write_text(json.dumps(cfg))
    args = build_parser().parse_args(["fit", "--config", str(cfg_path), "--quiet"])
    assert args.potential is None
    assert cmd_fit(args) == 0
    fitted = json.load(open(tmp_path / "out" / "fitted_potential.json"))
    assert fitted["potential_type"] == "lj"


def test_cmd_mlip_settings_print_the_atom_centred_defaults(tmp_path, capsys):
    """The [Settings] block printed n_hidden 512 (the tabular default) for
    an atom-centred ELM, which builds 400 hidden units."""
    # arbitrary labels: only the printed settings are checked; a 3x3x3
    # supercell keeps r_cut 4 A within half the cell width
    rng = np.random.default_rng(0)
    al = fcc_al_dict()
    lat = al["lattice"] * 3
    shifts = np.array([[i, j, k] for i in range(3) for j in range(3) for k in range(3)])
    pos = (al["positions"][None] + (shifts @ al["lattice"])[:, None]).reshape(-1, 3)
    types = al["atom_types"] * 27
    rows = ["target,label,weight,struct_idx"]
    for sid in range(8):
        sdir = tmp_path / "ts" / str(sid)
        os.makedirs(sdir, exist_ok=True)
        write_poscar(str(sdir / "POSCAR"), lat, pos + 0.05 * rng.normal(size=pos.shape), types)
        rows.append(f"{-3.0 + 0.01 * rng.normal()},E,1.0,{sid}")
        rows += [f"{f},F,1.0,{sid}" for f in 0.1 * rng.normal(size=3 * len(types))]
        rows += [f"{v},S,1.0,{sid}" for v in 1e-3 * rng.normal(size=6)]
    (tmp_path / "ts" / "structures.csv").write_text("\n".join(rows))
    cfg = {
        "structure": {"training_csv": str(tmp_path / "ts" / "structures.csv")},
        "potential": {"species": ["Al"]},
        "mlip": {"output_dir": str(tmp_path / "out"), "form": "atom_centred",
                 "model": "elm", "val_fraction": 0.25,
                 "descriptor": {"type": "acsf", "r_cut": 4.0,
                                "g2": {"eta": [4.0], "rs": [2.5, 3.0]}}},
    }
    yaml_path = tmp_path / "mlip.yaml"
    _write_yaml(yaml_path, cfg)
    assert cmd_mlip(_ns(yaml_path)) == 0
    out = capsys.readouterr().out
    assert "n_hidden   : 400" in out
    z = np.load(tmp_path / "out" / "atomic_elm_model.npz")
    assert '"n_hidden": 400' in str(z["hyper"])
