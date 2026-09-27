"""External engine (ExternalFileEngine, `potential: {type: external}`;
ADR-0011).

- EFS file round trip and the stress unit factor.
- akitsushima itself as the external code (tools.akitsushima_efs, SW Si):
  E, F, S equal the in-process values to the EFS print precision; forces
  and stress are derivatives of the energy received; a short relax through
  the external engine follows the in-process one.
- Cache (hit, miss past cache_tol, species in the key, copies returned);
  failures (exit status, timeout, atom count) are errors.
- YAML route and start-up rules (electrostatics refused, kappa refused,
  gcmc warns); relax through the CLI.
- ASE EMT as an independent external code (skipped without ASE): values
  equal ASE in process, stress sign by finite differences.
"""

import shlex
import sys

import numpy as np
import pytest
import yaml
from conftest import diamond_si_dict, randomly_perturbed

from akitsushima.fitting.external import ExternalFileEngine
from akitsushima.io.efs import read_efs_file, write_efs_file
from akitsushima.solvers.config_bridge import _build_force_fn_from_yaml

PY = shlex.quote(sys.executable)
SW_SI = {
    "type": "SW",
    "species": ["Si"],
    "params": {
        "epsilon": 2.1683,
        "sigma": 2.0951,
        "A": 7.049556277,
        "B": 0.6022245584,
        "p": 4,
        "q": 0,
        "a": 1.80,
        "lambda": 21.0,
        "gamma": 1.20,
        "costheta0": -1.0 / 3.0,
    },
}


def _si(seed=0, mag=0.05):
    s = randomly_perturbed(diamond_si_dict(2), magnitude=mag, seed=seed)
    return s["lattice"], s["positions"], s["atom_types"]


@pytest.fixture
def self_engine(tmp_path):
    pot = tmp_path / "sw.yaml"
    pot.write_text(yaml.safe_dump({"potential": SW_SI}))
    eng = ExternalFileEngine(
        cmd=f"{PY} -m akitsushima.tools.akitsushima_efs {shlex.quote(str(pot))}",
        work_dir=str(tmp_path / "run"),
        verbose=False,
    )
    return eng, _build_force_fn_from_yaml(SW_SI)[0]


def test_efs_round_trip_and_stress_factor(tmp_path):
    rng = np.random.default_rng(0)
    F = rng.normal(size=(5, 3))
    S = rng.normal(size=(3, 3))
    S = S + S.T
    p = tmp_path / "r.efs"
    write_efs_file(str(p), -12.345, F, S)
    E2, F2, S2 = read_efs_file(str(p), 5)
    assert E2 == pytest.approx(-12.345, abs=5e-10)
    np.testing.assert_allclose(F2, F, atol=1e-10)
    np.testing.assert_allclose(S2, S, atol=1e-10)
    _, _, S3 = read_efs_file(str(p), 5, stress_to_eV_Ang3=0.5)
    np.testing.assert_allclose(S3, 0.5 * S, atol=1e-10)
    with pytest.raises(ValueError):
        read_efs_file(str(p), 6)


def test_self_engine_matches_in_process_and_its_energy_derivatives(self_engine):
    eng, direct = self_engine
    lat, pos, types = _si()
    E, F, S = eng(lat, pos, types)
    E0, F0, S0 = direct(lat, pos, types)
    assert eng.kind == "external" and not eng.per_atom_energy
    # EFS carries 10 decimals: E per atom (x64), F, S
    assert E == pytest.approx(E0, abs=64 * 1e-10)
    np.testing.assert_allclose(F, F0, atol=2e-10)
    np.testing.assert_allclose(S, S0, atol=2e-10)
    # forces / stress are derivatives of the energy received (h sized to
    # the EFS rounding: 6e-9 eV / 2h = 3e-6 eV/A)
    h = 1e-3
    for k, c in [(0, 0), (17, 2)]:
        p = pos.copy()
        p[k, c] += h
        Ep = eng(lat, p, types)[0]
        p[k, c] -= 2 * h
        Em = eng(lat, p, types)[0]
        assert F[k, c] == pytest.approx(-(Ep - Em) / (2 * h), abs=5e-5)
    V = abs(np.linalg.det(lat))
    fr = pos @ np.linalg.inv(lat)
    T = np.eye(3)
    T[0, 0] += h
    Ep = eng(lat @ T, fr @ lat @ T, types)[0]
    T[0, 0] -= 2 * h
    Em = eng(lat @ T, fr @ lat @ T, types)[0]
    assert S[0, 0] == pytest.approx(-(Ep - Em) / (2 * h) / V, abs=1e-5)


def test_cache_hits_misses_species_and_copies(self_engine):
    eng, _ = self_engine
    lat, pos, types = _si()
    E1, F1, _ = eng(lat, pos, types)
    F1[:] = 0.0  # caller mutates its copy
    E2, F2, _ = eng(lat, pos + 1e-8, types)  # within cache_tol
    assert eng.n_calls == 1 and eng.n_cache_hits == 1
    assert E2 == E1 and np.abs(F2).max() > 0.1
    eng(lat, pos + np.r_[1e-4, 0, 0], types)  # past cache_tol
    assert eng.n_calls == 2
    lat_b = lat * 1.001
    eng(lat_b, pos, types)  # other cell
    assert eng.n_calls == 3


def test_failures_are_errors(tmp_path):
    lat, pos, types = _si()
    bad = ExternalFileEngine(cmd="exit 3", work_dir=str(tmp_path), verbose=False)
    with pytest.raises(RuntimeError, match="exit 3"):
        bad(lat, pos, types)
    slow = ExternalFileEngine(
        cmd="sleep 5", work_dir=str(tmp_path), timeout=0.3, verbose=False
    )
    import subprocess

    with pytest.raises(subprocess.TimeoutExpired):
        slow(lat, pos, types)
    write_efs_file(str(tmp_path / "two.efs"), -1.0, np.zeros((2, 3)), np.zeros((3, 3)))
    short = ExternalFileEngine(
        cmd="true", work_dir=str(tmp_path), output_file="two.efs", verbose=False
    )
    with pytest.raises(ValueError):
        short(lat, pos, types)
    missing = ExternalFileEngine(
        cmd="true", work_dir=str(tmp_path), output_file="none.efs", verbose=False
    )
    with pytest.raises(FileNotFoundError):
        missing(lat, pos, types)


def test_relax_through_the_external_engine_follows_in_process(self_engine, tmp_path):
    from akitsushima.solvers import GeometryOptimiser

    eng, direct = self_engine
    lat, pos, types = _si(seed=3, mag=0.08)
    out = []
    for fn, d in ((eng, "ext"), (direct, "dir")):
        r = GeometryOptimiser(
            lat,
            pos,
            types,
            force_fn=fn,
            method="fire",
            max_steps=8,
            fmax=1e-6,
            relax_cell=False,
            verbose=False,
            output_dir=str(tmp_path / d),
        ).run()
        out.append(r)
    np.testing.assert_allclose(out[0]["positions"], out[1]["positions"], atol=1e-7)
    assert out[0]["energy"] == pytest.approx(out[1]["energy"], abs=1e-7)
    assert eng.n_calls >= 8


def test_yaml_route_and_start_up_rules(tmp_path):
    fn, r_cut = _build_force_fn_from_yaml({"type": "external", "cmd": "true"})[:2]
    assert isinstance(fn, ExternalFileEngine) and r_cut == 1.0
    with pytest.raises(ValueError, match="electrostatics"):
        _build_force_fn_from_yaml(
            {"type": "external", "cmd": "true", "coulomb_method": "esm"}
        )
    with pytest.raises(ValueError, match="unknown key"):
        _build_force_fn_from_yaml({"type": "external", "cmd": "true", "cmdd": 1})
    with pytest.raises(ValueError, match="needs potential.cmd"):
        _build_force_fn_from_yaml({"type": "external"})

    from akitsushima.solvers.gcmc import GCMCMDSimulation
    from akitsushima.solvers.thermal_conductivity import (
        ThermalConductivityCalculator,
    )

    lat, pos, types = _si()
    with pytest.raises(ValueError, match="per-atom energies"):
        ThermalConductivityCalculator(lat, pos, types, fn)
    with pytest.warns(UserWarning, match="very slow"):
        GCMCMDSimulation(
            lat,
            pos,
            types,
            force_fn=fn,
            species="Si",
            n_cycles=1,
            verbose=False,
            output_dir=str(tmp_path),
        )


def test_relax_through_the_cli(tmp_path, monkeypatch):
    from akitsushima.cli.cmd_relax import cmd_relax
    from akitsushima.cli.templates import _DUMP_MAP
    from akitsushima.io.poscar import parse_poscar, write_poscar

    monkeypatch.chdir(tmp_path)
    lat, pos, types = _si(seed=5)
    write_poscar("POSCAR", lat, pos, types)
    (tmp_path / "sw.yaml").write_text(yaml.safe_dump({"potential": SW_SI}))
    cfg = yaml.safe_load(_DUMP_MAP["relax"])
    cfg["structure"] = {"poscar": "POSCAR"}
    cfg["potential"] = {
        "type": "external",
        "cmd": f"{sys.executable} -m akitsushima.tools.akitsushima_efs ../sw.yaml",
        "work_dir": "ext_run",
    }
    cfg["relax"].update(max_steps=3, relax_cell=False, output_dir="out", verbose=False)
    (tmp_path / "relax.yaml").write_text(yaml.safe_dump(cfg))

    class A:
        yaml = str(tmp_path / "relax.yaml")
        quiet = True

    assert cmd_relax(A()) == 0
    assert parse_poscar("ext_run/POSCAR")["atom_types"] == types


ase = pytest.importorskip("ase")


def test_ase_emt_as_an_external_code(tmp_path):
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from akitsushima.tools.ase_efs import ase_efs

    at = bulk("Cu", "fcc", a=3.62, cubic=True).repeat((2, 2, 2))
    at.rattle(0.05, seed=2)
    lat, pos, types = (
        at.cell.array.copy(),
        at.positions.copy(),
        at.get_chemical_symbols(),
    )
    eng = ExternalFileEngine(
        cmd=f"{PY} -m akitsushima.tools.ase_efs --calc emt",
        work_dir=str(tmp_path),
        verbose=False,
    )
    E, F, S = eng(lat, pos, types)
    at.calc = EMT()
    E0, F0, S0 = ase_efs(at)
    assert E == pytest.approx(E0, abs=32 * 1e-10)
    np.testing.assert_allclose(F, F0, atol=2e-10)
    np.testing.assert_allclose(S, S0, atol=2e-10)
    # akitsushima's sign: S = -(1/V) dE/d(strain), positive under compression
    V, h = at.get_volume(), 1e-4
    fr = pos @ np.linalg.inv(lat)
    Es = []
    for sgn in (1, -1):
        T = np.eye(3)
        T[1, 1] += sgn * h
        Es.append(eng(lat @ T, fr @ lat @ T, types)[0])
    assert S[1, 1] == pytest.approx(-(Es[0] - Es[1]) / (2 * h) / V, abs=2e-5)
