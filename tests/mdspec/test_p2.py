"""mdspec P2 (ADR-0052): extxyz / LAMMPS-dump trajectories, velocities in
the md trajectory, forces from the trajectory, dipole.model auto and the
Gamma modes computed with the potential."""

import argparse
import contextlib
import io
import json
import re

import numpy as np
import pytest
import yaml
from akitsushima.io.trajectory import TrajectoryWriter
from akitsushima.mdspec import config, prepare
from akitsushima.mdspec import trajectory as T
from akitsushima.mdspec.dipole import _check_orientation

from .conftest import langevin_crystal, write_xdatcar

SHEAR = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.2, 0.1, 0.0]])


@pytest.fixture(scope="module")
def three_formats(tmp_path_factory):
    """The same 300 frames (sheared Mg / O cell) as XDATCAR, extxyz and a
    LAMMPS dump, with random forces and velocities."""
    d = tmp_path_factory.mktemp("fmt")
    pos, lat, sym, *_ = langevin_crystal(n_steps=300)
    lat = lat + SHEAR
    frac = (pos @ np.linalg.inv(np.eye(3) * (lat[0, 0]))) % 1.0
    cart = frac @ lat
    n_mg = sym.count("Mg")
    write_xdatcar(d / "XDATCAR", ["Mg", "O"], [n_mg, len(sym) - n_mg], frac, lat)
    rng = np.random.default_rng(1)
    F, V = rng.normal(size=cart.shape), 0.01 * rng.normal(size=cart.shape)
    for fmt, name in (("extxyz", "t.extxyz"), ("lammps-dump", "t.lammpstrj")):
        with open(d / name, "w") as fh:
            w = TrajectoryWriter(fh, fmt)
            for k in range(len(cart)):
                w.frame(k, lat, cart[k], sym, energy=0.0, forces=F[k], velocities=V[k])
    return d, frac, lat, F, V, sym


def test_readers_agree_with_xdatcar(three_formats):
    d, frac, lat, F, V, sym = three_formats
    hx = T.read_header(d / "XDATCAR")
    fx = np.concatenate([c.frac for c in T.iter_chunks(d / "XDATCAR", hx)])
    for name, fmt in (("t.extxyz", "extxyz"), ("t.lammpstrj", "lammps-dump")):
        h = T.read_header(d / name)
        assert (h.fmt, h.species, h.counts) == (fmt, ["Mg", "O"], [32, 32])
        assert not h.variable
        chunks = list(T.iter_chunks(d / name, h, chunk_bytes=200_000))
        assert len(chunks) > 1  # several blocks
        f = np.concatenate([c.frac for c in chunks])
        df = f - fx
        df -= np.round(df)
        assert np.abs(df).max() < 1e-7  # 8 written decimals
        L = chunks[0].lattice[0]
        assert np.allclose(
            L @ L.T, lat @ lat.T, atol=1e-7
        )  # LAMMPS: rotated, same metric
        assert T.count_frames(d / name, h) == len(fx) == 300
        Fs = np.concatenate(list(T.iter_forces(d / name, h)))
        assert np.allclose(
            np.linalg.norm(Fs, axis=2), np.linalg.norm(F, axis=2), atol=1e-7
        )
        blk = next(T._iter_text_frames(d / name, fmt, h.n_atoms, 300, None))
        assert np.allclose(
            np.linalg.norm(blk["velocities"], axis=2),
            np.linalg.norm(V, axis=2),
            atol=1e-7,
        )  # A/ps -> A/fs
        assert len(
            list(T.iter_chunks(d / name, h, max_frames=7))
        )  # max_frames honoured
        assert sum(len(c.frac) for c in T.iter_chunks(d / name, h, max_frames=7)) == 7


def test_lammps_dump_velocity_units(three_formats):
    d, *_ = three_formats
    line = [ln for ln in open(d / "t.lammpstrj") if ln.startswith("ITEM: ATOMS")][0]
    assert line.split()[2:] == [
        "id",
        "type",
        "element",
        "x",
        "y",
        "z",
        "fx",
        "fy",
        "fz",
        "vx",
        "vy",
        "vz",
    ]
    head = open(d / "t.extxyz").readlines()[1]
    assert "velocities:R:3" in head


def test_interleaved_species_and_incomplete_frame(tmp_path):
    lat = np.eye(3) * 5.0
    with open(tmp_path / "a.extxyz", "w") as fh:
        w = TrajectoryWriter(fh, "extxyz")
        w.frame(0, lat, np.random.default_rng(0).random((3, 3)), ["Mg", "O", "Mg"])
    with pytest.raises(ValueError, match="contiguous"):
        T.read_header(tmp_path / "a.extxyz")
    with open(tmp_path / "b.extxyz", "w") as fh:
        w = TrajectoryWriter(fh, "extxyz")
        for k in range(3):
            w.frame(k, lat, np.full((2, 3), 0.1 * k), ["Mg", "O"])
    text = (tmp_path / "b.extxyz").read_text().splitlines()
    (tmp_path / "b.extxyz").write_text("\n".join(text[:-1]) + "\n")
    h = T.read_header(tmp_path / "b.extxyz")
    with pytest.warns(UserWarning, match="incomplete"):
        n = sum(len(c.frac) for c in T.iter_chunks(tmp_path / "b.extxyz", h))
    assert n == 2


def test_xdatcar_refuses_velocities(tmp_path):
    with open(tmp_path / "X", "w") as fh:
        w = TrajectoryWriter(fh, "xdatcar")
        with pytest.raises(ValueError, match="XDATCAR"):
            w.frame(0, np.eye(3), np.zeros((1, 3)), ["Si"], velocities=np.zeros((1, 3)))


def test_lammps_orientation_guard(tmp_path):
    from akitsushima.io.poscar import write_poscar

    write_poscar(str(tmp_path / "P"), np.eye(3) * 4.0 + SHEAR, np.zeros((1, 3)), ["Si"])
    h = T.Header("x", np.eye(3), ["Si"], [1], 0, False, "lammps-dump")
    rot = np.array([[4.0, 0, 0], [0.3, 4.0, 0], [0.2, 0.1, 4.0]])  # same as given: ok
    _check_orientation(h, rot, str(tmp_path / "P"), [1, 1, 1], "modes")
    turned = (np.eye(3) * 4.0 + SHEAR) @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
    with pytest.raises(ValueError, match="LAMMPS orientation"):
        _check_orientation(h, turned, str(tmp_path / "P"), [1, 1, 1], "modes")
    h.fmt = "extxyz"  # other formats keep their own orientation: no check
    _check_orientation(h, turned, str(tmp_path / "P"), [1, 1, 1], "modes")


def _write_cfg(d, files, extra=None):
    cfg = {
        "born": {"epsilon_inf": 1.0},
        "mdspec": {
            "trajectory": {"files": files, "timestep_fs": 1.0, "frame_every": 1},
            "temperature": 300.0,
            "dipole": {"model": "point_charge", "charges": {"Mg": 2.0, "O": -2.0}},
            "spectrum": {"max_lag_ps": 0.1, "freq_max": 30},
            "output": {
                "dir": "out_" + files[0].replace(".", "_"),
                "plot": False,
                "dipole_cache": False,
            },
        },
    }
    for k, v in (extra or {}).items():
        cfg["mdspec"][k] = v
    (d / "in.yaml").write_text(yaml.safe_dump(cfg))
    return d / "in.yaml"


def test_mdspec_same_result_from_three_formats(three_formats):
    from akitsushima import mdspec

    d, *_ = three_formats
    res = {}
    for name in ("XDATCAR", "t.extxyz", "t.lammpstrj"):
        with warnings_ignored():
            cfg = prepare.load(_write_cfg(d, [name]), log=lambda *_: None)
            res[name] = mdspec.run(cfg, log=lambda *_: None)
    a, b, c = (
        np.array(res[k]["eps_static"]) for k in ("XDATCAR", "t.extxyz", "t.lammpstrj")
    )
    # 8 written decimals (Cartesian in extxyz, fractional in the XDATCAR)
    # differ by ~1e-8 A; the fluctuation estimator carries it to ~1e-6.
    assert np.allclose(a, b, rtol=1e-5)
    # LAMMPS: the tensor is in the rotated frame; its invariants agree
    assert np.trace(c) == pytest.approx(np.trace(a), rel=1e-5)
    assert np.allclose(np.linalg.eigvalsh(c), np.linalg.eigvalsh(a), rtol=1e-5)


@contextlib.contextmanager
def warnings_ignored():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _resolve(cfg):
    from akitsushima.mdspec.yaml_io import to_internal

    raw = to_internal(cfg)
    log = []
    prepare._resolve_model(cfg, raw, prepare._potential(cfg), log.append)
    return raw["dipole"], log


def test_model_auto_resolution(tmp_path):
    """born.born_charges > the potential's point charges > dipole.charges."""
    from test_dielectric import MGO_JSON

    (tmp_path / "mgo.json").write_text(json.dumps(MGO_JSON))
    base = {"mdspec": {"trajectory": {"files": ["X"], "timestep_fs": 1, "frame_every": 1},
                       "temperature": 300, "modes": "off"}}
    c = json.loads(json.dumps(base))
    c["mdspec"]["dipole"] = {"charges": {"Mg": 1.0, "O": -1.0}}
    with pytest.warns(UserWarning, match="fixed charges"):
        dp, log = _resolve(c)
    assert dp["model"] == "point_charge" and dp["charges"] == {"Mg": 1.0, "O": -1.0}
    c["potential"] = {"type": "Buckingham", "potential_json": str(tmp_path / "mgo.json"),
                      "r_cut": 5.0}
    with pytest.warns(UserWarning, match="ignored"):
        dp, log = _resolve(c)
    assert dp["model"] == "point_charge" and dp["charges"] == {"Mg": 2.0, "O": -2.0}
    c["born"] = {"born_charges": {"per_species": {"Mg": 1.9, "O": -1.9}}}
    with warnings_ignored():
        dp, log = _resolve(c)
    assert dp["model"] == "born_tensor" and "born.born_charges" in log[0]
    with pytest.raises(config.ConfigError, match="found no charges"):
        _resolve(base)


def test_forces_trajectory_translation():
    from akitsushima.mdspec.yaml_io import to_internal

    raw = to_internal(
        {
            "mdspec": {
                "trajectory": {"files": ["a.extxyz", "b.extxyz"]},
                "forces": "trajectory",
            }
        }
    )
    assert raw["forces"] == {"files": ["a.extxyz", "b.extxyz"]}


# ---------------------------------------------------------------- Gamma modes


def _aln_md(d, n_steps):
    """Tersoff AlN (test_phonon_regressions._ALN), 3x3x2 wurtzite, Langevin
    30 K, extxyz with forces and velocities."""
    from akitsushima.cli.cmd_md import cmd_md
    from akitsushima.io.poscar import write_poscar
    from test_phonon_regressions import _ALN

    a, c, u = 3.381244, 5.521549, 0.375
    lat = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    frac = np.array(
        [
            [1 / 3, 2 / 3, 0],
            [2 / 3, 1 / 3, 0.5],
            [1 / 3, 2 / 3, u],
            [2 / 3, 1 / 3, 0.5 + u],
        ]
    )
    types = ["Al", "Al", "N", "N"]
    write_poscar(str(d / "POSCAR_unit"), lat, frac @ lat, types)
    S = np.array([3, 3, 2])
    cells = np.array([[i, j, k] for i in range(3) for j in range(3) for k in range(2)])
    fsc = ((frac[None] + cells[:, None]) / S).reshape(-1, 3)
    tsc = types * len(cells)
    order = np.argsort([0 if t == "Al" else 1 for t in tsc], kind="stable")
    latS = lat * S[:, None]
    write_poscar(str(d / "POSCAR"), latS, (fsc @ latS)[order], [tsc[i] for i in order])
    pot = {"type": "Tersoff", "species": ["Al", "N"], "params": _ALN}
    md = {
        "structure": {"poscar": str(d / "POSCAR")},
        "potential": pot,
        "md": {
            "ensemble": "nvt",
            "thermostat": "langevin",
            "damping": 200.0,
            "output_dir": str(d / "md"),
            "temperature": 30.0,
            "timestep": 1.0,
            "n_steps": n_steps,
            "equil_steps": 1000,
            "velocity_rescale_steps": 0,
            "traj_every": 2,
            "traj_format": "extxyz",
            "traj_velocities": True,
            "thermo_every": 2,
            "seed": 7,
            "verbose": False,
            "timing": False,
            "checkpoint_every": 0,
            "restart": False,
        },
    }
    (d / "md.yaml").write_text(yaml.safe_dump(md))
    with contextlib.redirect_stdout(io.StringIO()):
        assert cmd_md(argparse.Namespace(yaml=str(d / "md.yaml"), quiet=True)) == 0
    return pot


def _harmonic_eps(unit_path):
    from akitsushima.io.poscar import parse_poscar
    from akitsushima.solvers.config_bridge import _build_force_fn_from_yaml
    from akitsushima.solvers.dielectric import dielectric_response
    from akitsushima.solvers.phonon import PhononCalculator
    from test_phonon_regressions import _ALN

    fn = _build_force_fn_from_yaml(
        {"type": "Tersoff", "species": ["Al", "N"], "params": _ALN}
    )[0]
    u = parse_poscar(str(unit_path))
    pc = PhononCalculator(
        u["lattice"], u["positions"], u["atom_types"], fn, supercell=[4, 4, 3]
    )
    with contextlib.redirect_stdout(io.StringIO()):
        pc.compute_ifc(verbose=False)
    Z = np.array([np.eye(3) if t == "Al" else -np.eye(3) for t in u["atom_types"]])
    return dielectric_response(
        pc.build_dynamical_matrix(np.zeros(3)),
        pc.masses,
        Z,
        np.eye(3),
        abs(np.linalg.det(u["lattice"])),
    )["epsilon_0"]


@pytest.mark.slow
def test_gamma_modes_from_the_potential(tmp_path):
    """AlN with charges +-1: the Gamma modes computed with the MD's own
    potential make the mode estimator and the force-based value agree with
    the harmonic eps_0 of the same model; unit-cell tiling and the MD cell
    itself give the same modes' answer; above the size limit the mode
    estimator is not offered."""
    from akitsushima import mdspec

    pot = _aln_md(tmp_path, 6000)
    traj = str(tmp_path / "md" / "traj_nvt.extxyz")
    out = {}
    for tag, modes in (
        ("unit", {"unit_cell": "POSCAR_unit", "supercell": [3, 3, 2]}),
        ("cell", None),
    ):
        cfg = {
            "potential": pot,
            "born": {"epsilon_inf": 1.0},
            "mdspec": {
                "trajectory": {"files": [traj], "timestep_fs": 1.0, "frame_every": 2},
                "temperature": "auto",
                "temperature_source": [str(tmp_path / "md" / "thermo.log")],
                "dipole": {"charges": {"Al": 1.0, "N": -1.0}},
                "forces": "trajectory",
                "spectrum": {"max_lag_ps": 1.0, "freq_max": 30},
                "output": {"dir": str(tmp_path / tag), "prefix": tag, "plot": False},
            },
        }
        if modes:
            cfg["mdspec"]["modes"] = modes
        (tmp_path / f"{tag}.yaml").write_text(yaml.safe_dump(cfg))
        with warnings_ignored():
            c = prepare.load(tmp_path / f"{tag}.yaml", log=lambda *_: None)
            out[tag] = mdspec.run(c, log=lambda *_: None)
        assert out[tag]["estimator"] == "mode"
    ref = np.diag(_harmonic_eps(tmp_path / "unit" / "unit_gamma_unit.vasp")) - 1.0
    for tag in ("unit", "cell"):
        mode = np.diag(np.array(out[tag]["eps_static"])) - 1.0
        force = np.diag(np.array(out[tag]["eps_static_force"])) - 1.0
        assert np.allclose(force, ref, rtol=0.02)  # harmonic at 30 K, exact estimator
        assert np.allclose(mode, ref, rtol=0.05)  # statistics of a 12-ps run
    # the same modes either way: diagonals to 3e-5, off-diagonals (noise ~1e-4) absolute
    assert np.allclose(
        out["unit"]["eps_static_force"], out["cell"]["eps_static_force"], rtol=0, atol=2e-4
    )
    # above the size limit: no in-house modes, fsum
    cfg = yaml.safe_load((tmp_path / "cell.yaml").read_text())
    cfg["mdspec"].pop("forces")
    (tmp_path / "big.yaml").write_text(yaml.safe_dump(cfg))
    import akitsushima.mdspec.prepare as P

    old = P.DIRECT_MODES_MAX_ATOMS
    P.DIRECT_MODES_MAX_ATOMS = 10
    try:
        with pytest.warns(UserWarning, match="mode estimator unavailable"):
            c = prepare.load(tmp_path / "big.yaml", log=lambda *_: None)
    finally:
        P.DIRECT_MODES_MAX_ATOMS = old
    assert c["modes"] is None


@pytest.mark.slow
def test_md_writes_velocities_consistent_with_thermo(tmp_path):
    _aln_md(tmp_path, 40)
    traj = (tmp_path / "md" / "traj_nvt.extxyz").read_text().splitlines()
    from akitsushima.constants import ATOMIC_MASS

    h = T.read_header(tmp_path / "md" / "traj_nvt.extxyz")
    blk = next(
        T._iter_text_frames(
            tmp_path / "md" / "traj_nvt.extxyz", "extxyz", h.n_atoms, 5, None
        )
    )
    m = np.array([ATOMIC_MASS[s] for s in h.symbols])
    ke = 0.5 * np.einsum("i,kij->k", m, blk["velocities"] ** 2) * 103.642697  # eV
    steps = [
        int(re.search(r"step=(\d+)", ln).group(1)) for ln in traj if "Lattice=" in ln
    ][:5]
    th = {
        int(r.split()[0]): float(r.split()[3])
        for r in (tmp_path / "md" / "thermo.log").read_text().splitlines()[1:]
        if r.split() and r.split()[0].isdigit()
    }
    for s, k in zip(steps, ke):
        assert k == pytest.approx(th[s], abs=2e-6)


def test_md_refuses_velocities_in_xdatcar(tmp_path):
    from akitsushima.cli.cmd_md import cmd_md
    from akitsushima.cli.templates import _DUMP_MAP
    from test_cli_templates import _write_si

    cfg = yaml.safe_load(_DUMP_MAP["nvt"])
    _write_si(tmp_path / "POSCAR")
    cfg["structure"]["poscar"] = str(tmp_path / "POSCAR")
    cfg["md"].update(
        traj_velocities=True,
        traj_format="xdatcar",
        n_steps=2,
        output_dir=str(tmp_path / "o"),
    )
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(cfg))
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rc = cmd_md(argparse.Namespace(yaml=str(tmp_path / "m.yaml"), quiet=True))
    assert rc == 1 and "traj_velocities" in out.getvalue()
