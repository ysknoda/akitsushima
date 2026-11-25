"""Invariance checks (ADR-0011 D4): a rigid rotation of the whole system rotates the
permittivity tensor (raw and mode estimators; fsum scales Cartesian components
separately and is not rotation-covariant by construction), and the atom order in
the XDATCAR does not matter."""

import numpy as np
import pytest

from akitsushima import mdspec as nozuchi
from akitsushima.mdspec import config
from akitsushima.mdspec import trajectory as tj

from .conftest import write_xdatcar
from .test_modes import _cfg as _mode_cfg
from .test_modes import _harmonic_crystal


def _rotation(seed=7):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q *= np.sign(np.diag(r))[None]
    return q if np.linalg.det(q) > 0 else -q


def _quiet():
    return lambda *a: None


def _pc_cfg(d, estimator):
    return config.validate(
        {
            "trajectory": {"files": ["XDATCAR"], "timestep_fs": 1.0, "frame_every": 4},
            "temperature": 300.0,
            "dipole": {"model": "point_charge", "charges": {"Mg": 2.0, "O": -2.0}},
            "epsilon_inf": 1.0,
            "spectrum": {"max_lag_ps": 2.0, "estimator": estimator, "kk_check": False},
            "output": {"dir": "out", "prefix": estimator, "plot": False, "dipole_cache": False},
        },
        d,
    )


def _pc_traj(d, crystal_traj, lattice_rows, order=None, species=("Mg", "O")):
    pos, lattice, symbols, *_ = crystal_traj
    frac = (pos[::4] @ np.linalg.inv(lattice)) % 1.0
    n_mg = symbols.count("Mg")
    idx = np.arange(len(symbols)) if order is None else np.asarray(order)
    counts = [n_mg, len(symbols) - n_mg]
    if species[0] == "O":
        counts = counts[::-1]
    d.mkdir(exist_ok=True)
    write_xdatcar(d / "XDATCAR", list(species), counts, frac[:, idx], lattice_rows)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_rotation_raw_estimator(tmp_path, crystal_traj):
    _, lattice, *_ = crystal_traj
    rot = _rotation()
    _pc_traj(tmp_path / "a", crystal_traj, lattice)
    _pc_traj(tmp_path / "b", crystal_traj, lattice @ rot.T)
    ra = nozuchi.run(_pc_cfg(tmp_path / "a", "raw"), log=_quiet())
    rb = nozuchi.run(_pc_cfg(tmp_path / "b", "raw"), log=_quiet())
    ea, eb = np.asarray(ra["eps_static"]), np.asarray(rb["eps_static"])
    d_a, d_b = ea - np.eye(3), eb - np.eye(3)
    # lattice rows are written with 6 decimals: relative rounding ~1e-6
    assert np.allclose(d_b, rot @ d_a @ rot.T, atol=1e-5 * np.abs(d_a).max())
    assert not np.allclose(d_b, d_a, atol=1e-3 * np.abs(d_a).max())  # the rotation is not trivial


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("estimator", ["raw", "fsum"])
def test_atom_order_point_charges(tmp_path, crystal_traj, estimator):
    _, lattice, symbols, *_ = crystal_traj
    n_mg, n = symbols.count("Mg"), len(symbols)
    rng = np.random.default_rng(3)
    perm = np.concatenate([rng.permutation(n_mg), n_mg + rng.permutation(n - n_mg)])
    swapped = np.concatenate([np.arange(n_mg, n), np.arange(n_mg)])  # O block first
    _pc_traj(tmp_path / "a", crystal_traj, lattice)
    _pc_traj(tmp_path / "b", crystal_traj, lattice, order=perm)
    _pc_traj(tmp_path / "c", crystal_traj, lattice, order=swapped, species=("O", "Mg"))
    res = [nozuchi.run(_pc_cfg(tmp_path / x, estimator), log=_quiet()) for x in "abc"]
    ref = np.asarray(res[0]["eps_static"])
    for r in res[1:]:
        assert np.allclose(r["eps_static"], ref, rtol=1e-9, atol=1e-12)


@pytest.fixture(scope="module")
def harmonic(tmp_path_factory):
    d = tmp_path_factory.mktemp("inv_modes")
    _harmonic_crystal(d, n_frames=8000)
    h = tj.read_header(d / "XDATCAR")
    chunk = tj.read_frames(d / "XDATCAR", h, max_frames=10**6)
    return d, h, chunk.frac


def _copy_modes(src, dst, rot=None, order=None):
    """Rewrite the harmonic crystal rotated by rot and / or with the atoms reordered."""
    d, h, frac = src
    dst.mkdir()
    rot = np.eye(3) if rot is None else rot
    lat = h.lattice @ rot.T
    idx = np.arange(h.n_atoms) if order is None else np.asarray(order)
    write_xdatcar(dst / "XDATCAR", h.species, h.counts, frac[:, idx], lat, comment="akitsushima x")
    lines = (d / "POSCAR_unit").read_text().splitlines()
    ulat = np.array([[float(x) for x in ln.split()] for ln in lines[2:5]]) @ rot.T
    lines[2:5] = [" ".join(f"{x:.12f}" for x in row) for row in ulat]
    (dst / "POSCAR_unit").write_text("\n".join(lines) + "\n")
    m = np.load(d / "modes.npz")
    np.savez(
        dst / "modes.npz",
        frequencies_THz=m["frequencies_THz"],
        eigenvectors=m["eigenvectors"] @ rot.T,
    )


def test_rotation_and_order_mode_estimator(tmp_path, harmonic):
    rot = _rotation(11)
    _copy_modes(harmonic, tmp_path / "a")
    _copy_modes(harmonic, tmp_path / "b", rot=rot)
    _copy_modes(harmonic, tmp_path / "c", order=[1, 0, 5, 3, 2, 4])
    res = {}
    for x in "abc":
        res[x] = nozuchi.run(_mode_cfg(tmp_path / x, "mode"), log=_quiet())
        assert res[x]["estimator"] == "mode"
    d_a = np.asarray(res["a"]["eps_static"]) - np.eye(3)
    d_b = np.asarray(res["b"]["eps_static"]) - np.eye(3)
    d_c = np.asarray(res["c"]["eps_static"]) - np.eye(3)
    assert np.allclose(d_b, rot @ d_a @ rot.T, atol=1e-5 * np.abs(d_a).max())
    assert np.allclose(d_c, d_a, rtol=1e-9, atol=1e-12)
