"""Total dipole M(t), runs, the dipole cache and the external model (ADR-0002, ADR-0004)."""

import numpy as np
import pytest

from akitsushima.mdspec import config
from akitsushima.mdspec import dipole as dp
from akitsushima.mdspec import trajectory as tj

from .conftest import write_xdatcar

LAT = np.diag([5.0, 6.0, 7.0])


def _traj(path, n_frames=50, seed=0, start=None, step=0.02):
    """Two Mg and two O atoms on a random walk; returns the unwrapped fractional coordinates."""
    rng = np.random.default_rng(seed)
    first = rng.random((1, 4, 3)) if start is None else start[None]
    uw = first + np.cumsum(rng.normal(scale=step, size=(n_frames, 4, 3)), axis=0)
    uw = np.round(uw, 8)
    write_xdatcar(path, ["Mg", "O"], [2, 2], uw % 1.0, LAT)
    return uw


def _cfg(tmp_path, files, **over):
    raw = {
        "trajectory": {"files": files, "timestep_fs": 1.0, "frame_every": 1},
        "temperature": 300.0,
        "dipole": {"model": "point_charge", "charges": {"Mg": 2.0, "O": -2.0}},
        "epsilon_inf": 1.0,
        "output": {"dir": "out", "prefix": "t", "plot": False},
    }
    for sec, val in over.items():
        raw[sec] = {**raw.get(sec, {}), **val} if isinstance(val, dict) else val
    return config.validate(raw, tmp_path)


Q = np.array([2.0, 2.0, -2.0, -2.0])


def test_scalar_and_tensor_paths_agree():
    rng = np.random.default_rng(1)
    uw = rng.random((7, 4, 3))
    lat = np.broadcast_to(LAT, (7, 3, 3))
    z = Q[:, None, None] * np.eye(3)[None]
    assert np.allclose(dp.dipole_from_chunk(uw, lat, z, Q), dp.dipole_from_chunk(uw, lat, z, None))


def test_concatenate_and_independent_runs(tmp_path):
    uw1 = _traj(tmp_path / "X1", seed=1)
    _traj(tmp_path / "X2", seed=2, start=uw1[-1])
    files = ["X1", "X2"]
    runs, _z, header = dp.compute_runs(_cfg(tmp_path, files), log=lambda *a: None)
    assert len(runs) == 1 and len(runs[0].dipole) == 100 and header.n_atoms == 4
    m = runs[0].dipole
    assert np.allclose(m[:50] - m[0], ((uw1 - uw1[0]) @ LAT * Q[None, :, None]).sum(axis=1))
    # unwrapping continues across files: no jump between the two halves
    assert np.abs(m[50] - m[49]).max() < 1.0
    ind = _cfg(tmp_path, files, trajectory={"mode": "independent"}, output={"prefix": "i"})
    runs_i, _, _ = dp.compute_runs(ind, log=lambda *a: None)
    assert [len(r.dipole) for r in runs_i] == [50, 50]
    assert runs_i[0].files == [str((tmp_path / "X1").resolve())]


def test_skip_and_stride(tmp_path):
    _traj(tmp_path / "X1")
    full, _, _ = dp.compute_runs(_cfg(tmp_path, ["X1"]), log=lambda *a: None)
    cfg = _cfg(
        tmp_path, ["X1"], trajectory={"skip_frames": 10, "stride": 2}, output={"prefix": "s"}
    )
    with pytest.warns(UserWarning, match="aliasing"):
        sub, _, _ = dp.compute_runs(cfg, log=lambda *a: None)
    assert np.allclose(
        sub[0].dipole - sub[0].dipole[0], (full[0].dipole[10::2] - full[0].dipole[10])
    )
    assert sub[0].dt_fs == 2.0


def test_dipole_cache_hit_and_invalidation(tmp_path):
    _traj(tmp_path / "X1")
    cfg = _cfg(tmp_path, ["X1"])
    msgs = []
    first, _, _ = dp.compute_runs(cfg, log=msgs.append)
    assert (tmp_path / "out" / "t_dipole_cache.npz").exists()
    again, _, _ = dp.compute_runs(cfg, log=msgs.append)
    assert any("cache hit" in m for m in msgs)
    assert np.array_equal(first[0].dipole, again[0].dipole)
    # a changed trajectory file invalidates the cache
    _traj(tmp_path / "X1", seed=5)
    msgs.clear()
    new, _, _ = dp.compute_runs(cfg, log=msgs.append)
    assert not any("cache hit" in m for m in msgs)
    assert not np.allclose(new[0].dipole, first[0].dipole)


def test_external_dipole_units(tmp_path):
    _traj(tmp_path / "X1", n_frames=20)
    m = np.random.default_rng(3).normal(size=(20, 3))
    np.savetxt(tmp_path / "m.dat", np.column_stack([np.arange(20), m]))
    raw_d = {"model": "external", "external_file": "m.dat", "external_unit": "Debye"}
    cfg = _cfg(tmp_path, ["X1"], dipole=raw_d)
    runs, z, _ = dp.compute_runs(cfg, log=lambda *a: None)
    assert np.allclose(runs[0].dipole, m * dp.DEBYE_TO_E_ANG)
    assert np.allclose(runs[0].volume, abs(np.linalg.det(LAT)))
    assert not z.any()


# --------------------------------------------------------------------------
# A1: displacement form with a variable cell (ADR-0011)
# --------------------------------------------------------------------------

WURTZITE = np.array(
    [[1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, 0.382], [2 / 3, 1 / 3, 0.882]]
)


def _npt_lattices(n, seed=0, strain=1e-3):
    a, c = 3.25, 5.21
    lat0 = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    eps = strain * np.random.default_rng(seed).normal(size=(n, 3, 3))
    eps = 0.5 * (eps + np.transpose(eps, (0, 2, 1)))
    return np.einsum("ij,tjk->tik", lat0, np.eye(3)[None] + eps)


def _zno_cfg(tmp_path, d, **over):
    raw = {
        "trajectory": {"files": [str(d / "XDATCAR")], "timestep_fs": 1.0, "frame_every": 1},
        "temperature": 300.0,
        "dipole": {"model": "point_charge", "charges": {"Zn": 2.0, "O": -2.0}},
        "epsilon_inf": 1.0,
        "output": {"dir": str(d / "out"), "plot": False, "dipole_cache": False},
    }
    raw.update(over)
    return config.validate(raw, tmp_path)


def _run_npt(tmp_path, name, frac, lats):
    d = tmp_path / name
    d.mkdir()
    write_xdatcar(d / "XDATCAR", ["Zn", "O"], [2, 2], frac, lats, variable=True)
    with pytest.warns(UserWarning, match="variable cell"):
        runs, _, _ = dp.compute_runs(_zno_cfg(tmp_path, d), log=lambda *a: None)
    assert runs[0].info["variable_cell"]
    return runs[0].dipole


def test_frozen_fractional_coordinates_give_constant_dipole(tmp_path):
    n = 400
    lats = _npt_lattices(n)
    frac = np.repeat(WURTZITE[None], n, axis=0)
    m = _run_npt(tmp_path, "a", frac, lats)
    # absolute form sum q f h(t) would fluctuate by |sum q f_z| c 1e-3 ~ 8e-3 e A (ADR-0011)
    assert np.abs(m).max() < 1e-9
    shifted = frac.copy()
    shifted[:, 2, 2] += 1.0 - 1e-9  # same crystal, other periodic image of one O
    shifted = shifted % 1.0
    assert np.abs(_run_npt(tmp_path, "b", shifted, lats)).max() < 1e-6


def test_vibrating_atoms_in_fluctuating_cell(tmp_path):
    n = 600
    rng = np.random.default_rng(2)
    lats = _npt_lattices(n, seed=3)
    df = 0.01 * rng.normal(size=(n, 4, 3))
    frac = WURTZITE[None] + df
    m = _run_npt(tmp_path, "a", frac % 1.0, lats)
    q = np.array([2.0, 2.0, -2.0, -2.0])
    fr = np.round(frac, 8)  # as written to the XDATCAR
    expect = np.einsum("k,tkb,tbc->tc", q, fr - fr.mean(axis=0), np.round(lats, 6))
    assert np.allclose(m, expect, atol=1e-9)
    # the image of an atom does not matter
    alt = frac.copy()
    alt[:, 1] += [1.0, -1.0, 0.0]
    m2 = _run_npt(tmp_path, "b", alt % 1.0, lats)
    assert np.allclose(m2, m, atol=1e-6)


def test_variable_cell_refuses_forces(tmp_path):
    from .test_modes import _harmonic_crystal

    _harmonic_crystal(tmp_path, n_frames=200)
    hd = tj.read_header(tmp_path / "XDATCAR")
    fr = tj.read_frames(tmp_path / "XDATCAR", hd, max_frames=200).frac
    lats = np.array([hd.lattice * (1 + 1e-3 * np.sin(k)) for k in range(200)])
    write_xdatcar(tmp_path / "XDATCAR", hd.species, hd.counts, fr, lats, variable=True)
    cfg = config.validate(
        {
            "trajectory": {"files": ["XDATCAR"], "timestep_fs": 1.0, "frame_every": 1},
            "temperature": 300.0,
            "dipole": {"model": "point_charge", "charges": {"Mg": 2.0, "O": -1.0}},
            "epsilon_inf": 1.0,
            "modes": {"file": "modes.npz", "unit_cell": "POSCAR_unit", "supercell": [2, 1, 1]},
            "forces": {"files": ["forces.npy"]},
            "output": {"dir": "out", "plot": False},
        },
        tmp_path,
    )
    with pytest.raises(ValueError, match="variable cell"):
        dp.compute_runs(cfg, log=lambda *a: None)


def test_check_com_motion_npt_frozen(tmp_path):
    from akitsushima.mdspec import check as chk

    n = 300
    d = tmp_path / "a"
    d.mkdir()
    write_xdatcar(
        d / "XDATCAR", ["Zn", "O"], [2, 2], np.repeat(WURTZITE[None], n, 0), _npt_lattices(n), True
    )
    cfg = _zno_cfg(tmp_path, d)
    hd = tj.read_header(d / "XDATCAR")
    assert chk._com_motion(cfg, hd, np.array([65.38, 65.38, 16.0, 16.0])) < 1e-9


def test_count_frames_matches_reader(tmp_path):
    _traj(tmp_path / "X1", n_frames=37)
    h = tj.read_header(tmp_path / "X1")
    assert tj.count_frames(tmp_path / "X1", h, chunk_bytes=113) == 37
    text = (tmp_path / "X1").read_text().splitlines(keepends=True)
    (tmp_path / "X2").write_text("".join(text[:-2]))  # last frame incomplete
    assert tj.count_frames(tmp_path / "X2", h, chunk_bytes=97) == 36
    lats = _npt_lattices(9)
    write_xdatcar(tmp_path / "X3", ["Zn", "O"], [2, 2], np.repeat(WURTZITE[None], 9, 0), lats, True)
    assert tj.count_frames(tmp_path / "X3", tj.read_header(tmp_path / "X3"), chunk_bytes=50) == 9


def test_external_frame_count_checked(tmp_path):
    _traj(tmp_path / "X1", n_frames=20)
    np.savetxt(tmp_path / "m.dat", np.zeros((19, 3)))
    cfg = _cfg(tmp_path, ["X1"], dipole={"model": "external", "external_file": "m.dat"})
    with pytest.raises(ValueError, match="19 dipole rows for 20 trajectory frames"):
        dp.compute_runs(cfg, log=lambda *a: None)


def test_cache_ignores_skip_and_stride(tmp_path):
    _traj(tmp_path / "X1")
    msgs = []
    dp.compute_runs(_cfg(tmp_path, ["X1"]), log=msgs.append)
    cfg = _cfg(tmp_path, ["X1"], trajectory={"skip_frames": 5})
    msgs.clear()
    runs, _, _ = dp.compute_runs(cfg, log=msgs.append)
    assert any("cache hit" in m for m in msgs) and len(runs[0].dipole) == 45
    assert runs[0].info["file_frames"] == [50]
