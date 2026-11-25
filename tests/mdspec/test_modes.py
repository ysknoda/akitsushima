"""Mode-resolved estimator (ADR-0003 item 10): exact for a harmonic crystal with unequal mode energies."""

import numpy as np
import pytest

from akitsushima import mdspec as nozuchi
from akitsushima.mdspec import config
from akitsushima.mdspec import modes as md
from akitsushima.mdspec.constants import AMU, ANGSTROM, ATOMIC_MASS, E_CHARGE, EPS0, KB

from .conftest import write_xdatcar

UNIT = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5]])  # Mg, O, O
SPECIES, CHARGES = ["Mg", "O", "O"], np.array([2.0, -1.0, -1.0])
FREQ = np.array([4.0, 5.5, 7.0, 8.5, 10.0, 12.0])  # optical, THz
ENERGY = np.array([0.3, 2.5, 1.0, 0.5, 1.8, 0.7])  # mode energies in kT: far from equipartition


def _harmonic_crystal(
    tmp_path,
    freq=FREQ,
    n_frames=40000,
    dt_fs=1.0,
    a=4.0,
    temperature=300.0,
    seed=4,
    frame_every=1,
    outcar=False,
):
    """Pure q = 0 motion of a 2x1x1 supercell along random mass-weighted eigenvectors.

    Also writes the exact harmonic forces: forces.npy for the XDATCAR frames and,
    with outcar=True, an OUTCAR excerpt with a TOTAL-FORCE block for every MD step
    (the XDATCAR keeps every frame_every-th step, as VASP with NBLOCK).
    """
    rng = np.random.default_rng(seed)
    m = np.array([ATOMIC_MASS[s] for s in SPECIES])
    sq = np.repeat(np.sqrt(m), 3)
    acoustic = np.array(
        [[sq[3 * s + i] * (i == b) for s in range(3) for i in range(3)] for b in range(3)]
    )
    acoustic /= np.linalg.norm(acoustic, axis=1)[:, None]
    basis, _ = np.linalg.qr(np.vstack([acoustic, rng.normal(size=(6, 9))]).T)
    e = basis.T.reshape(9, 3, 3)  # first three are translations (up to sign)
    freqs = np.concatenate([[0.0, 0.0, 0.0], freq])
    nc = 2
    t = np.arange(1, n_frames * frame_every + 1) * dt_fs * 1e-15  # MD steps 1, 2, ...
    w = 2 * np.pi * np.asarray(freq) * 1e12
    amp = np.sqrt(2 * ENERGY * KB * temperature) / w / (np.sqrt(AMU) * ANGSTROM)  # sqrt(amu) A
    q = amp[None] * np.cos(w[None] * t[:, None] + rng.uniform(0, 2 * np.pi, 6)[None])
    u_unit = np.einsum("tm,msi->tsi", q, e[3:]) / np.sqrt(m)[None, :, None] / np.sqrt(nc)
    lattice = np.diag([2 * a, a, a])
    cells = [np.array([0.0, 0, 0]), np.array([1.0, 0, 0])]
    sites = np.array([(UNIT[s] + c) * [a, a, a] for c in cells for s in range(3)])  # Cartesian
    order = [0, 3, 1, 2, 4, 5]  # Mg, Mg, O, O, O, O (species-grouped XDATCAR)
    disp = np.concatenate([u_unit, u_unit], axis=1)
    # F_k = -sum_m w_m^2 Q_m sqrt(m_k) e_m(k) / sqrt(N_c)  [eV/A]
    f_unit = -np.einsum("tm,msi->tsi", q * w[None] ** 2, e[3:]) * np.sqrt(m)[None, :, None]
    f_unit *= AMU * ANGSTROM**2 / E_CHARGE / np.sqrt(nc)
    force = np.concatenate([f_unit, f_unit], axis=1)[:, order]
    pos_all = (sites[None] + disp)[:, order]
    if outcar:
        with open(tmp_path / "OUTCAR", "w") as fh:
            for x, f in zip(pos_all, force):
                fh.write(" POSITION                                       TOTAL-FORCE (eV/Angst)\n")
                fh.write(" " + "-" * 83 + "\n")
                fh.writelines(
                    "".join(f"{v:13.5f}" for v in r) + "".join(f"{v:15.6f}" for v in g) + "\n"
                    for r, g in zip(x, f)
                )
                fh.write(" " + "-" * 83 + "\n")
    pos = pos_all[frame_every - 1 :: frame_every]
    np.save(tmp_path / "forces.npy", force[frame_every - 1 :: frame_every])
    write_xdatcar(
        tmp_path / "XDATCAR",
        ["Mg", "O"],
        [2, 4],
        (pos @ np.linalg.inv(lattice)) % 1.0,
        lattice,
        comment="akitsushima synthetic",
    )
    (tmp_path / "POSCAR_unit").write_text(
        f"unit\n1.0\n{a} 0 0\n0 {a} 0\n0 0 {a}\nMg O\n1 2\nDirect\n"
        + "".join(f"{x} {y} {z}\n" for x, y, z in UNIT)
    )
    np.savez(tmp_path / "modes.npz", frequencies_THz=freqs, eigenvectors=e)
    p = np.sqrt(nc) * np.einsum(
        "s,msa->am", CHARGES, e[3:] / np.sqrt(m)[None, :, None]
    )  # e / sqrt(amu)
    vol = abs(np.linalg.det(lattice)) * ANGSTROM**3
    w2 = w**2 * AMU  # kg-based: p^2 [e^2/amu] / (w^2) -> e^2 s^2 / amu
    d_eps = (p**2 / w2[None]).sum(axis=1) * E_CHARGE**2 / (EPS0 * vol)
    return d_eps


def _cfg(tmp_path, estimator, forces=None, frame_every=1, modes=None):
    return config.validate(
        {
            "trajectory": {"files": ["XDATCAR"], "timestep_fs": 1.0, "frame_every": frame_every},
            "temperature": 300.0,
            "dipole": {"model": "point_charge", "charges": {"Mg": 2.0, "O": -1.0}},
            "epsilon_inf": 1.0,
            "modes": modes
            or {"file": "modes.npz", "unit_cell": "POSCAR_unit", "supercell": [2, 1, 1]},
            "forces": forces,
            "spectrum": {"max_lag_ps": 4.0, "estimator": estimator, "kk_check": False},
            "output": {"dir": "out", "prefix": estimator, "plot": False},
        },
        tmp_path,
    )


@pytest.fixture(scope="module")
def crystal(tmp_path_factory):
    d = tmp_path_factory.mktemp("modes")
    return d, _harmonic_crystal(d)


def test_mode_estimator_exact_with_unequal_mode_energies(crystal):
    d, d_eps = crystal
    res = nozuchi.run(_cfg(d, "auto"), log=lambda *a: None)
    assert res["estimator"] == "mode"
    got = np.diag(np.array(res["eps_static_mode"])) - 1
    assert np.allclose(got, d_eps, rtol=0.01), (got, d_eps)
    # the f-sum estimator, reported alongside, is biased by the unequal energies
    fsum = np.diag(np.array(res["eps_static_fsum"])) - 1
    assert np.abs(fsum / d_eps - 1).max() > 0.05
    m = res["modes"][0]
    assert np.allclose(m["freq_effective"], FREQ, rtol=2e-3)
    assert np.allclose(m["kinetic_ratio"], ENERGY, rtol=0.03)
    assert m["variance_outside_modes"] < 1e-6
    assert np.allclose(np.sum(m["d_eps_diag"], axis=1), got, rtol=0.02)  # cross terms average out


def test_fsum_warns_when_modes_show_several_ir_modes(crystal):
    d, _ = crystal
    res = nozuchi.run(_cfg(d, "fsum"), log=lambda *a: None)
    assert res["estimator"] == "fsum"
    assert any("IR modes per component" in w for w in res["warnings"])


def test_mode_estimator_requires_modes(tmp_path):
    raw = {
        "trajectory": {"files": ["X"], "timestep_fs": 1.0, "frame_every": 1},
        "temperature": 300.0,
        "dipole": {"model": "point_charge", "charges": {"Mg": 2.0, "O": -2.0}},
        "spectrum": {"estimator": "mode"},
    }
    with pytest.raises(config.ConfigError, match="needs a 'modes' block"):
        config.validate(raw, tmp_path)
    raw["modes"] = {"file": "m.npz", "outcar": "OUTCAR", "unit_cell": "P", "supercell": [1, 1, 1]}
    with pytest.raises(config.ConfigError, match="exactly one"):
        config.validate(raw, tmp_path)


def test_read_outcar_modes(tmp_path):
    block = [
        ("   1 f  =   12.000000 THz", (0.6, 0.0, 0.8)),
        ("   2 f  =    9.000000 THz", (0.8, 0.0, -0.6)),
        ("   3 f/i=    0.010000 THz", (0.0, 1.0, 0.0)),
    ]
    text = " Eigenvectors and eigenvalues of the dynamical matrix\n ----\n\n"
    for head, v in block:
        text += f"{head}    1.0 2PiTHz    1.0 cm-1     1.0 meV\n"
        text += "             X         Y         Z           dx          dy          dz\n"
        text += f"      0.000000  0.000000  0.000000  {v[0]:10.6f}  {v[1]:10.6f}  {v[2]:10.6f}\n\n"
    p = tmp_path / "OUTCAR"
    p.write_text(text)
    assert md.has_outcar_modes(p)
    f, e = md.read_outcar_modes(p, 1)
    assert np.allclose(f, [12.0, 9.0, -0.01])  # f/i = imaginary -> negative
    assert np.allclose(e[1, 0], [0.8, 0.0, -0.6])
    with pytest.raises(ValueError, match="eigenvector block ends after"):
        md.read_outcar_modes(p, 2)  # 2 sites need 6 modes


def test_degenerate_modes_basis_independent(tmp_path):
    # a degenerate pair (5.5 THz twice) with random eigenvectors and unequal energies along
    # that arbitrary basis: per-component values must still be exact (SVD alignment)
    d_eps = _harmonic_crystal(tmp_path, freq=np.array([4.0, 5.5, 5.5, 8.5, 10.0, 12.0]), seed=7)
    res = nozuchi.run(_cfg(tmp_path, "mode"), log=lambda *a: None)
    got = np.diag(np.array(res["eps_static_mode"])) - 1
    assert np.allclose(got, d_eps, rtol=0.01), (got, d_eps)


def test_force_estimator_exact_with_unequal_mode_energies(crystal):
    d, d_eps = crystal
    res = nozuchi.run(_cfg(d, "auto", forces={"files": ["forces.npy"]}), log=lambda *a: None)
    got = np.diag(np.array(res["eps_static_force"])) - 1
    assert np.allclose(got, d_eps, rtol=1e-3), (got, d_eps)
    assert res["estimator"] == "mode"  # reported only, never selected
    assert np.allclose(res["force_modes"][0]["freq_force"], FREQ, rtol=1e-3)
    # forces on every 7th frame only (.npy stride_frames)
    np.save(d / "forces7.npy", np.load(d / "forces.npy")[::7])
    res7 = nozuchi.run(
        _cfg(d, "auto", forces={"files": ["forces7.npy"], "stride_frames": 7}), log=lambda *a: None
    )
    assert res7["force_modes"][0]["n_frames"] == len(range(0, 40000, 7))
    assert np.allclose(np.diag(np.array(res7["eps_static_force"])) - 1, d_eps, rtol=1e-3)


def test_force_estimator_from_outcar_with_nblock(tmp_path):
    d_eps = _harmonic_crystal(tmp_path, n_frames=3000, frame_every=2, outcar=True)
    res = nozuchi.run(
        _cfg(tmp_path, "auto", forces={"files": ["OUTCAR"]}, frame_every=2), log=lambda *a: None
    )
    got = np.diag(np.array(res["eps_static_force"])) - 1
    assert np.allclose(got, d_eps, rtol=2e-3), (got, d_eps)  # OUTCAR rounding (1e-6 eV/A)
    # wrong NBLOCK: the first kept force step is not the first frame
    with pytest.raises(ValueError, match="first kept force step"):
        nozuchi.run(
            _cfg(tmp_path, "auto", forces={"files": ["OUTCAR"]}, frame_every=1),
            log=lambda *a: None,
        )


def test_forces_config_errors(tmp_path):
    raw = {
        "trajectory": {"files": ["X"], "timestep_fs": 1.0, "frame_every": 1},
        "temperature": 300.0,
        "dipole": {"model": "point_charge", "charges": {"Mg": 2.0, "O": -2.0}},
        "forces": {"files": ["F.npy"]},
    }
    with pytest.raises(config.ConfigError, match="need Gamma modes"):
        config.validate(raw, tmp_path)
    raw["modes"] = {"file": "m.npz", "unit_cell": "P", "supercell": [1, 1, 1]}
    raw["forces"] = {"files": ["a.npy", "b.npy"]}
    with pytest.raises(config.ConfigError, match="one per trajectory file"):
        config.validate(raw, tmp_path)
    raw["forces"] = {"files": ["OUTCAR"], "stride_frames": 2}
    with pytest.raises(config.ConfigError, match=".npy forces only"):
        config.validate(raw, tmp_path)


def _write_qpoints(path, freqs, e, rng, mix=()):
    """phonopy qpoints.yaml with a random phase per mode and complex mixing inside `mix`."""
    c = (
        e.reshape(len(e), -1).astype(complex)
        * np.exp(2j * np.pi * rng.uniform(size=len(e)))[:, None]
    )
    for g in mix:  # random unitary rotation of a degenerate set
        u, _ = np.linalg.qr(
            rng.normal(size=(len(g), len(g))) + 1j * rng.normal(size=(len(g), len(g)))
        )
        c[g] = u @ c[g]
    lines = [
        "nqpoint: 1",
        f"natom: {e.shape[1]}",
        "phonon:",
        "- q-position: [ 0.0, 0.0, 0.0 ]",
        "  band:",
    ]
    for i, f in enumerate(freqs):
        lines += [f"  - # {i + 1}", f"    frequency: {f:.10f}", "    eigenvector:"]
        for s in range(e.shape[1]):
            lines.append(f"    - # atom {s + 1}")
            for k in range(3):
                z = c[i, 3 * s + k]
                lines.append(f"      - [ {z.real:.14f}, {z.imag:.14f} ]")
    path.write_text("\n".join(lines) + "\n")


def test_phonopy_qpoints_matches_npz(tmp_path):
    freq = np.array([4.0, 5.5, 5.5, 8.5, 10.0, 12.0])
    d_eps = _harmonic_crystal(tmp_path, freq=freq, n_frames=20000, seed=7)
    g = np.load(tmp_path / "modes.npz")
    _write_qpoints(
        tmp_path / "qpoints.yaml",
        g["frequencies_THz"],
        g["eigenvectors"],
        np.random.default_rng(1),
        mix=([0, 1, 2], [4, 5]),
    )
    f, e = md.read_phonopy_qpoints(tmp_path / "qpoints.yaml")
    assert np.allclose(f, g["frequencies_THz"])
    assert np.allclose(np.einsum("mai,nai->mn", e, e), np.eye(9), atol=1e-10)
    for grp in md.degenerate_groups(f):  # same eigenspaces as the source
        pa, pb = e[grp].reshape(len(grp), -1), g["eigenvectors"][grp].reshape(len(grp), -1)
        assert np.allclose(pa.T @ pa, pb.T @ pb, atol=1e-10)
    modes = {"phonopy": "qpoints.yaml", "unit_cell": "POSCAR_unit", "supercell": [2, 1, 1]}
    a = nozuchi.run(_cfg(tmp_path, "mode", modes=modes), log=lambda *x: None)
    b = nozuchi.run(_cfg(tmp_path, "mode"), log=lambda *x: None)
    # another basis inside the 5.5 THz pair changes the finite-run statistics slightly (2e-5)
    assert np.allclose(a["eps_static_mode"], b["eps_static_mode"], rtol=1e-3)
    assert np.allclose(np.diag(np.array(a["eps_static_mode"])) - 1, d_eps, rtol=0.01)


def test_mode_resolved_spectra_harmonic(crystal):
    d, _ = crystal
    res = nozuchi.run(_cfg(d, "auto"), log=lambda *a: None)
    tab = res["mode_table"]
    assert [r["index"] for r in tab] == list(range(4, 10))  # optical modes of the source
    # an undamped line is located to the periodogram resolution 1 / T_run = 0.025 THz
    assert np.allclose([r["freq_peak"] for r in tab], FREQ, atol=0.025)
    assert not any(r["resolved"] for r in tab)  # undamped: narrower than 1 / T_run
    total = np.diag(np.array(res["eps_static_mode"])) - 1
    # split decomposition: exact; diagonal only: the cross terms average out here
    assert np.allclose(np.sum([r["d_eps"] for r in tab], axis=0), total, rtol=1e-10)
    assert np.allclose(np.sum([r["d_eps_diag_only"] for r in tab], axis=0), total, rtol=0.02)
    r = res["mode_residual"]
    assert max(map(abs, r["split_static"] + r["split_spectral"])) < 1e-10
    assert max(abs(x) for x in r["cross_share_static"]) < 0.02
    cols = np.loadtxt(d / "out" / "auto_modes.dat")
    assert cols.shape[1] == 1 + 2 * 3 * len(FREQ)  # split and diagonal-only columns


def test_mode_split_exact_with_correlated_modes(tmp_path):
    # Gamma "modes" rotated against the true ones (a 4.0 / 5.5 THz pair mixed by 30 degrees, as
    # DFPT vs AIMD eigenvectors): the coordinates are correlated, the diagonal-only sum misses
    # the cross terms, the split decomposition still sums to the estimator total
    freq = np.array([4.0, 5.5, 5.5, 8.5, 10.0, 12.0])  # with a degenerate pair as well
    _harmonic_crystal(tmp_path, freq=freq, n_frames=20000, seed=7)
    g = np.load(tmp_path / "modes.npz")
    e, f = g["eigenvectors"].copy(), g["frequencies_THz"].copy()
    c, s_ = np.cos(np.pi / 6), np.sin(np.pi / 6)
    e[3], e[6] = (
        c * g["eigenvectors"][3] + s_ * g["eigenvectors"][6],
        (-s_ * g["eigenvectors"][3] + c * g["eigenvectors"][6]),
    )
    np.savez(tmp_path / "rot.npz", frequencies_THz=f, eigenvectors=e)
    modes = {"file": "rot.npz", "unit_cell": "POSCAR_unit", "supercell": [2, 1, 1]}
    res = nozuchi.run(_cfg(tmp_path, "mode", modes=modes), log=lambda *a: None)
    tab, r = res["mode_table"], res["mode_residual"]
    total = np.diag(np.array(res["eps_static_mode"])) - 1
    assert np.allclose(np.sum([x["d_eps"] for x in tab], axis=0), total, rtol=1e-10)
    assert max(map(abs, r["split_static"] + r["split_spectral"])) < 1e-10
    assert max(abs(x) for x in r["cross_share_static"]) > 0.02  # real cross terms
    im = np.loadtxt(tmp_path / "out" / "mode_modes.dat")
    eps = np.loadtxt(tmp_path / "out" / "mode_eps.dat")
    n = len(tab)
    for a, col in enumerate((2, 4, 6)):  # eps_im_xx, _yy, _zz; mode columns m*3 + a
        assert np.allclose(im[:, 1 + a : 1 + 3 * n : 3].sum(axis=1), eps[:, col], atol=1e-8)


def test_line_fit_damped_oscillator():
    from akitsushima.mdspec import modespec

    rng = np.random.default_rng(11)
    f0, g, dt = 5.0, 0.2, 1e-3  # THz, THz (FWHM), ps
    w0, gam = 2 * np.pi * f0, 2 * np.pi * g
    n = 200000
    x, v = 0.0, 0.0
    c1 = np.exp(-gam * dt)
    c2 = np.sqrt(1 - c1**2)
    xs = np.empty(n)
    kick = rng.normal(size=n)
    for i in range(n):  # BAOAB Langevin, unit mass and kT
        v -= 0.5 * dt * w0**2 * x
        x += 0.5 * dt * v
        v = c1 * v + c2 * kick[i]
        x += 0.5 * dt * v
        v -= 0.5 * dt * w0**2 * x
        xs[i] = x
    out = modespec.line_fit(xs, dt * 1e-12, f0 * 1e12)
    assert out["f_peak"] / 1e12 == pytest.approx(f0, rel=0.01)
    assert out["fwhm"] / 1e12 == pytest.approx(g, rel=0.15)
    assert out["resolved"]
