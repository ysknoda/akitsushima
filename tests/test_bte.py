"""RTA lattice thermal conductivity (ADR-0042)."""

import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from akitsushima.cli.cmd_bte import mass_variances_from_config
from akitsushima.cli.templates import _DUMP_MAP
from akitsushima.constants import ATOMIC_MASS
from akitsushima.solvers.bte import (
    _THZ,
    BTECalculator,
    _permute_average,
    _translation_maps,
    mass_variance,
)
from akitsushima.solvers.config_bridge import _build_force_fn_from_yaml

pytest.importorskip("spglib")


def _si_fn():
    return _build_force_fn_from_yaml(yaml.safe_load(_DUMP_MAP["qha"])["potential"])[0]


def _si_conv(a=5.432):
    f8 = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    f8 = np.vstack([f8, f8 + 0.25])
    lat = a * np.eye(3)
    return lat, f8 @ lat, ["Si"] * 8


def _aln():
    import test_phonon_regressions as T

    fn = _build_force_fn_from_yaml(
        {"type": "Tersoff", "species": ["Al", "N"], "params": T._ALN}
    )[0]
    a, c, u = 3.11, 4.98, 0.382
    lat = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    fr = np.array(
        [
            [1 / 3, 2 / 3, 0],
            [2 / 3, 1 / 3, 0.5],
            [1 / 3, 2 / 3, u],
            [2 / 3, 1 / 3, 0.5 + u],
        ]
    )
    return lat, fr @ lat, ["Al", "Al", "N", "N"], fn


@pytest.fixture(scope="module")
def si():
    lat, pos, types = _si_conv()
    bc = BTECalculator(lat, pos, types, _si_fn(), supercell=2)
    bc.compute_fc2()
    bc.compute_fc3()
    return bc


def _freqs(bc, q):
    w2 = np.linalg.eigvalsh(bc.ph2.build_dynamical_matrix(q))
    from akitsushima.solvers.bte import _W2_SI

    return np.sign(w2) * np.sqrt(np.abs(w2) * _W2_SI) / (2 * np.pi * _THZ)


def _check_velocity(bc, q, tol):
    D = bc.ph2.build_dynamical_matrix(q)
    _, e = np.linalg.eigh(D)
    f = _freqs(bc, q)
    v = bc._velocities(q, f, e, np.eye(3)[None])
    h = 1e-5  # 1/A
    for a in range(3):
        dq = np.zeros(3)
        dq[a] = h
        fd = (
            (_freqs(bc, q + dq) - _freqs(bc, q - dq))
            / (2 * h)
            * 2
            * np.pi
            * _THZ
            * 1e-10
        )
        nondeg = np.r_[True, np.diff(f) > 1e-3] & np.r_[np.diff(f) > 1e-3, True]
        assert np.allclose(v[nondeg, a], fd[nondeg], rtol=tol, atol=1.0), (
            a,
            v[:, a],
            fd,
        )


def test_group_velocity_matches_finite_differences(si):
    q = np.array([0.11, 0.07, 0.19]) @ (2 * np.pi * np.linalg.inv(si.lattice).T)
    _check_velocity(si, q, 1e-5)


def test_group_velocity_with_nac_matches_finite_differences():
    lat, pos, types, fn = _aln()
    bc = BTECalculator(lat, pos, types, fn, supercell=[3, 3, 2])
    bc.compute_fc2()
    Z = np.array([np.diag([2.6, 2.6, 2.7])] * 2 + [-np.diag([2.6, 2.6, 2.7])] * 2)
    bc.ph2.set_nac(Z, np.diag([4.3, 4.3, 4.5]))  # model values for the test
    q = np.array([0.13, 0.05, 0.21]) @ (2 * np.pi * np.linalg.inv(lat).T)
    _check_velocity(bc, q, 1e-4)


def test_fc3_permutation_symmetry_and_sum_rule(si):
    ph = si.ph3
    basis = np.asarray(ph._sc_to_prim)
    origin = np.array(ph._origin_indices())
    T = _translation_maps(ph)
    scale = np.abs(si.fc3).max()
    P = _permute_average(si.fc3, basis, origin, T)
    assert np.abs(P - si.fc3).max() / scale < 1e-12
    assert np.abs(si.fc3.sum(axis=2)).max() / scale < 1e-8
    # unsymmetrised: permutation symmetry holds to the O(delta^2) of the
    # central differences
    lat, pos, types = _si_conv()
    raw = BTECalculator(lat, pos, types, _si_fn(), supercell=2, sym_fc3=False)
    raw.compute_fc2()
    raw.compute_fc3()
    Pr = _permute_average(raw.fc3, basis, origin, T)
    assert np.abs(Pr - raw.fc3).max() / scale < 2e-3


def test_classical_kappa_scales_as_one_over_T(si):
    r = si.compute_kappa(
        3, sigma=0.2, temperatures=[300.0, 600.0], occupation="classical"
    )
    k = r["kappa_W_mK"]
    assert np.allclose(k[0], 2.0 * k[1], rtol=1e-10, atol=1e-12)
    cv = r["heat_capacity_J_K"]
    on = np.broadcast_to(r["frequencies_THz"][None] > 1e-4, cv.shape)
    assert np.allclose(cv[on], 1.380649e-23)
    # quantum -> classical at high temperature
    hi = 4000.0
    kq = si.compute_kappa(3, sigma=0.2, temperatures=[hi])["kappa_W_mK"][0]
    kc = si.compute_kappa(3, sigma=0.2, temperatures=[hi], occupation="classical")[
        "kappa_W_mK"
    ][0]
    assert abs(kq[0, 0] / kc[0, 0] - 1) < 0.02


def test_tamura_on_a_cubic_monatomic_crystal():
    """Monatomic cubic crystal on a full Oh-symmetric mesh:
    sum_{q' l'} e' e'^T delta(f - f') is isotropic, so
    gamma_iso = pi f^2 g / (4 N) (1/3) sum_{q' l'} delta(f - f')."""
    fn = _build_force_fn_from_yaml(
        {
            "type": "LJ",
            "species": ["Ar"],
            "params": {"epsilon": 0.01032, "sigma": 3.405, "r_cut": 7.0},
        }
    )[0]
    a = 5.26
    lat = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    bc = BTECalculator(lat, np.zeros((1, 3)), ["Ar"], fn, supercell=5, symmetry=False)
    bc.compute_fc2()
    g = 1e-3
    bc.set_mass_variance([g])
    mesh, addr, _, _, _ = bc._mesh(6)
    q = (addr / mesh) @ (2 * np.pi * np.linalg.inv(lat).T)
    freqs, evecs = bc._harmonic(q)
    sigma = 0.1
    gp = 17
    got = bc._gamma_iso(gp, freqs, evecs, sigma)
    f0 = freqs[gp]
    d = f0[:, None, None] - freqs[None]
    delta = np.exp(-0.5 * (d / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    delta = delta * (freqs[None] > 1e-4)
    want = np.pi * f0**2 * g / (4 * len(q)) * delta.sum(axis=(1, 2)) / 3
    assert np.allclose(got, want, rtol=1e-10)


def test_mass_variance_inputs():
    # Si natural isotopes -> Tamura's g = 2.01e-4
    g = mass_variance([27.97693, 28.97649, 29.97377], [0.92223, 0.04685, 0.03092])
    assert g == pytest.approx(2.01e-4, rel=5e-3)
    types = ["Ge"] * 2
    m0 = np.array([ATOMIC_MASS["Ge"]] * 2)
    m, gg = mass_variances_from_config({"alloy": {"host": "Ge", "Sn": 0.1}}, types, m0)
    mbar = 0.9 * ATOMIC_MASS["Ge"] + 0.1 * ATOMIC_MASS["Sn"]
    assert np.allclose(m, mbar)
    want = (
        0.9 * (1 - ATOMIC_MASS["Ge"] / mbar) ** 2
        + 0.1 * (1 - ATOMIC_MASS["Sn"] / mbar) ** 2
    )
    assert np.allclose(gg, want)
    _, g2 = mass_variances_from_config({"mass_variance": {"Ge": 5e-4}}, types, m0)
    assert np.allclose(g2, 5e-4)
    with pytest.raises(ValueError, match="not in the structure"):
        mass_variances_from_config({"mass_variance": {"Si": 1e-4}}, types, m0)


def test_boundary_scattering_is_v_over_L(si):
    si.set_boundary_length(0.05)  # um
    try:
        r = si.compute_kappa(3, sigma=0.2, temperatures=[300.0])
    finally:
        si.set_boundary_length(None)
    v = np.linalg.norm(r["group_velocity_m_s"], axis=2)
    assert np.allclose(r["gamma_boundary_THz"], v / 0.05e-6 / (4 * np.pi * _THZ))
    k0 = si.compute_kappa(3, sigma=0.2, temperatures=[300.0])["kappa_W_mK"][0, 0, 0]
    assert r["kappa_W_mK"][0, 0, 0] < k0


@pytest.mark.slow
def test_symmetry_reduction_matches_the_full_displacement_set():
    """fc3 from the ADR-0040 first displacements and their images equals
    fc3 from every origin atom displaced along +/-x, y, z, to the O(delta^2)
    of the finite differences, with a quarter of the force calls."""
    lat, pos, types, fn = _aln()
    out = {}
    for sym in (True, False):
        bc = BTECalculator(lat, pos, types, fn, supercell=[3, 3, 2], symmetry=sym,
                           sym_fc3=False)
        bc.compute_fc3()
        out[sym] = (bc.fc3_info["n_force_calls"], bc.fc3)
    assert out[True][0] * 4 == out[False][0]
    scale = np.abs(out[False][1]).max()
    assert np.abs(out[True][1] - out[False][1]).max() / scale < 2e-3


@pytest.mark.slow
def test_phono3py_oracle(si):
    """ADR-0042 gate on Si: own fc2 / fc3 given to phono3py (kappa 0.5 %,
    gamma 1 %), and phono3py's own displacements with the same force_fn
    (kappa 2 %)."""
    pytest.importorskip("phono3py")
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms

    lat, pos, types = _si_conv()
    mesh, sigma, T = [4, 4, 4], 0.2, [300.0]
    r = si.compute_kappa(mesh, sigma=sigma, temperatures=T)
    k_own = np.diag(r["kappa_W_mK"][0])
    cell = PhonopyAtoms(
        symbols=types, cell=lat, scaled_positions=pos @ np.linalg.inv(lat)
    )

    def p3():
        return Phono3py(
            cell,
            supercell_matrix=[2, 2, 2],
            primitive_matrix="P",
            make_r0_average=False,
            log_level=0,
        )

    a = p3()
    L = si.ph3.lattice_sc
    d = (
        a.supercell.scaled_positions[:, None]
        - (si.ph3._pos_sc @ np.linalg.inv(L))[None]
    )
    d -= np.round(d)
    m = np.argmin(np.linalg.norm(d @ L, axis=2), axis=1)
    N = len(m)
    a.fc2 = np.ascontiguousarray(
        si.ph2.IFC.reshape(N, 3, N, 3)[np.ix_(m, range(3), m, range(3))].transpose(
            0, 2, 1, 3
        )
    )
    Tm = _translation_maps(si.ph3)
    basis = np.asarray(si.ph3._sc_to_prim)
    full = np.array([si.fc3[basis[i]][np.ix_(Tm[i], Tm[i])] for i in range(N)])
    a.fc3 = np.ascontiguousarray(full[np.ix_(m, m, m)])
    a.mesh_numbers = mesh
    a.sigmas = [sigma]
    a.init_phph_interaction()
    a.run_thermal_conductivity(temperatures=T)
    tc = a.thermal_conductivity
    assert np.allclose(k_own, tc.kappa[0, 0][:3], rtol=5e-3)
    go = np.sort(np.repeat(r["gamma_THz"][0], r["weights"], axis=0).ravel())
    gp = np.sort(np.repeat(tc.gamma[0, 0], tc.grid_weights, axis=0).ravel())
    sel = gp > 1e-3
    assert np.max(np.abs(go[sel] - gp[sel]) / gp[sel]) < 1e-2

    b = p3()
    b.generate_displacements(distance=0.03)
    fn = _si_fn()
    b.forces = np.array(
        [
            np.zeros((N, 3))
            if s is None
            else np.asarray(fn(s.cell, s.positions, list(s.symbols))[1])
            for s in b.supercells_with_displacements
        ]
    )
    b.produce_fc3(symmetrize_fc3r=True, fc_calculator="traditional")
    b.produce_fc2(symmetrize_fc2=True, fc_calculator="traditional")
    b.mesh_numbers = mesh
    b.sigmas = [sigma]
    b.init_phph_interaction()
    b.run_thermal_conductivity(temperatures=T)
    assert np.allclose(k_own, b.thermal_conductivity.kappa[0, 0][:3], rtol=2e-2)
