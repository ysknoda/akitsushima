"""Optical quantities from eps(omega) (ADR-0010 F1): analytic Lorentz oscillators (Tier 1)."""

import warnings

import numpy as np
import pytest

from akitsushima.mdspec import optics

THZ = 1e12


def _lorentz(f, f_to, d_eps, gamma, eps_inf):
    """Physics convention eps = eps_inf + sum d f_to^2 / (f_to^2 - f^2 - i gamma f)."""
    e = np.full(len(f), eps_inf, complex)
    for ft, de, g in zip(f_to, d_eps, gamma):
        e += de * ft**2 / (ft**2 - f**2 - 1j * g * f)
    return e


def _tensor(e):
    """nozuchi arrays (eps', eps'' with eps = eps' - i eps'') as isotropic 3x3 tensors."""
    eye = np.eye(3)[None]
    return e.real[:, None, None] * eye, e.imag[:, None, None] * eye


def test_single_oscillator_optics_and_lst():
    f = np.linspace(0.0, 40.0, 40001)  # THz
    f_to, d_eps, eps_inf, g = 10.0, 5.0, 4.0, 0.05
    e = _lorentz(f, [f_to], [d_eps], [g], eps_inf)
    er, ei = _tensor(e)
    q = optics.quantities(f * THZ, er, ei)
    nc = np.sqrt(e)  # n + i kappa in the physics convention
    assert np.allclose(q["n"][:, 0], nc.real)
    assert np.allclose(q["kappa"][:, 0], nc.imag)
    assert np.allclose(q["R"][:, 3], np.abs((nc - 1) / (nc + 1)) ** 2)
    assert np.allclose(q["loss"][:, 1], (-1 / e).imag)
    k = np.argmin(np.abs(f - 12.0))  # alpha = 4 pi f kappa / c
    assert q["alpha_cm-1"][k, 2] == pytest.approx(4 * np.pi * 12e12 * nc.imag[k] / 2.99792458e10)
    # reststrahlen band between TO and LO: R close to 1
    f_lo = f_to * np.sqrt((eps_inf + d_eps) / eps_inf)
    band = (f > f_to + 0.5) & (f < f_lo - 0.5)
    assert q["R"][band, 0].min() > 0.9
    static = (eps_inf + d_eps) * np.eye(3)
    _, s = optics.analyse(f, f * THZ, er, ei, static, eps_inf * np.eye(3))
    assert s["xx"]["to_peaks"] == pytest.approx([f_to], abs=2e-3)
    assert s["xx"]["lo_peaks"] == pytest.approx([f_lo], abs=2e-3)
    assert s["iso"]["lst"]["ratio"] == pytest.approx(1.0, abs=1e-3)


def test_two_oscillators_lst_and_anisotropy():
    f = np.linspace(0.0, 30.0, 30001)
    e = _lorentz(f, [6.0, 12.0], [3.0, 1.5], [0.05, 0.05], 3.0)
    ez = _lorentz(f, [8.0], [2.0], [0.05], 3.0)
    er, ei = _tensor(e)
    er[:, 2, 2], ei[:, 2, 2] = ez.real, ez.imag
    static = np.diag([7.5, 7.5, 5.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # diagonal tensor: no off-diagonal warning
        _, s = optics.analyse(f, f * THZ, er, ei, static, 3.0 * np.eye(3))
    assert len(s["yy"]["to_peaks"]) == 2 and len(s["yy"]["lo_peaks"]) == 2
    assert s["yy"]["lst"]["ratio"] == pytest.approx(1.0, abs=2e-3)
    assert s["zz"]["lo_peaks"] == pytest.approx([8.0 * np.sqrt(5.0 / 3.0)], abs=2e-3)


def test_offdiagonal_static_warns():
    f = np.linspace(0.0, 20.0, 2001)
    er, ei = _tensor(_lorentz(f, [8.0], [2.0], [0.2], 3.0))
    static = 5.0 * np.eye(3)
    static[0, 1] = static[1, 0] = 0.3  # 15 % of Delta eps
    with pytest.warns(UserWarning, match="principal axes"):
        optics.analyse(f, f * THZ, er, ei, static, 3.0 * np.eye(3))
