"""Tier 1: FDT transform against analytic models, f-sum estimator, KK, drift (ADR-0003)."""

import warnings

import numpy as np
import pytest
from scipy.fft import next_fast_len

from akitsushima.mdspec import spectrum as sp
from akitsushima.mdspec.dipole import Run


def _tensor_c(c1d):
    c = np.zeros((len(c1d), 3, 3))
    for a in range(3):
        c[:, a, a] = c1d
    return c


def test_fdt_debye_exact_correlation():
    dt, tau = 1e-15, 200e-15
    t = np.arange(20000) * dt
    freq, xr, xi = sp.fdt_transform(_tensor_c(np.exp(-t / tau)), dt, "none", 2 * len(t))
    w = 2 * np.pi * freq
    debye = 1 / (1 + 1j * w * tau)
    m = freq < 0.2 / dt
    assert np.abs(xr[m, 0, 0] - debye.real[m]).max() < 2e-3
    assert np.abs(xi[m, 0, 0] + debye.imag[m]).max() < 2e-3


def test_fdt_damped_oscillator_and_jump_correction():
    dt, f0, gam = 2e-15, 12.4e12, 2 * np.pi * 0.5e12
    w0 = 2 * np.pi * f0
    w1 = np.sqrt(w0**2 - gam**2 / 4)
    t = np.arange(200000) * dt
    c = np.exp(-gam * t / 2) * (np.cos(w1 * t) + gam / (2 * w1) * np.sin(w1 * t))
    freq, xr, xi = sp.fdt_transform(_tensor_c(c), dt, "none", 2 * len(t))
    w = 2 * np.pi * freq
    exact = w0**2 / (w0**2 - w**2 + 1j * gam * w)
    m = (freq > 0.5e12) & (freq < 200e12) & (np.abs(freq - f0) > 2e12)
    assert np.abs(xr[m, 0, 0] - exact.real[m]).max() < 2e-3  # 0.59 without the jump correction
    assert np.abs(xi[m, 0, 0] + exact.imag[m]).max() < 1e-5  # band-edge aliasing ~2e-6


def test_correlation_tensor_matches_direct():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(500, 3))
    x -= x.mean(0)
    c = sp.correlation_tensor(x, 10)
    for lag in (0, 3, 10):
        d = (x[: len(x) - lag].T @ x[lag:]) / (len(x) - lag)
        assert np.allclose(c[lag], 0.5 * (d + d.T))


def test_ou_process_debye_static_and_peak():
    """Ornstein-Uhlenbeck M(t): Debye relaxation with known eps_s (Tier 1, 1 %)."""
    rng = np.random.default_rng(7)
    dt_fs, tau_fs, sigma, n = 1.0, 20.0, 3.0, 400000
    a = np.exp(-dt_fs / tau_fs)
    noise = rng.normal(size=(n, 3)) * sigma * np.sqrt(1 - a**2)
    m = np.empty((n, 3))
    m[0] = rng.normal(size=3) * sigma
    for i in range(1, n):
        m[i] = a * m[i - 1] + noise[i]
    vol, temp = 1000.0, 300.0
    run = Run(m, np.full(n, vol), dt_fs, ["ou"])
    cfg = {
        "max_lag_ps": 0.4,
        "window": "hann",
        "zero_pad_factor": 4,
        "kk_check": False,
        "n_blocks": 5,
        "estimator": "raw",
    }
    res = sp.analyse([run], np.zeros((1, 3, 3)), None, temp, np.eye(3), cfg)
    pref = (sp.E_CHARGE * sp.ANGSTROM) ** 2 / (sp.EPS0 * vol * sp.ANGSTROM**3 * sp.KB * temp)
    expect = 1 + sigma**2 * pref
    assert abs(sp.iso(res["static_raw"]) / expect - 1) < 0.01
    f = res["freq_hz"]
    im = sp.iso(res["eps_im"])
    f_peak = 1 / (2 * np.pi * tau_fs * 1e-15)
    df = f[1] - f[0]
    sel = f < 3 * f_peak
    assert abs(f[sel][np.argmax(im[sel])] - f_peak) < 3 * df + 0.05 * f_peak


def test_fsum_estimator_on_langevin_crystal(crystal_traj):
    pos, lattice, _symbols, masses, q, d_eps = crystal_traj
    frac = pos @ np.linalg.inv(lattice)
    m = np.einsum("n,tnb->tb", q, frac) @ lattice
    run = Run(m[::2], np.full(len(m[::2]), abs(np.linalg.det(lattice))), 2.0, ["langevin"])
    z = q[:, None, None] * np.eye(3)[None]
    cfg = {
        "max_lag_ps": 2.0,
        "window": "hann",
        "zero_pad_factor": 2,
        "kk_check": True,
        "n_blocks": 5,
        "estimator": "fsum",
    }
    res = sp.analyse([run], z, masses, 300.0, np.eye(3), cfg)
    norm = sp.iso(res["static_fsum"]) - 1
    raw = sp.iso(res["static_raw"]) - 1
    # M is one collective variable with ~20 independent samples in 40 ps
    # (1/gamma = 2 ps): the raw estimator is off by ~15 % here, the
    # f-sum-normalised one is not, because the same sampling error appears in
    # <Mdot^2>_MD and cancels (ADR-0003 item 6).
    assert abs(norm / d_eps - 1) < 0.03
    ratio = np.mean(res["fsum_ratio"][0])
    assert abs((raw / d_eps) / ratio - 1) < 0.05
    assert res["kk_rms_dev_rel"] < 0.01


def test_fsum_zero_charges_gives_zero_response():
    run = Run(np.zeros((1000, 3)), np.full(1000, 100.0), 1.0, ["si"])
    cfg = {
        "max_lag_ps": "auto",
        "window": "hann",
        "zero_pad_factor": 1,
        "kk_check": False,
        "n_blocks": 2,
        "estimator": "fsum",
    }
    res = sp.analyse([run], np.zeros((4, 3, 3)), np.full(4, 28.0), 300.0, 11.7 * np.eye(3), cfg)
    assert np.allclose(res["static_fsum"], 11.7 * np.eye(3))
    assert np.allclose(res["eps_im"], 0)


def test_kk_maclaurin_lorentz():
    f = np.linspace(0, 250e12, 10001)
    eps = 6.0 * 12.4**2 / (12.4**2 - (f / 1e12) ** 2 + 1j * 0.5 * f / 1e12)
    kk = sp.kk_real(f, -eps.imag)
    lim = f < 200e12
    assert np.abs(kk[lim] - eps.real[lim]).max() < 1e-3


def test_drift_warning():
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.normal(size=(20000, 3)), axis=0)
    assert sp.drift_ratio(walk) > 2.5
    bounded = rng.normal(size=(20000, 3))
    assert sp.drift_ratio(bounded) < 1.5
    run = Run(walk, np.full(len(walk), 100.0), 1.0, ["w"])
    cfg = {
        "max_lag_ps": "auto",
        "window": "hann",
        "zero_pad_factor": 1,
        "kk_check": False,
        "n_blocks": 2,
        "estimator": "raw",
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.analyse([run], np.zeros((1, 3, 3)), None, 300.0, np.eye(3), cfg)
    msgs = [str(w.message) for w in caught]
    assert any("drifts" in m for m in msgs)
    assert any("has not decayed" in m for m in msgs)  # a random walk never decorrelates


def _ou(n, tau_steps, rng):
    a = np.exp(-1.0 / tau_steps)
    x = np.zeros((n, 3))
    noise = rng.normal(size=(n, 3)) * np.sqrt(1 - a**2)
    for i in range(1, n):
        x[i] = a * x[i - 1] + noise[i]
    return x


def test_acf_tail_warning_slow_relaxation():
    # OU relaxation time 4000 frames vs max_lag 1000: C(tau)/C(0) ~ exp(-0.25 ... -0.2) at the tail
    rng = np.random.default_rng(3)
    cfg = {
        "max_lag_ps": 1.0,
        "window": "hann",
        "zero_pad_factor": 1,
        "kk_check": False,
        "n_blocks": 2,
        "estimator": "raw",
    }
    slow = Run(_ou(100000, 4000, rng), np.full(100000, 100.0), 1.0, ["s"])
    with pytest.warns(UserWarning, match="has not decayed"):
        res = sp.analyse([slow], np.zeros((1, 3, 3)), None, 300.0, np.eye(3), cfg)
    assert min(res["acf_tail_ratio"][0]) > 0.5
    fast = Run(_ou(100000, 50, rng), np.full(100000, 100.0), 1.0, ["f"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sp.analyse([fast], np.zeros((1, 3, 3)), None, 300.0, np.eye(3), cfg)
    assert not any("has not decayed" in str(w.message) for w in caught)
    assert max(abs(x) for x in res["acf_tail_ratio"][0]) < 0.05


def test_single_run_warns_no_seed_scatter():
    rng = np.random.default_rng(4)
    cfg = {
        "max_lag_ps": 0.2,
        "window": "hann",
        "zero_pad_factor": 1,
        "kk_check": False,
        "n_blocks": 2,
        "estimator": "raw",
    }
    runs = [Run(_ou(5000, 20, rng), np.full(5000, 100.0), 1.0, [f"r{i}"]) for i in range(2)]
    with pytest.warns(UserWarning, match="single run"):
        sp.analyse(runs[:1], np.zeros((1, 3, 3)), None, 300.0, np.eye(3), cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.analyse(runs, np.zeros((1, 3, 3)), None, 300.0, np.eye(3), cfg)
    assert not any("single run" in str(w.message) for w in caught)


def test_next_fast_len_grid_shared_by_blocks():
    assert next_fast_len(1000) >= 1000


def test_mdot2_no_endpoint_jump_artefact():
    # a slow oscillation over a non-integer number of periods: the periodic FFT sees a jump
    dt, f0, n = 0.5e-15, 8.46e12, 20000
    t = np.arange(n) * dt
    x = np.cos(2 * np.pi * f0 * t + 0.3)[:, None]
    exact = 0.5 * (2 * np.pi * f0) ** 2
    assert abs(sp.mdot2_md(x - x.mean(0), dt)[0] / exact - 1) < 0.01
