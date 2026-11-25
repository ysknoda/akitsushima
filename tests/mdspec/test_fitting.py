"""Fit models recover known parameters (ADR-0005)."""

import numpy as np

from akitsushima.mdspec import fitting as ft


def test_lorentz_plus_debye_recovered():
    f = np.linspace(0.05, 30, 1500)
    true = (
        3.0 + ft.term_value("lorentz", (6.0, 12.4, 0.3), f) + ft.term_value("debye", (2.0, 0.5), f)
    )
    rng = np.random.default_rng(0)
    s = 0.01
    re, im = (
        true.real + rng.normal(scale=s, size=f.size),
        -true.imag + rng.normal(scale=s, size=f.size),
    )
    res = ft.fit(
        f,
        re,
        im,
        3.0,
        "additive",
        [{"type": "lorentz", "f0": 12.0}, {"type": "debye", "tau_ps": 0.3}],
        np.full(f.size, s),
        np.full(f.size, s),
    )
    p, e = np.array(res["params"]), np.array(res["errors"])
    truth = np.array([6.0, 12.4, 0.3, 2.0, 0.5])
    assert np.all(np.abs(p - truth) < 5 * e + 1e-6), (p, e)
    assert 0.8 < res["reduced_chi2"] < 1.2
    assert abs(res["static"] - 11.0) < 0.05


def test_gervais_piriou_recovers_lo_to():
    f = np.linspace(0.1, 40, 2000)
    true = ft.gervais_piriou(6.5, [23.9, 0.2, 29.1, 0.3], f)
    res = ft.fit(f, true.real, -true.imag, 6.5, "gervais_piriou", [{"f_to": 23.0, "f_lo": 28.0}])
    ft_to, _, f_lo, _ = res["params"]
    assert abs(ft_to - 23.9) < 1e-3 and abs(f_lo - 29.1) < 1e-3
    # Lyddane-Sachs-Teller: eps_s / eps_inf = (f_LO / f_TO)^2
    assert abs(res["static"] - 6.5 * (29.1 / 23.9) ** 2) < 1e-3


def test_bic_prefers_true_model():
    f = np.linspace(0.05, 30, 800)
    true = 2.0 + ft.term_value("lorentz", (4.0, 10.0, 0.5), f)
    rng = np.random.default_rng(1)
    s = 0.01
    re, im = (
        true.real + rng.normal(scale=s, size=f.size),
        -true.imag + rng.normal(scale=s, size=f.size),
    )
    cfg = {
        "model": "additive",
        "terms": [{"type": "lorentz", "f0": 9.0}],
        "candidates": [[{"type": "lorentz", "f0": 9.0}, {"type": "lorentz", "f0": 20.0}]],
    }
    best, allr = ft.fit_with_candidates(f, re, im, 2.0, cfg, np.full(f.size, s), np.full(f.size, s))
    assert best is allr[0]


def test_hn_special_cases():
    f = np.linspace(0.01, 10, 50)
    d = ft.term_value("debye", (2.0, 0.3), f)
    assert np.allclose(ft.term_value("havriliak_negami", (2.0, 0.3, 1.0, 1.0), f), d)
    assert np.allclose(ft.term_value("cole_cole", (2.0, 0.3, 1.0), f), d)
    assert np.allclose(ft.term_value("cole_davidson", (2.0, 0.3, 1.0), f), d)
    assert np.all(-ft.term_value("drude", (1.0, 0.5), f).imag > 0)  # absorption positive
