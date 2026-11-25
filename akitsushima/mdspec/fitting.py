"""Parametric permittivity models and joint eps' / eps'' fits (ADR-0005).

Convention eps = eps' - i eps'' (eps'' > 0).  Frequencies f in THz,
omega = 2 pi f in rad/ps, relaxation times tau in ps.  eps_inf is fixed (an
input, ADR-0001); additive terms model eps - eps_inf.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

# term: (parameter names, defaults, lower bounds, upper bounds)
TERMS = {
    "debye": (("delta", "tau_ps"), (1.0, 1.0), (0.0, 1e-6), (np.inf, np.inf)),
    "cole_cole": (("delta", "tau_ps", "alpha"), (1.0, 1.0, 0.8), (0.0, 1e-6, 1e-3), (np.inf, np.inf, 1.0)),
    "cole_davidson": (("delta", "tau_ps", "beta"), (1.0, 1.0, 0.8), (0.0, 1e-6, 1e-3), (np.inf, np.inf, 1.0)),
    "havriliak_negami": (
        ("delta", "tau_ps", "alpha", "beta"), (1.0, 1.0, 0.8, 0.8),
        (0.0, 1e-6, 1e-3, 1e-3), (np.inf, np.inf, 1.0, 1.0),
    ),
    "lorentz": (("delta", "f0", "gamma"), (1.0, None, None), (0.0, 1e-6, 1e-6), (np.inf, np.inf, np.inf)),
    "drude": (("fp", "gamma"), (1.0, 1.0), (0.0, 1e-6), (np.inf, np.inf)),
}  # fmt: skip
GP_PARAMS = ("f_to", "g_to", "f_lo", "g_lo")


def term_value(kind: str, p, f: np.ndarray) -> np.ndarray:
    """Complex contribution (eps' - i eps'' convention) of one additive term."""
    w = 2.0 * np.pi * f
    if kind == "debye":
        d, tau = p
        return d / (1.0 + 1j * w * tau)
    if kind == "cole_cole":
        d, tau, a = p
        return d / (1.0 + (1j * w * tau) ** a)
    if kind == "cole_davidson":
        d, tau, b = p
        return d / (1.0 + 1j * w * tau) ** b
    if kind == "havriliak_negami":
        d, tau, a, b = p
        return d / (1.0 + (1j * w * tau) ** a) ** b
    if kind == "lorentz":
        d, f0, g = p
        return d * f0**2 / (f0**2 - f**2 + 1j * g * f)
    if kind == "drude":
        fp, g = p
        return -(fp**2) / (f**2 - 1j * g * f + 1e-300)
    raise ValueError(kind)


def gervais_piriou(eps_inf: float, p, f: np.ndarray) -> np.ndarray:
    """eps_inf prod_j (f_LO^2 - f^2 + i g_LO f) / (f_TO^2 - f^2 + i g_TO f)."""
    out = np.full(f.shape, eps_inf, dtype=complex)
    for ft, gt, fl, gl in np.reshape(p, (-1, 4)):
        out *= (fl**2 - f**2 + 1j * gl * f) / (ft**2 - f**2 + 1j * gt * f)
    return out


def _setup(model: str, terms: list[dict], f, eps_im):
    names, x0, lo, hi, kinds = [], [], [], [], []
    for j, t in enumerate(terms):
        if model == "gervais_piriou":
            vals = (
                t["f_to"],
                t.get("g_to", 0.05 * t["f_to"]),
                t["f_lo"],
                t.get("g_lo", 0.05 * t["f_lo"]),
            )
            for nm, v in zip(GP_PARAMS, vals):
                names.append(f"{j}:{nm}")
                x0.append(float(v))
                lo.append(1e-6)
                hi.append(np.inf)
            kinds.append(("gp", 4))
            continue
        kind = t["type"]
        pn, dflt, lb, ub = TERMS[kind]
        vals = []
        for nm, d in zip(pn, dflt):
            v = t.get(nm, d)
            if v is None and nm == "f0":
                raise ValueError("lorentz terms need an initial f0")
            if v is None and nm == "gamma":
                v = 0.05 * float(t["f0"])
            vals.append(v)
        if kind == "lorentz" and "delta" not in t:  # eps''_peak ~ delta f0 / gamma
            k = int(np.argmin(np.abs(f - vals[1])))
            vals[0] = max(float(eps_im[k]) * vals[2] / vals[1], 1e-3)
        for nm, v, a, b in zip(pn, vals, lb, ub):
            names.append(f"{j}:{kind}.{nm}")
            x0.append(float(v))
            lo.append(a)
            hi.append(b)
        kinds.append((kind, len(pn)))
    return names, np.array(x0), np.array(lo), np.array(hi), kinds


def model_eps(model, kinds, x, eps_inf, f):
    if model == "gervais_piriou":
        return gervais_piriou(eps_inf, x, f)
    out = np.full(f.shape, eps_inf, dtype=complex)
    i = 0
    for kind, n in kinds:
        out += term_value(kind, x[i : i + n], f)
        i += n
    return out


def fit(f_thz, eps_re, eps_im, eps_inf, model, terms, sigma_re=None, sigma_im=None):
    """Joint weighted least-squares fit; returns a result dict."""
    names, x0, lo, hi, kinds = _setup(model, terms, f_thz, eps_im)
    x0 = np.clip(x0, lo + 1e-12, np.where(np.isfinite(hi), hi, np.inf))
    weighted = (
        sigma_re is not None and np.all(np.isfinite(sigma_re)) and np.all(np.isfinite(sigma_im))
    )
    if weighted:
        # guard against zero error bars only (a floor tied to the peak height
        # would swamp the real errors)
        floor = max(0.01 * float(np.median(np.concatenate([sigma_re, sigma_im]))), 1e-300)
        s_re, s_im = np.maximum(sigma_re, floor), np.maximum(sigma_im, floor)
    else:
        s_re = s_im = np.ones_like(eps_re)

    def resid(x):
        m = model_eps(model, kinds, x, eps_inf, f_thz)
        return np.concatenate([(m.real - eps_re) / s_re, (-m.imag - eps_im) / s_im])

    sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=20000)
    chi2 = float(np.sum(sol.fun**2))
    n, k = sol.fun.size, len(x0)
    dof = max(n - k, 1)
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac)
        if not weighted:
            cov *= chi2 / dof
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
        corr = cov / np.outer(err, err) if np.all(err > 0) else np.full_like(cov, np.nan)
    except np.linalg.LinAlgError:
        err = np.full(k, np.nan)
        corr = np.full((k, k), np.nan)
    bic = chi2 + k * np.log(n) if weighted else n * np.log(chi2 / n) + k * np.log(n)
    m = model_eps(model, kinds, sol.x, eps_inf, f_thz)
    return {
        "model": model,
        "terms": terms,
        "names": names,
        "params": sol.x.tolist(),
        "errors": err.tolist(),
        "correlation": corr.tolist(),
        "chi2": chi2,
        "reduced_chi2": chi2 / dof,
        "weighted": bool(weighted),
        "bic": float(bic),
        "converged": bool(sol.success),
        "static": float(model_eps(model, kinds, sol.x, eps_inf, np.array([0.0])).real[0]),
        "fit_re": m.real,
        "fit_im": -m.imag,
    }


def fit_with_candidates(f_thz, eps_re, eps_im, eps_inf, fit_cfg, sigma_re=None, sigma_im=None):
    """Fit fit_cfg['terms'] and every candidate; return (best, all results)."""
    sets = [fit_cfg["terms"]] + list(fit_cfg["candidates"] or [])
    results = [
        fit(f_thz, eps_re, eps_im, eps_inf, fit_cfg["model"], t, sigma_re, sigma_im) for t in sets
    ]
    best = min(results, key=lambda r: r["bic"])
    return best, results
