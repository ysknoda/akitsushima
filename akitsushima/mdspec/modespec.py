"""Mode-resolved spectra and line shapes (ADR-0010, F2; mode estimator only).

With amp_m = sqrt(s_m) dQ_m (the mode-estimator scaling) and the
time-symmetrised correlations C_mn(t), the estimator spectrum is
    Delta eps_aa = sum_m sum_n' p_am p_an FDT[C_mn] / (eps0 V kB T),
n' running over all modes except the other partners of m's degenerate
set (their cross terms are dropped by the estimator, ADR-0003 item 13).
Two decompositions are reported per mode m:
  * split (default, exact): Delta eps_m,aa = p_am sum_n' p_an FDT[C_mn],
    every cross term shared equally between its two modes, so that
    sum_m Delta eps_m,aa equals the estimator spectrum to round-off; a
    mode may come out negative where its cross terms are negative;
  * diagonal only: p_am^2 FDT[C_mm], which omits the cross terms between
    different modes (the 'cross share' says how much they carry: 0 for
    one IR mode or one degenerate set per component, 3-10 % in urea where
    the AIMD surface's modes are rotated against the DFPT eigenvectors).
The peak frequency and linewidth of every mode come from a damped-
harmonic-oscillator fit to the periodogram of Qdot_m over the whole run,
    S(f) = A gamma f^2 / ((f0^2 - f^2)^2 + gamma^2 f^2),
gamma = FWHM (Hz) for weak damping, lifetime 1 / (2 pi gamma).  The
periodogram resolves df = 1 / T_run; a fitted gamma below 2 df is
flagged as unresolved (an upper bound).
"""

from __future__ import annotations

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize

RESOLVED_BINS = 2.0  # gamma must exceed this many periodogram bins to count as resolved


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Unbiased autocorrelation of every column, shape (max_lag + 1, n_cols)."""
    n = len(x)
    nfft = next_fast_len(2 * n)
    f = rfft(x, n=nfft, axis=0)
    c = irfft(np.abs(f) ** 2, n=nfft, axis=0)[: max_lag + 1]
    return c / (n - np.arange(max_lag + 1))[:, None]


def mode_correlations(qd, pm, sm, groups, max_lag):
    """Per-mode correlations (lags, n_modes, 3): split (cross terms halved) and diagonal only."""
    n, nm = qd.shape
    nfft = next_fast_len(2 * n)
    amp = qd * np.sqrt(sm)[None, :]
    fa = rfft(amp, n=nfft, axis=0)  # (nf, n_modes)
    fx = fa @ pm.T  # (nf, 3): FFT of M~_a = sum_n p_an amp_n
    partner = {int(i): np.setdiff1d(g, [i]) for g in groups for i in g}
    norm = (n - np.arange(max_lag + 1)).astype(float)
    split = np.empty((max_lag + 1, nm, 3))
    for m in range(nm):
        fy = fx
        if m in partner:  # the estimator drops m's degenerate partners
            fy = fx - fa[:, partner[m]] @ pm[:, partner[m]].T
        # time-symmetrised <amp_m(0) Y_a(t)>: irfft of the real part of the cross spectrum
        cr = irfft((np.conj(fa[:, m])[:, None] * fy).real, n=nfft, axis=0)[: max_lag + 1]
        split[:, m, :] = cr / norm[:, None] * pm[:, m][None, :]
    auto = irfft(np.abs(fa) ** 2, n=nfft, axis=0)[: max_lag + 1] / norm[:, None]
    diag = auto[:, :, None] * (pm.T**2)[None]
    return split, diag


def mode_spectra(qd, pm, sm, groups, dt_s, max_lag, window, nfft, pref, fdt_transform):
    """(re, im) of the split and the diagonal-only decomposition, each (nfreq, n_modes, 3)."""
    split, diag = mode_correlations(qd, pm, sm, groups, max_lag)
    _, sr, si = fdt_transform(split, dt_s, window, nfft)
    _, dr, di = fdt_transform(diag, dt_s, window, nfft)
    return {
        "re": sr * pref,
        "im": si * pref,
        "re_diag": dr * pref,
        "im_diag": di * pref,
        "static": split[0] * pref,  # (n_modes, 3)
        "static_diag": diag[0] * pref,
    }


def _dho(f, a, f0, g):
    return a * g * f**2 / ((f0**2 - f**2) ** 2 + (g * f) ** 2)


def line_fit(q: np.ndarray, dt_s: float, f_guess_hz: float) -> dict:
    """Peak frequency and FWHM [Hz] of one mode from its Qdot periodogram.

    Whittle maximum likelihood (periodogram ordinates are exponentially
    distributed about the spectrum, so least squares is dominated by noise
    spikes): minimise sum(log S + P / S) with S = DHO + constant background,
    over 0.6-1.4 x the guessed frequency.
    """
    n = len(q)
    out = {"f_peak": None, "fwhm": None, "resolved": None}
    if n < 16 or not f_guess_hz > 0:
        return out
    q = q - (q[0] + (q[-1] - q[0]) * np.linspace(0.0, 1.0, n))  # no end-to-start jump
    f = rfftfreq(n, dt_s)
    p = (2 * np.pi * f) ** 2 * np.abs(rfft(q)) ** 2 / n
    df = f[1]
    win = np.flatnonzero((f > 0.6 * f_guess_hz) & (f < 1.4 * f_guess_hz))
    if len(win) < 8:
        return out
    fw, pw = f[win], p[win]
    ps = uniform_filter1d(pw, max(7, int(0.01 * f_guess_hz / df)))
    ip = int(np.argmax(ps))
    half = ps[ip] / 2
    lo, hi = ip, ip
    while lo > 0 and ps[lo] > half:
        lo -= 1
    while hi < len(ps) - 1 and ps[hi] > half:
        hi += 1
    g0 = max((hi - lo) * df, df)
    bg0 = max(float(np.median(pw)) * 0.1, 1e-300)

    def nll(theta):
        a, f0, g, b = np.exp(theta[0]), theta[1], np.exp(theta[2]), np.exp(theta[3])
        sm = _dho(fw, a, f0, g) + b
        return float(np.sum(np.log(sm) + pw / sm))

    k = max(7, int(0.01 * f_guess_hz / df))
    near = slice(max(0, ip - k), ip + k + 1)
    ip_raw = near.start + int(np.argmax(pw[near]))  # sharp lines: start on the raw maximum
    x0 = np.array([np.log(ps[ip] * g0), fw[ip_raw], np.log(g0), np.log(bg0)])
    bounds = [
        (None, None),
        (fw[0], fw[-1]),
        (np.log(0.1 * df), np.log(fw[-1] - fw[0])),
        (np.log(bg0) - 30, np.log(float(pw.max()))),
    ]
    best = None
    for gs in (1.0, 0.1, 10.0):  # a few starting widths against local minima
        x = x0.copy()
        x[0] += np.log(gs)
        x[2] = np.clip(x[2] + np.log(gs), bounds[2][0], bounds[2][1])
        r = minimize(nll, x, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    f0, g = float(best.x[1]), float(np.exp(best.x[2]))
    if not (fw[0] < f0 < fw[-1]):
        return out
    # 1-sigma errors from the curvature of the Whittle likelihood (finite differences) in
    # (log A, f0, log gamma); the background is held fixed (often at its flat lower end)
    h = np.array([1e-3, max(1e-3 * g, 1e-6 * f0), 1e-3])
    hess = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            ei = np.append(np.eye(3)[i] * h[i], 0.0)
            ej = np.append(np.eye(3)[j] * h[j], 0.0)
            hess[i, j] = (
                nll(best.x + ei + ej)
                - nll(best.x + ei - ej)
                - nll(best.x - ei + ej)
                + nll(best.x - ei - ej)
            ) / (4 * h[i] * h[j])
    f0_err = g_err = None
    try:
        cov = np.linalg.inv(hess)
        if cov[1, 1] > 0 and cov[2, 2] > 0:
            f0_err, g_err = float(np.sqrt(cov[1, 1])), float(g * np.sqrt(cov[2, 2]))
    except np.linalg.LinAlgError:
        pass
    out.update(
        {
            "f_peak": f0,
            "fwhm": g,
            "f_peak_err": f0_err,
            "fwhm_err": g_err,
            "resolved": bool(g > RESOLVED_BINS * df),
        }
    )
    return out


def table(
    mode_infos: list[dict],
    lines: list[list[dict]],
    d_eps: list[np.ndarray],
    d_eps_diag: list[np.ndarray],
) -> list[dict]:
    """One row per kept mode, averaged over runs (THz), with the run-to-run scatter of the width."""
    rows = []
    n = len(mode_infos[0]["freq_harmonic"])
    for m in range(n):
        lm = [ln[m] for ln in lines]
        fp = [x["f_peak"] for x in lm if x["f_peak"] is not None]
        fw = [x["fwhm"] for x in lm if x["fwhm"] is not None]
        rows.append(
            {
                "index": int(mode_infos[0]["index"][m]),
                "freq_harmonic": mode_infos[0]["freq_harmonic"][m],
                "freq_effective": float(np.mean([mi["freq_effective"][m] for mi in mode_infos])),
                "freq_peak": float(np.mean(fp)) / 1e12 if fp else None,
                "linewidth_fwhm": float(np.mean(fw)) / 1e12 if fw else None,
                "linewidth_scatter": float(np.std(fw, ddof=1)) / 1e12 if len(fw) > 1 else None,
                "linewidth_fit_err": (
                    float(np.sqrt(np.mean([x["fwhm_err"] ** 2 for x in lm if x.get("fwhm_err")])))
                    / 1e12
                    / np.sqrt(len(fw))
                    if any(x.get("fwhm_err") for x in lm)
                    else None
                ),
                "lifetime_ps": float(1e12 / (2 * np.pi * np.mean(fw))) if fw else None,
                "resolved": all(x["resolved"] for x in lm if x["resolved"] is not None),
                "d_eps": np.mean([d[m] for d in d_eps], axis=0).tolist(),
                "d_eps_diag_only": np.mean([d[m] for d in d_eps_diag], axis=0).tolist(),
            }
        )
    return rows
