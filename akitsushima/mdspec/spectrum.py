"""Permittivity spectrum from dipole fluctuations (ADR-0003).

For each Cartesian pair (a, b), with dM = M - <M> and the one-sided
correlation C_ab(t) = <dM_a(0) dM_b(t)>,

    eps_ab(w) - eps_inf_ab = [C_ab(0) - i w int_0^inf C_ab(t) win(t) e^{-i w t} dt] / (eps0 V kB T)

(conducting / tin-foil boundary; eps = eps' - i eps'').  With
A(w) = int C win cos(wt) dt and B(w) = int C win sin(wt) dt:

    eps' - eps_inf = (C(0) - w B) / (eps0 V kB T),    eps'' = w A / (eps0 V kB T).

The f-sum-normalised estimator rescales eps - eps_inf (component aa by
s_a = <Mdot_a^2>_exact / <Mdot_a^2>_MD, component ab by sqrt(s_a s_b)), with
the exact canonical value <Mdot_a Mdot_b> = kB T sum_k (Z_k Z_k^T)_ab / m_k.

The mode-resolved estimator (ADR-0003 item 10) normalises every Gamma
normal coordinate Q_m by s_m = kB T / <Qdot_m^2>_MD and builds the spectrum
from M~ = sum_m p_m sqrt(s_m) Q_m (auto terms x s_m, cross terms x
sqrt(s_m s_n)); exact for a harmonic crystal whatever the mode energies.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq

from . import forces as fr
from . import modespec
from .constants import AMU, ANGSTROM, CM1_TO_HZ, E_CHARGE, EPS0, FS, KB, MEV_TO_HZ, THZ

UNIT_SCALE = {"THz": THZ, "cm-1": CM1_TO_HZ, "meV": MEV_TO_HZ}  # Hz per unit


def correlation_tensor(dm: np.ndarray, max_lag: int) -> np.ndarray:
    """Unbiased, time-symmetrised C_ab(tau) for tau = 0..max_lag; shape (max_lag+1, 3, 3)."""
    n = len(dm)
    nfft = next_fast_len(2 * n)
    f = rfft(dm, n=nfft, axis=0)
    c = np.empty((max_lag + 1, 3, 3))
    norm = (n - np.arange(max_lag + 1)).astype(float)
    for a in range(3):
        for b in range(a, 3):
            # sum_t dm_a(t) dm_b(t + tau)
            x = irfft(np.conj(f[:, a]) * f[:, b], n=nfft)[: max_lag + 1] / norm
            if a != b:
                y = irfft(np.conj(f[:, b]) * f[:, a], n=nfft)[: max_lag + 1] / norm
                x = 0.5 * (x + y)
            c[:, a, b] = c[:, b, a] = x
    return c


def lag_window(name: str, n: int) -> np.ndarray:
    """One-sided lag window of length n (n-1 = max lag)."""
    if name == "hann":
        return 0.5 * (1.0 + np.cos(np.pi * np.arange(n) / (n - 1)))
    return np.ones(n)


def fdt_transform(c: np.ndarray, dt_s: float, window: str, nfft: int):
    """Return (freq_Hz, X', X'') with X = (eps - eps_inf) * eps0 V kB T in (e A)^2 units."""
    w = lag_window(window, len(c))
    cw = c * w[:, None, None]
    cw[0] *= 0.5  # trapezoid weight at t = 0
    spec = rfft(cw, n=nfft, axis=0) * dt_s  # sum c_n e^{-i w t_n} dt
    freq = rfftfreq(nfft, dt_s)
    om = 2.0 * np.pi * freq[:, None, None]
    a_cos, b_sin = spec.real, -spec.imag
    # Jump correction: the one-sided C(t) H(t) jumps by C(0) at t = 0, so the
    # trapezoid sine sum tends to C(0) (dt/2) cot(w dt/2) instead of C(0)/w and
    # eps' would not decay at high frequency (+0.59 C(0) at 0.8 x Nyquist).
    # Replacing C(0) by C(0) (theta/2) cot(theta/2), theta = w dt, removes the
    # aliased jump term exactly (analytic damped oscillator: eps' error
    # 1.2e-2 -> 1.5e-5 below 0.12 x Nyquist; tests/test_spectrum.py).
    th = (om * dt_s)[:, 0, 0]
    jump = np.ones_like(th)
    nz = th > 0
    jump[nz] = (th[nz] / 2.0) / np.tan(th[nz] / 2.0)
    return freq, c[0][None] * jump[:, None, None] - om * b_sin, om * a_cos


def mdot2_md(dm: np.ndarray, dt_s: float) -> np.ndarray:
    """<Mdot_a^2> per column from the spectrum (Parseval), (unit / s)^2.

    The line through the first and last values is removed first: the FFT
    treats the series as periodic, and the jump x(end) - x(0) would put a
    1/w^2 tail into |X|^2 that the w^2 weight turns into a spurious
    contribution (found 2026-10-23: +40 % for a slow 8.5 THz ice mode
    coordinate, up to +11 % for M of a 10-ps ZnO AIMD run; against central
    differences the corrected value differs only by their (w dt)^2 / 3
    attenuation).  The ramp's own derivative, (x(end) - x(0)) / T, is
    negligible for a bounded signal.
    """
    n = len(dm)
    if n > 1:
        ramp = np.linspace(0.0, 1.0, n)[:, None] if dm.ndim == 2 else np.linspace(0.0, 1.0, n)
        dm = dm - (dm[:1] + (dm[-1:] - dm[:1]) * ramp)
    f = rfft(dm, axis=0)
    om = 2.0 * np.pi * rfftfreq(n, dt_s)
    wt = np.full(len(om), 2.0)
    wt[0] = 1.0
    if n % 2 == 0:
        wt[-1] = 1.0
    return (wt[:, None] * om[:, None] ** 2 * np.abs(f) ** 2).sum(axis=0) / n**2


def mdot2_exact(z: np.ndarray, masses_amu: np.ndarray, temperature: float) -> np.ndarray:
    """Exact canonical <Mdot_a Mdot_b> (3, 3) in (e A / s)^2."""
    # SI: <Mdot^2> = kB T sum (Z e)^2 / m  [C^2 m^2 s^-2]; divide by (e A)^2
    zz = np.einsum("kab,kcb->kac", z, z) / (masses_amu * AMU)[:, None, None]
    return KB * temperature * zz.sum(axis=0) / ANGSTROM**2


def kk_real(freq: np.ndarray, eps_im: np.ndarray, chunk: int = 512) -> np.ndarray:
    """eps' - eps_inf from eps'' by Kramers-Kronig on a uniform grid starting at 0.

    Maclaurin's alternate-point formula (Ohta and Ishida, Appl. Spectrosc. 42,
    952 (1988)): eps'(w_i) - eps_inf = (4 h / pi) sum_{j - i odd} w_j eps''_j / (w_j^2 - w_i^2),
    which avoids the singular point and stays accurate for lines a few grid
    points wide (a trapezoid rule with singularity subtraction does not).
    """
    om = 2.0 * np.pi * np.asarray(freq, dtype=float)
    h = om[1] - om[0]
    g = om * eps_im
    n = len(om)
    idx = np.arange(n)
    out = np.empty(n)
    for i0 in range(0, n, chunk):
        ii = idx[i0 : i0 + chunk]
        odd = ((idx[None, :] - ii[:, None]) % 2) == 1
        den = om[None, :] ** 2 - om[ii, None] ** 2
        term = np.where(odd, g[None, :] / np.where(odd, den, 1.0), 0.0)
        out[ii] = (4.0 * h / np.pi) * term.sum(axis=1)
    return out


def drift_ratio(m: np.ndarray) -> float:
    """MSD of M at lag n/4 over lag n/16: ~1 for bounded fluctuations, ~4 for diffusion."""
    n = len(m)
    l1, l2 = max(1, n // 16), max(2, n // 4)
    msd = lambda lag: np.mean(np.sum((m[lag:] - m[:-lag]) ** 2, axis=1))
    d1 = msd(l1)
    return float(msd(l2) / d1) if d1 > 0 else 1.0


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den with 0 where den == 0 (zero charges: no ionic response)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    out = np.zeros_like(num)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


ACF_TAIL_WARN = 0.1


def acf_tail_ratio(c: np.ndarray, frac: float = 0.2) -> np.ndarray:
    """Mean of C_aa(tau) / C_aa(0) over the last `frac` of the lag window, per diagonal component.

    Near zero when the dipole autocorrelation has decayed (oscillating
    crystal correlations average out); a large value means slow relaxation
    not sampled within max_lag, e.g. molecular reorientation in a liquid.
    Measured 2026-10-21: crystals (AIMD and classical MD) <= 0.023, the
    unrelaxed 32-water AIMD 0.12-0.19 (ADR-0006).
    """
    d = np.einsum("tii->ti", c)
    tail = d[int((1 - frac) * len(d)) :].mean(axis=0)
    return np.divide(tail, d[0], out=np.zeros(3), where=d[0] > 0)


def _mode_single(qd, pm, sm, groups, dt_s, max_lag, window, nfft):
    """FDT input from M~ = sum_m p_m sqrt(s_m) Q_m without cross terms inside degenerate sets.

    Partners of a degenerate set are uncorrelated in equilibrium by symmetry;
    over a finite run a degenerate oscillation keeps its polarisation, and
    the resulting cross terms, scaled by unequal s_m, would mix the
    Cartesian components in a basis-dependent way (SiC TO triplet:
    xx / yy = 1.94 / 6.25 for 4.1; found 2026-10-23).
    """
    amp = qd * np.sqrt(sm)[None, :]
    c = correlation_tensor(amp @ pm.T, max_lag)
    for g in groups:
        c -= correlation_tensor(amp[:, g] @ pm[:, g].T, max_lag)
        for i in g:
            c += correlation_tensor(amp[:, [i]] * pm[:, i][None, :], max_lag)
    freq, xr, xi = fdt_transform(c, dt_s, window, nfft)
    return freq, xr, xi, c


def _single(dm, dt_s, max_lag, window, nfft):
    c = correlation_tensor(dm, max_lag)
    freq, xr, xi = fdt_transform(c, dt_s, window, nfft)
    return freq, xr, xi, c


def analyse(
    runs,
    z: np.ndarray,
    masses: np.ndarray | None,
    temperature: float | list[float],
    eps_inf: np.ndarray,
    spec_cfg: dict,
    log=print,
) -> dict:
    """Permittivity tensors, static values and errors for one or more runs.

    `temperature` is one value for all runs or one per run (ADR-0011 A2); it
    enters the raw estimator, the exact <Mdot^2> and the canonical <Qdot^2>,
    and cancels in the fsum and mode estimators.
    """
    temps = np.broadcast_to(np.asarray(temperature, float), (len(runs),))
    dt_s = runs[0].dt_fs * FS
    n_min = min(len(r.dipole) for r in runs)
    if spec_cfg["max_lag_ps"] == "auto":
        max_lag = n_min // 4
    else:
        max_lag = round(float(spec_cfg["max_lag_ps"]) * 1e-12 / dt_s)
    max_lag = max(2, min(max_lag, n_min - 1))
    nfft = next_fast_len(int(spec_cfg["zero_pad_factor"]) * 2 * max_lag)
    nb = int(spec_cfg["n_blocks"])
    window = spec_cfg["window"]
    exact = mdot2_exact(z, masses, temps[0]) if masses is not None else None
    proj = runs[0].info.get("modes")
    req = spec_cfg["estimator"]
    if req == "mode" and (proj is None or any(r.modes is None for r in runs)):
        raise ValueError(
            "estimator 'mode' needs Gamma modes (modes block or a DFPT OUTCAR with eigenvectors)"
        )
    use_mode = req in ("mode", "auto") and proj is not None and len(proj["freq_THz"]) > 0
    use_fsum = not use_mode and req in ("fsum", "auto") and exact is not None
    if use_fsum and proj is not None and max(proj["n_ir_per_component"]) > 1:
        warnings.warn(
            f"estimator fsum with {proj['n_ir_per_component']} IR modes per component: unequal mode "
            "energies bias it (ADR-0003 item 10); use estimator mode",
            stacklevel=2,
        )
    per_run = []
    for r, t_run in zip(runs, temps):
        q_exact = KB * t_run / (AMU * ANGSTROM**2)  # <Qdot_m^2> canonical, (sqrt(amu) A/s)^2
        if masses is not None:
            exact = mdot2_exact(z, masses, t_run)
        vol = r.volume.mean()
        if r.volume.std() / vol > 0.01:
            warnings.warn(
                f"volume fluctuates by {r.volume.std() / vol:.1%} (> 1 %); using <V>", stacklevel=2
            )
        pref = (E_CHARGE * ANGSTROM) ** 2 / (EPS0 * vol * ANGSTROM**3 * KB * t_run)
        dm = r.dipole - r.dipole.mean(axis=0)
        ratio = drift_ratio(r.dipole)
        if ratio > 2.5:
            warnings.warn(
                f"M(t) drifts diffusively (MSD ratio {ratio:.1f}): eps_s is ill-defined for a "
                "conducting system (ADR-0003 item 9)",
                stacklevel=2,
            )
        freq, xr, xi, c = _single(dm, dt_s, max_lag, window, nfft)
        tail = acf_tail_ratio(c)
        if np.abs(tail).max() > ACF_TAIL_WARN:
            warnings.warn(
                f"dipole autocorrelation has not decayed within max_lag (tail C/C(0) = "
                f"{np.round(tail, 2).tolist()} > {ACF_TAIL_WARN}): slow relaxation is not sampled "
                "and eps_s is unreliable; use longer runs or a larger max_lag (ADR-0006)",
                stacklevel=2,
            )
        md2 = mdot2_md(dm, dt_s)
        scale = np.ones((3, 3))
        fsum_ratio = None
        if exact is not None:
            fsum_ratio = _ratio(md2, np.diag(exact))
            if use_fsum:
                s = _ratio(np.diag(exact), md2)
                scale = np.sqrt(np.outer(s, s))
        mode_info, mode_static = None, None
        mode_spec = mode_lines = None
        if use_mode:
            pm = proj["p"]
            qd = r.modes - r.modes.mean(axis=0)
            qdot2 = mdot2_md(qd, dt_s)
            sm = _ratio(np.full(len(qdot2), q_exact), qdot2)
            groups = proj.get("groups", [])
            freq, xr_m, xi_m, c_m = _mode_single(qd, pm, sm, groups, dt_s, max_lag, window, nfft)
            mode_static = c_m[0] * pref
            resid = dm - qd @ pm.T
            outside = float(resid.var(axis=0).sum() / max(dm.var(axis=0).sum(), 1e-300))
            if outside > 0.01:
                warnings.warn(
                    f"{outside:.1%} of the M variance lies outside the projected Gamma modes "
                    "(supercell / site mapping / modes inconsistent?)",
                    stacklevel=2,
                )
            q2 = (qd**2).mean(axis=0)
            mode_info = {
                "freq_harmonic": proj["freq_THz"].tolist(),
                "freq_effective": (
                    np.sqrt(qdot2 / np.where(q2 > 0, q2, np.inf)) / (2 * np.pi * 1e12)
                ).tolist(),
                "kinetic_ratio": (qdot2 / q_exact).tolist(),
                "d_eps_diag": (pm**2 * (sm * q2)[None, :] * pref).tolist(),
                "variance_outside_modes": outside,
                "index": proj["index"].tolist(),
            }
            # mode-resolved spectra and line shapes (ADR-0010 F2)
            mode_spec = modespec.mode_spectra(
                qd, pm, sm, groups, dt_s, max_lag, window, nfft, pref, fdt_transform
            )
            f_eff = np.asarray(mode_info["freq_effective"]) * 1e12
            mode_lines = [modespec.line_fit(qd[:, m], dt_s, f_eff[m]) for m in range(qd.shape[1])]
        force_static, force_blocks, force_info = None, None, None
        if r.forces is not None and proj is not None and r.modes is not None:
            qf = r.modes[r.force_idx]
            force_static = fr.static_tensor(qf, r.forces, proj["p"], vol)
            force_blocks = [
                fr.static_tensor(qf[i], r.forces[i], proj["p"], vol)
                for i in np.array_split(np.arange(len(qf)), nb)
            ]
            force_info = {
                "n_frames": len(qf),
                "freq_harmonic": proj["freq_THz"].tolist(),
                "freq_force": fr.frequencies(qf, r.forces).tolist(),
            }
        # blocks (same frequency grid through the common nfft)
        blocks = []
        bq = np.array_split(r.modes, nb) if use_mode else [None] * nb
        for bm, bqm in zip(np.array_split(r.dipole, nb), bq):
            bd = bm - bm.mean(axis=0)
            lag_b = max(2, min(max_lag, len(bd) // 2))
            _, br, bi, bc = _single(bd, dt_s, lag_b, window, nfft)
            bscale = np.ones((3, 3))
            if exact is not None and use_fsum:
                sb = _ratio(np.diag(exact), mdot2_md(bd, dt_s))
                bscale = np.sqrt(np.outer(sb, sb))
            blk = {
                "re": br * pref * bscale,
                "im": bi * pref * bscale,
                "raw_s": bc[0] * pref,
                "norm_s": bc[0] * pref * bscale,
            }
            if exact is not None and not use_fsum:
                sb = _ratio(np.diag(exact), mdot2_md(bd, dt_s))
                blk["norm_s"] = bc[0] * pref * np.sqrt(np.outer(sb, sb))
            if use_mode:
                bqd = bqm - bqm.mean(axis=0)
                sbm = _ratio(np.full(bqd.shape[1], q_exact), mdot2_md(bqd, dt_s))
                _, bmr, bmi, bmc = _mode_single(
                    bqd, proj["p"], sbm, proj.get("groups", []), dt_s, lag_b, window, nfft
                )
                blk.update({"re": bmr * pref, "im": bmi * pref, "mode_s": bmc[0] * pref})
            blocks.append(blk)
        fsum_scale = scale
        if exact is not None and not use_fsum:
            sf = _ratio(np.diag(exact), md2)
            fsum_scale = np.sqrt(np.outer(sf, sf))
        per_run.append(
            {
                "freq": freq,
                "re": (xr_m if use_mode else xr * scale) * pref,
                "im": (xi_m if use_mode else xi * scale) * pref,
                "mode_static": mode_static,
                "mode_info": mode_info,
                "mode_spec": mode_spec,
                "mode_lines": mode_lines,
                "force_static": force_static,
                "force_blocks": force_blocks,
                "force_info": force_info,
                "c": c,
                "raw_static": c[0] * pref,
                "norm_static": c[0] * pref * fsum_scale,
                "fsum_ratio": fsum_ratio,
                "blocks": blocks,
                "volume": vol,
                "drift_ratio": ratio,
                "acf_tail_ratio": tail,
                "n_frames": len(r.dipole),
            }
        )

    freq = per_run[0]["freq"]

    def stack(key):
        return np.array([p[key] for p in per_run])

    def err(arrs):
        a = np.array(arrs)
        return (
            a.std(axis=0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.full(a.shape[1:], np.nan)
        )

    if len(per_run) == 1:
        warnings.warn(
            "single run: error bars are block errors of one trajectory and contain no "
            "seed-to-seed scatter, which was larger in validation (ADR-0006, ADR-0009); "
            "use independent runs (trajectory.mode: independent) for a realistic error",
            stacklevel=2,
        )
    re_mean, im_mean = stack("re").mean(axis=0), stack("im").mean(axis=0)
    if len(per_run) > 1:  # independent runs: run-to-run errors
        re_err, im_err = err(stack("re")), err(stack("im"))
        raw_err, norm_err = err(stack("raw_static")), err(stack("norm_static"))
        mode_err = err(stack("mode_static")) if use_mode else None
    else:
        b = per_run[0]["blocks"]
        re_err, im_err = err([x["re"] for x in b]), err([x["im"] for x in b])
        raw_err, norm_err = err([x["raw_s"] for x in b]), err([x["norm_s"] for x in b])
        mode_err = err([x["mode_s"] for x in b]) if use_mode else None

    mode_extra = {}
    if use_mode:

        def mstack(key):
            return np.array([p["mode_spec"][key] for p in per_run])

        tot_im = np.stack([im_mean[:, a, a] for a in range(3)], axis=1)
        tot_s = np.diag(stack("mode_static").mean(axis=0))
        m_im, m_s = mstack("im").mean(axis=0), mstack("static").mean(axis=0)
        md_im, md_s = mstack("im_diag").mean(axis=0), mstack("static_diag").mean(axis=0)
        den_s = np.where(tot_s != 0, tot_s, np.inf)
        den_im = np.maximum(np.abs(tot_im).sum(axis=0), 1e-300)
        mode_extra = {
            "mode_eps_re": mstack("re").mean(axis=0),
            "mode_eps_im": m_im,
            "mode_eps_im_diag": md_im,
            "mode_table": modespec.table(
                [p["mode_info"] for p in per_run],
                [p["mode_lines"] for p in per_run],
                [p["mode_spec"]["static"] for p in per_run],
                [p["mode_spec"]["static_diag"] for p in per_run],
            ),
            "mode_residual": {
                # split decomposition vs the estimator total: round-off only
                "split_static": ((tot_s - m_s.sum(axis=0)) / den_s).tolist(),
                "split_spectral": (np.abs(tot_im - m_im.sum(axis=1)).sum(axis=0) / den_im).tolist(),
                # share of the cross terms between different modes (ADR-0010 F2)
                "cross_share_static": ((tot_s - md_s.sum(axis=0)) / den_s).tolist(),
                "cross_share_spectral": (
                    np.abs(tot_im - md_im.sum(axis=1)).sum(axis=0) / den_im
                ).tolist(),
            },
        }
    use_force = all(p["force_static"] is not None for p in per_run)
    force_err = None
    if use_force:
        force_err = (
            err(stack("force_static")) if len(per_run) > 1 else err(per_run[0]["force_blocks"])
        )
    eps_re = eps_inf[None] + re_mean
    out = {
        "freq_hz": freq,
        "eps_re": eps_re,
        "eps_im": im_mean,
        "eps_re_err": re_err,
        "eps_im_err": im_err,
        "eps_inf": eps_inf,
        "static_raw": eps_inf + stack("raw_static").mean(axis=0),
        "static_raw_err": raw_err,
        "static_fsum": eps_inf + stack("norm_static").mean(axis=0) if exact is not None else None,
        "static_fsum_err": norm_err if exact is not None else None,
        "fsum_ratio": [
            p["fsum_ratio"].tolist() if p["fsum_ratio"] is not None else None for p in per_run
        ],
        "static_mode": eps_inf + stack("mode_static").mean(axis=0) if use_mode else None,
        "static_mode_err": mode_err,
        "modes": [p["mode_info"] for p in per_run] if use_mode else None,
        "static_force": eps_inf + stack("force_static").mean(axis=0) if use_force else None,
        "static_force_err": force_err,
        "force_modes": [p["force_info"] for p in per_run] if use_force else None,
        **mode_extra,
        "estimator": "mode" if use_mode else ("fsum" if use_fsum else "raw"),
        "acf": stack("c").mean(axis=0),
        "dt_s": dt_s,
        "max_lag": max_lag,
        "nfft": nfft,
        "volume_A3": float(np.mean([p["volume"] for p in per_run])),
        "n_frames": [p["n_frames"] for p in per_run],
        "drift_ratio": [p["drift_ratio"] for p in per_run],
        "acf_tail_ratio": [p["acf_tail_ratio"].tolist() for p in per_run],
        "temperature": float(temps.mean()),
    }
    if spec_cfg["kk_check"]:
        lim = freq <= 0.8 * freq[-1]
        iso_im = np.trace(im_mean, axis1=1, axis2=2) / 3
        iso_re = np.trace(re_mean, axis1=1, axis2=2) / 3
        kk = kk_real(freq, iso_im)
        out["kk_iso"] = kk + np.trace(eps_inf) / 3
        dev = kk[lim] - iso_re[lim]
        out["kk_rms_dev_rel"] = float(
            np.sqrt(np.mean(dev**2) / max(np.mean(iso_re[lim] ** 2), 1e-30))
        )
    return out


def iso(t: np.ndarray) -> np.ndarray:
    """Isotropic value (trace / 3) of (..., 3, 3) tensors."""
    return np.trace(t, axis1=-2, axis2=-1) / 3.0
