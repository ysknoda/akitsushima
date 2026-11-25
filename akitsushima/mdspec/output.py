"""Output files (ADR-0007): text tables, JSON summary, npz and figure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COMP = (("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2))


def _iso(t):
    return np.trace(t, axis1=-2, axis2=-1) / 3.0


def write_eps(path: Path, freq, unit: str, res: dict) -> None:
    cols, names = [freq], [f"freq_{unit}"]
    for lab, i, j in COMP:
        cols += [res["eps_re"][:, i, j], res["eps_im"][:, i, j]]
        names += [f"eps_re_{lab}", f"eps_im_{lab}"]
    cols += [
        _iso(res["eps_re"]),
        _iso(res["eps_im"]),
        _iso(res["eps_re_err"]),
        _iso(res["eps_im_err"]),
    ]
    names += ["eps_re_iso", "eps_im_iso", "err_re_iso", "err_im_iso"]
    np.savetxt(path, np.column_stack(cols), header=" ".join(names), fmt="%.10e")


def write_modes(path: Path, freq, unit: str, res: dict) -> None:
    """eps''_m,aa(omega) of every kept Gamma mode (ADR-0010 F2): split (cross terms halved,
    sums to the total) and diagonal-only columns per mode and component."""
    cols, names = [freq], [f"freq_{unit}"]
    for key, tag in (("mode_eps_im", ""), ("mode_eps_im_diag", "_diag")):
        for k, row in enumerate(res["mode_table"]):
            for a, lab in enumerate(("xx", "yy", "zz")):
                cols.append(res[key][:, k, a])
                names.append(f"m{row['index']}_{lab}{tag}")
    np.savetxt(path, np.column_stack(cols), header=" ".join(names), fmt="%.8e")


def write_acf(path: Path, res: dict) -> None:
    c = res["acf"]
    lag_ps = np.arange(len(c)) * res["dt_s"] * 1e12
    cols = [lag_ps] + [c[:, i, j] for _, i, j in COMP] + [_iso(c)]
    np.savetxt(
        path,
        np.column_stack(cols),
        header="lag_ps C_xx C_yy C_zz C_iso  [(e*Angstrom)^2]",
        fmt="%.10e",
    )


def write_dipole(path: Path, run) -> None:
    t_ps = np.arange(len(run.dipole)) * run.dt_fs * 1e-3
    np.savetxt(
        path,
        np.column_stack([t_ps, run.dipole]),
        header="time_ps Mx My Mz [e*Angstrom]",
        fmt="%.10e",
    )


def to_jsonable(x):
    """Plain Python types for json; NaN and infinities become null (strict JSON, ADR-0011 A4)."""
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_json(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(to_jsonable(summary), indent=1, allow_nan=False))


def plot(
    path: Path,
    run,
    freq,
    unit,
    res,
    fit_res=None,
    fit_comp="iso",
    freq_max=None,
    optics=None,
    n_runs: int = 1,
) -> None:
    """Overview figure; panel (a) shows the first run, (c) / (d) the iso error band (ADR-0011 A3)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = 2 if optics is None else 3
    fig, ax = plt.subplots(rows, 2, figsize=(12, 4.25 * rows), constrained_layout=True)
    t_ps = np.arange(len(run.dipole)) * run.dt_fs * 1e-3
    step = max(1, len(t_ps) // 20000)
    for k, lab in enumerate("xyz"):
        ax[0, 0].plot(
            t_ps[::step], run.dipole[::step, k] - run.dipole[:, k].mean(), lw=0.6, label=f"M{lab}"
        )
    title_a = "(a) total dipole" + (f" (run 1 of {n_runs})" if n_runs > 1 else "")
    ax[0, 0].set(xlabel="time (ps)", ylabel="M - <M> (e Angstrom)", title=title_a)
    ax[0, 0].legend(loc="upper right")
    c = _iso(res["acf"])
    ax[0, 1].plot(np.arange(len(c)) * res["dt_s"] * 1e12, c / c[0], color="k", lw=1)
    ax[0, 1].axhline(0, color="0.6", lw=0.8)
    ax[0, 1].set(xlabel="lag (ps)", ylabel="C(t) / C(0)", title="(b) dipole autocorrelation (iso)")
    sel = freq <= (freq_max if freq_max else freq[-1])
    colors = {"xx": "C0", "yy": "C1", "zz": "C2", "iso": "k"}
    for lab, i, j in COMP:
        ax[1, 0].plot(freq[sel], res["eps_im"][sel, i, j], lw=0.8, color=colors[lab], label=lab)
        ax[1, 1].plot(freq[sel], res["eps_re"][sel, i, j], lw=0.8, color=colors[lab], label=lab)
    for a, key in ((ax[1, 0], "eps_im"), (ax[1, 1], "eps_re")):
        y, e = _iso(res[key])[sel], _iso(res[f"{key}_err"])[sel]
        a.plot(freq[sel], y, lw=1.4, color="k", label="iso")
        if np.isfinite(e).all():  # same error as the err_*_iso columns of _eps.dat
            a.fill_between(freq[sel], y - e, y + e, color="k", alpha=0.2, lw=0, label="iso error")
    if "kk_iso" in res:
        ax[1, 1].plot(freq[sel], res["kk_iso"][sel], "--", color="0.5", lw=1, label="KK(iso)")
    if fit_res is not None:
        ff = fit_res["freq"]
        ax[1, 0].plot(ff, fit_res["fit_im"], "r--", lw=1.2, label=f"fit ({fit_comp})")
        ax[1, 1].plot(ff, fit_res["fit_re"], "r--", lw=1.2, label=f"fit ({fit_comp})")
    ax[1, 0].set(xlabel=f"frequency ({unit})", ylabel="eps''", title="(c) imaginary part")
    ax[1, 1].set(xlabel=f"frequency ({unit})", ylabel="eps'", title="(d) real part")
    if optics is not None:  # ADR-0010 F1
        for k, lab in enumerate(("xx", "yy", "zz", "iso")):
            kw = {"lw": 1.4 if lab == "iso" else 0.8, "color": colors[lab], "label": lab}
            ax[2, 0].plot(freq[sel], optics["R"][sel, k], **kw)
            ax[2, 1].plot(freq[sel], optics["loss"][sel, k], **kw)
        ax[2, 0].set(
            xlabel=f"frequency ({unit})", ylabel="R", title="(e) reflectivity", ylim=(0, 1)
        )
        ax[2, 1].set(
            xlabel=f"frequency ({unit})", ylabel="-Im(1/eps)", title="(f) loss function (LO)"
        )
    for a in ax[1:].ravel():
        a.legend(fontsize=8)
        a.grid(ls=":", lw=0.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)
