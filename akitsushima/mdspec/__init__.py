"""akitsushima.mdspec: spectra from MD trajectories -- complex permittivity
via the fluctuation-dissipation theorem (ADR-0050).

Ported from nozuchi 0.2.0a2 (commit 84dec4c of
/Users/yusuke/Claude_Code/dielectric_analysis) without behaviour changes
(ADR-0051): ``run`` takes the nozuchi-format configuration; the akitsushima
YAML (``born:`` + ``mdspec:``) is translated by ``mdspec.yaml_io``.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np

from .. import __version__  # noqa: E402  (akitsushima's own version)

NOZUCHI_BASE = "0.2.0a2"  # the nozuchi release this port follows

from . import charges as ch
from . import config as _config
from . import fitting, optics, output, spectrum
from .dipole import compute_runs


def _eps_inf(value) -> np.ndarray:
    if isinstance(value, dict):
        return ch.read_outcar_eps_inf(value["outcar"])
    return ch._tensor(value)


def _source_temperatures(src: str, n_frames: int, n_skip: int, every: int) -> np.ndarray:
    """Instantaneous temperatures of one source over the kept frames of its trajectory file.

    n_frames frames were read from the file and its first n_skip are skipped.
    OUTCAR: one value per MD step, frame j (0-based) at step (j + 1) * every.
    thermo.log: aligned to the end of the log (the analysed frames are the last ones).
    """
    if ch.is_thermo_log(src):
        steps, tt = ch.read_thermo_log(src)
        return tt[steps > steps[-1] - (n_frames - n_skip) * every]
    return ch.read_outcar_temperatures(src)[n_skip * every : n_frames * every]


def _run_temperatures(cfg: dict, runs, log) -> list[float]:
    """Temperature of every analysed run [K] (ADR-0011 A2).

    A number in the YAML applies to all runs.  With `auto` and one
    temperature_source per trajectory file (same order), each run averages
    the sources of its own files over its kept frames; otherwise one mean
    over all sources is used for every run (warned for independent runs).
    """
    t = cfg["temperature"]
    if t != "auto":
        return [float(t)] * len(runs)
    tr, srcs = cfg["trajectory"], cfg["temperature_source"]
    every, skip = int(tr["frame_every"]), int(tr["skip_frames"])
    if len(srcs) == len(tr["files"]) and all("file_frames" in r.info for r in runs):
        out, k = [], 0
        for i, r in enumerate(runs):
            vals, left = [], skip
            for n in r.info["file_frames"]:
                drop = min(left, n)
                left -= drop
                vals.append(_source_temperatures(srcs[k], n, drop, every))
                k += 1
            allt = np.concatenate(vals)
            if not len(allt):
                raise ValueError(f"temperature: auto found no temperatures for run {i + 1}")
            out.append(float(allt.mean()))
            log(
                f"[temperature] auto, run {i + 1}: {allt.mean():.2f} K from {len(allt)} values "
                f"(std {allt.std():.1f} K)"
            )
        return out
    if len(runs) > 1:
        warnings.warn(
            f"temperature: auto with {len(srcs)} temperature_source entries for "
            f"{len(tr['files'])} trajectory files: one mean temperature is used for every run "
            "(give one source per file for per-run temperatures, ADR-0011)",
            stacklevel=2,
        )
    return [_temperature(cfg, len(runs[0].dipole), log)] * len(runs)


def _temperature(cfg: dict, n_frames: int, log) -> float:
    """One mean temperature over all sources (0.2.0a1 behaviour, used as the fallback)."""
    tr = cfg["trajectory"]
    every, skip, stride = int(tr["frame_every"]), int(tr["skip_frames"]), int(tr["stride"])
    temps = []
    for src in cfg["temperature_source"]:
        if ch.is_thermo_log(src):
            steps, tt = ch.read_thermo_log(src)
            # align to the end of the run: the analysed frames are the last ones
            first_step = steps[-1] - n_frames * every * stride
            temps.append(tt[steps > first_step])
        else:
            temps.append(ch.read_outcar_temperatures(src))
    if not any(len(x) for x in temps):
        raise ValueError("temperature: auto found no temperatures in temperature_source")
    if not ch.is_thermo_log(cfg["temperature_source"][0]):
        allt = np.concatenate(temps)[skip * every :]
    else:
        allt = np.concatenate(temps)
    log(f"[temperature] auto: {allt.mean():.2f} K from {len(allt)} values (std {allt.std():.1f} K)")
    return float(allt.mean())


def run(cfg: dict | str | Path, log=print) -> dict:
    """Run the full analysis for a validated config (or a YAML path); return the summary."""
    if not isinstance(cfg, dict):
        cfg = _config.load(cfg)
    t0 = time.perf_counter()
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / cfg["output"]["prefix"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runs, z, header = compute_runs(cfg, log)
        temps = _run_temperatures(cfg, runs, log)
        temperature = float(np.mean(temps))
        masses, msrc = None, None
        if cfg["dipole"]["model"] != "external":
            masses, msrc, outcars = ch.run_masses(cfg, header)
            if (
                not outcars
                and not cfg["masses"]
                and not header.comment.lower().startswith("akitsushima")
            ):
                warnings.warn(
                    "no OUTCAR or masses given: standard atomic weights used for the f-sum "
                    "normalisation (VASP POMASS may differ, ADR-0002)",
                    stacklevel=2,
                )
        eps_inf = _eps_inf(cfg["epsilon_inf"])
        res = spectrum.analyse(runs, z, masses, temps, eps_inf, cfg["spectrum"], log)

        unit = cfg["spectrum"]["freq_unit"]
        freq = res["freq_hz"] / spectrum.UNIT_SCALE[unit]
        fmax = cfg["spectrum"]["freq_max"]
        keep = freq <= float(fmax) if fmax else np.ones(len(freq), bool)
        for k in (
            "eps_re",
            "eps_im",
            "eps_re_err",
            "eps_im_err",
            "kk_iso",
            "mode_eps_im",
            "mode_eps_im_diag",
        ):
            if k in res:
                res[k] = res[k][keep]
        freq = freq[keep]

        fit_out, fit_plot, fit_all = None, None, None
        if cfg["fit"] is not None:
            fc = cfg["fit"]
            comp = fc["use_component"]
            ij = {"xx": 0, "yy": 1, "zz": 2}.get(comp)

            def pick(t, ij=ij):
                return spectrum.iso(t) if ij is None else t[:, ij, ij]

            f_thz = freq * spectrum.UNIT_SCALE[unit] / 1e12
            lo, hi = fc["freq_range"] or (freq[1], freq[-1])
            sel = (freq >= lo) & (freq <= hi)
            einf = float(spectrum.iso(eps_inf)) if ij is None else float(eps_inf[ij, ij])
            best, fit_all = fitting.fit_with_candidates(
                f_thz[sel],
                pick(res["eps_re"])[sel],
                pick(res["eps_im"])[sel],
                einf,
                fc,
                pick(res["eps_re_err"])[sel],
                pick(res["eps_im_err"])[sel],
            )
            fit_plot = {"freq": freq[sel], "fit_re": best["fit_re"], "fit_im": best["fit_im"]}
            fit_out = {k: v for k, v in best.items() if k not in ("fit_re", "fit_im")}
            np.savetxt(
                f"{prefix}_fit.dat",
                np.column_stack([freq[sel], best["fit_re"], best["fit_im"]]),
                header=f"freq_{unit} eps_re_fit eps_im_fit  ({comp}, {best['model']})",
                fmt="%.10e",
            )
            log(f"[fit] {comp}: BIC {best['bic']:.2f}, reduced chi2 {best['reduced_chi2']:.3f}")
            for nm, v, e in zip(best["names"], best["params"], best["errors"]):
                log(f"      {nm:28s} {v:12.6g} +/- {e:.3g}")

        est = res["estimator"]
        opt_q, opt_sum = optics.analyse(
            freq,
            freq * spectrum.UNIT_SCALE[unit],
            res["eps_re"],
            res["eps_im"],
            res[f"static_{est}"],
            eps_inf,
        )
        optics.write(Path(f"{prefix}_optics.dat"), freq, unit, opt_q)
        output.write_eps(Path(f"{prefix}_eps.dat"), freq, unit, res)
        if "mode_eps_im" in res:
            output.write_modes(Path(f"{prefix}_modes.dat"), freq, unit, res)
        output.write_acf(Path(f"{prefix}_acf.dat"), res)
        # one dipole file per run when there are several (ADR-0011 A3)
        if len(runs) == 1:
            output.write_dipole(Path(f"{prefix}_dipole.dat"), runs[0])
        else:
            for i, r in enumerate(runs, start=1):
                output.write_dipole(Path(f"{prefix}_dipole_run{i}.dat"), r)
        np.savez(
            f"{prefix}.npz",
            freq=freq,
            eps_re=res["eps_re"],
            eps_im=res["eps_im"],
            eps_re_err=res["eps_re_err"],
            eps_im_err=res["eps_im_err"],
            acf=res["acf"],
            dipole=runs[0].dipole,  # run 1, kept for compatibility
            n_runs=len(runs),
            **{f"dipole_run{i}": r.dipole for i, r in enumerate(runs, start=1)},
        )
        # akitsushima's optional-matplotlib convention: without it the
        # figure is skipped with one warning (ADR-0051), not an ImportError.
        from ..utils.optional_matplotlib import have_matplotlib

        if cfg["output"]["plot"] and have_matplotlib():
            output.plot(
                Path(f"{prefix}_dielectric.png"),
                runs[0],
                freq,
                unit,
                res,
                fit_plot,
                cfg["fit"]["use_component"] if cfg["fit"] else "iso",
                optics=opt_q,
                n_runs=len(runs),
            )

    est = res["estimator"]
    static = res[f"static_{est}"]
    static_err = res[f"static_{est}_err"]
    summary = {
        "akitsushima_version": __version__,
        "nozuchi_base": NOZUCHI_BASE,
        "estimator": est,
        "eps_static": static,
        "eps_static_err": static_err,
        "eps_static_iso": float(spectrum.iso(static)),
        "eps_static_iso_err": float(np.sqrt(np.sum(np.diag(static_err) ** 2)) / 3),
        "eps_static_raw": res["static_raw"],
        "eps_static_raw_err": res["static_raw_err"],
        "eps_static_fsum": res["static_fsum"],
        "eps_static_fsum_err": res["static_fsum_err"],
        "eps_static_mode": res["static_mode"],
        "eps_static_mode_err": res["static_mode_err"],
        "modes": res["modes"],
        "eps_static_force": res["static_force"],
        "eps_static_force_err": res["static_force_err"],
        "force_modes": res["force_modes"],
        "optics": opt_sum,
        "mode_table": res.get("mode_table"),
        "mode_residual": res.get("mode_residual"),
        "fsum_ratio_md_over_exact": res["fsum_ratio"],
        "eps_inf": eps_inf,
        "temperature_K": temperature,
        "temperature_K_per_run": temps,
        "volume_A3": res["volume_A3"],
        "n_frames": res["n_frames"],
        "dt_fs": runs[0].dt_fs,
        "max_lag_ps": res["max_lag"] * res["dt_s"] * 1e12,
        "freq_resolution": 1.0 / (res["nfft"] * res["dt_s"]) / spectrum.UNIT_SCALE[unit],
        "freq_unit": unit,
        "drift_ratio": res["drift_ratio"],
        "acf_tail_ratio": res["acf_tail_ratio"],
        "kk_rms_dev_rel": res.get("kk_rms_dev_rel"),
        "masses_source": msrc,
        "fit": fit_out,
        "fit_candidates_bic": [r["bic"] for r in fit_all] if fit_all else None,
        "warnings": [str(w.message) for w in caught],
        "config": cfg,
        "wall_time_s": time.perf_counter() - t0,
    }
    for w in caught:
        log(f"[warning] {w.message}")
    output.write_json(Path(f"{prefix}_results.json"), summary)
    iso_s, iso_e = summary["eps_static_iso"], summary["eps_static_iso_err"]
    log(
        f"[result] eps_s (iso, {est}) = {iso_s:.4f} +/- {iso_e:.4f}; diag "
        f"{np.round(np.diag(static), 4).tolist()}"
    )
    if res["static_force"] is not None:
        log(
            f"[result] force-based eps_s diag = {np.round(np.diag(res['static_force']), 4).tolist()}"
            f" +/- {np.round(np.diag(res['static_force_err']), 4).tolist()} (reported, ADR-0009)"
        )
    if res["fsum_ratio"][0] is not None:
        log(
            f"[result] f-sum ratio <Mdot^2>_MD / exact = {np.round(res['fsum_ratio'][0], 3).tolist()}"
        )
    log(f"[done] outputs in {out_dir} ({summary['wall_time_s']:.1f} s)")
    return summary
