"""`nozuchi check`: quality diagnostics of the input runs (ADR-0010, F3).

Reads the same configuration as `mdspec.run` and reports, per item, OK / WARN / INFO
with the measured value and the threshold (justified in ADR-0010), without
writing permittivity files:

  temperature   instantaneous T: mean vs the INCAR TEBEG (VASP), std vs the
                canonical kinetic std T sqrt(2 / 3N), change over the run
  energy        conserved-energy drift (OSZICAR "E=" or akitsushima
                thermo.log E_tot) relative to 3 kB T per atom over the run
  rigid drift   centre-of-mass motion during the run, and the rigid offset
                removed by the site mapping
  equipartition mean kinetic energy of the IR Gamma modes, <Qdot_m^2> / kB T
                (INFO on the spread: a canonical draw spreads like Exp(1))
  relaxation    dipole ACF tail and diffusive M(t) drift
  statistics    relative error of eps_s per component and the run length /
                number of runs needed for a target relative error
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np

from . import charges as ch
from . import spectrum
from .constants import EV, KB
from .dipole import RunStream, compute_runs, remove_reference
from .trajectory import iter_chunks, read_header

# thresholds (ADR-0010; values measured on the validation runs, ADR-0006)
T_STD_RATIO = (0.4, 1.5)  # NVE harmonic ~0.71; Nose-Hoover SMASS = 0 NVT ~2.5
T_CHANGE = 0.05  # |T_end - T_start| / T over the run (urea 18 %, ZnO 8 %)
E_DRIFT = 0.05  # energy drift over the run / (3 kB T per atom)
COM_DRIFT_A = 0.1  # centre-of-mass motion during the run
KIN_MEAN_SIGMA = 3.0  # |mean(<Qdot^2> / kT) - 1| sqrt(n_modes)
T_TARGET_SIGMA = 3.0  # |<T> / TEBEG - 1| in units of the canonical 1 / sqrt(3N)
ACF_TAIL = 0.1
DRIFT_RATIO = 2.5


def _item(name, status, value, threshold, note=""):
    return {"item": name, "status": status, "value": value, "threshold": threshold, "note": note}


def _temperatures(cfg):
    out = []
    for src in cfg["temperature_source"] or []:
        if ch.is_thermo_log(src):
            _, t = ch.read_thermo_log(src)
            out.append((src, t))
        else:
            out.append((src, ch.read_outcar_temperatures(src)))
    return out


def _tebeg(outcar) -> float | None:
    """TEBEG of the INCAR next to a VASP OUTCAR, if any."""
    inc = Path(outcar).with_name("INCAR")
    if ch.is_thermo_log(outcar) or not inc.exists():
        return None
    m = re.search(r"^\s*TEBEG\s*=\s*([\d.]+)", inc.read_text(), re.MULTILINE)
    return float(m.group(1)) if m else None


def read_energy(path) -> np.ndarray:
    """Conserved energy per MD step [eV]: OSZICAR 'E=' or thermo.log E_tot (every logged step)."""
    path = Path(path)
    if ch.is_thermo_log(path):
        with open(path) as fh:
            head = fh.readline().split()
        col = next((i for i, h in enumerate(head) if h.startswith("E_tot")), None)
        if col is None:
            raise ValueError(f"{path}: no E_tot column")
        return np.loadtxt(path, skiprows=1, ndmin=2)[:, col]
    rx = re.compile(r"\sE=\s*([-+.\dEe]+)")
    vals = []
    with open(path) as fh:
        for ln in fh:
            if " T=" in ln:
                m = rx.search(ln)
                if m:
                    vals.append(float(m.group(1)))
    return np.array(vals)


def _energy_files(cfg, given):
    if given:
        return list(given)
    found = []
    for src in cfg["temperature_source"] or []:
        p = Path(src)
        cand = p if ch.is_thermo_log(p) else p.with_name("OSZICAR")
        if cand.exists():
            found.append(str(cand))
    return found


def _com_motion(cfg, header, masses) -> float:
    """Largest centre-of-mass displacement [A] during a run (worst run).

    Displacement form as for M (ADR-0011 A1): with a variable cell the
    absolute centre of mass would follow the cell fluctuations.
    """
    m = np.asarray(masses, float)
    files = cfg["trajectory"]["files"]
    groups = [files] if cfg["trajectory"]["mode"] == "concatenate" else [[f] for f in files]
    worst = 0.0
    for grp in groups:
        stream, com = RunStream(), []
        for f in grp:
            for chunk in iter_chunks(f, read_header(f, cfg["trajectory"]["species"])):
                rel = stream(chunk.frac, chunk.lattice)
                com.append(np.einsum("k,tkb,tbc->tc", m, rel, chunk.lattice))
        com = remove_reference(np.concatenate(com), stream.lattice, stream.offset, m, "mass")
        com /= m.sum()
        worst = max(worst, float(np.linalg.norm(com - com[0], axis=1).max()))
    return worst


def check(cfg: dict, log=print, energy_files=None, target: float = 0.02) -> dict:
    """Run the diagnostics; return {'items': [...], 'n_warn': int}."""
    from . import _eps_inf, _run_temperatures

    items = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runs, z, header = compute_runs(cfg, log=lambda *a: None)
        n_atoms = header.n_atoms
        temps = _run_temperatures(cfg, runs, lambda *a: None)
        temperature = float(np.mean(temps))
        masses = None
        if cfg["dipole"]["model"] != "external":
            masses, _, _ = ch.run_masses(cfg, header)
        res = spectrum.analyse(
            runs, z, masses, temps, _eps_inf(cfg["epsilon_inf"]), cfg["spectrum"], log
        )
    dt_ps = runs[0].dt_fs * 1e-3
    t_run_ps = float(np.mean([len(r.dipole) for r in runs]) * dt_ps)

    # temperature
    for src, t in _temperatures(cfg):
        if len(t) < 4:
            continue
        sig_can = t.mean() * np.sqrt(2.0 / (3.0 * n_atoms))
        ratio = float(t.std() / sig_can)
        ok = T_STD_RATIO[0] <= ratio <= T_STD_RATIO[1]
        items.append(
            _item(
                f"temperature std / canonical ({Path(src).parent.name}/{Path(src).name})",
                "OK" if ok else "WARN",
                round(ratio, 3),
                list(T_STD_RATIO),
                f"T = {t.mean():.1f} +/- {t.std():.1f} K; NVE of a harmonic solid gives ~0.71",
            )
        )
        tebeg = _tebeg(src)
        if tebeg:
            sig = 1.0 / np.sqrt(3.0 * n_atoms)  # canonical relative std of E (harmonic solid)
            rel_t = float(t.mean() / tebeg - 1.0)
            items.append(
                _item(
                    f"mean T vs INCAR TEBEG ({Path(src).parent.name})",
                    "OK" if abs(rel_t) <= T_TARGET_SIGMA * sig else "WARN",
                    round(rel_t, 4),
                    round(T_TARGET_SIGMA * sig, 4),
                    f"<T> {t.mean():.1f} K vs {tebeg:.1f} K; an NVE start drawn from the canonical "
                    f"ensemble scatters by {sig:.1%} (1 sigma); a larger offset means a poor "
                    "equilibration (e.g. Nose-Hoover SMASS = 0)",
                )
            )
        q = max(1, len(t) // 10)
        change = float((t[-q:].mean() - t[:q].mean()) / t.mean())
        items.append(
            _item(
                f"temperature change over the run ({Path(src).name})",
                "OK" if abs(change) <= T_CHANGE else "WARN",
                round(change, 4),
                T_CHANGE,
                f"{t[:q].mean():.1f} K -> {t[-q:].mean():.1f} K (first / last 10 %)",
            )
        )

    # energy drift
    for ef in _energy_files(cfg, energy_files):
        e = read_energy(ef)
        if len(e) < 4:
            continue
        step_ps = float(cfg["trajectory"]["timestep_fs"]) * 1e-3
        if ch.is_thermo_log(ef):
            steps, _ = ch.read_thermo_log(ef)
            tt = steps * step_ps
        else:
            tt = np.arange(len(e)) * step_ps
        slope = float(np.polyfit(tt, e, 1)[0])  # eV / ps
        per_atom = slope / n_atoms * 1e3  # meV / atom / ps
        rel = abs(slope * (tt[-1] - tt[0]) / n_atoms) / (3 * KB * temperature / EV)
        items.append(
            _item(
                f"energy drift ({Path(ef).parent.name}/{Path(ef).name})",
                "OK" if rel <= E_DRIFT else "WARN",
                round(float(rel), 4),
                E_DRIFT,
                f"{per_atom:+.3f} meV/atom/ps over {tt[-1] - tt[0]:.1f} ps (relative to 3 kT/atom)",
            )
        )

    # rigid drift
    if masses is not None:
        com = _com_motion(cfg, header, masses)
        items.append(
            _item(
                "centre-of-mass motion during the run (A)",
                "OK" if com <= COM_DRIFT_A else "WARN",
                round(com, 4),
                COM_DRIFT_A,
                "VASP removes the drift at the NVE start; motion here is a thermostat or input issue",
            )
        )
    for msg in dict.fromkeys(str(w.message) for w in caught):
        if "rigid translation" in msg:
            items.append(_item("rigid offset removed by the site mapping", "INFO", None, None, msg))

    # equipartition per Gamma mode.  A single NVE run of a weakly anharmonic crystal keeps the
    # mode energies it started with, and those are a canonical draw: each E_m ~ kT Exp(1), so
    # <Qdot_m^2> / kT spreads widely (0.27-1.5 for SiC after a Langevin NVT) even for a perfect
    # start.  Only the mean over the modes, 1 +/- 1 / sqrt(n) canonically, is tested (ADR-0010).
    if res.get("modes"):
        kr = np.concatenate([np.asarray(m["kinetic_ratio"]) for m in res["modes"]])
        dev = abs(kr.mean() - 1.0) * np.sqrt(len(kr))
        items.append(
            _item(
                "Gamma-mode kinetic energy <Qdot^2> / kT: mean over IR modes",
                "OK" if dev <= KIN_MEAN_SIGMA else "WARN",
                round(float(kr.mean()), 3),
                f"1 +/- {KIN_MEAN_SIGMA:g}/sqrt({len(kr)})",
                f"range {kr.min():.2f}-{kr.max():.2f}: a canonical draw spreads like Exp(1); fsum "
                "is biased by the spread, mode and force estimators are not; average over seeds",
            )
        )

    # relaxation
    tail = np.abs(np.asarray(res["acf_tail_ratio"])).max()
    items.append(
        _item(
            "dipole ACF tail C/C(0)",
            "OK" if tail <= ACF_TAIL else "WARN",
            round(float(tail), 4),
            ACF_TAIL,
        )
    )
    dr = max(res["drift_ratio"])
    items.append(
        _item(
            "M(t) diffusive drift ratio",
            "OK" if dr <= DRIFT_RATIO else "WARN",
            round(float(dr), 3),
            DRIFT_RATIO,
        )
    )

    # statistics and the run length needed for `target`
    est = res["estimator"]
    d_eps = np.diag(np.asarray(res[f"static_{est}"]) - res["eps_inf"])
    err = np.diag(np.asarray(res[f"static_{est}_err"]))
    rel = np.abs(err / np.where(d_eps != 0, d_eps, np.nan))
    worst = float(np.nanmax(rel)) if np.isfinite(rel).any() else float("nan")
    n_runs = len(runs)
    factor = (worst / target) ** 2 if np.isfinite(worst) else float("nan")
    if factor <= 1:
        need = "met by the present data"
        if n_runs == 1:
            need += " (block error only; use 3 or more independent runs for seed scatter)"
    elif n_runs > 1:
        need = f"about {int(np.ceil(n_runs * factor))} runs of {t_run_ps:.1f} ps (run-to-run error)"
    else:
        need = (
            f"about {t_run_ps * factor:.0f} ps in one run (block error); seed-to-seed scatter is "
            "not included -- use 3 or more independent runs"
        )
    items.append(
        _item(
            f"relative error of Delta eps_s ({est}, worst component)",
            "OK" if worst <= target else "WARN",
            round(worst, 4),
            target,
            f"per component {np.round(rel, 4).tolist()}; for {target:.0%}: {need}",
        )
    )
    n_warn = sum(1 for x in items if x["status"] == "WARN")
    for x in items:
        log(
            f"[{x['status']:4s}] {x['item']}: {x['value']} (threshold {x['threshold']}) {x['note']}"
        )
    log(f"[check] {n_warn} warning(s); {n_runs} run(s) of {t_run_ps:.1f} ps, estimator {est}")
    return {
        "items": items,
        "n_warn": n_warn,
        "estimator": est,
        "n_runs": n_runs,
        "run_length_ps": t_run_ps,
        "temperature_K": temperature,
        "warnings": [str(w.message) for w in caught],
    }
