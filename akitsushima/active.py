"""Active learning driven by the ensemble uncertainty (ADR-0037).

One round is four stages, each recorded in ``active_ledger.jsonl`` so that
running the same configuration again resumes where it stopped:

explore
    md and / or anneal runs (the CLI commands, in process) with the
    current ensemble MLIP, from given start structures.  Trajectories are
    written as extxyz; ``stop_sigma`` (default: threshold hi, 3 logged
    steps) stops a trajectory that has left the trusted region.
select
    Every saved frame (and every anneal minimum) is evaluated again with
    the ensemble.  sigma_F,max < lo: accurate; > hi: failed; in between:
    candidate.  lo and hi default to the ``percentile`` of sigma_F,max
    over the training structures and ``hi_factor`` times it; absolute
    values override them.  The candidates are thinned to at most ``k`` by
    farthest-point sampling in the space of per-species mean ACSF vectors
    standardised over the training structures, starting from the largest
    sigma.  ``strategy: random`` draws ``k`` of the explored frames that
    did not fail at random instead (the baseline of the ADR's validation).
    With fewer than ``k`` to choose from, both fill up from the failed
    frames with the smallest sigma (exact duplicates skipped).  A frame
    whose shortest interatomic distance is below ``min_distance``
    (default ``min_distance_factor`` = 0.75 times the shortest distance in
    the training structures) is unphysical: never selected, filled or
    labelled.  The loop ends when the explored frames are at least
    ``accurate_fraction`` accurate and none failed or was unphysical.
label
    ``mode: external``: an ADR-0011 ``type: external`` potential section
    run for each selected structure (``n_jobs`` at a time).
    ``mode: vasp_inputs``: one VASP directory per structure (POSCAR, the
    INCAR template, a Gamma KPOINTS with k_i = ceil(K / L_i), and
    ``make_potcar.sh`` that concatenates the given POTCAR paths -- the
    POTCARs are referenced, never copied); the command stops, and the next
    run reads ``vasprun.xml``.  A structure that fails (no output, SCF not
    converged) is recorded and dropped.
train
    The ensemble is refitted from scratch on the original training data
    plus every labelled structure so far.  Added structures never enter
    the validation split (``train_only``), so validation and test stay
    those of the start.

Layout: ``output_dir/initial/`` (the round-0 model when trained here),
``output_dir/round_NNN/{explore,select,label,train}/`` and
``output_dir/active_ledger.jsonl``.  Round r explores with the model of
round r - 1 (or the initial model) and its ``train/`` holds the model for
round r + 1.
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
import math
import os
import shutil
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

__all__ = ["ActiveConfigError", "active"]

LEDGER = "active_ledger.jsonl"
MODEL_FILE = "atomic_ensemble_model.npz"

_TOP_KEYS = {
    "output_dir",
    "rounds",
    "seed",
    "model",
    "train",
    "explore",
    "select",
    "label",
    "converge",
    "verbose",
}
_MODEL_KEYS = {"file", "train_dir", "test_dir"}
_EXPLORE_KEYS = {"name", "kind", "structures", "md", "anneal", "potential"}
_SELECT_KEYS = {
    "lo",
    "hi",
    "percentile",
    "hi_factor",
    "k",
    "strategy",
    "stop_consecutive",
    "stop_sigma",
    "min_distance",
    "min_distance_factor",
}
_LABEL_KEYS = {
    "mode",
    "potential",
    "n_jobs",
    "isolated_atom_energy",
    "incar",
    "potcar",
    "kpoint_density",
    "energy",
}
_CONVERGE_KEYS = {"accurate_fraction"}
_ENERGY_KEYS = {
    "wo_entropy": "energy_wo_entropy",
    "free": "energy_free",
    "sigma0": "energy_sigma0",
}


class ActiveConfigError(ValueError):
    """An error in the ``active`` configuration."""


# --------------------------------------------------------------------- config
def _unknown(section, given, allowed):
    bad = sorted(set(given) - set(allowed))
    if bad:
        raise ActiveConfigError(
            f"active: unknown key(s) in {section}: {', '.join(bad)} "
            f"(allowed: {', '.join(sorted(allowed))})"
        )


def _abs(path, base):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def _load_config(config, base_dir):
    if isinstance(config, str):
        base_dir = os.path.dirname(os.path.abspath(config))
        with open(config) as fh:
            config = yaml.safe_load(fh) or {}
    cfg = dict(config.get("active", config))
    _unknown("active", cfg, _TOP_KEYS)
    for key, allowed in (
        ("model", _MODEL_KEYS),
        ("select", _SELECT_KEYS),
        ("label", _LABEL_KEYS),
        ("converge", _CONVERGE_KEYS),
    ):
        cfg[key] = dict(cfg.get(key) or {})
        _unknown(f"active.{key}", cfg[key], allowed)
    model = cfg["model"]
    if not model.get("train_dir"):
        raise ActiveConfigError(
            "active.model.train_dir is required (the fit-layout "
            "training set: structures.csv + <k>/POSCAR)."
        )
    for k in ("file", "train_dir", "test_dir"):
        if model.get(k):
            model[k] = _abs(model[k], base_dir)
    train = dict(cfg.get("train") or {})
    if not train:
        raise ActiveConfigError(
            "active.train is required: the mlip settings of "
            "the ensemble (model, form, descriptor, ensemble, ...)."
        )
    if (
        str(train.get("form", "atom_centred")).lower().replace("-", "_")
        != "atom_centred"
    ):
        raise ActiveConfigError("active.train.form must be atom_centred (ADR-0033).")
    if str(train.get("model", "elm")).lower() not in ("elm", "qelm"):
        raise ActiveConfigError("active.train.model must be elm or qelm (ADR-0033).")
    n_members = train.get("ensemble") or len(train.get("ensemble_seeds") or [])
    if int(n_members or 0) < 2:
        raise ActiveConfigError(
            "active.train needs an ensemble of at least 2 members "
            "(ensemble: 5 or ensemble_seeds)."
        )
    if not train.get("descriptor"):
        raise ActiveConfigError("active.train.descriptor (ACSF) is required.")
    for bad in ("weight_csv",):
        if train.get(bad):
            raise ActiveConfigError(
                f"active.train.{bad} is not supported: the rows change every round."
            )
    if any("fit_to" in t for t in _terms(train.get("baseline"))):
        raise ActiveConfigError(
            "active.train.baseline: a fitted (fit_to) term is "
            "not supported in active learning."
        )
    cfg["train"] = train
    explore = cfg.get("explore") or []
    if isinstance(explore, dict):
        explore = [explore]
    if not explore:
        raise ActiveConfigError("active.explore needs at least one explorer.")
    out = []
    for i, ex in enumerate(explore):
        ex = dict(ex)
        _unknown(f"active.explore[{i}]", ex, _EXPLORE_KEYS)
        kind = str(ex.get("kind", "md")).lower()
        if kind not in ("md", "anneal"):
            raise ActiveConfigError(f"active.explore[{i}].kind must be md or anneal.")
        ex["kind"] = kind
        ex.setdefault("name", f"{kind}{i}")
        pats = ex.get("structures")
        if not pats:
            raise ActiveConfigError(f"active.explore[{i}].structures is required.")
        files = []
        for pat in [pats] if isinstance(pats, str) else pats:
            hits = sorted(glob.glob(_abs(pat, base_dir)))
            if not hits:
                raise ActiveConfigError(
                    f"active.explore[{i}]: no file matches {pat!r}."
                )
            files += hits
        ex["structures"] = files
        ex[kind] = dict(ex.get(kind) or {})
        out.append(ex)
    names = [e["name"] for e in out]
    if len(set(names)) != len(names):
        raise ActiveConfigError(f"active.explore: names must be unique, got {names}.")
    cfg["explore"] = out
    sel = cfg["select"]
    sel.setdefault("percentile", 99.0)
    sel.setdefault("hi_factor", 10.0)
    sel.setdefault("k", 10)
    sel.setdefault("strategy", "uncertainty")
    sel.setdefault("stop_consecutive", 3)
    sel.setdefault("stop_sigma", True)
    sel.setdefault("min_distance_factor", 0.75)
    if sel.get("min_distance") is None and not float(sel["min_distance_factor"]) > 0:
        raise ActiveConfigError("active.select.min_distance_factor must be > 0.")
    if sel["strategy"] not in ("uncertainty", "random"):
        raise ActiveConfigError("active.select.strategy must be uncertainty or random.")
    lab = cfg["label"]
    mode = str(lab.get("mode", "vasp_inputs")).lower()
    lab["mode"] = mode
    if mode == "external":
        pot = dict(lab.get("potential") or {})
        if str(pot.get("type", "")).lower() != "external":
            raise ActiveConfigError(
                "active.label.mode external needs a potential "
                "section of type external (ADR-0011)."
            )
        lab["potential"] = pot
    elif mode == "vasp_inputs":
        for k in ("incar", "potcar", "kpoint_density"):
            if not lab.get(k):
                raise ActiveConfigError(
                    f"active.label.{k} is required for vasp_inputs."
                )
        lab["incar"] = _abs(lab["incar"], base_dir)
        lab["potcar"] = {el: _abs(p, base_dir) for el, p in lab["potcar"].items()}
        energy = str(lab.get("energy", "wo_entropy")).lower()
        if energy not in _ENERGY_KEYS:
            raise ActiveConfigError(
                f"active.label.energy must be one of {list(_ENERGY_KEYS)}."
            )
        lab["energy"] = energy
    else:
        raise ActiveConfigError("active.label.mode must be external or vasp_inputs.")
    lab.setdefault("n_jobs", 1)
    cfg["converge"].setdefault("accurate_fraction", 0.99)
    cfg.setdefault("rounds", 5)
    cfg.setdefault("seed", 42)
    cfg.setdefault("verbose", True)
    cfg["output_dir"] = _abs(cfg.get("output_dir", "./active_output"), base_dir)
    return cfg


def _terms(b_cfg):
    return [] if not b_cfg else ([b_cfg] if isinstance(b_cfg, dict) else list(b_cfg))


# --------------------------------------------------------------------- ledger
class _Ledger:
    def __init__(self, path):
        self.path = path
        self.rows = []
        if os.path.isfile(path):
            with open(path) as fh:
                self.rows = [json.loads(ln) for ln in fh if ln.strip()]

    def add(self, **row):
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row, default=_jsonable) + "\n")
        self.rows.append(row)
        return row

    def find(self, **match):
        for r in reversed(self.rows):
            if all(r.get(k) == v for k, v in match.items()):
                return r
        return None


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _say(cfg, msg):
    if cfg["verbose"]:
        print(msg, flush=True)


# ---------------------------------------------------------------- structures
def _load_split(train_dir):
    from .io.training_set import load_training_set

    csv = os.path.join(train_dir, "structures.csv")
    return load_training_set(csv, train_dir)


def _frame(lattice, positions, atom_types, source):
    return {
        "lattice": np.asarray(lattice, float),
        "positions": np.asarray(positions, float),
        "atom_types": list(atom_types),
        "source": source,
    }


def _species_mean_features(desc, s):
    """Per-species mean of the ACSF vectors (decision 16), species blocks
    concatenated; zeros for a species the structure lacks."""
    G = desc.atomic(s["lattice"], s["positions"], s["atom_types"])
    types = list(s["atom_types"])
    out = []
    for sp in desc.species:
        mask = np.array([t == sp for t in types])
        out.append(G[mask].mean(axis=0) if mask.any() else np.zeros(G.shape[1]))
    return np.concatenate(out)


def _sigma(ff, s):
    ff(s["lattice"], s["positions"], s["atom_types"])
    u = ff.uncertainty() or {}
    return float(u.get("sigma_F_max", float("nan")))


def farthest_point(features, first, k):
    """Indices of ``k`` rows of ``features`` by farthest-point sampling
    (Euclidean), starting from row ``first``."""
    n = len(features)
    k = min(int(k), n)
    if k <= 0:
        return []
    chosen = [int(first)]
    d = np.linalg.norm(features - features[first], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(d))
        if d[nxt] <= 0.0:
            break  # only duplicates left
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(features - features[nxt], axis=1))
    return chosen


# --------------------------------------------------------------------- stages
def _run_cli(cmd_fn, cfg_dict, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "input.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(cfg_dict, fh, sort_keys=False)
    buf = io.StringIO()
    cwd = os.getcwd()
    try:
        os.chdir(run_dir)
        with contextlib.redirect_stdout(buf):
            rc = cmd_fn(Namespace(yaml=path, quiet=False))
    finally:
        os.chdir(cwd)
        with open(os.path.join(run_dir, "run.log"), "w") as fh:
            fh.write(buf.getvalue())
    return rc


def _explore(cfg, led, r, model_path, lo, hi):
    from .cli.cmd_anneal import cmd_anneal
    from .cli.cmd_md import cmd_md

    rdir = os.path.join(cfg["output_dir"], f"round_{r:03d}", "explore")
    sel = cfg["select"]
    stop = None
    if sel["stop_sigma"]:
        stop = {"threshold": float(hi), "consecutive": int(sel["stop_consecutive"])}
    pot = {"type": "MLIP", "model": model_path}
    for ex in cfg["explore"]:
        for j, st in enumerate(ex["structures"]):
            item = f"{ex['name']}/{j:03d}"
            if led.find(round=r, stage="explore_item", item=item):
                continue
            run_dir = os.path.join(rdir, ex["name"], f"{j:03d}")
            seed = int(cfg["seed"]) + 1000 * r + j
            if ex["kind"] == "md":
                md = dict(ex["md"])
                md.update(
                    output_dir="./md_output",
                    traj_format="extxyz",
                    seed=md.get("seed", 42) + seed,
                    verbose=True,
                    timing=False,
                )
                if stop and "stop_sigma" not in ex["md"]:
                    md["stop_sigma"] = stop
                md.pop("restart", None)
                rc = _run_cli(
                    cmd_md,
                    {"structure": {"file": st}, "potential": pot, "md": md},
                    run_dir,
                )
            else:
                an = dict(ex["anneal"])
                an_md = dict(an.get("md") or {})
                an_md["traj_format"] = "extxyz"
                an.update(
                    md=an_md,
                    output_dir="./anneal_output",
                    seed=an.get("seed", 42) + seed,
                    plot=False,
                    restart=True,
                )
                if stop and "stop_sigma" not in ex["anneal"]:
                    an["stop_sigma"] = stop
                rc = _run_cli(
                    cmd_anneal,
                    {"structure": {"file": st}, "potential": pot, "anneal": an},
                    run_dir,
                )
            led.add(
                round=r,
                stage="explore_item",
                item=item,
                rc=int(rc or 0),
                stopped=_stopped(run_dir),
            )
            _say(cfg, f"  round {r}: explored {item} (rc {rc})")
    led.add(round=r, stage="explore", status="done")


def _stopped(run_dir):
    p = os.path.join(run_dir, "run.log")
    if not os.path.isfile(p):
        return 0
    with open(p) as fh:
        return sum(1 for ln in fh if "stop_sigma" in ln and "stopp" in ln)


def _harvest(cfg, r):
    """Every saved frame of the round's explorers, plus anneal minima."""
    from .io.poscar import parse_poscar
    from .io.structure import read_extxyz

    rdir = os.path.join(cfg["output_dir"], f"round_{r:03d}", "explore")
    frames = []
    for path in sorted(glob.glob(os.path.join(rdir, "**", "*.extxyz"), recursive=True)):
        try:
            fr = read_extxyz(path)
        except (ValueError, OSError):
            continue
        rel = os.path.relpath(path, rdir)
        for i, f in enumerate(fr):
            frames.append(
                _frame(f["lattice"], f["positions"], f["atom_types"], f"{rel}#{i}")
            )
    for path in sorted(
        glob.glob(
            os.path.join(rdir, "**", "cycles", "*", "relax", "POSCAR_relaxed"),
            recursive=True,
        )
    ):
        d = parse_poscar(path)
        frames.append(
            _frame(
                d["lattice"],
                d["positions"],
                d["atom_types"],
                os.path.relpath(path, rdir),
            )
        )
    return frames


def _shortest_distance(s, r_search=4.0):
    """Shortest interatomic distance [A] of a structure, all periodic
    images (inf when no pair is closer than ``r_search``)."""
    from .fitting.descriptors.acsf import _neighbours

    _, _, vec = _neighbours(
        np.asarray(s["lattice"], float), np.asarray(s["positions"], float), r_search
    )
    return float(np.linalg.norm(vec, axis=1).min()) if len(vec) else float("inf")


def _distance_guard(cfg, train_structs):
    """(min_distance, d_ref): frames with a shorter interatomic distance
    are unphysical.  One value for every species pair: the shortest
    distance over the training structures times ``min_distance_factor``.
    Not per pair, because a dilute species (8 Sn in 64) may have no
    nearest-neighbour pair of its own in the training data, and its
    reference would then be a second-neighbour distance."""
    sel = cfg["select"]
    d_ref = min(_shortest_distance(s) for s in train_structs)
    if sel.get("min_distance") is not None:
        return float(sel["min_distance"]), d_ref
    return float(sel["min_distance_factor"]) * d_ref, d_ref


def _thresholds(cfg, ff, train_structs):
    sel = cfg["select"]
    sig = np.array([_sigma(ff, s) for s in train_structs])
    sig = sig[np.isfinite(sig)]
    lo = (
        float(np.percentile(sig, float(sel["percentile"])))
        if len(sig)
        else float("nan")
    )
    hi = float(sel["hi_factor"]) * lo
    if sel.get("lo") is not None:
        lo = float(sel["lo"])
    if sel.get("hi") is not None:
        hi = float(sel["hi"])
    if not (lo > 0 and hi > lo):
        raise ActiveConfigError(
            f"active.select: need 0 < lo < hi, got lo {lo}, hi {hi}."
        )
    return lo, hi, (float(sig.min()), float(np.median(sig)), float(sig.max()))


def _select(cfg, led, r, ff, lo, hi, train_structs, d_guard):
    from .io.poscar import write_poscar

    sdir = os.path.join(cfg["output_dir"], f"round_{r:03d}", "select")
    os.makedirs(sdir, exist_ok=True)
    frames = _harvest(cfg, r)
    sel = cfg["select"]
    sig = np.array([_sigma(ff, f) for f in frames])
    dmin = np.array([_shortest_distance(f) for f in frames])
    n = len(frames)
    # Unphysical: atoms closer than the distance guard.  A collapsed
    # trajectory of a poor model can stay below hi, and its oracle labels
    # (E up to 1e13 eV/atom with SW in the unguarded validation run) wreck
    # the least-squares fit; DFT on such a structure is wasted.
    bad = ~(dmin >= d_guard)
    acc = (sig < lo) & ~bad
    fail = ~(sig <= hi)  # nan counts as failed
    cand = np.flatnonzero(~acc & ~fail & ~bad)
    rng = np.random.default_rng(int(cfg["seed"]) + 7919 * (r + 1))
    k = int(sel["k"])
    if sel["strategy"] == "random":
        # the baseline draws from the frames both strategies keep: failed
        # ones (sigma > hi, unphysical) are discarded by either
        pool = np.flatnonzero(~fail & ~bad)
        chosen = sorted(
            int(i) for i in rng.choice(pool, size=min(k, len(pool)), replace=False)
        )
    elif len(cand):
        desc = ff.descriptor
        Xt = np.array([_species_mean_features(desc, s) for s in train_structs])
        mu, sd = Xt.mean(axis=0), Xt.std(axis=0)
        sd[sd < 1e-12] = 1.0
        Xc = np.array(
            [(_species_mean_features(desc, frames[i]) - mu) / sd for i in cand]
        )
        first = int(np.argmax(sig[cand]))
        chosen = [int(cand[i]) for i in farthest_point(Xc, first, k)]
    else:
        chosen = []
    # Too few candidates (a model trained on narrow data puts almost every
    # explored frame above hi): fill up from the failed frames with the
    # smallest sigma, skipping exact duplicates (several trajectories start
    # from one structure).  stop_sigma ends a trajectory soon after it
    # leaves the window, so these are mostly the trajectories' first,
    # physical frames.  Decided after the first validation run (ADR-0037,
    # implementation).
    filled = []
    if len(chosen) < k:
        seen = {_frame_key(frames[i]) for i in chosen}
        ok = fail & ~bad
        for i in np.flatnonzero(ok)[np.argsort(sig[ok], kind="stable")]:
            if len(chosen) + len(filled) >= k or not np.isfinite(sig[i]):
                break
            key = _frame_key(frames[i])
            if key not in seen:
                seen.add(key)
                filled.append(int(i))
        chosen = list(chosen) + filled
    with open(os.path.join(sdir, "frames.csv"), "w") as fh:
        fh.write("index,source,n_atoms,sigma_F_max,d_min,class,selected\n")
        cs = set(chosen)
        for i, f in enumerate(frames):
            c = "accurate" if acc[i] else ("failed" if fail[i] else "candidate")
            if bad[i]:
                c = "unphysical"
            if i in filled:
                c = "failed_filled"
            fh.write(
                f"{i},{f['source']},{len(f['atom_types'])},{sig[i]:.8g},"
                f"{dmin[i]:.4f},{c},"
                f"{int(i in cs)}\n"
            )
    for m, i in enumerate(chosen):
        d = os.path.join(sdir, f"{m:03d}")
        os.makedirs(d, exist_ok=True)
        f = frames[i]
        write_poscar(
            os.path.join(d, "POSCAR"),
            f["lattice"],
            f["positions"],
            f["atom_types"],
            title=f"active round {r} {f['source']}",
        )
    frac_acc = float(acc.mean()) if n else 0.0
    n_fail = int(fail.sum())
    n_bad = int(bad.sum())
    converged = bool(
        n
        and frac_acc >= float(cfg["converge"]["accurate_fraction"])
        and n_fail == 0
        and n_bad == 0
    )
    row = led.add(
        round=r,
        stage="select",
        status="done",
        n_frames=n,
        n_accurate=int(acc.sum()),
        n_candidate=len(cand),
        n_failed=n_fail,
        n_unphysical=n_bad,
        min_distance=float(d_guard),
        accurate_fraction=frac_acc,
        n_selected=len(chosen),
        n_filled_from_failed=len(filled),
        selected=[frames[i]["source"] for i in chosen],
        sigma_selected=[float(sig[i]) for i in chosen],
        strategy=sel["strategy"],
        converged=converged,
    )
    _say(
        cfg,
        f"  round {r}: {n} frames, accurate {acc.sum()}, candidates {len(cand)}, "
        f"failed {n_fail}, unphysical {n_bad}; selected {len(chosen)}"
        + (f" ({len(filled)} filled from failed)" if filled else "")
        + ("  -> converged" if converged else ""),
    )
    return row


def _frame_key(f):
    """Identity of a frame (types, cell and positions to 1e-6 A)."""
    pos = np.round(np.asarray(f["positions"]), 6) + 0.0
    lat = np.round(np.asarray(f["lattice"]), 6) + 0.0
    return (tuple(f["atom_types"]), lat.tobytes(), pos.tobytes())


def _kmesh(lattice, density):
    return [
        max(1, math.ceil(float(density) / float(np.linalg.norm(v)))) for v in lattice
    ]


def _label(cfg, led, r):
    """Returns True when every selected structure is labelled or dropped,
    False when VASP results are still missing."""
    from .io.poscar import parse_poscar

    lab = cfg["label"]
    sdir = os.path.join(cfg["output_dir"], f"round_{r:03d}", "select")
    ldir = os.path.join(cfg["output_dir"], f"round_{r:03d}", "label")
    os.makedirs(ldir, exist_ok=True)
    items = sorted(d for d in os.listdir(sdir) if os.path.isdir(os.path.join(sdir, d)))
    todo = [m for m in items if not led.find(round=r, stage="label_item", item=m)]

    def one_external(m):
        from .solvers.config_bridge import force_fn_from_config

        st = parse_poscar(os.path.join(sdir, m, "POSCAR"))
        pot = dict(lab["potential"], work_dir=os.path.join(ldir, m))
        try:
            E, F, S = force_fn_from_config(pot)(
                st["lattice"], st["positions"], st["atom_types"]
            )
            res = dict(energy=float(E), forces=np.asarray(F), stress=np.asarray(S))
            if not (np.isfinite(res["energy"]) and np.all(np.isfinite(res["forces"]))):
                raise ValueError("non-finite result")
            return m, st, res, None
        except Exception as err:  # recorded and dropped (decision 11)
            return m, st, None, str(err)[:300]

    if lab["mode"] == "external":
        with ThreadPoolExecutor(max_workers=max(1, int(lab["n_jobs"]))) as pool:
            for m, st, res, err in pool.map(one_external, todo):
                _store_label(cfg, led, r, ldir, m, st, res, err)
        return True

    # vasp_inputs
    waiting = []
    for m in todo:
        d = os.path.join(ldir, m)
        st = parse_poscar(os.path.join(sdir, m, "POSCAR"))
        vr = os.path.join(d, "vasprun.xml")
        if not os.path.isdir(d):
            _write_vasp_dir(d, st, lab)
        if not os.path.isfile(vr):
            waiting.append(m)
            continue
        res, err = _read_vasp(d, lab)
        _store_label(cfg, led, r, ldir, m, st, res, err)
    if waiting:
        _say(
            cfg,
            f"  round {r}: {len(waiting)} VASP run(s) waiting in {ldir} "
            f"(run make_potcar.sh, then VASP, then this command again)",
        )
        return False
    return True


def _write_vasp_dir(d, st, lab):
    from .io.poscar import write_poscar

    os.makedirs(d, exist_ok=True)
    write_poscar(
        os.path.join(d, "POSCAR"),
        st["lattice"],
        st["positions"],
        st["atom_types"],
        title="active learning",
    )
    shutil.copyfile(lab["incar"], os.path.join(d, "INCAR"))
    k = _kmesh(st["lattice"], lab["kpoint_density"])
    with open(os.path.join(d, "KPOINTS"), "w") as fh:
        fh.write(
            f"auto (k_i = ceil({lab['kpoint_density']} / L_i))\n0\nGamma\n"
            f" {k[0]} {k[1]} {k[2]}\n 0 0 0\n"
        )
    order = list(dict.fromkeys(st["atom_types"]))
    missing = [el for el in order if el not in lab["potcar"]]
    if missing:
        raise ActiveConfigError(f"active.label.potcar has no path for {missing}.")
    with open(os.path.join(d, "make_potcar.sh"), "w") as fh:
        fh.write(
            "#!/bin/sh\n# POTCARs in the POSCAR's species order "
            f"({' '.join(order)}); referenced, not copied\ncat "
            + " ".join(f"'{lab['potcar'][el]}'" for el in order)
            + " > POTCAR\n"
        )
    os.chmod(os.path.join(d, "make_potcar.sh"), 0o755)


def _read_vasp(d, lab):
    import xml.etree.ElementTree as ET

    from .io.vasp import read_vasprun

    path = os.path.join(d, "vasprun.xml")
    try:
        fr = read_vasprun(path)[-1]
        root = ET.parse(path).getroot()
        nelm = None
        for i in root.iter("i"):
            if i.get("name") == "NELM":
                nelm = int(float(i.text))
                break
        calc = root.findall("calculation")[-1]
        n_sc = len(calc.findall("scstep"))
        if nelm is not None and n_sc >= nelm:
            return None, f"SCF not converged ({n_sc} steps = NELM)"
        e = fr[_ENERGY_KEYS[lab["energy"]]]
        # the geometry VASP computed, not the POSCAR written for it
        return dict(
            energy=float(e),
            forces=fr["forces"],
            stress=fr["stress"],
            lattice=fr["lattice"],
            positions=fr["positions"],
            atom_types=fr["atom_types"],
        ), None
    except Exception as err:
        return None, f"vasprun.xml unreadable: {str(err)[:200]}"


def _store_label(cfg, led, r, ldir, m, st, res, err):
    if res is None:
        led.add(round=r, stage="label_item", item=m, status="dropped", error=err)
        _say(cfg, f"  round {r}: structure {m} dropped: {err}")
        return
    if "lattice" in res:
        st = {k: res[k] for k in ("lattice", "positions", "atom_types")}
    np.savez(
        os.path.join(ldir, f"{m}.npz"),
        lattice=st["lattice"],
        positions=st["positions"],
        atom_types=np.array(st["atom_types"]),
        energy=res["energy"],
        forces=res["forces"],
        stress=np.zeros((3, 3)) if res["stress"] is None else res["stress"],
        has_stress=res["stress"] is not None,
    )
    led.add(
        round=r,
        stage="label_item",
        item=m,
        status="ok",
        energy=res["energy"],
        n_atoms=len(st["atom_types"]),
    )


def _added_items(cfg, led, upto):
    """Labelled structures of rounds 0..upto as _write_split items."""
    e_atom = {
        str(k): float(v)
        for k, v in (cfg["label"].get("isolated_atom_energy") or {}).items()
    }
    items = []
    for r in range(upto + 1):
        ldir = os.path.join(cfg["output_dir"], f"round_{r:03d}", "label")
        for row in led.rows:
            if (
                row.get("round") != r
                or row.get("stage") != "label_item"
                or row.get("status") != "ok"
            ):
                continue
            z = np.load(os.path.join(ldir, f"{row['item']}.npz"))
            types = [str(t) for t in z["atom_types"]]
            missing = sorted(set(types) - set(e_atom)) if e_atom else []
            if missing:
                raise ActiveConfigError(
                    f"active.label.isolated_atom_energy lacks {missing}."
                )
            shift = sum(e_atom[t] for t in types) if e_atom else 0.0
            frame = {
                "lattice": z["lattice"],
                "positions": z["positions"],
                "atom_types": types,
                "forces": z["forces"],
                "stress": z["stress"] if bool(z["has_stress"]) else None,
            }
            items.append(
                {
                    "frame": frame,
                    "target": (float(z["energy"]) - shift) / len(types),
                    "run": f"al_round_{r:03d}",
                    "step": row["item"],
                }
            )
    return items


def _train(cfg, out_dir, added_items):
    """Fit the ensemble on the start data plus ``added_items``; write the
    model and a summary into ``out_dir``; return the model path."""
    from .cli.cmd_mlip import _build_atomic_model, _physics, _structures_and_descriptor
    from .io.dataset import _write_split

    ml = dict(cfg["train"])
    ml["verbose"] = False
    model_cfg = cfg["model"]
    os.makedirs(out_dir, exist_ok=True)
    base_csv = os.path.join(model_cfg["train_dir"], "structures.csv")
    structs, desc = _structures_and_descriptor(
        ml["descriptor"], base_csv, model_cfg["train_dir"], None, False
    )
    n_added = 0
    if added_items:
        add_dir = os.path.join(out_dir, "added")
        _write_split(add_dir, added_items)
        extra, _ = _structures_and_descriptor(
            desc, os.path.join(add_dir, "structures.csv"), add_dir, desc.species, False
        )
        for s in extra:
            s["train_only"] = True
        structs = list(structs) + list(extra)
        n_added = len(extra)
    model = _build_atomic_model(str(ml.get("model", "elm")).lower(), ml, False)
    _, model.physics_config = _physics(ml, desc.species)
    t0 = time.perf_counter()
    model.fit(
        structs,
        desc,
        val_fraction=ml.get("val_fraction", 0.2),
        val_seed=ml.get("val_seed", 0),
    )
    path = os.path.join(out_dir, MODEL_FILE)
    model.save(path)
    summary = {
        "model": path,
        "n_train_structures": len(structs),
        "n_added": n_added,
        "fit_s": time.perf_counter() - t0,
    }
    for name, sc in (
        ("train", getattr(model, "_train_sc", None)),
        ("val", getattr(model, "_val_sc", None)),
    ):
        if sc:
            summary[f"rmse_{name}"] = {
                c: sc[c]["RMSE"] for c in ("E", "F", "S") if sc.get(c)
            }
    if model_cfg.get("test_dir"):
        test, _ = _structures_and_descriptor(
            desc,
            os.path.join(model_cfg["test_dir"], "structures.csv"),
            model_cfg["test_dir"],
            desc.species,
            False,
        )
        sc = model.score_structures(test, desc)
        summary["rmse_test"] = {c: sc[c]["RMSE"] for c in ("E", "F", "S") if sc.get(c)}
    with open(os.path.join(out_dir, "train_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=_jsonable)
    return path, summary


# ------------------------------------------------------------------- driver
def active(config, base_dir: str | None = None) -> dict:
    """Run (or resume) active learning from ``config``: a YAML path, or a
    dict with an ``active`` section (or that section's keys).  Relative
    paths are taken from the YAML's directory (or ``base_dir``, default
    the working directory).  Returns a summary dict with ``status``
    ("converged" | "rounds_done" | "waiting_for_labels" | "stalled"), ``rounds`` (one
    entry per finished round) and ``model`` (the latest model path)."""
    from .fitting.mlip_engine import MLIPForceField

    cfg = _load_config(config, base_dir or os.getcwd())
    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "active_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1, default=_jsonable)
    led = _Ledger(os.path.join(out, LEDGER))

    # the model of round 0
    row = led.find(stage="initial")
    if row is None:
        if cfg["model"].get("file"):
            path, summ = cfg["model"]["file"], {"model": cfg["model"]["file"]}
        else:
            _say(cfg, "  training the initial ensemble")
            path, summ = _train(cfg, os.path.join(out, "initial"), [])
        row = led.add(stage="initial", status="done", model=path, summary=summ)
    model_path = row["model"]
    train_structs = _load_split(cfg["model"]["train_dir"])

    rounds = []
    status = "rounds_done"
    for r in range(int(cfg["rounds"])):
        done = led.find(round=r, stage="train", status="done")
        if done is not None:
            model_path = done["model"]
            rounds.append(done)
            continue
        ff = MLIPForceField.from_file(model_path)
        thr = led.find(round=r, stage="thresholds")
        if thr is None:
            lo, hi, stats = _thresholds(cfg, ff, train_structs)
            d_guard, d_ref = _distance_guard(cfg, train_structs)
            thr = led.add(
                round=r,
                stage="thresholds",
                lo=lo,
                hi=hi,
                min_distance=d_guard,
                train_shortest_distance=d_ref,
                train_sigma_min_median_max=stats,
                model=model_path,
            )
        lo, hi = thr["lo"], thr["hi"]
        d_guard = thr.get("min_distance")
        if d_guard is None:  # ledger written before the distance guard
            d_guard = _distance_guard(cfg, train_structs)[0]
        _say(
            cfg,
            f"round {r}: model {model_path}  lo {lo:.4g}  hi {hi:.4g} eV/A  "
            f"min_distance {d_guard:.3f} A",
        )
        if not led.find(round=r, stage="explore", status="done"):
            _explore(cfg, led, r, model_path, lo, hi)
        sel = led.find(round=r, stage="select", status="done")
        if sel is None:
            sel = _select(cfg, led, r, ff, lo, hi, train_structs, d_guard)
        if sel["converged"]:
            status = "converged"
            break
        if sel["n_selected"] == 0:
            # nothing to label and not converged: retraining would give the
            # same model again
            status = "stalled"
            _say(cfg, f"round {r}: nothing selected and not converged -- stalled "
                      f"(check lo / hi and the explorers)")
            break
        if not led.find(round=r, stage="label", status="done"):
            if not _label(cfg, led, r):
                status = "waiting_for_labels"
                break
            led.add(round=r, stage="label", status="done")
        added = _added_items(cfg, led, r)
        model_path, summ = _train(
            cfg, os.path.join(out, f"round_{r:03d}", "train"), added
        )
        rows = led.add(
            round=r, stage="train", status="done", model=model_path, summary=summ
        )
        rounds.append(rows)
        _say(
            cfg,
            f"round {r}: retrained on {summ['n_train_structures']} structures "
            f"({summ['n_added']} added); test RMSE {summ.get('rmse_test')}",
        )
    return {
        "status": status,
        "rounds": rounds,
        "model": model_path,
        "output_dir": out,
        "initial": led.find(stage="initial"),
    }
