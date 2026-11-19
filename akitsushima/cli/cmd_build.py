"""``akitsushima build`` -- write a structure built by ``akitsushima.build``
(ADR-0035).

    build:
      start: {bulk: {type, species, a, ...}}   # or {file: path, format, index, species}
      steps:                                    # applied in order
        - supercell: [2, 2, 2]
        - substitute: {from: Ge, to: {Sn: 8}, seed: 1}
      output: POSCAR_GeSn                       # format from the name, or format:
      repeat: 1                                 # n > 1: every seed + k

The record ``<output stem>_build.json`` holds the input, the seeds used,
the indices chosen and a summary (composition, volume, shortest distance
per species pair) of every structure.
"""

import copy
import json
import os
import time as _time

from .. import __version__
from .. import build as B
from ..io.structure import detect_format, read_structure, write_structure
from .common import _check_yaml, _print_header, _print_results_header

__all__ = ["cmd_build"]

_STEPS = {
    "supercell": lambda s, kw: B._supercell(s, **kw),
    "substitute": lambda s, kw: B._substitute(
        s, kw.pop("from", kw.pop("from_", None)), **kw
    ),
    "vacancy": lambda s, kw: B._vacancy(s, **kw),
    "slab": lambda s, kw: B._slab(s, **kw),
    "rattle": lambda s, kw: (B.rattle(s, **kw), {}),
    "strain": lambda s, kw: B._strain(s, **kw),
    "scale": lambda s, kw: (B.scale(s, **kw), {}),
}


def _read_start(start: dict):
    if isinstance(start, dict) and "file" in start:
        f = start["file"]
        if isinstance(f, dict):
            f = dict(f)
            return read_structure(f.pop("path"), **f)
        return read_structure(str(f))
    if isinstance(start, dict) and len(start) == 1 and "bulk" in start:
        return B.bulk(**dict(start["bulk"]))
    raise ValueError(
        "build.start: one of {bulk: {type, species, a, ...}}, {file: path} or "
        "{file: {path, format, index, species}}."
    )


def _step(s, item: dict, k: int):
    """Apply one step; seeds offset by the repeat index k."""
    if not isinstance(item, dict) or len(item) != 1:
        raise ValueError(
            f"build.steps: each step is one {{name: options}}, got {item}."
        )
    ((name, opts),) = item.items()
    if name not in _STEPS:
        raise ValueError(f"build.steps: unknown step '{name}' ({' | '.join(_STEPS)}).")
    if name == "supercell" and not isinstance(opts, dict):
        opts = {"matrix": opts}
    kw = copy.deepcopy(dict(opts or {}))
    if kw.get("seed") is not None:
        kw["seed"] = int(kw["seed"]) + k
    used = {key: v for key, v in kw.items()}
    try:
        s, rec = _STEPS[name](s, kw)
    except TypeError as exc:
        raise ValueError(f"build.steps.{name}: {exc}") from None
    return s, {"step": name, "options": used, **rec}


def _outputs(output: str, n: int, fmt):
    fmt = detect_format(output) if fmt in (None, "auto") else str(fmt)
    if n == 1 or fmt == "extxyz":
        return fmt, [output]
    stem, ext = os.path.splitext(output)
    if fmt == "poscar" and ext.lower() not in (".vasp", ".poscar"):
        stem, ext = output, ""
    return fmt, [f"{stem}_{k:03d}{ext}" for k in range(n)]


def build_from_config(sec: dict, verbose: bool = True) -> dict:
    """Run a ``build:`` section; return the record written to JSON."""
    output = sec.get("output")
    if not output:
        raise ValueError("build.output is required.")
    n = int(sec.get("repeat", 1))
    if n < 1:
        raise ValueError(f"build.repeat must be >= 1, got {n}.")
    steps = list(sec.get("steps") or [])
    fmt, paths = _outputs(str(output), n, sec.get("format", "auto"))
    records = []
    for k in range(n):
        s = _read_start(sec.get("start", {}))
        recs = []
        for item in steps:
            s, r = _step(s, item, k)
            recs.append(r)
        path = paths[0] if fmt == "extxyz" else paths[k]
        title = f"akitsushima build {k}" if n > 1 else "akitsushima build"
        if fmt == "extxyz":
            from ..io.structure import write_extxyz

            write_extxyz(
                path, s["lattice"], s["positions"], s["atom_types"], append=k > 0
            )
        else:
            write_structure(
                path,
                s["lattice"],
                s["positions"],
                s["atom_types"],
                format=fmt,
                title=title,
            )
        summ = B.summary(s)
        records.append({"index": k, "output": path, "steps": recs, **summ})
        if verbose:
            comp = " ".join(f"{sp}{c}" for sp, c in summ["composition"].items())
            dmin = ", ".join(
                f"{p} {d:.4f}" for p, d in summ["min_distance"].items() if d is not None
            )
            print(
                f"  [{k}] {path}: {summ['natoms']} atoms ({comp}), "
                f"V {summ['volume']:.3f} A^3, d_min {dmin} A"
            )
            for r in recs:
                if r["step"] == "slab":
                    for j, p in enumerate(r["planes"]):
                        mark = "  <- top" if j == r["termination"] else ""
                        print(
                            f"        plane {j}: z {p['height']:.4f} A  "
                            f"{p['composition']}{mark}"
                        )
    stem = os.path.splitext(str(output))[0] if fmt != "poscar" else str(output)
    log = {
        "akitsushima": __version__,
        "input": sec,
        "format": fmt,
        "structures": records,
    }
    with open(f"{stem}_build.json", "w") as fh:
        json.dump(
            log,
            fh,
            indent=1,
            default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
        )
    log["json"] = f"{stem}_build.json"
    return log


def cmd_build(args) -> int:
    """Build a structure (bulk / file -> steps -> output)."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1
    sec = cfg.get("build", {}) or {}
    verbose = sec.get("verbose", True) and not getattr(args, "quiet", False)
    _print_header("Structure builder (build)", cfg)
    t0 = _time.perf_counter()
    _print_results_header("build")
    try:
        log = build_from_config(sec, verbose=verbose)
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"\n  Record           : {log['json']}")
    print(f"  Wall time        : {_time.perf_counter() - t0:.2f} s")
    return 0
