"""``akitsushima dataset`` -- build a ``fit`` / ``mlip`` training set from
VASP runs (ADR-0021).  The work is done by
:func:`akitsushima.io.build_training_set` (ADR-0024); this module reads
the YAML and prints.
"""

import os
import time as _time
import warnings

from ..io.dataset import (
    _ENERGY_KEYS,
    _READERS,
    _find_runs,
    build_training_set,
    format_isolated_atom_check,
)
from .common import _check_yaml, _print_header, _print_results_header

__all__ = ["cmd_dataset"]


def cmd_dataset(args) -> int:
    """Build a training set from VASP runs."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1
    sec = cfg.get("dataset", {}) or {}
    verbose = sec.get("verbose", True) and not getattr(args, "quiet", False)
    _print_header("Training set from VASP runs (dataset)", cfg)
    t0 = _time.perf_counter()

    reader = str(sec.get("reader", "vasprun")).lower()
    if reader not in _READERS:
        print(f"ERROR: dataset.reader must be vasprun | outcar, got '{reader}'.")
        return 1
    energy = str(sec.get("energy", "wo_entropy")).lower()
    if energy not in _ENERGY_KEYS:
        print(
            f"ERROR: dataset.energy must be one of {list(_ENERGY_KEYS)}, "
            f"got '{energy}'."
        )
        return 1
    sources = sec.get("sources")
    if not sources:
        print("ERROR: dataset.sources is required (VASP run directories).")
        return 1
    sources = [sources] if isinstance(sources, str) else list(sources)
    cross = bool(sec.get("cross_check", True))
    # a number (eV) or the path of the isolated-atom run (ADR-0026)
    e_atom = {
        str(k): (v if isinstance(v, str) else float(v))
        for k, v in (sec.get("isolated_atom_energy") or {}).items()
    }
    split = sec.get("split", {}) or {}
    out_dir = sec.get("output_dir", "./training_set")
    r_cut = sec.get("r_cut_check")

    fname, _ = _READERS[reader]
    try:
        runs = _find_runs(sources, fname)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("\n[Input]")
    print(
        f"  Runs        : {len(runs)}  (reader {fname}"
        f"{', cross-checked with ' + ('OUTCAR' if reader == 'vasprun' else 'vasprun.xml') if cross else ''})"
    )
    print(
        f"  Energy      : {_ENERGY_KEYS[energy]}"
        f"{'  minus isolated atoms ' + str(e_atom) if e_atom else '  (raw DFT, eV/atom)'}"
    )
    print(f"  Frames      : {sec.get('frames', 'all')}")

    try:
        with warnings.catch_warnings():
            # printed below as a table, from the summary
            warnings.filterwarnings("ignore", message="isolated-atom runs differ")
            res = build_training_set(
                sources, output_dir=out_dir, reader=reader, energy=energy,
                frames=sec.get("frames", "all"), isolated_atom_energy=e_atom,
                test_fraction=float(split.get("test_fraction", 0.0)),
                stratify=str(split.get("stratify", "composition")).lower(),
                seed=int(split.get("seed", 0)), cross_check=cross)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    if e_atom:
        print("  Isolated atoms : "
              + ", ".join(f"{sp} {e:.8f} eV" + (f" ({e_atom[sp]})" if isinstance(e_atom[sp], str)
                                                else "")
                          for sp, e in res["isolated_atom_energy"].items()))
        chk = res["isolated_atom_check"]
        for line in format_isolated_atom_check(chk):
            print(f"  WARNING: {line}" if line.startswith("isolated") else f"  {line}")
        if chk["unchecked"]:
            print(f"  NOTE: isolated_atom_energy of {chk['unchecked']} given as numbers -- "
                  f"their VASP settings cannot be checked against the training runs "
                  f"(give the run directory instead, ADR-0026).")

    _print_results_header("dataset")
    is_test = res["is_test"]
    n_tr_runs = is_test.count(False)
    print(f"  Species          : {res['species']}")
    print(f"  Train            : {n_tr_runs} runs, {res['n_train']} structures")
    print(f"  Test             : {len(runs) - n_tr_runs} runs, {res['n_test']} structures")
    print(f"  Target E/N       : {res['target_min']:.6f} .. {res['target_max']:.6f} eV/atom")
    print(f"  Min half width   : {res['min_half_width']:.4f} A")
    if r_cut is not None and res["min_half_width"] <= float(r_cut):
        print(
            f"  WARNING: r_cut_check {r_cut} A >= half the smallest cell "
            f"width {res['min_half_width']:.4f} A -- the minimum-image check of the "
            f"many-body engines will refuse these cells."
        )
    if verbose and res["n_test"]:
        print(
            "  Test runs        : "
            + ", ".join(
                os.path.basename(os.path.normpath(r))
                for r, t in zip(runs, is_test)
                if t
            )
        )
    print(
        f"\n  Output           : {out_dir}/{{train,test}}/structures.csv, "
        f"<k>/POSCAR, manifest.csv"
    )
    print(f"  Wall time        : {_time.perf_counter() - t0:.1f} s")
    return 0
