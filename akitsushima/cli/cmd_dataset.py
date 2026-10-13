"""``akitsushima dataset`` -- build a ``fit`` / ``mlip`` training set from
VASP runs (ADR-0021).

Every ionic step of every run becomes one structure of the ``fit`` layout
(``structures.csv`` + ``<k>/POSCAR``, see example/fit/README.md).  Runs are
split into train and test as a whole -- the steps of one relaxation are
strongly correlated, so splitting by frame would leak a trajectory into
both sides -- optionally stratified by composition.  Energies can be
shifted to the isolated-atom reference that classical potentials use:
target E/N = (E - sum_s n_s E_atom,s) / N.
"""

import csv
import os
import time as _time

import numpy as np

from ..constants import VASP_KB_TO_EV_ANG3
from ..fitting.descriptors import sort_species
from ..io import read_outcar, read_vasprun, write_poscar
from .common import _check_yaml, _print_header, _print_results_header

__all__ = ["cmd_dataset"]

_READERS = {"vasprun": ("vasprun.xml", read_vasprun), "outcar": ("OUTCAR", read_outcar)}
_ENERGY_KEYS = {
    "free": "energy_free",
    "wo_entropy": "energy_wo_entropy",
    "sigma0": "energy_sigma0",
}
# vasprun.xml vs OUTCAR agreement, set by OUTCAR's printed precision
# (forces 6 decimals, positions 5, stress 5 in kB, energies 8).
_CROSS_TOL = {
    "energy": 1e-6,
    "lattice": 1e-6,
    "positions": 2e-5,
    "forces": 2e-6,
    "stress_kB": 2e-5,
}


def _find_runs(sources: list[str], filename: str) -> list[str]:
    """A source holding ``filename`` is one run; otherwise each immediate
    sub-directory holding it is (sorted by name)."""
    runs = []
    for src in sources:
        if os.path.isfile(os.path.join(src, filename)):
            runs.append(src)
            continue
        if not os.path.isdir(src):
            raise FileNotFoundError(f"dataset: source not found: '{src}'")
        subs = sorted(
            d for d in os.listdir(src) if os.path.isfile(os.path.join(src, d, filename))
        )
        if not subs:
            raise FileNotFoundError(
                f"dataset: no '{filename}' in '{src}' or its sub-directories."
            )
        runs.extend(os.path.join(src, d) for d in subs)
    return runs


def _select(frames: list, mode) -> list[int]:
    n = len(frames)
    if mode == "all":
        return list(range(n))
    if mode == "last":
        return [n - 1]
    if mode == "first_last":
        return sorted({0, n - 1})
    if isinstance(mode, int) and mode > 0:
        idx = list(range(0, n, mode))
        return idx if idx[-1] == n - 1 else idx + [n - 1]
    raise ValueError(
        f"dataset.frames must be all | last | first_last | a "
        f"positive stride, got {mode!r}."
    )


def _cross_check(run: str, a: list, b: list) -> None:
    if len(a) != len(b):
        raise ValueError(
            f"{run}: vasprun.xml has {len(a)} ionic steps, OUTCAR {len(b)}."
        )
    for k, (x, y) in enumerate(zip(a, b)):
        if x["atom_types"] != y["atom_types"]:
            raise ValueError(f"{run} step {k + 1}: species differ.")
        diffs = {
            "energy": max(abs(x[e] - y[e]) for e in _ENERGY_KEYS.values()),
            "lattice": np.abs(x["lattice"] - y["lattice"]).max(),
            "positions": np.abs(x["positions"] - y["positions"]).max(),
            "forces": np.abs(x["forces"] - y["forces"]).max(),
            "stress_kB": (
                0.0
                if x["stress"] is None and y["stress"] is None
                else np.abs(x["stress"] - y["stress"]).max() / VASP_KB_TO_EV_ANG3
            ),
        }
        bad = {q: d for q, d in diffs.items() if d > _CROSS_TOL[q]}
        if bad:
            raise ValueError(
                f"{run} step {k + 1}: vasprun.xml and OUTCAR "
                f"disagree: {bad} (tolerances {_CROSS_TOL})."
            )


def _half_width(lattice: np.ndarray) -> float:
    """Half the smallest perpendicular width of the cell [A]."""
    vol = abs(np.linalg.det(lattice))
    a, b, c = lattice
    return 0.5 * min(
        vol / np.linalg.norm(np.cross(b, c)),
        vol / np.linalg.norm(np.cross(c, a)),
        vol / np.linalg.norm(np.cross(a, b)),
    )


def _split_runs(
    comps: list[tuple], test_fraction: float, stratify: str, seed: int
) -> list[bool]:
    """True = test.  Stratified: runs sorted by composition, cut into
    n_test consecutive groups, one drawn at random from each."""
    n = len(comps)
    n_test = int(round(test_fraction * n))
    is_test = [False] * n
    if n_test == 0:
        return is_test
    if n_test >= n:
        raise ValueError(
            f"dataset.split.test_fraction {test_fraction} leaves "
            f"no training run ({n} runs)."
        )
    rng = np.random.default_rng(seed)
    if stratify == "composition":
        order = sorted(range(n), key=lambda i: (comps[i], i))
        for group in np.array_split(np.array(order), n_test):
            is_test[int(rng.choice(group))] = True
    elif stratify == "none":
        for i in rng.choice(n, size=n_test, replace=False):
            is_test[int(i)] = True
    else:
        raise ValueError(
            f"dataset.split.stratify must be composition | none, got {stratify!r}."
        )
    return is_test


def _grouped(frame: dict):
    """Atoms grouped by species in first-appearance order -- the order
    write_poscar writes -- so the F rows follow the POSCAR's atoms."""
    types = frame["atom_types"]
    order = []
    for sp in dict.fromkeys(types):
        order.extend(i for i, t in enumerate(types) if t == sp)
    order = np.array(order)
    return (
        [types[i] for i in order],
        frame["positions"][order],
        frame["forces"][order],
    )


def _write_split(out: str, items: list) -> None:
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "structures.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "label", "weight", "struct_idx"])
        for k, it in enumerate(items):
            types, pos, forces = _grouped(it["frame"])
            d = os.path.join(out, str(k))
            os.makedirs(d, exist_ok=True)
            write_poscar(
                os.path.join(d, "POSCAR"),
                it["frame"]["lattice"],
                pos,
                types,
                title=f"{it['run']} step {it['step']}",
            )
            w.writerow([f"{it['target']:.12g}", "E", 1.0, k])
            for f in forces.reshape(-1):
                w.writerow([f"{f:.10g}", "F", 1.0, k])
            s = it["frame"]["stress"]
            if s is not None:
                for r, c in ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1)):
                    w.writerow([f"{s[r, c]:.10g}", "S", 1.0, k])
            it["struct_idx"] = k


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
    e_atom: dict[str, float] = {
        str(k): float(v) for k, v in (sec.get("isolated_atom_energy") or {}).items()
    }
    split = sec.get("split", {}) or {}
    out_dir = sec.get("output_dir", "./training_set")
    r_cut = sec.get("r_cut_check")

    fname, read = _READERS[reader]
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

    items, comps, run_items = [], [], []
    try:
        for run in runs:
            frames = read(os.path.join(run, fname))
            if cross:
                other = "OUTCAR" if reader == "vasprun" else "vasprun.xml"
                oread = read_outcar if reader == "vasprun" else read_vasprun
                _cross_check(run, frames, oread(os.path.join(run, other)))
            types = frames[0]["atom_types"]
            missing = sorted(set(types) - set(e_atom)) if e_atom else []
            if missing:
                raise ValueError(f"{run}: no isolated_atom_energy for {missing}.")
            shift = sum(e_atom[t] for t in types) if e_atom else 0.0
            mine = []
            for k in _select(frames, sec.get("frames", "all")):
                f = frames[k]
                mine.append(
                    {
                        "run": run,
                        "step": k + 1,
                        "frame": f,
                        "e_raw": f[_ENERGY_KEYS[energy]],
                        "target": (f[_ENERGY_KEYS[energy]] - shift) / f["natoms"],
                    }
                )
            run_items.append(mine)
            n = len(types)
            comps.append(tuple(types.count(sp) / n for sp in sort_species(types)))
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        is_test = _split_runs(
            comps,
            float(split.get("test_fraction", 0.0)),
            str(split.get("stratify", "composition")).lower(),
            int(split.get("seed", 0)),
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    train = [it for t, mine in zip(is_test, run_items) if not t for it in mine]
    test = [it for t, mine in zip(is_test, run_items) if t for it in mine]

    _write_split(os.path.join(out_dir, "train"), train)
    if test:
        _write_split(os.path.join(out_dir, "test"), test)

    species = sort_species(
        [t for mine in run_items for t in mine[0]["frame"]["atom_types"]]
    )
    hw = [_half_width(it["frame"]["lattice"]) for mine in run_items for it in mine]
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["split", "struct_idx", "run", "step", "natoms"]
            + [f"n_{sp}" for sp in species]
            + ["energy_eV", "target_eV_per_atom", "max_force_eV_A", "half_width_A"]
        )
        for split_name, lst in (("train", train), ("test", test)):
            for it in lst:
                f = it["frame"]
                w.writerow(
                    [split_name, it["struct_idx"], it["run"], it["step"], f["natoms"]]
                    + [f["atom_types"].count(sp) for sp in species]
                    + [
                        f"{it['e_raw']:.8f}",
                        f"{it['target']:.10f}",
                        f"{np.linalg.norm(f['forces'], axis=1).max():.6f}",
                        f"{_half_width(f['lattice']):.4f}",
                    ]
                )

    _print_results_header("dataset")
    n_tr_runs = is_test.count(False)
    print(f"  Species          : {species}")
    print(f"  Train            : {n_tr_runs} runs, {len(train)} structures")
    print(f"  Test             : {len(runs) - n_tr_runs} runs, {len(test)} structures")
    tgt = np.array([it["target"] for mine in run_items for it in mine])
    print(f"  Target E/N       : {tgt.min():.6f} .. {tgt.max():.6f} eV/atom")
    print(f"  Min half width   : {min(hw):.4f} A")
    if r_cut is not None and min(hw) <= float(r_cut):
        print(
            f"  WARNING: r_cut_check {r_cut} A >= half the smallest cell "
            f"width {min(hw):.4f} A -- the minimum-image check of the "
            f"many-body engines will refuse these cells."
        )
    if verbose and test:
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
