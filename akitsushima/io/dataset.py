"""Training sets from VASP runs (ADR-0021; a library function since
ADR-0024, Q24 -- it was the private body of ``akitsushima dataset``).

Every ionic step of every run becomes one structure of the fit layout
(``structures.csv`` + ``<k>/POSCAR``).  Runs are split into train and test
as a whole -- the steps of one relaxation are strongly correlated, so
splitting by frame would leak a trajectory into both sides -- optionally
stratified by composition.  Energies can be shifted to the isolated-atom
reference: target E/N = (E - sum_s n_s E_atom,s) / N.
"""

import csv
import os

import numpy as np

from ..constants import VASP_KB_TO_EV_ANG3
from .poscar import write_poscar
from .vasp import read_outcar, read_vasprun

__all__ = ["build_training_set"]

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



def build_training_set(sources, output_dir="./training_set", reader="vasprun",
                       energy="wo_entropy", frames="all", isolated_atom_energy=None,
                       test_fraction=0.0, stratify="composition", seed=0,
                       cross_check=True) -> dict:
    """Write a fit-layout training set from VASP runs; return a summary.

    sources : a run directory (holding vasprun.xml / OUTCAR) or a directory
        of run sub-directories, or a list of those.
    reader : "vasprun" | "outcar" (the other file cross-checks it when
        ``cross_check``, within OUTCAR's printed precision).
    energy : "wo_entropy" | "free" | "sigma0".
    frames : "all" | "last" | "first_last" | a positive stride.
    isolated_atom_energy : {species: eV}; subtracted per atom when given.
    test_fraction, stratify ("composition" | "none"), seed : run-level split.

    Writes ``output_dir/{train,test}/structures.csv``, ``<k>/POSCAR`` and
    ``output_dir/manifest.csv``; read them back with
    :func:`akitsushima.io.load_training_set`.  Raises ValueError or
    FileNotFoundError on bad input.  The summary has ``runs``, ``is_test``,
    ``species``, ``n_train`` / ``n_test`` (structures), ``target_min`` /
    ``target_max`` (eV/atom) and ``min_half_width`` (A).
    """
    from ..fitting.descriptors import sort_species

    reader = str(reader).lower()
    if reader not in _READERS:
        raise ValueError(f"dataset.reader must be vasprun | outcar, got '{reader}'.")
    energy = str(energy).lower()
    if energy not in _ENERGY_KEYS:
        raise ValueError(f"dataset.energy must be one of {list(_ENERGY_KEYS)}, got '{energy}'.")
    if not sources:
        raise ValueError("dataset.sources is required (VASP run directories).")
    sources = [sources] if isinstance(sources, str) else list(sources)
    e_atom = {str(k): float(v) for k, v in (isolated_atom_energy or {}).items()}
    fname, read = _READERS[reader]
    runs = _find_runs(sources, fname)

    comps, run_items = [], []
    for run in runs:
        fr = read(os.path.join(run, fname))
        if cross_check:
            other = "OUTCAR" if reader == "vasprun" else "vasprun.xml"
            oread = read_outcar if reader == "vasprun" else read_vasprun
            _cross_check(run, fr, oread(os.path.join(run, other)))
        types = fr[0]["atom_types"]
        missing = sorted(set(types) - set(e_atom)) if e_atom else []
        if missing:
            raise ValueError(f"{run}: no isolated_atom_energy for {missing}.")
        shift = sum(e_atom[t] for t in types) if e_atom else 0.0
        mine = []
        for k in _select(fr, frames):
            f = fr[k]
            mine.append({"run": run, "step": k + 1, "frame": f,
                         "e_raw": f[_ENERGY_KEYS[energy]],
                         "target": (f[_ENERGY_KEYS[energy]] - shift) / f["natoms"]})
        run_items.append(mine)
        n = len(types)
        comps.append(tuple(types.count(sp) / n for sp in sort_species(types)))

    is_test = _split_runs(comps, float(test_fraction), str(stratify).lower(), int(seed))
    train = [it for t, mine in zip(is_test, run_items) if not t for it in mine]
    test = [it for t, mine in zip(is_test, run_items) if t for it in mine]
    _write_split(os.path.join(output_dir, "train"), train)
    if test:
        _write_split(os.path.join(output_dir, "test"), test)

    species = sort_species([t for mine in run_items for t in mine[0]["frame"]["atom_types"]])
    hw = [_half_width(it["frame"]["lattice"]) for mine in run_items for it in mine]
    with open(os.path.join(output_dir, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "struct_idx", "run", "step", "natoms"]
                   + [f"n_{sp}" for sp in species]
                   + ["energy_eV", "target_eV_per_atom", "max_force_eV_A", "half_width_A"])
        for split_name, lst in (("train", train), ("test", test)):
            for it in lst:
                f = it["frame"]
                w.writerow([split_name, it["struct_idx"], it["run"], it["step"], f["natoms"]]
                           + [f["atom_types"].count(sp) for sp in species]
                           + [f"{it['e_raw']:.8f}", f"{it['target']:.10f}",
                              f"{np.linalg.norm(f['forces'], axis=1).max():.6f}",
                              f"{_half_width(f['lattice']):.4f}"])
    tgt = np.array([it["target"] for mine in run_items for it in mine])
    return {"runs": runs, "is_test": is_test, "species": species,
            "n_train": len(train), "n_test": len(test),
            "target_min": float(tgt.min()), "target_max": float(tgt.max()),
            "min_half_width": float(min(hw))}
