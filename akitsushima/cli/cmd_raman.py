"""``akitsushima raman`` -- first-order Raman spectra in the harmonic
approximation (ADR-0046 stage S1, ADR-0049).

For every structure: the Gamma dynamical matrix, its degenerate sets,
Raman / IR activity from characters (spglib, optional) and, with a
``polarizability:`` block, relative Raman intensities for named
polarisation geometries and the powder average, broadened with a constant
FWHM.  Several structures (random alloy configurations) are averaged per
atom, with the standard deviation.
"""

import json
import math
import os
import time as _time

import numpy as np

from ..solvers import PhononCalculator
from ..solvers.qha import relax_at_volume
from ..solvers.raman import RAMAN_DEGENERACY_CM, raman_modes, spectrum_columns
from ..solvers.response import polarizability_from_config, species_without_bonds
from ..utils.mpi import _mpi_rank
from .common import (
    _born_setup,
    _check_structure,
    _check_yaml,
    _cli_force_fn,
    _input_hash,
    _print_header,
    _print_output_files,
    _print_timing_summary,
    _resolve_masses,
    _structure_file,
    _symmetry_kw,
)

__all__ = ["auto_supercell", "cmd_raman"]

_SECTION_KEYS = {
    "output_dir",
    "structures",
    "supercell",
    "delta",
    "pre_relax",
    "relax",
    "n_jobs",
    "masses",
    "symmetry",
    "symprec",
    "activity",
    "geometries",
    "temperature",
    "laser_nm",
    "anti_stokes",
    "fwhm_cm",
    "grid",
    "degeneracy_cm",
    "load_ifc",
    "verbose",
}

# Console table: above this many sets only the strongest are printed.
_PRINT_MAX_SETS = 40
_PRINT_TOP_SETS = 20

# Dense diagonalisation guide (ADR-0046 decision 9): eigh of the 3N x 3N
# Gamma matrix took 42 s at N = 1024 and 249 s at N = 2048 (loaded machine).
_DENSE_GUIDE_ATOMS = 1000

# Neighbour-list skin of the pre-relaxation [A]; the automatic supercell
# keeps r_cut + skin below half its perpendicular width.
_SKIN = 0.5


def auto_supercell(lattice, r_cut: float, z_fixed: bool = False) -> list:
    """Smallest replication whose perpendicular widths all exceed
    2 (r_cut + skin).  At Gamma the dynamical matrix of the cell itself is
    exact from any supercell the engine accepts (the images fold onto the
    same atoms with phase 1), so the smallest valid one is used."""
    lattice = np.asarray(lattice, dtype=float)
    vol = abs(np.linalg.det(lattice))
    S = []
    for i in range(3):
        width = vol / np.linalg.norm(
            np.cross(lattice[(i + 1) % 3], lattice[(i + 2) % 3])
        )
        S.append(max(1, math.floor(2.0 * (r_cut + _SKIN) / width) + 1))
    if z_fixed:
        S[2] = 1
    return S


def _geometries(sec: dict, lattice) -> list:
    """Named (incident, scattered) pairs, Cartesian or lattice directions."""
    out = []
    for g in sec.get("geometries") or []:
        unknown = set(g) - {"name", "incident", "scattered", "basis"}
        if unknown or not {"name", "incident", "scattered"} <= set(g):
            raise ValueError(
                "raman.geometries: each entry needs name, incident "
                f"and scattered (basis optional); got {sorted(g)}"
            )
        basis = str(g.get("basis", "cartesian")).lower()
        if basis not in ("cartesian", "lattice"):
            raise ValueError(
                f"raman.geometries[{g['name']}]: basis must be "
                f"cartesian or lattice, got {basis!r}"
            )
        vi = np.asarray(g["incident"], float)
        vs = np.asarray(g["scattered"], float)
        if basis == "lattice":  # [uvw] -> u a1 + v a2 + w a3
            vi, vs = vi @ lattice, vs @ lattice
        out.append({"name": str(g["name"]), "incident": vi, "scattered": vs})
    names = [g["name"] for g in out]
    reserved = {"powder", "powder_par", "powder_perp"}
    if len(set(names)) != len(names) or reserved & set(names):
        raise ValueError(
            f"raman.geometries: names must be unique and not "
            f"{sorted(reserved)}; got {names}"
        )
    return out


def _structure_list(cfg: dict, sec: dict) -> list:
    """[(label, structure dict)] from raman.structures or structure:."""
    items = sec.get("structures")
    if items is None:
        st = _check_structure(cfg.get("structure", {}), "POSCAR")
        if st is None:
            return None
        return [(_structure_file(cfg.get("structure", {})), st)]
    if not isinstance(items, list) or not items:
        raise ValueError("raman.structures must be a non-empty list of files")
    out = []
    for it in items:
        scfg = it if isinstance(it, dict) else {"file": str(it)}
        st = _check_structure(scfg)
        if st is None:
            return None
        out.append((_structure_file(scfg), st))
    return out


def _load_phonon(sec, input_hash, force_fn):
    """phonon output reused when ``load_ifc`` names a directory whose
    phonon_results.json has this run's input hash (POSCAR + potential)."""
    d = sec["load_ifc"]
    meta = json.load(open(os.path.join(d, "phonon_results.json")))
    if meta.get("input_hash") != input_hash:
        raise ValueError(
            f"raman.load_ifc: {d} was computed for another "
            "structure or potential (input hash differs)."
        )
    return PhononCalculator.load_ifc(
        os.path.join(d, "ifc.npz"),
        force_fn=force_fn,
        supercell=meta["supercell"],
        delta=meta["delta_A"],
    )


def _operations(lattice, positions, types, masses, symprec, mode):
    """spglib operations for the activity, or None (with the reason)."""
    if mode is False:
        return None, "off"
    try:
        from ..solvers.symmetry import spglib_operations

        ops = spglib_operations(lattice, positions, types, masses, symprec)
    except ImportError:
        if mode is True:
            raise ValueError(
                "raman.activity: true needs spglib (pip install akitsushima[symmetry])"
            ) from None
        return None, "spglib not installed"
    return ops, f"{ops.symbol} ({ops.number}), {ops.n_ops} operations"


def _write_modes(path, label, res):
    geos = [g["name"] for g in res["geometries"]]
    with open(path, "w") as fh:
        fh.write(f"# Gamma modes of {label}: degenerate sets\n")
        fh.write(
            "# freq_cm deg acoustic ir_active raman_active n_ir n_raman "
            "powder_intensity I_par I_perp rho thermal_factor "
            + " ".join(f"I_{g}" for g in geos)
            + "\n"
        )
        for r in res["sets"]:

            def b(k):
                return "-" if k not in r else str(int(r[k]))

            def x(k, fmt="{:.6e}"):
                return "nan" if k not in r else fmt.format(r[k])

            fh.write(
                f"{r['frequency_cm']:12.4f} {r['degeneracy']:3d} "
                f"{int(r['acoustic'])} {b('ir_active')} {b('raman_active')} "
                f"{x('n_ir', '{:.4f}')} {x('n_raman', '{:.4f}')} "
                f"{x('powder_intensity')} {x('I_par')} {x('I_perp')} "
                f"{x('rho', '{:.5f}')} {x('thermal_factor')} "
                + " ".join(
                    "{:.6e}".format(r["geometry"][g]) for g in geos if "geometry" in r
                )
                + "\n"
            )


_POWDER = ("powder", "powder_par", "powder_perp")


def _norm_scales(cols: dict) -> dict:
    """Normalisation of each spectrum column: the powder columns by the
    maximum of the powder total, every polarisation geometry by the largest
    maximum among the geometries (they share one prefactor, so their ratios
    are physical and a forbidden geometry stays at zero)."""
    pmax = np.max(cols["powder"]["mean"])
    gmax = max([np.max(v["mean"]) for k, v in cols.items() if k not in _POWDER]
               + [0.0])
    return {k: ((pmax if k in _POWDER else gmax) or 1.0) for k in cols}


def _plot(path, grid, cols, title):
    from ..utils.optional_matplotlib import have_matplotlib, plt

    if not have_matplotlib():
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    scale = _norm_scales(cols)
    for k, v in cols.items():
        if k in ("powder_par", "powder_perp"):
            continue
        y = v["mean"]
        m = scale[k]
        ax.plot(grid, y / m, label=k, lw=1.2)
        if v["std"] is not None:
            ax.fill_between(grid, (y - v["std"]) / m, (y + v["std"]) / m, alpha=0.25)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("intensity (powder; geometries jointly normalised)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cmd_raman(args) -> int:
    """Raman spectra from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1
    pot_cfg = cfg.get("potential", {})
    sec = cfg.get("raman", {}) or {}
    unknown = set(sec) - _SECTION_KEYS
    if unknown:
        raise ValueError(f"raman: unknown keys {sorted(unknown)}")
    is_root = _mpi_rank == 0
    out_dir = sec.get("output_dir", "./raman_output")
    verbose = sec.get("verbose", True) and not getattr(args, "quiet", False) and is_root
    structures = _structure_list(cfg, sec)
    if structures is None:
        return 1
    if sec.get("load_ifc") and len(structures) > 1:
        raise ValueError("raman.load_ifc takes one structure, not a structures list")
    force_fn, r_cut = _cli_force_fn(pot_cfg, "cmd_raman")
    from ..fitting.force_fn import uses_esm
    from ..fitting.mlip_engine import check_mlip_for_solver

    species = sorted({t for _, st in structures for t in st["atom_types"]})
    check_mlip_for_solver(force_fn, "raman", species=set(species))
    sym_kw = _symmetry_kw(sec)
    act_mode = sec.get("activity", "auto")
    if act_mode not in (True, False, "auto"):
        raise ValueError(
            f"raman.activity must be true, false or auto, got {act_mode!r}"
        )
    model = polarizability_from_config(cfg.get("polarizability"))
    T = float(sec.get("temperature", 300.0))
    laser = sec.get("laser_nm")
    laser = None if laser is None else float(laser)
    anti = bool(sec.get("anti_stokes", False))
    fwhm = float(sec.get("fwhm_cm", 3.0))
    deg_tol = float(sec.get("degeneracy_cm", RAMAN_DEGENERACY_CM))
    if fwhm <= 0.0 or T < 0.0 or deg_tol < 0.0 or (laser is not None and laser <= 0):
        raise ValueError(
            "raman: fwhm_cm and laser_nm must be positive, "
            "temperature and degeneracy_cm non-negative"
        )
    sc_cfg = sec.get("supercell", "auto")
    if is_root:
        os.makedirs(out_dir, exist_ok=True)
        _print_header(
            "Raman spectra (raman)",
            cfg,
            structures[0][0] if len(structures) == 1 else None,
            pot_cfg,
        )
        print("\n[Settings]")
        print(f"  structures     : {len(structures)}")
        print(f"  supercell      : {sc_cfg}   delta {sec.get('delta', 0.01)} A")
        print(
            "  polarizability : "
            + (
                "none -> frequencies and activity only"
                if model is None
                else f"{type(model).__name__}"
            )
        )
        print(
            f"  temperature    : {T} K   laser {laser or '-'} nm   "
            f"{'anti-Stokes' if anti else 'Stokes'}   FWHM {fwhm} cm^-1"
        )
        missing = species_without_bonds(model, species) if model else []
        if missing:
            print(f"  WARNING: no BPM bond pair contains {missing}")
    t_sections, results = [], []
    for k, (label, st) in enumerate(structures):
        lat, pos, types = st["lattice"], st["positions"], st["atom_types"]
        in_hash = _input_hash(lat, pos, types, pot_cfg)
        if is_root and len(types) > _DENSE_GUIDE_ATOMS:
            print(f"  WARNING: {label} has {len(types)} atoms (> {_DENSE_GUIDE_ATOMS}): "
                  f"6 N = {6 * len(types)} force calls and a dense "
                  f"{3 * len(types)} x {3 * len(types)} diagonalisation "
                  "(about N^3; 42 s at N = 1024, 249 s at N = 2048)")
        masses = _resolve_masses(sec, types)
        t0 = _time.perf_counter()
        if sec.get("load_ifc"):
            pc = _load_phonon(sec, in_hash, force_fn)
            lat, pos = pc.lattice_prim, pc.positions
            masses = pc.masses
            S = pc.supercell
        else:
            S = (
                auto_supercell(lat, r_cut, z_fixed=uses_esm(force_fn))
                if sc_cfg == "auto"
                else [int(x) for x in np.broadcast_to(sc_cfg, 3)]
            )
            if sec.get("pre_relax", True):
                r = relax_at_volume(
                    lat,
                    pos,
                    types,
                    force_fn,
                    None,
                    S,
                    sec.get("relax", {}) or {},
                    output_dir=os.path.join(out_dir, f"pre_relax_{k:03d}"),
                )
                lat, pos = r["lattice"], r["positions"]
                if is_root and not r["converged"]:
                    print(f"  WARNING: pre-relaxation of {label} did not converge")
                t_sections.append((f"Pre-relaxation [{k}]", _time.perf_counter() - t0))
                t0 = _time.perf_counter()
            pc = PhononCalculator(
                lat,
                pos,
                types,
                force_fn,
                supercell=S,
                delta=sec.get("delta", 0.01),
                masses=masses,
                **sym_kw,
            )
            pc.compute_ifc(n_jobs=sec.get("n_jobs", 1), verbose=False)
        t_sections.append((f"Force constants [{k}]", _time.perf_counter() - t0))
        born = _born_setup(cfg, types, pot_cfg, lattice=lat, positions=pos)
        t0 = _time.perf_counter()
        D0 = pc.build_dynamical_matrix(np.zeros(3))  # analytic (TO)
        ops, ops_line = _operations(
            lat, pos, types, pc.masses, sym_kw["symprec"], act_mode
        )
        A = (
            model.polarizability_derivative(lat, pos, types)
            if model is not None
            else None
        )
        res = raman_modes(
            D0,
            pc.masses,
            A,
            ops,
            _geometries(sec, lat),
            temperature=T,
            laser_nm=laser,
            anti_stokes=anti,
            degeneracy_cm=deg_tol,
        )
        res.update(
            label=label,
            natoms=len(types),
            supercell=list(S),
            symmetry=ops_line,
            input_hash=in_hash,
        )
        if born is not None:
            pc.set_nac(born["Z"], born["epsilon_inf"])
            res["lo_cm"] = {}
            for name, qd in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
                w = np.sort(
                    PhononCalculator._eig_to_freq(
                        np.linalg.eigvalsh(
                            pc.build_dynamical_matrix(np.zeros(3), np.array(qd, float))
                        )
                    )
                )
                res["lo_cm"][name] = w[3:].tolist()
        t_sections.append((f"Modes and intensities [{k}]", _time.perf_counter() - t0))
        results.append(res)
        if verbose:
            _print_structure(res)
    if not is_root:
        return 0
    _write_outputs(out_dir, results, model is not None, fwhm, sec)
    _print_output_files(out_dir)
    _print_timing_summary(t_sections, sum(t for _, t in t_sections))
    return 0


def _print_structure(res):
    print(
        f"\n[{res['label']}]  {res['natoms']} atoms, supercell {res['supercell']}, "
        f"activity: {res['symmetry']}"
    )
    if res["activity_deviation"] is not None:
        print(
            f"  largest deviation of a multiplicity from an integer: "
            f"{res['activity_deviation']:.2e}"
        )
    have_i = any("powder_intensity" in r for r in res["sets"])
    top = max([r.get("powder_intensity", 0.0) for r in res["sets"]] + [0.0])
    geos = [g["name"] for g in res["geometries"]]
    print(
        "  freq (cm^-1) deg  IR  Raman"
        + ("   powder    rho  " + " ".join(f"{g:>9}" for g in geos) if have_i else "")
    )
    rows = [r for r in res["sets"] if not r["acoustic"]]
    if len(rows) > _PRINT_MAX_SETS:
        key = "powder_intensity" if have_i else "frequency_cm"
        rows = sorted(sorted(rows, key=lambda r: -r.get(key, 0.0))[:_PRINT_TOP_SETS],
                      key=lambda r: r["frequency_cm"])
        print(f"  ({len(res['sets'])} sets; the {_PRINT_TOP_SETS} "
              + ("strongest" if have_i else "highest")
              + " shown, all of them in the modes file)")
    for r in rows:
        ir = "-" if "ir_active" not in r else ("yes" if r["ir_active"] else "no")
        ra = "-" if "raman_active" not in r else ("yes" if r["raman_active"] else "no")
        line = f"  {r['frequency_cm']:11.2f} {r['degeneracy']:3d} {ir:>4} {ra:>5}"
        if have_i:
            rel = r["powder_intensity"] / top if top > 0 else 0.0
            line += f"  {rel:7.4f} {r['rho']:6.3f}  " + " ".join(
                f"{r['geometry'][g]:9.3e}" for g in geos
            )
        if r["imaginary"]:
            line += "  imaginary"
        print(line)
    if "lo_cm" in res:
        for k, w in res["lo_cm"].items():
            print(f"  LO, q || {k} : " + " ".join(f"{v:.2f}" for v in w))


def _grid(results, fwhm, sec):
    g = sec.get("grid") or {}
    fmax = max(r["frequency_cm"] for res in results for r in res["sets"])
    lo = float(g.get("min", 0.0))
    hi = float(g.get("max", 1.1 * fmax + 10.0 * fwhm))
    step = float(g.get("step", min(0.5, fwhm / 6.0)))
    if step <= 0.0 or hi <= lo:
        raise ValueError("raman.grid: need step > 0 and max > min")
    return np.arange(lo, hi + 0.5 * step, step)


def _write_outputs(out_dir, results, have_intensity, fwhm, sec):
    for k, res in enumerate(results):
        name = "raman_modes.dat" if len(results) == 1 else f"raman_modes_{k:03d}.dat"
        _write_modes(os.path.join(out_dir, name), res["label"], res)
    summary = {
        "command": "raman",
        "fwhm_cm": fwhm,
        "temperature_K": results[0]["temperature_K"],
        "laser_nm": results[0]["laser_nm"],
        "anti_stokes": results[0]["anti_stokes"],
        "structures": [
            {
                k: v
                for k, v in res.items()
                if k
                not in ("eigenvectors", "raman_tensors", "frequencies_cm", "geometries")
            }
            for res in results
        ],
    }
    for s, res in zip(summary["structures"], results):
        s["geometries"] = [
            {
                "name": g["name"],
                "incident": g["incident"].tolist(),
                "scattered": g["scattered"].tolist(),
            }
            for g in res["geometries"]
        ]
        if res["raman_tensors"] is not None:
            s["raman_tensors_A2_amu-1/2"] = {
                str(i): res["raman_tensors"][r["modes"]].tolist()
                for i, r in enumerate(res["sets"])
                if not r["acoustic"]
            }
    if have_intensity:
        grid = _grid(results, fwhm, sec)
        per = [
            spectrum_columns(res, grid, fwhm, per_atom=res["natoms"]) for res in results
        ]
        keys = list(per[0])
        cols = {}
        for key in keys:
            stack = np.array([p[key] for p in per])
            cols[key] = {
                "mean": stack.mean(axis=0),
                "std": stack.std(axis=0, ddof=1) if len(per) > 1 else None,
            }
        hdr = "freq_cm " + " ".join(
            f"{k} {k}_std" if cols[k]["std"] is not None else k for k in keys
        )

        scale = _norm_scales(cols)

        def table(norm):
            out = [grid]
            for k in keys:
                m = scale[k] if norm else 1.0
                out.append(cols[k]["mean"] / m)
                if cols[k]["std"] is not None:
                    out.append(cols[k]["std"] / m)
            return np.column_stack(out)

        note = (
            f"# Raman spectrum per atom, Lorentzian FWHM {fwhm} cm^-1, "
            f"{len(results)} structure(s)\n"
        )
        np.savetxt(
            os.path.join(out_dir, "raman_spectrum.dat"),
            table(False),
            header=note[2:] + hdr,
            fmt="%.6e",
        )
        np.savetxt(
            os.path.join(out_dir, "raman_spectrum_normalised.dat"),
            table(True),
            header=note[2:] + "powder columns / max(powder); geometry columns / the largest geometry maximum\n" + hdr,
            fmt="%.6e",
        )
        summary["spectrum_maxima"] = {k: float(np.max(cols[k]["mean"])) for k in keys}
        _plot(
            os.path.join(out_dir, "raman_spectrum.png"),
            grid,
            cols,
            f"Raman ({len(results)} structure(s), FWHM {fwhm} cm$^{{-1}}$)",
        )
    with open(os.path.join(out_dir, "raman_results.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o).__name__)
