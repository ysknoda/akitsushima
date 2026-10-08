"""``akitsushima qha`` -- quasi-harmonic thermal expansion, bulk modulus,
C_p and thermodynamic Grueneisen parameter (ADR-0018); also the shared
volume-scan set-up used by ``gruneisen``.
"""
import json
import os
import time as _time

import numpy as np

from ..solvers.qha import phonons_at, quasi_harmonic, relax_at_volume
from ..utils.mpi import _mpi_rank
from .common import (
    _check_poscar,
    _born_setup,
    _check_yaml,
    _cli_force_fn,
    _input_hash,
    _print_header,
    _print_output_files,
    _print_results_header,
    _print_timing_summary,
    _resolve_masses,
)

__all__ = ["cmd_qha", "_volume_setup"]


def _volume_setup(args, section: str, title: str):
    """Shared start of gruneisen / qha: config, primitive cell, force
    function, ESM / MLIP refusals and the optional full pre-relaxation.
    Returns None on an input error, else a dict."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return None
    pot_cfg = cfg.get("potential", {})
    sec = cfg.get(section, {})
    pos_data = _check_poscar(cfg.get("structure", {}).get("poscar", "POSCAR"))
    if pos_data is None:
        return None
    force_fn, _ = _cli_force_fn(pot_cfg, f"cmd_{section}")
    from ..fitting.force_fn import check_esm_for_solver
    from ..fitting.mlip_engine import check_mlip_for_solver
    check_esm_for_solver(force_fn, section, refuse="changing the volume moves "
                         "the ESM boundaries (ADR-0018)")
    check_mlip_for_solver(force_fn, section,
                          species=set(pos_data["atom_types"]))
    is_root = (_mpi_rank == 0)
    out_dir = sec.get("output_dir", f"./{section}_output")
    if is_root:
        os.makedirs(out_dir, exist_ok=True)
        _print_header(title, cfg, cfg.get("structure", {}).get("poscar"), pot_cfg)
    lat, pos, types = pos_data["lattice"], pos_data["positions"], pos_data["atom_types"]
    sc = sec.get("supercell", 4)
    relax_kw = sec.get("relax", {}) or {}
    t_sections = []
    pre = None
    # load_ifc (dielectric): the stored phonon run's structure is used as is.
    if sec.get("pre_relax", True) and not sec.get("load_ifc"):
        t0 = _time.perf_counter()
        r = relax_at_volume(lat, pos, types, force_fn, None, sc, relax_kw,
                            output_dir=os.path.join(out_dir, "pre_relax"))
        lat, pos = r["lattice"], r["positions"]
        pre = {k: r[k] for k in ("energy", "pressure_GPa", "converged", "n_steps")}
        t_sections.append(("Pre-relaxation (cell + ions)", _time.perf_counter() - t0))
        if is_root and not r["converged"]:
            print("  WARNING: pre-relaxation did not converge")
    born = _born_setup(cfg, types, pot_cfg,
                       required=section in ("dielectric", "piezo"))
    return {"cfg": cfg, "sec": sec, "pot_cfg": pot_cfg, "force_fn": force_fn,
            "born": born, "input_hash": _input_hash(
                pos_data["lattice"], pos_data["positions"], types, pot_cfg),
            "lattice": lat, "positions": pos, "types": types, "supercell": sc,
            "relax_kw": relax_kw, "out_dir": out_dir, "is_root": is_root,
            "masses": _resolve_masses(sec, types), "pre_relax": pre,
            "t_sections": t_sections,
            "verbose": sec.get("verbose", True) and not getattr(args, "quiet", False)
                       and is_root}


def cmd_qha(args) -> int:
    """Quasi-harmonic approximation from YAML input."""
    st = _volume_setup(args, "qha", "Quasi-harmonic approximation (qha)")
    if st is None:
        return 1
    sec, fn, types = st["sec"], st["force_fn"], st["types"]
    out_dir, is_root = st["out_dir"], st["is_root"]
    V0 = abs(float(np.linalg.det(st["lattice"])))
    rng = float(sec.get("volume_range", 0.05))
    n_v = int(sec.get("n_volumes", 7))
    if n_v < 5:
        print("ERROR: qha.n_volumes must be >= 5 (C_v and S are fitted as "
              "fourth-order polynomials in V)")
        return 1
    t_max, t_step = float(sec.get("t_max", 1000.0)), float(sec.get("t_step", 10.0))
    # One step past t_max so that alpha (a central difference) exists at
    # t_max; the extra point is cut before output (as phonopy does).
    T = np.arange(float(sec.get("t_min", 0.0)), t_max + t_step + 1e-9, t_step)
    tol = float(sec.get("imaginary_tol_THz", 0.05))
    skip = bool(sec.get("skip_unstable_volumes", False))
    eos = sec.get("eos", "birch-murnaghan")
    if is_root:
        print(f"\n[Settings]")
        print(f"  V0 (reference) : {V0:.4f} A^3 per primitive cell")
        print(f"  volumes        : {n_v} in V0 x [1 -+ {rng}]")
        print(f"  supercell      : {st['supercell']}   q_mesh {sec.get('q_mesh', 20)}")
        print(f"  EOS            : {eos}")

    rows, F, S, C, dropped = [], [], [], [], []
    t0 = _time.perf_counter()
    for k, f in enumerate(np.linspace(1.0 - rng, 1.0 + rng, n_v)):
        r = relax_at_volume(st["lattice"], st["positions"], types, fn, V0 * f,
                            st["supercell"], st["relax_kw"],
                            output_dir=os.path.join(out_dir, f"relax_v{k}"))
        pc = phonons_at(r["lattice"], r["positions"], types, fn, st["supercell"],
                        delta=sec.get("delta", 0.01), masses=st["masses"],
                        n_jobs=sec.get("n_jobs", 1), born=st["born"])
        th = pc.thermal_properties(T, q_mesh=sec.get("q_mesh", 20))
        V = abs(float(np.linalg.det(r["lattice"])))
        if st["verbose"]:
            print(f"  V = {V:9.4f} A^3  E = {r['energy']:.6f} eV  P = "
                  f"{r['pressure_GPa']:+8.3f} GPa  f_min = "
                  f"{th['min_frequency_THz']:+.3f} THz"
                  + ("" if r["converged"] else "  (relaxation NOT converged)"))
        if th["min_frequency_THz"] < -tol:
            if not skip:
                if is_root:
                    print(f"ERROR: imaginary mode {th['min_frequency_THz']:.3f} THz "
                          f"at V = {V:.4f} A^3 (tolerance {tol} THz).  Narrow "
                          f"volume_range, or set skip_unstable_volumes: true.")
                return 1
            dropped.append(V)
            continue
        rows.append((V, r["energy"], r["pressure_GPa"], r["converged"]))
        F.append(th["F_eV"]); S.append(th["S_kB"]); C.append(th["Cv_kB"])
    st["t_sections"].append((f"Volumes ({n_v}): relax + IFC + mesh",
                             _time.perf_counter() - t0))
    if len(rows) < 5:
        if is_root:
            print(f"ERROR: only {len(rows)} stable volumes left (need >= 5)")
        return 1
    Vs = np.array([r[0] for r in rows])
    Es = np.array([r[1] for r in rows])
    q = quasi_harmonic(Vs, Es, T, np.array(F).T, np.array(S).T, np.array(C).T,
                       eos=eos)
    keep = q["temperatures"] <= t_max + 1e-9
    if q["truncated"]:            # the last kept point has no alpha
        keep[np.flatnonzero(keep)[-1]] = False
    for k, v in list(q.items()):
        if isinstance(v, np.ndarray):
            q[k] = v[keep]
    q["t_max_K"] = float(q["temperatures"][-1])
    if not is_root:
        return 0

    n_at = len(types)
    out = {"command": "qha",
           "input_hash": st["input_hash"], "nac": st["born"] is not None,
           "natoms_cell": n_at, "eos": eos, "pre_relax": st["pre_relax"],
           "volumes_A3": Vs.tolist(), "energies_eV": Es.tolist(),
           "pressures_GPa": [r[2] for r in rows],
           "dropped_unstable_volumes_A3": dropped,
           "t_max_K": q["t_max_K"], "truncated": q["truncated"]}
    for k, v in q.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
    with open(os.path.join(out_dir, "qha_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    cols = np.column_stack([q["temperatures"], q["V_A3"], q["G_eV"], q["B_GPa"],
                            q["alpha_per_K"], q["Cv_J_K_mol"], q["Cp_J_K_mol"],
                            q["S_J_K_mol"], q["gamma"]])
    np.savetxt(os.path.join(out_dir, "qha.dat"), cols, fmt="%.10g",
               header=f"Quasi-harmonic properties per primitive cell ({n_at} atoms), "
                      f"EOS {eos}; per mole of cells\n"
                      "T_K V_A3 G_eV B_GPa alpha_per_K Cv_J_K_mol Cp_J_K_mol "
                      "S_J_K_mol gamma")
    np.savetxt(os.path.join(out_dir, "e_v.dat"),
               np.column_stack([Vs, Es, [r[2] for r in rows]]), fmt="%.10g",
               header="V_A3 E_eV P_GPa (structures relaxed at fixed volume)")

    _print_results_header("qha")
    if q["truncated"]:
        print(f"\n  NOTE: above {q['t_max_K']:.0f} K V(T) leaves the sampled "
              f"volumes; nothing above is reported (widen volume_range).")
    if dropped:
        print(f"\n  WARNING: {len(dropped)} unstable volume(s) left out: {dropped}")
    print(f"\n  {'T (K)':>8} {'V (A^3)':>10} {'B (GPa)':>9} {'alpha (1/K)':>12} "
          f"{'Cp (J/K/mol)':>13} {'gamma':>7}")
    Tq = q["temperatures"]
    shown = sorted({int(np.argmin(np.abs(Tq - t))) for t in
                    (0.0, 100.0, 300.0, 600.0, 1000.0, Tq[-1])})
    for k in shown:
        print(f"  {Tq[k]:8.1f} {q['V_A3'][k]:10.4f} {q['B_GPa'][k]:9.2f} "
              f"{q['alpha_per_K'][k]:12.4e} {q['Cp_J_K_mol'][k]:13.4f} "
              f"{q['gamma'][k]:7.3f}")
    _print_output_files(out_dir)
    _print_timing_summary(st["t_sections"], sum(t for _, t in st["t_sections"]))
    return 0
