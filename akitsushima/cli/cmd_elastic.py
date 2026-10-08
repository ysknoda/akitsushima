"""``akitsushima elastic`` -- elastic tensor, polycrystalline moduli,
sound velocities, Debye temperature and internal strain from a YAML
config (ADR-0017).
"""
import json
import os
import time as _time

import numpy as np

from ..constants import EV_ANG3_TO_GPA
from ..io.poscar import write_poscar
from ..solvers import ElasticConstantsCalculator, GeometryOptimiser
from ..solvers.elastic import DEFAULT_STRAINS, elastic_moduli
from ..utils.mpi import _mpi_rank
from .common import (
    _check_poscar,
    _check_yaml,
    _cli_force_fn,
    _input_hash,
    _print_header,
    _print_output_files,
    _print_results_header,
    _print_timing_summary,
    _resolve_masses,
)

__all__ = ["cmd_elastic"]

# A residual stress above this after the pre-relaxation is reported
# (ADR-0017 decision 2; the relax default pmax).
_RESIDUAL_WARN_GPA = 0.1

_VOIGT_LABELS = ["xx", "yy", "zz", "yz", "xz", "xy"]


def _fmt6(M, fmt="{:12.4f}") -> str:
    return "\n".join("  " + " ".join(fmt.format(v) for v in row) for row in M)


def cmd_elastic(args) -> int:
    """Elastic constants from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1

    struct_cfg = cfg.get("structure", {})
    pot_cfg    = cfg.get("potential", {})
    el_cfg     = cfg.get("elastic",   {})
    is_root    = (_mpi_rank == 0)
    verbose    = (el_cfg.get("verbose", True) and not getattr(args, "quiet", False)
                  and is_root)

    poscar = struct_cfg.get("poscar", "POSCAR")
    pos_data = _check_poscar(poscar)
    if pos_data is None:
        return 1
    lattice, positions = pos_data["lattice"], pos_data["positions"]
    atom_types = pos_data["atom_types"]

    force_fn, r_cut = _cli_force_fn(pot_cfg, "cmd_elastic")

    strains   = el_cfg.get("strains", list(DEFAULT_STRAINS))
    skin      = float(el_cfg.get("skin_width", 0.5))
    rel_cfg   = el_cfg.get("relax", {}) or {}
    pre_relax = bool(el_cfg.get("pre_relax", True))
    output_dir = el_cfg.get("output_dir", "./elastic_output")
    if is_root:
        os.makedirs(output_dir, exist_ok=True)
        _print_header("Elastic constants", cfg, poscar, pot_cfg)
        print(f"\n[Settings]")
        print(f"  pre_relax  : {pre_relax}")
        print(f"  strains    : {strains} (each +-)")
        print(f"  relax_ions : {el_cfg.get('relax_ions', True)}  "
              f"fmax {el_cfg.get('fmax', 1e-4)} eV/A")
        print(f"  n_jobs     : {el_cfg.get('n_jobs', 1)}")
    t_sections = []

    pre = None
    if pre_relax:
        t0 = _time.perf_counter()
        go = GeometryOptimiser(
            lattice, positions, atom_types, force_fn=force_fn, r_cut=r_cut,
            relax_cell=True, method=rel_cfg.get("method", "fire"),
            fmax=rel_cfg.get("fmax", 1e-3), pmax=rel_cfg.get("pmax", 0.01),
            max_steps=rel_cfg.get("max_steps", 3000), skin_width=skin,
            verbose=False, output_dir=os.path.join(output_dir, "pre_relax"))
        r = go.run()
        lattice, positions = r["lattice"], r["positions"]
        pre = {"converged": bool(r["converged"]), "n_steps": int(r["n_steps"]),
               "energy_eV": float(r["energy"])}
        t_sections.append(("Pre-relaxation (cell + ions)", _time.perf_counter() - t0))
        if is_root and not r["converged"]:
            print(f"  WARNING: pre-relaxation did not converge in "
                  f"{r['n_steps']} steps")

    _, _, s0 = force_fn(lattice, positions, atom_types)
    pmax0 = float(np.max(np.abs(s0))) * EV_ANG3_TO_GPA
    if is_root and pmax0 > _RESIDUAL_WARN_GPA:
        print(f"  WARNING: residual stress {pmax0:.3f} GPa > "
              f"{_RESIDUAL_WARN_GPA} GPa at the reference structure; C is "
              f"biased at about that order")

    t0 = _time.perf_counter()
    calc = ElasticConstantsCalculator(
        lattice, positions, atom_types, force_fn, strains=strains,
        relax_ions=el_cfg.get("relax_ions", True),
        fmax=el_cfg.get("fmax", 1e-4), max_steps=el_cfg.get("max_steps", 1000),
        skin_width=skin, n_jobs=el_cfg.get("n_jobs", 1))
    res = calc.compute()
    t_sections.append(("Strained cells (" + str(len(calc._points())) + ")",
                       _time.perf_counter() - t0))

    volume = float(abs(np.linalg.det(lattice)))
    masses = _resolve_masses(el_cfg, atom_types)
    mod = elastic_moduli(res["C"], volume=volume, masses=masses)
    mod0 = elastic_moduli(res["C_clamped"])

    if not is_root:
        return 0

    write_poscar(os.path.join(output_dir, "POSCAR_reference"), lattice,
                 positions, atom_types,
                 title="Reference structure of akitsushima elastic")
    out = {
        "command": "elastic",
        "input_hash": _input_hash(lattice, positions, atom_types, pot_cfg),
        "natoms": len(atom_types),
        "volume_A3": volume,
        "pre_relax": pre,
        "reference_pmax_GPa": pmax0,
        "strains": list(res["strains"]),
        "C_GPa": res["C"].tolist(),
        "C_clamped_GPa": res["C_clamped"].tolist(),
        "S_per_GPa": mod["S"].tolist(),
        "eigenvalues_GPa": mod["eigenvalues"].tolist(),
        "born_stable": mod["born_stable"],
        "B_GPa": mod["B"], "G_GPa": mod["G"], "E_GPa": mod["E"], "nu": mod["nu"],
        "B_clamped_GPa": mod0["B"], "G_clamped_GPa": mod0["G"],
        "pugh_ratio": mod["pugh_ratio"],
        "universal_anisotropy": mod["universal_anisotropy"],
        "density_kg_m3": mod["density_kg_m3"],
        "v_l_m_s": mod["v_l_m_s"], "v_t_m_s": mod["v_t_m_s"],
        "v_m_m_s": mod["v_m_m_s"],
        "debye_temperature_K": mod["debye_temperature_K"],
        "fit_rms_GPa": {k: v.tolist() for k, v in res["fit_rms_GPa"].items()},
        "residual_stress_GPa": res["residual_stress_GPa"].tolist(),
        "unconverged": [list(p) for p in res["unconverged"]],
        "internal_strain_A": res["internal_strain"].tolist(),
    }
    with open(os.path.join(output_dir, "elastic_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    with open(os.path.join(output_dir, "elastic_tensor.dat"), "w") as fh:
        fh.write("# Relaxed-ion C [GPa], Voigt order xx yy zz yz xz xy\n")
        fh.write(_fmt6(res["C"]) + "\n")
        fh.write("# Clamped-ion C [GPa]\n")
        fh.write(_fmt6(res["C_clamped"]) + "\n")
        fh.write("# Compliance S = C^-1 [1/GPa] (relaxed-ion)\n")
        fh.write(_fmt6(mod["S"], "{:14.6e}") + "\n")
    with open(os.path.join(output_dir, "internal_strain.dat"), "w") as fh:
        fh.write("# du_ia / d eta_j [A per unit Voigt strain]; columns: "
                 "atom type axis " + " ".join(_VOIGT_LABELS) + "\n")
        for i, t in enumerate(atom_types):
            for a, ax in enumerate("xyz"):
                fh.write(f"{i + 1:6d} {t:>3s} {ax} " + " ".join(
                    f"{v:12.6f}" for v in res["internal_strain"][i, a]) + "\n")

    _print_results_header("elastic")
    print(f"\n[Reference structure]")
    print(f"  Volume     : {volume:.4f} A^3  ({len(atom_types)} atoms)")
    print(f"  Pmax       : {pmax0:.4f} GPa")
    print(f"\n[Relaxed-ion C (GPa)]")
    print(_fmt6(res["C"], "{:10.2f}"))
    print(f"\n[Clamped-ion C (GPa)]")
    print(_fmt6(res["C_clamped"], "{:10.2f}"))
    print(f"\n[Fit]  RMS residual per mode (GPa): "
          + " ".join(f"{v:.3g}" for v in res["fit_rms_GPa"]["relaxed"]))
    if res["unconverged"]:
        print(f"  WARNING: unconverged ionic relaxations at {res['unconverged']}")
    print(f"\n[Born stability]  eigenvalues (GPa): "
          + " ".join(f"{v:.2f}" for v in mod["eigenvalues"])
          + f"  -> {'stable' if mod['born_stable'] else 'UNSTABLE'}")
    print(f"\n[Polycrystalline moduli (GPa)]   Voigt     Reuss      Hill")
    for name, key in (("B", "B"), ("G", "G"), ("E", "E")):
        d = mod[key]
        print(f"  {name:<30s} {d['voigt']:9.2f} {d['reuss']:9.2f} {d['hill']:9.2f}")
    d = mod["nu"]
    print(f"  {'nu (Poisson ratio)':<30s} {d['voigt']:9.4f} {d['reuss']:9.4f} {d['hill']:9.4f}")
    print(f"  Pugh ratio B_H/G_H : {mod['pugh_ratio']:.3f}")
    print(f"  Universal anisotropy A^U : {mod['universal_anisotropy']:.4f}")
    print(f"\n[Sound velocities and Debye temperature]")
    print(f"  density    : {mod['density_kg_m3']:.1f} kg/m^3")
    print(f"  v_l / v_t / v_m : {mod['v_l_m_s']:.1f} / {mod['v_t_m_s']:.1f} / "
          f"{mod['v_m_m_s']:.1f} m/s")
    print(f"  Theta_D    : {mod['debye_temperature_K']:.1f} K")
    _print_output_files(output_dir)
    _print_timing_summary(t_sections, sum(t for _, t in t_sections))
    return 0
