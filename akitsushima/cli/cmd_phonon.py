"""``akitsushima phonon`` -- phonon dispersion/DOS/Debye-temperature/free-
energy calculation from a YAML config.
"""
import json
import os
import time as _time

import numpy as np

from ..solvers import PhononCalculator
from ..solvers.config_bridge import _build_force_fn_from_yaml
from .common import (
    _check_poscar,
    _check_yaml,
    _input_hash,
    _print_header,
    _print_output_files,
    _print_results_header,
    _print_timing_summary,
    _resolve_masses,
)

__all__ = ["cmd_phonon"]


def cmd_phonon(args) -> int:
    """Phonon calculation from YAML input (IFC, band structure, DOS, Debye T, free energy)."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1

    struct_cfg = cfg.get("structure", {})
    pot_cfg    = cfg.get("potential", {})
    ph_cfg     = cfg.get("phonon",    {})
    verbose    = ph_cfg.get("verbose", True) and not getattr(args, "quiet", False)

    poscar = struct_cfg.get("poscar", "POSCAR")
    pos_data = _check_poscar(poscar)
    if pos_data is None:
        return 1

    force_fn, r_cut, _, _ = _build_force_fn_from_yaml(pot_cfg)
    supercell = ph_cfg.get("supercell", 2)

    _print_header("Phonon Calculation", cfg, poscar, pot_cfg)
    print(f"\n[Settings]")
    print(f"  Supercell : {supercell}")
    print(f"  Delta     : {ph_cfg.get('delta',0.01)} A")
    print(f"  n_jobs    : {ph_cfg.get('n_jobs',1)}")

    output_dir = ph_cfg.get("output_dir", "./phonon_output")
    os.makedirs(output_dir, exist_ok=True)
    t_sections = []

    ifc_path = ph_cfg.get("load_ifc")
    if ifc_path and os.path.isfile(ifc_path):
        print(f"\nLoading IFC from {ifc_path}")
        t0 = _time.perf_counter()
        calc = PhononCalculator.load_ifc(ifc_path, force_fn=force_fn,
                                          supercell=supercell,
                                          delta=ph_cfg.get("delta",0.01))
        t_sections.append(("Load IFC", _time.perf_counter()-t0))
    else:
        calc = PhononCalculator(
            lattice=pos_data["lattice"], positions=pos_data["positions"],
            atom_types=pos_data["atom_types"], force_fn=force_fn,
            supercell=supercell, delta=ph_cfg.get("delta",0.01),
            masses=_resolve_masses(ph_cfg,pos_data["atom_types"])
                   if ph_cfg.get("masses") else None)
        t0 = _time.perf_counter()
        calc.compute_ifc(n_jobs=ph_cfg.get("n_jobs",1), verbose=verbose)
        t_sections.append(("Compute IFC", _time.perf_counter()-t0))

    bs_cfg = ph_cfg.get("band_structure", {})
    if bs_cfg and bs_cfg.get("q_path"):
        q_path = [(e[0],e[1]) if isinstance(e,(list,tuple)) else (e["label"],e["q"])
                  for e in bs_cfg["q_path"]]
        t0 = _time.perf_counter()
        calc.compute_band_structure(q_path=q_path,
            n_points=bs_cfg.get("n_points",50),
            coordinates=bs_cfg.get("coordinates","reduced"), verbose=verbose)
        t_sections.append(("Band structure", _time.perf_counter()-t0))

    dos_cfg = ph_cfg.get("dos", {})
    if dos_cfg:
        t0 = _time.perf_counter()
        calc.compute_dos(q_mesh=dos_cfg.get("q_mesh",10),
                         n_dos=dos_cfg.get("n_dos",200),
                         smearing=dos_cfg.get("smearing",5.0), verbose=verbose)
        t_sections.append(("DOS", _time.perf_counter()-t0))

    T_D = None
    if ph_cfg.get("debye_temperature", False):
        t0 = _time.perf_counter()
        T_D = calc.debye_temperature_from_vsound(
            n_q=ph_cfg.get("debye_n_q",20), verbose=verbose)
        t_sections.append(("Debye temperature", _time.perf_counter()-t0))

    F_phonon = {}
    fe_cfg = ph_cfg.get("free_energy")
    if fe_cfg:
        T_list = fe_cfg if isinstance(fe_cfg,list) else [fe_cfg.get("T",300.0)]
        t0 = _time.perf_counter()
        for T_val in T_list:
            F_phonon[float(T_val)] = calc.phonon_free_energy(
                T=float(T_val), n_q=ph_cfg.get("free_energy_n_q",20),
                verbose=verbose)
        t_sections.append(("Phonon free energy", _time.perf_counter()-t0))

    thermal = None
    th_cfg = ph_cfg.get("thermal")
    if th_cfg:
        th_cfg = th_cfg if isinstance(th_cfg, dict) else {}
        temps = np.arange(float(th_cfg.get("t_min", 0.0)),
                          float(th_cfg.get("t_max", 1000.0)) + 1e-9,
                          float(th_cfg.get("t_step", 10.0)))
        t0 = _time.perf_counter()
        thermal = calc.thermal_properties(
            temps, q_mesh=th_cfg.get("q_mesh", 20),
            cutoff_THz=float(th_cfg.get("cutoff_THz", 1e-3)), verbose=verbose)
        t_sections.append(("Thermal properties", _time.perf_counter()-t0))

    t0 = _time.perf_counter()
    calc.save_results(output_dir=output_dir, verbose=verbose)
    if thermal is not None:
        _write_thermal_dat(os.path.join(output_dir, "thermal_properties.dat"),
                           thermal, calc.N_basis)
    _write_phonon_json(os.path.join(output_dir, "phonon_results.json"), calc,
                       pot_cfg, supercell, ph_cfg, T_D, F_phonon, thermal)
    t_sections.append(("Save results", _time.perf_counter()-t0))

    _print_results_header("phonon")
    print(f"\n[IFC]")
    print(f"  Supercell : {supercell}  Delta : {ph_cfg.get('delta',0.01)} A")
    if bs_cfg and bs_cfg.get("q_path"):
        print(f"\n[Band structure]")
        print(f"  q-points  : {len(bs_cfg.get('q_path',[]))} special points")
        print(f"  n_points  : {bs_cfg.get('n_points',50)}")
    if dos_cfg:
        print(f"\n[DOS]")
        print(f"  q_mesh    : {dos_cfg.get('q_mesh',10)}")
        print(f"  n_dos     : {dos_cfg.get('n_dos',200)}")
    if T_D is not None:
        print(f"\n[Debye temperature]")
        print(f"  T_Debye   : {T_D:.1f} K")
    if F_phonon:
        print(f"\n[Phonon free energy]")
        for T_val, F in F_phonon.items():
            print(f"  F({T_val:.0f} K)  : {F:.6f} eV")
    if thermal is not None:
        n = calc.N_basis
        print(f"\n[Thermal properties]  q_mesh {thermal['q_mesh']}, "
              f"per atom ({n} atoms / cell)")
        if thermal["n_imaginary"]:
            print(f"  WARNING: {thermal['n_imaginary']} imaginary modes left out "
                  f"(lowest {thermal['min_frequency_THz']:.3f} THz)")
        print(f"  ZPE        : {thermal['ZPE_eV'] / n * 1e3:.3f} meV/atom")
        print(f"  {'T (K)':>8} {'F (meV)':>11} {'S (kB)':>9} {'Cv (kB)':>9} "
              f"{'Theta_D (K)':>12}")
        T = thermal["temperatures"]
        for k in sorted(set(np.searchsorted(T, [0.0, 100.0, 300.0, T[-1]]).clip(0, len(T)-1))):
            print(f"  {T[k]:8.1f} {thermal['F_eV_per_atom'][k]*1e3:11.3f} "
                  f"{thermal['S_kB_per_atom'][k]:9.4f} "
                  f"{thermal['Cv_kB_per_atom'][k]:9.4f} {thermal['theta_D_K'][k]:12.1f}")

    _print_output_files(output_dir)
    _print_timing_summary(t_sections, sum(t for _,t in t_sections))
    return 0


def _write_thermal_dat(path, th, n_atoms) -> None:
    """Thermal properties table (ADR-0018): per primitive cell and per atom."""
    cols = np.column_stack([
        th["temperatures"], th["F_eV"], th["U_eV"], th["S_kB"], th["Cv_kB"],
        th["F_kJ_mol"], th["S_J_K_mol"], th["Cv_J_K_mol"],
        th["F_eV_per_atom"], th["S_kB_per_atom"], th["Cv_kB_per_atom"],
        th["theta_D_K"]])
    header = (f"Harmonic thermal properties, q-mesh {th['q_mesh']} (Gamma-centred "
              f"mode sum), {n_atoms} atoms per cell, modes <= {th['cutoff_THz']} THz "
              f"left out ({th['n_imaginary']} imaginary)\n"
              "T_K F_eV_cell U_eV_cell S_kB_cell Cv_kB_cell F_kJ_mol S_J_K_mol "
              "Cv_J_K_mol F_eV_atom S_kB_atom Cv_kB_atom theta_D_K")
    np.savetxt(path, cols, header=header, fmt="%.10g")


def _write_phonon_json(path, calc, pot_cfg, supercell, ph_cfg, T_D,
                       F_phonon, thermal) -> None:
    """phonon_results.json (ADR-0018, Q25): settings, input hash and the
    scalar / tabulated results.  Rank 0 only."""
    from ..utils.mpi import _mpi_rank
    if _mpi_rank != 0:
        return
    out = {
        "command": "phonon",
        "input_hash": _input_hash(calc.lattice_prim, calc.positions,
                                  calc.atom_types, pot_cfg),
        "natoms_cell": calc.N_basis,
        "supercell": list(calc.supercell),
        "delta_A": calc.delta,
        "masses_amu": calc.masses.tolist(),
        "debye_temperature_vsound_K": T_D,
        "free_energy_eV_cell": {str(k): v for k, v in F_phonon.items()},
    }
    if thermal is not None:
        out["thermal"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in thermal.items()}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
