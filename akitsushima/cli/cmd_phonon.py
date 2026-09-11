"""``akitsushima phonon`` -- phonon dispersion/DOS/Debye-temperature/free-
energy calculation from a YAML config.
"""
import os
import time as _time

from ..solvers import PhononCalculator
from ..solvers.config_bridge import _build_force_fn_from_yaml
from .common import (
    _check_poscar,
    _check_yaml,
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

    t0 = _time.perf_counter()
    calc.save_results(output_dir=output_dir, verbose=verbose)
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

    _print_output_files(output_dir)
    _print_timing_summary(t_sections, sum(t for _,t in t_sections))
    return 0
