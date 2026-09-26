"""``akitsushima esm`` -- Effective Screening Medium (ESM) electrostatics
calculation from a YAML config.
"""
import os
import time as _time

import numpy as np

from ..engine.coulomb.esm import ESMCalculator
from ..fitting.io_utils import _load_potential_json
from ..solvers import GeometryOptimiser
from .common import (
    _check_poscar,
    _check_yaml,
    _print_header,
    _print_output_files,
    _print_results_header,
    _print_timing_summary,
)

__all__ = ["cmd_esm"]


def cmd_esm(args) -> int:
    """Effective Screening Medium calculation from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1

    struct_cfg = cfg.get("structure",{})
    pot_cfg    = cfg.get("potential",{})
    esm_cfg    = cfg.get("esm",      {})
    verbose    = esm_cfg.get("verbose",True) and not getattr(args,"quiet",False)

    poscar = struct_cfg.get("poscar","POSCAR_slab")
    pos_data = _check_poscar(poscar)
    if pos_data is None:
        return 1

    _print_header("Effective Screening Medium (ESM)", cfg, poscar, pot_cfg)
    print(f"\n[Settings]")
    print(f"  BC            : {esm_cfg.get('bc','bc1')}")
    print(f"  sigma         : {esm_cfg.get('sigma',0.5)} A")
    print(f"  n_grid_xy     : {esm_cfg.get('n_grid_xy',64)}")
    print(f"  n_grid_z      : {esm_cfg.get('n_grid_z',256)}")
    print(f"  E_field       : {esm_cfg.get('external_field',0.0)} V/A")

    output_dir = esm_cfg.get("output_dir","./esm_output")
    os.makedirs(output_dir, exist_ok=True)
    t_sections = []

    if str(pot_cfg.get("type", "")).strip().lower() == "mlip":
        # ADR-0007: esm computes electrostatics only; it has no short-range
        # term to add an MLIP to, and an MLIP trained on total energies
        # would count the electrostatics twice.
        print("ERROR: esm cannot use an MLIP (potential.type: MLIP): it "
              "computes electrostatics only (ADR-0007).")
        return 1

    charges = pot_cfg.get("charges")
    if charges is None:
        pot_json = pot_cfg.get("potential_json")
        if pot_json and os.path.isfile(pot_json):
            _, charges = _load_potential_json(pot_json)
        else:
            print("ERROR: potential.charges or potential.potential_json required.")
            return 1

    esm = ESMCalculator(
        lattice=pos_data["lattice"], positions=pos_data["positions"],
        atom_types=pos_data["atom_types"], charges=charges,
        bc=esm_cfg.get("bc","bc1"), sigma=esm_cfg.get("sigma",0.5),
        n_grid_xy=esm_cfg.get("n_grid_xy",64), n_grid_z=esm_cfg.get("n_grid_z",256),
        external_field=esm_cfg.get("external_field",0.0), r_cut=esm_cfg.get("r_cut",15.0))

    lattice, positions, atom_types = (pos_data["lattice"], pos_data["positions"],
                                      pos_data["atom_types"])

    rel_cfg = esm_cfg.get("relax")
    if rel_cfg:
        t0 = _time.perf_counter()
        go = GeometryOptimiser(
            lattice=lattice, positions=positions,
            atom_types=atom_types, force_fn=esm,
            r_cut=esm_cfg.get("r_cut",15.0), method=rel_cfg.get("method","fire"),
            fmax=rel_cfg.get("fmax",0.01), max_steps=rel_cfg.get("max_steps",200),
            relax_cell=False, output_dir=os.path.join(output_dir,"relax"),
            verbose=verbose)
        res_go = go.run()
        lattice, positions = res_go["lattice"], res_go["positions"]
        esm.update_geometry(lattice, positions, atom_types)
        t_sections.append(("ESM geometry relax", _time.perf_counter()-t0))

    t0 = _time.perf_counter()
    # Bug-fix note (S21, newly found during the port): get_potential_
    # profile()/charge_density_profile() only return valid data after at
    # least one __call__ has populated ESMCalculator's internal state
    # (update_geometry() resets it back to None, requiring a fresh call
    # too) -- without a `relax:` section requesting it, `esm` was never
    # actually evaluated at all, so this call would previously raise
    # RuntimeError unconditionally. An explicit evaluation on the
    # (possibly just-relaxed) geometry guarantees this always succeeds.
    esm(lattice, positions, atom_types)
    # Bug-fix note (S7): get_potential_profile()/charge_density_profile()
    # return plain (z_grid, values) tuples, not dicts -- the original
    # treated them as dicts (V_profile.get("z", None), V_profile["V"]),
    # which raises AttributeError on a tuple immediately.
    z_axis, V_z = esm.get_potential_profile()
    _, rho_z    = esm.charge_density_profile()
    wf          = esm.work_function()
    t_sections.append(("ESM electrostatics", _time.perf_counter()-t0))

    np.savetxt(os.path.join(output_dir,"potential_profile.dat"),
               np.column_stack([z_axis, V_z]),
               header="z(A)  V(eV)")
    np.savetxt(os.path.join(output_dir,"charge_density.dat"),
               np.column_stack([z_axis, rho_z]),
               header="z(A)  rho(e/A^3)")
    with open(os.path.join(output_dir,"work_function.txt"),"w") as fh:
        fh.write(f"work_function_eV = {wf:.6f}\n")

    _print_results_header("esm")
    print(f"\n[Electrostatics]")
    print(f"  Work function : {wf:.6f} eV")
    print(f"  V_min         : {float(np.min(V_z)):.4f} eV  at z={z_axis[int(np.argmin(V_z))]:.2f} A")
    print(f"  V_max         : {float(np.max(V_z)):.4f} eV  at z={z_axis[int(np.argmax(V_z))]:.2f} A")
    esm.print_summary()

    _print_output_files(output_dir)
    _print_timing_summary(t_sections, sum(t for _,t in t_sections))
    return 0
