"""``akitsushima relax`` -- geometry optimisation from a YAML config."""
import os
import time as _time

import numpy as np

from ..constants import EV_ANG3_TO_GPA
from ..solvers import GeometryOptimiser
from ..solvers.config_bridge import _build_force_fn_from_yaml
from ..utils.mpi import _mpi_rank
from .common import (
    _check_poscar,
    _check_yaml,
    _fmt_time,
    _print_header,
    _print_output_files,
    _print_results_header,
    _print_timing_summary,
)

__all__ = ["cmd_relax"]


def cmd_relax(args) -> int:
    """Geometry optimisation from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1

    struct_cfg = cfg.get("structure", {})
    pot_cfg    = cfg.get("potential", {})
    rel_cfg    = cfg.get("relax",     {})
    # Only rank 0 prints; see the identical guard/rationale in cmd_md.
    is_root    = (_mpi_rank == 0)
    verbose    = (rel_cfg.get("verbose", True) and not getattr(args, "quiet", False)
                 and is_root)

    poscar = struct_cfg.get("poscar", "POSCAR")
    pos_data = _check_poscar(poscar)
    if pos_data is None:
        return 1

    force_fn, r_cut, buck_params, charges = _build_force_fn_from_yaml(pot_cfg)

    # --- Header (rank 0 only) ------------------------------------------------
    if is_root:
        _print_header("Geometry Optimisation (relax)", cfg, poscar, pot_cfg)
        print(f"\n[Settings]")
        print(f"  Method     : {rel_cfg.get('method','fire').upper()}")
        print(f"  relax_cell : {rel_cfg.get('relax_cell', False)}")
        print(f"  fmax       : {rel_cfg.get('fmax', 0.01)} eV/A")
        print(f"  pmax       : {rel_cfg.get('pmax', 0.1)} GPa")
        print(f"  check_forces : {rel_cfg.get('check_forces', True)}")
        print(f"  check_stress : {rel_cfg.get('check_stress', None)} (null=auto)")
        print(f"  check_energy : {rel_cfg.get('check_energy', False)}")
        print(f"  max_steps  : {rel_cfg.get('max_steps', 2000)}")
        print(f"  save_best  : {rel_cfg.get('save_best', False)}")
        if rel_cfg.get("relax_cell", False):
            print(f"  cell_update_scheme : {rel_cfg.get('cell_update_scheme', 'generalized_force')}")
            print(f"  lattice_constraints: {rel_cfg.get('lattice_constraints', None)} (null=all free)")

    # Initial energy
    t_fe0 = _time.perf_counter()
    E0, F0, S0 = force_fn(pos_data["lattice"], pos_data["positions"],
                           pos_data["atom_types"])
    t_fe0 = _time.perf_counter() - t_fe0
    fmax0 = float(np.max(np.abs(F0)))
    pmax0 = float(np.max(np.abs(S0))) * EV_ANG3_TO_GPA
    if is_root:
        print(f"\n[Initial state]")
        print(f"  E_pot      : {E0:.6f} eV  ({E0/pos_data['natoms']:.6f} eV/atom)")
        print(f"  Fmax       : {fmax0:.4e} eV/A")
        print(f"  Pmax       : {pmax0:.4f} GPa")
        print(f"  1 force_fn : {_fmt_time(t_fe0)}")

    output_dir = rel_cfg.get("output_dir", "./relax_output")
    os.makedirs(output_dir, exist_ok=True)

    kw = dict(
        lattice      = pos_data["lattice"],
        positions    = pos_data["positions"],
        atom_types   = pos_data["atom_types"],
        fix_atoms    = pos_data.get("fix_atoms"),
        r_cut        = r_cut,
        relax_cell   = rel_cfg.get("relax_cell",  False),
        check_forces = rel_cfg.get("check_forces", True),
        check_stress = rel_cfg.get("check_stress", None),
        check_energy = rel_cfg.get("check_energy", False),
        fmax         = rel_cfg.get("fmax",         0.01),
        pmax         = rel_cfg.get("pmax",         0.1),
        emax         = rel_cfg.get("emax",         1e-6),
        max_steps    = rel_cfg.get("max_steps",    2000),
        skin_width   = rel_cfg.get("skin_width",   0.5),
        method       = rel_cfg.get("method",       "fire"),
        output_dir   = output_dir,
        verbose      = verbose,
        save_best    = rel_cfg.get("save_best",    False),
        dt_start     = rel_cfg.get("dt_start",     None),
        dt_max       = rel_cfg.get("dt_max",       None),
        cell_update_scheme = rel_cfg.get("cell_update_scheme", "generalized_force"),
        lattice_constraints = rel_cfg.get("lattice_constraints", None),
        timing       = True,
    )
    if force_fn is not None:
        kw["force_fn"] = force_fn
    else:
        kw.update(buck_params=buck_params, charges=charges,
                  ewald_alpha=pot_cfg.get("ewald_alpha"),
                  coulomb_method=pot_cfg.get("coulomb_method","ewald"),
                  pppm_accuracy=pot_cfg.get("pppm_accuracy",1e-5),
                  pppm_mesh=pot_cfg.get("pppm_mesh"),
                  pppm_order=pot_cfg.get("pppm_order",6))

    t0 = _time.perf_counter()
    result = GeometryOptimiser(**kw).run()
    t_relax = _time.perf_counter() - t0

    if not is_root:
        return 0

    # --- Results summary (rank 0 only) --------------------------------------
    _print_results_header("relax")
    N = pos_data["natoms"]
    print(f"\n[Initial structure]")
    print(f"  E_pot : {E0:.6f} eV  ({E0/N:.6f} eV/atom)")
    print(f"  Fmax  : {fmax0:.4e} eV/A    Pmax : {pmax0:.4f} GPa")

    E1 = result["energy"]; F1 = result["forces"]
    fmax1 = float(np.max(np.abs(F1)))
    print(f"\n[Final (relaxed) structure]")
    print(f"  E_pot : {E1:.6f} eV  ({E1/N:.6f} eV/atom)")
    print(f"  Fmax  : {fmax1:.4e} eV/A")
    print(f"  dE    : {E1-E0:+.6f} eV")
    print(f"  Conv  : {result.get('converged', '?')}")
    print(f"  Steps : {result.get('n_steps','?')}")

    if rel_cfg.get("save_best") and result.get("best_energy") is not None:
        be = result["best_energy"]; bs = result["best_step"]
        print(f"\n[Best structure (lowest E)]")
        print(f"  E_pot : {be:.6f} eV  (at step {bs})")
        print(f"  File  : {os.path.join(output_dir,'POSCAR_best')}")

    _print_output_files(output_dir)

    td = result.get("timing", {})
    sections = [("1 force_fn call (benchmark)", t_fe0),
                (f"GeometryOptimiser.run() [{result.get('n_steps','?')} steps]", t_relax)]
    if td and td.get("enabled") and td.get("n_steps", 0) > 0:
        sections.append((f"  mean/step (TimingCollector)",
                         td["per_step_ms"]/1000))
    _print_timing_summary(sections, t_fe0 + t_relax)
    return 0
