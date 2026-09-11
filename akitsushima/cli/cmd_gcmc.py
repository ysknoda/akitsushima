"""``akitsushima gcmc`` -- Grand-Canonical Monte Carlo MD (muVT) from a
YAML config.
"""
import os
import time as _time

import numpy as np

from ..solvers import GCMCMDSimulation
from ..solvers.config_bridge import _build_force_fn_from_yaml
from ..utils.mpi import _mpi_rank
from .common import (
    _check_poscar,
    _check_yaml,
    _print_header,
    _print_output_files,
    _print_results_header,
    _print_timing_summary,
    _resolve_mu_from_cfg,
)

__all__ = ["cmd_gcmc"]


def cmd_gcmc(args) -> int:
    """Grand-Canonical Monte Carlo MD from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1

    struct_cfg = cfg.get("structure", {})
    pot_cfg    = cfg.get("potential", {})
    gc_cfg     = cfg.get("gcmc",      {})
    # Only rank 0 prints; see the identical guard/rationale in cmd_md.
    is_root    = (_mpi_rank == 0)
    verbose    = (gc_cfg.get("verbose", True) and not getattr(args, "quiet", False)
                 and is_root)

    poscar = struct_cfg.get("poscar", "POSCAR")
    pos_data = _check_poscar(poscar)
    if pos_data is None:
        return 1

    force_fn, r_cut, _, _ = _build_force_fn_from_yaml(pot_cfg)
    mu = _resolve_mu_from_cfg(gc_cfg, default=-0.1, section_name="gcmc")
    if is_root:
        _print_header("Grand-Canonical MC (muVT)", cfg, poscar, pot_cfg)
        print(f"\n[Settings]")
        print(f"  Species   : {gc_cfg.get('species','?')}")
        print(f"  mu        : {mu} eV")
        print(f"  T         : {gc_cfg.get('temperature',300.0)} K")
        print(f"  n_cycles  : {gc_cfg.get('n_cycles',1000)}")

    output_dir = gc_cfg.get("output_dir", "./gcmc_output")
    os.makedirs(output_dir, exist_ok=True)

    sim = GCMCMDSimulation(
        lattice                  = pos_data["lattice"],
        positions                = pos_data["positions"],
        atom_types               = pos_data["atom_types"],
        fix_atoms                = pos_data.get("fix_atoms"),
        force_fn                 = force_fn,
        temperature              = gc_cfg.get("temperature",             300.0),
        mu                       = mu,
        species                  = gc_cfg.get("species",                 "Ar"),
        timestep                 = gc_cfg.get("timestep",                1.0),
        n_cycles                 = gc_cfg.get("n_cycles",                1000),
        md_steps_per_cycle       = gc_cfg.get("md_steps_per_cycle",      20),
        gcmc_attempts_per_cycle  = gc_cfg.get("gcmc_attempts_per_cycle", 50),
        ins_del_ratio            = gc_cfg.get("ins_del_ratio",           0.5),
        max_disp                 = gc_cfg.get("max_disp",                0.2),
        tau_nhc                  = gc_cfg.get("tau_nhc",                 100.0),
        traj_every               = gc_cfg.get("traj_every",              10),
        thermo_every             = gc_cfg.get("thermo_every",            1),
        output_dir               = output_dir,
        skin_width               = gc_cfg.get("skin_width",              0.5),
        seed                     = gc_cfg.get("seed",                    42),
        verbose                  = verbose,
        timing                   = True,
    )

    t0 = _time.perf_counter()
    result = sim.run()
    t_run = _time.perf_counter() - t0

    if not is_root:
        return 0

    _print_results_header("gcmc")
    thermo = result.get("thermo_log", [])
    if thermo:
        # Bug-fix note (S7): the original read r.get("N_guest", ...) and a
        # nested result["acceptance"]["insert"/"delete"] dict, but
        # GCMCMDSimulation's thermo_log rows use the key "N" (not
        # "N_guest"), and its run() result carries flat top-level
        # "acc_ins"/"acc_del"/"acc_disp" keys (not a nested "acceptance"
        # dict at all) -- so this summary silently printed NaN/nothing
        # for every GCMC run.
        N_v = [r.get("N", float("nan")) for r in thermo]
        E_v = [r.get("e_pot", float("nan")) for r in thermo]
        print(f"\n[GCMC statistics  ({len(thermo)} thermo frames)]")
        print(f"  <N_guest>   : {np.nanmean(N_v):.2f} atoms")
        print(f"  <E_pot>     : {np.nanmean(E_v):.4f} eV")
        print(f"  Ins accept  : {result.get('acc_ins', float('nan')):.4f}")
        print(f"  Del accept  : {result.get('acc_del', float('nan')):.4f}")

    _print_output_files(output_dir)
    _print_timing_summary([(f"GCMC run ({gc_cfg.get('n_cycles',1000)} cycles)", t_run)], t_run)
    return 0
