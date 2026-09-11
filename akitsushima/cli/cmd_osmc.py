"""``akitsushima osmc`` -- OS-MC (continuous soft-core lambda-dynamics
grand-canonical MC/MD) from a YAML config.
"""
import os
import time as _time

import numpy as np

from ..solvers import OSMMCSimulation
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

__all__ = ["cmd_osmc"]


def cmd_osmc(args) -> int:
    """OS-MC (OSMMCSimulation) from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1

    struct_cfg = cfg.get("structure",{})
    pot_cfg    = cfg.get("potential",{})
    os_cfg     = cfg.get("osmc",     {})
    # Only rank 0 prints; see the identical guard/rationale in cmd_md.
    is_root    = (_mpi_rank == 0)
    verbose    = (os_cfg.get("verbose",True) and not getattr(args,"quiet",False)
                 and is_root)

    poscar = struct_cfg.get("poscar","POSCAR")
    pos_data = _check_poscar(poscar)
    if pos_data is None:
        return 1

    force_fn, r_cut, _, _ = _build_force_fn_from_yaml(pot_cfg)
    mu = _resolve_mu_from_cfg(os_cfg, default=-0.1, section_name="osmc")
    if is_root:
        _print_header("OS-MC (soft-core lambda GCMC)", cfg, poscar, pot_cfg)
        print(f"\n[Settings]")
        print(f"  Species   : {os_cfg.get('species','?')}")
        print(f"  mu        : {mu} eV")
        print(f"  T         : {os_cfg.get('temperature',300.0)} K")
        print(f"  n_cycles  : {os_cfg.get('n_cycles',1000)}")
        print(f"  m_lam     : {os_cfg.get('m_lam',50.0)}")
        print(f"  sc_alpha  : {os_cfg.get('sc_alpha',0.5)}")

    output_dir = os_cfg.get("output_dir","./osmc_output")
    os.makedirs(output_dir, exist_ok=True)

    sim = OSMMCSimulation(
        lattice=pos_data["lattice"], positions=pos_data["positions"],
        atom_types=pos_data["atom_types"], fix_atoms=pos_data.get("fix_atoms"),
        force_fn=force_fn,
        temperature=os_cfg.get("temperature",300.0), mu=mu,
        species=os_cfg.get("species","Ar"), timestep=os_cfg.get("timestep",1.0),
        n_cycles=os_cfg.get("n_cycles",1000),
        md_steps_per_cycle=os_cfg.get("md_steps_per_cycle",20),
        gcmc_attempts_per_cycle=os_cfg.get("gcmc_attempts_per_cycle",50),
        tau_nhc=os_cfg.get("tau_nhc",100.0),
        traj_every=os_cfg.get("traj_every",10), thermo_every=os_cfg.get("thermo_every",1),
        output_dir=output_dir, skin_width=os_cfg.get("skin_width",0.5),
        seed=os_cfg.get("seed",42), verbose=verbose,
        m_lam=os_cfg.get("m_lam",50.0), sc_alpha=os_cfg.get("sc_alpha",0.5),
        sc_n=os_cfg.get("sc_n",4), lam_dt_frac=os_cfg.get("lam_dt_frac",1.0),
        insertion_mode=os_cfg.get("insertion_mode","random"),
        n_probe=os_cfg.get("n_probe",20), lambda_probe=os_cfg.get("lambda_probe",0.1))

    t0 = _time.perf_counter()
    result = sim.run()
    t_run = _time.perf_counter() - t0

    if not is_root:
        return 0

    _print_results_header("osmc")
    # Bug-fix note (S7): the original read thermo-row key "N_guest" (the
    # actual key is "N") and a top-level "lambda_integral" key that
    # OSMMCSimulation.run() has never returned -- its actual extra keys
    # (beyond GCMCMDSimulation.run()) are n_insertions/n_deletions/
    # insertion_mode/species_stats/multi_species, used below instead.
    thermo = result.get("thermo_log",[])
    if thermo:
        N_v = [r.get("N", float("nan")) for r in thermo]
        print(f"\n[OSMC statistics  ({len(thermo)} frames)]")
        print(f"  <N_guest>    : {np.nanmean(N_v):.3f}")
        print(f"  n_insertions : {result.get('n_insertions','?')}")
        print(f"  n_deletions  : {result.get('n_deletions','?')}")

    species_stats = result.get("species_stats", {})
    if species_stats:
        print(f"\n[Per-species statistics]"
              + ("  (multi-species)" if result.get("multi_species") else ""))
        for sp, st in species_stats.items():
            print(f"  {sp:<4s}: mu={st['mu']:.4f} eV  "
                  f"n_ins={st['n_ins']}  n_del={st['n_del']}  "
                  f"N_current={st.get('N_current','?')}")

    _print_output_files(output_dir)
    _print_timing_summary([(f"OS-MC run ({os_cfg.get('n_cycles',1000)} cycles)", t_run)], t_run)
    return 0
