"""``akitsushima osmc`` -- grand-canonical MC/MD with a continuous fractional
atom (CFCMC, ADR-0009) from a YAML config.
"""

import os
import time as _time

from ..solvers import CFCMCSimulation
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

# keys of the old lambda-dynamics osmc (removed with ADR-0009)
_OLD_KEYS = (
    "m_lam",
    "sc_alpha",
    "sc_n",
    "lam_dt_frac",
    "insertion_mode",
    "n_probe",
    "lambda_probe",
    "species_params",
)


def cmd_osmc(args) -> int:
    """CFCMC (CFCMCSimulation) from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1

    struct_cfg = cfg.get("structure", {})
    pot_cfg = cfg.get("potential", {})
    os_cfg = cfg.get("osmc", {})
    # Only rank 0 prints; see the identical guard/rationale in cmd_md.
    is_root = _mpi_rank == 0
    verbose = (
        os_cfg.get("verbose", True) and not getattr(args, "quiet", False) and is_root
    )

    old = [k for k in _OLD_KEYS if k in os_cfg]
    if old:
        raise ValueError(
            f"osmc: {old} belonged to the old lambda-dynamics osmc, which did "
            f"not sample the muVT ensemble; osmc is now CFCMC (ADR-0009).  "
            f"Remove them (see `akitsushima --write-config osmc`)."
        )

    poscar = struct_cfg.get("poscar", "POSCAR")
    pos_data = _check_poscar(poscar)
    if pos_data is None:
        return 1

    force_fn, r_cut, _, _ = _build_force_fn_from_yaml(pot_cfg)
    mu = _resolve_mu_from_cfg(os_cfg, default=-0.1, section_name="osmc")
    n_cycles = int(os_cfg.get("n_cycles", 1000))
    if is_root:
        _print_header("osmc (CFCMC, continuous fractional atom)", cfg, poscar, pot_cfg)
        print("\n[Settings]")
        print(f"  Species      : {os_cfg.get('species', '?')}")
        print(f"  mu           : {mu} eV")
        print(f"  T            : {os_cfg.get('temperature', 300.0)} K")
        print(f"  n_cycles     : {n_cycles}")
        print(f"  equil_cycles : {os_cfg.get('equil_cycles', n_cycles // 5)}")
        print(f"  max_dlambda  : {os_cfg.get('max_dlambda', 0.2)}")

    output_dir = os_cfg.get("output_dir", "./osmc_output")
    os.makedirs(output_dir, exist_ok=True)

    sim = CFCMCSimulation(
        lattice=pos_data["lattice"],
        positions=pos_data["positions"],
        atom_types=pos_data["atom_types"],
        fix_atoms=pos_data.get("fix_atoms"),
        force_fn=force_fn,
        temperature=os_cfg.get("temperature", 300.0),
        mu=mu,
        species=os_cfg.get("species", "Ar"),
        timestep=os_cfg.get("timestep", 1.0),
        n_cycles=n_cycles,
        md_steps_per_cycle=os_cfg.get("md_steps_per_cycle", 20),
        gcmc_attempts_per_cycle=os_cfg.get("gcmc_attempts_per_cycle", 50),
        max_disp=os_cfg.get("max_disp", 0.2),
        tau_nhc=os_cfg.get("tau_nhc", 100.0),
        traj_every=os_cfg.get("traj_every", 10),
        thermo_every=os_cfg.get("thermo_every", 1),
        output_dir=output_dir,
        skin_width=os_cfg.get("skin_width", 0.5),
        seed=os_cfg.get("seed", 42),
        verbose=verbose,
        lambda_move_ratio=os_cfg.get("lambda_move_ratio", 0.5),
        max_dlambda=os_cfg.get("max_dlambda", 0.2),
        n_lambda_bins=os_cfg.get("n_lambda_bins", 10),
        equil_cycles=os_cfg.get("equil_cycles"),
        adapt_weights=os_cfg.get("adapt_weights", True),
        softcore_kT=os_cfg.get("softcore_kT", 10.0),
    )

    t0 = _time.perf_counter()
    result = sim.run()
    t_run = _time.perf_counter() - t0

    if not is_root:
        return 0

    _print_results_header("osmc")
    est = result["cfcmc"]
    print(
        f"\n[CFCMC estimators  ({est['n_samples']} samples after "
        f"{sim.equil_cycles} equilibration cycles)]"
    )
    print(f"  <N> (lambda = 0) : {result['N_mean']:.3f} +/- {result['N_mean_se']:.3f}")
    print(
        f"  <N> (lambda = 1) : {result['N_mean_lambda1']:.3f} +/- "
        f"{result['N_mean_lambda1_se']:.3f}"
    )
    print(f"  mu_ex            : {result['mu_ex']:.5f} +/- {result['mu_ex_se']:.5f} eV")
    print(
        f"  acc (lambda / ins / del) : {result['acc_lambda']:.3f} / "
        f"{result['acc_ins']:.3f} / {result['acc_del']:.3f}"
    )
    print(f"  lambda histogram : {est['lambda_hist']}")

    _print_output_files(output_dir)
    _print_timing_summary([(f"CFCMC run ({n_cycles} cycles)", t_run)], t_run)
    return 0
