"""``akitsushima kappa`` -- thermal conductivity (Green-Kubo + optional
Slack-Debye) from a YAML config.
"""
import os
import time as _time

from ..solvers import PhononCalculator, ThermalConductivityCalculator
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

__all__ = ["cmd_kappa"]


def cmd_kappa(args) -> int:
    """Thermal conductivity (Green-Kubo + optional Slack-Debye) from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1

    struct_cfg = cfg.get("structure", {})
    pot_cfg    = cfg.get("potential", {})
    kp_cfg     = cfg.get("kappa",     {})
    verbose    = kp_cfg.get("verbose", True) and not getattr(args, "quiet", False)

    poscar = struct_cfg.get("poscar", "POSCAR")
    pos_data = _check_poscar(poscar)
    if pos_data is None:
        return 1

    force_fn, r_cut, _, _ = _build_force_fn_from_yaml(pot_cfg)
    _print_header("Thermal Conductivity (Green-Kubo)", cfg, poscar, pot_cfg)
    T_target = kp_cfg.get("temperature", 300.0)
    n_steps  = kp_cfg.get("n_steps", 100_000)
    dt       = kp_cfg.get("timestep", 0.5)
    print(f"\n[Settings]")
    print(f"  Temperature    : {T_target} K")
    print(f"  NVE steps      : {n_steps}  ({n_steps*dt/1000:.3f} ps)")
    print(f"  Equil steps    : {kp_cfg.get('equil_steps',10000)}")
    print(f"  Timestep       : {dt} fs")
    print(f"  tau_max (HCACF): {kp_cfg.get('tau_max',20.0)} ps")

    output_dir = kp_cfg.get("output_dir", "./kappa_output")
    os.makedirs(output_dir, exist_ok=True)
    masses = _resolve_masses(kp_cfg, pos_data["atom_types"]) \
             if kp_cfg.get("masses") else None
    t_sections = []

    calc = ThermalConductivityCalculator(
        lattice=pos_data["lattice"], positions=pos_data["positions"],
        atom_types=pos_data["atom_types"], force_fn=force_fn,
        masses=masses, temperature=T_target)

    t0 = _time.perf_counter()
    calc.run_md(n_steps=n_steps, timestep=dt,
                equil_steps=kp_cfg.get("equil_steps",10_000),
                heat_flux_every=kp_cfg.get("heat_flux_every",5),
                traj_every=kp_cfg.get("traj_every",100),
                thermo_every=kp_cfg.get("thermo_every",1000),
                seed=kp_cfg.get("seed", 42),
                output_dir=output_dir, verbose=verbose,
                decompose=kp_cfg.get("decompose"))
    t_sections.append(("NVE MD run", _time.perf_counter()-t0))

    t0 = _time.perf_counter()
    kappa = calc.compute_kappa(tau_max=kp_cfg.get("tau_max",20.0),
                               n_blocks=kp_cfg.get("n_blocks",5),
                               verbose=verbose)
    t_sections.append(("Green-Kubo integration", _time.perf_counter()-t0))

    decomp = None
    if kp_cfg.get("decompose"):
        decomp = calc.compute_kappa_decomposition(
            tau_max=kp_cfg.get("tau_max", 20.0), verbose=verbose)
        import json
        with open(os.path.join(output_dir, "kappa_decomposition.json"), "w") as fh:
            json.dump({k: ({g: v.tolist() for g, v in d.items()} if isinstance(d, dict)
                           else (d.tolist() if hasattr(d, "tolist") else d))
                       for k, d in decomp.items()}, fh, indent=1)

    kappa_slack = None
    slack_cfg = kp_cfg.get("slack_debye")
    if slack_cfg:
        ifc_path  = slack_cfg.get("ifc_path")
        supercell = slack_cfg.get("supercell", 2)
        t0 = _time.perf_counter()
        if ifc_path and os.path.isfile(ifc_path):
            ph_calc = PhononCalculator.load_ifc(
                ifc_path, force_fn=force_fn, supercell=supercell)
        else:
            ph_calc = PhononCalculator(
                lattice=pos_data["lattice"], positions=pos_data["positions"],
                atom_types=pos_data["atom_types"], force_fn=force_fn,
                supercell=supercell)
            ph_calc.compute_ifc(verbose=verbose)
        kappa_slack = calc.slack_debye(
            phonon_calc=ph_calc, T=T_target,
            gruneisen=slack_cfg.get("gruneisen"),
            use_vsound=slack_cfg.get("use_vsound",False), verbose=verbose)
        t_sections.append(("Slack-Debye estimate", _time.perf_counter()-t0))

    t0 = _time.perf_counter()
    calc.save_results(output_dir=output_dir, verbose=verbose)
    t_sections.append(("Save results", _time.perf_counter()-t0))

    _print_results_header("kappa")
    print(f"\n[Green-Kubo thermal conductivity  (T={T_target:.0f} K)]")
    print(f"  kappa_xx : {kappa[0,0]:.4f} W/(m K)")
    print(f"  kappa_yy : {kappa[1,1]:.4f} W/(m K)")
    print(f"  kappa_zz : {kappa[2,2]:.4f} W/(m K)")
    print(f"  kappa_iso: {(kappa[0,0]+kappa[1,1]+kappa[2,2])/3:.4f} W/(m K)")
    if decomp is not None:
        for name, parts in decomp.items():
            if name in ("sum_rule_residual", "total"):
                continue
            print(f"\n[kappa by {name}]  (sum-rule residual "
                  f"{decomp['sum_rule_residual'][name]:.1e} W/m/K)")
            for g, k in parts.items():
                print(f"  {g:>18s} : {k[0,0]:9.4f} {k[1,1]:9.4f} {k[2,2]:9.4f}")
    if kappa_slack is not None:
        print(f"\n[Slack-Debye estimate]")
        print(f"  kappa_Slack : {kappa_slack:.4f} W/(m K)")

    _print_output_files(output_dir)
    _print_timing_summary(t_sections, sum(t for _,t in t_sections))
    return 0
