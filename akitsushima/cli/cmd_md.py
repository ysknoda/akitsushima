"""``akitsushima md`` -- molecular dynamics (NVT / NVE / NPT) from a YAML
config.
"""
import os
import time as _time

import numpy as np

from ..solvers import NPTSimulation, NVESimulation, NVTSimulation
from ..solvers.config_bridge import _build_force_fn_from_yaml
from ..utils.mpi import _mpi_rank
from .common import (
    _check_poscar,
    _check_yaml,
    _print_header,
    _print_output_files,
    _print_results_header,
    _print_timing_summary,
    _resolve_masses,
)

__all__ = ["cmd_md"]


def cmd_md(args) -> int:
    """Molecular dynamics from YAML input (NVT / NVE / NPT)."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1

    struct_cfg = cfg.get("structure", {})
    pot_cfg    = cfg.get("potential", {})
    md_cfg     = cfg.get("md",        {})
    ens        = str(md_cfg.get("ensemble", "nvt")).lower()
    # Only rank 0 prints anything and runs with verbose=True; every other
    # MPI rank stays silent. Without this guard, `mpirun -n N` would have
    # ALL N ranks independently execute cmd_md() and print the SAME
    # header/settings/simulation-progress output N times over, since each
    # rank runs the identical Python script with no awareness that its
    # console output is redundant with the other ranks'.
    is_root    = (_mpi_rank == 0)
    verbose    = (md_cfg.get("verbose", True) and not getattr(args, "quiet", False)
                 and is_root)

    poscar = struct_cfg.get("poscar", "POSCAR")
    pos_data = _check_poscar(poscar)
    if pos_data is None:
        return 1

    force_fn, r_cut, buck_params, charges = _build_force_fn_from_yaml(pot_cfg)

    # --- Header (rank 0 only) ------------------------------------------------
    if is_root:
        _print_header(f"Molecular Dynamics ({ens.upper()})", cfg, poscar, pot_cfg)
    N = pos_data["natoms"]
    dt = md_cfg.get("timestep", 1.0)
    n_steps = md_cfg.get("n_steps", 10000)
    equil   = md_cfg.get("equil_steps", 0)
    total   = equil + n_steps
    if is_root:
        print(f"\n[Settings]")
        print(f"  Ensemble     : {ens.upper()}")
        print(f"  Temperature  : {md_cfg.get('temperature', 300.0)} K")
        print(f"  Timestep     : {dt} fs")
        print(f"  Equil steps  : {equil}  ({equil*dt/1000:.4f} ps)")
        print(f"  Prod  steps  : {n_steps}  ({n_steps*dt/1000:.4f} ps)")
        print(f"  Total steps  : {total}  ({total*dt/1000:.4f} ps)")
        if ens in ("nvt", "npt"):
            # The defaults differ per ensemble and must match what the
            # constructors below actually receive: printing 20.0 for an
            # NPT run that then ran at 100.0 was simply a false report.
            _tau_default = 100.0
            print(f"  Tau NHC      : {md_cfg.get('tau_nhc', _tau_default)} fs")
        if ens == "npt":
            print(f"  Tau baro     : {md_cfg.get('tau_baro', 1000.0)} fs")
            print(f"  Pressure     : {md_cfg.get('pressure', 0.0)} GPa")

    output_dir = md_cfg.get("output_dir", f"./{ens}_output")
    os.makedirs(output_dir, exist_ok=True)
    masses = _resolve_masses(md_cfg, pos_data["atom_types"])

    common = dict(
        lattice      = pos_data["lattice"],
        positions    = pos_data["positions"],
        atom_types   = pos_data["atom_types"],
        fix_atoms    = pos_data.get("fix_atoms"),
        masses       = masses,
        r_cut        = r_cut,
        timestep     = dt,
        n_steps      = n_steps,
        equil_steps  = equil,
        traj_every   = md_cfg.get("traj_every",   10),
        thermo_every = md_cfg.get("thermo_every", 100),
        output_dir   = output_dir,
        skin_width   = md_cfg.get("skin_width",   0.5),
        seed         = md_cfg.get("seed",         42),
        verbose      = verbose,
        # The MD templates all ship a `timing:` key; it used to be
        # ignored here, so setting `timing: false` had no effect.
        timing       = bool(md_cfg.get("timing", True)),
    )
    if force_fn is not None:
        common["force_fn"] = force_fn
    else:
        common.update(buck_params=buck_params, charges=charges,
                      ewald_alpha=pot_cfg.get("ewald_alpha"),
                      coulomb_method=pot_cfg.get("coulomb_method","ewald"),
                      pppm_accuracy=pot_cfg.get("pppm_accuracy",1e-5),
                      pppm_mesh=pot_cfg.get("pppm_mesh"),
                      pppm_order=pot_cfg.get("pppm_order",6))

    if ens == "nve":
        sim = NVESimulation(**common,
                            temperature=md_cfg.get("temperature",300.0))
    elif ens == "npt":
        sim = NPTSimulation(**common,
                            temperature=md_cfg.get("temperature",300.0),
                            pressure   =md_cfg.get("pressure",   0.0),
                            tau_nhc    =md_cfg.get("tau_nhc",    100.0),
                            tau_baro   =md_cfg.get("tau_baro",   1000.0),
                            velocity_rescale_steps=md_cfg.get(
                                "velocity_rescale_steps", 0))
    else:
        sim = NVTSimulation(**common,
                            temperature=md_cfg.get("temperature",300.0),
                            tau_nhc    =md_cfg.get("tau_nhc",    100.0),
                            velocity_rescale_steps=md_cfg.get(
                                "velocity_rescale_steps", 0))

    t0 = _time.perf_counter()
    result = sim.run()
    t_md = _time.perf_counter() - t0

    if not is_root:
        return 0

    # --- Results summary (rank 0 only) --------------------------------------
    _print_results_header(f"md ({ens.upper()})")
    thermo = result.get("thermo_log", [])
    # `>= equil`, matching the simulations' own `is_prod = step >=
    # self.equil_steps`.  `> equil` dropped the first production
    # frame from every CLI average.
    prod   = [r for r in thermo if r["step"] >= equil] or thermo
    if prod:
        T_v   = [r["T_inst"] for r in prod]
        Ep_v  = [r["e_pot"]  for r in prod]
        Ek_v  = [r["e_kin"]  for r in prod]
        H_v   = [h for h in [r.get("H_nhc", float("nan")) for r in prod]
                 if np.isfinite(h)]
        Hdr   = (max(H_v)-min(H_v)) if len(H_v) > 1 else float("nan")
        print(f"\n[Thermodynamics  ({len(prod)} production frames)]")
        print(f"  T_mean      : {np.nanmean(T_v):.2f} K  (target {md_cfg.get('temperature',300.0):.1f} K)")
        print(f"  T_std       : {np.nanstd(T_v):.2f} K")
        print(f"  T_min/max   : {np.nanmin(T_v):.2f} / {np.nanmax(T_v):.2f} K")
        print(f"  <E_pot>     : {np.nanmean(Ep_v):.4f} eV  ({np.nanmean(Ep_v)/N:.6f} eV/atom)")
        print(f"  <E_kin>     : {np.nanmean(Ek_v):.4f} eV")
        if np.isfinite(Hdr):
            print(f"  H_NHC drift : {Hdr:.4e} eV  ({Hdr/N:.4e} eV/atom)")
        if thermo:
            print(f"  NL builds   : {thermo[-1].get('nl_builds','?')}")
    else:
        # Only NVTSimulation accumulates a per-frame thermo_log; NVE and
        # NPT return flat summary keys instead.  Without this branch the
        # entire [Thermodynamics] block was silently skipped for those two
        # ensembles -- an NPT run reported neither pressure nor volume,
        # the two quantities the ensemble exists to control.
        print("\n[Thermodynamics  (run summary)]")
        if result.get("T_mean") is not None:
            # NVE has no target temperature -- `temperature` there is the
            # Maxwell-Boltzmann draw the velocities start from, and
            # equipartition sends about half of it into potential energy,
            # so an NVE run started at 600 K settles near 300 K.  Labelling
            # that "target 600.0 K" reads as a run that missed by 2x.
            t_set = md_cfg.get("temperature", 300.0)
            note = (f"(initial MB draw {t_set:.1f} K; equipartition halves it)"
                    if ens == "nve" else f"(target {t_set:.1f} K)")
            print(f"  T_mean      : {result['T_mean']:.2f} K  {note}")
            print(f"  T_std       : {result.get('T_std', float('nan')):.2f} K")
        if result.get("e_pot_mean") is not None:
            ep = result["e_pot_mean"]
            print(f"  <E_pot>     : {ep:.4f} eV  ({ep/N:.6f} eV/atom)")
        if result.get("P_mean_gpa") is not None:
            print(f"  P_mean      : {result['P_mean_gpa']:.4f} GPa  "
                  f"(target {md_cfg.get('pressure',0.0):.4f} GPa)")
            print(f"  P_std       : {result.get('P_std_gpa', float('nan')):.4f} GPa")
        if result.get("V_mean") is not None:
            print(f"  V_mean      : {result['V_mean']:.3f} A^3")
        for key, label in (("e_tot_drift", "E_tot drift "),
                           ("H_npt_drift", "H_NPT drift ")):
            if result.get(key) is not None:
                print(f"  {label}: {result[key]:.4e} eV  "
                      f"({result[key]/N:.4e} eV/atom)")

    _print_output_files(output_dir)

    td = result.get("timing", {})
    prod_ps = n_steps * dt / 1000.0
    sections = [(f"MD run ({total} steps, {N} atoms)", t_md)]
    _print_timing_summary(sections, t_md)
    if td and td.get("enabled") and td.get("n_steps", 0) > 0:
        print(f"\n  Per-step     : {td['per_step_ms']:.2f} ms  "
              f"({td['throughput_steps_per_s']:.1f} steps/s)")
        ps_s = prod_ps / t_md if t_md > 0 else 0
        print(f"  Performance  : {ps_s:.4f} ps/s  ({ps_s*86.4:.5f} ns/day)")
    return 0
