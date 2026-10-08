"""``akitsushima nemd`` -- non-equilibrium MD with two Langevin heat baths:
temperature profile, local heat flux, kappa from the gradient and Kapitza
resistances (ADR-0020).
"""
import json
import os
import time as _time

import numpy as np

from ..solvers.nemd import NEMDSimulation
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

__all__ = ["cmd_nemd"]

# Steady-state indicators reported (warnings only; ADR-0020, Q27).
_BALANCE_WARN = 0.05
_PROFILE_WARN_K = 1.0


def cmd_nemd(args) -> int:
    """NEMD from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1
    pot_cfg = cfg.get("potential", {})
    sec = cfg.get("nemd", {}) or {}
    poscar = cfg.get("structure", {}).get("poscar", "POSCAR")
    pos = _check_poscar(poscar)
    if pos is None:
        return 1
    is_root = (_mpi_rank == 0)
    verbose = sec.get("verbose", True) and not getattr(args, "quiet", False) and is_root
    out_dir = sec.get("output_dir", "./nemd_output")
    os.makedirs(out_dir, exist_ok=True)
    fn, _ = _cli_force_fn(pot_cfg, "cmd_nemd")
    if is_root:
        _print_header("Non-equilibrium MD (nemd)", cfg, poscar, pot_cfg)
    hot, cold = sec.get("hot", {}) or {}, sec.get("cold", {}) or {}
    sim = NEMDSimulation(
        pos["lattice"], pos["positions"], pos["atom_types"], fn,
        masses=_resolve_masses(sec, pos["atom_types"]),
        T_hot=hot.get("temperature", 330.0), T_cold=cold.get("temperature", 270.0),
        axis=sec.get("axis", "c"),
        hot=hot.get("region", [0.0, 0.05]), cold=cold.get("region", [0.5, 0.55]),
        hot_atoms=None if hot.get("atoms") is None else [i - 1 for i in hot["atoms"]],
        cold_atoms=None if cold.get("atoms") is None else [i - 1 for i in cold["atoms"]],
        damping=sec.get("damping", 100.0), timestep=sec.get("timestep", 1.0),
        n_bins=sec.get("n_bins", 20), local_flux=sec.get("local_flux", True),
        seed=sec.get("seed", 42))
    if is_root:
        print(f"\n[Settings]")
        print(f"  axis {sec.get('axis', 'c')}: length {sim.length:.3f} A, "
              f"cross-section {sim.area:.3f} A^2")
        print(f"  hot  {sim.T_hot} K, {len(sim.hot)} atoms;  cold {sim.T_cold} K, "
              f"{len(sim.cold)} atoms;  damping {sim.damping} fs")
        print(f"  dt {sim.dt} fs, equilibration {sec.get('equil_steps', 20000)}, "
              f"production {sec.get('n_steps', 100000)} steps")
    ckpt = sec.get("checkpoint", os.path.join(out_dir, "nemd_checkpoint.npz"))
    if sec.get("restart") is False and os.path.isfile(ckpt):
        os.remove(ckpt)
    t0 = _time.perf_counter()
    sim.run(int(sec.get("equil_steps", 20000)), int(sec.get("n_steps", 100000)),
            sample_every=int(sec.get("sample_every", 10)), checkpoint=ckpt,
            checkpoint_every=int(sec.get("checkpoint_every", 1000)),
            thermo_path=os.path.join(out_dir, "nemd_thermo.log"), verbose=verbose)
    t_sections = [("NEMD run", _time.perf_counter() - t0)]
    fit = sec.get("fit", {}) or {}
    kap = sec.get("kapitza", {}) or {}
    r = sim.results(fit_fraction=fit.get("fraction", 0.5),
                    interfaces=kap.get("interfaces", []) or [],
                    kapitza_window=kap.get("window", 0.1),
                    kapitza_skip=kap.get("skip_bins", 1))
    if not is_root:
        return 0
    out = {"command": "nemd",
           "input_hash": _input_hash(pos["lattice"], pos["positions"],
                                     pos["atom_types"], pot_cfg)}
    for k, v in r.items():
        out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    with open(os.path.join(out_dir, "nemd_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    np.savetxt(os.path.join(out_dir, "nemd_profile.dat"),
               np.column_stack([r["z_frac"], r["z_A"], r["T_K"], r["atoms_per_bin"],
                                r["local_flux_W_m2"]]), fmt="%.8g",
               header="z_frac z_A T_K atoms_per_bin local_heat_flux_W_m2 (along +axis)")

    _print_results_header("nemd")
    print(f"\n[Temperature profile and local flux]")
    print(f"  {'z (A)':>8} {'T (K)':>8} {'J_local (MW/m^2)':>17}")
    for zz, T, j in zip(r["z_A"], r["T_K"], r["local_flux_W_m2"]):
        print(f"  {zz:8.2f} {T:8.2f} {j / 1e6:17.3f}")
    print(f"\n[Heat flux]  from hot bath {r['flux_hot_W_m2'] / 1e6:.3f}, from cold "
          f"bath {r['flux_cold_W_m2'] / 1e6:.3f} MW/m^2 (Qdot / 2A)")
    for sgm in r["segments"]:
        print(f"  {sgm['segment']:>20s}: dT/dz {sgm['gradient_K_per_A']:+.4f} K/A, "
              f"kappa {sgm['kappa_W_mK']:.4f} +/- {sgm['kappa_err_W_mK']:.4f} W/m/K")
    print(f"  kappa (mean of the two segments): {r['kappa_W_mK']:.4f} W/m/K "
          f"(one length: extrapolate 1/kappa vs 1/L over several lengths)")
    for k in r["kapitza"]:
        print(f"  interface at {k['interface_frac']:.3f}: dT {k['delta_T_K']:+.3f} K, "
              f"R_K {k['R_m2K_per_W']:.3e} m^2 K/W")
    print(f"\n[Steady state]  bath balance |Qh + Qc| / |Qh| = {r['bath_balance']:.3f}; "
          f"profile change between the last blocks "
          f"{max(r['profile_change_last_blocks_K']):.3f} K; energy drift beyond "
          f"the baths {r['E_minus_Q_drift_eV']:.3e} eV")
    if not all(sg["gradient_sign_ok"] for sg in r["segments"]):
        print("  WARNING: a segment's temperature gradient has the wrong sign "
              "(T should fall from the hot to the cold bath); its kappa is "
              "meaningless.")
    if r["bath_balance"] > _BALANCE_WARN or max(r["profile_change_last_blocks_K"]) > _PROFILE_WARN_K:
        print(f"  WARNING: not at steady state (balance > {_BALANCE_WARN} or profile "
              f"change > {_PROFILE_WARN_K} K): lengthen equil_steps / n_steps.")
    _print_output_files(out_dir)
    _print_timing_summary(t_sections, sum(t for _, t in t_sections))
    return 0
