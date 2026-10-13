"""``akitsushima anneal`` -- MD-quench annealing or basin hopping (ADR-0022).

The YAML has the usual ``structure`` and ``potential`` sections and an
``anneal`` section.  ``anneal.md`` and ``anneal.relax`` take the same key
names as the ``md`` and ``relax`` sections of those commands' configs, so
settings can be copied across; keys the driver does not use are refused
rather than silently ignored.
"""

import time as _time

from ..fitting.force_fn import _resolve_force_fn
from ..solvers.config_bridge import _build_force_fn_from_yaml
from ..solvers.global_opt import GlobalOptimiser
from ..utils.mpi import _mpi_rank
from .common import (
    _check_poscar,
    _check_yaml,
    _fmt_time,
    _print_header,
    _print_output_files,
    _print_results_header,
    _resolve_masses,
)

__all__ = ["cmd_anneal"]

# relax-section keys GeometryOptimiser takes (output_dir / verbose are set
# per quench by the driver).
_RELAX_KEYS = {
    "method",
    "relax_cell",
    "check_forces",
    "check_stress",
    "check_energy",
    "fmax",
    "pmax",
    "emax",
    "max_steps",
    "skin_width",
    "max_displacement",
    "dt_start",
    "dt_max",
    "cell_update_scheme",
    "lattice_constraints",
    "memory",
    "fix_volume",
    "timing",
}
_RELAX_DEFAULTS = {"method": "fire", "fmax": 0.05, "max_steps": 2000}

# md-section keys NVTSimulation / NPTSimulation take; temperature, steps,
# output_dir, seed and velocities come from the driver.
_MD_KEYS = {
    "ensemble",
    "timestep",
    "tau_nhc",
    "tau_baro",
    "traj_every",
    "thermo_every",
    "skin_width",
    "masses",
    "timing",
}

_ANNEAL_KEYS = {
    "method",
    "output_dir",
    "seed",
    "pressure",
    "restart",
    "max_consecutive_failures",
    "min_energy_per_atom",
    "keep_best",
    "dedup_energy",
    "dedup_volume",
    "schedule",
    "segment_steps",
    "quench_every",
    "md",
    "basin_hopping",
    "relax",
    "verbose",
    "plot",
}
_BH_KEYS = {
    "n_trials",
    "temperature",
    "step",
    "cell_step",
    "adapt_every",
    "adapt_factor",
    "target_acceptance",
}


def _unknown(section, given, allowed):
    bad = sorted(set(given) - allowed)
    if bad:
        raise ValueError(
            f"anneal: unknown key(s) in {section}: {', '.join(bad)} "
            f"(allowed: {', '.join(sorted(allowed))})"
        )


def cmd_anneal(args) -> int:
    """Global structure search from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1
    struct_cfg = cfg.get("structure", {})
    pot_cfg = cfg.get("potential", {})
    an = dict(cfg.get("anneal", {}) or {})
    is_root = _mpi_rank == 0
    verbose = an.get("verbose", True) and not getattr(args, "quiet", False) and is_root

    try:
        _unknown("anneal", an, _ANNEAL_KEYS)
        relax_cfg = dict(an.get("relax", {}) or {})
        md_cfg = dict(an.get("md", {}) or {})
        bh_cfg = dict(an.get("basin_hopping", {}) or {})
        _unknown("anneal.relax", relax_cfg, _RELAX_KEYS)
        _unknown("anneal.md", md_cfg, _MD_KEYS)
        _unknown("anneal.basin_hopping", bh_cfg, _BH_KEYS)
    except ValueError as err:
        print(f"[ERROR] {err}")
        return 1

    poscar = struct_cfg.get("poscar", "POSCAR")
    pos_data = _check_poscar(poscar)
    if pos_data is None:
        return 1

    force_fn, r_cut, buck_params, charges = _build_force_fn_from_yaml(pot_cfg)
    force_fn, r_cut = _resolve_force_fn(
        force_fn,
        r_cut,
        buck_params,
        charges,
        pot_cfg.get("ewald_alpha"),
        pot_cfg.get("coulomb_method", "ewald"),
        pot_cfg.get("pppm_accuracy", 1e-5),
        pot_cfg.get("pppm_mesh"),
        pot_cfg.get("pppm_order", 6),
        "cmd_anneal",
    )

    relax_kw = {**_RELAX_DEFAULTS, **relax_cfg}
    md_kw = {k: v for k, v in md_cfg.items() if k != "masses"}
    md_kw["masses"] = _resolve_masses(md_cfg, pos_data["atom_types"])
    method = str(an.get("method", "md_quench")).lower()
    output_dir = an.get("output_dir", "./anneal_output")

    if is_root:
        _print_header(f"Anneal ({method})", cfg, poscar, pot_cfg)
        print("\n[Settings]")
        print(f"  Method     : {method}")
        print(f"  Pressure   : {an.get('pressure', 0.0)} GPa")
        print(
            f"  Relax      : {relax_kw.get('method')}  fmax {relax_kw.get('fmax')} eV/A  "
            f"relax_cell {relax_kw.get('relax_cell', False)}"
        )
        if method == "md_quench":
            print(f"  Schedule   : {an.get('schedule', {})}")
            print(
                f"  MD         : {md_cfg.get('ensemble', 'nvt').upper()}  "
                f"segment_steps {an.get('segment_steps', 2000)}  "
                f"quench_every {an.get('quench_every', 500)}"
            )
        else:
            print(f"  BH         : {bh_cfg}")
        print(f"  Restart    : {an.get('restart', False)}")

    try:
        opt = GlobalOptimiser(
            pos_data["lattice"],
            pos_data["positions"],
            pos_data["atom_types"],
            force_fn,
            r_cut,
            method=method,
            output_dir=output_dir,
            relax=relax_kw,
            md=md_kw,
            schedule=an.get("schedule"),
            segment_steps=an.get("segment_steps", 2000),
            quench_every=an.get("quench_every", 500),
            pressure=an.get("pressure", 0.0),
            seed=an.get("seed", 42),
            restart=an.get("restart", False),
            max_consecutive_failures=an.get("max_consecutive_failures", 5),
            min_energy_per_atom=an.get("min_energy_per_atom"),
            keep_best=an.get("keep_best", 10),
            dedup_energy=an.get("dedup_energy", 1e-4),
            dedup_volume=an.get("dedup_volume", 1e-3),
            fix_atoms=pos_data.get("fix_atoms"),
            verbose=verbose,
            plot=an.get("plot", True),
            **bh_cfg,
        )
    except ValueError as err:
        print(f"[ERROR] {err}")
        return 1

    t0 = _time.perf_counter()
    result = opt.run()
    t_run = _time.perf_counter() - t0
    if not is_root:
        return 0

    _print_results_header("anneal")
    N = pos_data["natoms"]
    minima = result["minima"]
    if minima:
        first = [m for m in minima if m["cycle"] == 0]
        print(
            f"\n[Distinct minima]  {len(minima)} found "
            f"(written: {min(len(minima), opt.keep_best)})"
        )
        for r, m in enumerate(minima[: opt.keep_best], start=1):
            print(
                f"  {r:2d}  cycle {m['cycle']:5d}  E/N {m['E'] / N:.6f} eV  "
                f"H/N {m['H'] / N:.6f} eV  V {m['V']:.3f} A^3"
            )
        best = minima[0]
        print(f"\n[Best]  E/N = {best['E'] / N:.6f} eV  (cycle {best['cycle']})")
        if first:
            print(
                f"  relax only (cycle 0): E/N = {first[0]['E'] / N:.6f} eV  "
                f"-> {(best['H'] - first[0]['H']) / N * 1e3:+.2f} meV/atom"
            )
    else:
        print("\n[Distinct minima]  none converged")
    print(f"  Failed cycles : {result['n_failed']}")
    if "acceptance" in result:
        print(f"  Acceptance    : {result['acceptance']:.3f}")
    print(f"  Wall time     : {_fmt_time(t_run)}")
    _print_output_files(output_dir)
    if result["aborted"]:
        print("[ERROR] anneal stopped after max_consecutive_failures failures in a row")
        return 1
    return 0 if minima else 1
