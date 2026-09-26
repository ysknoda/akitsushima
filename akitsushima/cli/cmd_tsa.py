"""``akitsushima tsa`` -- transition-state analysis (Hessian/normal-mode,
IRC, Eyring TST rate) from a YAML config.
"""
import os
import time as _time

from ..solvers import TransitionStateAnalyser
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

__all__ = ["cmd_tsa"]


def cmd_tsa(args) -> int:
    """Transition State Analysis from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1

    struct_cfg = cfg.get("structure",{})
    pot_cfg    = cfg.get("potential",{})
    tsa_cfg    = cfg.get("tsa",      {})
    verbose    = tsa_cfg.get("verbose",True) and not getattr(args,"quiet",False)

    poscar = struct_cfg.get("poscar","POSCAR_saddle")
    pos_data = _check_poscar(poscar)
    if pos_data is None:
        return 1

    force_fn, r_cut, _, _ = _build_force_fn_from_yaml(pot_cfg)
    _print_header("Transition State Analysis (TSA)", cfg, poscar, pot_cfg)
    delta = tsa_cfg.get("delta",0.01)
    T     = tsa_cfg.get("temperature",300.0)
    print(f"\n[Settings]")
    print(f"  delta         : {delta} A")
    print(f"  mass_weighted : {tsa_cfg.get('mass_weighted',True)}")
    print(f"  run_irc       : {tsa_cfg.get('run_irc',True)}")
    print(f"  temperature   : {T} K  (for TST rate)")
    print(f"  wigner        : {tsa_cfg.get('wigner',False)}")

    output_dir = tsa_cfg.get("output_dir","./tsa_output")
    os.makedirs(output_dir, exist_ok=True)
    masses = _resolve_masses(tsa_cfg, pos_data["atom_types"]) \
             if tsa_cfg.get("masses") else None
    t_sections = []

    tsa = TransitionStateAnalyser(
        lattice=pos_data["lattice"], positions=pos_data["positions"],
        atom_types=pos_data["atom_types"], force_fn=force_fn, masses=masses)

    t0 = _time.perf_counter()
    mode_result = tsa.analyse_modes(delta=delta,
                                    mass_weighted=tsa_cfg.get("mass_weighted",True))
    t_sections.append(("Hessian + mode analysis", _time.perf_counter()-t0))

    # Bug-fix note (S11): is_first_order_saddle's threshold argument used
    # to be silently ignored; it is now exposed here too (optional,
    # defaults to the solver's own -1e-3 default).
    is_saddle = tsa.is_first_order_saddle(
        delta=delta, threshold=tsa_cfg.get("threshold", -1e-3))

    irc_result = None
    if tsa_cfg.get("run_irc",True):
        t0 = _time.perf_counter()
        irc_result = tsa.run_irc(
            irc_delta=tsa_cfg.get("irc_delta",0.1),
            fmax=tsa_cfg.get("irc_fmax",0.05),
            max_steps=tsa_cfg.get("irc_max_steps",500),
            output_dir=output_dir, verbose=verbose)
        t_sections.append(("IRC path", _time.perf_counter()-t0))

    # Bug-fix note (S22, newly found during the port): reaction_rate()
    # requires either Ea= or E_init= (there is no way for it to compute
    # an activation energy on its own -- it only ever evaluates the
    # saddle-point structure's own energy), but the original call passed
    # neither, so this line raised ValueError unconditionally, on every
    # single cmd_tsa run. The natural E_init here is the lower-energy
    # side of the two IRC-relaxed minima (the "reactant" by definition);
    # when run_irc is disabled, fall back to an explicit tsa.E_init
    # config value, or skip the rate calculation entirely with a clear
    # message rather than crash.
    E_init = None
    if irc_result is not None:
        E_init = min(irc_result["e_forward"], irc_result["e_backward"])
    elif tsa_cfg.get("E_init") is not None:
        E_init = float(tsa_cfg["E_init"])

    k = None
    if E_init is not None:
        t0 = _time.perf_counter()
        k = tsa.reaction_rate(T=T, E_init=E_init, delta_hess=delta,
                              wigner=tsa_cfg.get("wigner",False))
        t_sections.append(("Reaction rate (TST)", _time.perf_counter()-t0))
    else:
        print("\n  Skipping TST rate: no E_init available "
              "(enable tsa.run_irc, or set tsa.E_init explicitly).")

    tsa.report(output_dir=output_dir, delta=delta)

    _print_results_header("tsa")
    print(f"\n[Saddle-point analysis]")
    print(f"  First-order saddle : {is_saddle}")
    # Bug-fix note (S7): the original checked mode_result for
    # "imaginary_freq_cm"/"n_imag", but analyse_modes() actually returns
    # "imag_freq_cm"/"n_negative" -- so these two lines never printed.
    print(f"  Imag. frequency    : {mode_result['imag_freq_cm']:.2f} cm^-1")
    print(f"  # negative modes   : {mode_result['n_negative']}")
    if k is not None:
        print(f"\n[Reaction rate  (T={T:.0f} K)]")
        print(f"  k (TST)            : {k:.4e} s^-1")
        # ADR-0003 (tsa finding): say which approximation this is -- for
        # the SW Si vacancy hop the harmonic (Vineyard) rate is 11x / 5x
        # larger (63 / 215 atoms, example/neb/check_neb_tsa.json).
        print("                       (k_B T / h) exp(-E_a / k_B T): "
              "Q_TS/Q_IS = 1 assumed,")
        print("                       not the harmonic (Vineyard) "
              "prefactor from the mode frequencies")
    if irc_result is not None:
        # Bug-fix note (S6/S7): run_irc() (fixed in S6 to return a proper
        # dict instead of a bare tuple) returns e_forward/e_backward/
        # converged_forward/converged_backward -- not E_ts/E_min1/E_min2,
        # which this summary used to read (always NaN). E_ts is the
        # saddle-point structure's own energy, obtained with one extra
        # force_fn call (matching the plan's fix for this command).
        E_ts, _, _ = force_fn(pos_data["lattice"], pos_data["positions"],
                              pos_data["atom_types"])
        E_min1 = irc_result["e_forward"]
        E_min2 = irc_result["e_backward"]
        print(f"\n[IRC path]")
        print(f"  E_TS    : {float(E_ts):.4f} eV")
        print(f"  E_min1  : {E_min1:.4f} eV  (forward side, "
              f"converged={irc_result['converged_forward']})")
        print(f"  E_min2  : {E_min2:.4f} eV  (backward side, "
              f"converged={irc_result['converged_backward']})")
        print(f"  Ea_fwd  : {float(E_ts)-E_min1:.4f} eV")
        print(f"  Ea_rev  : {float(E_ts)-E_min2:.4f} eV")

    _print_output_files(output_dir)
    _print_timing_summary(t_sections, sum(t for _,t in t_sections))
    return 0
