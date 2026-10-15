"""``akitsushima fit`` -- classical potential parameter fitting (LJ / Morse
/ BornMayer / Buckingham / Stillinger-Weber / Tersoff) from a training CSV
+ per-structure POSCAR directory.

Consolidation note (F1, per project decision): the original script also had
a separate ``fitpot`` sub-command for classical potentials. Its
implementation called ``ParameterSet(**flat_kwargs)`` (every ParameterSet
actually takes one ``initial_*: dict`` argument, not flattened keyword
arguments -- immediate TypeError) and passed a directory path where
``fit`` expects a training CSV file path (immediate "file not found"
failure) -- it never worked. Rather than fix a second, redundant code
path, ``fitpot`` has been removed entirely and every potential type is
fit through this single ``fit`` command (see also F3: the CLI parser's
``--potential`` choices now include all six types, not just Buckingham).
"""
import argparse
import contextlib
import os
from ..fitting import registry
from ..fitting.driver import (
    DEFAULT_LOCAL_METHOD,
    evaluate_diagnostics_generic,
    fit_potential,
    print_fit_result,
    resolve_local_method,
    save_fit_result_generic,
)
from ..fitting.models.csv_io import load_mlip_csv
from ..io.training_set import _structures_from_csv  # moved there (ADR-0024)
from ..solvers.config_bridge import _load_cfg, _write_cfg
from ..utils.mpi import _mpi_rank
from .templates import _DEFAULT_FIT_CFG, _FIT_CFG_TEMPLATES

__all__ = ["cmd_fit"]


# ============================================================================
#  Layout B: reconstruct structures from a training CSV + poscar_dir
# ============================================================================

# ============================================================================
#  ParameterSet construction
# ============================================================================

def _build_param_set(potential  : str,
                      cfg        : dict,
                      all_species: list,
                      pairs      : list,
                      r_cut      : float):
    """
    Instantiate the appropriate FittableParameterSet from a config dict.

    Parameters
    ----------
    potential   : potential name string (lowercase).
    cfg         : config dict (loaded from JSON or defaults).
    all_species : list of element symbols found in training data.
    pairs       : list of frozenset pairs to fit.
    r_cut       : cutoff radius [A].

    Returns
    -------
    FittableParameterSet subclass instance.

    Raises
    ------
    KeyError   when required config keys are missing.
    ValueError when the potential string is unknown.

    Consolidation note (Phase 1, potential registry): this used to be a
    ~120-line if/elif chain, one branch per potential type -- one of 4
    locations that had to be kept manually in sync whenever a potential
    was added (see fitting/registry.py's module docstring for the other
    3, and for where each branch's logic actually moved to).
    """
    return registry.get(potential).build_param_set(cfg, all_species, pairs, r_cut)


# ============================================================================
#  cmd_fit
# ============================================================================

def cmd_fit(args: argparse.Namespace) -> int:
    """Handle the 'fit' sub-command (multi-potential parameter fitting).

    Under mpirun every rank runs the fit (the loss and the diagnostics are
    collective), but only rank 0 prints: the other ranks' stdout is
    discarded, and fitting.driver writes files on rank 0 only.  A
    --write-config template is written by rank 0 alone.
    """
    if _mpi_rank == 0:
        return _cmd_fit(args)
    if args.write_config:
        return 0
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        return _cmd_fit(args)


def _cmd_fit(args: argparse.Namespace) -> int:

    # --potential wins; otherwise the config's own `potential:` key, and
    # only then buckingham.  It used to be --potential's default
    # (buckingham) that silently replaced the config's key (ADR-0024 O4).
    potential = getattr(args, 'potential', None)
    if not potential and args.config and not args.write_config \
            and os.path.isfile(args.config):
        potential = _load_cfg(args.config, {}).get("potential")
    potential = str(potential or 'buckingham').lower()

    # -- Write template config and exit --------------------
    if args.write_config:
        tmpl = _FIT_CFG_TEMPLATES.get(potential, _DEFAULT_FIT_CFG)
        variant = getattr(args, "template", None)
        if variant:
            variants = registry.get(potential).templates
            if variant not in variants:
                print(f"ERROR: no template '{variant}' for {potential}; "
                      f"available: {sorted(variants) or 'none'}")
                return 1
            tmpl = variants[variant]
        _write_cfg(args.write_config, tmpl)
        return 0

    # -- Load config ---------------------------------------
    default_cfg = _FIT_CFG_TEMPLATES.get(potential, _DEFAULT_FIT_CFG)
    cfg = (_load_cfg(args.config, default_cfg)
           if args.config else dict(default_cfg))

    # CLI overrides
    if getattr(args, "training_csv", None):
        cfg["training_csv"] = args.training_csv
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    # Bug-fix note (F12, CLI remainder): `if args.r_cut:` is a truthiness
    # check, so `--r-cut 0` (a legitimate, if unusual, explicit override)
    # was silently ignored and the config/default value used instead.
    # argparse's own default for --r-cut is None (see parser.py), so
    # `is not None` correctly distinguishes "not provided" from
    # "explicitly provided as 0".
    if args.r_cut is not None:
        cfg["r_cut"] = args.r_cut
    if getattr(args, "r0",   None) is not None:
        cfg["r0"]   = args.r0
    if getattr(args, "eps0", None) is not None:
        cfg["eps0"] = args.eps0
    if args.use_de:
        cfg["use_de"] = True
    # Merge potential from CLI (takes precedence over config file)
    cfg["potential"] = potential
    verbose = not args.quiet

    # Checked before any data is read: fit_potential validates too, but
    # only after the CSV and every POSCAR have been loaded.
    try:
        resolve_local_method(cfg.get("local_method", DEFAULT_LOCAL_METHOD))
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    # -- Load training data --------------------------------
    csv_path = cfg.get("training_csv", "")
    if not os.path.isfile(csv_path):
        print(f"ERROR: Training CSV not found: '{csv_path}'")
        return 1

    print(f"\n  Potential      : {potential.upper()}")
    print(f"  Training CSV   : {csv_path}")
    targets, labels, weights, X_desc = load_mlip_csv(csv_path, verbose=verbose)

    try:
        structures = _structures_from_csv(
            targets, labels, X_desc,
            cfg.get("poscar_dir", os.path.dirname(csv_path)),
            verbose=verbose)
    except Exception as exc:
        print(f"ERROR: Could not reconstruct structures from CSV: {exc}")
        return 1

    if getattr(args, "no_forces", False):
        for s in structures:
            s["ref_forces"] = None
    if getattr(args, "no_stress", False):
        for s in structures:
            s["ref_stress"] = None

    all_species = sorted({a for s in structures for a in s["atom_types"]})
    print(f"  Detected species: {all_species}")
    print(f"  Structures      : {len(structures)}")

    # -- Build ParameterSet --------------------------------
    r_cut   = float(cfg["r_cut"])
    pairs   = [frozenset({s1, s2})
               for i, s1 in enumerate(all_species)
               for s2 in all_species[i:]]

    try:
        param_set = _build_param_set(potential, cfg, all_species, pairs, r_cut)
    except (KeyError, ValueError) as exc:
        print(f"ERROR: Cannot build {potential.upper()} ParameterSet: {exc}")
        print("  Tip: use --write-config to generate a template config file.")
        return 1

    # -- Run fitting ---------------------------------------
    # Merge r0/eps0 from cfg into the weights dict so that
    # generic_loss_function receives the correct unit-conversion factors.
    # r0 and eps0 may be specified either as top-level cfg keys or inside
    # the "weights" sub-dict; the top-level value takes precedence.
    _weights = dict(cfg.get("weights", {"energy": 1.0, "forces": 1.0, "stress": 0.1}))
    _weights.setdefault("r0",   float(cfg.get("r0",   _weights.get("r0",   1.0))))
    _weights.setdefault("eps0", float(cfg.get("eps0", _weights.get("eps0", 1.0))))
    # Allow top-level cfg["r0"]/cfg["eps0"] to override weights sub-dict values
    if "r0"   in cfg: _weights["r0"]   = float(cfg["r0"])
    if "eps0" in cfg: _weights["eps0"] = float(cfg["eps0"])

    # Phase 3: optional robust (Huber) loss -- opt-in via
    # cfg["loss_type"]: "huber"; the three huber_delta_* keys must then
    # also be present (top-level cfg or inside the weights sub-dict, top
    # level taking precedence), matching generic_loss_function's own
    # explicit-required-delta contract (no silent default is guessed).
    loss_type = cfg.get("loss_type", "squared")
    if loss_type == "huber":
        for key in ("huber_delta_energy", "huber_delta_forces", "huber_delta_stress"):
            if key in cfg:
                _weights[key] = float(cfg[key])

    # Bug-fix note (consolidation, see module docstring / F1): the
    # original branched here into a Buckingham-specific `run_fitting` +
    # `evaluate_diagnostics` + `save_fit_results` + `print_parameters`
    # stack, kept alongside the generic `fit_potential` path used by
    # every other potential type. That whole Buckingham-only stack was
    # already nothing but a thin wrapper around `fit_potential` (removed
    # in fitting/driver.py -- see its module docstring), so every
    # potential type, including Buckingham, now goes through the exact
    # same generic path below.
    result = fit_potential(
        param_set    = param_set,
        structures   = structures,
        r_cut        = r_cut,
        ewald_alpha  = cfg.get("ewald_alpha"),
        weights      = _weights,
        loss_type    = loss_type,
        method       = cfg.get("local_method", DEFAULT_LOCAL_METHOD),
        use_de       = cfg.get("use_de", False),
        de_maxiter   = cfg.get("de_maxiter", 300),
        de_popsize   = cfg.get("de_popsize", 10),
        val_fraction = cfg.get("val_fraction", 0.2),
        val_seed     = cfg.get("val_seed", 0),
        output_dir   = cfg["output_dir"],
        verbose      = verbose,
        restarts     = int(cfg.get("restarts", 0)),
        restart_rtol = float(cfg.get("restart_rtol", 1e-3)),
        local_maxiter= cfg.get("local_maxiter"),
        restart_seed = int(cfg.get("restart_seed", 0)),
        time_limit_hours = cfg.get("time_limit_hours"),
    )
    print_fit_result(result, potential)
    # Bug-fix note (F5): evaluate_diagnostics_generic requires the FITTED
    # parameters (result["params"]) as an explicit argument -- the
    # original omitted it, so the positional r_cut/output_dir arguments
    # silently shifted into the wrong parameter slots (this function's
    # actual signature is (structures, param_set, fitted_params, r_cut,
    # output_dir, ...)) on top of always evaluating diagnostics at the
    # untrained initial guess x0 rather than the fitted result.
    diag = evaluate_diagnostics_generic(
        result["train_structs"], param_set, result["params"], r_cut,
        cfg["output_dir"],
        ewald_alpha=cfg.get("ewald_alpha"),
        val_structs=result.get("val_structs"))
    # Fixed (not-fit) construction hyperparameters that must be persisted
    # verbatim to reconstruct an identical potential on load -- Mie's
    # n/m exponents and Yukawa's kappa are not part of the fitted
    # `params` dict itself (see MieParameterSet/YukawaParameterSet).
    extra_meta = None
    if potential == "mie":
        extra_meta = {"n": cfg.get("n", 12.0), "m": cfg.get("m", 6.0)}
    elif potential == "yukawa":
        extra_meta = {"kappa": cfg.get("kappa", 1.0)}
    save_fit_result_generic(result, potential, cfg["output_dir"], diag,
                            extra_meta=extra_meta, r_cut=r_cut,
                            species=all_species)

    return 0
