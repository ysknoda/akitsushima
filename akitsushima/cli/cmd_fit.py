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
from typing import Dict

import numpy as np

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
from ..io.poscar import parse_poscar
from ..solvers.config_bridge import _load_cfg, _write_cfg
from ..utils.mpi import _mpi_rank
from .templates import _DEFAULT_FIT_CFG, _FIT_CFG_TEMPLATES

__all__ = ["cmd_fit"]


# ============================================================================
#  Layout B: reconstruct structures from a training CSV + poscar_dir
# ============================================================================

def _parse_geometry_layout_b(unique_ids: list, poscar_dir: str) -> Dict:
    """
    Layout B (only supported layout for classical force-field fitting).

    Loads geometry from POSCAR files located at::

        poscar_dir/<struct_idx>/POSCAR

    Parameters
    ----------
    unique_ids : sorted list of integer structure indices from the CSV.
    poscar_dir : root directory containing numbered sub-directories.

    Returns
    -------
    dict  {struct_id: structure_dict}

    Raises
    ------
    FileNotFoundError
        When ``poscar_dir/<sid>/POSCAR`` does not exist.
    """
    struct_map: Dict = {}
    for sid in unique_ids:
        poscar_path = os.path.join(poscar_dir, str(sid), "POSCAR")
        if not os.path.isfile(poscar_path):
            raise FileNotFoundError(
                f"POSCAR not found for structure {sid}: '{poscar_path}'\n"
                f"  Expected: {poscar_dir}/<struct_idx>/POSCAR\n"
                f"  Ensure poscar_dir is set correctly in the config "
                f"(default: directory of training_csv).")
        pd = parse_poscar(poscar_path)
        struct_map[sid] = {
            "name"       : str(sid),
            "lattice"    : pd["lattice"],
            "positions"  : pd["positions"],
            "atom_types" : pd["atom_types"],
            "natoms"     : pd["natoms"],
            "volume"     : float(abs(np.linalg.det(pd["lattice"]))),
            "fix_atoms"  : pd["fix_atoms"],
            "ref_energy" : None,
            "ref_forces" : None,
            "ref_stress" : None,
        }
    return struct_map


def _fill_efs_from_csv(struct_map    : Dict,
                        targets       : np.ndarray,
                        categories    : np.ndarray,
                        struct_idx_col: np.ndarray) -> list:
    """
    Fill ref_energy / ref_forces / ref_stress into the struct_map.

    Runs in O(M) time (M = total number of CSV rows) by pre-grouping
    rows per structure rather than rescanning the full array per row.

    Returns a list of structure dicts that have at least ref_energy set.
    """
    # Pre-group row indices by (struct_id, category) in one O(M) pass
    from collections import defaultdict
    grp: dict = defaultdict(list)   # (sid, cat) -> [row_indices]
    for row_idx, (sid, cat) in enumerate(zip(struct_idx_col, categories)):
        grp[(int(sid), str(cat))].append(row_idx)

    for sid, s in struct_map.items():
        # Energy (1 row expected)
        e_rows = grp.get((sid, 'E'), [])
        if e_rows:
            s['ref_energy'] = float(targets[e_rows[-1]])  # last wins if dup

        # Forces (3N rows expected)
        f_rows = grp.get((sid, 'F'), [])
        if f_rows:
            N      = s['natoms']
            f_vals = targets[np.array(f_rows, dtype=int)]
            if len(f_vals) == N * 3:
                s['ref_forces'] = f_vals.reshape(N, 3)

        # Stress (6 Voigt or 9 full rows expected)
        s_rows = grp.get((sid, 'S'), [])
        if s_rows:
            s_vals = targets[np.array(s_rows, dtype=int)]
            if len(s_vals) == 6:        # Voigt order: xx yy zz yz zx xy
                xx, yy, zz, yz, xz, xy = s_vals
                s['ref_stress'] = np.array(
                    [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            elif len(s_vals) == 9:
                s['ref_stress'] = s_vals.reshape(3, 3)

    return [s for s in struct_map.values() if s['ref_energy'] is not None]


def _structures_from_csv(targets    : np.ndarray,
                          categories : np.ndarray,
                          X_desc     : np.ndarray,
                          poscar_dir : str,
                          verbose    : bool = True) -> list:
    """
    Load classical force-field training structures using Layout B.

    **Layout B is the only supported CSV format** for classical force-field
    fitting.  The CSV must have exactly **one** descriptor column (col 2)
    containing an integer structure index.  Geometry is read from the
    corresponding POSCAR file::

        poscar_dir/<struct_idx>/POSCAR

    CSV format
    ----------
    ::

        col 0  target value  [eV/atom | eV/A | eV/A^3]   (float)
        col 1  category      "E" | "F" | "S"             (str)
        col 2  struct_idx    integer structure index      (int)

    Row ordering within one structure (N atoms):

    * 1 E row  -- total energy [eV/atom]; multiplied by N internally
    * 3N F rows -- force components, row-major (atom 1 x/y/z, atom 2 ...) [eV/A]
    * 6 S rows  -- Voigt stress: xx yy zz yz zx xy [eV/A^3],
      compressive-positive (a compressed cell has positive diagonal
      stress -- the engine's own convention, docs/adr/0001)

    If S rows are derived from a real VASP OUTCAR "in kB" line, apply a
    unit conversion and a REORDERING, but NO sign flip.  VASP's raw value
    is positive under compression, exactly like the engine's.  Multiply
    by ``akitsushima.constants.VASP_KB_TO_EV_ANG3`` (= +1/1602.176621)
    and reorder VASP's xx yy zz xy yz zx into xx yy zz yz zx xy.  (This
    docstring used to require a sign flip and point at a constant of
    -1/1602.176621; see that constant's bug-fix note.)

    Note that the CSV has FOUR columns, with a header row:
    ``target, label, weight, struct_idx``.  The ``weight`` column is
    read by :func:`load_mlip_csv` but not used by ``fit``: classical-
    potential residuals are normalised by the config's
    ``weights: {energy, forces, stress, r0, eps0}`` instead.

    Parameters
    ----------
    targets     : (M,) float  -- target values from CSV col 0.
    categories  : (M,) str    -- "E", "F", or "S" from CSV col 1.
                  (Parameter named ``labels`` in :func:`load_mlip_csv`
                  return value; both names refer to the same array.)
    X_desc      : (M, 1) float -- struct_idx column (col 2 of the CSV).
    poscar_dir  : root directory containing numbered POSCAR sub-directories.
    verbose     : print a summary line when True.

    Returns
    -------
    list of structure dicts with keys:
        name, lattice, positions, atom_types, natoms, volume,
        fix_atoms, ref_energy, ref_forces, ref_stress.

    Raises
    ------
    ValueError
        When the CSV has zero or more than one descriptor column ---
        i.e. Layout A data or an ELM descriptor CSV was passed.
    FileNotFoundError
        When a required POSCAR file is missing.
    RuntimeError
        When no structures with energy rows are found.
    """
    if X_desc.shape[1] == 0:
        raise ValueError(
            "_structures_from_csv: CSV has no descriptor column (col 2).\n"
            "  Layout B requires exactly one column: the integer structure index.")

    if X_desc.shape[1] > 1:
        raise ValueError(
            f"_structures_from_csv: CSV has {X_desc.shape[1]} descriptor "
            f"columns (cols 2+), but Layout B requires exactly 1 column "
            f"(the integer structure index).\n"
            f"  - If you passed an ELM descriptor CSV by mistake, use "
            f"ELMModel.fit() instead.\n"
            f"  - If you intended Layout B, keep only col 2 (struct_idx) "
            f"and place geometry in poscar_dir/<struct_idx>/POSCAR.")

    struct_idx_col = X_desc[:, 0].astype(int)
    unique_ids     = sorted(set(struct_idx_col))

    struct_map = _parse_geometry_layout_b(unique_ids, poscar_dir)
    result     = _fill_efs_from_csv(struct_map, targets, categories,
                                     struct_idx_col)

    if not result:
        raise RuntimeError(
            "_structures_from_csv: no structures with energy rows found.\n"
            "  Check that at least one row per structure has category 'E'.")

    if verbose:
        n_f = sum(1 for s in result if s["ref_forces"] is not None)
        n_s = sum(1 for s in result if s["ref_stress"] is not None)
        print(f"  Loaded {len(result)} structures from Layout B CSV  "
              f"(forces: {n_f}  stress: {n_s})")
    return result


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

    potential = getattr(args, 'potential', 'buckingham') or 'buckingham'
    potential = potential.lower()

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
