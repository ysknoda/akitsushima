"""``akitsushima mlip`` -- fit a machine-learning interatomic potential
(ELM / QELM / NN / linear) from a YAML config.

Two kinds of training data (ADR-0004):

* ``structure.training_csv`` alone: a pre-computed descriptor table in the
  ``load_mlip_csv`` layout (e.g. from ``atomic_descriptor_ver2.0.0.py``).
* ``mlip.descriptor`` given: ``structure.training_csv`` is a structure
  list in the classical ``fit`` layout (``target,label,weight,struct_idx``
  with ``<poscar_dir>/<k>/POSCAR``), and the descriptor table is computed
  in-package.  The descriptor config is stored in the model file, so the
  model can then drive the solvers as ``potential: {type: MLIP,
  model: ...}`` (which ones: ADR-0007, README).

``mlip.form: atom_centred`` (ADR-0005) trains E = sum_i eps_s(G_i) on the
per-atom ACSF vectors instead (models elm, qelm, nn), from the same
structure list.
"""

import json
import os
import time as _time

from ..fitting.models.csv_io import load_mlip_csv
from .common import (
    _check_yaml,
    _print_header,
    _print_output_files,
    _print_results_header,
    _print_timing_summary,
)

__all__ = ["cmd_mlip"]

_MODEL_FILE = {
    "elm": "elm_model.npz",
    "qelm": "qelm_model.npz",
    "nn": "nn_model.pt",
    "linear": "linear_model.npz",
}


# ELM / QELM options passed on only when given (ADR-0013; the model
# defaults apply otherwise).
_ELM_KEYS = (
    "w_init_scale",
    "w_init_scale_search",
    "bias_init",
    "standardize",
    "pca_eps",
    "pca_drop",
    "w_correlation",
    "w_corr_length",
    "w_corr_length_search",
)
_QELM_KEYS = ("max_exponent", "n_bits", "levels", "n_levels", "p_zero",
              "use_sparse_w", "matmul", "w_memory")


def _elm_options(model_type, ml):
    keys = _ELM_KEYS + (_QELM_KEYS if model_type == "qelm" else ())
    return {k: ml[k] for k in keys if k in ml}


def _build_model(model_type, ml, verbose):
    common = dict(
        w_energy=ml.get("w_energy", 1.0),
        w_forces=ml.get("w_forces", 0.1),
        w_stress=ml.get("w_stress", 10.0),
        verbose=verbose,
    )
    if model_type == "nn":
        from ..fitting import NNModel

        return NNModel(
            hidden_layers=ml.get("hidden_layers", [128, 64]),
            activation=ml.get("activation", "softplus"),
            r0=ml.get("r0", 1.0),
            eps0=ml.get("eps0", 0.01),
            learning_rate=ml.get("learning_rate", 1.0e-3),
            epochs=ml.get("epochs", 1500),
            batch_size=ml.get("batch_size", 256),
            weight_decay=ml.get("weight_decay", 1.0e-6),
            early_stopping_patience=ml.get("early_stopping_patience", 150),
            device=ml.get("device", "auto"),
            seed=ml.get("seed", 42),
            row_independent=ml.get("row_independent", False),
            direct_link=ml.get("direct_link", True),
            deterministic=ml.get("deterministic", False),
            batch_structures=ml.get("batch_structures", 30),
            optimizer=ml.get("optimizer", "adam"),
            **common,
        )
    if model_type in ("lgbm", "lightgbm"):
        raise ValueError(
            "mlip.model lgbm was removed (ADR-0012): trees give no "
            "conservative force, so a LightGBM model is no force field.")
    if model_type == "linear":
        from ..fitting import LinearModel

        return LinearModel(
            method=ml.get("method", "ridge"),
            alpha=ml.get("alpha", 1.0e-6),
            l1_ratio=ml.get("l1_ratio", 0.5),
            n_components=ml.get("n_components", 20),
            alpha_search=ml.get("alpha_search", None),
            **common,
        )
    if model_type in ("elm", "qelm"):
        from ..fitting import ELMModel, QuantizedELMModel

        cls = QuantizedELMModel if model_type == "qelm" else ELMModel
        kw = dict(
            n_hidden=ml.get("n_hidden", 512),
            activation=ml.get("activation", "tanh"),
            reg=ml.get("reg", 1e-6),
            r0=ml.get("r0", 1.0),
            eps0=ml.get("eps0", 0.01),
            seed=ml.get("seed", 42),
            row_independent=ml.get("row_independent", False),
            direct_link=ml.get("direct_link", True),
            **common,
        )
        if model_type == "elm":
            kw["solver"] = ml.get("solver", "auto")
        kw["reg_search"] = ml.get("reg_search", None)
        kw.update(_elm_options(model_type, ml))
        return cls(**kw)
    raise ValueError(
        f"mlip.model must be one of {sorted(_MODEL_FILE)}, got '{model_type}'."
    )


def _structures_and_descriptor(desc_cfg, csv_path, poscar_dir, species, verbose):
    """Structures (fit layout) and the ACSF built from ``desc_cfg`` (or
    ``desc_cfg`` itself when it is already an ACSF, for test data)."""
    from ..fitting.descriptors import ACSF
    from .cmd_fit import _structures_from_csv

    t, lab, _, Xs = load_mlip_csv(csv_path, verbose=False)
    structs = _structures_from_csv(t, lab, Xs, poscar_dir, verbose=verbose)
    if species is None:
        species = sorted({el for s in structs for el in s["atom_types"]})
    if isinstance(desc_cfg, ACSF):
        return structs, desc_cfg
    kind = str(desc_cfg.get("type", "acsf")).lower()
    if kind != "acsf":
        raise ValueError(
            f"mlip.descriptor.type '{kind}' is not supported "
            f"(in-package descriptors: acsf)."
        )
    return structs, ACSF.from_grid(
        desc_cfg.get("species", species),
        desc_cfg["r_cut"],
        desc_cfg.get("g2"),
        desc_cfg.get("g4"),
        desc_cfg.get("g5"),
        desc_cfg.get("inner_cutoff"),
    )


def _physics(ml, species):
    """(PairBaseline or None, physics_config JSON) from mlip.baseline and
    mlip.isolated_atom_energy (ADR-0014)."""
    import json

    from ..fitting.physics import PairBaseline

    b_cfg, iso = ml.get("baseline"), ml.get("isolated_atom_energy")
    # terms with fit_to are fitted after training (_fit_short_range)
    b_cfg = [t for t in _terms(b_cfg) if "fit_to" not in t]
    if not b_cfg and not iso:
        return None, ""
    base = PairBaseline(species, b_cfg) if b_cfg else None
    cfg = {}
    if base is not None:
        cfg["baseline"] = base.config()
    if iso:
        cfg["isolated_atom_energy"] = {str(k): float(v) for k, v in iso.items()}
    return base, json.dumps(cfg)


def _terms(b_cfg):
    return [] if not b_cfg else ([b_cfg] if isinstance(b_cfg, dict) else list(b_cfg))


def _fit_short_range(model, desc, ml, fixed, train_structs, refit, record):
    """
    Fit the ``born_mayer`` baseline term that carries ``fit_to`` (a
    structure list of short-range configurations, fit layout) to the
    residual E_ref - E_model of the trained model (ADR-0014 follow-up).
    If the fitted term reaches the training data (it is not zero there),
    the model is refitted once on labels minus the full baseline.
    Returns the new PairBaseline.
    """
    import json

    from ..fitting.mlip_engine import MLIPForceField
    from ..fitting.physics import PairBaseline, fit_born_mayer

    fit_terms = [t for t in _terms(ml.get("baseline")) if "fit_to" in t]
    if len(fit_terms) != 1 or str(fit_terms[0].get("type")).lower() != "born_mayer":
        raise ValueError("mlip.baseline: exactly one term may carry fit_to, "
                         "and it must be of type born_mayer.")
    t = fit_terms[0]
    fit = t["fit_to"]
    fit = {"training_csv": fit} if isinstance(fit, str) else dict(fit)
    csv = fit["training_csv"]
    pdir = fit.get("poscar_dir", os.path.dirname(os.path.abspath(csv)))
    sr, _ = _structures_and_descriptor(desc, csv, pdir, desc.species, False)
    ff = MLIPForceField(model)  # the model plus any fixed baseline
    term, info = fit_born_mayer(
        sr, lambda lat, pos, typ: ff(lat, pos, typ)[0], desc.species,
        float(t["r_inner"]), float(t["r_outer"]))
    new = PairBaseline(desc.species, (fixed.terms if fixed else []) + [term])
    cfg = json.loads(model.physics_config) if model.physics_config else {}
    cfg["baseline"] = new.config()
    model.physics_config = json.dumps(cfg)
    only = PairBaseline(desc.species, [term])
    touch = max(abs(only.compute(s["lattice"], s["positions"], s["atom_types"])[0])
                for s in train_structs)
    info["refit"] = bool(touch > 1e-6)
    print(f"  Short-range fit : {len(sr)} structures from {csv}")
    for k, v in term["pairs"].items():
        print(f"    {k}: A = {v['A']:.6g} eV  rho = {v['rho']:.5f} A")
    print(f"    residual rms {info['rms_before_eV']:.4f} -> "
          f"{info['rms_after_eV']:.4f} eV")
    if info["refit"]:
        print(f"    the fitted term reaches the training data (up to {touch:.2e} "
              f"eV per structure): refitting the model on labels minus the "
              f"full baseline")
        refit(new)
    record["baseline_fit"] = info
    return new


def _report_baseline(base, structs):
    d = base.min_pair_distance(structs)
    print(
        f"  Baseline    : {[t['type'] for t in base.terms]}  r_outer="
        f"{base.r_max} A  (shortest distance in the data: {d:.3f} A)"
    )


def _descriptor_table(desc_cfg, csv_path, poscar_dir, ml, species, verbose,
                      baseline=None):
    """Structures (fit layout) -> ACSF table, labels minus the pair
    baseline (``baseline``, or the one of mlip.baseline when None).
    Returns (y, lab, w, X, descriptor, baseline, physics_config)."""
    from ..fitting.descriptors import build_table

    structs, desc = _structures_and_descriptor(
        desc_cfg, csv_path, poscar_dir, species, verbose
    )
    cfg = ""
    if baseline is None:
        baseline, cfg = _physics(ml, desc.species)
        if baseline is not None and verbose:
            _report_baseline(baseline, structs)
    if baseline is not None:
        structs = baseline.subtract(structs)
    y, lab, w, X = build_table(
        structs, desc, r0=ml.get("r0", 1.0), eps0=ml.get("eps0", 0.01)
    )
    return y, lab, w, X, desc, baseline, cfg


def _build_atomic_model(model_type, ml, verbose):
    """Atom-centred model (ADR-0005): elm, qelm or nn."""
    common = dict(
        w_energy=ml.get("w_energy", 1.0),
        w_forces=ml.get("w_forces", 0.1),
        w_stress=ml.get("w_stress", 10.0),
        r0=ml.get("r0", 1.0),
        eps0=ml.get("eps0", 0.01),
        seed=ml.get("seed", 42),
        direct_link=ml.get("direct_link", True),
        verbose=verbose,
    )
    if model_type == "nn":
        from ..fitting.models.atomic_nn import AtomicNNModel

        return AtomicNNModel(
            hidden_layers=ml.get("hidden_layers", [64, 64]),
            activation=ml.get("activation", "softplus"),
            learning_rate=ml.get("learning_rate", 1.0e-3),
            epochs=ml.get("epochs", 1500),
            weight_decay=ml.get("weight_decay", 1.0e-6),
            early_stopping_patience=ml.get("early_stopping_patience", 150),
            batch_structures=ml.get("batch_structures", 30),
            deterministic=ml.get("deterministic", False),
            device=ml.get("device", "auto"),
            **common,
        )
    if model_type in ("elm", "qelm"):
        from ..fitting.models.atomic import AtomicELMModel, AtomicQELMModel

        cls = AtomicQELMModel if model_type == "qelm" else AtomicELMModel
        return cls(
            n_hidden=ml.get("n_hidden", 400),
            activation=ml.get("activation", "tanh"),
            reg=ml.get("reg", 1e-6),
            reg_search=ml.get("reg_search", None),
            **common,
            **_elm_options(model_type, ml),
        )
    raise ValueError(
        f"mlip.form 'atom_centred' supports model elm, qelm or nn, got "
        f"'{model_type}' (a linear tabular model already is atom-centred, "
        f"ADR-0004 decision 9)."
    )


def cmd_mlip(args) -> int:
    """Fit a machine-learning potential from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1

    struct_cfg = cfg.get("structure", {})
    pot_cfg = cfg.get("potential", {})
    ml_cfg = cfg.get("mlip", {})
    verbose = ml_cfg.get("verbose", True) and not getattr(args, "quiet", False)

    _print_header("Machine-Learning Potential (mlip)", cfg, pot_cfg=pot_cfg)
    model_type = str(ml_cfg.get("model", "elm")).lower()
    desc_cfg = ml_cfg.get("descriptor")
    form = str(ml_cfg.get("form", "tabular")).lower().replace("-", "_")
    if form not in ("tabular", "atom_centred"):
        print(f"ERROR: mlip.form must be 'tabular' or 'atom_centred', got '{form}'.")
        return 1
    atomic = form == "atom_centred"
    if atomic and not desc_cfg:
        print(
            "ERROR: mlip.form 'atom_centred' needs mlip.descriptor (ACSF): the "
            "per-atom descriptors are computed in-package (ADR-0005)."
        )
        return 1
    print("\n[Settings]")
    print(f"  Model type : {model_type.upper()}")
    print(f"  Model form : {form}")
    print(f"  w_energy   : {ml_cfg.get('w_energy', 1.0)}")
    print(f"  w_forces   : {ml_cfg.get('w_forces', 0.1)}")
    print(f"  w_stress   : {ml_cfg.get('w_stress', 10.0)}")
    if model_type in ("elm", "qelm", "nn") and not atomic:
        print(
            f"  form       : "
            f"{'row-independent' if ml_cfg.get('row_independent') else 'chain rule'}"
        )
    if model_type in ("elm", "qelm"):
        print(f"  n_hidden   : {ml_cfg.get('n_hidden', 512)}")
        print(f"  activation : {ml_cfg.get('activation', 'tanh')}")
        print(f"  reg        : {ml_cfg.get('reg', 1e-6)}")
    elif model_type == "nn":
        for k, d in (
            ("hidden_layers", [128, 64]),
            ("activation", "softplus"),
            ("learning_rate", 1.0e-3),
            ("epochs", 1500),
            ("batch_size", 256),
            ("weight_decay", 1.0e-6),
            ("device", "auto"),
        ):
            print(f"  {k:<13}: {ml_cfg.get(k, d)}")
    elif model_type == "linear":
        for k, d in (("method", "ridge"), ("alpha", 1.0e-6), ("alpha_search", None)):
            print(f"  {k:<13}: {ml_cfg.get(k, d)}")
    print(f"  descriptor : {desc_cfg.get('type', 'acsf') if desc_cfg else 'from CSV'}")

    output_dir = ml_cfg.get("output_dir", "./mlip_output")
    os.makedirs(output_dir, exist_ok=True)

    training_csv = struct_cfg.get("training_csv")
    if struct_cfg.get("training_dir") and not training_csv:
        print(
            "ERROR: structure.training_dir is not supported for mlip; set "
            "structure.training_csv (a descriptor CSV, or with "
            "mlip.descriptor a structure list with poscar_dir)."
        )
        return 1
    if not (training_csv and os.path.isfile(training_csv)):
        print("ERROR: Specify structure.training_csv (an existing file).")
        return 1

    try:
        model = (_build_atomic_model if atomic else _build_model)(
            model_type, ml_cfg, verbose
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    t0 = _time.perf_counter()
    sections = []
    fit_kw = dict(
        val_fraction=ml_cfg.get("val_fraction", 0.2), val_seed=ml_cfg.get("val_seed", 0)
    )
    test_table = None
    test_structs = None
    extra_summary = {}
    if atomic:
        poscar_dir = struct_cfg.get(
            "poscar_dir", os.path.dirname(os.path.abspath(training_csv))
        )
        print(
            f"\nTraining data : {training_csv}  (structures, POSCARs in {poscar_dir})"
        )
        structs, desc = _structures_and_descriptor(
            desc_cfg, training_csv, poscar_dir, pot_cfg.get("species"), verbose
        )
        print(
            f"  Descriptor  : ACSF  r_cut={desc.r_cut} A  species="
            f"{desc.species}  per atom={desc.n_per_atom}"
        )
        try:
            base, model.physics_config = _physics(ml_cfg, desc.species)
        except (ValueError, KeyError) as exc:
            print(f"ERROR: mlip.baseline: {exc}")
            return 1
        raw_structs = structs
        if base is not None:
            _report_baseline(base, structs)
            structs = base.subtract(structs)
        t1 = _time.perf_counter()
        try:
            model.fit(structs, desc, weight_csv=ml_cfg.get("weight_csv"), **fit_kw)
            if any("fit_to" in t for t in _terms(ml_cfg.get("baseline"))):
                base = _fit_short_range(
                    model, desc, ml_cfg, base, raw_structs,
                    lambda b: model.fit(b.subtract(raw_structs), desc,
                                        weight_csv=ml_cfg.get("weight_csv"),
                                        **fit_kw),
                    extra_summary)
        except (ValueError, KeyError) as exc:
            print(f"ERROR: {exc}")
            return 1
        sections.append(
            (f"atom-centred {model_type.upper()} fitting", _time.perf_counter() - t1)
        )
        test_csv = ml_cfg.get("test_csv")
        if test_csv and os.path.isfile(test_csv):
            tdir = ml_cfg.get(
                "test_poscar_dir", os.path.dirname(os.path.abspath(test_csv))
            )
            test_structs, _ = _structures_and_descriptor(
                desc, test_csv, tdir, desc.species, False
            )
            if base is not None:
                test_structs = base.subtract(test_structs)
    elif desc_cfg:
        poscar_dir = struct_cfg.get(
            "poscar_dir", os.path.dirname(os.path.abspath(training_csv))
        )
        print(
            f"\nTraining data : {training_csv}  (structures, POSCARs in {poscar_dir})"
        )
        try:
            y, lab, w, X, desc, base, model.physics_config = _descriptor_table(
                desc_cfg, training_csv, poscar_dir, ml_cfg,
                pot_cfg.get("species"), verbose)
        except (ValueError, KeyError) as exc:
            print(f"ERROR: mlip.baseline: {exc}")
            return 1
        print(
            f"  Descriptor  : ACSF  r_cut={desc.r_cut} A  species="
            f"{desc.species}  D={desc.n_features}"
        )
        sections.append(("Descriptors", _time.perf_counter() - t0))
        if ml_cfg.get("write_table", False):
            from ..fitting.descriptors import write_table_csv

            write_table_csv(
                os.path.join(output_dir, "descriptor_table.csv"),
                y,
                lab,
                w,
                X,
                desc.feature_names(),
            )
        t1 = _time.perf_counter()
        extra = {"columns": desc.feature_names()} if model_type == "linear" else {}
        # set before fit: w_correlation reads the channel
        # layout from it (ADR-0013)
        model.descriptor_config = desc.to_json()
        try:
            model.fit(y, lab, X, weights=w, **fit_kw, **extra)
            if any("fit_to" in t for t in _terms(ml_cfg.get("baseline"))):
                raw_structs, _ = _structures_and_descriptor(
                    desc, training_csv, poscar_dir, desc.species, False)

                def _refit(b):
                    y2, l2, w2, X2, *_ = _descriptor_table(
                        desc, training_csv, poscar_dir, ml_cfg, desc.species,
                        False, baseline=b)
                    model.fit(y2, l2, X2, weights=w2, **fit_kw, **extra)

                base = _fit_short_range(model, desc, ml_cfg, base, raw_structs,
                                        _refit, extra_summary)
        except (ValueError, KeyError) as exc:
            print(f"ERROR: {exc}")
            return 1
        model.descriptor_config = desc.to_json()
        sections.append((f"{model_type.upper()} fitting", _time.perf_counter() - t1))
        test_csv = ml_cfg.get("test_csv")
        if test_csv and os.path.isfile(test_csv):
            tdir = ml_cfg.get(
                "test_poscar_dir", os.path.dirname(os.path.abspath(test_csv))
            )
            ty, tl, tw, tX, *_ = _descriptor_table(
                desc, test_csv, tdir, ml_cfg, desc.species, False, baseline=base
            )
            test_table = (ty, tl, tX)
            if ml_cfg.get("write_table", False):
                # the test set's table too, so a model trained from
                # descriptor_table.csv can be scored on the same rows
                from ..fitting.descriptors import write_table_csv

                write_table_csv(
                    os.path.join(output_dir, "descriptor_table_test.csv"),
                    ty, tl, tw, tX, desc.feature_names(),
                )
    else:
        if ml_cfg.get("baseline") or ml_cfg.get("isolated_atom_energy"):
            print("ERROR: mlip.baseline / isolated_atom_energy need in-package "
                  "descriptors (mlip.descriptor): the baseline is computed "
                  "from the structures (ADR-0014).")
            return 1
        print(f"\nTraining data : {training_csv}  (descriptor CSV)")
        model.fit_csv(training_csv, **fit_kw)
        sections.append((f"{model_type.upper()} fitting", _time.perf_counter() - t0))

    model_path = os.path.join(
        output_dir, ("atomic_" if atomic else "") + _MODEL_FILE[model_type]
    )
    model.save(model_path)

    score_test = {}
    t0 = _time.perf_counter()
    if test_structs is not None:
        score_test = model.score_structures(test_structs, desc)
    elif test_table is not None:
        from ..fitting.models.elm import ELMModel

        ty, tl, tX = test_table
        score_test = ELMModel._score(ty, model.predict_table(tl, tX), tl, "test")
    elif not desc_cfg and ml_cfg.get("test_csv") and os.path.isfile(ml_cfg["test_csv"]):
        score_test = model.score_csv(ml_cfg["test_csv"]) or {}
    if score_test:
        sections.append(("Test evaluation", _time.perf_counter() - t0))

    _print_results_header("mlip")
    print("\n[Training]")
    print(f"  Model saved : {model_path}")
    engine = (
        bool(desc_cfg)
        and not getattr(model, "row_independent", True)
    )
    print(
        f"  Usable as a force field   : {'yes' if engine else 'no'}"
        + ("  (potential: {type: MLIP, model: " + model_path + "})" if engine else "")
    )
    summary = {"model": model_path, "engine": engine, "form": form, **extra_summary}
    for name, sc in (
        ("train", getattr(model, "_train_sc", None)),
        ("val", getattr(model, "_val_sc", None)),
        ("test", score_test or None),
    ):
        if not sc:
            continue
        rmse = {c: sc[c]["RMSE"] for c in ("E", "F", "S") if sc.get(c)}
        summary[f"rmse_{name}"] = rmse
        print(
            f"  RMSE {name:<5}: " + "  ".join(f"{c}={v:.6g}" for c, v in rmse.items())
        )
    with open(os.path.join(output_dir, "mlip_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    _print_output_files(output_dir)
    _print_timing_summary(sections, sum(t for _, t in sections))
    return 0
