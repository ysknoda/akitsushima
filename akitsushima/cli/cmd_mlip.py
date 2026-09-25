"""``akitsushima mlip`` -- fit a machine-learning interatomic potential
(ELM / QELM / NN / linear / LightGBM) from a YAML config.

Two kinds of training data (ADR-0004):

* ``structure.training_csv`` alone: a pre-computed descriptor table in the
  ``load_mlip_csv`` layout (e.g. from ``atomic_descriptor_ver2.0.0.py``).
* ``mlip.descriptor`` given: ``structure.training_csv`` is a structure
  list in the classical ``fit`` layout (``target,label,weight,struct_idx``
  with ``<poscar_dir>/<k>/POSCAR``), and the descriptor table is computed
  in-package.  The descriptor config is stored in the model file, so the
  model can then run ``relax``/``md`` as ``potential: {type: MLIP,
  model: ...}``.
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
    "lgbm": "lgbm_model.npz",
    "linear": "linear_model.npz",
}


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
    if model_type == "lgbm":
        from ..fitting import LightGBMModel

        return LightGBMModel(
            r0=ml.get("r0", 1.0),
            eps0=ml.get("eps0", 0.01),
            lgbm_params=ml.get("lgbm_params", None),
            optuna_trials=ml.get("optuna_trials", 100),
            optuna_timeout=ml.get("optuna_timeout", None),
            optuna_storage=ml.get("optuna_storage", None),
            optuna_study_name=ml.get("optuna_study_name", "akitsushima_lgbm"),
            label_feature=ml.get("label_feature", True),
            early_stopping_rounds=ml.get("early_stopping_rounds", 100),
            seed=ml.get("seed", 42),
            **common,
        )
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
        return cls(**kw)
    raise ValueError(
        f"mlip.model must be one of {sorted(_MODEL_FILE)}, got '{model_type}'."
    )


def _descriptor_table(desc_cfg, csv_path, poscar_dir, ml, species, verbose):
    """Structures (fit layout) -> ACSF table.  Returns (y, lab, w, X,
    descriptor)."""
    from ..fitting.descriptors import ACSF, build_table
    from .cmd_fit import _structures_from_csv

    t, lab, _, Xs = load_mlip_csv(csv_path, verbose=False)
    structs = _structures_from_csv(t, lab, Xs, poscar_dir, verbose=verbose)
    if species is None:
        species = sorted({el for s in structs for el in s["atom_types"]})
    if isinstance(desc_cfg, ACSF):
        desc = desc_cfg          # the training descriptor, for test data
    else:
        kind = str(desc_cfg.get("type", "acsf")).lower()
        if kind != "acsf":
            raise ValueError(
                f"mlip.descriptor.type '{kind}' is not supported "
                f"(in-package descriptors: acsf)."
            )
        desc = ACSF.from_grid(
            desc_cfg.get("species", species),
            desc_cfg["r_cut"],
            desc_cfg.get("g2"),
            desc_cfg.get("g4"),
            desc_cfg.get("g5"),
        )
    y, lab, w, X = build_table(
        structs, desc, r0=ml.get("r0", 1.0), eps0=ml.get("eps0", 0.01)
    )
    return y, lab, w, X, desc


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
    print("\n[Settings]")
    print(f"  Model type : {model_type.upper()}")
    print(f"  w_energy   : {ml_cfg.get('w_energy', 1.0)}")
    print(f"  w_forces   : {ml_cfg.get('w_forces', 0.1)}")
    print(f"  w_stress   : {ml_cfg.get('w_stress', 10.0)}")
    if model_type in ("elm", "qelm", "nn"):
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
    elif model_type == "lgbm":
        for k in ("optuna_trials", "optuna_timeout", "optuna_storage", "lgbm_params"):
            print(f"  {k:<14}: {ml_cfg.get(k)}")
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
        model = _build_model(model_type, ml_cfg, verbose)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    t0 = _time.perf_counter()
    sections = []
    fit_kw = dict(
        val_fraction=ml_cfg.get("val_fraction", 0.2), val_seed=ml_cfg.get("val_seed", 0)
    )
    test_table = None
    if desc_cfg:
        poscar_dir = struct_cfg.get(
            "poscar_dir", os.path.dirname(os.path.abspath(training_csv))
        )
        print(
            f"\nTraining data : {training_csv}  (structures, POSCARs in {poscar_dir})"
        )
        y, lab, w, X, desc = _descriptor_table(
            desc_cfg, training_csv, poscar_dir, ml_cfg, pot_cfg.get("species"), verbose
        )
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
        model.fit(y, lab, X, weights=w, **fit_kw, **extra)
        model.descriptor_config = desc.to_json()
        sections.append((f"{model_type.upper()} fitting", _time.perf_counter() - t1))
        test_csv = ml_cfg.get("test_csv")
        if test_csv and os.path.isfile(test_csv):
            tdir = ml_cfg.get(
                "test_poscar_dir", os.path.dirname(os.path.abspath(test_csv))
            )
            ty, tl, _, tX, _ = _descriptor_table(
                desc, test_csv, tdir, ml_cfg, desc.species, False
            )
            test_table = (ty, tl, tX)
    else:
        print(f"\nTraining data : {training_csv}  (descriptor CSV)")
        model.fit_csv(training_csv, **fit_kw)
        sections.append((f"{model_type.upper()} fitting", _time.perf_counter() - t0))

    model_path = os.path.join(output_dir, _MODEL_FILE[model_type])
    model.save(model_path)

    score_test = {}
    t0 = _time.perf_counter()
    if test_table is not None:
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
        and model_type != "lgbm"
        and not getattr(model, "row_independent", True)
    )
    print(
        f"  Usable as relax/md engine : {'yes' if engine else 'no'}"
        + ("  (potential: {type: MLIP, model: " + model_path + "})" if engine else "")
    )
    summary = {"model": model_path, "engine": engine}
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
