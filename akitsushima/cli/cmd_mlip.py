"""``akitsushima mlip`` -- fit a machine-learning interatomic potential
(ELM / QELM / NN / LightGBM) from a YAML config.
"""
import os
import time as _time

from ..fitting.models.csv_io import load_mlip_csv
from .common import _check_yaml, _print_header, _print_output_files, _print_results_header, _print_timing_summary

__all__ = ["cmd_mlip"]


def cmd_mlip(args) -> int:
    """Fit a machine-learning potential (ELM / QELM / NN / LightGBM) from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1

    struct_cfg = cfg.get("structure", {})
    pot_cfg    = cfg.get("potential", {})
    ml_cfg     = cfg.get("mlip",      {})
    verbose    = ml_cfg.get("verbose", True) and not getattr(args, "quiet", False)

    _print_header("Machine-Learning Potential (mlip)", cfg, pot_cfg=pot_cfg)
    model_type = str(ml_cfg.get("model", "elm")).lower()
    print(f"\n[Settings]")
    print(f"  Model type : {model_type.upper()}")
    print(f"  w_energy   : {ml_cfg.get('w_energy',1.0)}")
    print(f"  w_forces   : {ml_cfg.get('w_forces',0.1)}")
    print(f"  w_stress   : {ml_cfg.get('w_stress',10.0)}")
    print(f"  r_cut      : {pot_cfg.get('r_cut',5.0)} A")

    if model_type in ("elm", "qelm"):
        print(f"  n_hidden   : {ml_cfg.get('n_hidden',512)}")
        print(f"  activation : {ml_cfg.get('activation','tanh')}")
        print(f"  reg        : {ml_cfg.get('reg',1e-6)}")
    elif model_type == "nn":
        print(f"  hidden_layers : {ml_cfg.get('hidden_layers',[128,64])}")
        print(f"  activation    : {ml_cfg.get('activation','silu')}")
        print(f"  learning_rate : {ml_cfg.get('learning_rate',1.0e-3)}")
        print(f"  epochs        : {ml_cfg.get('epochs',500)}")
        print(f"  batch_size    : {ml_cfg.get('batch_size',256)}")
        print(f"  weight_decay  : {ml_cfg.get('weight_decay',1.0e-6)}")
        print(f"  device        : {ml_cfg.get('device','auto')}")
    elif model_type == "lgbm":
        print(f"  optuna_trials : {ml_cfg.get('optuna_trials',100)}")
        print(f"  optuna_timeout: {ml_cfg.get('optuna_timeout',None)}")
        print(f"  lgbm_params   : {ml_cfg.get('lgbm_params',None)}")

    output_dir = ml_cfg.get("output_dir", "./mlip_output")
    os.makedirs(output_dir, exist_ok=True)

    training_csv = struct_cfg.get("training_csv")
    training_dir = struct_cfg.get("training_dir")

    if model_type == "nn":
        from ..fitting import NNModel
        model = NNModel(
            hidden_layers = ml_cfg.get("hidden_layers", [128, 64]),
            activation    = ml_cfg.get("activation", "silu"),
            w_energy      = ml_cfg.get("w_energy", 1.0),
            w_forces      = ml_cfg.get("w_forces", 0.1),
            w_stress      = ml_cfg.get("w_stress", 10.0),
            r0            = ml_cfg.get("r0", 1.0),
            eps0          = ml_cfg.get("eps0", 0.01),
            learning_rate = ml_cfg.get("learning_rate", 1.0e-3),
            epochs        = ml_cfg.get("epochs", 500),
            batch_size    = ml_cfg.get("batch_size", 256),
            weight_decay  = ml_cfg.get("weight_decay", 1.0e-6),
            early_stopping_patience = ml_cfg.get("early_stopping_patience", 30),
            device        = ml_cfg.get("device", "auto"),
            seed          = ml_cfg.get("seed", 42),
            verbose       = verbose)
        model_filename = "nn_model.pt"
    elif model_type == "lgbm":
        from ..fitting import LightGBMModel
        model = LightGBMModel(
            w_energy      = ml_cfg.get("w_energy", 1.0),
            w_forces      = ml_cfg.get("w_forces", 0.1),
            w_stress      = ml_cfg.get("w_stress", 10.0),
            r0            = ml_cfg.get("r0", 1.0),
            eps0          = ml_cfg.get("eps0", 0.01),
            lgbm_params   = ml_cfg.get("lgbm_params", None),
            optuna_trials = ml_cfg.get("optuna_trials", 100),
            optuna_timeout= ml_cfg.get("optuna_timeout", None),
            early_stopping_rounds = ml_cfg.get("early_stopping_rounds", 100),
            seed          = ml_cfg.get("seed", 42),
            verbose       = verbose)
        model_filename = "lgbm_model.npz"
    else:
        from ..fitting import ELMModel, QuantizedELMModel
        ModelCls = QuantizedELMModel if model_type == "qelm" else ELMModel
        model = ModelCls(
            n_hidden=ml_cfg.get("n_hidden",512), activation=ml_cfg.get("activation","tanh"),
            w_energy=ml_cfg.get("w_energy",1.0), w_forces=ml_cfg.get("w_forces",0.1),
            w_stress=ml_cfg.get("w_stress",10.0), reg=ml_cfg.get("reg",1e-6),
            r0=ml_cfg.get("r0",1.0), eps0=ml_cfg.get("eps0",0.01),
            seed=ml_cfg.get("seed",42), verbose=verbose)
        model_filename = "qelm_model.npz" if model_type == "qelm" else "elm_model.npz"

    t0 = _time.perf_counter()
    # Bug-fix note (S23, newly found during the port): fit_csv() takes
    # only a pre-computed descriptor CSV path, for ALL FOUR model types
    # (ELM/QELM/NN/LightGBM) -- none of them accept r_cut (r_cut only
    # matters for GENERATING the descriptor CSV, a step that has already
    # happened by the time this CSV exists). The original passed
    # r_cut=r_cut to the ELM/QELM branch specifically, raising TypeError
    # on every single ELM/QELM fit. Separately, structure.training_dir
    # was never actually functional for any model type either way:
    # fit_csv()/load_mlip_csv() only open a single CSV file -- there is
    # no directory-of-files convention for MLIP training data (unlike
    # cmd_fit's classical fitting, which has a real poscar_dir
    # convention) -- so a clear, actionable error is raised here instead
    # of the previous confusing low-level one (IsADirectoryError, from
    # trying to open() a directory as a text file).
    if training_csv and os.path.isfile(training_csv):
        print(f"\nTraining data : {training_csv}  (CSV)")
        model.fit_csv(training_csv)
    elif training_dir and os.path.isdir(training_dir):
        print(f"ERROR: structure.training_dir is not supported for mlip "
              f"models -- fit_csv() requires a single pre-computed "
              f"descriptor CSV file (see load_mlip_csv's format for the "
              f"layout). Set structure.training_csv instead.")
        return 1
    else:
        print("ERROR: Specify structure.training_csv (a pre-computed descriptor CSV)")
        return 1
    t_fit = _time.perf_counter() - t0

    model_path = os.path.join(output_dir, model_filename)
    model.save(model_path)

    score_train = {}
    if hasattr(model, "uncertainty_summary"):
        # Bug-fix note (F6): uncertainty_summary() requires targets/labels
        # (the same training data just fitted above) -- the original
        # called it with no arguments at all, raising TypeError
        # immediately after every successful ELM/QELM fit.
        train_targets, train_labels, _, _ = load_mlip_csv(training_csv, verbose=False)
        score_train = model.uncertainty_summary(train_targets, train_labels, label="train")

    t_eval = 0.0
    score_test = {}
    test_csv = ml_cfg.get("test_csv")
    if test_csv and os.path.isfile(test_csv):
        t0 = _time.perf_counter()
        score_test = model.score_csv(test_csv) or {}
        t_eval = _time.perf_counter() - t0

    _print_results_header("mlip")
    print(f"\n[Training]")
    print(f"  Model saved : {model_path}")
    for k,v in score_train.items():
        print(f"  {k:<16}: {v}")
    if score_test:
        print(f"\n[Test evaluation  ({test_csv})]")
        for k,v in score_test.items():
            print(f"  {k:<16}: {v}")

    _print_output_files(output_dir)
    sections = [(f"{model_type.upper()} fitting", t_fit)]
    if t_eval > 0:
        sections.append(("Test evaluation", t_eval))
    _print_timing_summary(sections, t_fit + t_eval)
    return 0
