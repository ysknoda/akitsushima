"""LightGBM gradient-boosted-tree machine-learning potential, with a
standard, always-on Optuna hyperparameter search.
"""
from typing import Optional

import numpy as np

from ...utils.optional_lightgbm import HAS_LIGHTGBM, HAS_OPTUNA, lgb, optuna
from .csv_io import load_mlip_csv, load_weights_csv, structure_grouped_split
from .elm import ELMModel

__all__ = ["LightGBMModel"]


class LightGBMModel:
    """
    LightGBM gradient-boosted-tree machine-learning potential trained
    from the same CSV descriptor data used by ELMModel / NNModel (see
    load_mlip_csv for the file format).

    Hyperparameter optimisation (Optuna) is a STANDARD, always-on part
    of fit() / fit_csv() (not an optional extra step): the 9
    hyperparameters listed below are searched automatically over
    `optuna_trials` trials, minimising the weighted validation RMSE,
    before the final model is trained with the best combination found.

    Role (ADR-0004, decision 3)
    ---------------------------
    Every row is regressed on its own descriptor, as before: trees have
    zero gradient almost everywhere, so the chain rule the other models
    use cannot be applied and the model is not a force field.  Two
    additions help it learn E, F and S rows side by side:

    * the row label (E/F/S) is an extra categorical feature, so the trees
      can learn separate functions for the three kinds of row;
    * targets are standardised per label (mean/std of the training rows),
      and the weights are rescaled by std_c^2 and then by their mean.
      The objective is a constant times sum_i w_i (y_i - p_i)^2 in the
      original units, but its scale no longer depends on the units of the
      targets or of the weights -- which is what ``lambda_l1/l2`` and the
      split gain are measured against.

    Optuna-optimised hyperparameters (ADR-0004 search space)
    --------------------------------------------------------
        learning_rate      : trial.suggest_float(..., 0.01, 0.1, log=True)
        num_leaves         : trial.suggest_int(..., 15, 255)
        max_depth          : trial.suggest_int(..., 3, 20)
        min_data_in_leaf   : trial.suggest_int(..., 2, 30)
        bagging_fraction   : trial.suggest_float(..., 0.5, 1.0)
        feature_fraction   : trial.suggest_float(..., 0.5, 1.0)
        lambda_l1          : trial.suggest_float(..., 1e-8, 10, log=True)
        lambda_l2          : trial.suggest_float(..., 1e-8, 10, log=True)
        max_bin            : trial.suggest_int(..., 200, 500)

    ``min_gain_to_split`` is fixed at 0 (in the fixed parameters).  Its
    former search range [0.1, 10] was the main reason this wrapper
    trailed the bare library (test RMSE 1.77-2.03 sigma against
    1.45-1.47 sigma once fixed at 0; ADR-0004, evidence table): a split
    gain is a loss decrease in the units of the weighted squared target,
    so a fixed lower bound of 0.1 stops splitting on small-unit rows.
    The lambda ranges were [1, 100]; their lower bound was the secondary
    cause.

    Fixed (non-optimised) LightGBM parameters
    -------------------------------------------
    Supplied via the `lgbm_params` constructor argument, e.g.:

        lgbm_params = {
            "objective": "regression",
            "boosting": "gbdt",
            "num_iterations": 100000,
            "bagging_freq": 1,
            "min_gain_to_split": 0.0,
            "min_data_in_bin": 3,
            "metric": "rmse",
            "seed": 0,
            "verbosity": -1,
        }

    These are merged with the Optuna-selected hyperparameters (Optuna
    values take precedence for any overlapping key) before each trial's
    lgb.train() call, and again for the final model.

    Weighting scheme
    ----------------
    Weight on the squared residual = w_cat * w_unit^2 * w_user, the rule
    shared by every model (ADR-0004; w_unit was not squared before
    0.2.0a3).  See NNModel's docstring for the three factors.

    Parameters
    ----------
    w_energy, w_forces, w_stress : category weights (as in NNModel).
    r0, eps0       : as in NNModel (unit-normalisation consistency).
    lgbm_params    : dict of fixed (non-optimised) LightGBM parameters
                     (see example above). Defaults to a reasonable
                     regression configuration if not given.
    optuna_trials  : number of Optuna trials (the `n_trials` argument
                     of optuna.study.optimize()).
    optuna_timeout : optional wall-clock time limit in seconds for the
                     Optuna search (None = no timeout, run all trials).
    optuna_storage : optional Optuna storage URL, e.g.
                     ``"sqlite:///lgbm_study.db"``.  Trials are saved as
                     they finish, and a rerun with the same storage and
                     ``optuna_study_name`` resumes the study: only the
                     trials still missing from ``optuna_trials`` are run.
    optuna_study_name : study name inside the storage (default
                     ``"akitsushima_lgbm"``).
    label_feature  : pass the row label as a categorical feature (default
                     True).  False lets E, F and S rows share every split,
                     which helps when they follow one function.
    seed           : RNG seed (Optuna sampler and the train/val split).
    verbose        : print search progress and diagnostics.
    """

    _OPTUNA_RANGES = {
        'learning_rate'     : ('float', 0.01, 0.1,  True),
        'num_leaves'        : ('int',   15,   255,  False),
        'max_depth'         : ('int',   3,    20,   False),
        'min_data_in_leaf'  : ('int',   2,    30,   False),
        'bagging_fraction'  : ('float', 0.5,  1.0,  False),
        'feature_fraction'  : ('float', 0.5,  1.0,  False),
        'lambda_l1'         : ('float', 1e-8, 10,   True),
        'lambda_l2'         : ('float', 1e-8, 10,   True),
        'max_bin'           : ('int',   200,  500,  False),
    }

    _DEFAULT_LGBM_PARAMS = {
        "objective"      : "regression",
        "boosting"       : "gbdt",
        # Early stopping (early_stopping_rounds) ends training; the cap
        # only guards against a run that never stops.  At 10000 the
        # ablation's best runs reached 9511 iterations, i.e. the cap.
        "num_iterations" : 100000,
        "bagging_freq"   : 1,
        "min_gain_to_split": 0.0,
        "min_data_in_bin": 3,
        "metric"         : "rmse",
        "seed"           : 0,
        "verbosity"      : -1,
    }

    def __init__(self,
                 w_energy       : float = 1.0,
                 w_forces       : float = 0.1,
                 w_stress       : float = 10.0,
                 r0             : float = 1.0,
                 eps0           : float = 0.01,
                 lgbm_params    : Optional[dict] = None,
                 optuna_trials  : int   = 100,
                 optuna_timeout : Optional[float] = None,
                 optuna_storage : Optional[str] = None,
                 optuna_study_name: str = "akitsushima_lgbm",
                 label_feature  : bool  = True,
                 early_stopping_rounds: int = 100,
                 seed           : int   = 42,
                 verbose        : bool  = True):
        if not HAS_LIGHTGBM:
            raise ImportError(
                "LightGBMModel requires the lightgbm package, which is not "
                "installed. Install it with: "
                "pip install lightgbm --break-system-packages")
        if not HAS_OPTUNA:
            raise ImportError(
                "LightGBMModel requires the optuna package for its "
                "standard hyperparameter search, which is not installed. "
                "Install it with: pip install optuna --break-system-packages")

        self.w_energy      = float(w_energy)
        self.w_forces      = float(w_forces)
        self.w_stress      = float(w_stress)
        self.r0            = float(r0)
        self.eps0          = float(eps0)
        self.lgbm_params   = dict(self._DEFAULT_LGBM_PARAMS)
        if lgbm_params:
            self.lgbm_params.update(lgbm_params)
        self.optuna_trials  = int(optuna_trials)
        self.optuna_timeout = optuna_timeout
        self.optuna_storage = optuna_storage
        self.optuna_study_name = str(optuna_study_name)
        self.label_feature  = bool(label_feature)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.seed           = int(seed)
        self.verbose        = bool(verbose)

        self._booster    = None
        self._best_params = None
        # Per-label target standardisation: label -> (mean, std)
        self._label_stats: dict = {}
        self._D          = None
        self._fitted     = False

    # ------------------------------------------------------------------
    def _suggest_params(self, trial) -> dict:
        """Build one trial's hyperparameter dict from _OPTUNA_RANGES."""
        params = {}
        for name, (kind, lo, hi, log) in self._OPTUNA_RANGES.items():
            if kind == 'float':
                params[name] = trial.suggest_float(name, lo, hi, log=log)
            else:
                params[name] = trial.suggest_int(name, lo, hi, log=log)
        return params

    # ------------------------------------------------------------------
    def fit(self,
            targets      : np.ndarray,
            labels       : np.ndarray,
            X            : np.ndarray,
            weights      : Optional[np.ndarray] = None,
            weight_csv   : Optional[str]        = None,
            val_fraction : float = 0.2,
            val_seed     : int   = 0,
            struct_ids   : Optional[np.ndarray] = None,
            n_atoms      : Optional[np.ndarray] = None,
            ) -> 'LightGBMModel':
        """
        Train the model on (targets, labels, X) descriptor rows,
        including the standard Optuna hyperparameter search.

        See the class docstring for the CSV row convention, the
        sample-weighting scheme, and the Optuna search space.

        Returns
        -------
        self
        """
        N, D = X.shape
        self._D = D

        if weights is not None:
            w_unit = np.asarray(weights, dtype=np.float64)
            if w_unit.shape != (N,):
                raise ValueError(
                    f"LightGBMModel.fit: weights.shape={w_unit.shape} "
                    f"must be ({N},).")
            if np.any(w_unit < 0):
                raise ValueError(
                    "LightGBMModel.fit: weights contains negative values.")
        else:
            w_unit = np.ones(N, dtype=np.float64)

        if weight_csv is not None:
            w_user = load_weights_csv(weight_csv, n_total=N, verbose=self.verbose)
        else:
            w_user = np.ones(N, dtype=np.float64)

        labels = np.array([str(c).strip().upper() for c in labels])
        _wmap = {'E': self.w_energy, 'F': self.w_forces, 'S': self.w_stress}
        w_cat = np.array([_wmap.get(c, 1.0) for c in labels])
        w_all = w_cat * w_unit ** 2 * w_user

        # Train / val split, grouped by structure.
        #
        # Bug-fix note (F17): this used to be a plain row-wise shuffle,
        #     idx_all = np.random.default_rng(val_seed).permutation(N)
        # over CSV ROWS.  Each structure contributes 1 energy + 3N force +
        # 6 stress rows derived from the SAME atomic configuration, so a
        # row-wise split puts rows of one structure on both sides and the
        # validation score measures memorisation rather than
        # generalisation -- which then drives Optuna's hyperparameter
        # selection, since it minimises exactly that leaked score.
        # structure_grouped_split() assigns whole structures to one side.
        # See csv_io.structure_grouped_split for the full note, the
        # boundary-inference rules and the edge cases (single-structure
        # dataset, file with no E rows, rows before the first E row) --
        # including the val_fraction >= 1 case this branch used to warn
        # about itself, which that helper now handles for every model.
        idx_train, idx_val = structure_grouped_split(
            labels, val_fraction=val_fraction, val_seed=val_seed,
            caller='LightGBMModel.fit', struct_ids=struct_ids)
        n_val = int(idx_val.size)

        # Per-label standardisation from the training rows; the weights
        # carry std_c^2 so that the objective stays sum w (y - p)^2 in the
        # original units, up to one overall constant (the mean weight).
        self._label_stats = {}
        for c in ('E', 'F', 'S'):
            m = labels[idx_train] == c
            if m.any():
                mu = float(targets[idx_train][m].mean())
                sd = float(targets[idx_train][m].std())
                self._label_stats[c] = (mu, sd if sd > 1e-300 else 1.0)
        y_std, w_std = self._standardise(targets, labels, w_all)
        w_std = w_std / max(float(w_std[idx_train].mean()), 1e-300)
        X_aug = self._augment(X, labels)

        X_tr = X_aug[idx_train]; y_tr = y_std[idx_train]; w_tr = w_std[idx_train]
        X_va = X_aug[idx_val]    if n_val else None
        y_va = y_std[idx_val] if n_val else None
        w_va = w_std[idx_val]   if n_val else None

        if self.verbose:
            print(f"\nLightGBMModel fitting")
            print(f"  Samples      : {N}  (D={D})")
            print(f"  Category wts : wE={self.w_energy}  "
                  f"wF={self.w_forces}  wS={self.w_stress}")
            print(f"  Train/val split: train={len(idx_train)}  val={n_val}")
            print(f"  Optuna trials: {self.optuna_trials}"
                  + (f"  timeout={self.optuna_timeout}s" if self.optuna_timeout else ""))

        def _make_datasets(params):
            # Bug-fix note (found via a LightGBMError -- "Cannot change
            # max_bin after constructed Dataset handle" -- while
            # investigating an unrelated environment-specific segfault):
            # max_bin (one of the searched Optuna hyperparameters) is a
            # Dataset-construction-time setting in LightGBM, not a
            # training-time one. The original code built ds_train/ds_val
            # ONCE outside the objective and reused them across every
            # Optuna trial, passing each trial's (possibly different)
            # max_bin only to lgb.train() -- which raises as soon as any
            # trial after the first suggests a different max_bin than the
            # one implicitly baked into the already-constructed Dataset.
            # Rebuilding the Dataset per trial, with that trial's own
            # params, is the correct way to search over max_bin.
            cat = [X_tr.shape[1] - 1] if self.label_feature else []
            ds_tr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, params=params,
                                categorical_feature=cat)
            ds_va = (lgb.Dataset(X_va, label=y_va, weight=w_va,
                                 reference=ds_tr, params=params,
                                 categorical_feature=cat)
                    if n_val else None)
            return ds_tr, ds_va

        def _objective(trial):
            trial_params = dict(self.lgbm_params)
            trial_params.update(self._suggest_params(trial))
            ds_tr, ds_va = _make_datasets(trial_params)
            if n_val:
                booster = lgb.train(
                    trial_params, ds_tr,
                    valid_sets=[ds_va],
                    callbacks=[lgb.early_stopping(self.early_stopping_rounds,
                                                  verbose=False)])
                pred = booster.predict(X_va, num_iteration=booster.best_iteration)
                rmse = float(np.sqrt(np.mean(w_va * (pred - y_va) ** 2)))
            else:
                booster = lgb.train(trial_params, ds_tr)
                pred = booster.predict(X_tr)
                rmse = float(np.sqrt(np.mean(w_tr * (pred - y_tr) ** 2)))
            return rmse

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction='minimize', sampler=sampler,
                                    storage=self.optuna_storage,
                                    study_name=(self.optuna_study_name
                                                if self.optuna_storage
                                                else None),
                                    load_if_exists=bool(self.optuna_storage))
        n_done = sum(1 for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE)
        n_left = max(0, self.optuna_trials - n_done)
        if self.verbose and n_done:
            print(f"  Resuming study '{self.optuna_study_name}': "
                  f"{n_done} trials done, {n_left} to run")
        if n_left:
            study.optimize(_objective, n_trials=n_left,
                           timeout=self.optuna_timeout,
                           show_progress_bar=False)

        self._best_params = dict(self.lgbm_params)
        self._best_params.update(study.best_params)

        if self.verbose:
            print(f"  Optuna best RMSE (weighted): {study.best_value:.6e}")
            print(f"  Optuna best params: {study.best_params}")

        # Final model: retrain on the full training split (with early
        # stopping against the validation split, if available) using
        # the best hyperparameters found.
        ds_train, ds_val = _make_datasets(self._best_params)
        if n_val:
            self._booster = lgb.train(
                self._best_params, ds_train,
                valid_sets=[ds_val],
                callbacks=[lgb.early_stopping(self.early_stopping_rounds,
                                              verbose=self.verbose)])
        else:
            self._booster = lgb.train(self._best_params, ds_train)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    _LABEL_CODE = {'E': 0, 'F': 1, 'S': 2}

    def _augment(self, X: np.ndarray, labels) -> np.ndarray:
        """Append the label code (E=0, F=1, S=2) as a categorical column."""
        if not self.label_feature:
            return np.asarray(X, dtype=np.float64)
        code = np.array([self._LABEL_CODE[str(c).strip().upper()]
                         for c in labels], dtype=np.float64)
        return np.column_stack([X, code])

    def _standardise(self, y, labels, w):
        y2, w2 = np.array(y, dtype=np.float64), np.array(w, dtype=np.float64)
        for c, (mu, sd) in self._label_stats.items():
            m = labels == c
            y2[m] = (y2[m] - mu) / sd
            w2[m] = w2[m] * sd ** 2
        return y2, w2

    # ------------------------------------------------------------------
    def fit_csv(self,
                csv_path     : str,
                weight_csv   : Optional[str] = None,
                val_fraction : float = 0.2,
                val_seed     : int   = 0,
                ) -> 'LightGBMModel':
        """Convenience wrapper: load CSV (load_mlip_csv format) then fit()."""
        targets, labels, weights, X, meta = load_mlip_csv(
            csv_path, verbose=self.verbose, return_meta=True)
        return self.fit(targets, labels, X,
                        weights=weights, weight_csv=weight_csv,
                        val_fraction=val_fraction, val_seed=val_seed,
                        struct_ids=meta["struct_id"], n_atoms=meta["n_atoms"])

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, labels=None) -> np.ndarray:
        """
        Predict target values for descriptor matrix X.

        ``labels`` ('E'/'F'/'S' per row) is required: the label is a model
        feature and the targets are standardised per label.
        """
        if not self._fitted:
            raise RuntimeError("LightGBMModel: call fit() or fit_csv() first.")
        if labels is None:
            raise ValueError(
                "LightGBMModel.predict: labels= is required (the row label "
                "is a model feature since 0.2.0a3).")
        labels = np.array([str(c).strip().upper() for c in labels])
        num_iter = getattr(self._booster, 'best_iteration', None)
        p = self._booster.predict(self._augment(X, labels),
                                  num_iteration=num_iter)
        for c, (mu, sd) in self._label_stats.items():
            m = labels == c
            p[m] = p[m] * sd + mu
        return p

    def predict_table(self, labels, X, struct_ids=None, n_atoms=None):
        """Predict every row of a training-format table (rows are
        independent for this model)."""
        return self.predict(X, labels)

    # ------------------------------------------------------------------
    def score_csv(self, csv_path: str, label: str = "test") -> dict:
        """Load a CSV and return a score dict without re-fitting."""
        targets, labels, weights, X = load_mlip_csv(csv_path, verbose=False)
        y_pred = self.predict(X, labels)
        sc = ELMModel._score(targets, y_pred, np.asarray(labels), label)
        ELMModel._print_scores(sc)
        return sc

    # ------------------------------------------------------------------
    #  Model persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Save the booster (as LightGBM's native text format) and
        metadata to a single .npz archive.
        """
        if not self._fitted:
            raise RuntimeError("LightGBMModel: model has not been fitted yet.")
        p = path if path.endswith('.npz') else path + '.npz'
        model_str = self._booster.model_to_string()
        np.savez(p,
                 model_str    = np.array(model_str),
                 best_params  = np.array(str(self._best_params)),
                 lgbm_params  = np.array(str(self.lgbm_params)),
                 w_energy     = np.array(self.w_energy),
                 w_forces     = np.array(self.w_forces),
                 w_stress     = np.array(self.w_stress),
                 r0           = np.array(self.r0),
                 eps0         = np.array(self.eps0),
                 optuna_trials= np.array(self.optuna_trials),
                 seed         = np.array(self.seed),
                 D            = np.array(self._D),
                 label_stats  = np.array(str(self._label_stats)),
                 label_feature= np.array(self.label_feature))
        if self.verbose:
            print(f"  LightGBMModel saved -> {p}")

    @classmethod
    def load(cls, path: str, verbose: bool = False) -> 'LightGBMModel':
        """Load a saved LightGBMModel from a .npz archive."""
        if not HAS_LIGHTGBM:
            raise ImportError(
                "LightGBMModel.load requires the lightgbm package, "
                "which is not installed.")
        # Bug-fix note (F12): model_str/best_params/lgbm_params are all
        # saved as plain strings (best_params/lgbm_params via str(dict),
        # parsed back below with ast.literal_eval) -- nothing here needs
        # pickle support.
        p = path if path.endswith('.npz') else path + '.npz'
        npz = np.load(p)
        obj = cls(
            w_energy      = float(npz['w_energy']),
            w_forces      = float(npz['w_forces']),
            w_stress      = float(npz['w_stress']),
            r0            = float(npz['r0']),
            eps0          = float(npz['eps0']),
            optuna_trials = int(npz['optuna_trials']),
            seed          = int(npz['seed']),
            verbose       = verbose,
        )
        model_str    = str(npz['model_str'])
        obj._booster = lgb.Booster(model_str=model_str)
        import ast as _ast
        obj._best_params = _ast.literal_eval(str(npz['best_params']))
        obj._D       = int(npz['D'])
        if 'label_stats' not in npz:
            raise ValueError(
                f"LightGBMModel.load: '{p}' was written before 0.2.0a3, "
                f"when the row label was not a model feature; refit it.")
        obj._label_stats = _ast.literal_eval(str(npz['label_stats']))
        obj.label_feature = bool(npz['label_feature'])
        obj._fitted  = True
        return obj
