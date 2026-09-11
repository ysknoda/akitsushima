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

    Optuna-optimised hyperparameters
    ---------------------------------
        learning_rate      : trial.suggest_float(..., 0.01, 0.1, log=True)
        num_leaves         : trial.suggest_int(..., 15, 255)
        max_depth          : trial.suggest_int(..., 3, 20)
        min_data_in_leaf   : trial.suggest_int(..., 2, 30)
        bagging_fraction   : trial.suggest_float(..., 0.5, 1.0)
        feature_fraction   : trial.suggest_float(..., 0.5, 1.0)
        min_gain_to_split  : trial.suggest_float(..., 0.1, 10, log=True)
        lambda_l1          : trial.suggest_float(..., 1, 100, log=True)
        lambda_l2          : trial.suggest_float(..., 1, 100, log=True)
        max_bin            : trial.suggest_int(..., 200, 500)

    Fixed (non-optimised) LightGBM parameters
    -------------------------------------------
    Supplied via the `lgbm_params` constructor argument, e.g.:

        lgbm_params = {
            "objective": "regression",
            "boosting": "gbdt",
            "num_iterations": 10000,
            "bagging_freq": 1,
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
    Sample weight = w_cat * w_unit * w_user (see NNModel's docstring
    for the exact definition and why Uncertainty Weighting, as used by
    ELMModel, is not implemented here: LightGBM's boosting procedure
    fits each tree using a FIXED sample-weight vector supplied up
    front -- there is no mechanism to jointly learn additional
    continuous "uncertainty scale" parameters alongside the tree
    ensemble the way a backpropagation-trained model can).

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
        'min_gain_to_split' : ('float', 0.1,  10,   True),
        'lambda_l1'         : ('float', 1,    100,  True),
        'lambda_l2'         : ('float', 1,    100,  True),
        'max_bin'           : ('int',   200,  500,  False),
    }

    _DEFAULT_LGBM_PARAMS = {
        "objective"      : "regression",
        "boosting"       : "gbdt",
        "num_iterations" : 10000,
        "bagging_freq"   : 1,
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
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.seed           = int(seed)
        self.verbose        = bool(verbose)

        self._booster    = None
        self._best_params = None
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

        _wmap = {'E': self.w_energy, 'F': self.w_forces, 'S': self.w_stress}
        w_cat = np.array([_wmap.get(c, 1.0) for c in labels])
        w_all = w_cat * w_unit * w_user

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
            caller='LightGBMModel.fit')
        n_val = int(idx_val.size)

        X_tr = X[idx_train]; y_tr = targets[idx_train]; w_tr = w_all[idx_train]
        X_va = X[idx_val]    if n_val else None
        y_va = targets[idx_val] if n_val else None
        w_va = w_all[idx_val]   if n_val else None

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
            ds_tr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, params=params)
            ds_va = (lgb.Dataset(X_va, label=y_va, weight=w_va,
                                 reference=ds_tr, params=params)
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
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(_objective, n_trials=self.optuna_trials,
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
    def fit_csv(self,
                csv_path     : str,
                weight_csv   : Optional[str] = None,
                val_fraction : float = 0.2,
                val_seed     : int   = 0,
                ) -> 'LightGBMModel':
        """Convenience wrapper: load CSV (load_mlip_csv format) then fit()."""
        targets, labels, weights, X = load_mlip_csv(csv_path, verbose=self.verbose)
        return self.fit(targets, labels, X,
                        weights=weights, weight_csv=weight_csv,
                        val_fraction=val_fraction, val_seed=val_seed)

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for descriptor matrix X."""
        if not self._fitted:
            raise RuntimeError("LightGBMModel: call fit() or fit_csv() first.")
        num_iter = getattr(self._booster, 'best_iteration', None)
        return self._booster.predict(X, num_iteration=num_iter)

    # ------------------------------------------------------------------
    def score_csv(self, csv_path: str, label: str = "test") -> dict:
        """Load a CSV and return a score dict without re-fitting."""
        targets, labels, weights, X = load_mlip_csv(csv_path, verbose=False)
        y_pred = self.predict(X)
        sc = ELMModel._score(targets, y_pred, labels, label)
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
                 D            = np.array(self._D))
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
        obj._fitted  = True
        return obj
