"""Linear machine-learning potential on the tabular E/F/S descriptor format
(ADR-0004, decision 9).

Model
-----
E/N = f(d) = w . d + b, with d the intensive structure descriptor of an
E row.  Because an F row's descriptor is X = -dd/dr and an S row's is
X = -(1/V) dd/dT (see :func:`load_mlip_csv`), every row is linear in
(w, b) without any approximation::

    E row : y = w . d + b
    F row : y = N w . X          (F_{i,a})
    S row : y = N w . X          (sigma_{ab}, compressive-positive)

With d the per-atom average of atomic descriptors G_i this is
E = sum_i w . G_i, i.e. an atom-centred linear potential.

Intercept and centring act on E rows only
------------------------------------------
F and S rows are derivatives, in which the constant b vanishes.  The
intercept column is therefore 1 on E rows and 0 elsewhere, and it is
left unpenalised by projecting it out of the least-squares system
(``_project_intercept``) before any method runs.  Library routines that
centre every row and fit a common intercept (e.g. scikit-learn's) would
silently put a constant into the force rows; that is why the methods are
implemented here with numpy/scipy.

Loss (all methods)
------------------
With row weights w_i = w_cat * w_unit^2 * w_user (ADR-0004, decision 2)
and W = sum_i w_i, on standardised descriptor columns (divided by the std
of d over the training E rows)::

    L(w) = (1/(2W)) sum_i w_i (y_i - a_i . w - c_i b)^2 + penalty(w)

    ols          : 0
    ridge        : (alpha/2) |w|^2
    smooth_ridge : (alpha/2) |Dw|^2,  D = first differences between
                   adjacent bins of the same descriptor block
    lasso        : alpha |w|_1
    elastic_net  : alpha (l1_ratio |w|_1 + (1 - l1_ratio)/2 |w|^2)
    pls          : no penalty; w restricted to n_components PLS
                   directions (NIPALS, PLS1)
    bayes_ridge  : Gaussian prior with one precision, set with the noise
                   precision by evidence maximisation (MacKay 1992)
    ard          : one prior precision per coefficient (Tipping 2001)

References: MacKay, Neural Comput. 4, 415 (1992); Tipping, J. Mach. Learn.
Res. 1, 211 (2001); Friedman, Hastie & Tibshirani, J. Stat. Softw. 33, 1
(2010) (coordinate descent); Wold et al., Chemom. Intell. Lab. Syst. 58,
109 (2001) (PLS).
"""

import re
from collections.abc import Sequence

import numpy as np

from .csv_io import (
    build_structure_index,
    load_mlip_csv,
    load_weights_csv,
    structure_grouped_split,
)
from .elm import ELMModel, feature_std

__all__ = ["LinearModel", "descriptor_blocks"]

METHODS = (
    "ols",
    "ridge",
    "smooth_ridge",
    "lasso",
    "elastic_net",
    "pls",
    "bayes_ridge",
    "ard",
)


def descriptor_blocks(columns: Sequence[str] | None, D: int) -> np.ndarray:
    """
    Block index of each descriptor column, for ``smooth_ridge``.

    A column named ``<block>_b<k>`` (e.g. ``RDF_Si-O_b0012``, the layout
    of ``atomic_descriptor_ver2.0.0.py``) belongs to ``<block>``; adjacent
    columns of one block are adjacent bins.  Any other name is a block of
    its own.  Without names every column is its own block, which makes
    the difference penalty empty.
    """
    if columns is None:
        return np.arange(D)
    keys = [
        re.sub(r"_b\d+$", "", str(c)) if re.search(r"_b\d+$", str(c)) else f"__col{j}"
        for j, c in enumerate(columns)
    ]
    _, idx = np.unique(keys, return_inverse=True)
    return idx.astype(np.int64)


class LinearModel:
    """
    Linear potential E/N = w . d + b trained on E, F and S rows together.

    Parameters
    ----------
    method        : one of ``ols``, ``ridge``, ``smooth_ridge``, ``lasso``,
                    ``elastic_net``, ``pls``, ``bayes_ridge``, ``ard``.
    alpha         : regularisation strength (ridge, smooth_ridge, lasso,
                    elastic_net), on standardised columns; see the module
                    docstring for the loss it enters.
    l1_ratio      : L1 share of the Elastic Net penalty, in (0, 1].
    n_components  : number of PLS components.
    alpha_search  : optional list of alphas (or, for pls, of component
                    counts); each is fitted on the training split and the
                    one with the lowest weighted validation loss is kept.
    w_energy, w_forces, w_stress : category weights.
    max_iter, tol : coordinate-descent / evidence-iteration limits.
    verbose       : print progress and the score table.
    """

    def __init__(
        self,
        method: str = "ridge",
        alpha: float = 1e-6,
        l1_ratio: float = 0.5,
        n_components: int = 20,
        alpha_search: Sequence | None = None,
        w_energy: float = 1.0,
        w_forces: float = 0.1,
        w_stress: float = 10.0,
        max_iter: int = 5000,
        tol: float = 1e-8,
        verbose: bool = True,
        min_feature_std: float = 1e-12,
    ):
        method = str(method).lower()
        if method not in METHODS:
            raise ValueError(
                f"LinearModel: unknown method '{method}'. "
                f"Choose from {', '.join(METHODS)}."
            )
        if method == "elastic_net" and not (0.0 < l1_ratio <= 1.0):
            raise ValueError(
                f"LinearModel: l1_ratio must be in (0, 1], got {l1_ratio}."
            )
        self.method = method
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.n_components = int(n_components)
        self.alpha_search = list(alpha_search) if alpha_search else None
        self.w_energy = float(w_energy)
        self.w_forces = float(w_forces)
        self.w_stress = float(w_stress)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        # mask columns with training std below it (ADR-0010 A4)
        self.min_feature_std = float(min_feature_std)
        # There is no row-independent form: every row is exact.
        self.row_independent = False
        self.descriptor_config = ""  # see ELMModel.descriptor_config
        # physics-informed parts (ADR-0014): pair baseline, isolated atom
        self.physics_config = ""

        self.coef_: np.ndarray | None = None  # w on raw d
        self.intercept_: float = 0.0  # b
        self._cov_std: np.ndarray | None = None  # posterior cov (std. cols)
        self._noise_prec: float | None = None
        self._X_mean = self._X_std = None
        self._columns = None
        self._fitted = False

    # ------------------------------------------------------------------
    def _row_weights(self, labels, w_unit, w_user):
        _wmap = {"E": self.w_energy, "F": self.w_forces, "S": self.w_stress}
        w_cat = np.array([_wmap.get(c, 1.0) for c in labels], dtype=np.float64)
        return w_cat * w_unit**2 * w_user

    def _design(self, labels, X, sid, n_at):
        """Rows a_i on standardised columns, and the intercept column c."""
        isE = labels == "E"
        A = np.empty_like(X, dtype=np.float64)
        A[isE] = (X[isE] - self._X_mean) / self._X_std
        A[~isE] = X[~isE] * n_at[sid[~isE]][:, None] / self._X_std
        return A, isE.astype(np.float64)

    @staticmethod
    def _project_intercept(A, t, c):
        """Remove the unpenalised intercept (column c, E rows only):
        returns P A, P t with P = I - c c^T / c^T c, and the map that
        recovers b from w."""
        cc = float(c @ c)
        if cc == 0.0:
            return A, t, lambda w: 0.0
        cA = (c @ A) / cc
        ct = float(c @ t) / cc
        return (A - np.outer(c, cA), t - c * ct, lambda w: ct - float(cA @ w))

    # ------------------------------------------------------------------
    #  Solvers on the projected, weighted system (rows scaled by sqrt(w/W))
    # ------------------------------------------------------------------
    def _solve(self, A, t, method, alpha, n_comp, blocks):
        D = A.shape[1]
        G = A.T @ A
        q = A.T @ t
        if method == "ols":
            return np.linalg.lstsq(A, t, rcond=None)[0]
        if method == "ridge":
            return np.linalg.solve(G + alpha * np.eye(D), q)
        if method == "smooth_ridge":
            same = blocks[1:] == blocks[:-1]
            rows = np.flatnonzero(same)
            Dm = np.zeros((rows.size, D))
            Dm[np.arange(rows.size), rows] = -1.0
            Dm[np.arange(rows.size), rows + 1] = 1.0
            # A tiny ridge keeps the system definite in a direction the
            # difference penalty leaves free and the data do not fix.
            jit = 1e-12 * np.trace(G) / D
            return np.linalg.solve(G + alpha * Dm.T @ Dm + jit * np.eye(D), q)
        if method in ("lasso", "elastic_net"):
            rho = 1.0 if method == "lasso" else self.l1_ratio
            return self._coordinate_descent(G, q, alpha * rho, alpha * (1.0 - rho))
        if method == "pls":
            return self._pls1(A, t, n_comp)
        if method in ("bayes_ridge", "ard"):
            return self._evidence(A, t, ard=(method == "ard"))
        raise AssertionError(method)

    def _coordinate_descent(self, G, q, l1, l2):
        """min (1/2) w^T G w - q^T w + l1 |w|_1 + (l2/2) |w|^2 by cyclic
        coordinate descent on the Gram matrix."""
        D = G.shape[0]
        w = np.zeros(D)
        diag = np.diag(G) + l2
        Gw = np.zeros(D)
        scale = max(float(np.abs(q).max()), 1e-300)
        for it in range(self.max_iter):
            dmax = 0.0
            for j in range(D):
                if diag[j] <= 0.0:
                    continue
                rho_j = q[j] - Gw[j] + G[j, j] * w[j]
                new = np.sign(rho_j) * max(abs(rho_j) - l1, 0.0) / diag[j]
                d = new - w[j]
                if d != 0.0:
                    Gw += G[:, j] * d
                    w[j] = new
                    dmax = max(dmax, abs(d) * diag[j])
            if dmax <= self.tol * scale:
                break
        self._cd_iter = it + 1
        return w

    @staticmethod
    def _pls1(A, t, n_comp):
        """PLS1 by NIPALS; returns the regression vector on A's columns."""
        X, y = A.copy(), t.copy()
        n_comp = int(min(n_comp, A.shape[1]))
        Wm, Pm, qv = [], [], []
        for _ in range(n_comp):
            w = X.T @ y
            nw = np.linalg.norm(w)
            if nw < 1e-300:
                break
            w /= nw
            s = X @ w
            ss = float(s @ s)
            if ss < 1e-300:
                break
            p = X.T @ s / ss
            qk = float(y @ s) / ss
            X -= np.outer(s, p)
            y -= qk * s
            Wm.append(w)
            Pm.append(p)
            qv.append(qk)
        if not Wm:
            return np.zeros(A.shape[1])
        Wm, Pm = np.array(Wm).T, np.array(Pm).T
        return Wm @ np.linalg.solve(Pm.T @ Wm, np.array(qv))

    def _evidence(self, A, t, ard):
        """Bayesian ridge / ARD by evidence maximisation.  The (scaled) row
        weights act as relative noise precisions.  Sets the posterior
        covariance and noise precision for predictive variances."""
        # MacKay's fixed-point updates (MacKay 1992; Tipping 2001 for ARD):
        #   Sigma = (beta G + diag(lam))^-1,  m = beta Sigma q,
        #   gamma_j = 1 - lam_j Sigma_jj,
        #   lam = gamma / m^2  (ARD, per coefficient) or sum(gamma) / |m|^2,
        #   beta = (n - sum(gamma)) / |t - A m|^2.
        # Convergence is judged on lam and beta (relative change), not on
        # m: starting from a strong prior, m can sit near 0 and change
        # little per step long before the hyperparameters have settled.
        # Bayesian ridge runs in the eigenbasis of G, where Sigma is
        # diagonal and no ill-conditioned inverse is formed.
        n, D = A.shape
        n_eff = max(n - 1, 1)  # one dof went to the intercept
        G = A.T @ A
        q = A.T @ t
        tt = float(t @ t)
        s, U = np.linalg.eigh(G)
        s = np.clip(s, 0.0, None)
        qU = U.T @ q
        # Directions G cannot resolve (eigenvalues at round-off level, e.g.
        # ACSF channels that are ~0 for every atom) carry no information:
        # their qU is round-off too.  Left in, m = beta qU / (beta s + lam)
        # blows up along them, lam = gamma / |m|^2 collapses, and the
        # iteration settles on a degenerate fixed point (SW-Si v2 after
        # ADR-0006: lam 1.5e-11, test error 14-38 sigma).  Zero them.
        null = s <= s.max() * D * np.finfo(float).eps
        s[null] = 0.0
        qU[null] = 0.0
        beta = n_eff / max(tt, 1e-300)
        lam = np.full(D, max(float(s.mean()), 1e-300) * beta * 1e-6)

        def _r2(m):
            # Direct residual: the expanded form tt - 2 m.q + m G m loses
            # all digits to cancellation once the fit is close.
            r = t - A @ m
            return max(float(r @ r), 1e-300)

        for it in range(self.max_iter):
            if ard:
                H = beta * G + np.diag(lam)
                try:
                    Sig = np.linalg.inv(H)
                except np.linalg.LinAlgError:
                    Sig = np.linalg.pinv(H)
                m = beta * Sig @ q
                gam = 1.0 - lam * np.diag(Sig)
                lam_new = np.clip(gam / np.maximum(m**2, 1e-300), 1e-12, 1e12)
            else:
                d = beta * s + lam[0]
                m = U @ (beta * qU / d)
                gam_tot = float(np.sum(beta * s / d))
                gam = np.full(D, gam_tot / D)
                lam_new = np.full(D, gam_tot / max(float(m @ m), 1e-300))
            beta_new = max(n_eff - float(gam.sum()), 1e-12) / _r2(m)
            dl = np.max(np.abs(np.log(lam_new) - np.log(lam)))
            db = abs(np.log(beta_new) - np.log(beta))
            lam, beta = lam_new, beta_new
            if max(dl, db) < 1e-8:
                break
        self._evidence_iter = it + 1
        if ard:
            H = beta * G + np.diag(lam)
            Sig = np.linalg.pinv(H)
            m = beta * Sig @ q
        else:
            d = beta * s + lam[0]
            Sig = (U / d) @ U.T
            m = U @ (beta * qU / d)
        self._cov_std = Sig
        self._noise_prec = beta
        self._prior_prec = lam
        return m

    # ------------------------------------------------------------------
    def fit(
        self,
        targets: np.ndarray,
        labels: np.ndarray,
        X: np.ndarray,
        weights: np.ndarray | None = None,
        weight_csv: str | None = None,
        val_fraction: float = 0.2,
        val_seed: int = 0,
        struct_ids: np.ndarray | None = None,
        n_atoms: np.ndarray | None = None,
        columns: Sequence[str] | None = None,
    ) -> "LinearModel":
        """
        Fit on a training-format table (see ELMModel.fit for the arguments;
        ``columns`` are the descriptor column names, used by smooth_ridge).
        """
        from ..physics import physics_from_config
        if physics_from_config(self.physics_config)[1]:
            raise ValueError(
                "LinearModel: isolated_atom_energy is supported by elm / qelm "
                "(tabular with one species, atom-centred) and the "
                "atom-centred nn (ADR-0014).")
        labels = np.array([str(c).strip().upper() for c in labels])
        N, D = X.shape
        w_unit = (
            np.asarray(weights, dtype=np.float64) if weights is not None else np.ones(N)
        )
        if w_unit.shape != (N,) or np.any(w_unit < 0):
            raise ValueError("LinearModel.fit: weights must be (N,) and >= 0.")
        w_user = (
            load_weights_csv(weight_csv, n_total=N, verbose=self.verbose)
            if weight_csv is not None
            else np.ones(N)
        )
        w_row = self._row_weights(labels, w_unit, w_user)

        sid, e_row, n_at = build_structure_index(
            labels, struct_ids, n_atoms, caller="LinearModel.fit"
        )
        idx_tr, idx_va = structure_grouped_split(
            labels,
            val_fraction=val_fraction,
            val_seed=val_seed,
            caller="LinearModel.fit",
            struct_ids=struct_ids,
        )

        E_tr = e_row[np.unique(sid[idx_tr])]
        self._X_mean = X[E_tr].mean(axis=0)
        self._X_std = feature_std(X[E_tr], self.min_feature_std)
        self._columns = list(columns) if columns is not None else None
        blocks = descriptor_blocks(self._columns, D)

        A_all, c_all = self._design(labels, X, sid, n_at)

        def _system(idx):
            s = np.sqrt(w_row[idx] / w_row[idx].sum())
            return (A_all[idx] * s[:, None], targets[idx] * s, c_all[idx] * s)

        def _fit_one(idx, alpha, n_comp):
            A, t, c = _system(idx)
            Ap, tp, b_of = self._project_intercept(A, t, c)
            w_std = self._solve(Ap, tp, self.method, alpha, n_comp, blocks)
            return w_std, b_of(w_std)

        def _val_loss(idx, w_std, b_std):
            A, t, c = _system(idx)
            return float(np.sum((t - A @ w_std - c * b_std) ** 2))

        if self.verbose:
            print(f"\nLinearModel fitting  [method={self.method}]")
            print(f"  Samples : {N}  (D={D})  train={idx_tr.size}  val={idx_va.size}")

        self.search_results_ = None
        if self.alpha_search and self.method not in ("ols", "bayes_ridge", "ard"):
            if idx_va.size == 0:
                raise ValueError(
                    "LinearModel.fit: alpha_search needs a "
                    "validation split (val_fraction > 0 and "
                    "at least two structures)."
                )
            res = []
            for a in self.alpha_search:
                alpha, nc = (
                    (self.alpha, int(a))
                    if self.method == "pls"
                    else (float(a), self.n_components)
                )
                w_s, b_s = _fit_one(idx_tr, alpha, nc)
                res.append((a, _val_loss(idx_va, w_s, b_s)))
                if self.verbose:
                    print(f"  search {a!s:>10}: weighted val loss {res[-1][1]:.6e}")
            best = min(res, key=lambda r: r[1])[0]
            if self.method == "pls":
                self.n_components = int(best)
            else:
                self.alpha = float(best)
            self.search_results_ = res
            if self.verbose:
                print(f"  selected: {best}")

        w_std, b_std = _fit_one(idx_tr, self.alpha, self.n_components)
        self._w_std = w_std
        self.coef_ = w_std / self._X_std
        self.intercept_ = b_std - float(self.coef_ @ self._X_mean)
        self._fitted = True

        if self.verbose and self.method in ("lasso", "elastic_net"):
            nz = int(np.count_nonzero(self.coef_))
            print(f"  non-zero coefficients: {nz} / {D}")

        def _sub(idx):
            return dict(
                labels=labels[idx],
                X=X[idx],
                struct_ids=(
                    None if struct_ids is None else np.asarray(struct_ids)[idx]
                ),
                n_atoms=(None if n_atoms is None else np.asarray(n_atoms)[idx]),
            )

        self._train_sc = ELMModel._score(
            targets[idx_tr], self.predict_table(**_sub(idx_tr)), labels[idx_tr], "train"
        )
        self._val_sc = (
            ELMModel._score(
                targets[idx_va],
                self.predict_table(**_sub(idx_va)),
                labels[idx_va],
                "val",
            )
            if idx_va.size
            else None
        )
        if self.verbose:
            ELMModel._print_scores(self._train_sc, self._val_sc)
        return self

    def fit_csv(self, csv_path, weight_csv=None, val_fraction=0.2, val_seed=0):
        """Load a training CSV (load_mlip_csv format) and fit."""
        targets, labels, weights, X, meta = load_mlip_csv(
            csv_path, verbose=self.verbose, return_meta=True
        )
        return self.fit(
            targets,
            labels,
            X,
            weights=weights,
            weight_csv=weight_csv,
            val_fraction=val_fraction,
            val_seed=val_seed,
            struct_ids=meta["struct_id"],
            n_atoms=meta["n_atoms"],
            columns=meta["columns"],
        )

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """E/N for structure descriptors X (rows are E rows)."""
        if not self._fitted:
            raise RuntimeError("LinearModel: call fit() first.")
        return X @ self.coef_ + self.intercept_

    def predict_table(self, labels, X, struct_ids=None, n_atoms=None):
        """Predict every row of a training-format table."""
        if not self._fitted:
            raise RuntimeError("LinearModel: call fit() first.")
        labels = np.array([str(c).strip().upper() for c in labels])
        sid, _, n_at = build_structure_index(
            labels, struct_ids, n_atoms, caller="LinearModel.predict_table"
        )
        isE = labels == "E"
        out = np.empty(len(labels))
        out[isE] = X[isE] @ self.coef_ + self.intercept_
        out[~isE] = n_at[sid[~isE]] * (X[~isE] @ self.coef_)
        return out

    def predict_std_table(self, labels, X, struct_ids=None, n_atoms=None):
        """
        Posterior standard deviation of the model part of each row's
        prediction (bayes_ridge / ard only), sqrt(a^T Sigma a), in target
        units.  The intercept's own uncertainty is not included.
        """
        if self._cov_std is None:
            raise RuntimeError(
                "LinearModel.predict_std_table: available for bayes_ridge / ard only."
            )
        labels = np.array([str(c).strip().upper() for c in labels])
        sid, _, n_at = build_structure_index(labels, struct_ids, n_atoms)
        A, _ = self._design(labels, X, sid, n_at)
        return np.sqrt(np.einsum("ij,jk,ik->i", A, self._cov_std, A))

    def energy_and_gradient(self, d: np.ndarray):
        """f(d) = E/N and its gradient w.r.t. d (used by the MLIP engine)."""
        return float(d @ self.coef_ + self.intercept_), self.coef_.copy()

    def score_csv(self, csv_path: str, label: str = "test") -> dict:
        targets, labels, weights, X, meta = load_mlip_csv(
            csv_path, verbose=False, return_meta=True
        )
        y = self.predict_table(labels, X, meta["struct_id"], meta["n_atoms"])
        sc = ELMModel._score(targets, y, np.asarray(labels), label)
        ELMModel._print_scores(sc)
        return sc

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError("LinearModel: model has not been fitted yet.")
        p = path if path.endswith(".npz") else path + ".npz"
        np.savez(
            p,
            model_class=np.array("LinearModel"),
            method=np.array(self.method),
            alpha=np.array(self.alpha),
            l1_ratio=np.array(self.l1_ratio),
            n_components=np.array(self.n_components),
            w_energy=np.array(self.w_energy),
            w_forces=np.array(self.w_forces),
            w_stress=np.array(self.w_stress),
            coef=self.coef_,
            intercept=np.array(self.intercept_),
            X_mean=self._X_mean,
            X_std=self._X_std,
            cov_std=(self._cov_std if self._cov_std is not None else np.zeros((0, 0))),
            descriptor_config=np.array(self.descriptor_config),
            physics_config=np.array(self.physics_config),
        )
        if self.verbose:
            print(f"  LinearModel saved -> {p}")

    @classmethod
    def load(cls, path: str, verbose: bool = False) -> "LinearModel":
        p = path if path.endswith(".npz") else path + ".npz"
        z = np.load(p)
        obj = cls(
            method=str(z["method"]),
            alpha=float(z["alpha"]),
            l1_ratio=float(z["l1_ratio"]),
            n_components=int(z["n_components"]),
            w_energy=float(z["w_energy"]),
            w_forces=float(z["w_forces"]),
            w_stress=float(z["w_stress"]),
            verbose=verbose,
        )
        obj.coef_ = z["coef"]
        obj.intercept_ = float(z["intercept"])
        obj._X_mean, obj._X_std = z["X_mean"], z["X_std"]
        obj._cov_std = z["cov_std"] if z["cov_std"].size else None
        obj.descriptor_config = str(z["descriptor_config"])
        obj.physics_config = str(z["physics_config"]) if "physics_config" in z else ""
        obj._fitted = True
        return obj
