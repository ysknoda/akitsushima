"""Power-of-2 weight quantization ELM: multiply-free inference via
per-column power-of-two + zero-sparsified hidden-layer weights.
"""
import time as _time
from typing import Optional

import numpy as np

from .elm import ELMModel

__all__ = ["QuantizedELMModel"]


class QuantizedELMModel(ELMModel):
    """
    Power-of-2 weight quantization ELM.

    Each entry of the hidden-layer weight matrix W is rounded to the
    nearest signed power of two (times a per-column scale s_j):

        levels(n) = { -2^n, ..., -2, -1, 0, 1, 2, ..., 2^n }

    A fraction p_zero of the weights is forced to exactly zero, so the
    matrix H can be stored as a scipy.sparse CSC matrix for fast,
    multiply-free inference (products become bit shifts and adds).

    Parameters
    ----------
    n_bits       : exponent range n (levels +/-1, +/-2, +/-4, ..., +/-2^n, and 0)
    p_zero       : target fraction in [0, 1) of weights forced to zero
    use_sparse_w : True = store W as CSC sparse (recommended when p_zero is high)

    Examples
    --------
    >>> model = QuantizedELMModel(n_hidden=512, n_bits=3, p_zero=0.5)
    >>> model.fit_csv("training.csv")
    >>> model.print_quantization_report()
    """

    def __init__(self,
                 n_hidden        : int   = 512,
                 activation      : str   = 'tanh',
                 w_energy        : float = 1.0,
                 w_forces        : float = 0.1,
                 w_stress        : float = 10.0,
                 r0              : float = 1.0,
                 eps0            : float = 1.0,
                 reg             : float = 1e-6,
                 lsqr_maxiter    : int   = 2000,
                 lsqr_atol       : float = 1e-8,
                 seed            : int   = 42,
                 full_matrix_free: bool  = False,
                 chunk_size      : int   = 1024,
                 use_float32     : bool  = False,
                 snake_a         : float = 1.0,
                 rff_gamma       : Optional[float] = None,
                 w_init_scale    : Optional[float] = None,
                 n_bits          : int   = 3,
                 p_zero          : float = 0.5,
                 use_sparse_w    : bool  = True,
                 verbose         : bool  = True,
                 row_independent : bool  = False,
                 direct_link     : bool  = True,
                 reg_search      : Optional[list] = None):

        super().__init__(
            n_hidden         = n_hidden,
            activation       = activation,
            w_energy         = w_energy,
            w_forces         = w_forces,
            w_stress         = w_stress,
            r0               = r0,
            eps0             = eps0,
            reg              = reg,
            lsqr_maxiter     = lsqr_maxiter,
            lsqr_atol        = lsqr_atol,
            seed             = seed,
            full_matrix_free = full_matrix_free,
            chunk_size       = chunk_size,
            use_float32      = use_float32,
            snake_a          = snake_a,
            rff_gamma        = rff_gamma,
            w_init_scale     = w_init_scale,
            verbose          = verbose,
            row_independent  = row_independent,
            direct_link      = direct_link,
            reg_search       = reg_search,
        )
        if not (0 <= p_zero < 1):
            raise ValueError(f"p_zero must be in [0, 1), got {p_zero}")
        if n_bits < 1:
            raise ValueError(f"n_bits must be >= 1, got {n_bits}")

        self.n_bits       = n_bits
        self.p_zero       = p_zero
        self.use_sparse_w = use_sparse_w

        # Filled by _quantize_W()
        self._W_scale   : Optional[np.ndarray] = None  # (N_h,) per-col scale
        self._W_q_dense : Optional[np.ndarray] = None  # (D, N_h) quantized
        self._W_sparse  = None                          # scipy CSC sparse
        self._quant_stats: Optional[dict]      = None
        # Filled by fit(): full-precision W before quantisation (bug-fix
        # F8, see the note in fit() below).
        self._W_fp: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _pow2_levels(n: int) -> np.ndarray:
        """levels(n) = {-2^n, ..., -1, 0, 1, ..., 2^n}."""
        pos = [2**k for k in range(n + 1)]
        neg = [-2**k for k in range(n + 1)]
        return np.array(sorted(set(neg + [0] + pos)), dtype=np.float64)

    # ------------------------------------------------------------------
    def _quantize_W(self, W_raw: np.ndarray) -> np.ndarray:
        """
        Quantize the raw weight matrix W_raw to powers of two.

        Algorithm
        ---------
        For each column j (each hidden unit j):
          1. scale:    s_j = max|W_raw[:,j]| / 2^n_bits
          2. normalise: w_scaled = W_raw[:,j] / s_j
          3. snap:     w_q = argmin_{lambda in levels} |w_scaled - lambda|
          4. sparsify: force p_zero fraction to 0 (smallest |w_scaled|)
          5. rescale:  W_q[:,j] = w_q * s_j

        Parameters
        ----------
        W_raw : (D, N_h) raw random weight matrix.

        Returns
        -------
        W_q : (D, N_h) quantized weight matrix (powers of two, sparsified).
        """
        levels = self._pow2_levels(self.n_bits)
        D, N_h = W_raw.shape
        W_q     = np.empty_like(W_raw)
        scales  = np.empty(N_h)

        zero_counts = np.zeros(N_h, dtype=np.int64)

        for j in range(N_h):
            col     = W_raw[:, j]
            max_abs = float(np.max(np.abs(col)))

            if max_abs < 1e-12:
                W_q[:, j]   = 0.0
                scales[j]   = 1.0
                zero_counts[j] = D
                continue

            s      = max_abs / (2 ** self.n_bits)
            scales[j] = s
            col_s  = col / s                        # scaled to [-2^n, 2^n]

            # vectorised nearest-level snap
            idx   = np.argmin(np.abs(col_s[:, None] - levels[None, :]), axis=1)
            col_q = levels[idx].copy()              # quantized scaled values

            # p_zero: force the smallest-magnitude entries of |col_s| to 0
            if self.p_zero > 0.0:
                thr     = float(np.quantile(np.abs(col_s), self.p_zero))
                mask_z  = np.abs(col_s) <= thr
                col_q[mask_z] = 0.0

            zero_counts[j] = int(np.sum(col_q == 0))
            W_q[:, j]      = col_q * s              # back to original scale

        total_elems   = D * N_h
        total_zeros   = int(np.sum(W_q == 0))
        sparsity      = total_zeros / total_elems
        unique_levels = np.unique(W_q)

        self._W_scale    = scales
        self._W_q_dense  = W_q
        self._quant_stats = {
            "n_bits"       : self.n_bits,
            "p_zero"       : self.p_zero,
            "levels"       : levels.tolist(),
            "n_levels"     : len(levels),
            "sparsity_W"   : float(sparsity),
            "total_zeros"  : total_zeros,
            "total_elems"  : total_elems,
            "unique_W_vals": len(unique_levels),
            "W_mem_dense_MB" : W_q.nbytes / 1e6,
        }

        # scipy CSC sparse matrix for multiply-free inference
        if self.use_sparse_w:
            from scipy.sparse import csc_matrix
            dtype = np.float32 if self.use_float32 else np.float64
            self._W_sparse = csc_matrix(W_q.astype(dtype))
            nnz = self._W_sparse.nnz
            self._quant_stats["nnz_W"]       = nnz
            self._quant_stats["W_mem_csc_MB"] = (
                self._W_sparse.data.nbytes
                + self._W_sparse.indices.nbytes
                + self._W_sparse.indptr.nbytes) / 1e6
        else:
            self._W_sparse = None

        return W_q

    # ------------------------------------------------------------------
    def _hidden(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Compute the hidden matrix using the quantized W.

        When use_sparse_w=True, uses scipy sparse matmul for the product.
        In RFF mode the quantized W is used directly.
        """
        # Fall back to the dense ELMModel path before fit() has quantised W.
        if self._W_q_dense is None:
            return super()._hidden(X_norm)

        if self.activation == 'rff':
            # RFF: W shape (D, N_omega), b=None
            if self.use_sparse_w and self._W_sparse is not None:
                X_sp = X_norm.astype(np.float32) if self.use_float32 else X_norm
                Z = np.array(X_sp @ self._W_sparse)   # dense result
                if self.use_float32: Z = Z.astype(np.float64)
            else:
                W_use = (self._W_q_dense.astype(np.float32)
                         if self.use_float32 else self._W_q_dense)
                X_use = (X_norm.astype(np.float32)
                         if self.use_float32 else X_norm)
                Z = X_use @ W_use
                if self.use_float32: Z = Z.astype(np.float64)
            N_omega = self._W_q_dense.shape[1]
            return np.hstack([np.cos(Z), np.sin(Z)]) / np.sqrt(N_omega)

        # Standard activations
        b = self._b
        if self.use_sparse_w and self._W_sparse is not None:
            X_sp = X_norm.astype(np.float32) if self.use_float32 else X_norm
            Z = np.array(X_sp @ self._W_sparse)
            if self.use_float32: Z = Z.astype(np.float64)
            Z = Z + b
        else:
            W_use = (self._W_q_dense.astype(np.float32)
                     if self.use_float32 else self._W_q_dense)
            X_use = (X_norm.astype(np.float32)
                     if self.use_float32 else X_norm)
            Z = X_use @ W_use + (b.astype(np.float32) if self.use_float32 else b)
            if self.use_float32: Z = Z.astype(np.float64)
        return self._act(Z)

    # ------------------------------------------------------------------
    def fit(self,
            targets      : np.ndarray,
            labels       : np.ndarray,
            X            : np.ndarray,
            weights      : Optional[np.ndarray] = None,
            weight_csv   : Optional[str]         = None,
            val_fraction : float = 0.2,
            val_seed     : int   = 0,
            struct_ids   : Optional[np.ndarray] = None,
            n_atoms      : Optional[np.ndarray] = None,
            ) -> 'QuantizedELMModel':
        """
        Fit and quantise the ELM.

        Quantises the random hidden-layer weight matrix W to n_bits-bit
        fixed-point, then delegates all training (weight composition,
        chain-rule design rows, LSQR, score reporting) to
        :meth:`ELMModel.fit` via ``super().fit()``.  The chain-rule rows use
        the quantised W, so forces are derivatives of the quantised model.

        Parameters
        ----------
        targets      : (N,) float -- targets from :func:`load_mlip_csv`.
        labels       : (N,) str   -- labels from :func:`load_mlip_csv`.
        X            : (N, D)     -- descriptor matrix.
        weights      : (N,) float or None -- unit weights from :func:`load_mlip_csv`.
        weight_csv   : str or None -- path to user weight file.
        val_fraction : float
        val_seed     : int
        struct_ids, n_atoms : optional CSV metadata columns (see ELMModel.fit).

        Returns
        -------
        self
        """
        _, D = X.shape
        # Initialise raw random W before quantisation
        rng = np.random.default_rng(self.seed)
        if self.activation == 'rff':
            N_omega = max(1, self.n_hidden // 2)
            gamma   = (self.rff_gamma or float(np.sqrt(D / 2.0)))
            W_raw   = rng.standard_normal((D, N_omega)) / gamma
            self._b = None
        else:
            sc      = (float(self.w_init_scale) if self.w_init_scale
                       else float(np.sqrt(2.0 / (D + self.n_hidden))))
            W_raw   = rng.standard_normal((D, self.n_hidden)) * sc
            self._b = rng.standard_normal(self.n_hidden) * sc

        # Bug-fix note (F8): the original never stored W_raw anywhere, so
        # quantization_error() -- whose entire purpose is to compare
        # quantised vs full-precision predictions -- could never find a
        # full-precision reference and silently fell back to comparing
        # the quantised model against itself (always reporting exactly
        # zero error).  Store it before quantising so that
        # quantization_error() has something real to compare against.
        self._W_fp = W_raw.copy()

        # Quantise W; ELMModel.fit will use self._W as-is
        t_q0    = _time.perf_counter()
        self._W = self._quantize_W(W_raw)
        t_q     = _time.perf_counter() - t_q0
        if self.verbose:
            qs = self._quant_stats
            print(f'  Quantization ({t_q*1000:.0f} ms):  '
                  f'levels={qs["levels"]}  sparsity={qs["sparsity_W"]:.3f}')
        # Delegate training to ELMModel.fit (weights, LSQR, scoring)
        _orig_seed = self.seed
        self.seed  = None   # skip W/b re-init inside ELMModel.fit
        try:
            super().fit(targets, labels, X,
                        weights      = weights,
                        weight_csv   = weight_csv,
                        val_fraction = val_fraction,
                        val_seed     = val_seed,
                        struct_ids   = struct_ids,
                        n_atoms      = n_atoms)
        finally:
            self.seed = _orig_seed
        return self

    def print_quantization_report(self) -> None:
        """
        Display a detailed report of quantisation settings and statistics.
        """
        if not self._fitted or self._quant_stats is None:
            print("  [not fitted] call fit() first.")
            return

        qs = self._quant_stats
        print("\n" + "=" * 58)
        print("  Quantized ELM -- report")
        print("=" * 58)
        print(f"  n_bits      : {qs['n_bits']}")
        print(f"  p_zero      : {qs['p_zero']}")
        print(f"  levels      : {qs['levels']}")
        print(f"  n_levels    : {qs['n_levels']}  "
              f"(2n+3 = {2*qs['n_bits']+3} levels)")
        print()
        print(f"  W sparsity  : {qs['sparsity_W']:.4f}  "
              f"({qs['total_zeros']:,} / {qs['total_elems']:,} are 0)")
        print(f"  Unique W    : {qs['unique_W_vals']}")
        print()
        mem_dense = qs["W_mem_dense_MB"]
        print(f"  W mem (dense) : {mem_dense:.2f} MB")
        if "W_mem_csc_MB" in qs:
            mem_csc = qs["W_mem_csc_MB"]
            print(f"  W mem (CSC)   : {mem_csc:.2f} MB  "
                  f"(compression {mem_dense/max(mem_csc,1e-6):.1f}x  "
                  f"nnz={qs['nnz_W']:,})")
        print()
        print("  Notes:")
        print("    W_q values are of the form +/- 2^k . s_j (powers of two).")
        print("    x . (2^k . s_j)  ==  s_j . (x << k)  (bit-shift product).")
        print("    -> multiply-free inference via shifts and adds.")
        sp = qs['sparsity_W']
        print(f"    -> about {sp*100:.0f}% of weights are zero "
              f"({(1-sp)*100:.0f}% non-zero).")
        print("=" * 58)

    # ------------------------------------------------------------------
    def quantization_error(self,
                            X       : np.ndarray,
                            y_true  : np.ndarray,
                            ) -> dict:
        """
        Quantify the accuracy loss caused by weight quantization.

        Compares predictions of the quantized model against those of the
        same model with full-precision (float64) weights, computed on the
        provided descriptor matrix ``X``.

        Parameters
        ----------
        X      : (N, D) descriptor matrix (same format as used in ``fit``).
        y_true : (N,) reference target values.

        Returns
        -------
        dict with keys
            mae_quant    -- MAE of quantized predictions vs y_true
            rmse_quant   -- RMSE of quantized predictions vs y_true
            mae_fp       -- MAE of full-precision predictions vs y_true
            rmse_fp      -- RMSE of full-precision predictions vs y_true
            mae_delta    -- MAE of (quant - fp) prediction difference
            rmse_delta   -- RMSE of (quant - fp) prediction difference
            rel_delta_pct -- 100 x rmse_delta / rmse_fp  (% accuracy loss)

        Raises
        ------
        RuntimeError  if the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "QuantizedELMModel.quantization_error: call fit() first.")

        # Quantized predictions.
        y_quant = self.predict(X)

        # Full-precision predictions: temporarily swap in the full-precision
        # weights and re-predict.
        #
        # Bug-fix note (found alongside F8): the original swapped
        # ``self._W`` here, but ``QuantizedELMModel._hidden()`` -- once
        # ``fit()`` has run -- never reads ``self._W`` at all; it always
        # forwards through ``self._W_q_dense`` / ``self._W_sparse`` (see
        # ``_hidden()`` above).  So even after fixing F8 (storing
        # ``self._W_fp``), swapping ``self._W`` alone had zero effect on
        # ``self.predict()`` and ``y_fp`` was silently identical to
        # ``y_quant`` -- the "quantisation error" was still always exactly
        # 0.  The swap must instead target ``self._W_q_dense`` (and disable
        # ``self._W_sparse`` so the dense fallback path in ``_hidden()`` is
        # taken) to actually force a full-precision forward pass.
        W_q_dense = self._W_q_dense
        W_sparse  = self._W_sparse
        W_fp      = self._W_fp
        if W_fp is None:
            # No fp reference available (e.g. a reloaded model -- see
            # QuantizedELMModel.load); fall back to comparing against
            # itself rather than crashing.
            y_fp = y_quant.copy()
        else:
            self._W_q_dense = W_fp
            self._W_sparse  = None
            try:
                y_fp = self.predict(X)
            finally:
                self._W_q_dense = W_q_dense
                self._W_sparse  = W_sparse

        def _mae(a, b):  return float(np.mean(np.abs(a - b)))
        def _rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))

        mae_q   = _mae(y_quant, y_true)
        rmse_q  = _rmse(y_quant, y_true)
        mae_fp  = _mae(y_fp, y_true)
        rmse_fp = _rmse(y_fp, y_true)
        delta   = y_quant - y_fp
        mae_d   = float(np.mean(np.abs(delta)))
        rmse_d  = float(np.sqrt(np.mean(delta**2)))
        rel_pct = 100.0 * rmse_d / max(rmse_fp, 1e-30)

        return {
            "mae_quant"    : mae_q,
            "rmse_quant"   : rmse_q,
            "mae_fp"       : mae_fp,
            "rmse_fp"      : rmse_fp,
            "mae_delta"    : mae_d,
            "rmse_delta"   : rmse_d,
            "rel_delta_pct": rel_pct,
        }

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the Quantized ELM model to a .npz file."""
        if not self._fitted:
            raise RuntimeError("QuantizedELMModel: model has not been fitted.")
        np.savez(path,
                 # ELMModel
                 W               = self._W,
                 b               = (self._b if self._b is not None
                                    else np.array([])),
                 beta            = self._beta,
                 X_mean          = self._X_mean,
                 X_std           = self._X_std,
                 n_hidden        = np.array(self.n_hidden),
                 activation      = np.array(self.activation),
                 w_energy        = np.array(self.w_energy),
                 w_forces        = np.array(self.w_forces),
                 w_stress        = np.array(self.w_stress),
                 reg             = np.array(self.reg),
                 seed            = np.array(self.seed),
                 full_matrix_free= np.array(self.full_matrix_free),
                 chunk_size      = np.array(self.chunk_size),
                 use_float32     = np.array(self.use_float32),
                 snake_a         = np.array(self.snake_a),
                 rff_gamma       = np.array(self.rff_gamma
                                            if self.rff_gamma is not None
                                            else -1.0),
                 w_init_scale    = np.array(self.w_init_scale
                                            if self.w_init_scale is not None
                                            else -1.0),
                 r0              = np.array(self.r0),
                 eps0            = np.array(self.eps0),
                 # QuantizedELMModel
                 model_class     = np.array("QuantizedELMModel"),
                 n_bits          = np.array(self.n_bits),
                 p_zero          = np.array(self.p_zero),
                 use_sparse_w    = np.array(self.use_sparse_w),
                 W_q_dense       = self._W_q_dense,
                 W_scale         = self._W_scale,
                 **self._extra_state())
        if self.verbose:
            print(f"  QuantizedELMModel saved -> {path}.npz")

    @classmethod
    def load(cls, path: str, verbose: bool = False) -> 'QuantizedELMModel':
        """Load a saved QuantizedELMModel from a .npz archive."""
        # Bug-fix note (F12): see ELMModel.load() -- nothing saved here
        # needs pickle support either.
        p   = path if path.endswith('.npz') else path + '.npz'
        npz = np.load(p)
        obj = cls(
            n_hidden         = int(npz['n_hidden']),
            activation       = str(npz['activation']),
            w_energy         = float(npz['w_energy']),
            w_forces         = float(npz['w_forces']),
            w_stress         = float(npz['w_stress']),
            reg              = float(npz['reg']),
            seed             = int(npz['seed']),
            full_matrix_free = bool(npz.get('full_matrix_free', False)),
            chunk_size       = int(npz.get('chunk_size', 1024)),
            use_float32      = bool(npz.get('use_float32', False)),
            snake_a          = float(npz.get('snake_a', 1.0)),
            rff_gamma        = (float(npz['rff_gamma'])
                                if 'rff_gamma' in npz
                                   and float(npz['rff_gamma']) > 0
                                else None),
            w_init_scale     = (float(npz['w_init_scale'])
                                if 'w_init_scale' in npz
                                   and float(npz['w_init_scale']) > 0
                                else None),
            r0               = float(npz['r0'])   if 'r0'   in npz else 1.0,
            eps0             = float(npz['eps0']) if 'eps0' in npz else 1.0,
            n_bits           = int(npz['n_bits']),
            p_zero           = float(npz['p_zero']),
            use_sparse_w     = bool(npz.get('use_sparse_w', True)),
            verbose          = verbose,
        )
        obj._W         = npz['W']
        b_arr          = npz['b'] if 'b' in npz else np.array([])
        obj._b         = b_arr if b_arr.size > 0 else None
        obj._beta      = npz['beta']
        obj._X_mean    = npz['X_mean']
        obj._X_std     = npz['X_std']
        obj._D         = obj._W.shape[0]
        obj._W_q_dense = npz['W_q_dense']
        obj._W_scale   = npz['W_scale']
        obj._restore_extra_state(npz)
        obj._fitted    = True
        # A reloaded model has no full-precision reference available
        # (it was never saved); quantization_error() falls back to
        # comparing the quantised model against itself in that case.
        obj._W_fp      = None
        if obj.use_sparse_w:
            from scipy.sparse import csc_matrix
            dtype = np.float32 if obj.use_float32 else np.float64
            obj._W_sparse = csc_matrix(obj._W_q_dense.astype(dtype))
        return obj
