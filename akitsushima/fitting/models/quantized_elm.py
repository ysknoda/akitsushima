"""Quantised-weight ELM: hidden-layer weights rounded per column to a
small level set (powers of two by default, ternary or uniform integers,
ADR-0013) and partly zeroed.
"""
import time as _time
import warnings
from typing import Optional

import numpy as np

from .elm import ELMModel
from ._api import tabular_mlip

__all__ = ["QuantizedELMModel"]


@tabular_mlip
class QuantizedELMModel(ELMModel):
    """
    Power-of-2 weight quantization ELM.

    Each entry of the hidden-layer weight matrix W is rounded to the
    nearest level of a small set (times a per-column scale s_j):

        levels 'pow2'    : {0, +-1, +-2, ..., +-2^n}, n = max_exponent,
                           s_j = max|W[:, j]| / 2^n  (default; 2n + 3 levels)
        levels 'ternary' : {-1, 0, 1}, s_j chosen so that the column keeps
                           the variance of the raw draw
        levels 'uniform' : {-k, ..., k}, k = (n_levels - 1) / 2,
                           s_j = max|W[:, j]| / k  (integer arithmetic)

    A fraction p_zero of each column (the smallest raw magnitudes) is set
    to exactly zero.

    Storage (ADR-0013): W is kept as integer levels K (int8; int16 when
    |level| > 127, i.e. pow2 with max_exponent > 6) and the column scales
    s, W = K * s exactly.  Model files store K and s only.  ``w_memory``
    chooses what stays in memory after fit: 'int8' (default) keeps K and
    s only and forms X W as (X K) * s on each call (8x less W memory; a
    transient float copy of K is made inside the product; engine calls up
    to 12 % slower); 'float64' also keeps the dense float W.

    Parameters
    ----------
    levels       : 'pow2' | 'ternary' | 'uniform'
    max_exponent : largest power-of-two exponent n of 'pow2' (formerly
                   ``n_bits``, which stays as a deprecated alias: it is not
                   a bit count -- 9 levels need 4 bits)
    n_levels     : odd number of levels of 'uniform' (>= 3)
    p_zero       : target fraction in [0, 1) of weights forced to zero
    use_sparse_w : allow a CSC copy of W (built only for matmul 'sparse')
    matmul       : 'dense' (default) or 'sparse' -- the product X W of the
                   chain-rule rows and the engine uses the CSC copy
    w_memory     : 'int8' (default) | 'float64' -- see Storage above
    keep_fp      : keep the full-precision draw after fit (needed only by
                   :meth:`quantization_error`; default False)

    Examples
    --------
    >>> model = QuantizedELMModel(n_hidden=512, max_exponent=3, p_zero=0.5)
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
                 eps0            : float = 0.01,
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
                 max_exponent    : Optional[int] = None,
                 p_zero          : float = 0.5,
                 use_sparse_w    : bool  = True,
                 verbose         : bool  = True,
                 row_independent : bool  = False,
                 direct_link     : bool  = True,
                 reg_search      : Optional[list] = None,
                 levels          : str   = 'pow2',
                 n_levels        : int   = 3,
                 matmul          : str   = 'dense',
                 n_bits          : Optional[int] = None,
                 w_memory        : str   = 'int8',
                 keep_fp         : bool  = False,
                 **elm_kw):

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
            **elm_kw,
        )
        if n_bits is not None:
            warnings.warn("QuantizedELMModel: n_bits is deprecated, use "
                          "max_exponent (ADR-0013).", DeprecationWarning,
                          stacklevel=2)
            if max_exponent is None:
                max_exponent = n_bits
        max_exponent = 3 if max_exponent is None else int(max_exponent)
        if not (0 <= p_zero < 1):
            raise ValueError(f"p_zero must be in [0, 1), got {p_zero}")
        if levels not in ('pow2', 'ternary', 'uniform'):
            raise ValueError(f"levels must be 'pow2', 'ternary' or "
                             f"'uniform', got {levels!r}")
        if levels == 'pow2' and not 1 <= max_exponent <= 14:
            raise ValueError(f"max_exponent must be 1..14, got {max_exponent} "
                             f"(use levels: ternary for {{-1, 0, 1}})")
        if levels == 'uniform' and (n_levels < 3 or n_levels % 2 == 0
                                    or n_levels > 255):
            raise ValueError(f"n_levels must be odd, 3..255, got {n_levels}")
        if matmul not in ('dense', 'sparse'):
            raise ValueError(f"matmul must be 'dense' or 'sparse', got "
                             f"{matmul!r}")
        if matmul == 'sparse' and not use_sparse_w:
            raise ValueError("matmul 'sparse' needs use_sparse_w.")
        if w_memory not in ('float64', 'int8'):
            raise ValueError(f"w_memory must be 'float64' or 'int8', got "
                             f"{w_memory!r}")

        self.levels       = levels
        self.max_exponent = max_exponent
        self.n_levels     = int(n_levels)
        self.p_zero       = p_zero
        self.use_sparse_w = use_sparse_w
        self.matmul       = matmul
        self.w_memory     = w_memory
        self.keep_fp      = bool(keep_fp)

        # Filled by _quantize_W(): W = _W_int * _W_scale
        self._W_int     : Optional[np.ndarray] = None  # (D, N_h) int8 / int16
        self._W_scale   : Optional[np.ndarray] = None  # (N_h,) per-col scale
        self._W_sparse  = None                          # CSC (matmul sparse)
        self._quant_stats: Optional[dict]      = None
        # Full-precision draw, kept with keep_fp (bug-fix F8, see
        # _finish_W and quantization_error).
        self._W_fp: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    @property
    def n_bits(self) -> int:
        """Deprecated alias of max_exponent."""
        return self.max_exponent

    @staticmethod
    def _pow2_levels(n: int) -> np.ndarray:
        """levels(n) = {-2^n, ..., -1, 0, 1, ..., 2^n}."""
        pos = [2**k for k in range(n + 1)]
        neg = [-2**k for k in range(n + 1)]
        return np.array(sorted(set(neg + [0] + pos)), dtype=np.float64)

    @staticmethod
    def _int_dtype(kmax: int):
        return np.int8 if kmax <= 127 else np.int16

    def _int_from_float(self, Wq: np.ndarray, scale=None):
        """(K, s) of a float quantised W (files written before int
        storage).  Without stored scales s_j = max|W[:, j]| / kmax: the
        largest entry of a column always sits on +-kmax (ternary: +-1)."""
        _, kmax = self._level_set()
        if scale is None:
            m = np.abs(Wq).max(axis=0)
            scale = np.where(m > 0, m / kmax, 1.0)
        K = np.rint(Wq / scale)
        if np.abs(K * scale - Wq).max() > 1e-9 * max(np.abs(Wq).max(), 1e-300) \
                or np.abs(K).max() > kmax:
            raise ValueError("QuantizedELMModel: stored W is not an integer "
                             "multiple of its column scales.")
        return K.astype(self._int_dtype(kmax)), np.asarray(scale, dtype=np.float64)

    def _set_int_W(self, K, scale):
        """Install W = K * s: the dense float copy (w_memory float64) and
        the CSC copy (matmul sparse)."""
        self._W_int, self._W_scale = K, scale
        self._W = (K * scale) if self.w_memory == 'float64' else None
        self._W_sparse = None
        if self.matmul == 'sparse' and self.use_sparse_w:
            from scipy.sparse import csc_matrix
            dtype = np.float32 if self.use_float32 else np.float64
            self._W_sparse = csc_matrix((K * scale).astype(dtype))

    def _w_shape(self):
        return self._W_int.shape if self._W_int is not None else self._W.shape

    def _W_eff(self) -> np.ndarray:
        """Dense float W (formed from K and s with w_memory 'int8')."""
        if self._W is not None or self._W_int is None:
            return self._W
        return self._W_int * self._W_scale

    def _level_set(self):
        """(integer levels, kmax) of the chosen level set."""
        if self.levels == 'pow2':
            return self._pow2_levels(self.max_exponent), 2 ** self.max_exponent
        if self.levels == 'ternary':
            return np.array([-1.0, 0.0, 1.0]), 1
        k = (self.n_levels - 1) // 2
        return np.arange(-k, k + 1, dtype=np.float64), k

    # ------------------------------------------------------------------
    def _quantize_W(self, W_raw: np.ndarray) -> np.ndarray:
        """
        Quantize the raw weight matrix W_raw to powers of two.

        Algorithm
        ---------
        For each column j (each hidden unit j):
          1. scale:    s_j = max|W_raw[:,j]| / kmax  (pow2: kmax = 2^n,
                       uniform: kmax = k)
          2. normalise: w_scaled = W_raw[:,j] / s_j
          3. snap:     w_q = argmin_{lambda in levels} |w_scaled - lambda|
          4. sparsify: force p_zero fraction to 0 (smallest |w_scaled|)
          5. rescale:  W_q[:,j] = w_q * s_j
        Ternary: sign of the entries kept by step 4, s_j = sqrt(sum w^2 /
        n_kept), so that the column keeps the sum of squares of the raw draw.

        Parameters
        ----------
        W_raw : (D, N_h) raw random weight matrix.

        Returns
        -------
        W_q : (D, N_h) quantized weight matrix (powers of two, sparsified).
        """
        levels, kmax = self._level_set()
        D, N_h = W_raw.shape
        W_q     = np.empty_like(W_raw)       # integer levels (float dtype)
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

            if self.levels == 'ternary':
                keep = np.ones(D, dtype=bool)
                if self.p_zero > 0.0:
                    thr = float(np.quantile(np.abs(col), self.p_zero))
                    keep = np.abs(col) > thr
                if not keep.any():
                    keep[np.argmax(np.abs(col))] = True
                s = float(np.sqrt(np.sum(col ** 2) / keep.sum()))
                scales[j] = s
                col_q = np.where(keep, np.sign(col), 0.0)
                zero_counts[j] = int(np.sum(col_q == 0))
                W_q[:, j] = col_q
                continue

            s      = max_abs / kmax
            scales[j] = s
            col_s  = col / s                        # scaled to [-kmax, kmax]

            # vectorised nearest-level snap
            idx   = np.argmin(np.abs(col_s[:, None] - levels[None, :]), axis=1)
            col_q = levels[idx].copy()              # quantized scaled values

            # p_zero: force the smallest-magnitude entries of |col_s| to 0
            if self.p_zero > 0.0:
                thr     = float(np.quantile(np.abs(col_s), self.p_zero))
                mask_z  = np.abs(col_s) <= thr
                col_q[mask_z] = 0.0

            zero_counts[j] = int(np.sum(col_q == 0))
            W_q[:, j]      = col_q                  # integer level

        K = W_q.astype(self._int_dtype(kmax))
        self._set_int_W(K, scales)
        total_elems   = D * N_h
        total_zeros   = int(np.sum(K == 0))
        self._quant_stats = {
            "level_set"    : self.levels,
            "max_exponent" : self.max_exponent,
            "p_zero"       : self.p_zero,
            "levels"       : levels.tolist(),
            "n_levels"     : len(levels),
            "sparsity_W"   : total_zeros / total_elems,
            "total_zeros"  : total_zeros,
            "total_elems"  : total_elems,
            "unique_W_vals": len(np.unique(K)),
            "W_int_dtype"  : str(K.dtype),
            "W_mem_dense_MB" : 8 * total_elems / 1e6,
            # integer levels + one float64 scale per column (what is saved)
            "W_mem_int8_MB"  : (K.nbytes + scales.nbytes) / 1e6,
        }
        if self._W_sparse is not None:
            self._quant_stats["nnz_W"] = self._W_sparse.nnz
            self._quant_stats["W_mem_csc_MB"] = (
                self._W_sparse.data.nbytes
                + self._W_sparse.indices.nbytes
                + self._W_sparse.indptr.nbytes) / 1e6
        return K * scales

    # ------------------------------------------------------------------
    def _hidden(self, X_norm: np.ndarray) -> np.ndarray:
        """Hidden matrix with the quantised W (through :meth:`_xw`)."""
        if self._W_int is None:          # before quantisation
            return super()._hidden(X_norm)
        Z = self._xw(X_norm)
        if self.activation == 'rff':
            return np.hstack([np.cos(Z), np.sin(Z)]) / np.sqrt(Z.shape[1])
        return self._act(Z + self._b)

    # ------------------------------------------------------------------
    def _finish_W(self, W_raw: np.ndarray) -> np.ndarray:
        """Quantise the raw draw of ELMModel._init_hidden (called by
        ELMModel.fit and by the per-species units of AtomicQELMModel)."""
        # Bug-fix note (F8): the original never stored W_raw anywhere, so
        # quantization_error() -- whose entire purpose is to compare
        # quantised vs full-precision predictions -- could never find a
        # full-precision reference and silently fell back to comparing
        # the quantised model against itself (always reporting exactly
        # zero error).  Store it before quantising so that
        # quantization_error() has something real to compare against.
        # (ADR-0013: only with keep_fp, to save memory; quantization_error
        # refuses without it instead of reporting zero.)
        self._W_fp = W_raw.copy() if self.keep_fp else None

        t_q0 = _time.perf_counter()
        W_q  = self._quantize_W(W_raw)
        t_q  = _time.perf_counter() - t_q0
        if self.verbose:
            qs = self._quant_stats
            print(f'  Quantization ({t_q*1000:.0f} ms):  '
                  f'levels={qs["levels"]}  sparsity={qs["sparsity_W"]:.3f}')
        return W_q if self.w_memory == 'float64' else None

    def _xw(self, X: np.ndarray) -> np.ndarray:
        if self.matmul == 'sparse' and self._W_sparse is not None:
            return np.asarray(X @ self._W_sparse, dtype=np.float64)
        if self._W is not None:
            if self.use_float32:
                return (X.astype(np.float32)
                        @ self._W.astype(np.float32)).astype(np.float64)
            return X @ self._W
        return (X @ self._W_int) * self._W_scale

    def _uwT(self, U: np.ndarray) -> np.ndarray:
        if self.matmul == 'sparse' and self._W_sparse is not None:
            return np.asarray((self._W_sparse @ U.T).T, dtype=np.float64)
        if self._W is not None:
            return U @ self._W.T
        return (U * self._W_scale) @ self._W_int.T

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

        Quantises the random hidden-layer weight matrix W (see
        :meth:`_quantize_W`), then delegates all training (weight composition,
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
        return super().fit(targets, labels, X,
                           weights      = weights,
                           weight_csv   = weight_csv,
                           val_fraction = val_fraction,
                           val_seed     = val_seed,
                           struct_ids   = struct_ids,
                           n_atoms      = n_atoms)

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
        print(f"  level set   : {qs['level_set']}  "
              f"(max_exponent {qs['max_exponent']})")
        print(f"  p_zero      : {qs['p_zero']}")
        print(f"  levels      : {qs['levels']}")
        print(f"  n_levels    : {qs['n_levels']}")
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
        print("    W_q values are integer levels times a column scale s_j.")
        if qs['level_set'] == 'pow2':
            print("    x . (2^k . s_j)  ==  s_j . (x << k)  (bit-shift product).")
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
        # (Swapping the float W now reaches every product: _xw reads
        # self._W first, and the CSC copy is switched off for the swap.)
        if self._W_fp is None:
            raise RuntimeError(
                "QuantizedELMModel.quantization_error: no full-precision "
                "reference -- fit with keep_fp=True (reloaded models have "
                "none; ADR-0013).")
        W_saved, W_sparse = self._W, self._W_sparse
        self._W, self._W_sparse = self._W_fp, None
        try:
            y_fp = self.predict(X)
        finally:
            self._W, self._W_sparse = W_saved, W_sparse

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
                 n_bits          = np.array(self.max_exponent),
                 max_exponent    = np.array(self.max_exponent),
                 levels          = np.array(self.levels),
                 n_levels        = np.array(self.n_levels),
                 matmul          = np.array(self.matmul),
                 p_zero          = np.array(self.p_zero),
                 use_sparse_w    = np.array(self.use_sparse_w),
                 w_memory        = np.array(self.w_memory),
                 # W = W_int * W_scale (ADR-0013); no float W is stored
                 W_int           = self._W_int,
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
            max_exponent     = int(npz['max_exponent'] if 'max_exponent' in npz
                                   else npz['n_bits']),
            levels           = str(npz['levels']) if 'levels' in npz else 'pow2',
            n_levels         = int(npz['n_levels']) if 'n_levels' in npz else 3,
            matmul           = str(npz['matmul']) if 'matmul' in npz else 'dense',
            p_zero           = float(npz['p_zero']),
            use_sparse_w     = bool(npz.get('use_sparse_w', True)),
            w_memory         = (str(npz['w_memory']) if 'w_memory' in npz
                                else 'int8'),
            verbose          = verbose,
        )
        b_arr          = npz['b'] if 'b' in npz else np.array([])
        obj._b         = b_arr if b_arr.size > 0 else None
        obj._beta      = npz['beta']
        obj._X_mean    = npz['X_mean']
        obj._X_std     = npz['X_std']
        obj._D         = obj._X_mean.shape[0]
        if 'W_int' in npz:
            obj._set_int_W(npz['W_int'], npz['W_scale'])
        else:  # files written before integer storage: float W_q_dense
            obj._set_int_W(*obj._int_from_float(npz['W_q_dense'],
                                                npz['W_scale']))
        obj._restore_extra_state(npz)
        obj._fitted    = True
        # A reloaded model has no full-precision reference (never saved).
        obj._W_fp      = None
        return obj
