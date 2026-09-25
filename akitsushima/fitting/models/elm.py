"""Physics-Informed Extreme Learning Machine (PI-ELM) that reads
pre-computed descriptors and DFT reference values from a single CSV file.

Descriptor computation is performed EXTERNALLY (e.g., DScribe, RDKit, or
any other tool) and the results are stored in the CSV.  This module is
responsible only for training (ELM) and prediction.

Matrix-free LSQR algorithm
--------------------------
The hidden-layer matrix H_tr in R^{N_eq x N_h} is NEVER formed explicitly.
A scipy LinearOperator wrapping matvec/rmatvec replaces it so that LSQR
only needs O(N_h) working memory regardless of N_eq.

References
----------
  ELM : Huang et al., Neurocomputing 70, 489 (2006)
  LSQR: Paige & Saunders, ACM TOMS 8, 43 (1982)
"""
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .csv_io import (
    build_structure_index,
    load_mlip_csv,
    load_weights_csv,
    structure_grouped_split,
)

__all__ = ["ELMModel"]


class ELMModel:
    """
    Physics-Informed Extreme Learning Machine trained from CSV descriptor data.

    The model is completely agnostic about how the descriptors were generated.
    It reads pre-computed (target, label, weight, features) rows from a CSV
    file and solves the PI-ELM linear system via matrix-free LSQR.

    Architecture
    ------------
    Input  : descriptor vector x in R^D
    Hidden : h(x) = sigma(x W + b)  in R^{N_h}
             W in R^{D x N_h}, b in R^{N_h}  -- fixed random (ELM paradigm)
    Output : E/N = f(d) = h(d) . beta + beta0   -- learned by LSQR

    Every F and S row is trained as N grad f(d) . X with d the descriptor
    of the row's structure (chain rule, ADR-0004); see :meth:`fit`.
    ``row_independent=True`` restores the pre-0.2.0a3 form y = h(x) . beta
    applied to each row alone.

    Weighting
    ---------
    Each row's squared residual is weighted by w_cat * w_unit^2 * w_user
    (category weight w_energy / w_forces / w_stress, CSV col 2, optional
    weights.csv); see :meth:`fit`.

    CSV format  (4-column layout)
    ------------------------------
    Row 0    : header  --  labels for every column
    Row 1+   : one sample per row

    Column layout (fixed):
      col 0  : target value  (float) -- E [eV/atom], F_a [eV/A], sigma [eV/A^3]
      col 1  : label         (str)   -- "E", "F", or "S"
      col 2  : weight        (float) -- unit-normalisation weight (see below)
      col 3+ : descriptor features  (float)

    Correct weight values (col 2) that bring all residuals to eV^2:
      E row  : weight = N                (number of atoms)
      F row  : weight = r0  [A]          (typical atomic displacement)
      S row  : weight = eps0 * V  [A^3]  (typical strain * cell volume)

    Typical usage::

        targets, labels, weights, X = load_mlip_csv("structures.csv")
        model = ELMModel(n_hidden=512, r0=0.1, eps0=0.01)
        model.fit(targets, labels, X, weights=weights)

    With user-defined block weights (weights.csv)::

        model.fit(targets, labels, X, weights=weights,
                  weight_csv="weights.csv")

    Via the convenience wrapper::

        model.fit_csv("structures.csv", weight_csv="weights.csv")

    Two LSQR modes  (controlled by full_matrix_free)
    -------------------------------------------------
    full_matrix_free=False  (default -- partial matrix-free):
        Compute H = sigma(X W + b) once and keep it in RAM.
        Each LSQR matvec/rmatvec is an O(N x N_h) matrix product.
        Memory O(N x N_h).  Example: N=50k, N_h=1k -> H ~ 400 MB.

    full_matrix_free=True  (fully matrix-free):
        Never store H.  Recompute X mini-batches on the fly per iteration.
        Memory O(chunk x N_h) only.  Essential for very large N or N_h.

    Parameters
    ----------
    n_hidden       : number of hidden neurons.
    activation     : 'tanh' | 'relu' | 'sigmoid' | 'rff'.
    w_energy       : category weight applied to 'E' rows.
    w_forces       : category weight applied to 'F' rows.
    w_stress       : category weight applied to 'S' rows.
    r0, eps0       : recorded with the model for reference only; the unit
                     weights themselves come from CSV col 2.
    reg            : Tikhonov regularisation lambda (LSQR damp = sqrt(lambda)).
    lsqr_maxiter   : maximum LSQR iterations.
    lsqr_atol      : LSQR convergence tolerance.
    seed           : RNG seed for the random hidden layer.
    full_matrix_free : False = store H in RAM; True = recompute per iteration.
    chunk_size     : rows per mini-batch in full matrix-free mode.
    use_float32    : compute H in float32 to halve memory and bandwidth.
    verbose        : print progress and diagnostics.
    """

    # Category -> weight mapping
    _CAT_WEIGHT_KEY = {'E': 'w_energy', 'F': 'w_forces', 'S': 'w_stress'}

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
                 verbose         : bool  = True,
                 masses          : Optional[np.ndarray] = None,
                 row_independent : bool  = False,
                 solver          : str   = 'auto'):
        """
        Parameters
        ----------
        n_hidden   : int
            Number of hidden units.  Larger values improve accuracy but
            increase memory and LSQR iteration count.
        activation : str
            Hidden-layer activation: ``'tanh'`` (default), ``'relu'``,
            ``'elu'``, ``'snake'``, ``'rff'`` (Random Fourier Features).
        w_energy   : float
            Category weight applied to E rows in the loss function.
        w_forces   : float
            Category weight applied to F rows in the loss function.
        w_stress   : float
            Category weight applied to S rows in the loss function.
        r0, eps0   : float
            Typical displacement [A] and strain used to build the CSV's
            unit weights (F row: r0, S row: eps0 * V).  Kept with the
            model for reference; training reads the weights from CSV
            col 2 and does not rescale by these again (the former
            Uncertainty Weighting did, see ADR-0003 M1).
        reg        : float
            Ridge (L2) regularisation coefficient lambda for LSQR.
        lsqr_maxiter : int
            Maximum LSQR iterations.
        lsqr_atol  : float
            LSQR convergence tolerance.
        seed       : int
            Random seed for W and b initialisation.
        full_matrix_free : bool
            When True, use a fully matrix-free LinearOperator (on-the-fly
            H computation, lower memory).  When False, cache the hidden
            matrix H (faster LSQR, higher memory).
        use_float32 : bool
            Cast W to float32 before the matrix product (reduces memory
            bandwidth by ~2*; accuracy loss < 1e-6).
        w_init_scale : float or None
            Standard deviation sigma for W and b initialisation:
                W ~ N(0, sigma^2),  b ~ N(0, sigma^2)
            None (default) -> Glorot scale sqrt(2 / (D + N_h)).
            Larger sigma pushes tanh into saturation, making columns of H
            more orthogonal and reducing LSQR iteration count:
                None    -> Glorot  (~100 iterations)
                sigma = 0.5 -> medium  (~50 iterations)
                sigma = 1.0 -> upper recommended limit (~45 iterations)
                sigma = 2.0 -> near-binary H (~44 iterations or fewer)
            Ignored in RFF mode (rff_gamma plays the same role).
        verbose    : bool
            Print training progress and score table.
        row_independent : bool
            False (default): chain-rule training, a force field.
            True: pre-0.2.0a3 per-row regression (not a force field).
        solver     : str
            ``'direct'``: normal equations (A^T A + reg I) beta = A^T t,
            accumulated chunk by chunk and solved by Cholesky -- the same
            minimiser as LSQR with damp = sqrt(reg), in one pass.
            ``'lsqr'``: iterative LSQR (the pre-0.2.0a3 solver).
            ``'auto'`` (default): direct when there are at most 6000
            output weights, LSQR otherwise.  The chain-rule system mixes
            rows of very different scale; LSQR then often stops at its
            iteration limit (flag 7) short of the minimum.
        """
        from scipy.sparse.linalg import lsqr as _lsqr, LinearOperator as _LO
        self._lsqr = _lsqr
        self._LO   = _LO

        self.n_hidden         = n_hidden
        self.activation       = activation.lower()
        self.w_energy         = w_energy
        self.w_forces         = w_forces
        self.w_stress         = w_stress
        self.r0               = float(r0)
        self.eps0             = float(eps0)
        self.reg              = reg
        self.lsqr_maxiter     = lsqr_maxiter
        self.lsqr_atol        = lsqr_atol
        self.seed             = seed
        self.full_matrix_free = full_matrix_free
        self.chunk_size       = chunk_size
        self.use_float32      = use_float32
        self.snake_a          = float(snake_a)
        self.rff_gamma        = rff_gamma
        self.w_init_scale     = w_init_scale   # None -> Glorot at fit() time
        self.verbose          = verbose
        self.row_independent  = bool(row_independent)
        if solver not in ('auto', 'direct', 'lsqr'):
            raise ValueError(
                f"ELMModel: solver must be 'auto', 'direct' or 'lsqr', "
                f"got {solver!r}.")
        self.solver           = solver
        # Descriptor the model was trained on, when generated in-package
        # (JSON string; empty for external CSV descriptors).  Needed to
        # run the model as a relax/md engine (ADR-0004, decision 6).
        self.descriptor_config = ""

        # Set by fit()
        self._W    : Optional[np.ndarray] = None
        self._b    : Optional[np.ndarray] = None
        self._beta : Optional[np.ndarray] = None
        self._beta0: float                = 0.0
        self._D    : Optional[int]        = None
        self._fitted = False

        # Activation function and derivative
        self._act, self._dact = self._build_activation(self.activation)

        # Snake: override static placeholder with closure that captures snake_a
        if self.activation == 'snake':
            a = self.snake_a
            self._act  = lambda z, _a=a: z + np.sin(_a * z)**2 / _a
            self._dact = lambda z, _a=a: 1.0 + np.sin(2.0 * _a * z)

        # Normalisation stats (set by fit)
        self._X_mean : Optional[np.ndarray] = None
        self._X_std  : Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _build_activation(name: str):
        """
        Build (act_fn, dact_fn) pair for the named activation.

        Supported activations
        ---------------------
        tanh     : sigma(z) = tanh(z)
        relu     : sigma(z) = max(0, z)
        sigmoid  : sigma(z) = 1/(1+e^{-z})
        silu     : sigma(z) = z / (1+e^{-z})   [= Swish]
        mish     : sigma(z) = z . tanh(softplus(z))
        gelu     : sigma(z) = z . Phi(z)  (Phi: Gaussian CDF, tanh approx)
        softplus : sigma(z) = ln(1+e^z)
        snake    : sigma(z) = z + sin^2(az)/a   [a = snake_a parameter]
        rff      : phi(x) = [cos(Wx), sin(Wx)] / sqrt(N_omega)
                   Random Fourier Features (RBF kernel approximation)
                   stacks sin and cos of omega-projected inputs
                   output dim N_h = 2 N_omega
                   handled internally by _hidden()

        Returns: (act_fn, dact_fn)
          act_fn  : (N,D)->(N,N_h), or elementwise (z,)->(z,)
          dact_fn : derivative used by the PI-ELM force computation
        """
        n = name.lower()

        if n == 'tanh':
            return (np.tanh,
                    lambda z: 1.0 - np.tanh(z)**2)

        elif n == 'relu':
            return (lambda z: np.maximum(z, 0.0),
                    lambda z: (z > 0.0).astype(float))

        elif n == 'sigmoid':
            def _sig(z):
                return np.where(z >= 0,
                                1.0 / (1.0 + np.exp(-z)),
                                np.exp(z) / (1.0 + np.exp(z)))
            return (_sig, lambda z: _sig(z) * (1.0 - _sig(z)))

        elif n in ('silu', 'swish'):
            # SiLU(z) = z sigma(z)    sigma = sigmoid
            def _silu(z):
                s = np.where(z >= 0, 1.0/(1.0+np.exp(-z)),
                             np.exp(z)/(1.0+np.exp(z)))
                return z * s
            def _dsilu(z):
                s  = np.where(z >= 0, 1.0/(1.0+np.exp(-z)),
                              np.exp(z)/(1.0+np.exp(z)))
                return s + z * s * (1.0 - s)
            return (_silu, _dsilu)

        elif n == 'mish':
            # Mish(z) = z . tanh(ln(1+e^z))
            # clip to avoid overflow in exp; tanh(ln(1+e^z)) -> 1 for z->+inf
            def _mish(z):
                sp = np.log1p(np.exp(np.clip(z, -500.0, 20.0)))
                return z * np.tanh(sp)
            def _dmish(z):
                zc  = np.clip(z, -500.0, 20.0)
                ex  = np.exp(zc)
                ex2 = np.exp(2.0 * zc)
                ex3 = np.exp(3.0 * zc)
                num = ex * (4.0*(z+1.0) + 4.0*ex2 + ex3
                            + ex*(4.0*z + 6.0))
                den = (2.0*ex + ex2 + 2.0)**2
                return num / (den + 1e-30)
            return (_mish, _dmish)

        elif n == 'gelu':
            # GELU(z) ~= 0.5z(1 + tanh(sqrt(2/pi)(z + 0.044715z^3)))
            _c1 = np.sqrt(2.0 / np.pi)
            _c2 = 0.044715
            def _gelu(z):
                inner = _c1 * (z + _c2 * z**3)
                return 0.5 * z * (1.0 + np.tanh(inner))
            def _dgelu(z):
                inner = _c1 * (z + _c2 * z**3)
                t     = np.tanh(inner)
                dt    = (1.0 - t**2) * _c1 * (1.0 + 3.0*_c2*z**2)
                return 0.5 * (1.0 + t) + 0.5 * z * dt
            return (_gelu, _dgelu)

        elif n == 'softplus':
            # Softplus(z) = ln(1+e^z),  beta=1
            def _sp(z):
                return np.log1p(np.exp(np.clip(z, -500.0, 20.0)))
            def _dsp(z):
                return np.where(z >= 0,
                                1.0 / (1.0 + np.exp(-z)),
                                np.exp(z) / (1.0 + np.exp(z)))
            return (_sp, _dsp)

        elif n == 'snake':
            # Snake(z;a) = z + sin^2(az)/a   Liu et al., ICLR 2021
            # 'snake' uses snake_a=1.0 (set in __init__);
            # access via the closure to allow per-instance a
            # NOTE: _build_activation is @staticmethod so 'a' must be
            #       passed through a different path. We return a placeholder
            #       that will be overwritten by __init__ after calling this.
            def _snake_default(z, a=1.0):
                return z + np.sin(a * z)**2 / a
            def _dsnake_default(z, a=1.0):
                return 1.0 + np.sin(2.0 * a * z)
            return (_snake_default, _dsnake_default)

        elif n == 'rff':
            # Random Fourier Features: handled specially in _hidden()
            # Return identity placeholders; actual computation is in _hidden.
            return (lambda z: z, lambda z: np.ones_like(z))

        else:
            raise ValueError(
                f"ELMModel: unknown activation '{name}'. "
                f"Choose from: tanh, relu, sigmoid, silu, mish, gelu, "
                f"softplus, snake, rff.")

    # ------------------------------------------------------------------
    def _hidden(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Forward pass: (N, D) -> (N, N_h).

        float32 (use_float32=True)
        ----------------------------------------
        When use_float32 is set, X_norm, W, b are cast to float32 for the
        matrix product, then the result is returned (LSQR still runs in
        float64).  This halves memory bandwidth, typically giving a 2 to 5x
        speedup with negligible accuracy loss (max error ~= 7x10^-7).

        RFF mode (activation='rff')
        --------------------------------
        W has shape (D, N_omega); b is not used.
        phi(x) = [cos(WX), sin(WX)] / sqrt(N_omega)  -> (N, 2N_omega)
        """
        if self.activation == 'rff':
            if self.use_float32:
                Z = (X_norm.astype(np.float32)
                     @ self._W.astype(np.float32)).astype(np.float64)
            else:
                Z = X_norm @ self._W
            N_omega = self._W.shape[1]
            return np.hstack([np.cos(Z), np.sin(Z)]) / np.sqrt(N_omega)
        # ---- standard activations ----
        if self.use_float32:
            Z = (X_norm.astype(np.float32)
                 @ self._W.astype(np.float32)
                 + self._b.astype(np.float32)).astype(np.float64)
        else:
            Z = X_norm @ self._W + self._b
        return self._act(Z)

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        """Z-score normalise using statistics from fit()."""
        return (X - self._X_mean) / self._X_std

    # ------------------------------------------------------------------
    #  LinearOperator builders
    # ------------------------------------------------------------------

    def _build_operator_cached(self,
                                X_tr_n : np.ndarray,
                                w_tr   : np.ndarray,
                                ) -> Tuple[object, np.ndarray]:
        """
        Build the cached LinearOperator (full_matrix_free=False).

        H = sigma(X_tr_n W + b) is computed once and stored in RAM.
        LSQR then uses H_w (weight-scaled H) at each iteration.

        use_float32=True effect:
          H is stored as float32 to reduce RAM,
          matvec / rmatvec use the float32 H_w then cast to float64,
          so LSQR still operates in float64.

        RAM: O(N_tr x N_h)  (float32 halves it)
        matmul cost per iteration: O(N_tr x N_h)
        """
        N_tr, N_h = len(w_tr), self.n_hidden if self.activation != 'rff' \
                    else self._W.shape[1] * 2

        H_tr = self._hidden(X_tr_n)      # (N_tr, N_h)  always float64 here

        # float32 storage -> matvec cast back to float64
        if self.use_float32:
            H_tr = H_tr.astype(np.float32)

        H_w = H_tr * w_tr[:, None].astype(H_tr.dtype)

        if self.verbose:
            mem_MB  = H_tr.nbytes / 1e6
            peak_GB = mem_MB / 1e3
            dtype_s = "float32" if self.use_float32 else "float64"
            print(f"  [cached MF] H stored: "
                  f"({N_tr}, {N_h})  {mem_MB:.1f} MB  [{dtype_s}]  "
                  f"(peak ~= {peak_GB:.2f} GB)")

        # matvec / rmatvec run in float64 so that LSQR stays float64.
        def matvec(x: np.ndarray) -> np.ndarray:
            return (H_w @ x.astype(H_w.dtype)).astype(np.float64)

        def rmatvec(y: np.ndarray) -> np.ndarray:
            return (H_w.T @ y.astype(H_w.dtype)).astype(np.float64)

        A = self._LO((N_tr, N_h),
                     matvec=matvec, rmatvec=rmatvec, dtype=float)
        A.row_blocks = lambda: iter([(slice(0, N_tr), H_w)])
        return A

    def _build_operator_full_mf(self,
                                 X_tr_n : np.ndarray,
                                 w_tr   : np.ndarray,
                                 ) -> Tuple[object, np.ndarray]:
        """
        Build the fully matrix-free LinearOperator (full_matrix_free=True).

        H is never stored in RAM; each LSQR matvec / rmatvec
        recomputes H from X_tr_n on the fly.

        RAM: O(chunk x N_h)
        cost: O(N_tr x D x N_h / chunk + N_tr x N_h)
              ~= O(N_tr x N_h)  (when D << N_h)

        matvec: A.x = sum_chunk H_chunk_w . x
        ---------------------------------------------------
        for each chunk [s, s+chunk):
            Z_chunk    = X_tr_n[s:s+chunk] @ W + b      (chunk, N_h)
            H_chunk    = sigma(Z_chunk)                      (chunk, N_h)
            H_chunk_w  = H_chunk * w[s:s+chunk, None]   (chunk, N_h)
            y[s:s+c] = H_chunk_w @ x                    (chunk,)

        rmatvec: A^T.y = sum_chunk  H_chunk_w^T . y[s:s+c]
        ---------------------------------------------------
        for each chunk [s, s+chunk):
            <same H_chunk_w computation>
            out += H_chunk_w.T @ y[s:s+c]               (N_h,)

        Returns
        -------
        A     : LinearOperator  (N_tr, N_h)
        """
        N_tr  = len(w_tr)
        N_h   = self.n_hidden
        chunk = self.chunk_size
        W     = self._W        # (D, N_h)  ---  captured by closure
        b     = self._b        # (N_h,)

        # Compute peak memory per chunk for diagnostics
        chunk_act = min(chunk, N_tr)
        chunk_MB  = chunk_act * N_h * 8 / 1e6
        total_MB  = N_tr * N_h * 8 / 1e6

        if self.verbose:
            print(f"  [full MF] H recomputed on-the-fly: "
                  f"chunk={chunk}  "
                  f"peak_RAM ~= {chunk_MB:.1f} MB/iter  "
                  f"(vs cached: {total_MB:.0f} MB)")

        act = self._act    # activation function

        def _h_chunk(X_chunk: np.ndarray,
                     w_chunk: np.ndarray) -> np.ndarray:
            """
            Compute one weighted mini-batch of the hidden matrix.
            Honours use_float32 / activation='rff' via _hidden().
            """
            H_c = self._hidden(X_chunk)                    # float64
            return H_c * w_chunk[:, None]                  # (chunk, N_h)

        def matvec(x: np.ndarray) -> np.ndarray:
            """A . x  =  sum_chunk  H_w_chunk . x"""
            out = np.empty(N_tr, dtype=np.float64)
            for s in range(0, N_tr, chunk):
                e     = min(s + chunk, N_tr)
                H_w_c = _h_chunk(X_tr_n[s:e], w_tr[s:e])
                out[s:e] = H_w_c @ x
            return out

        def rmatvec(y: np.ndarray) -> np.ndarray:
            """A^T . y  =  sum_chunk  H_w_chunk^T . y[s:e]"""
            out = np.zeros(N_h, dtype=np.float64)
            for s in range(0, N_tr, chunk):
                e     = min(s + chunk, N_tr)
                H_w_c = _h_chunk(X_tr_n[s:e], w_tr[s:e])
                out  += H_w_c.T @ y[s:e]
            return out

        A = self._LO((N_tr, N_h),
                     matvec=matvec, rmatvec=rmatvec, dtype=float)
        A.row_blocks = lambda: ((slice(s, min(s + chunk, N_tr)),
                                 _h_chunk(X_tr_n[s:s + chunk],
                                          w_tr[s:s + chunk]))
                                for s in range(0, N_tr, chunk))
        return A

    # ------------------------------------------------------------------
    #  Weighting (ADR-0004, decision 2)
    # ------------------------------------------------------------------
    def _row_weights(self, labels, w_unit, w_user) -> np.ndarray:
        """Weight on each row's SQUARED residual: w_cat * w_unit^2 * w_user."""
        _wmap = {'E': self.w_energy, 'F': self.w_forces, 'S': self.w_stress}
        w_cat = np.array([_wmap.get(c, 1.0) for c in labels], dtype=np.float64)
        return w_cat * w_unit ** 2 * w_user

    # ------------------------------------------------------------------
    #  Chain-rule design rows (ADR-0004, decision 1)
    # ------------------------------------------------------------------
    def _W_eff(self) -> np.ndarray:
        """Dense hidden-layer weights actually used by _hidden()."""
        return self._W

    def _n_out(self) -> int:
        """Number of output weights (hidden units, plus the E-row bias
        in chain-rule mode)."""
        n_h = (2 * self._W.shape[1] if self.activation == 'rff'
               else self._W.shape[1])
        return n_h if self.row_independent else n_h + 1

    def _preact(self, X_norm: np.ndarray) -> np.ndarray:
        """Z = X_norm W (+ b), float64, with the effective weights."""
        Z = X_norm @ self._W_eff()
        if self.activation != 'rff':
            Z = Z + self._b
        return Z

    def _hidden_from_Z(self, Z: np.ndarray) -> np.ndarray:
        if self.activation == 'rff':
            return np.hstack([np.cos(Z), np.sin(Z)]) / np.sqrt(Z.shape[1])
        return self._act(Z)

    def _hidden_dir(self, Z: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Directional derivative of the hidden layer: dh = h'(Z) (*) V,
        where V = (direction / X_std) W is the pre-activation change."""
        if self.activation == 'rff':
            s = 1.0 / np.sqrt(Z.shape[1])
            return np.hstack([-np.sin(Z) * V, np.cos(Z) * V]) * s
        return self._dact(Z) * V

    def _chain_rows(self, X_rows, lab_rows, sid_rows, Zs, n_at) -> np.ndarray:
        """
        Design matrix of the given rows, one column per output weight
        (hidden units, then the E-row bias):

            E row : [ h(z_s),                         1 ]
            F/S   : [ N_s h'(z_s) (*) ((X / X_std) W), 0 ]

        z_s is the pre-activation of the row's structure descriptor d_s.
        An F row's X is -dd/dr and an S row's X is -(1/V) dd/dT, so these
        rows are exactly N grad f(d) . X for f(d) = beta . h(d) + beta0.
        """
        n = len(lab_rows)
        A = np.zeros((n, self._n_out()), dtype=np.float64)
        isE = lab_rows == 'E'
        if isE.any():
            A[isE, :-1] = self._hidden_from_Z(Zs[sid_rows[isE]])
            A[isE, -1] = 1.0
        isD = ~isE
        if isD.any():
            V = (X_rows[isD] / self._X_std) @ self._W_eff()
            A[isD, :-1] = (self._hidden_dir(Zs[sid_rows[isD]], V)
                           * n_at[sid_rows[isD]][:, None])
        return A

    def _structure_Z(self, X, e_row) -> np.ndarray:
        """Pre-activations of every structure's descriptor (its E row)."""
        return self._preact(self._normalise(X[e_row]))

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
            ) -> 'ELMModel':
        r"""
        Fit the ELM output weights by weighted linear least squares.

        Weight on each row's squared residual (ADR-0004, decision 2)::

            w_i = w_cat_i  x  w_unit_i^2  x  w_user_i

        (1) w_cat  : category weight (w_energy / w_forces / w_stress).
        (2) w_unit : unit-normalisation weight from CSV col 2, which makes
                     every squared residual eV^2:
                       E: (N * dy [eV/atom])^2,  F: (r0 * dF)^2,
                       S: (eps0 * V * dsigma)^2.
        (3) w_user : block-wise user weight from ``weight_csv``.

        Each row of the least-squares system is multiplied by sqrt(w_i).
        (The former Uncertainty Weighting, which applied 1/sigma^2 to the
        residual and squared w_unit a second time, is removed: ADR-0003
        finding M1.)

        Model form
        ----------
        Default (chain rule, ADR-0004 decision 1): E/N = f(d) =
        beta . h(d) + beta0, and each F/S row is N grad f(d) . X with d
        the descriptor of the row's structure -- see :meth:`_chain_rows`
        and :func:`~akitsushima.fitting.models.csv_io.load_mlip_csv` for
        the descriptor convention.  All rows are linear in (beta, beta0),
        so E, F and S are solved together in closed form.

        ``row_independent=True`` (constructor): the pre-0.2.0a3 form
        y = beta . h(x) applied to every row by itself.  It is not a
        force field and exists to reproduce old fits and for comparison.

        Parameters
        ----------
        targets      : (N,) float  target values (eV/atom, eV/A, eV/A^3).
        labels       : (N,) str    'E', 'F', or 'S' per row.
        X            : (N, D)      descriptor feature matrix.
        weights      : (N,) float or None
            Unit-normalisation weights from load_mlip_csv col 2.
            None -> all 1.0 (no unit normalisation).
        weight_csv   : str or None
            Path to user weight file (weights.csv).  None -> all 1.0.
        val_fraction : float
            Fraction of **structures** (not rows) held out for validation;
            the split is grouped by structure (F17, see
            :func:`~akitsushima.fitting.models.csv_io.structure_grouped_split`).
        val_seed     : int    RNG seed for the train/val split.
        struct_ids   : (N,) int or None -- explicit structure id per row
            (CSV ``struct_id`` column); None opens a structure at each E row.
        n_atoms      : (N,) int or None -- CSV ``n_atoms`` column.

        Returns
        -------
        self
        """
        import time as _time

        labels = np.array([str(c).strip().upper() for c in labels])
        N, D = X.shape

        if weights is not None:
            w_unit = np.asarray(weights, dtype=np.float64)
            if w_unit.shape != (N,):
                raise ValueError(
                    f"ELMModel.fit: weights.shape={w_unit.shape} "
                    f"must be ({N},).")
            if np.any(w_unit < 0):
                raise ValueError(
                    "ELMModel.fit: weights contains negative values.")
        else:
            w_unit = np.ones(N, dtype=np.float64)

        if weight_csv is not None:
            w_user = load_weights_csv(weight_csv, n_total=N,
                                       verbose=self.verbose)
        else:
            w_user = np.ones(N, dtype=np.float64)

        w_row = self._row_weights(labels, w_unit, w_user)

        if not self.row_independent:
            sid, e_row, n_at = build_structure_index(
                labels, struct_ids, n_atoms, caller='ELMModel.fit')

        mode_str = ('full MF' if self.full_matrix_free else 'cached H')
        form_str = ('row-independent' if self.row_independent
                    else 'chain rule')
        if self.verbose:
            print(f'\nELM fitting  [{mode_str}, {form_str}]')
            print(f'  Samples      : {N}  (D={D}  N_h={self.n_hidden})')
            print(f'  Category wts : wE={self.w_energy}  '
                  f'wF={self.w_forces}  wS={self.w_stress}')
            print(f'  reg={self.reg}  activation={self.activation}')

        # Train / val split, grouped by structure (F17): whole structures
        # go to one side, since the 1 E + 3N F + 6 S rows of a structure
        # all derive from one configuration.
        idx_train, idx_val = structure_grouped_split(
            labels, val_fraction=val_fraction, val_seed=val_seed,
            caller='ELMModel.fit', struct_ids=struct_ids)
        n_val = int(idx_val.size)

        if self.verbose:
            print(f'  Train/val split: train={len(idx_train)}  val={n_val}')

        # Descriptor normalisation.  In chain-rule mode the statistics are
        # those of the structure descriptors d (E rows); F/S rows hold
        # derivatives of d, which live in a different space.
        if self.row_independent:
            X_stat = X[idx_train]
        else:
            tr_struct = np.unique(sid[idx_train])
            X_stat = X[e_row[tr_struct]]
        self._X_mean = X_stat.mean(axis=0)
        self._X_std  = X_stat.std(axis=0)
        self._X_std[self._X_std < 1e-12] = 1.0
        self._D = D

        # Initialise random hidden layer (fixed)
        # When seed is None, W and b have been pre-initialised by the
        # calling class (e.g. QuantizedELMModel), so skip re-init.
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            if self.activation == 'rff':
                N_omega = max(1, self.n_hidden // 2)
                gamma   = (self.rff_gamma if self.rff_gamma is not None
                           else float(np.sqrt(D / 2.0)))
                self._W    = rng.standard_normal((D, N_omega)) / gamma
                self._b    = None
            else:
                if self.w_init_scale is not None:
                    scale = float(self.w_init_scale)
                else:
                    scale = float(np.sqrt(2.0 / (D + self.n_hidden)))
                self._W    = rng.standard_normal((D, self.n_hidden)) * scale
                self._b    = rng.standard_normal(self.n_hidden) * scale

        mult_tr = np.sqrt(w_row[idx_train])
        t_vec   = mult_tr * targets[idx_train]
        N_tr    = len(idx_train)
        N_out   = self._n_out()

        if self.verbose:
            print('  Row weights w_cat * w_unit^2 * w_user (training set):')
            for c in ('E', 'F', 'S'):
                mask = labels[idx_train] == c
                if mask.any():
                    wc = w_row[idx_train][mask]
                    print(f'    [{c}]  mean={wc.mean():.4e}  '
                          f'min={wc.min():.4e}  max={wc.max():.4e}')
            print(f'  Building LinearOperator (N_eq={N_tr}  N_out={N_out}) ...')

        t_build0 = _time.perf_counter()
        if self.row_independent:
            X_tr_n = self._normalise(X[idx_train])
            if self.full_matrix_free:
                A = self._build_operator_full_mf(X_tr_n, mult_tr)
            else:
                A = self._build_operator_cached(X_tr_n, mult_tr)
        else:
            Zs = self._structure_Z(X, e_row)
            rows = (X[idx_train], labels[idx_train], sid[idx_train])
            A = self._build_chain_operator(rows, Zs, n_at, mult_tr)
        t_build = _time.perf_counter() - t_build0

        if self.verbose:
            print(f'  Operator ready in {t_build*1000:.0f} ms')
            print(f'  Running LSQR (maxiter={self.lsqr_maxiter}) ...')

        use_direct = (self.solver == 'direct'
                      or (self.solver == 'auto' and N_out <= 6000))
        t_lsqr0 = _time.perf_counter()
        if use_direct:
            result = self._solve_direct(A, t_vec)
        else:
            result = self._lsqr(A, t_vec,
                                damp     = np.sqrt(self.reg),
                                iter_lim = self.lsqr_maxiter,
                                atol     = self.lsqr_atol,
                                btol     = self.lsqr_atol,
                                show     = False)
        t_lsqr  = _time.perf_counter() - t_lsqr0

        coef = result[0]
        if self.row_independent:
            self._beta, self._beta0 = coef, 0.0
        else:
            self._beta, self._beta0 = coef[:-1], float(coef[-1])
        flag, itn, rnorm = result[1], result[2], result[3]
        self._fitted = True

        self._solver_info = {'solver': 'direct' if use_direct else 'lsqr',
                             'flag': int(flag), 'iter': int(itn),
                             'rnorm': float(rnorm), 'time_s': t_lsqr}
        if self.verbose:
            if use_direct:
                print(f'  Direct solve: r={rnorm:.4e}  total={t_lsqr:.2f}s')
            else:
                t_ms = t_lsqr / itn * 1000 if itn > 0 else 0.0
                print(f'  LSQR: flag={flag}  iter={itn}  r={rnorm:.4e}  '
                      f'total={t_lsqr:.1f}s  ({t_ms:.1f} ms/iter)')

        def _sub(idx):
            return dict(labels=labels[idx], X=X[idx],
                        struct_ids=(None if struct_ids is None
                                    else np.asarray(struct_ids)[idx]),
                        n_atoms=(None if n_atoms is None
                                 else np.asarray(n_atoms)[idx]))

        train_sc = self._score(targets[idx_train],
                               self.predict_table(**_sub(idx_train)),
                               labels[idx_train], 'train')
        val_sc   = (self._score(targets[idx_val],
                                self.predict_table(**_sub(idx_val)),
                                labels[idx_val], 'val')
                    if n_val else None)
        self._train_sc = train_sc
        self._val_sc   = val_sc
        if self.verbose:
            self._print_scores(train_sc, val_sc)

        return self

    def _solve_direct(self, A, t_vec):
        """
        Minimise ||A beta - t||^2 + reg ||beta||^2 through the normal
        equations G = A^T A, accumulated over the operator's row blocks
        (one block in cached mode; chunks in full matrix-free mode).
        Returns an LSQR-shaped tuple (beta, flag, iter, rnorm).
        """
        import scipy.linalg as _sla
        n, m = A.shape
        G = np.zeros((m, m))
        rhs = np.zeros(m)
        for sl, B in A.row_blocks():
            B = np.asarray(B, dtype=np.float64)
            G += B.T @ B
            rhs += B.T @ t_vec[sl]
        G[np.diag_indices(m)] += self.reg
        try:
            beta = _sla.cho_solve(_sla.cho_factor(G, lower=True), rhs)
        except np.linalg.LinAlgError:
            beta = _sla.lstsq(G, rhs)[0]
        rnorm = float(np.linalg.norm(A.matvec(beta) - t_vec))
        return beta, 0, 0, rnorm

    def _build_chain_operator(self, rows, Zs, n_at, mult):
        """LinearOperator of the weighted chain-rule design matrix.
        Cached mode builds it once; full matrix-free mode rebuilds it
        chunk by chunk on every product."""
        X_r, lab_r, sid_r = rows
        n, n_out = len(lab_r), self._n_out()
        if not self.full_matrix_free:
            A = self._chain_rows(X_r, lab_r, sid_r, Zs, n_at) * mult[:, None]
            if self.use_float32:
                A = A.astype(np.float32)
            if self.verbose:
                print(f'  [cached] design matrix ({n}, {n_out})  '
                      f'{A.nbytes / 1e6:.1f} MB')
            op = self._LO(
                (n, n_out),
                matvec=lambda x: (A @ x.astype(A.dtype)).astype(np.float64),
                rmatvec=lambda y: (A.T @ y.astype(A.dtype)).astype(np.float64),
                dtype=float)
            op.row_blocks = lambda: iter([(slice(0, n), A)])
            return op

        chunk = self.chunk_size

        def _block(s, e):
            return (self._chain_rows(X_r[s:e], lab_r[s:e], sid_r[s:e], Zs, n_at)
                    * mult[s:e, None])

        def matvec(x):
            out = np.empty(n)
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                out[s:e] = _block(s, e) @ x
            return out

        def rmatvec(y):
            out = np.zeros(n_out)
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                out += _block(s, e).T @ y[s:e]
            return out

        if self.verbose:
            print(f'  [full MF] design rows rebuilt per product, chunk={chunk}')
        op = self._LO((n, n_out), matvec=matvec, rmatvec=rmatvec, dtype=float)
        op.row_blocks = lambda: ((slice(s, min(s + chunk, n)),
                                  _block(s, min(s + chunk, n)))
                                 for s in range(0, n, chunk))
        return op

    # ------------------------------------------------------------------
    def predict_table(self,
                      labels     : np.ndarray,
                      X          : np.ndarray,
                      struct_ids : Optional[np.ndarray] = None,
                      n_atoms    : Optional[np.ndarray] = None,
                      ) -> np.ndarray:
        """
        Predict every row of a training-format table (E, F and S rows).

        In chain-rule mode F/S rows need their structure's descriptor, so
        the whole table is passed, not single rows.  In row-independent
        mode this is :meth:`predict`.
        """
        if not self._fitted:
            raise RuntimeError("ELMModel: call fit() or fit_csv() first.")
        if self.row_independent:
            return self.predict(X)
        labels = np.array([str(c).strip().upper() for c in labels])
        sid, e_row, n_at = build_structure_index(
            labels, struct_ids, n_atoms, caller='ELMModel.predict_table')
        Zs = self._structure_Z(X, e_row)
        coef = np.append(self._beta, self._beta0)
        out = np.empty(len(labels))
        for s in range(0, len(labels), self.chunk_size):
            e = min(s + self.chunk_size, len(labels))
            out[s:e] = self._chain_rows(X[s:e], labels[s:e], sid[s:e],
                                        Zs, n_at) @ coef
        return out

    def energy_and_gradient(self, d: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        f(d) = E/N and its gradient with respect to the raw structure
        descriptor d (chain-rule models only; used by the MLIP engine).
        """
        if self.row_independent:
            raise RuntimeError(
                "ELMModel: a row_independent model has no energy function "
                "whose derivative gives its force rows (ADR-0004).")
        z = self._preact(self._normalise(d[None, :]))       # (1, n_h')
        f = float(self._hidden_from_Z(z)[0] @ self._beta + self._beta0)
        # grad = sum_k beta_k dh_k/dd = W (h'(z) (*) beta) / X_std
        Dn = self._W_eff().shape[0]
        g = np.empty(Dn)
        eye = np.eye(Dn) / self._X_std[:, None]             # directions
        V = eye @ self._W_eff()                              # (D, n_w)
        g[:] = self._hidden_dir(np.repeat(z, Dn, axis=0), V) @ self._beta
        return f, g

    # ------------------------------------------------------------------
    def fit_csv(self,
                csv_path     : str,
                weight_csv   : Optional[str]  = None,
                val_fraction : float = 0.2,
                val_seed     : int   = 0,
                ) -> 'ELMModel':
        """
        Convenience wrapper: load CSV then call fit().

        Unit-normalisation weights are read from col 2 of the training CSV
        automatically.  The optional ``weight_csv`` file provides additional
        user-defined block weights (see :func:`load_weights_csv`).

        Parameters
        ----------
        csv_path     : path to the 4-column training CSV
                       (target, label, weight, feature_0, feature_1, ...).
                       See :func:`load_mlip_csv` for the format.
        weight_csv   : optional path to a ``weights.csv`` file for
                       user-defined block weights.
                       Format: ``#samples`` and ``weight`` columns; see
                       :func:`load_weights_csv` for the exact layout.
                       ``None`` (default) -> user weights all 1.0.
        val_fraction : fraction of **structures** (not rows) held out for
                       validation.  The split is grouped by structure so
                       that no structure contributes rows to both splits
                       (F17); structure boundaries are inferred from the
                       'E' rows of the CSV.  See :meth:`fit`.
        val_seed     : RNG seed for the train/val split.

        Returns
        -------
        self

        Examples
        --------
        >>> model.fit_csv("structures.csv")
        >>> model.fit_csv("structures.csv", weight_csv="weights.csv")
        """
        targets, labels, weights, X, meta = load_mlip_csv(
            csv_path, verbose=self.verbose, return_meta=True)

        return self.fit(targets, labels, X,
                        weights=weights,
                        weight_csv=weight_csv,
                        val_fraction=val_fraction,
                        val_seed=val_seed,
                        struct_ids=meta["struct_id"],
                        n_atoms=meta["n_atoms"])

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for descriptor matrix X.

        Parameters
        ----------
        X : (N, D) descriptor matrix.

        Returns
        -------
        y_hat : (N,) predicted values.
        """
        if not self._fitted:
            raise RuntimeError("ELMModel: call fit() or fit_csv() first.")
        X_n = self._normalise(X)
        return self._hidden(X_n) @ self._beta + self._beta0

    def compute(self,
                lattice          : np.ndarray,
                positions        : np.ndarray,
                atom_types       : list,
                descriptor_fn    : Optional[Callable] = None,
                descriptor_jac_fn: Optional[Callable] = None,
                neigh_list       = None,
                delta            : float = 1e-4,
                ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        PotentialEngine-compatible interface for FIRE / MD.

        Evaluates the total energy E = sum_i (sum of per-atom predictions).

        Forces and stress can be computed either analytically (preferred)
        or by central-difference numerical differentiation.

        Analytical mode  (fast, O(1) descriptor evaluations for forces)
        ---------------------------------------------------------------
        Requires ``descriptor_jac_fn`` to be supplied.

        ``descriptor_jac_fn(lattice, positions, atom_types)``
            Must return a tuple ``(X, dX_dr, dX_dh)`` where

            X      : (N, D) descriptor matrix  [same as descriptor_fn]
            dX_dr  : (N, D, N, 3) Jacobian dX[i,d] / dr[j,alpha]
            dX_dh  : (N, D, 3, 3) Jacobian dX[i,d] / dh[alpha,beta]
                     (h is the lattice matrix, rows = lattice vectors)

        With these, the force on atom j and the stress are

            F[j,alpha] = -sum_{i,d} (dy_i/dX[i,d]) x dX_dr[i,d,j,alpha]
            sigma[alpha,beta] = -(1/V) sum_{i,d} (dy_i/dX[i,d]) x dX_dh[i,d,alpha,beta]

        where dy_i/dX[i,d] = dE/dX is the output-layer weight
        times the activation derivative (one backpropagation pass
        through the single hidden layer).

        Numerical mode  (fallback when descriptor_jac_fn is None)
        ----------------------------------------------------------
        Uses central differences: cost = (6N + 18) x descriptor_fn calls.

        Parameters
        ----------
        lattice            : (3,3) lattice vectors [A].
        positions          : (N,3) Cartesian positions [A].
        atom_types         : list of element symbols, length N.
        descriptor_fn      : callable(lattice, positions, atom_types) -> (N, D).
                             Required in numerical mode.  In analytical mode
                             may be None if descriptor_jac_fn returns X too.
        descriptor_jac_fn  : callable -> (X, dX_dr, dX_dh) as described above.
                             If provided, analytical forces are used.
        neigh_list         : ignored (kept for interface compatibility).
        delta              : finite-difference displacement [A].
                             Scaled automatically by lattice_mean / sqrt(N)
                             unless set explicitly below 1e-3.

        Returns
        -------
        energy : float        [eV]
        forces : (N,3) array  [eV/A]
        stress : (3,3) array  [eV/A^3]
        """
        if not self._fitted:
            raise RuntimeError("ELMModel.compute: model has not been fitted.")
        if descriptor_fn is None and descriptor_jac_fn is None:
            raise ValueError(
                "ELMModel.compute: supply descriptor_fn (numerical mode) or "
                "descriptor_jac_fn (analytical mode).")

        N = len(positions)

        # -- Analytical mode -----------------------------------
        if descriptor_jac_fn is not None:
            X, dX_dr, dX_dh = descriptor_jac_fn(lattice, positions, atom_types)
            # X: (N, D)  dX_dr: (N, D, N, 3)  dX_dh: (N, D, 3, 3)

            X_norm = self._normalise(X)         # (N, D)
            H      = self._hidden(X_norm)       # (N, n_hidden or 2*n_rff)
            # _beta: (n_hidden,) -- output weights
            beta   = self._beta                 # (n_hidden,)
            E0     = float(np.dot(H.sum(axis=0), beta))

            # dH / dZ: activation derivative depends on activation type.
            b_term = self._b if self._b is not None else 0.0
            if self.activation == 'rff':
                # RFF: H[i,k] = [cos(Z[i,k]), sin(Z[i,k])] / sqrt(N_omega)
                # where Z = X_norm @ _W  (no bias in RFF mode)
                # dH_cos / dZ = -sin(Z) / sqrt(N_omega)
                # dH_sin / dZ =  cos(Z) / sqrt(N_omega)
                N_omega    = self._W.shape[1]
                scale      = 1.0 / np.sqrt(float(N_omega))
                Z_rff      = X_norm @ self._W        # (N, N_omega)
                # beta is (2*N_omega,) = [beta_cos | beta_sin]
                beta_cos   = beta[:N_omega]
                beta_sin   = beta[N_omega:]
                # dE/dZ[i,k] = beta_cos[k]*(-sin) + beta_sin[k]*(cos)
                dE_dZ      = scale * (
                    -np.sin(Z_rff) * beta_cos[None, :]
                    + np.cos(Z_rff) * beta_sin[None, :])   # (N, N_omega)
                # dE/dX_norm[i,d] = sum_k dE_dZ[i,k] * _W[d,k]
                dE_dXnorm  = dE_dZ @ self._W.T        # (N, D)
            else:
                # Standard activations: H = sigma(Z),  Z = X_norm @ _W + b
                # dE / dH[i,k] = beta[k]  (output layer is linear)
                dE_dH  = np.tile(beta, (N, 1))      # (N, n_hidden)
                Z          = X_norm @ self._W + b_term    # (N, n_hidden)
                if self.activation == 'tanh':
                    act_d  = 1.0 - np.tanh(Z) ** 2
                elif self.activation == 'relu':
                    act_d  = (Z > 0).astype(Z.dtype)
                elif self.activation in ('sigmoid', 'logistic'):
                    sig    = 1.0 / (1.0 + np.exp(-Z))
                    act_d  = sig * (1.0 - sig)
                elif self.activation == 'snake':
                    a      = getattr(self, 'snake_a', 1.0)
                    act_d  = 1.0 + 2.0 * a * np.sin(a * Z) * np.cos(a * Z)
                elif self.activation == 'elu':
                    act_d  = np.where(Z >= 0, 1.0, np.exp(Z))
                else:
                    act_d  = 1.0 - np.tanh(Z) ** 2   # fallback: tanh
                # dE/dX_norm[i,d] = sum_k (dE_dH[i,k] * act_d[i,k]) * _W[d,k]
                dE_dXnorm  = (dE_dH * act_d) @ self._W.T  # (N, D)

            # Through normalisation: dX_norm/dX = 1/_X_std
            X_std     = getattr(self, '_X_std', np.ones(X.shape[1]))
            X_std     = np.where(X_std > 1e-12, X_std, 1.0)
            dE_dX     = dE_dXnorm / X_std[None, :]   # (N, D)

            # Forces: F[j,alpha] = -sum_{i,d} dE/dX[i,d] * dX_dr[i,d,j,alpha]
            forces    = -np.einsum('id,idjk->jk', dE_dX, dX_dr)  # (N, 3)

            # Stress: sigma[alpha,beta] = -(1/V) sum_{i,d} dE/dX[i,d] * dX_dh[i,d,alpha,beta]
            vol    = float(abs(np.linalg.det(lattice)))
            stress = (-1.0 / vol) * np.einsum('id,idab->ab', dE_dX, dX_dh)

            return E0, forces, stress

        # -- Numerical mode ------------------------------------
        # D-1: auto-scale delta by lattice parameter and system size
        if delta >= 1e-3:
            lat_mean = float(np.mean(np.linalg.norm(lattice, axis=1)))
            delta_eff = max(1e-4, delta * lat_mean / max(np.sqrt(N), 1.0))
        else:
            delta_eff = float(delta)

        X0 = descriptor_fn(lattice, positions, atom_types)
        E0 = float(self.predict(X0).sum())

        forces = np.zeros((N, 3))
        for i in range(N):
            for alpha in range(3):
                p_p = positions.copy(); p_p[i, alpha] += delta_eff
                p_m = positions.copy(); p_m[i, alpha] -= delta_eff
                Ep = float(self.predict(
                    descriptor_fn(lattice, p_p, atom_types)).sum())
                Em = float(self.predict(
                    descriptor_fn(lattice, p_m, atom_types)).sum())
                forces[i, alpha] = -(Ep - Em) / (2.0 * delta_eff)

        vol    = float(abs(np.linalg.det(lattice)))
        stress = np.zeros((3, 3))
        frac   = np.linalg.solve(lattice.T, positions.T).T
        for a in range(3):
            for b in range(3):
                # Bug-fix note (F9b): the original perturbed by
                # +/- delta_eff/2 here (half the step size used by the force
                # loop above) but kept the *same* `2.0 * delta_eff`
                # denominator below, silently halving every stress
                # component regardless of the position-reconstruction fix
                # (F9, see below): central difference with a +/-h probe
                # needs a `2h` denominator, and h=delta_eff/2 needs `delta_eff`,
                # not `2*delta_eff`.  Using a full +/- delta_eff probe here
                # instead (matching the force loop's convention exactly)
                # keeps the existing `2.0 * delta_eff` denominator correct.
                sp = np.eye(3); sp[a, b] += delta_eff
                sm = np.eye(3); sm[a, b] -= delta_eff
                # Bug-fix note (F9): the original computed
                # `lat_p = sp @ lattice; pos_p = frac @ lat_p.T` (deformation
                # matrix on the LEFT of lattice, then an extra transpose on
                # the way back to Cartesian positions).  Every other
                # finite-difference stress check in this codebase (see
                # test_engine_forces.py's `finite_diff_stress`, itself
                # cross-validated against Ewald's independently-derived
                # real-space stress to 8 significant figures) uses the
                # affine convention `lattice' = lattice @ T`,
                # `positions' = frac @ lattice'` (deformation matrix on the
                # RIGHT, fixed fractional coordinates, no transpose) -- and
                # that is also this codebase's general row-vectors
                # convention (positions = frac @ lattice, e.g. parse_poscar,
                # NeighbourList).  The original combination of
                # left-multiplication plus a transpose reconstructed a
                # displaced structure that agreed with the correct one only
                # for a diagonal (orthogonal-cell, axis-aligned strain)
                # `sp`/`sm`; for a general (off-diagonal / sheared)
                # perturbation both the multiplication order and the
                # transpose needed to change for the finite-difference
                # stress to come out right.
                lat_p = lattice @ sp; pos_p = frac @ lat_p
                lat_m = lattice @ sm; pos_m = frac @ lat_m
                Ep = float(self.predict(
                    descriptor_fn(lat_p, pos_p, atom_types)).sum())
                Em = float(self.predict(
                    descriptor_fn(lat_m, pos_m, atom_types)).sum())
                stress[a, b] = -(Ep - Em) / (2.0 * delta_eff * vol)

        return E0, forces, stress

    # ------------------------------------------------------------------
    #  Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score(y_true   : np.ndarray,
               y_pred   : np.ndarray,
               categories: np.ndarray,
               label    : str) -> dict:
        """
        Compute MAE / RMSE / MaxAE per category and overall.
        """
        result: Dict = {'label': label, 'n': len(y_true)}
        for cat in ('E', 'F', 'S', 'all'):
            if cat == 'all':
                mask = np.ones(len(y_true), dtype=bool)
            else:
                mask = categories == cat
            if not mask.any():
                result[cat] = None
                continue
            err   = y_true[mask] - y_pred[mask]
            abs_e = np.abs(err)
            result[cat] = {
                'MAE'  : float(abs_e.mean()),
                'RMSE' : float(np.sqrt((err**2).mean())),
                'MaxAE': float(abs_e.max()),
                'N'    : int(mask.sum()),
            }
        return result

    def score_csv(self,
                  csv_path  : str,
                  label     : str = "test") -> dict:
        """Load a CSV and return a score dict without re-fitting."""
        targets, labels, weights, X, meta = load_mlip_csv(
            csv_path, verbose=False, return_meta=True)
        y_pred = self.predict_table(labels, X, struct_ids=meta["struct_id"],
                                    n_atoms=meta["n_atoms"])
        # Bug-fix note (F7): the original referenced an undefined name
        # `categories` here (NameError on every call) instead of the
        # `labels` array actually returned by load_mlip_csv.
        sc     = self._score(targets, y_pred, np.asarray(labels), label)
        self._print_scores(sc)
        return sc

    # ------------------------------------------------------------------
    #  Printing
    # ------------------------------------------------------------------

    @staticmethod
    def _print_scores(train_sc: dict,
                      val_sc  : Optional[dict] = None) -> None:
        """Print a formatted MAE/RMSE/MaxAE comparison table."""
        _units = {'E': 'eV', 'F': 'eV/A', 'S': 'eV/A^3', 'all': '(mixed)'}
        sep = "=" * 70
        print("\n" + sep)
        title = "ELM prediction errors"
        if val_sc is not None:
            title += f"  [{train_sc['label']} / {val_sc['label']}]"
        else:
            title += f"  [{train_sc['label']}]"
        print(f"  {title}")
        print(sep)

        hdr = f"  {'Cat':4s} | {'Metric':7s} | {'Train':>14s}"
        if val_sc:
            hdr += f" | {'Val':>14s}"
        hdr += " | Unit"
        print(hdr)
        mid_sep = "  " + "-" * 5 + "+" + "-" * 9 + "+" + "-" * 16
        if val_sc:
            mid_sep += "+" + "-" * 16
        mid_sep += "+" + "-" * 11
        print(mid_sep)

        for cat in ('E', 'F', 'S', 'all'):
            tr = train_sc.get(cat)
            if tr is None:
                continue
            va = val_sc.get(cat) if val_sc else None
            unit = _units[cat]
            for metric in ('MAE', 'RMSE', 'MaxAE'):
                tr_v  = f"{tr[metric]:14.6f}"
                row   = f"  {cat:4s} | {metric:7s} | {tr_v}"
                if val_sc:
                    va_v = f"{va[metric]:14.6f}" if va else f"{'N/A':>14s}"
                    row += f" | {va_v}"
                row += f" | {unit}"
                print(row)
            print(mid_sep)

        ns_tr = train_sc['n']
        ns_va = val_sc['n'] if val_sc else ''
        row   = f"  {'n':4s} | {'':7s} | {ns_tr:>14d}"
        if val_sc:
            row += f" | {ns_va:>14d}"
        print(row + " |")
        print(sep)

    # ------------------------------------------------------------------
    #  Model persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and normalisation stats to a .npz archive."""
        if not self._fitted:
            raise RuntimeError("ELMModel: model has not been fitted yet.")
        np.savez(path,
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
                 **self._extra_state())
        if self.verbose:
            print(f"  ELMModel saved -> {path}.npz")

    def _extra_state(self) -> dict:
        """Arrays added in 0.2.0a3 (chain rule, ADR-0004)."""
        return dict(beta0             = np.array(self._beta0),
                    row_independent   = np.array(self.row_independent),
                    descriptor_config = np.array(self.descriptor_config))

    def _restore_extra_state(self, npz) -> None:
        # A file written before 0.2.0a3 holds a row-independent model.
        self.row_independent = (bool(npz['row_independent'])
                                if 'row_independent' in npz else True)
        self._beta0 = float(npz['beta0']) if 'beta0' in npz else 0.0
        self.descriptor_config = (str(npz['descriptor_config'])
                                  if 'descriptor_config' in npz else "")

    @classmethod
    def load(cls, path: str, verbose: bool = False) -> 'ELMModel':
        """Load a saved ELMModel from a .npz archive."""
        # Bug-fix note (F12): every array save() writes is a plain
        # numeric/boolean/unicode-string array (see save() above) -- none
        # of them are Python objects needing pickle support, so
        # allow_pickle=True (which would let a malicious .npz execute
        # arbitrary code on load) is unnecessary here.
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
            full_matrix_free = bool(npz['full_matrix_free'])
                               if 'full_matrix_free' in npz else False,
            chunk_size       = int(npz['chunk_size'])
                               if 'chunk_size' in npz else 1024,
            use_float32      = bool(npz['use_float32'])
                               if 'use_float32' in npz else False,
            snake_a          = float(npz['snake_a'])
                               if 'snake_a' in npz else 1.0,
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
            verbose          = verbose,
        )
        obj._W       = npz['W']
        b_arr        = npz['b'] if 'b' in npz else np.array([])
        obj._b       = b_arr if b_arr.size > 0 else None
        obj._beta    = npz['beta']
        obj._X_mean  = npz['X_mean']
        obj._X_std   = npz['X_std']
        obj._D       = obj._W.shape[0]
        obj._restore_extra_state(npz)
        obj._fitted  = True
        return obj
