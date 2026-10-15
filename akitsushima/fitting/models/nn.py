"""Feedforward neural-network machine-learning potential (Adam-trained
multilayer perceptron), sharing the same CSV descriptor format as
ELMModel / QuantizedELMModel / LinearModel.
"""
from typing import Optional

import numpy as np

from ...utils.optional_torch import HAS_TORCH, _torch_nn, torch
from .csv_io import (
    build_structure_index,
    load_mlip_csv,
    load_weights_csv,
    structure_grouped_split,
)
from .elm import ELMModel, feature_std, pca_whitening
from ._api import tabular_mlip

__all__ = ["NNModel"]


@tabular_mlip
class NNModel:
    """
    Feedforward neural-network machine-learning potential trained from
    the same CSV descriptor data used by ELMModel / QuantizedELMModel
    (see load_mlip_csv for the file format).

    Unlike ELMModel (a single hidden layer with a FIXED random basis and
    a closed-form linear solve), NNModel is a fully trainable multi-layer
    perceptron optimized end-to-end via backpropagation (Adam optimizer).

    Architecture
    ------------
    Input  : descriptor vector x in R^D
    Hidden : an arbitrary number of fully-connected layers, each
             followed by an activation function. The NUMBER of hidden
             layers is given implicitly by the length of the
             `hidden_layers` list (e.g. hidden_layers=[128, 64] builds
             two hidden layers of size 128 and 64); there is no separate
             "number of layers" parameter to avoid redundant/inconsistent
             configuration.
    Output : E/N = f(d) for a structure descriptor d.  F and S rows are
             predicted as N grad f(d) . X by autograd (chain rule,
             ADR-0004); ``row_independent=True`` instead predicts every
             row as f(x) (the pre-0.2.0a3 behaviour).

    Supported activation functions: 'tanh', 'relu', 'gelu', 'silu'
    (a.k.a. swish), 'mish', 'softplus'.

    Weighting scheme
    ----------------
    Weight on the squared residual = w_cat * w_unit^2 * w_user
    (ADR-0004; w_unit was not squared before 0.2.0a3), where:
      w_cat  : category weight (w_energy / w_forces / w_stress).
      w_unit : unit-normalisation weight from CSV column 2 (see
               load_mlip_csv for the exact convention -- N for E rows,
               r0 for F rows, eps0*V for S rows).
      w_user : optional user-defined block weight from a weights.csv
               file (see load_weights_csv).

    This is the same rule as ELMModel and LinearModel and as the
    classical ``fit`` (w_cat x w_unit^2).

    Parameters
    ----------
    hidden_layers  : list of int -- hidden layer sizes, e.g. [128, 64].
                      len(hidden_layers) is the number of hidden layers.
    activation     : 'tanh' | 'relu' | 'gelu' | 'silu' | 'mish' | 'softplus'
                      (default softplus since 0.2.0a3: on SW-Si test cells
                      outside the training strains it extrapolated best;
                      relu gives discontinuous forces and is unsuited to MD).
    w_energy       : category weight applied to 'E' rows.
    w_forces       : category weight applied to 'F' rows.
    w_stress       : category weight applied to 'S' rows.
    r0             : typical atomic displacement [A] (for F-row weight
                      consistency with load_mlip_csv; not otherwise used).
    eps0           : typical cell strain [dimensionless] (for S-row
                      weight consistency with load_mlip_csv).
    learning_rate  : Adam optimizer learning rate.
    epochs         : maximum number of training epochs.
    batch_size     : mini-batch size in rows (chain-rule mode batches whole
                      structures, about batch_size rows per batch).
    weight_decay   : L2 regularisation coefficient (Adam's weight_decay).
    batch_structures : chain-rule mode: structures per mini-batch (takes
                      precedence over batch_size).  Default 30 since
                      0.2.0a3 (SW-Si comparison, ADR-0004: fastest setting
                      passing every seed; 4-6x faster than ~1-2 structures
                      per batch).  None -> about batch_size rows per batch
                      (batch_size is also what row-independent mode uses).
    optimizer      : 'adam' (mini-batch) or 'lbfgs' (full batch, strong
                      Wolfe line search, 20 iterations per epoch; with
                      lbfgs, learning_rate is the initial step, 1.0 is
                      the usual choice).
    deterministic  : make training bit-reproducible for a given seed:
                      torch's deterministic algorithms and a single CPU
                      thread during fit() (multi-threaded reductions sum
                      in a varying order).  Slower; off by default.
    direct_link    : (default True) add a linear term of the standardised descriptor to
                      the output, f(x) = MLP(x) + gamma . x_n (as the ELM's
                      RVFL option), so the network only has to learn the
                      nonlinear remainder of a nearly linear energy.
    early_stopping_patience : (default 150, epochs default 1500 since
                      0.2.0a3: E rows are 1 in ~3N+7 rows and converge last;
                      patience 30-40 left E at up to 1.9 sigma on SW-Si,
                      150 brought every seed to the noise, ADR-0004)
                      stop training if validation loss does not
                      improve for this many consecutive epochs (0 disables
                      early stopping, always running the full `epochs`).
    standardize    : 'zscore' (default) or 'pca'.  Inputs are z-scored
                      with the training structures' mean and std; 'pca'
                      then applies the whitening map of
                      :func:`pca_whitening`, as ELMModel does (ADR-0013;
                      for NN since 0.2.0a12).  F and S rows get the same
                      linear map without the mean.
    pca_eps, pca_drop : whitening regulariser and relative eigenvalue
                      cut (see :func:`pca_whitening`); used with 'pca'.
    min_feature_std : mask input columns whose training std is below it
                      (default 1e-12: none; see :func:`feature_std`).
    device         : 'auto' | 'cpu' | 'cuda'. 'auto' selects CUDA if
                      available, otherwise CPU.
    seed           : RNG seed (numpy, torch, and the train/val split).
    verbose        : print training progress and diagnostics.
    """

    def __init__(self,
                 hidden_layers  : list  = None,
                 activation     : str   = 'softplus',
                 w_energy       : float = 1.0,
                 w_forces       : float = 0.1,
                 w_stress       : float = 10.0,
                 r0             : float = 1.0,
                 eps0           : float = 0.01,
                 learning_rate  : float = 1.0e-3,
                 epochs         : int   = 1500,
                 batch_size     : int   = 256,
                 weight_decay   : float = 1.0e-6,
                 early_stopping_patience: int = 150,
                 device         : str   = 'auto',
                 seed           : int   = 42,
                 verbose        : bool  = True,
                 row_independent: bool  = False,
                 direct_link    : bool  = True,
                 deterministic  : bool  = False,
                 batch_structures: Optional[int] = 30,
                 optimizer      : str   = 'adam',
                 standardize    : str   = 'zscore',
                 pca_eps        : float = 1e-3,
                 pca_drop       : float = 0.0,
                 min_feature_std: float = 1e-12):
        if not HAS_TORCH:
            raise ImportError(
                "NNModel requires PyTorch, which is not installed. "
                'Install it with: pip install "akitsushima[torch]" (or pip install torch).')

        self.hidden_layers  = list(hidden_layers) if hidden_layers else [128, 64]
        self.activation     = str(activation).lower()
        self.w_energy       = float(w_energy)
        self.w_forces       = float(w_forces)
        self.w_stress       = float(w_stress)
        self.r0             = float(r0)
        self.eps0           = float(eps0)
        self.learning_rate  = float(learning_rate)
        self.epochs         = int(epochs)
        self.batch_size     = int(batch_size)
        self.weight_decay   = float(weight_decay)
        self.early_stopping_patience = int(early_stopping_patience)
        self.seed           = int(seed)
        self.verbose        = bool(verbose)
        self.row_independent = bool(row_independent)
        self.direct_link    = bool(direct_link)
        self.deterministic  = bool(deterministic)
        self.batch_structures = (int(batch_structures) if batch_structures
                                 else None)
        self.optimizer      = str(optimizer).lower()
        if self.optimizer not in ('adam', 'lbfgs'):
            raise ValueError(f"NNModel: optimizer must be 'adam' or 'lbfgs', "
                             f"got {optimizer!r}.")
        self.standardize    = str(standardize).lower()
        if self.standardize not in ('zscore', 'pca'):
            raise ValueError(f"NNModel: standardize must be 'zscore' or 'pca', "
                             f"got {standardize!r}.")
        self.pca_eps        = float(pca_eps)
        self.pca_drop       = float(pca_drop)
        self.min_feature_std = float(min_feature_std)
        self.descriptor_config = ""   # see ELMModel.descriptor_config
        # physics-informed parts (ADR-0014): pair baseline, isolated atom
        self.physics_config = ""

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self._ACTIVATIONS = {
            'tanh':     _torch_nn.Tanh,
            'relu':     _torch_nn.ReLU,
            'gelu':     _torch_nn.GELU,
            'silu':     _torch_nn.SiLU,
            'mish':     _torch_nn.Mish,
            'softplus': _torch_nn.Softplus,
        }
        if self.activation not in self._ACTIVATIONS:
            raise ValueError(
                f"NNModel: unknown activation '{self.activation}'. "
                f"Choose from {sorted(self._ACTIVATIONS.keys())}.")

        self._net      = None   # built lazily in fit(), once D is known
        self._X_mean   = None
        self._X_std    = None
        self._P        = None   # PCA whitening map (D, D') or None
        self._D        = None
        self._fitted   = False

    # ------------------------------------------------------------------
    def _build_network(self, D: int):
        """Construct the feedforward network for input dimension D."""
        act_cls = self._ACTIVATIONS[self.activation]
        layers  = []
        in_dim  = D
        for h in self.hidden_layers:
            layers.append(_torch_nn.Linear(in_dim, h))
            layers.append(act_cls())
            in_dim = h
        layers.append(_torch_nn.Linear(in_dim, 1))
        net = _torch_nn.Sequential(*layers)
        if self.direct_link:
            net = _DirectLink(net, D)
        return net.to(self.device)

    # ------------------------------------------------------------------
    def _normalise(self, X: np.ndarray) -> np.ndarray:
        """Network input of descriptors X: z-score, then the PCA map."""
        Z = (X - self._X_mean) / self._X_std
        return Z if self._P is None else Z @ self._P

    def _direction(self, X: np.ndarray) -> np.ndarray:
        """The same linear map without the mean (F / S rows): d x_n for
        a change X in the raw descriptor."""
        Z = X / self._X_std
        return Z if self._P is None else Z @ self._P

    def _raw_gradient(self, g: np.ndarray) -> np.ndarray:
        """Gradient w.r.t. the network input -> w.r.t. the raw descriptor."""
        if self._P is not None:
            g = g @ self._P.T
        return g / self._X_std

    @property
    def _D_in(self) -> int:
        return self._D if self._P is None else int(self._P.shape[1])

    # ------------------------------------------------------------------
    def _row_weights(self, labels, w_unit, w_user) -> np.ndarray:
        """Weight on each row's SQUARED residual: w_cat * w_unit^2 * w_user
        (ADR-0004, decision 2)."""
        _wmap = {'E': self.w_energy, 'F': self.w_forces, 'S': self.w_stress}
        w_cat = np.array([_wmap.get(c, 1.0) for c in labels], dtype=np.float64)
        return w_cat * w_unit ** 2 * w_user

    def _chain_predict_t(self, d_n, X_dir, lab_isE, sid_loc, n_at,
                         create_graph):
        """
        Predictions of a block of rows in chain-rule mode (torch).

        d_n     : (S, D) normalised descriptors of the block's structures
        X_dir   : (R, D) row descriptors divided by X_std (F/S rows use it)
        lab_isE : (R,) bool, True for E rows
        sid_loc : (R,) long, row -> structure index into d_n
        n_at    : (S,) float, atom count per structure

        E row: f(d);  F/S row: N grad f(d) . X   (grad w.r.t. raw d,
        hence X / X_std against the gradient in normalised variables).
        """
        d_n = d_n.detach().requires_grad_(True)
        f = self._net(d_n).view(-1)                          # (S,)
        g = torch.autograd.grad(f.sum(), d_n, create_graph=create_graph)[0]
        pred_E = f[sid_loc]
        pred_D = n_at[sid_loc] * (g[sid_loc] * X_dir).sum(dim=1)
        return torch.where(lab_isE, pred_E, pred_D)

    # ------------------------------------------------------------------
    def fit(self, *args, **kwargs) -> 'NNModel':
        """Train the network; see :meth:`_fit` for the arguments.  With
        ``deterministic=True`` the run is bit-reproducible for a seed."""
        from ..physics import physics_from_config
        if physics_from_config(self.physics_config)[1]:
            raise ValueError(
                "NNModel: isolated_atom_energy is supported by elm / qelm "
                "(tabular with one species, atom-centred) and the "
                "atom-centred nn (ADR-0014).")
        if not self.deterministic:
            return self._fit(*args, **kwargs)
        n_thr = torch.get_num_threads()
        was_det = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        try:
            return self._fit(*args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(was_det)
            torch.set_num_threads(n_thr)

    def _fit(self,
            targets      : np.ndarray,
            labels       : np.ndarray,
            X            : np.ndarray,
            weights      : Optional[np.ndarray] = None,
            weight_csv   : Optional[str]        = None,
            val_fraction : float = 0.2,
            val_seed     : int   = 0,
            struct_ids   : Optional[np.ndarray] = None,
            n_atoms      : Optional[np.ndarray] = None,
            ) -> 'NNModel':
        """
        Train the network on (targets, labels, X) descriptor rows.

        Model form: E/N = f(d); every F/S row is N grad f(d) . X with d the
        descriptor of the row's structure (chain rule, ADR-0004), computed
        by autograd.  ``row_independent=True`` (constructor) restores the
        pre-0.2.0a3 per-row regression y = f(x).

        Loss: mean over rows of w_i (pred_i - y_i)^2 with
        w_i = w_cat * w_unit^2 * w_user (ADR-0004, decision 2).

        Parameters
        ----------
        targets      : (N,) float  target values.
        labels       : (N,) str    'E', 'F', or 'S' per row.
        X            : (N, D)      descriptor feature matrix.
        weights      : (N,) float or None -- unit-normalisation weights
                       from load_mlip_csv col 2. None -> all 1.0.
        weight_csv   : optional path to a weights.csv file for
                       user-defined block weights.
        val_fraction : fraction of structures held out for validation /
                       early stopping.
        val_seed     : RNG seed for the train/val split.
        struct_ids, n_atoms : optional CSV metadata columns (see
                       ELMModel.fit).

        Returns
        -------
        self
        """
        labels = np.array([str(c).strip().upper() for c in labels])
        N, D = X.shape

        if weights is not None:
            w_unit = np.asarray(weights, dtype=np.float64)
            if w_unit.shape != (N,):
                raise ValueError(
                    f"NNModel.fit: weights.shape={w_unit.shape} must be ({N},).")
            if np.any(w_unit < 0):
                raise ValueError("NNModel.fit: weights contains negative values.")
        else:
            w_unit = np.ones(N, dtype=np.float64)

        if weight_csv is not None:
            w_user = load_weights_csv(weight_csv, n_total=N, verbose=self.verbose)
        else:
            w_user = np.ones(N, dtype=np.float64)

        w_all = self._row_weights(labels, w_unit, w_user)

        # Train / val split, grouped by structure (F17): the rows of one
        # structure all derive from one configuration, and the validation
        # loss is this model's early-stopping criterion.
        idx_train, idx_val = structure_grouped_split(
            labels, val_fraction=val_fraction, val_seed=val_seed,
            caller='NNModel.fit', struct_ids=struct_ids)
        n_val = int(idx_val.size)

        if self.verbose:
            print(f"\nNNModel fitting  "
                  f"[{'row-independent' if self.row_independent else 'chain rule'}]")
            print(f"  Samples      : {N}  (D={D})")
            print(f"  Architecture : {D} -> " +
                  " -> ".join(str(h) for h in self.hidden_layers) + " -> 1"
                  f"  (activation={self.activation})")
            print(f"  Category wts : wE={self.w_energy}  "
                  f"wF={self.w_forces}  wS={self.w_stress}")
            print(f"  Train/val split: train={len(idx_train)}  val={n_val}")
            print(f"  Device       : {self.device}")

        if self.row_independent:
            X_stat = X[idx_train]
        else:
            sid, e_row, n_at = build_structure_index(
                labels, struct_ids, n_atoms, caller='NNModel.fit')
            X_stat = X[e_row[np.unique(sid[idx_train])]]
        self._X_mean = X_stat.mean(axis=0)
        self._X_std  = feature_std(X_stat, self.min_feature_std)
        self._P = (pca_whitening((X_stat - self._X_mean) / self._X_std,
                                 self.pca_eps, self.pca_drop)
                   if self.standardize == 'pca' else None)
        self._D = D
        if self.verbose and self._P is not None:
            print(f"  PCA whitening: {D} -> {self._D_in} input directions")

        torch.manual_seed(self.seed)
        self._net = self._build_network(self._D_in)
        if self.optimizer == 'lbfgs':
            opt = torch.optim.LBFGS(self._net.parameters(),
                                    lr=self.learning_rate, max_iter=20,
                                    history_size=20,
                                    line_search_fn='strong_wolfe')
        else:
            opt = torch.optim.Adam(self._net.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=self.weight_decay)

        dev, f32 = self.device, torch.float32

        def _t(a, dtype=f32):
            return torch.as_tensor(a, dtype=dtype, device=dev)

        # All data are converted to tensors once, here; a batch is then
        # only index arithmetic on the device (ADR-0004 follow-up: the
        # per-batch numpy -> torch conversion was 20-40% of the time).
        y_all = _t(targets); w_all_t = _t(w_all)
        if self.row_independent:
            Xn_all = _t(self._normalise(X))

            def _blocks(idx, shuffle, full=False):
                order = (idx[torch.randperm(len(idx)).numpy()]
                         if shuffle else idx)
                step = len(order) if full else self.batch_size
                for s in range(0, len(order), max(step, 1)):
                    yield _t(order[s:s + step], torch.long), None

            def _pred(rows_t, structs, create_graph):
                return self._net(Xn_all[rows_t]).view(-1)
        else:
            # Batches are whole structures: batch_structures of them, or
            # enough to make ~batch_size rows.
            order_rows = np.argsort(sid, kind="stable")
            starts = np.searchsorted(sid[order_rows], np.arange(len(e_row)))
            ends = np.append(starts[1:], len(order_rows))
            dn_all = _t(self._normalise(X[e_row]))
            nat_all = _t(n_at.astype(np.float64))
            Xdir_all = _t(self._direction(X))
            isE_all = _t(labels == 'E', torch.bool)
            sid_all = _t(sid, torch.long)
            pos = torch.full((len(e_row),), -1, dtype=torch.long, device=dev)

            def _blocks(idx, shuffle, full=False):
                structs = np.unique(sid[idx])
                if shuffle:
                    structs = structs[torch.randperm(len(structs)).numpy()]
                if full:
                    per = len(structs)
                elif self.batch_structures:
                    per = self.batch_structures
                else:
                    per = max(1, int(round(self.batch_size * len(structs)
                                           / max(len(idx), 1))))
                for s in range(0, len(structs), per):
                    st = structs[s:s + per]
                    rows = np.concatenate([order_rows[starts[k]:ends[k]]
                                           for k in st])
                    yield _t(rows, torch.long), np.sort(st)

            def _pred(rows_t, structs, create_graph):
                st = _t(structs, torch.long)
                pos[st] = torch.arange(len(structs), device=dev)
                loc = pos[sid_all[rows_t]]
                return self._chain_predict_t(
                    dn_all[st], Xdir_all[rows_t], isE_all[rows_t], loc,
                    nat_all[st], create_graph)

        def _loss(rows_t, structs, create_graph):
            p = _pred(rows_t, structs, create_graph)
            return (w_all_t[rows_t] * (p - y_all[rows_t]) ** 2).sum(), len(rows_t)

        def _eval_loss(idx):
            tot, n = 0.0, 0
            for rows_t, structs in _blocks(idx, shuffle=False):
                with torch.enable_grad():
                    l, k = _loss(rows_t, structs, create_graph=False)
                tot += float(l.detach()); n += k
            return tot / max(n, 1)

        if self.optimizer == 'lbfgs':
            # Full batch: every training structure in one closure.  L-BFGS
            # has no weight_decay argument; the same L2 term Adam applies
            # (gradient weight_decay * theta) is added to the loss.
            full_rows, full_structs = next(_blocks(idx_train, False, full=True))

            def _closure():
                opt.zero_grad()
                l, k = _loss(full_rows, full_structs, create_graph=True)
                obj = l / k
                if self.weight_decay:
                    obj = obj + 0.5 * self.weight_decay * sum(
                        (p ** 2).sum() for p in self._net.parameters())
                obj.backward()
                return obj

        best_val = float('inf')
        best_state = None
        patience_ctr = 0
        for epoch in range(self.epochs):
            self._net.train()
            epoch_loss, n_seen = 0.0, 0
            if self.optimizer == 'lbfgs':
                opt.step(_closure)
                with torch.enable_grad():
                    l, k = _loss(full_rows, full_structs, create_graph=False)
                epoch_loss, n_seen = float(l.detach()), k
            else:
                for rows_t, structs in _blocks(idx_train, shuffle=True):
                    opt.zero_grad()
                    l, k = _loss(rows_t, structs, create_graph=True)
                    (l / k).backward()
                    opt.step()
                    epoch_loss += float(l.detach()); n_seen += k
            epoch_loss /= max(n_seen, 1)

            if n_val:
                self._net.eval()
                val_loss = _eval_loss(idx_val)
                if val_loss < best_val - 1e-12:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1

                if self.verbose and (epoch % max(1, self.epochs // 10) == 0
                                     or epoch == self.epochs - 1):
                    print(f"  epoch {epoch:5d}  train_loss={epoch_loss:.6e}  "
                          f"val_loss={val_loss:.6e}")

                if (self.early_stopping_patience > 0
                        and patience_ctr >= self.early_stopping_patience):
                    if self.verbose:
                        print(f"  Early stopping at epoch {epoch} "
                              f"(no improvement for {self.early_stopping_patience} epochs)")
                    break
            else:
                if self.verbose and (epoch % max(1, self.epochs // 10) == 0
                                     or epoch == self.epochs - 1):
                    print(f"  epoch {epoch:5d}  train_loss={epoch_loss:.6e}")

        self.n_epochs_run_ = epoch + 1
        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._fitted = True
        self._net.eval()

        def _sub(idx):
            return dict(labels=labels[idx], X=X[idx],
                        struct_ids=(None if struct_ids is None
                                    else np.asarray(struct_ids)[idx]),
                        n_atoms=(None if n_atoms is None
                                 else np.asarray(n_atoms)[idx]))
        self._train_sc = ELMModel._score(
            targets[idx_train], self.predict_table(**_sub(idx_train)),
            labels[idx_train], 'train')
        self._val_sc = (ELMModel._score(
            targets[idx_val], self.predict_table(**_sub(idx_val)),
            labels[idx_val], 'val') if n_val else None)
        if self.verbose:
            ELMModel._print_scores(self._train_sc, self._val_sc)
        return self

    # ------------------------------------------------------------------
    def predict_table(self,
                      labels     : np.ndarray,
                      X          : np.ndarray,
                      struct_ids : Optional[np.ndarray] = None,
                      n_atoms    : Optional[np.ndarray] = None,
                      ) -> np.ndarray:
        """Predict every row of a training-format table (E, F and S rows);
        see ELMModel.predict_table."""
        if not self._fitted:
            raise RuntimeError("NNModel: call fit() or fit_csv() first.")
        if self.row_independent:
            return self.predict(X)
        labels = np.array([str(c).strip().upper() for c in labels])
        sid, e_row, n_at = build_structure_index(
            labels, struct_ids, n_atoms, caller='NNModel.predict_table')
        dev, f32 = self.device, torch.float32
        self._net.eval()
        with torch.enable_grad():
            p = self._chain_predict_t(
                torch.as_tensor(self._normalise(X[e_row]), dtype=f32, device=dev),
                torch.as_tensor(self._direction(X), dtype=f32, device=dev),
                torch.as_tensor(labels == 'E', device=dev),
                torch.as_tensor(sid, dtype=torch.long, device=dev),
                torch.as_tensor(n_at, dtype=f32, device=dev),
                create_graph=False)
        return p.detach().cpu().numpy().astype(np.float64)

    def energy_and_gradient(self, d: np.ndarray):
        """f(d) = E/N and its gradient w.r.t. the raw descriptor d
        (chain-rule models only; used by the MLIP engine)."""
        if self.row_independent:
            raise RuntimeError(
                "NNModel: a row_independent model has no energy function "
                "whose derivative gives its force rows (ADR-0004).")
        self._net.eval()
        dn = torch.as_tensor(self._normalise(d[None, :]), dtype=torch.float32,
                             device=self.device).requires_grad_(True)
        with torch.enable_grad():
            f = self._net(dn).sum()
            g = torch.autograd.grad(f, dn)[0]
        return (float(f.detach()),
                self._raw_gradient(g.detach().cpu().numpy()[0].astype(np.float64)))

    # ------------------------------------------------------------------
    def fit_csv(self,
                csv_path     : str,
                weight_csv   : Optional[str] = None,
                val_fraction : float = 0.2,
                val_seed     : int   = 0,
                ) -> 'NNModel':
        """Convenience wrapper: load CSV (load_mlip_csv format) then fit()."""
        targets, labels, weights, X, meta = load_mlip_csv(
            csv_path, verbose=self.verbose, return_meta=True)
        return self.fit(targets, labels, X,
                        weights=weights, weight_csv=weight_csv,
                        val_fraction=val_fraction, val_seed=val_seed,
                        struct_ids=meta["struct_id"], n_atoms=meta["n_atoms"])

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for descriptor matrix X."""
        if not self._fitted:
            raise RuntimeError("NNModel: call fit() or fit_csv() first.")
        X_n = self._normalise(X)
        self._net.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(X_n, dtype=torch.float32, device=self.device)
            y_hat = self._net(X_t).view(-1).cpu().numpy()
        return y_hat

    # ------------------------------------------------------------------
    def score_csv(self, csv_path: str, label: str = "test") -> dict:
        """Load a CSV and return a score dict without re-fitting."""
        targets, labels, weights, X, meta = load_mlip_csv(
            csv_path, verbose=False, return_meta=True)
        y_pred = self.predict_table(labels, X, struct_ids=meta["struct_id"],
                                    n_atoms=meta["n_atoms"])
        sc = ELMModel._score(targets, y_pred, np.asarray(labels), label)
        ELMModel._print_scores(sc)
        return sc

    # ------------------------------------------------------------------
    #  Model persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save network weights and metadata to a single .pt file."""
        if not self._fitted:
            raise RuntimeError("NNModel: model has not been fitted yet.")
        p = path if path.endswith('.pt') else path + '.pt'
        torch.save({
            'state_dict'    : self._net.state_dict(),
            'hidden_layers' : self.hidden_layers,
            'activation'    : self.activation,
            'w_energy'      : self.w_energy,
            'w_forces'      : self.w_forces,
            'w_stress'      : self.w_stress,
            'r0'            : self.r0,
            'eps0'          : self.eps0,
            'seed'          : self.seed,
            'D'             : self._D,
            'X_mean'        : self._X_mean,
            'X_std'         : self._X_std,
            'standardize'   : self.standardize,
            'pca_eps'       : self.pca_eps,
            'pca_drop'      : self.pca_drop,
            'min_feature_std': self.min_feature_std,
            'X_P'           : self._P,
            'row_independent'  : self.row_independent,
            'direct_link'      : self.direct_link,
            'descriptor_config': self.descriptor_config,
            'physics_config'   : self.physics_config,
        }, p)
        if self.verbose:
            print(f"  NNModel saved -> {p}")

    @classmethod
    def load(cls, path: str, verbose: bool = False) -> 'NNModel':
        """Load a saved NNModel from a .pt file."""
        if not HAS_TORCH:
            raise ImportError("NNModel.load requires PyTorch, which is not installed. "
                              'Install it with: pip install "akitsushima[torch]" (or pip install torch).')
        p = path if path.endswith('.pt') else path + '.pt'
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        obj = cls(
            hidden_layers = ckpt['hidden_layers'],
            activation    = ckpt['activation'],
            w_energy      = ckpt['w_energy'],
            w_forces      = ckpt['w_forces'],
            w_stress      = ckpt['w_stress'],
            r0            = ckpt['r0'],
            eps0          = ckpt['eps0'],
            seed          = ckpt['seed'],
            direct_link   = bool(ckpt.get('direct_link', False)),
            device        = 'cpu',
            verbose       = verbose,
            # checkpoints before 0.2.0a12 are z-score models
            standardize   = ckpt.get('standardize', 'zscore'),
            pca_eps       = ckpt.get('pca_eps', 1e-3),
            pca_drop      = ckpt.get('pca_drop', 0.0),
            min_feature_std = ckpt.get('min_feature_std', 1e-12),
        )
        obj._D      = int(ckpt['D'])
        obj._P      = ckpt.get('X_P')
        obj._net    = obj._build_network(obj._D_in)
        obj._net.load_state_dict(ckpt['state_dict'])
        obj._X_mean = ckpt['X_mean']
        obj._X_std  = ckpt['X_std']
        # A checkpoint written before 0.2.0a3 holds a row-independent model.
        obj.row_independent   = bool(ckpt.get('row_independent', True))
        obj.descriptor_config = str(ckpt.get('descriptor_config', ''))
        obj.physics_config    = str(ckpt.get('physics_config', ''))
        obj._fitted = True
        return obj


if HAS_TORCH:
    class _DirectLink(_torch_nn.Module):
        """MLP(x) + gamma . x (no bias: the MLP's last layer has one)."""

        def __init__(self, mlp, D):
            super().__init__()
            self.mlp = mlp
            self.lin = _torch_nn.Linear(D, 1, bias=False)
            _torch_nn.init.zeros_(self.lin.weight)

        def forward(self, x):
            return self.mlp(x) + self.lin(x)
