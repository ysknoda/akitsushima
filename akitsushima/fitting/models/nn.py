"""Feedforward neural-network machine-learning potential (Adam-trained
multilayer perceptron), sharing the same CSV descriptor format as
ELMModel / QuantizedELMModel / LightGBMModel.
"""
from typing import Optional

import numpy as np

from ...utils.optional_torch import HAS_TORCH, _torch_nn, torch
from .csv_io import load_mlip_csv, load_weights_csv, structure_grouped_split
from .elm import ELMModel

__all__ = ["NNModel"]


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
    Output : a single scalar prediction per row (matching the row-based
             E / F / S target scheme used throughout this codebase's
             CSV format).

    Supported activation functions: 'tanh', 'relu', 'gelu', 'silu'
    (a.k.a. swish), 'mish', 'softplus'.

    Weighting scheme
    ----------------
    Sample weight = w_cat * w_unit * w_user, where:
      w_cat  : category weight (w_energy / w_forces / w_stress).
      w_unit : unit-normalisation weight from CSV column 2 (see
               load_mlip_csv for the exact convention -- N for E rows,
               r0 for F rows, eps0*V for S rows).
      w_user : optional user-defined block weight from a weights.csv
               file (see load_weights_csv).

    NOTE: unlike ELMModel, this does NOT include an Uncertainty
    Weighting (UW) term. ELMModel's UW is a simplified, closed-form,
    one-shot weighting scheme (target-variance-based) that works
    because ELM's training is a single linear least-squares solve.
    Implementing the genuine Kendall et al. (2018) UW scheme for a
    trainable deep network requires promoting log(sigma_c) to a
    LEARNABLE parameter jointly optimized with the network weights via
    backpropagation -- a loss-function redesign beyond the scope of
    this class. Sample weighting here uses only w_cat * w_unit * w_user.

    Parameters
    ----------
    hidden_layers  : list of int -- hidden layer sizes, e.g. [128, 64].
                      len(hidden_layers) is the number of hidden layers.
    activation     : 'tanh' | 'relu' | 'gelu' | 'silu' | 'mish' | 'softplus'.
    w_energy       : category weight applied to 'E' rows.
    w_forces       : category weight applied to 'F' rows.
    w_stress       : category weight applied to 'S' rows.
    r0             : typical atomic displacement [A] (for F-row weight
                      consistency with load_mlip_csv; not otherwise used).
    eps0           : typical cell strain [dimensionless] (for S-row
                      weight consistency with load_mlip_csv).
    learning_rate  : Adam optimizer learning rate.
    epochs         : maximum number of training epochs.
    batch_size     : mini-batch size for stochastic gradient descent.
    weight_decay   : L2 regularisation coefficient (Adam's weight_decay).
    early_stopping_patience : stop training if validation loss does not
                      improve for this many consecutive epochs (0 disables
                      early stopping, always running the full `epochs`).
    device         : 'auto' | 'cpu' | 'cuda'. 'auto' selects CUDA if
                      available, otherwise CPU.
    seed           : RNG seed (numpy, torch, and the train/val split).
    verbose        : print training progress and diagnostics.
    """

    def __init__(self,
                 hidden_layers  : list  = None,
                 activation     : str   = 'silu',
                 w_energy       : float = 1.0,
                 w_forces       : float = 0.1,
                 w_stress       : float = 10.0,
                 r0             : float = 1.0,
                 eps0           : float = 0.01,
                 learning_rate  : float = 1.0e-3,
                 epochs         : int   = 500,
                 batch_size     : int   = 256,
                 weight_decay   : float = 1.0e-6,
                 early_stopping_patience: int = 30,
                 device         : str   = 'auto',
                 seed           : int   = 42,
                 verbose        : bool  = True):
        if not HAS_TORCH:
            raise ImportError(
                "NNModel requires PyTorch, which is not installed. "
                "Install it with: pip install torch --break-system-packages")

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
        return net.to(self.device)

    # ------------------------------------------------------------------
    def _normalise(self, X: np.ndarray) -> np.ndarray:
        """Z-score normalise using statistics from fit()."""
        return (X - self._X_mean) / self._X_std

    # ------------------------------------------------------------------
    def fit(self,
            targets      : np.ndarray,
            labels       : np.ndarray,
            X            : np.ndarray,
            weights      : Optional[np.ndarray] = None,
            weight_csv   : Optional[str]        = None,
            val_fraction : float = 0.2,
            val_seed     : int   = 0,
            ) -> 'NNModel':
        """
        Train the network on (targets, labels, X) descriptor rows.

        See the class docstring for the CSV row convention and the
        sample-weighting scheme (w_cat * w_unit * w_user).

        Parameters
        ----------
        targets      : (N,) float  target values.
        labels       : (N,) str    'E', 'F', or 'S' per row.
        X            : (N, D)      descriptor feature matrix.
        weights      : (N,) float or None -- unit-normalisation weights
                       from load_mlip_csv col 2. None -> all 1.0.
        weight_csv   : optional path to a weights.csv file for
                       user-defined block weights.
        val_fraction : fraction of samples held out for validation /
                       early stopping.
        val_seed     : RNG seed for the train/val split.

        Returns
        -------
        self
        """
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

        _wmap  = {'E': self.w_energy, 'F': self.w_forces, 'S': self.w_stress}
        w_cat  = np.array([_wmap.get(c, 1.0) for c in labels])
        w_all  = w_cat * w_unit * w_user

        # Train / val split, grouped by structure.
        #
        # Bug-fix note (F17): this used to be a plain row-wise shuffle over
        # the N CSV *rows*, which leaked essentially every structure into
        # both splits -- one structure contributes 1 E + 3N_atoms F + 6 S
        # rows that are all derived from the same atomic configuration.
        # That mattered doubly here: the leaked validation loss is also
        # this model's early-stopping criterion and best-state selector, so
        # training stopped on a signal the network had effectively already
        # seen.  See csv_io.structure_grouped_split for the full note.
        idx_train, idx_val = structure_grouped_split(
            labels, val_fraction=val_fraction, val_seed=val_seed,
            caller='NNModel.fit')
        n_val = int(idx_val.size)

        X_tr = X[idx_train]; y_tr = targets[idx_train]; w_tr = w_all[idx_train]
        X_va = X[idx_val]    if n_val else None
        y_va = targets[idx_val] if n_val else None
        w_va = w_all[idx_val]   if n_val else None

        if self.verbose:
            print(f"\nNNModel fitting")
            print(f"  Samples      : {N}  (D={D})")
            print(f"  Architecture : {D} -> " +
                  " -> ".join(str(h) for h in self.hidden_layers) + " -> 1"
                  f"  (activation={self.activation})")
            print(f"  Category wts : wE={self.w_energy}  "
                  f"wF={self.w_forces}  wS={self.w_stress}")
            print(f"  Train/val split: train={len(idx_train)}  val={n_val}")
            print(f"  Device       : {self.device}")

        # Normalise descriptors (statistics from training split only)
        self._X_mean = X_tr.mean(axis=0)
        self._X_std  = X_tr.std(axis=0)
        self._X_std[self._X_std < 1e-12] = 1.0
        X_tr_n = self._normalise(X_tr)
        X_va_n = self._normalise(X_va) if n_val else None
        self._D = D

        # Bug-fix note (F12): the original also called the legacy global
        # `np.random.seed(self.seed)` here.  Nothing in this method (or
        # anywhere downstream in this class) draws from the legacy
        # global `np.random.*` state -- the train/val split above already
        # uses its own `np.random.default_rng(val_seed)` Generator, and
        # network weight initialisation is seeded by `torch.manual_seed`
        # below -- so the call was dead and has been removed.
        torch.manual_seed(self.seed)
        self._net = self._build_network(D)

        opt = torch.optim.Adam(self._net.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)

        X_tr_t = torch.as_tensor(X_tr_n, dtype=torch.float32, device=self.device)
        y_tr_t = torch.as_tensor(y_tr,   dtype=torch.float32, device=self.device).view(-1, 1)
        w_tr_t = torch.as_tensor(w_tr,   dtype=torch.float32, device=self.device).view(-1, 1)
        if n_val:
            X_va_t = torch.as_tensor(X_va_n, dtype=torch.float32, device=self.device)
            y_va_t = torch.as_tensor(y_va,   dtype=torch.float32, device=self.device).view(-1, 1)
            w_va_t = torch.as_tensor(w_va,   dtype=torch.float32, device=self.device).view(-1, 1)

        n_train  = X_tr_t.shape[0]
        best_val = float('inf')
        best_state = None
        patience_ctr = 0

        for epoch in range(self.epochs):
            self._net.train()
            perm = torch.randperm(n_train, device=self.device)
            epoch_loss = 0.0
            for start in range(0, n_train, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb, yb, wb = X_tr_t[idx], y_tr_t[idx], w_tr_t[idx]
                opt.zero_grad()
                pred = self._net(xb)
                loss = (wb * (pred - yb) ** 2).mean()
                loss.backward()
                opt.step()
                # .detach() before the scalar conversion: numerically
                # identical, but float() on a requires_grad tensor makes
                # torch emit a UserWarning about the silent graph exit.
                epoch_loss += float(loss.detach()) * xb.shape[0]
            epoch_loss /= n_train

            if n_val:
                self._net.eval()
                with torch.no_grad():
                    pred_va = self._net(X_va_t)
                    val_loss = float((w_va_t * (pred_va - y_va_t) ** 2).mean())
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

        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def fit_csv(self,
                csv_path     : str,
                weight_csv   : Optional[str] = None,
                val_fraction : float = 0.2,
                val_seed     : int   = 0,
                ) -> 'NNModel':
        """Convenience wrapper: load CSV (load_mlip_csv format) then fit()."""
        targets, labels, weights, X = load_mlip_csv(csv_path, verbose=self.verbose)
        return self.fit(targets, labels, X,
                        weights=weights, weight_csv=weight_csv,
                        val_fraction=val_fraction, val_seed=val_seed)

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
        targets, labels, weights, X = load_mlip_csv(csv_path, verbose=False)
        y_pred = self.predict(X)
        sc = ELMModel._score(targets, y_pred, labels, label)
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
        }, p)
        if self.verbose:
            print(f"  NNModel saved -> {p}")

    @classmethod
    def load(cls, path: str, verbose: bool = False) -> 'NNModel':
        """Load a saved NNModel from a .pt file."""
        if not HAS_TORCH:
            raise ImportError("NNModel.load requires PyTorch, which is not installed.")
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
            device        = 'cpu',
            verbose       = verbose,
        )
        obj._D      = int(ckpt['D'])
        obj._net    = obj._build_network(obj._D)
        obj._net.load_state_dict(ckpt['state_dict'])
        obj._X_mean = ckpt['X_mean']
        obj._X_std  = ckpt['X_std']
        obj._fitted = True
        return obj
