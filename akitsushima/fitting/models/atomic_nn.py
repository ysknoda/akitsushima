"""Atom-centred neural-network potential (Behler-Parrinello high-dimensional
NN, PRL 98, 146401 (2007)), ADR-0005.

E = sum_i eps_{s(i)}(x_i), x_i = (G_i - mean_s) / std_s, one MLP per
species (with NNModel's options: hidden_layers, activation, direct_link).
With ``standardize: pca`` each species' MLP sees x_i P_s, P_s the
per-species whitening map that AtomicELMModel uses (ADR-0013; for the NN
since 0.2.0a12), held as a fixed first layer, so everything outside the
networks stays in z-score space.
Forces and stress come from g_i = d eps_i / dx_i (torch autograd,
create_graph during training) contracted with the stored ACSF Jacobians:

    F_j = -sum_i g_i . dx_i/dr_j        sigma = -(1/V) sum_i g_i . dx_i/dT

Training rows, weights and the validation split are those of
:mod:`.atomic` (and therefore of the tabular models on the same
structures).
"""

import time as _time

import numpy as np

from ...utils.optional_torch import HAS_TORCH, _torch_nn, torch
from .atomic import _VOIGT, _prepare
from .elm import ELMModel
from .nn import NNModel

__all__ = ["AtomicNNModel"]


class AtomicNNModel:
    """
    Parameters as NNModel (hidden_layers, activation, direct_link,
    learning_rate, epochs, batch_structures, weight_decay,
    early_stopping_patience, optimizer 'adam' only, deterministic, seed,
    device, category weights, r0, eps0) and the input standardisation of
    the atom-centred ELM: standardize ('zscore', the default, or 'pca'),
    pca_eps, pca_drop, min_feature_std.
    """

    atom_centred = True
    row_independent = False

    def __init__(
        self,
        hidden_layers=None,
        activation: str = "softplus",
        w_energy: float = 1.0,
        w_forces: float = 0.1,
        w_stress: float = 10.0,
        r0: float = 1.0,
        eps0: float = 0.01,
        learning_rate: float = 1.0e-3,
        epochs: int = 1500,
        weight_decay: float = 1.0e-6,
        early_stopping_patience: int = 150,
        batch_structures: int = 30,
        device: str = "auto",
        seed: int = 42,
        direct_link: bool = True,
        deterministic: bool = False,
        verbose: bool = True,
        standardize: str = "zscore",
        pca_eps: float = 1e-3,
        pca_drop: float = 0.0,
        min_feature_std: float = 1e-12,
    ):
        if not HAS_TORCH:
            raise ImportError("AtomicNNModel requires PyTorch, which is not installed. "
                              'Install it with: pip install "akitsushima[torch]" (or pip install torch).')
        standardize = str(standardize).lower()
        if standardize not in ("zscore", "pca"):
            raise ValueError(
                f"AtomicNNModel: standardize must be 'zscore' or 'pca', got {standardize!r}."
            )
        # read by atomic._prepare, which builds the per-species maps
        self.unit_kw = dict(standardize=standardize, pca_eps=float(pca_eps),
                            pca_drop=float(pca_drop),
                            min_feature_std=float(min_feature_std))
        self._X_P = None
        self.hidden_layers = list(hidden_layers) if hidden_layers else [64, 64]
        self.activation = str(activation).lower()
        self.w_energy, self.w_forces, self.w_stress = (
            float(w_energy),
            float(w_forces),
            float(w_stress),
        )
        self.r0, self.eps0 = float(r0), float(eps0)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.weight_decay = float(weight_decay)
        self.early_stopping_patience = int(early_stopping_patience)
        self.batch_structures = int(batch_structures)
        self.seed = int(seed)
        self.direct_link = bool(direct_link)
        self.deterministic = bool(deterministic)
        self.verbose = bool(verbose)
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        self.species: list = []
        self.descriptor_config = ""
        # physics-informed parts (ADR-0014): pair baseline, isolated atom
        self.physics_config = ""
        self._nets = None
        self._fitted = False

    # ------------------------------------------------------------------
    def _build(self, n_sp, Dc, P=None):
        # One NNModel per species builds the MLP (same layers, activation
        # and direct link as the tabular NN).  With a whitening map P_s the
        # MLP takes x P_s through a fixed (untrained) first layer.
        proto = NNModel(
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            direct_link=self.direct_link,
            device=self.device,
            verbose=False,
        )
        nets = []
        for s in range(n_sp):
            Ps = None if P is None else P[s]
            if Ps is None:
                nets.append(proto._build_network(Dc))
            else:
                nets.append(_Projected(Ps, proto._build_network(Ps.shape[1])))
        return _torch_nn.ModuleList(nets).to(self.device)

    @staticmethod
    def _last_linear(net):
        """The output layer of one species' network (its bias is the
        energy offset)."""
        net = getattr(net, "core", net)
        return (net.mlp if hasattr(net, "mlp") else net)[-1]

    def _init_energy_offsets(self, data, y, ptr, tr):
        """Start each species' output bias at its least-squares reference
        energy (total energies against composition), so training does not
        spend its first epochs learning a constant of ~ -4 eV/atom."""
        n_sp = len(self.species)
        C = np.array([np.bincount(data.jac(k).sp, minlength=n_sp) for k in tr],
                     dtype=np.float64)
        E = y[ptr[tr]] * data.n_atoms[tr]
        e_ref = np.linalg.lstsq(C, E, rcond=None)[0]
        with torch.no_grad():
            for s, net in enumerate(self._nets):
                self._last_linear(net).bias.fill_(float(e_ref[s]))

    def _eps(self, xn, sp):
        """Per-atom energies of normalised descriptors xn (A, Dc).  While
        training with isolated_atom_energy: net(x) - net(x0) + E_atom, so
        eps_s(G = 0) = E_atom,s holds exactly (ADR-0014)."""
        eps = torch.zeros(len(xn), dtype=xn.dtype, device=xn.device)
        iso = getattr(self, "_iso_t", None)
        for s, net in enumerate(self._nets):
            ma = sp == s
            if bool(ma.any()):
                e = net(xn[ma]).view(-1)
                if iso is not None:
                    e = e - net(iso[0][s : s + 1]).view(-1) + iso[1][s]
                eps = eps.index_put((ma.nonzero().view(-1),), e)
        return eps

    def _iso_setup(self):
        """Tensors (x0 per species, E_atom per species) of the isolated-
        atom limit, or None."""
        from ..physics import physics_from_config

        _, iso = physics_from_config(self.physics_config)
        if not iso:
            return None
        missing = [s for s in self.species if s not in iso]
        if missing:
            raise ValueError(
                f"AtomicNNModel: isolated_atom_energy has no entry for {missing}."
            )
        x0 = -np.asarray(self._X_mean) / np.asarray(self._X_std)
        f32, dev = torch.float32, self.device
        return (
            torch.as_tensor(x0, dtype=f32, device=dev),
            torch.as_tensor([float(iso[e]) for e in self.species], dtype=f32, device=dev),
        )

    def _iso_fold(self):
        """Move the constraint into the output bias: bias += E_atom -
        net(x0), after which the plain network satisfies it."""
        x0, ea = self._iso_t
        self._iso_t = None
        with torch.no_grad():
            for s, net in enumerate(self._nets):
                self._last_linear(net).bias += ea[s] - net(x0[s : s + 1]).view(-1)[0]

    # ------------------------------------------------------------------
    def _tensors(self, data, structs):
        """Stack the Jacobians of ``structs`` into device tensors."""
        f32, dev = torch.float32, self.device
        xs, sps, P, cen, nbj, vec, ent_st = [], [], [], [], [], [], []
        a0, e0 = [0], [0]
        for q, k in enumerate(structs):
            J = data.jac(k)
            sp = J.sp
            xs.append((J.G - self._X_mean[sp]) / self._X_std[sp])
            sps.append(sp)
            c = J.centre
            # x-space Jacobian: dG/dv / std of the centre's species
            P.append(J.dGdv / self._X_std[sp[c]][:, None, :])
            cen.append(c + a0[-1])
            nbj.append(J.nbj + a0[-1])
            vec.append(J.vec)
            ent_st.append(np.full(len(c), q))
            a0.append(a0[-1] + len(sp))
            e0.append(e0[-1] + len(c))
        cat = np.concatenate
        return dict(
            x=torch.as_tensor(cat(xs), dtype=f32, device=dev),
            sp=torch.as_tensor(cat(sps), device=dev),
            P=torch.as_tensor(cat(P), dtype=f32, device=dev),
            cen=torch.as_tensor(cat(cen), device=dev),
            nbj=torch.as_tensor(cat(nbj), device=dev),
            vec=torch.as_tensor(cat(vec), dtype=f32, device=dev),
            ent_st=torch.as_tensor(cat(ent_st), device=dev),
            a_ptr=np.array(a0),
            e_ptr=np.array(e0),
            atom_st=torch.as_tensor(
                np.repeat(np.arange(len(structs)), np.diff(a0)), device=dev
            ),
            nat=torch.as_tensor(np.diff(a0), dtype=f32, device=dev),
            vol=torch.as_tensor(
                [data.jac(k).volume for k in structs], dtype=f32, device=dev
            ),
        )

    def _efs(self, T, st, create_graph):
        """E/N (S,), F (A, 3), sigma (S, 3, 3) of the structures st
        (positions in T) -- atoms and entries of those structures only."""
        dev = self.device
        a_idx = torch.as_tensor(
            np.concatenate([np.arange(T["a_ptr"][k], T["a_ptr"][k + 1]) for k in st]),
            device=dev,
        )
        e_idx = torch.as_tensor(
            np.concatenate([np.arange(T["e_ptr"][k], T["e_ptr"][k + 1]) for k in st]),
            device=dev,
        )
        amap = torch.full((len(T["x"]),), -1, dtype=torch.long, device=dev)
        amap[a_idx] = torch.arange(len(a_idx), device=dev)
        smap = torch.full((len(T["nat"]),), -1, dtype=torch.long, device=dev)
        st_t = torch.as_tensor(np.asarray(st), device=dev)
        smap[st_t] = torch.arange(len(st), device=dev)
        x = T["x"][a_idx].detach().requires_grad_(True)
        eps = self._eps(x, T["sp"][a_idx])
        g = torch.autograd.grad(eps.sum(), x, create_graph=create_graph)[0]
        cen = amap[T["cen"][e_idx]]
        nbj = amap[T["nbj"][e_idx]]
        c = torch.einsum("mad,md->ma", T["P"][e_idx], g[cen])
        F = torch.zeros(len(a_idx), 3, dtype=c.dtype, device=dev)
        F = F.index_add(0, nbj, -c).index_add(0, cen, c)
        ast = smap[T["atom_st"][a_idx]]
        E = torch.zeros(len(st), dtype=eps.dtype, device=dev).index_add(0, ast, eps)
        W = torch.zeros(len(st), 9, dtype=c.dtype, device=dev).index_add(
            0,
            smap[T["ent_st"][e_idx]],
            (T["vec"][e_idx][:, :, None] * c[:, None, :]).reshape(-1, 9),
        )
        S = -W.view(-1, 3, 3) / T["vol"][st_t][:, None, None]
        return E / T["nat"][st_t], F, 0.5 * (S + S.transpose(1, 2))

    def _rows_pred(self, T, st, create_graph):
        """Predictions in table-row order for structures st (every
        structure with F and S rows)."""
        E, F, S = self._efs(T, st, create_graph)
        a_ptr = T["a_ptr"]
        out, off = [], 0
        vo = torch.as_tensor([3 * a + b for a, b in _VOIGT], device=self.device)
        for q, k in enumerate(st):
            n = int(a_ptr[k + 1] - a_ptr[k])
            out += [E[q : q + 1], F[off : off + n].reshape(-1), S[q].reshape(-1)[vo]]
            off += n
        return torch.cat(out)

    # ------------------------------------------------------------------
    def fit(
        self,
        structures,
        descriptor=None,
        weight_csv=None,
        val_fraction: float = 0.2,
        val_seed: int = 0,
    ) -> "AtomicNNModel":
        if not self.deterministic:
            return self._fit(structures, descriptor, weight_csv, val_fraction, val_seed)
        n_thr = torch.get_num_threads()
        was = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        try:
            return self._fit(structures, descriptor, weight_csv, val_fraction, val_seed)
        finally:
            torch.use_deterministic_algorithms(was)
            torch.set_num_threads(n_thr)

    def _fit(self, structures, descriptor, weight_csv, val_fraction, val_seed):
        t0 = _time.perf_counter()
        data, y, lab, w_row, _, ptr, tr, val = _prepare(
            self, structures, descriptor, weight_csv, val_fraction, val_seed
        )
        for k in range(len(data)):
            nk = ptr[k + 1] - ptr[k]
            if nk != 1 + 3 * data.n_atoms[k] + 6:
                raise ValueError(
                    "AtomicNNModel: every structure needs reference forces "
                    f"and stress (structure {k} has {nk} rows)."
                )
        Dc = data.descriptor.n_per_atom
        torch.manual_seed(self.seed)
        if self.unit_kw["standardize"] != "pca":
            self._X_P = None
        self._nets = self._build(len(self.species), Dc, self._X_P)
        if self.verbose and self._X_P is not None:
            dims = [Dc if Ps is None else Ps.shape[1] for Ps in self._X_P]
            print(f"  PCA whitening per species: {Dc} -> {dims} input directions")
        self._init_energy_offsets(data, y, ptr, tr)
        self._iso_t = self._iso_setup()
        opt = torch.optim.Adam(
            self._nets.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        T = self._tensors(data, np.arange(len(data)))
        dev, f32 = self.device, torch.float32
        y_t = torch.as_tensor(y, dtype=f32, device=dev)
        w_t = torch.as_tensor(w_row, dtype=f32, device=dev)

        def _rows(st):
            return np.concatenate([np.arange(ptr[k], ptr[k + 1]) for k in st])

        def _loss(st, create_graph):
            rows = _rows(st)
            p = self._rows_pred(T, st, create_graph)
            r_t = torch.as_tensor(rows, device=dev)
            return (w_t[r_t] * (p - y_t[r_t]) ** 2).sum(), len(rows)

        def _eval(structs):
            tot, n = 0.0, 0
            for s in range(0, len(structs), self.batch_structures):
                with torch.enable_grad():
                    l, k = _loss(structs[s : s + self.batch_structures], False)
                tot += float(l.detach())
                n += k
            return tot / max(n, 1)

        best, best_state, wait = float("inf"), None, 0
        t_fit0 = _time.perf_counter()
        for epoch in range(self.epochs):
            self._nets.train()
            order = tr[torch.randperm(len(tr)).numpy()]
            for s in range(0, len(order), self.batch_structures):
                st = np.sort(order[s : s + self.batch_structures])
                opt.zero_grad()
                l, k = _loss(st, True)
                (l / k).backward()
                opt.step()
            if len(val):
                self._nets.eval()
                vl = _eval(val)
                if vl < best - 1e-12:
                    best, wait = vl, 0
                    best_state = {
                        k: v.clone() for k, v in self._nets.state_dict().items()
                    }
                else:
                    wait += 1
                if self.verbose and epoch % 50 == 0:
                    print(f"  epoch {epoch:5d}  val_loss={vl:.6e}", flush=True)
                if (
                    self.early_stopping_patience > 0
                    and wait >= self.early_stopping_patience
                ):
                    break
        self.n_epochs_run_ = epoch + 1
        if best_state is not None:
            self._nets.load_state_dict(best_state)
        self._nets.eval()
        if self._iso_t is not None:
            self._iso_fold()
        self._fitted = True
        self.fit_time_ = {
            "descriptors": data.t_descriptor,
            "training": _time.perf_counter() - t_fit0,
            "total": _time.perf_counter() - t0,
        }

        def _sc(structs, name):
            with torch.enable_grad():
                p = torch.cat(
                    [
                        self._rows_pred(T, structs[s : s + 50], False).detach()
                        for s in range(0, len(structs), 50)
                    ]
                )
            rows = _rows(structs)
            return ELMModel._score(
                y[rows], p.cpu().numpy().astype(np.float64), lab[rows], name
            )

        self._train_sc = _sc(tr, "train")
        self._val_sc = _sc(val, "val") if len(val) else None
        if self.verbose:
            print(
                f"\nAtomic NN  species={self.species}  Dc={Dc}  "
                f"epochs={self.n_epochs_run_}  ({self.fit_time_['total']:.1f} s)"
            )
            ELMModel._print_scores(self._train_sc, self._val_sc)
        return self

    # ------------------------------------------------------------------
    def atomic_energy_and_gradient(self, G, sp):
        """eps_i (N,) and g_i = d eps_i / dG_i (N, Dc), float64 arrays."""
        if not self._fitted:
            raise RuntimeError("AtomicNNModel: call fit() first.")
        x = torch.as_tensor(
            (G - self._X_mean[sp]) / self._X_std[sp],
            dtype=torch.float32,
            device=self.device,
        )
        x.requires_grad_(True)
        self._nets.eval()
        with torch.enable_grad():
            eps = self._eps(x, torch.as_tensor(sp, device=self.device))
            g = torch.autograd.grad(eps.sum(), x)[0]
        return (
            eps.detach().cpu().numpy().astype(np.float64),
            g.detach().cpu().numpy().astype(np.float64) / self._X_std[sp],
        )

    def predict_structures(self, structures, descriptor=None):
        from .atomic import AtomicELMModel

        return AtomicELMModel.predict_structures(self, structures, descriptor)

    def score_structures(self, structures, descriptor=None, label="test"):
        from .atomic import AtomicELMModel

        return AtomicELMModel.score_structures(self, structures, descriptor, label)

    def _predict_structure(self, J):
        eps, g = self.atomic_energy_and_gradient(J.G, J.sp)
        F, S = J.forces_and_stress(g)
        return np.concatenate(
            [[eps.sum() / len(eps)], F.ravel(), [S[a, b] for a, b in _VOIGT]]
        )

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError("AtomicNNModel: model has not been fitted.")
        p = path if path.endswith(".pt") else path + ".pt"
        hyper = {
            k: getattr(self, k)
            for k in (
                "hidden_layers",
                "activation",
                "w_energy",
                "w_forces",
                "w_stress",
                "r0",
                "eps0",
                "learning_rate",
                "epochs",
                "weight_decay",
                "early_stopping_patience",
                "batch_structures",
                "seed",
                "direct_link",
            )
        }
        hyper.update(self.unit_kw)
        torch.save(
            {
                "atom_centred": True,
                "hyper": hyper,
                "species": self.species,
                "X_mean": self._X_mean,
                "X_std": self._X_std,
                "X_P": self._X_P,
                "state_dict": self._nets.state_dict(),
                "descriptor_config": self.descriptor_config,
                "physics_config": self.physics_config,
            },
            p,
        )

    @classmethod
    def load(cls, path: str, verbose: bool = False) -> "AtomicNNModel":
        p = path if path.endswith(".pt") else path + ".pt"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        obj = cls(device="cpu", verbose=verbose, **ck["hyper"])
        obj.species = list(ck["species"])
        obj._X_mean, obj._X_std = ck["X_mean"], ck["X_std"]
        obj.descriptor_config = ck["descriptor_config"]
        obj.physics_config = ck.get("physics_config", "")
        # files before 0.2.0a12 have no X_P: z-score models
        obj._X_P = ck.get("X_P")
        obj._nets = obj._build(len(obj.species), obj._X_mean.shape[1], obj._X_P)
        obj._nets.load_state_dict(ck["state_dict"])
        obj._nets.eval()
        obj._fitted = True
        return obj



if HAS_TORCH:
    class _Projected(_torch_nn.Module):
        """core(x @ P): a fixed whitening map in front of a species' MLP."""

        def __init__(self, P, core):
            super().__init__()
            self.register_buffer("P", torch.as_tensor(np.asarray(P), dtype=torch.float32))
            self.core = core

        def forward(self, x):
            return self.core(x @ self.P)
