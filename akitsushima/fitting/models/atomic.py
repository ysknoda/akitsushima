"""Atom-centred (Behler-Parrinello) energy models, ADR-0005.

    E = sum_i eps_{s(i)}(G_i)

with G_i the ACSF vector of atom i (in-package ACSF, ADR-0004 decision 4)
and eps_s one function per species s (J. Behler and M. Parrinello, PRL 98,
146401 (2007)).  Forces F_j = -dE/dr_j and the stress
sigma = -(1/V) dE/dT (compressive-positive, ADR-0001) follow from the
per-neighbour-entry Jacobians of :class:`~..descriptors.acsf.AtomicJacobian`.

This module holds what the atom-centred models share -- the training rows
of a list of structures, in the order of the tabular table
(``build_table``: per structure one E row [eV/atom], 3N F rows, 6 S rows),
and the ADR-0004 weighting w_cat * w_unit^2 * w_user -- and the ELM form:

    eps_s(G) = beta_s . g(W_s x + b_s) + gamma_s . x + beta0_s,
    x = (G - mean_s) / std_s

E, F and S are all linear in (beta, gamma, beta0), so every row goes into
one weighted least-squares system, solved in closed form through the
normal equations.  Each species is a unit of the tabular ELM class
(ELMModel, or QuantizedELMModel for AtomicQELMModel), so activations,
the direct link and the quantisation come by inheritance.
"""

import json
import time as _time

import numpy as np

from .csv_io import load_weights_csv, structure_grouped_split
from .elm import ELMModel, pca_whitening
from .quantized_elm import QuantizedELMModel

__all__ = ["AtomicData", "AtomicELMModel", "AtomicQELMModel"]

_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))


class AtomicData:
    """
    Structures (dicts with ``lattice``, ``positions``, ``atom_types``,
    ``ref_energy`` [eV/atom], optional ``ref_forces`` (N, 3) and
    ``ref_stress`` (3, 3) -- the classical ``fit`` loader's dicts) with
    their ACSF Jacobians, computed once and kept (``cache=True``).

    Row layout (``rows``): as :func:`~..descriptors.acsf.build_table`.
    """

    def __init__(
        self, structures, descriptor, cache: bool = True, verbose: bool = False
    ):
        self.structures = list(structures)
        self.descriptor = descriptor
        self.cache = cache
        self._jac = {}
        self.n_atoms = np.array([len(s["atom_types"]) for s in self.structures])
        t0 = _time.perf_counter()
        if cache:
            for k in range(len(self.structures)):
                self.jac(k)
                if verbose and (k + 1) % 200 == 0:
                    print(f"  descriptors: {k + 1}/{len(self.structures)} structures")
        self.t_descriptor = _time.perf_counter() - t0

    def __len__(self):
        return len(self.structures)

    def jac(self, k):
        if k in self._jac:
            return self._jac[k]
        s = self.structures[k]
        J = self.descriptor.atomic_jacobian(
            s["lattice"], s["positions"], s["atom_types"]
        )
        if self.cache:
            self._jac[k] = J
        return J

    def rows(self, r0: float, eps0: float):
        """(targets, labels, w_unit, struct_id, row_ptr) -- unit weights
        N (E), r0 (F) and eps0 * V (S), as csv_io / build_table."""
        y, lab, w, sid, ptr = [], [], [], [], [0]
        for k, s in enumerate(self.structures):
            N = len(s["atom_types"])
            V = abs(np.linalg.det(np.asarray(s["lattice"], dtype=np.float64)))
            y.append(float(s["ref_energy"]))
            lab.append("E")
            w.append(float(N))
            if s.get("ref_forces") is not None:
                y += list(np.asarray(s["ref_forces"], dtype=np.float64).ravel())
                lab += ["F"] * (3 * N)
                w += [r0] * (3 * N)
            if s.get("ref_stress") is not None:
                S = np.asarray(s["ref_stress"], dtype=np.float64)
                y += [S[a, b] for a, b in _VOIGT]
                lab += ["S"] * 6
                w += [eps0 * V] * 6
            sid += [k] * (len(y) - ptr[-1])
            ptr.append(len(y))
        return (np.array(y), np.array(lab), np.array(w), np.array(sid), np.array(ptr))


def _row_weights(model, labels, w_unit, w_user):
    """w_cat * w_unit^2 * w_user (ADR-0004, decision 2)."""
    wmap = {"E": model.w_energy, "F": model.w_forces, "S": model.w_stress}
    w_cat = np.array([wmap[c] for c in labels], dtype=np.float64)
    return w_cat * w_unit**2 * w_user


def _prepare(model, structures, descriptor, weight_csv, val_fraction, val_seed):
    """Common part of the atom-centred fits: data, rows, weights, split,
    per-species descriptor statistics over the training atoms."""
    data = (
        structures
        if isinstance(structures, AtomicData)
        else AtomicData(structures, descriptor, verbose=model.verbose)
    )
    desc = data.descriptor
    y, lab, w_unit, sid, ptr = data.rows(model.r0, model.eps0)
    w_user = (
        load_weights_csv(weight_csv, n_total=len(y), verbose=model.verbose)
        if weight_csv is not None
        else np.ones(len(y))
    )
    w_row = _row_weights(model, lab, w_unit, w_user)
    idx_tr, idx_val = structure_grouped_split(
        lab,
        val_fraction=val_fraction,
        val_seed=val_seed,
        caller=type(model).__name__ + ".fit",
    )
    tr = np.unique(sid[idx_tr])
    val = np.unique(sid[idx_val])
    n_sp = len(desc.species)
    Dc = desc.n_per_atom
    mean, std = np.zeros((n_sp, Dc)), np.ones((n_sp, Dc))
    Gs = [data.jac(k) for k in tr]
    kw = getattr(model, "unit_kw", {})
    # default pca (ADR-0013), zscore with w_correlation radial
    method = kw.get("standardize") or (
        "zscore" if kw.get("w_correlation") == "radial" else "pca"
    )
    pca = method == "pca"
    P, G_train = [], []
    for s in range(n_sp):
        Gsp = np.concatenate([J.G[J.sp == s] for J in Gs])
        if len(Gsp):
            mean[s] = Gsp.mean(axis=0)
            sd = Gsp.std(axis=0)
            std[s] = np.where(sd < 1e-12, 1.0, sd)
        # ADR-0013: per-species whitening map; training atoms kept for
        # bias_init 'data' (dropped after the fit)
        P.append(
            pca_whitening(
                (Gsp - mean[s]) / std[s], kw.get("pca_eps", 1e-3), kw.get("pca_drop", 0.0)
            )
            if pca and len(Gsp)
            else None
        )
        G_train.append(Gsp)
    model.species = list(desc.species)
    model.descriptor_config = desc.to_json()
    model._X_mean, model._X_std = mean, std
    model._X_P = P
    model._G_train = G_train
    return data, y, lab, w_row, sid, ptr, tr, val


def _scores(model, data, y, lab, ptr, structs, name):
    rows = np.concatenate([np.arange(ptr[k], ptr[k + 1]) for k in structs])
    pred = np.concatenate([model._predict_structure(data.jac(k)) for k in structs])
    return ELMModel._score(y[rows], pred, lab[rows], name)


class AtomicELMModel:
    """
    Atom-centred ELM: eps_s(G) = beta_s . g(W_s x + b_s) (+ gamma_s . x)
    + beta0_s per species, W_s, b_s fixed random (drawn by the species
    unit, seed + s), (beta, gamma, beta0) of every species from one
    closed-form weighted least-squares solve over the E, F and S rows.

    Parameters follow ELMModel (n_hidden, activation, reg, reg_search,
    seed, direct_link, category weights, r0, eps0); extra keyword
    arguments go to the species units (e.g. rff_gamma, w_init_scale).
    """

    atom_centred = True
    row_independent = False
    _unit_cls = ELMModel

    def __init__(
        self,
        n_hidden: int = 400,
        activation: str = "tanh",
        w_energy: float = 1.0,
        w_forces: float = 0.1,
        w_stress: float = 10.0,
        r0: float = 1.0,
        eps0: float = 0.01,
        reg: float = 1e-6,
        reg_search: list | None = None,
        seed: int = 42,
        direct_link: bool = True,
        verbose: bool = True,
        **unit_kw,
    ):
        self.n_hidden = int(n_hidden)
        self.activation = str(activation).lower()
        self.w_energy, self.w_forces, self.w_stress = (
            float(w_energy),
            float(w_forces),
            float(w_stress),
        )
        self.r0, self.eps0 = float(r0), float(eps0)
        self.reg = float(reg)
        self.reg_search = [float(r) for r in reg_search] if reg_search else None
        self.seed = int(seed)
        self.direct_link = bool(direct_link)
        self.verbose = bool(verbose)
        self.unit_kw = dict(unit_kw)
        self.species: list = []
        self.descriptor_config = ""
        self.units: list = []
        self._X_P: list = []
        # (scale factor, correlation length) of the units being drawn
        self._trial = (1.0, float(self.unit_kw.get("w_corr_length", 0.2)))
        self._fitted = False

    # ------------------------------------------------------------------
    def _make_unit(self, s: int, Dc: int):
        u = self._unit_cls(
            n_hidden=self.n_hidden,
            activation=self.activation,
            seed=self.seed + s,
            direct_link=self.direct_link,
            reg=self.reg,
            verbose=False,
            **self.unit_kw,
        )
        u._X_mean, u._X_std, u._D = self._X_mean[s], self._X_std[s], Dc
        u._P = self._X_P[s] if self._X_P else None
        u.descriptor_config = self.descriptor_config
        if u.w_correlation == "radial":
            u._groups = u._groups_from_config(per_atom=True)
            if not u._groups:
                raise ValueError(
                    "AtomicELMModel: w_correlation needs G2 "
                    "channels that differ in r_s (ADR-0013)."
                )
        u._w_scale_factor, u.w_corr_length = self._trial
        G_s = getattr(self, "_G_train", None)
        X_ref = (
            u._normalise(G_s[s]) if u.bias_init == "data" and G_s is not None else None
        )
        u._init_hidden(Dc if u._P is None else u._P.shape[1], X_ref)
        return u

    @property
    def _off(self) -> np.ndarray:
        """Column offsets of the species blocks (the blocks differ in
        width when PCA whitening keeps different numbers of directions)."""
        return np.concatenate([[0], np.cumsum([u._n_out() for u in self.units])])

    def _unit_rows(self, u, J, mask_atoms, mask_ent):
        """Contributions of one species' atoms: E-row sum over the atoms,
        and per-entry derivative rows R (M_s, 3, n_out_s) of
        d eps/d(v_m) with respect to the output weights."""
        xn = u._normalise(J.G[mask_atoms])
        Z = u._preact(xn)
        e_row = np.concatenate(
            [
                u._hidden_from_Z(Z).sum(axis=0),
                xn.sum(axis=0) if self.direct_link else [],
                [float(len(xn))],
            ]
        )
        n_ent = int(mask_ent.sum())
        R = np.zeros((n_ent, 3, u._n_out()))
        if n_ent:
            loc = np.cumsum(mask_atoms) - 1  # atom -> row of Z
            Zm = np.repeat(Z[loc[J.centre[mask_ent]]], 3, axis=0)
            P = u._scale_dir(J.dGdv[mask_ent]).reshape(3 * n_ent, -1)
            nh = u._n_hid()
            R.reshape(3 * n_ent, -1)[:, :nh] = u._hidden_dir(Zm, u._xw(P))
            if self.direct_link:
                R.reshape(3 * n_ent, -1)[:, nh : nh + P.shape[1]] = P
        return e_row, R

    def _design(self, J):
        """Design rows of one structure: [E, F (3N), S (6)] x
        (n_species * n_out_s).  The E row is per atom (E/N), F rows are
        -dE/dr, S rows -(1/V) dE/dT (Voigt)."""
        from scipy.sparse import csr_matrix

        N = len(J.sp)
        off = self._off
        A = np.zeros((1 + 3 * N + 6, off[-1]))
        cen = J.centre
        for s, u in enumerate(self.units):
            ma = J.sp == s
            if not ma.any():
                continue
            me = ma[cen]
            e_row, R = self._unit_rows(u, J, ma, me)
            blk = slice(off[s], off[s + 1])
            n_o = off[s + 1] - off[s]
            A[0, blk] = e_row / N
            # F_j = -(sum_{m -> j} R_m - sum_{m of j} R_m)
            n_ent = len(R)
            inc = csr_matrix(
                (np.r_[-np.ones(n_ent), np.ones(n_ent)],
                 (np.r_[J.nbj[me], cen[me]], np.tile(np.arange(n_ent), 2))),
                shape=(N, n_ent))
            A[1:1 + 3 * N, blk] = (inc @ R.reshape(n_ent, -1)).reshape(3 * N, n_o)
            T = np.einsum("ma,mbo->abo", J.vec[me], R)
            A[1 + 3 * N:, blk] = np.array(
                [-0.5 * (T[a, b] + T[b, a]) for a, b in _VOIGT]) / J.volume
        return A

    def _struct_rows(self, J, k, y, lab, ptr):
        """Design rows matching the table rows ptr[k]:ptr[k+1] (F or S
        rows absent from the data are dropped)."""
        A = self._design(J)
        lk = lab[ptr[k] : ptr[k + 1]]
        keep = np.concatenate(
            [[True], np.full(3 * len(J.sp), "F" in lk), np.full(6, "S" in lk)]
        )
        return A[keep]

    # ------------------------------------------------------------------
    def fit(
        self,
        structures,
        descriptor=None,
        weight_csv=None,
        val_fraction: float = 0.2,
        val_seed: int = 0,
    ) -> "AtomicELMModel":
        """
        Fit on a list of structure dicts (or an :class:`AtomicData`) with
        an ACSF ``descriptor``.  Rows, weights and the structure-grouped
        validation split are those of the tabular models on the same
        structures (ADR-0005, decisions 3 and 4).
        """
        t0 = _time.perf_counter()
        data, y, lab, w_row, _, ptr, tr, val = _prepare(
            self, structures, descriptor, weight_csv, val_fraction, val_seed
        )
        Dc = data.descriptor.n_per_atom
        t1 = _time.perf_counter()

        def _normal(structs):
            m = self._off[-1]
            G, r, tt = np.zeros((m, m)), np.zeros(m), 0.0
            for k in structs:
                J = data.jac(k)
                A = self._struct_rows(J, k, y, lab, ptr)
                sl = slice(ptr[k], ptr[k + 1])
                w, t = w_row[sl], y[sl]
                Aw = A * w[:, None]
                G += A.T @ Aw
                r += Aw.T @ t
                tt += float(np.sum(w * t * t))
            return 0.5 * (G + G.T), r, tt

        kw = self.unit_kw
        regs = self.reg_search or [self.reg]
        factors = kw.get("w_init_scale_search") or [1.0]
        if factors is True:
            factors = [0.5, 1.0, 2.0, 4.0, 8.0]
        lengths = (
            kw.get("w_corr_length_search")
            if kw.get("w_correlation") == "radial" and kw.get("w_corr_length_search")
            else [kw.get("w_corr_length", 0.2)]
        )
        cand = [(float(f), float(ell)) for f in factors for ell in lengths]
        if (len(cand) > 1 or len(regs) > 1) and not len(val):
            if len(cand) > 1:
                raise ValueError(
                    "AtomicELMModel: w_init_scale_search / w_corr_length_search "
                    "need a validation split."
                )
            regs = [self.reg]
        need_val = bool(len(val)) and (len(cand) > 1 or len(regs) > 1)

        def _trial(fl):
            """Units for one (scale factor, correlation length); the
            normal equations; the reg search."""
            import scipy.linalg as sla

            self._trial = fl
            self.units = [self._make_unit(s, Dc) for s in range(len(self.species))]
            off = self._off
            m = off[-1]
            G, rhs, _ = _normal(tr)
            nh = self.units[0]._n_hid()
            hid = np.zeros(m, bool)
            for s in range(len(self.units)):
                hid[off[s] : off[s] + nh] = True
            jitter = 1e-12 * float(np.mean(np.diag(G)))

            def _solve(reg):
                pen = np.where(hid, reg, jitter if self.direct_link else reg)
                Gp = G.copy()
                Gp[np.diag_indices(m)] += pen
                try:
                    return sla.cho_solve(sla.cho_factor(Gp, lower=True), rhs)
                except np.linalg.LinAlgError:
                    return sla.lstsq(Gp, rhs)[0]

            if need_val:
                Gv, rv, ttv = _normal(val)
            res = []
            for reg in regs:
                c = _solve(reg)
                vl = float(c @ Gv @ c - 2 * c @ rv + ttv) if need_val else 0.0
                res.append((reg, vl, c))
            return min(res, key=lambda x: x[1]), res

        self._trial = cand[0]
        search = []
        for fl in cand:
            best, res = _trial(fl)
            search.append((fl, best, res))
            if self.verbose and (len(cand) > 1 or len(res) > 1):
                for reg, vl, _ in res:
                    print(
                        f"  scale x{fl[0]:g}  l={fl[1]:g}  reg {reg:10.3g}: "
                        f"weighted val loss {vl:.6e}"
                    )
        fl, best, res = min(search, key=lambda x: x[1][1])
        if len(cand) > 1:
            best, res = _trial(fl)  # rebuild the winning units
            self.hyper_search_results_ = [
                (f, ell, b[0], b[1]) for (f, ell), b, _ in search
            ]
        t_build = _time.perf_counter() - t1
        self.reg = float(best[0])
        if len(res) > 1:
            self.reg_search_results_ = [(r, vl) for r, vl, _ in res]
        coef = best[2]
        off = self._off
        for u in self.units:
            u.reg = self.reg
        self._G_train = None
        for s, u in enumerate(self.units):
            u._beta, u._gamma, u._beta0 = u._split_coef(coef[off[s] : off[s + 1]])
            u._fitted = True
        self._fitted = True
        self.fit_time_ = {
            "descriptors": data.t_descriptor,
            "rows+normal": t_build,
            "total": _time.perf_counter() - t0,
        }
        self._train_sc = _scores(self, data, y, lab, ptr, tr, "train")
        self._val_sc = (
            _scores(self, data, y, lab, ptr, val, "val") if len(val) else None
        )
        if self.verbose:
            print(
                f"\nAtomic ELM  species={self.species}  Dc={Dc}  "
                f"N_h={self.n_hidden}  reg={self.reg:g}  "
                f"({self.fit_time_['total']:.1f} s)"
            )
            ELMModel._print_scores(self._train_sc, self._val_sc)
        return self

    # ------------------------------------------------------------------
    def atomic_energy_and_gradient(self, G, sp):
        """eps_i (N,) and g_i = d eps_i / dG_i (N, Dc)."""
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__}: call fit() first.")
        eps = np.zeros(len(sp))
        g = np.zeros_like(G)
        for s, u in enumerate(self.units):
            ma = sp == s
            if not ma.any():
                continue
            xn = u._normalise(G[ma])
            Z = u._preact(xn)
            e = u._hidden_from_Z(Z) @ u._beta + u._beta0
            if u.activation == "rff":
                n = Z.shape[1]
                b = u._beta
                up = (-np.sin(Z) * b[:n] + np.cos(Z) * b[n:]) / np.sqrt(n)
            else:
                up = u._dact(Z) * u._beta
            gn = u._uwT(up)
            if self.direct_link:
                e = e + xn @ u._gamma
                gn = gn + u._gamma
            eps[ma], g[ma] = e, u._to_raw(gn)
        return eps, g

    def _predict_structure(self, J):
        """[E/N, F (3N), S (Voigt 6)] of one structure."""
        eps, g = self.atomic_energy_and_gradient(J.G, J.sp)
        F, S = J.forces_and_stress(g)
        return np.concatenate(
            [[eps.sum() / len(eps)], F.ravel(), [S[a, b] for a, b in _VOIGT]]
        )

    def predict_structures(self, structures, descriptor=None):
        """Rows [E/N, F, S] of every structure, in the table order (F / S
        rows only where the structure has reference forces / stress)."""
        data = (
            structures
            if isinstance(structures, AtomicData)
            else AtomicData(
                structures,
                descriptor or _descriptor(self.descriptor_config),
                cache=False,
            )
        )
        out = []
        for k, s in enumerate(data.structures):
            p = self._predict_structure(data.jac(k))
            N = len(s["atom_types"])
            keep = np.concatenate(
                [
                    [True],
                    np.full(3 * N, s.get("ref_forces") is not None),
                    np.full(6, s.get("ref_stress") is not None),
                ]
            )
            out.append(p[keep])
        return np.concatenate(out)

    def score_structures(self, structures, descriptor=None, label="test"):
        data = (
            structures
            if isinstance(structures, AtomicData)
            else AtomicData(
                structures,
                descriptor or _descriptor(self.descriptor_config),
                cache=False,
            )
        )
        y, lab, *_ = data.rows(self.r0, self.eps0)
        return ELMModel._score(y, self.predict_structures(data), lab, label)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__}: model has not been fitted.")
        arr = {}
        for s, u in enumerate(self.units):
            arr[f"W_{s}"] = u._W
            arr[f"b_{s}"] = u._b if u._b is not None else np.zeros(0)
            arr[f"beta_{s}"] = u._beta
            arr[f"gamma_{s}"] = u._gamma if u._gamma is not None else np.zeros(0)
            arr[f"beta0_{s}"] = np.array(u._beta0)
            arr[f"X_P_{s}"] = u._P if u._P is not None else np.zeros((0, 0))
            arr[f"options_{s}"] = np.array(json.dumps(u._options()))
        hyper = dict(
            n_hidden=self.n_hidden,
            activation=self.activation,
            w_energy=self.w_energy,
            w_forces=self.w_forces,
            w_stress=self.w_stress,
            r0=self.r0,
            eps0=self.eps0,
            reg=self.reg,
            seed=self.seed,
            direct_link=self.direct_link,
            unit_kw=self.unit_kw,
        )
        np.savez(
            path,
            model_class=np.array(type(self).__name__),
            atom_centred=np.array(True),
            species=np.array(self.species),
            X_mean=self._X_mean,
            X_std=self._X_std,
            hyper=np.array(json.dumps(hyper)),
            descriptor_config=np.array(self.descriptor_config),
            **arr,
        )

    @classmethod
    def load(cls, path: str, verbose: bool = False):
        p = path if path.endswith(".npz") else path + ".npz"
        z = np.load(p)
        hyper = json.loads(str(z["hyper"]))
        unit_kw = hyper.pop("unit_kw", {})
        if "n_bits" in unit_kw:  # files written before ADR-0013
            unit_kw["max_exponent"] = unit_kw.pop("n_bits")
        obj = cls(verbose=verbose, **hyper, **unit_kw)
        obj.species = [str(x) for x in z["species"]]
        obj._X_mean, obj._X_std = z["X_mean"], z["X_std"]
        obj.descriptor_config = str(z["descriptor_config"])
        Dc = obj._X_mean.shape[1]
        obj.units = []
        for s in range(len(obj.species)):
            u = obj._unit_cls(
                n_hidden=obj.n_hidden,
                activation=obj.activation,
                seed=obj.seed + s,
                direct_link=obj.direct_link,
                verbose=False,
                **obj.unit_kw,
            )
            u._X_mean, u._X_std, u._D = obj._X_mean[s], obj._X_std[s], Dc
            u._W = z[f"W_{s}"]
            b = z[f"b_{s}"]
            u._b = b if b.size else None
            u._beta = z[f"beta_{s}"]
            gm = z[f"gamma_{s}"]
            u._gamma = gm if gm.size else None
            u._beta0 = float(z[f"beta0_{s}"])
            if f"X_P_{s}" in z and z[f"X_P_{s}"].size:
                u._P = z[f"X_P_{s}"]
            if f"options_{s}" not in z:  # saved before ADR-0013: z-score
                u.standardize = "zscore"
            else:
                o = json.loads(str(z[f"options_{s}"]))
                u._w_scale_factor = float(o.pop("w_scale_factor", 1.0))
                for k, v in o.items():
                    setattr(u, k, v)
            u._fitted = True
            obj.units.append(u)
        obj._X_P = [u._P for u in obj.units]
        obj._fitted = True
        return obj


class AtomicQELMModel(AtomicELMModel):
    """Atom-centred ELM whose species units are QuantizedELMModel
    (quantised hidden weights); max_exponent, p_zero, use_sparse_w and the
    other QELM options (levels, n_levels, matmul) are passed to the
    units."""

    _unit_cls = QuantizedELMModel

    def __init__(
        self,
        *args,
        max_exponent: int = 3,
        p_zero: float = 0.5,
        use_sparse_w: bool = True,
        **kw,
    ):
        if "n_bits" in kw:
            import warnings

            warnings.warn(
                "AtomicQELMModel: n_bits is deprecated, use max_exponent "
                "(ADR-0013).",
                DeprecationWarning,
                stacklevel=2,
            )
            max_exponent = kw.pop("n_bits")
        super().__init__(
            *args,
            max_exponent=max_exponent,
            p_zero=p_zero,
            use_sparse_w=use_sparse_w,
            **kw,
        )


def _descriptor(cfg):
    from ..descriptors import descriptor_from_config

    return descriptor_from_config(cfg)
