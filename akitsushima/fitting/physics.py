"""Physics-informed parts of the MLIP (ADR-0014).

1. **Pair baseline (Delta-learning).**  A fixed short-range repulsion
   E_base = 1/2 sum_i sum_{j (images)} phi_{s_i s_j}(r_ij) s(r_ij) is
   subtracted from the training labels (E, F, S) and added back by the
   MLIP engine, so the model learns E - E_base and the total keeps a
   physical repulsion where the training data have no pairs.

       zbl        : phi(r) = K_E Z_i Z_j / r * sum_k c_k exp(-d_k r / a),
                    a = 0.46850 / (Z_i^0.23 + Z_j^0.23) A,
                    c = (0.18175, 0.50986, 0.28022, 0.02817),
                    d = (3.19980, 0.94229, 0.40290, 0.20162)
                    (Ziegler, Biersack & Littmark, The Stopping and Range
                    of Ions in Matter, Vol. 1, Pergamon 1985; constants as
                    in the LAMMPS pair_style zbl documentation)
       born_mayer : phi(r) = A exp(-r / rho), per species pair

   s(r) is a C2 switch: 1 for r <= r_inner, 0 for r >= r_outer,
   1 - 10 t^3 + 15 t^4 - 6 t^5 between (t = (r - r_inner) / (r_outer -
   r_inner)).  Unlike LAMMPS's zbl (which adds a polynomial shift to the
   energy), the baseline is multiplied by s, so it is exactly zero beyond
   r_outer.

2. **Isolated-atom limit.**  With ``isolated_atom_energy`` every atom-
   centred energy obeys eps_s(G = 0) = E_atom,s exactly: an atom with no
   neighbour inside the cutoff has the isolated-atom energy and no force.
   The ELM enforces it as an equality constraint of its least-squares
   solve; the NN by construction (eps(x) - eps(x0) + E_atom during
   training, folded into the output bias afterwards).
"""

import json

import numpy as np

from ..constants import K_E

__all__ = ["PairBaseline", "physics_from_config"]

_ZBL_C = np.array([0.18175, 0.50986, 0.28022, 0.02817])
_ZBL_D = np.array([3.19980, 0.94229, 0.40290, 0.20162])


def _switch(r, r_in, r_out):
    """(s, ds/dr): 1 below r_in, 0 above r_out, quintic smoothstep between
    (value, first and second derivatives continuous)."""
    t = np.clip((r - r_in) / (r_out - r_in), 0.0, 1.0)
    s = 1.0 - t**3 * (10.0 - 15.0 * t + 6.0 * t * t)
    ds = -30.0 * t * t * (1.0 - t) ** 2 / (r_out - r_in)
    return s, ds


def _pair_key(a, b):
    from .descriptors.acsf import ATOMIC_NUMBER

    x, y = sorted((a, b), key=lambda e: ATOMIC_NUMBER[e])
    return f"{x}-{y}"


class PairBaseline:
    """
    Fixed pair baseline: a list of terms, each
    ``{type: zbl, r_inner, r_outer}`` or
    ``{type: born_mayer, r_inner, r_outer, pairs: {"Si-Si": {A, rho}}}``
    (A in eV, rho in A; pair keys in either order).
    """

    def __init__(self, species, terms):
        from .descriptors.acsf import ATOMIC_NUMBER, sort_species

        self.species = sort_species(species)
        if isinstance(terms, dict):
            terms = [terms]
        self.terms = []
        for t in terms:
            t = dict(t)
            kind = str(t.get("type", "")).lower()
            r_in, r_out = float(t["r_inner"]), float(t["r_outer"])
            if not 0.0 <= r_in < r_out:
                raise ValueError(
                    f"PairBaseline: need 0 <= r_inner < r_outer, got {r_in}, {r_out}."
                )
            if kind == "zbl":
                self.terms.append(dict(type="zbl", r_inner=r_in, r_outer=r_out))
            elif kind == "born_mayer":
                pairs = {}
                for k, v in t["pairs"].items():
                    a, b = [x.strip() for x in k.split("-")]
                    pairs[_pair_key(a, b)] = dict(A=float(v["A"]), rho=float(v["rho"]))
                self.terms.append(
                    dict(type="born_mayer", r_inner=r_in, r_outer=r_out, pairs=pairs)
                )
            else:
                raise ValueError(
                    f"PairBaseline: term type must be zbl or born_mayer, got {kind!r}."
                )
        self._Z = np.array([ATOMIC_NUMBER[e] for e in self.species], float)
        self.r_max = max(t["r_outer"] for t in self.terms)

    # ------------------------------------------------------------------
    def config(self) -> dict:
        return {"species": self.species, "terms": self.terms}

    def to_json(self) -> str:
        return json.dumps(self.config())

    @classmethod
    def from_config(cls, cfg):
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        return cls(cfg["species"], cfg["terms"])

    # ------------------------------------------------------------------
    def phi(self, r, si, sj):
        """Pair energy and its r-derivative for arrays r, species indices
        si, sj (all shape (M,))."""
        e = np.zeros_like(r)
        de = np.zeros_like(r)
        for t in self.terms:
            s, ds = _switch(r, t["r_inner"], t["r_outer"])
            on = s > 0
            if not on.any():
                continue
            rr = r[on]
            if t["type"] == "zbl":
                zi, zj = self._Z[si[on]], self._Z[sj[on]]
                a = 0.46850 / (zi**0.23 + zj**0.23)
                x = rr[:, None] / a[:, None]
                ex = _ZBL_C * np.exp(-_ZBL_D * x)
                f = ex.sum(axis=1)
                df = -(ex * _ZBL_D).sum(axis=1) / a
                pre = K_E * zi * zj
                p = pre * f / rr
                dp = pre * (df / rr - f / rr**2)
            else:
                A = np.zeros_like(rr)
                rho = np.ones_like(rr)
                ssp = [self.species[k] for k in si[on]]
                tsp = [self.species[k] for k in sj[on]]
                for n, (x, y) in enumerate(zip(ssp, tsp)):
                    c = t["pairs"].get(_pair_key(x, y))
                    if c is not None:
                        A[n], rho[n] = c["A"], c["rho"]
                p = A * np.exp(-rr / rho)
                dp = -p / rho
            e[on] += p * s[on]
            de[on] += dp * s[on] + p * ds[on]
        return e, de

    def from_entries(self, sp, centre, nbj, vec, volume):
        """
        E, F (N, 3), S (3, 3), per-atom energies (N,) and per-entry
        gradients dU_i/dv_m (M, 3) from a neighbour list (entry m: atom
        centre[m], neighbour nbj[m], vector v_m = r_j + shift - r_i; every
        pair appears from both sides).  S = -(1/V) dE/d(strain)
        (compressive-positive, ADR-0001).
        """
        N = len(sp)
        r = np.linalg.norm(vec, axis=1)
        e, de = self.phi(r, sp[centre], sp[nbj])
        eps = 0.5 * np.bincount(centre, weights=e, minlength=N)
        dEdv = (0.5 * de / np.where(r > 0, r, 1.0))[:, None] * vec
        F = np.zeros((N, 3))
        np.add.at(F, centre, dEdv)
        np.add.at(F, nbj, -dEdv)
        S = -(dEdv.T @ vec) / volume
        return float(eps.sum()), F, 0.5 * (S + S.T), eps, dEdv

    def compute(self, lattice, positions, atom_types):
        """(E, F, S) of one structure."""
        from .descriptors.acsf import _neighbours

        lattice = np.asarray(lattice, dtype=np.float64)
        sp = np.array([self.species.index(t) for t in atom_types])
        ptr, nbj, vec = _neighbours(lattice, positions, self.r_max)
        centre = np.repeat(np.arange(len(sp)), np.diff(ptr))
        E, F, S, _, _ = self.from_entries(
            sp, centre, nbj, vec, abs(np.linalg.det(lattice))
        )
        return E, F, S

    def subtract(self, structures):
        """Copies of the structure dicts with the baseline removed from
        ref_energy [eV/atom], ref_forces and ref_stress."""
        out = []
        for s in structures:
            s = dict(s)
            E, F, S = self.compute(s["lattice"], s["positions"], s["atom_types"])
            s["ref_energy"] = float(s["ref_energy"]) - E / len(s["atom_types"])
            if s.get("ref_forces") is not None:
                s["ref_forces"] = np.asarray(s["ref_forces"], float) - F
            if s.get("ref_stress") is not None:
                s["ref_stress"] = np.asarray(s["ref_stress"], float) - S
            out.append(s)
        return out

    def min_pair_distance(self, structures):
        """Shortest interatomic distance (all images) in the structures,
        to compare with r_outer."""
        from .descriptors.acsf import _neighbours

        best = np.inf
        for s in structures:
            _, _, vec = _neighbours(
                np.asarray(s["lattice"], float), s["positions"], 4.0 * self.r_max
            )
            if len(vec):
                best = min(best, float(np.linalg.norm(vec, axis=1).min()))
        return best


def physics_from_config(cfg):
    """(PairBaseline or None, isolated_atom_energy dict or None) of a
    model's ``physics_config`` JSON string (empty: neither)."""
    if not cfg:
        return None, None
    d = json.loads(cfg) if isinstance(cfg, str) else cfg
    b = d.get("baseline")
    return (
        PairBaseline.from_config(b) if b else None,
        d.get("isolated_atom_energy") or None,
    )


def fit_born_mayer(structures, energy_fn, species, r_inner, r_outer,
                   x0=(1000.0, 0.3)):
    """
    Fit a switched Born-Mayer term A exp(-r / rho) s(r) (one (A, rho) per
    species pair present within r_outer) to short-range reference data
    (ADR-0014, follow-up).

    structures : dicts with lattice, positions, atom_types and ref_energy
                 [eV/atom] -- e.g. one atom pushed towards a neighbour,
                 labelled by DFT single points
    energy_fn  : (lattice, positions, atom_types) -> total energy [eV] of
                 the trained model *with every other baseline term* (the
                 MLIP engine); the fit minimises
                 sum_k (E_ref,k - energy_fn_k - E_BM,k)^2
    Returns (term, info): the baseline term dict (type born_mayer, with
    the fitted pairs) and {pairs, rms_before_eV, rms_after_eV,
    n_structures}.
    """
    from scipy.optimize import least_squares

    from .descriptors.acsf import _neighbours

    probe = PairBaseline(species, {"type": "born_mayer", "r_inner": r_inner,
                                   "r_outer": r_outer, "pairs": {}})
    sp_list = probe.species
    entries, keys = [], set()
    for s in structures:
        sp = np.array([sp_list.index(t) for t in s["atom_types"]])
        ptr, nbj, vec = _neighbours(np.asarray(s["lattice"], float),
                                    s["positions"], r_outer)
        centre = np.repeat(np.arange(len(sp)), np.diff(ptr))
        r = np.linalg.norm(vec, axis=1)
        entries.append((r, sp[centre], sp[nbj]))
        keys |= {_pair_key(sp_list[a], sp_list[b])
                 for a, b in zip(sp[centre], sp[nbj])}
    pairs = sorted(keys)
    if not pairs:
        raise ValueError(
            f"fit_born_mayer: no pair closer than r_outer = {r_outer} A in the "
            f"short-range structures; nothing to fit.")
    res = np.array([float(s["ref_energy"]) * len(s["atom_types"])
                    - energy_fn(s["lattice"], s["positions"], s["atom_types"])
                    for s in structures])

    def make(p):
        return PairBaseline(sp_list, {
            "type": "born_mayer", "r_inner": r_inner, "r_outer": r_outer,
            "pairs": {k: {"A": float(np.exp(p[2 * n])), "rho": float(p[2 * n + 1])}
                      for n, k in enumerate(pairs)}})

    def f(p):
        b = make(p)
        return np.array([0.5 * b.phi(r, si, sj)[0].sum()
                         for r, si, sj in entries]) - res

    lo = np.tile([np.log(1e-3), 0.05], len(pairs))
    hi = np.tile([15.0, 1.0], len(pairs))
    sol = least_squares(f, x0=np.tile([np.log(x0[0]), x0[1]], len(pairs)),
                        bounds=(lo, hi))
    term = make(sol.x).terms[0]
    return term, {
        "pairs": term["pairs"],
        "rms_before_eV": float(np.sqrt(np.mean(res**2))),
        "rms_after_eV": float(np.sqrt(np.mean(sol.fun**2))),
        "n_structures": len(structures),
    }
