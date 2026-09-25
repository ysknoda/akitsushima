"""Atom-centred symmetry functions (ACSF) with analytic derivatives with
respect to atomic positions and to the cell (ADR-0004, decision 4).

Functions (J. Behler, J. Chem. Phys. 134, 074106 (2011), eqs. 4 and 8,
cosine cutoff eq. 2)::

    fc(r)   = 0.5 [cos(pi r / r_c) + 1]                  r < r_c, else 0

    G2_i    = sum_{j != i}  exp(-eta (r_ij - r_s)^2) fc(r_ij)

    G4_i    = 2^(1-zeta) sum_{j<k; j,k != i} (1 + lambda cos theta_ijk)^zeta
              exp(-eta (r_ij^2 + r_ik^2 + r_jk^2)) fc(r_ij) fc(r_ik) fc(r_jk)

    G5_i    = 2^(1-zeta) sum_{j<k; j,k != i} (1 + lambda cos theta_ijk)^zeta
              exp(-eta (r_ij^2 + r_ik^2)) fc(r_ij) fc(r_ik)

G5 (Behler 2011, eq. 9) drops the r_jk terms.  It matters when r_c is
chosen short: in diamond Si with r_c = 3.771 A (Stillinger-Weber's own
range a*sigma) two neighbours of an atom are 3.84 A apart, so fc(r_jk) = 0
removes almost every bonded triplet from G4 and only G5 carries the bond
angles.

G2 is resolved by the neighbour's species, G4 and G5 by the unordered
species pair of the two neighbours.  Each unordered neighbour pair (j, k) is
counted once; Behler's sum over all j, k != i counts it twice, a factor 2
convention.  Neighbours include every periodic image within r_c, so cells
smaller than 2 r_c are handled.  Species axes follow atomic number
(Z order).

Structure descriptor for the tabular models
-------------------------------------------
``structure_table()`` returns, with N atoms and species blocks in Z order,

    d          : (n_sp * Dc,)  per-species block s holds
                 (1/N) sum_{i of species s} G_i, so d is intensive and
                 sum_s blocks = mean_i G_i
    X_F        : (N, 3, D)     X_F[j, a] = -dd/dr_{j,a}
    X_S        : (3, 3, D)     X_S[a, b] = -(1/V) dd/dT_{ab}, with
                 lattice' = lattice @ T and positions' = frac @ lattice'
                 (ADR-0001)

which is exactly the E/F/S row convention of ``load_mlip_csv``: for
E/N = f(d), F_{j,a} = N grad f . X_F[j,a] and
sigma_{ab} = N grad f . X_S[a,b] (compressive-positive).
"""

import json
import math
from collections.abc import Sequence

import numpy as np

from ...constants import ATOMIC_MASS
from ...utils.jit import _numba_njit

__all__ = ["ACSF", "ATOMIC_NUMBER", "AtomicJacobian", "sort_species"]

# ATOMIC_MASS is listed H, He, Li, ... in atomic-number order (Z = 1..86).
ATOMIC_NUMBER = {el: z for z, el in enumerate(ATOMIC_MASS, start=1)}
assert (
    ATOMIC_NUMBER["Si"],
    ATOMIC_NUMBER["Ge"],
    ATOMIC_NUMBER["Sn"],
    ATOMIC_NUMBER["Rn"],
) == (14, 32, 50, 86)


def sort_species(species: Sequence[str]) -> list:
    """Unique species ordered by atomic number."""
    uniq = sorted(set(species), key=lambda el: ATOMIC_NUMBER.get(el, 10_000))
    unknown = [el for el in uniq if el not in ATOMIC_NUMBER]
    if unknown:
        raise ValueError(f"sort_species: unknown element(s) {unknown}.")
    return uniq


# ---------------------------------------------------------------------------
#  Neighbour list (all periodic images within r_c)
# ---------------------------------------------------------------------------


def _neighbours(lattice, positions, r_cut):
    """CSR neighbour list: for atom i, entries ptr[i]:ptr[i+1] give the
    neighbour atom index and the vector r_j + shift - r_i (|.| < r_cut)."""
    lattice = np.asarray(lattice, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    N = len(pos)
    vol = abs(np.linalg.det(lattice))
    n_img = []
    for k in range(3):
        cross = np.cross(lattice[(k + 1) % 3], lattice[(k + 2) % 3])
        width = vol / np.linalg.norm(cross)  # perpendicular width
        n_img.append(int(math.ceil(r_cut / width)))
    rng = [np.arange(-n, n + 1) for n in n_img]
    shifts = np.array(np.meshgrid(*rng, indexing="ij")).reshape(3, -1).T
    shift_cart = shifts @ lattice  # (S, 3)
    # Wrap into the cell so that the image range above is sufficient.
    frac = pos @ np.linalg.inv(lattice)
    pos_w = (frac - np.floor(frac)) @ lattice
    delta = pos_w - pos  # wrapping offset
    dv = (
        pos_w[None, :, None, :] + shift_cart[None, None, :, :] - pos_w[:, None, None, :]
    )  # (i, j, S, 3)
    r2 = np.einsum("ijsk,ijsk->ijs", dv, dv)
    mask = (r2 < r_cut * r_cut) & (r2 > 1e-20)
    ii, jj, ss = np.nonzero(mask)
    vec = dv[ii, jj, ss]
    # The vectors are between wrapped positions; the displacement of an
    # atom moves its wrapped copy identically, so derivatives w.r.t. r_j
    # are unaffected by the wrap (delta is constant).
    del delta
    ptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(ptr, ii + 1, 1)
    ptr = np.cumsum(ptr)
    return ptr, jj.astype(np.int64), vec


# ---------------------------------------------------------------------------
#  Kernels
# ---------------------------------------------------------------------------


@_numba_njit
def _acsf_kernel(
    ptr,
    nbj,
    vec,
    sp,
    n_sp,
    r_cut,
    g2_eta,
    g2_rs,
    g4_eta,
    g4_zeta,
    g4_lam,
    g4_jk,
    pair_index,
    Dc,
    want_deriv,
):
    """
    Returns G (N, Dc) and dGdv (M, 3, Dc): for neighbour entry m of atom
    i (ptr[i] <= m < ptr[i+1]), dGdv[m, a, c] = dG_{i,c} / dv_{m,a}, the
    derivative with respect to the entry's vector v_m = r_j + shift - r_i.
    Every position and cell derivative follows from these (see
    :func:`_aggregate` and :func:`_contract`).
    """
    N = ptr.shape[0] - 1
    n2 = g2_eta.shape[0]
    n4 = g4_eta.shape[0]
    G = np.zeros((N, Dc))
    M = vec.shape[0] if want_deriv else 1
    dGdv = np.zeros((M, 3, Dc))
    pi_rc = math.pi / r_cut
    off4 = n_sp * n2
    for i in range(N):
        s0 = ptr[i]
        s1 = ptr[i + 1]
        # ---- G2 ----
        for m in range(s0, s1):
            j = nbj[m]
            vx = vec[m, 0]
            vy = vec[m, 1]
            vz = vec[m, 2]
            r = math.sqrt(vx * vx + vy * vy + vz * vz)
            fc = 0.5 * (math.cos(pi_rc * r) + 1.0)
            dfc = -0.5 * pi_rc * math.sin(pi_rc * r)
            for q in range(n2):
                ch = sp[j] * n2 + q
                x = r - g2_rs[q]
                e = math.exp(-g2_eta[q] * x * x)
                G[i, ch] += e * fc
                if want_deriv:
                    dgdr = e * (dfc - 2.0 * g2_eta[q] * x * fc)
                    dGdv[m, 0, ch] += dgdr * vx / r
                    dGdv[m, 1, ch] += dgdr * vy / r
                    dGdv[m, 2, ch] += dgdr * vz / r
        # ---- angular: G4 (jk = 1) and G5 (jk = 0) ----
        if n4 == 0:
            continue
        for m in range(s0, s1):
            j = nbj[m]
            ax = vec[m, 0]
            ay = vec[m, 1]
            az = vec[m, 2]
            ra = math.sqrt(ax * ax + ay * ay + az * az)
            fa = 0.5 * (math.cos(pi_rc * ra) + 1.0)
            dfa = -0.5 * pi_rc * math.sin(pi_rc * ra)
            for n in range(m + 1, s1):
                k = nbj[n]
                bx = vec[n, 0]
                by = vec[n, 1]
                bz = vec[n, 2]
                cx = bx - ax
                cy = by - ay
                cz = bz - az
                rc_ = math.sqrt(cx * cx + cy * cy + cz * cz)
                if rc_ < 1e-10:
                    continue
                in_jk = rc_ < r_cut
                rb = math.sqrt(bx * bx + by * by + bz * bz)
                fb = 0.5 * (math.cos(pi_rc * rb) + 1.0)
                dfb = -0.5 * pi_rc * math.sin(pi_rc * rb)
                if in_jk:
                    fcc = 0.5 * (math.cos(pi_rc * rc_) + 1.0)
                    dfcc = -0.5 * pi_rc * math.sin(pi_rc * rc_)
                else:
                    fcc = 0.0
                    dfcc = 0.0
                ab = ax * bx + ay * by + az * bz
                cos_t = ab / (ra * rb)
                # dcos/da, dcos/db
                ia = 1.0 / (ra * rb)
                dca_x = bx * ia - cos_t * ax / (ra * ra)
                dca_y = by * ia - cos_t * ay / (ra * ra)
                dca_z = bz * ia - cos_t * az / (ra * ra)
                dcb_x = ax * ia - cos_t * bx / (rb * rb)
                dcb_y = ay * ia - cos_t * by / (rb * rb)
                dcb_z = az * ia - cos_t * bz / (rb * rb)
                pidx = pair_index[sp[j], sp[k]]
                for q in range(n4):
                    jk = g4_jk[q]
                    if jk > 0.5:
                        # G4: fc(r_ij) fc(r_ik) fc(r_jk), c = b - a
                        if not in_jk:
                            continue
                        F = fa * fb * fcc
                        t1 = dfa * fb * fcc / ra
                        t2 = fa * dfb * fcc / rb
                        t3 = fa * fb * dfcc / rc_
                        rsum = ra * ra + rb * rb + rc_ * rc_
                    else:
                        # G5: fc(r_ij) fc(r_ik) only, no r_jk term
                        F = fa * fb
                        t1 = dfa * fb / ra
                        t2 = fa * dfb / rb
                        t3 = 0.0
                        rsum = ra * ra + rb * rb
                    eta = g4_eta[q]
                    zeta = g4_zeta[q]
                    lam = g4_lam[q]
                    ch = off4 + pidx * n4 + q
                    base_c = 1.0 + lam * cos_t
                    if base_c <= 0.0:
                        continue
                    P = base_c**zeta
                    E = math.exp(-eta * rsum)
                    K = 2.0 ** (1.0 - zeta)
                    G[i, ch] += K * P * E * F
                    if not want_deriv:
                        continue
                    dFa_x = t1 * ax - t3 * cx
                    dFa_y = t1 * ay - t3 * cy
                    dFa_z = t1 * az - t3 * cz
                    dFb_x = t2 * bx + t3 * cx
                    dFb_y = t2 * by + t3 * cy
                    dFb_z = t2 * bz + t3 * cz
                    dP = K * zeta * base_c ** (zeta - 1.0) * lam * E * F
                    KPE = K * P * E
                    # d(exp)/da = -2 eta E (a - jk c), d/db = -2 eta E (b + jk c)
                    dGdv[m, 0, ch] += dP * dca_x + KPE * (
                        -2.0 * eta * (ax - jk * cx) * F + dFa_x
                    )
                    dGdv[m, 1, ch] += dP * dca_y + KPE * (
                        -2.0 * eta * (ay - jk * cy) * F + dFa_y
                    )
                    dGdv[m, 2, ch] += dP * dca_z + KPE * (
                        -2.0 * eta * (az - jk * cz) * F + dFa_z
                    )
                    dGdv[n, 0, ch] += dP * dcb_x + KPE * (
                        -2.0 * eta * (bx + jk * cx) * F + dFb_x
                    )
                    dGdv[n, 1, ch] += dP * dcb_y + KPE * (
                        -2.0 * eta * (by + jk * cy) * F + dFb_y
                    )
                    dGdv[n, 2, ch] += dP * dcb_z + KPE * (
                        -2.0 * eta * (bz + jk * cz) * F + dFb_z
                    )
    return G, dGdv


@_numba_njit
def _aggregate(ptr, nbj, vec, sp, dGdv, n_sp):
    """
    Structure-level sums of the entry derivatives, each atom's features
    placed in its species block (width n_sp * Dc):

        dS[j, a] = sum_i dG_i/dr_{j,a}      Wv[a, b] = sum_i dG_i/dT_{ab}

    with dG_i/dr_j = sum_{m of i -> j} dGdv[m] - [i == j] sum_{m of i}
    dGdv[m], and dG_i/dT_ab = sum_{m of i} v_{m,a} dGdv[m, b].
    """
    N = ptr.shape[0] - 1
    Dc = dGdv.shape[2]
    dS = np.zeros((N, 3, n_sp * Dc))
    Wv = np.zeros((3, 3, n_sp * Dc))
    for i in range(N):
        base = sp[i] * Dc
        for m in range(ptr[i], ptr[i + 1]):
            j = nbj[m]
            for a in range(3):
                for c in range(Dc):
                    g = dGdv[m, a, c]
                    dS[j, a, base + c] += g
                    dS[i, a, base + c] -= g
                    for b in range(3):
                        Wv[b, a, base + c] += vec[m, b] * g
    return dS, Wv


@_numba_njit
def _contract(ptr, nbj, vec, dGdv, g):
    """
    Forces and virial of E = sum_i eps_i(G_i) given g[i] = d eps_i/dG_i:

        F_j = -sum_i g_i . dG_i/dr_j        W_ab = sum_i g_i . dG_i/dT_ab

    (stress = -W / V, compressive-positive).
    """
    N = ptr.shape[0] - 1
    Dc = dGdv.shape[2]
    F = np.zeros((N, 3))
    W = np.zeros((3, 3))
    for i in range(N):
        for m in range(ptr[i], ptr[i + 1]):
            j = nbj[m]
            for a in range(3):
                c = 0.0
                for k in range(Dc):
                    c += dGdv[m, a, k] * g[i, k]
                F[j, a] -= c
                F[i, a] += c
                for b in range(3):
                    W[b, a] += vec[m, b] * c
    return F, W

# ---------------------------------------------------------------------------


class AtomicJacobian:
    """
    Per-atom ACSF vectors of one structure and their derivatives, stored
    per neighbour entry (see :func:`_acsf_kernel`):

    sp (N,) species index; G (N, Dc); ptr (N+1,), nbj (M,), vec (M, 3):
    the CSR neighbour list (entry m belongs to atom ``centre[m]``);
    dGdv (M, 3, Dc); volume [A^3].
    """

    def __init__(self, sp, G, ptr, nbj, vec, dGdv, volume):
        self.sp, self.G, self.ptr, self.nbj = sp, G, ptr, nbj
        self.vec, self.dGdv, self.volume = vec, dGdv, float(volume)

    @property
    def centre(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.sp)), np.diff(self.ptr))

    def forces_and_stress(self, g):
        """Forces (N, 3) and stress (3, 3) [compressive-positive, ADR-0001]
        of E = sum_i eps_i(G_i), from g[i] = d eps_i / dG_i."""
        F, W = _contract(
            self.ptr, self.nbj, self.vec, self.dGdv, np.ascontiguousarray(g)
        )
        S = -W / self.volume
        return F, 0.5 * (S + S.T)


class ACSF:
    """
    Behler-Parrinello symmetry functions G2 (radial), G4 and G5 (angular).

    Parameters
    ----------
    species : element symbols; stored in atomic-number order.
    r_cut   : cutoff radius [A] of the cosine cutoff function.
    g2      : list of (eta [1/A^2], r_s [A]) pairs.
    g4, g5  : lists of (eta [1/A^2], zeta, lambda) triples, lambda = +1 or
              -1, zeta >= 1.

    Feature layout of one atom (Dc columns): G2 blocks for each neighbour
    species (Z order) x each (eta, r_s); then angular blocks for each
    unordered neighbour species pair (Z order, b <= c) x each G4 then each
    G5 (eta, zeta, lambda).
    """

    def __init__(self, species, r_cut, g2=(), g4=(), g5=()):
        self.species = sort_species(species)
        self.r_cut = float(r_cut)
        self.g2 = [(float(e), float(rs)) for e, rs in g2]
        self.g4 = [(float(e), float(z), float(l)) for e, z, l in g4]
        self.g5 = [(float(e), float(z), float(l)) for e, z, l in g5]
        for _, z, lam in self.g4 + self.g5:
            if z < 1.0 or lam not in (1.0, -1.0):
                raise ValueError("ACSF: G4/G5 need zeta >= 1 and lambda = +/-1.")
        if not (self.g2 or self.g4 or self.g5):
            raise ValueError("ACSF: give at least one G2, G4 or G5 function.")
        n = len(self.species)
        self._pairs = [(b, c) for b in range(n) for c in range(b, n)]
        self._pair_index = np.zeros((n, n), dtype=np.int64)
        for p, (b, c) in enumerate(self._pairs):
            self._pair_index[b, c] = self._pair_index[c, b] = p

    # ------------------------------------------------------------------
    @classmethod
    def from_grid(cls, species, r_cut, g2=None, g4=None, g5=None):
        """
        Build from YAML-style grids: ``g2 = {eta: [...], rs: [...]}`` and
        ``g4`` / ``g5 = {eta: [...], zeta: [...], lambda: [...]}``
        (Cartesian products).
        """
        g2l = [(e, r) for e in g2["eta"] for r in g2.get("rs", [0.0])] if g2 else []

        def _ang(g):
            if not g:
                return []
            return [
                (e, z, l)
                for e in g["eta"]
                for z in g["zeta"]
                for l in g.get("lambda", [-1, 1])
            ]

        return cls(species, r_cut, g2l, _ang(g4), _ang(g5))

    def config(self) -> dict:
        return {
            "type": "acsf",
            "species": self.species,
            "r_cut": self.r_cut,
            "g2": self.g2,
            "g4": self.g4,
            "g5": self.g5,
        }

    def to_json(self) -> str:
        return json.dumps(self.config())

    @classmethod
    def from_config(cls, cfg):
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        if cfg.get("type", "acsf") != "acsf":
            raise ValueError(f"ACSF.from_config: type {cfg.get('type')!r}.")
        return cls(
            cfg["species"], cfg["r_cut"], cfg["g2"], cfg["g4"], cfg.get("g5", [])
        )

    @property
    def _angular(self):
        """G4 then G5 as (eta, zeta, lambda, jk) with jk = 1 for G4."""
        return [t + (1.0,) for t in self.g4] + [t + (0.0,) for t in self.g5]

    @property
    def n_per_atom(self) -> int:
        n = len(self.species)
        return n * len(self.g2) + len(self._pairs) * len(self._angular)

    @property
    def n_features(self) -> int:
        """Width of the structure descriptor d."""
        return len(self.species) * self.n_per_atom

    def feature_names(self) -> list:
        sp = self.species
        per = [
            f"g2_{sp[b]}_eta{e:g}_rs{r:g}" for b in range(len(sp)) for e, r in self.g2
        ] + [
            f"g{4 if jk else 5}_{sp[b]}-{sp[c]}_eta{e:g}_zeta{z:g}_lam{l:+g}"
            for b, c in self._pairs
            for e, z, l, jk in self._angular
        ]
        return [f"{s}:{name}" for s in sp for name in per]

    # ------------------------------------------------------------------
    def _run(self, lattice, positions, atom_types, want_deriv):
        missing = set(atom_types) - set(self.species)
        if missing:
            raise ValueError(
                f"ACSF: element(s) {sorted(missing)} not in the "
                f"descriptor's species {self.species}."
            )
        sp = np.array([self.species.index(t) for t in atom_types], dtype=np.int64)
        ptr, nbj, vec = _neighbours(lattice, positions, self.r_cut)
        vec = np.ascontiguousarray(vec)
        g2 = np.array(self.g2, dtype=np.float64).reshape(-1, 2)
        g4 = np.array(self._angular, dtype=np.float64).reshape(-1, 4)
        G, dGdv = _acsf_kernel(
            ptr,
            nbj,
            vec,
            sp,
            len(self.species),
            self.r_cut,
            np.ascontiguousarray(g2[:, 0]),
            np.ascontiguousarray(g2[:, 1]),
            np.ascontiguousarray(g4[:, 0]),
            np.ascontiguousarray(g4[:, 1]),
            np.ascontiguousarray(g4[:, 2]),
            np.ascontiguousarray(g4[:, 3]),
            self._pair_index,
            self.n_per_atom,
            want_deriv,
        )
        return sp, ptr, nbj, vec, G, dGdv

    def atomic(self, lattice, positions, atom_types) -> np.ndarray:
        """Per-atom descriptors G (N, n_per_atom)."""
        return self._run(lattice, positions, atom_types, False)[4]

    def atomic_jacobian(self, lattice, positions, atom_types) -> "AtomicJacobian":
        """Per-atom descriptors with their derivatives (ADR-0005)."""
        lattice = np.asarray(lattice, dtype=np.float64)
        sp, ptr, nbj, vec, G, dGdv = self._run(lattice, positions, atom_types, True)
        return AtomicJacobian(sp, G, ptr, nbj, vec, dGdv, abs(np.linalg.det(lattice)))

    def structure(self, lattice, positions, atom_types) -> np.ndarray:
        """Structure descriptor d (n_features,) -- see the module docstring."""
        sp, *_, G, _ = self._run(lattice, positions, atom_types, False)
        return self._blocks(G, sp)

    def _blocks(self, G, sp):
        N, Dc = G.shape
        d = np.zeros(len(self.species) * Dc)
        for s in range(len(self.species)):
            d[s * Dc : (s + 1) * Dc] = G[sp == s].sum(axis=0)
        return d / N

    def structure_table(self, lattice, positions, atom_types):
        """(d, X_F, X_S) in the E/F/S row convention of load_mlip_csv."""
        J = self.atomic_jacobian(lattice, positions, atom_types)
        N = len(J.sp)
        dS, Wv = _aggregate(J.ptr, J.nbj, J.vec, J.sp, J.dGdv, len(self.species))
        return self._blocks(J.G, J.sp), -dS / N, -Wv / (N * J.volume)

    # ------------------------------------------------------------------
    def table_rows(
        self,
        lattice,
        positions,
        atom_types,
        energy,
        forces=None,
        stress=None,
        r0=1.0,
        eps0=1.0,
    ):
        """
        Rows of one structure in the load_mlip_csv layout.

        energy : total energy [eV]; forces (N, 3) [eV/A]; stress (3, 3)
        [eV/A^3], compressive-positive.  Unit weights: E -> N, F -> r0,
        S -> eps0 * V (csv_io's convention).

        Returns (targets, labels, weights, X).
        """
        d, XF, XS = self.structure_table(lattice, positions, atom_types)
        N = len(atom_types)
        V = abs(np.linalg.det(np.asarray(lattice, dtype=np.float64)))
        y, lab, w, X = [energy / N], ["E"], [float(N)], [d]
        if forces is not None:
            F = np.asarray(forces, dtype=np.float64)
            for i in range(N):
                for a in range(3):
                    y.append(F[i, a])
                    lab.append("F")
                    w.append(r0)
                    X.append(XF[i, a])
        if stress is not None:
            S = np.asarray(stress, dtype=np.float64)
            for a, b in ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1)):
                y.append(S[a, b])
                lab.append("S")
                w.append(eps0 * V)
                X.append(0.5 * (XS[a, b] + XS[b, a]))
        return (np.array(y), np.array(lab), np.array(w), np.array(X))


def build_table(
    structures,
    descriptor: "ACSF",
    r0: float = 1.0,
    eps0: float = 1.0,
    verbose: bool = False,
):
    """
    Training table from a list of structure dicts (``lattice``,
    ``positions``, ``atom_types``, ``ref_energy`` [eV/atom], optional
    ``ref_forces`` (N, 3), ``ref_stress`` (3, 3)) -- the dicts produced by
    the classical ``fit`` loader.  Returns (targets, labels, weights, X).
    """
    parts = []
    for k, s in enumerate(structures):
        N = len(s["atom_types"])
        parts.append(
            descriptor.table_rows(
                s["lattice"],
                s["positions"],
                s["atom_types"],
                energy=float(s["ref_energy"]) * N,
                forces=s.get("ref_forces"),
                stress=s.get("ref_stress"),
                r0=r0,
                eps0=eps0,
            )
        )
        if verbose and (k + 1) % 100 == 0:
            print(f"  descriptors: {k + 1}/{len(structures)} structures")
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def write_table_csv(
    path, targets, labels, weights, X, columns: Sequence[str] | None = None
) -> None:
    """Write a table in the load_mlip_csv layout."""
    names = (
        list(columns) if columns is not None else [f"f{j}" for j in range(X.shape[1])]
    )
    with open(path, "w") as fh:
        fh.write("target,label,weight," + ",".join(names) + "\n")
        fh.writelines(
            f"{y:.12g},{l},{w:.12g}," + ",".join(f"{v:.12g}" for v in x) + "\n"
            for y, l, w, x in zip(targets, labels, weights, X)
        )
