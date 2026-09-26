"""Site-energy decomposition of an interatomic potential (ADR-0007).

For a potential written as E = sum_i U_i, where U_i depends on the vectors
v_m = r_n - r_c from atom c = centre[m] to its neighbours n = nbr[m], a
``SiteTerms`` holds

    e_site (N,)    U_i
    centre, nbr (M,), vec (M, 3), dUdv (M, 3) = dU_centre / dv_m

From these alone follow the forces, the stress and the exact heat flux of
a many-body potential::

    F_k   = sum_m dUdv_m (delta_{k,centre} - delta_{k,nbr})
    sigma = -(1/V) sum_m dUdv_m (x) v_m        (compressive-positive, ADR-0001)
    J V   = sum_i E_i v_i - sum_m v_m (dUdv_m . vel_nbr)

The last is d/dt sum_i r_i E_i with E_i = K_i + U_i (Fan et al., Phys. Rev.
B 92, 094301 (2015), eq. 21, written per neighbour entry): the r_i F_i.v_i
terms of the kinetic and potential parts cancel, which leaves a form that
needs only neighbour vectors and so holds under periodic boundaries.

Each ``*_site_terms`` function below reproduces its engine's energy
convention exactly; tests check that the forces and stress rebuilt from
the site terms equal the engine's own.
"""

from dataclasses import dataclass

import numpy as np

from .neighborlist import _check_min_image


@dataclass
class SiteTerms:
    e_site: np.ndarray
    centre: np.ndarray
    nbr: np.ndarray
    vec: np.ndarray
    dUdv: np.ndarray

    @property
    def energy(self) -> float:
        return float(self.e_site.sum())

    def forces(self) -> np.ndarray:
        F = np.zeros((len(self.e_site), 3))
        np.add.at(F, self.centre, self.dUdv)
        np.add.at(F, self.nbr, -self.dUdv)
        return F

    def stress(self, volume: float) -> np.ndarray:
        S = -np.einsum("ma,mb->ab", self.dUdv, self.vec) / volume
        return 0.5 * (S + S.T)

    def virial_flux(self, velocities: np.ndarray) -> np.ndarray:
        """-sum_m v_m (dUdv_m . vel_nbr)  [eV A/fs], not divided by V."""
        c = np.einsum("ma,ma->m", self.dUdv, velocities[self.nbr])
        return -np.einsum("m,ma->a", c, self.vec)

    @staticmethod
    def combine(parts, n_atoms):
        parts = [p for p in parts if p is not None]
        if not parts:
            z = np.zeros(0, dtype=np.int64)
            return SiteTerms(
                np.zeros(n_atoms), z, z, np.zeros((0, 3)), np.zeros((0, 3))
            )
        return SiteTerms(
            np.sum([p.e_site for p in parts], axis=0),
            np.concatenate([p.centre for p in parts]),
            np.concatenate([p.nbr for p in parts]),
            np.concatenate([p.vec for p in parts]),
            np.concatenate([p.dUdv for p in parts]),
        )


# ---------------------------------------------------------------------------
#  Neighbour entries
# ---------------------------------------------------------------------------


def _pairs(lattice, positions, rc, caller):
    """Unordered pairs i < j within rc under the minimum image: (i, j,
    v_ij = r_j - r_i, r).  Refuses a cell that cannot host rc."""
    _check_min_image(lattice, rc, caller)
    N = len(positions)
    ii, jj = np.triu_indices(N, k=1)
    dr = positions[jj] - positions[ii]
    s = dr @ np.linalg.inv(lattice)
    s -= np.round(s)
    v = s @ lattice
    r = np.linalg.norm(v, axis=1)
    m = (r < rc) & (r > 1e-12)
    return ii[m], jj[m], v[m], r[m]


def _entries(ii, jj, v, r):
    """Both directions of each pair, sorted by centre: (centre, nbr, vec,
    r, pair index, ptr)."""
    P = len(ii)
    centre = np.concatenate([ii, jj])
    nbr = np.concatenate([jj, ii])
    vec = np.concatenate([v, -v])
    rr = np.concatenate([r, r])
    pid = np.concatenate([np.arange(P), np.arange(P)])
    o = np.argsort(centre, kind="stable")
    return centre[o], nbr[o], vec[o], rr[o], pid[o]


def _ordered_entry_pairs(centre, n_atoms):
    """All ordered pairs (m1, m2), m1 != m2, of entries sharing a centre
    (entries sorted by centre)."""
    cnt = np.bincount(centre, minlength=n_atoms)
    ptr = np.concatenate([[0], np.cumsum(cnt)])
    k = cnt[centre]
    m1 = np.repeat(np.arange(len(centre)), k)
    first = np.repeat(ptr[centre], k)
    off = np.arange(len(m1)) - np.repeat(np.cumsum(k) - k, k)
    m2 = first + off
    keep = m1 != m2
    return m1[keep], m2[keep]


def _cos_grad(u, w, ru, rw):
    """cos(u, w) and its gradients with respect to u and w."""
    cos = np.einsum("ma,ma->m", u, w) / (ru * rw)
    dcu = w / (ru * rw)[:, None] - (cos / ru**2)[:, None] * u
    dcw = u / (ru * rw)[:, None] - (cos / rw**2)[:, None] * w
    return cos, dcu, dcw


# ---------------------------------------------------------------------------
#  Pair potentials (BasePairPotential.compute_pairs)
# ---------------------------------------------------------------------------


def pair_site_terms(pot, lattice, positions, atom_types):
    """V_ij split equally between i and j.  compute_pairs returns
    r_ij = r_j - r_i and f_ij, the force on i, which equals dV/dv_ij."""
    from ..utils.mpi import local_only

    # compute_pairs splits pairs across MPI ranks; every rank needs them all
    # here, as the many-body site terms are computed whole on each rank.
    with local_only():
        ii, jj, rij, fij, V = pot.compute_pairs(lattice, positions, atom_types)
    N = len(positions)
    e = np.zeros(N)
    np.add.at(e, ii, 0.5 * V)
    np.add.at(e, jj, 0.5 * V)
    return SiteTerms(
        e,
        np.concatenate([ii, jj]),
        np.concatenate([jj, ii]),
        np.concatenate([rij, -rij]),
        np.concatenate([0.5 * fij, -0.5 * fij]),
    )


# ---------------------------------------------------------------------------
#  Stillinger-Weber
# ---------------------------------------------------------------------------

_SW_DEFAULTS = {
    "epsilon": 2.1682,
    "sigma": 2.0951,
    "A": 7.049556,
    "B": 0.6022246,
    "p": 4.0,
    "q": 0.0,
    "a": 1.80,
    "lambda": 21.0,
    "gamma": 1.20,
    "costheta0": -1.0 / 3.0,
}


def sw_site_terms(pot, lattice, positions, atom_types):
    """U_i = (1/2) sum_j V2(r_ij) + sum over angles centred on i.

    Two-body: each unordered pair once, from the record (s_i, s_j, s_j)
    (StillingerWeberPotential checks both records agree).  Three-body: both
    orders (j, k) of a neighbour pair, each h_ijk / 2, legs from their pair
    records -- the engine's convention (ADR-0006).
    """
    species = list(pot.species)
    if pot._multi:
        rec = pot.p
        rc = float(pot.r_cut)
    else:
        s0 = species[0]
        full = dict(_SW_DEFAULTS)
        full.update(pot.p)
        rec = {(s0, s0, s0): full}
        rc = float(full["a"]) * float(full["sigma"])
    ns = len(species)
    idx = {s: k for k, s in enumerate(species)}
    t = np.array([idx[a] for a in atom_types], dtype=np.int64)

    # pair tables (two-body and legs both read (i, j, j))
    tb_ok = np.zeros((ns, ns), bool)
    leg_ok = np.zeros((ns, ns), bool)
    T = {
        k: np.zeros((ns, ns))
        for k in ("epsilon", "sigma", "a", "A", "B", "p", "q", "gamma")
    }
    T["sigma"][:] = 1.0
    for x, si in enumerate(species):
        for y, sj in enumerate(species):
            r_ = rec.get((si, sj, sj))
            if r_ is None:
                continue
            leg_ok[x, y] = True
            T["sigma"][x, y] = float(r_["sigma"])
            T["a"][x, y] = float(r_["a"])
            T["gamma"][x, y] = float(r_.get("gamma", 1.2))
            if float(r_.get("A", 0.0)) != 0.0:
                tb_ok[x, y] = True
                T["epsilon"][x, y] = float(r_.get("epsilon", 1.0))
                T["A"][x, y] = float(r_["A"])
                T["B"][x, y] = float(r_.get("B", 0.0))
                T["p"][x, y] = float(r_.get("p", 4))
                T["q"][x, y] = float(r_.get("q", 0))
    tri_ok = np.zeros((ns, ns, ns), bool)
    tri_el = np.zeros((ns, ns, ns))
    tri_c0 = np.full((ns, ns, ns), -1.0 / 3.0)
    for x, si in enumerate(species):
        for y, sj in enumerate(species):
            for z, sk in enumerate(species):
                r_ = rec.get((si, sj, sk))
                if r_ is None or not leg_ok[x, y] or not leg_ok[x, z]:
                    continue
                tri_ok[x, y, z] = True
                tri_el[x, y, z] = float(r_.get("epsilon", 1.0)) * float(r_["lambda"])
                tri_c0[x, y, z] = float(r_.get("costheta0", -1.0 / 3.0))

    N = len(positions)
    ii, jj, v, r = _pairs(lattice, positions, rc, "StillingerWeberPotential.site_terms")
    centre, nbr, vec, rr, _ = _entries(ii, jj, v, r)
    e = np.zeros(N)
    dUdv = np.zeros_like(vec)

    # two-body, half to each end
    tc, tn = t[centre], t[nbr]
    ok = tb_ok[tc, tn]
    sig, a = T["sigma"][tc, tn], T["a"][tc, tn]
    x = rr / sig
    ok &= x < a
    xs, as_ = x[ok], a[ok]
    eps, A, B = T["epsilon"][tc, tn][ok], T["A"][tc, tn][ok], T["B"][tc, tn][ok]
    p, q = T["p"][tc, tn][ok], T["q"][tc, tn][ok]
    tt = 1.0 / (xs - as_)
    ex = np.exp(tt)
    poly = B * xs ** (-p) - xs ** (-q)
    V2 = eps * A * poly * ex
    dV2dx = (
        eps
        * A
        * ex
        * ((-p * B * xs ** (-p - 1) + q * xs ** (-q - 1)) + poly * (-(tt**2)))
    )
    np.add.at(e, centre[ok], 0.5 * V2)
    dUdv[ok] += (0.5 * dV2dx / sig[ok] / rr[ok])[:, None] * vec[ok]

    # three-body, both orders, each weighted 1/2
    m1, m2 = _ordered_entry_pairs(centre, N)
    c = centre[m1]
    ts, tj, tk = t[c], t[nbr[m1]], t[nbr[m2]]
    ok = tri_ok[ts, tj, tk]
    m1, m2, ts, tj, tk = m1[ok], m2[ok], ts[ok], tj[ok], tk[ok]
    sj, sk = T["sigma"][ts, tj], T["sigma"][ts, tk]
    aj, ak = T["a"][ts, tj], T["a"][ts, tk]
    gj, gk = T["gamma"][ts, tj], T["gamma"][ts, tk]
    rj, rk = rr[m1], rr[m2]
    xj, xk = rj / sj, rk / sk
    ok = (xj < aj) & (xk < ak)
    m1, m2, ts, tj, tk = m1[ok], m2[ok], ts[ok], tj[ok], tk[ok]
    sj, sk, aj, ak, gj, gk = sj[ok], sk[ok], aj[ok], ak[ok], gj[ok], gk[ok]
    rj, rk, xj, xk = rj[ok], rk[ok], xj[ok], xk[ok]
    el = 0.5 * tri_el[ts, tj, tk]
    c0 = tri_c0[ts, tj, tk]
    u, w = vec[m1], vec[m2]
    cos, dcu, dcw = _cos_grad(u, w, rj, rk)
    E = np.exp(gj / (xj - aj) + gk / (xk - ak))
    d = cos - c0
    h = el * E * d**2
    dlnE_drj = -gj / (xj - aj) ** 2 / sj
    dlnE_drk = -gk / (xk - ak) ** 2 / sk
    dhu = (h * dlnE_drj / rj)[:, None] * u + (2 * el * E * d)[:, None] * dcu
    dhw = (h * dlnE_drk / rk)[:, None] * w + (2 * el * E * d)[:, None] * dcw
    np.add.at(e, centre[m1], h)
    np.add.at(dUdv, m1, dhu)
    np.add.at(dUdv, m2, dhw)
    return SiteTerms(e, centre, nbr, vec, dUdv)


# ---------------------------------------------------------------------------
#  Tersoff
# ---------------------------------------------------------------------------


def _fc(r, R, D):
    """Tersoff cutoff 1/2 - 1/2 sin(pi/2 (r - R)/D) and its derivative."""
    arg = 0.5 * np.pi * (r - R) / D
    inner = (r >= R - D) & (r <= R + D)
    fc = np.where(r < R - D, 1.0, np.where(inner, 0.5 - 0.5 * np.sin(arg), 0.0))
    dfc = np.where(inner, -0.25 * np.pi / D * np.cos(arg), 0.0)
    return fc, dfc


def tersoff_site_terms(pot, lattice, positions, atom_types):
    """E = sum_{i<j} fc(r_ij) [fR + (b_ij + b_ji)/2 fA], the engine's form
    (pair terms from the record of the lower-index atom, (s_i, s_j, s_j);
    beta, n of b_ij from (s_i, s_j, s_j); angular terms from (s_i, s_j,
    s_k), the i-k cutoff from (s_i, s_k, s_k)).  Site split:
    U_i += fc (fR + b_ij fA) / 2, U_j += fc (fR + b_ji fA) / 2.
    """
    species = list(pot.species)
    rec = pot.p
    ns = len(species)
    idx = {s: k for k, s in enumerate(species)}
    t = np.array([idx[a] for a in atom_types], dtype=np.int64)

    tb_ok = np.zeros((ns, ns), bool)
    P2 = {
        k: np.zeros((ns, ns))
        for k in ("A", "B", "lambda1", "lambda2", "R", "D", "beta", "n")
    }
    P2["D"][:] = 1.0
    P2["n"][:] = 1.0
    for x, si in enumerate(species):
        for y, sj in enumerate(species):
            r_ = rec.get((si, sj, sj))
            if r_ is None:
                continue
            tb_ok[x, y] = True
            for k in ("A", "B", "lambda1", "lambda2", "R", "D"):
                if k in r_:
                    P2[k][x, y] = float(r_[k])
            P2["beta"][x, y] = float(r_.get("beta", 0.0))
            P2["n"][x, y] = float(r_.get("n", 1.0))
    tri_ok = np.zeros((ns, ns, ns), bool)
    P3 = {k: np.zeros((ns, ns, ns)) for k in ("c", "d", "h", "lambda3", "m", "gamma")}
    P3["d"][:] = 1.0
    for x, si in enumerate(species):
        for y, sj in enumerate(species):
            for z, sk in enumerate(species):
                r_ = rec.get((si, sj, sk))
                if r_ is None or not tb_ok[x, z]:
                    continue
                tri_ok[x, y, z] = True
                for k in ("c", "d", "h", "lambda3"):
                    P3[k][x, y, z] = float(r_[k])
                P3["m"][x, y, z] = float(r_.get("m", 3.0))
                P3["gamma"][x, y, z] = float(r_.get("gamma", 1.0))

    N = len(positions)
    ii, jj, v, r = _pairs(
        lattice, positions, float(pot.r_cut), "TersoffPotential.site_terms"
    )
    centre, nbr, vec, rr, _ = _entries(ii, jj, v, r)
    M = len(centre)
    tc, tn = t[centre], t[nbr]
    # the pair exists iff the lower-index atom's record does
    lo = np.minimum(centre, nbr)
    hi = np.maximum(centre, nbr)
    tlo, thi = t[lo], t[hi]
    bond_ok = tb_ok[tlo, thi]

    # zeta per entry (bond centre -> nbr), sum over the centre's other entries
    m1, m2 = _ordered_entry_pairs(centre, N)
    ts, tj, tk = t[centre[m1]], t[nbr[m1]], t[nbr[m2]]
    ok = bond_ok[m1] & tri_ok[ts, tj, tk]
    m1, m2, ts, tj, tk = m1[ok], m2[ok], ts[ok], tj[ok], tk[ok]
    fck, dfck = _fc(rr[m2], P2["R"][ts, tk], P2["D"][ts, tk])
    keep = (fck != 0.0) | (dfck != 0.0)
    m1, m2, ts, tj, tk = m1[keep], m2[keep], ts[keep], tj[keep], tk[keep]
    fck, dfck = fck[keep], dfck[keep]
    c_, d_, h_ = P3["c"][ts, tj, tk], P3["d"][ts, tj, tk], P3["h"][ts, tj, tk]
    l3, mm, gam = (
        P3["lambda3"][ts, tj, tk],
        P3["m"][ts, tj, tk],
        P3["gamma"][ts, tj, tk],
    )
    u, w = vec[m1], vec[m2]
    ru, rw = rr[m1], rr[m2]
    cos, dcu, dcw = _cos_grad(u, w, ru, rw)
    X = d_**2 + (h_ - cos) ** 2
    g = 1.0 + c_**2 / d_**2 - c_**2 / X
    dg = -2.0 * c_**2 * (h_ - cos) / X**2
    dr_ = ru - rw
    ex = np.exp(l3**mm * dr_**mm)
    dex = ex * mm * l3**mm * dr_ ** (mm - 1.0)  # d ex / d r_u
    z = gam * fck * g * ex
    dz_du = (gam * fck * ex * dg)[:, None] * dcu + (gam * fck * g * dex / ru)[
        :, None
    ] * u
    dz_dw = (
        (gam * dfck * g * ex / rw)[:, None] * w
        + (gam * fck * ex * dg)[:, None] * dcw
        - (gam * fck * g * dex / rw)[:, None] * w
    )
    zeta = np.zeros(M)
    np.add.at(zeta, m1, z)

    # bond order of each entry, beta and n from (s_centre, s_nbr, s_nbr)
    beta, nn = P2["beta"][tc, tn], P2["n"][tc, tn]
    b = np.ones(M)
    dbdz = np.zeros(M)
    act = (zeta > 0) & (beta != 0.0) & tb_ok[tc, tn]
    bz = (beta[act] * zeta[act]) ** nn[act]
    b[act] = (1.0 + bz) ** (-0.5 / nn[act])
    dbdz[act] = -0.5 * b[act] * bz / (zeta[act] * (1.0 + bz))

    # pair functions from the lower-index record
    A, B = P2["A"][tlo, thi], P2["B"][tlo, thi]
    l1, l2 = P2["lambda1"][tlo, thi], P2["lambda2"][tlo, thi]
    fc, dfc = _fc(rr, P2["R"][tlo, thi], P2["D"][tlo, thi])
    fR = A * np.exp(-l1 * rr)
    fA = -B * np.exp(-l2 * rr)
    act = bond_ok & ((fc != 0.0) | (dfc != 0.0))
    fc, dfc, fR, fA = (np.where(act, a_, 0.0) for a_ in (fc, dfc, fR, fA))

    e = np.zeros(N)
    np.add.at(e, centre, 0.5 * fc * (fR + b * fA))
    dEdr = 0.5 * (dfc * (fR + b * fA) + fc * (-l1 * fR - l2 * b * fA))
    dUdv = (dEdr / rr)[:, None] * vec
    pref = 0.5 * fc * fA * dbdz  # dU_centre / dzeta
    np.add.at(dUdv, m1, pref[m1][:, None] * dz_du)
    np.add.at(dUdv, m2, pref[m1][:, None] * dz_dw)
    return SiteTerms(e, centre, nbr, vec, dUdv)


# ---------------------------------------------------------------------------
#  EAM
# ---------------------------------------------------------------------------


def eam_site_terms(pot, lattice, positions, atom_types):
    """U_i = F_i(rho_bar_i) + (1/2) sum_j phi_ij(r_ij), rho_bar_i =
    sum_j rho_{s_j}(r_ij) (the neighbour's density function)."""
    N = len(positions)
    ii, jj, v, r = _pairs(
        lattice, positions, float(pot.r_cut), "EAMPotential.site_terms"
    )
    centre, nbr, vec, rr, _ = _entries(ii, jj, v, r)
    sp = np.array(atom_types)
    rho_v = np.zeros(len(rr))
    drho_v = np.zeros(len(rr))
    for s, (f, df) in pot._rho.items():
        m = sp[nbr] == s
        if np.any(m):
            rho_v[m], drho_v[m] = pot._eval(f, df, rr[m])
    rho_bar = np.zeros(N)
    np.add.at(rho_bar, centre, rho_v)
    rho_safe = np.maximum(rho_bar, 0.0)
    Fv = np.zeros(N)
    dF = np.zeros(N)
    for s, (f, df) in pot._F.items():
        m = sp == s
        if np.any(m):
            Fv[m], dF[m] = pot._eval(f, df, rho_safe[m])
    phi = np.zeros(len(rr))
    dphi = np.zeros(len(rr))
    for key, (f, df) in pot._phi.items():
        m = np.array(
            [frozenset({a, b}) == key for a, b in zip(sp[centre], sp[nbr])], dtype=bool
        )
        if np.any(m):
            phi[m], dphi[m] = pot._eval(f, df, rr[m])
    e = Fv.copy()
    np.add.at(e, centre, 0.5 * phi)
    coef = dF[centre] * drho_v + 0.5 * dphi
    return SiteTerms(e, centre, nbr, vec, (coef / rr)[:, None] * vec)
