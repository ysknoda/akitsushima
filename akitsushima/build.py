"""Structure builders (ADR-0035).

Every function takes a structure dict -- the dict of
:func:`akitsushima.io.parse_poscar` / :func:`read_structure` -- and returns a
new one; the input is never modified.  Functions that draw random numbers
require a ``seed``.  numpy only.

    bulk(type, species, a, ...)         -> a crystal from scratch
    supercell(s, matrix | min_width)    -> repeated cell (cell-major order)
    substitute(s, from_, to, seed)      -> random alloy, exact counts
    vacancy(s, species, count, seed)    -> random vacancies (or indices)
    slab(s, miller, layers, vacuum)     -> (hkl) slab, c along z
    rattle(s, sigma, seed)              -> Gaussian displacements
    strain(s, epsilon | volume, shear)  -> deformed cell, fractions kept
    scale(s, factor | volume)           -> isotropic scaling

``summary(s)`` gives the composition, volume and the shortest distance per
species pair, which the ``build`` command records.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np

from .io.structure import lammps_orientation, structure_dict

__all__ = [
    "BULK_TYPES",
    "bulk",
    "rattle",
    "scale",
    "slab",
    "strain",
    "substitute",
    "summary",
    "supercell",
    "vacancy",
]

BULK_TYPES = ("diamond", "fcc", "bcc", "hcp", "zincblende", "wurtzite", "rocksalt")
_TWO_SPECIES = ("zincblende", "wurtzite", "rocksalt")
_HEXAGONAL = ("hcp", "wurtzite")


def _new(s, lattice, positions, types, fix=None):
    return structure_dict(lattice, positions, types, fix)


def _rng(seed, what):
    if seed is None:
        raise ValueError(f"{what}: seed is required (ADR-0035).")
    return np.random.default_rng(int(seed))


# ---------------------------------------------------------------------------
#  bulk
# ---------------------------------------------------------------------------

_FCC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def bulk(
    type: str,
    species,
    a: float,
    cell: str = "conventional",
    c: float | None = None,
    u: float | None = None,
) -> dict:
    """
    A crystal of ``type`` (``BULK_TYPES``) with lattice constant ``a`` [A].

    species : one element (diamond, fcc, bcc, hcp) or two (zincblende,
              wurtzite, rocksalt; the first on the origin site, the order
              kept).  A string is one element.
    cell    : "conventional" (cubic types: the cubic cell) or "primitive".
              hcp and wurtzite are hexagonal only.
    c, u    : hcp / wurtzite: c [A] (default ideal, c/a = sqrt(8/3)) and the
              wurtzite internal parameter (default 3/8).
    """
    t = str(type).lower()
    if t not in BULK_TYPES:
        raise ValueError(f"bulk: type must be one of {list(BULK_TYPES)}, got '{type}'.")
    sp = [species] if isinstance(species, str) else [str(x) for x in species]
    need = 2 if t in _TWO_SPECIES else 1
    if len(sp) != need:
        raise ValueError(f"bulk: {t} needs {need} species, got {sp}.")
    a = float(a)
    if not a > 0:
        raise ValueError(f"bulk: a must be > 0, got {a}.")
    cell = str(cell).lower()
    if cell not in ("conventional", "primitive"):
        raise ValueError(f"bulk: cell must be conventional | primitive, got '{cell}'.")
    if t in _HEXAGONAL:
        c = a * np.sqrt(8.0 / 3.0) if c is None else float(c)
        lat = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3.0) / 2, 0], [0, 0, c]])
        if t == "hcp":
            frac = np.array([[0, 0, 0], [1 / 3, 2 / 3, 0.5]])
            types = sp * 2
        else:
            u = 0.375 if u is None else float(u)
            frac = np.array(
                [
                    [1 / 3, 2 / 3, 0],
                    [1 / 3, 2 / 3, u],
                    [2 / 3, 1 / 3, 0.5],
                    [2 / 3, 1 / 3, 0.5 + u],
                ]
            )
            types = [sp[0], sp[1], sp[0], sp[1]]
        return _new(None, lat, frac @ lat, types)
    if c is not None or u is not None:
        raise ValueError(f"bulk: c and u apply to hcp / wurtzite, not {t}.")

    q = np.array([0.25, 0.25, 0.25])
    if cell == "conventional":
        lat = np.eye(3) * a
        if t == "fcc":
            frac, types = _FCC, sp * 4
        elif t == "bcc":
            frac, types = np.array([[0, 0, 0], [0.5, 0.5, 0.5]]), sp * 2
        elif t in ("diamond", "zincblende"):
            frac = np.vstack([[f, f + q] for f in _FCC]).reshape(-1, 3)
            types = [sp[0], sp[-1]] * 4
        else:  # rocksalt
            frac = np.vstack([[f, f + [0.5, 0, 0]] for f in _FCC]).reshape(-1, 3)
            types = [sp[0], sp[1]] * 4
        frac = np.mod(frac, 1.0)
        return _new(None, lat, frac @ lat, types)

    if t == "bcc":
        lat = a / 2 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], float)
        return _new(None, lat, np.zeros((1, 3)), sp)
    lat = a / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    if t == "fcc":
        return _new(None, lat, np.zeros((1, 3)), sp)
    if t in ("diamond", "zincblende"):
        return _new(None, lat, np.array([[0, 0, 0], q]) @ lat, [sp[0], sp[-1]])
    return _new(None, lat, np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) @ lat, sp)


# ---------------------------------------------------------------------------
#  supercell
# ---------------------------------------------------------------------------


def _perp_widths(lattice) -> np.ndarray:
    L = np.asarray(lattice, dtype=float)
    vol = abs(np.linalg.det(L))
    return np.array(
        [
            vol / np.linalg.norm(np.cross(L[(i + 1) % 3], L[(i + 2) % 3]))
            for i in range(3)
        ]
    )


def _matrix(matrix) -> np.ndarray:
    M = np.asarray(matrix)
    if M.ndim == 0 or M.shape == (3,):
        M = np.diag(np.broadcast_to(M, 3))
    if M.shape != (3, 3) or not np.all(np.asarray(M) == np.round(M)):
        raise ValueError(
            f"supercell: matrix must be 3 integers or 3x3 integers, got {matrix}."
        )
    M = np.round(M).astype(int)
    if round(np.linalg.det(M)) == 0:
        raise ValueError(f"supercell: singular matrix {M.tolist()}.")
    return M


def _supercell(s, matrix=None, min_width=None):
    if (matrix is None) == (min_width is None):
        raise ValueError("supercell: give matrix or min_width.")
    lattice = np.asarray(s["lattice"], dtype=float)
    positions = np.asarray(s["positions"], dtype=float)
    types = list(s["atom_types"])
    fix = np.asarray(s.get("fix_atoms", np.zeros((len(types), 3), bool)), bool)
    if min_width is not None:
        n = np.ceil(float(min_width) / _perp_widths(lattice) - 1e-12).astype(int)
        matrix = np.maximum(n, 1)
    M = _matrix(matrix)
    det = int(round(np.linalg.det(M)))
    if np.count_nonzero(M - np.diag(np.diag(M))) == 0 and det > 0:
        # diagonal: the order and arithmetic of qha.build_supercell
        S = [int(x) for x in np.diag(M)]
        pos, tt = [], []
        for i in range(S[0]):
            for j in range(S[1]):
                for k in range(S[2]):
                    pos.append(positions + np.array([i, j, k]) @ lattice)
                    tt += types
        new = _new(s, np.diag(S) @ lattice, np.vstack(pos), tt, np.tile(fix, (det, 1)))
        return new, {"matrix": np.diag(S).tolist()}
    if det < 0:
        raise ValueError(
            f"supercell: det(matrix) = {det} < 0 makes a left-handed cell."
        )
    corners = np.array(list(itertools.product((0, 1), repeat=3))) @ M
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    Minv = np.linalg.inv(M)
    pts = []
    for n in itertools.product(*[range(lo[d], hi[d] + 1) for d in range(3)]):
        f = np.array(n) @ Minv
        if np.all(f > -1e-9) and np.all(f < 1 - 1e-9):
            pts.append(n)
    if len(pts) != det:
        raise RuntimeError(
            f"supercell: found {len(pts)} lattice points, expected {det}."
        )
    new_lat = M @ lattice
    inv_new = np.linalg.inv(new_lat)
    pos, tt = [], []
    for n in pts:
        p = positions + np.array(n) @ lattice
        f = p @ inv_new
        f -= np.floor(f + 1e-12)
        pos.append(f @ new_lat)
        tt += types
    new = _new(s, new_lat, np.vstack(pos), tt, np.tile(fix, (det, 1)))
    return new, {"matrix": M.tolist()}


def supercell(s, matrix=None, min_width=None) -> dict:
    """
    Repeat the cell by ``matrix`` (3 integers or a 3x3 integer matrix M;
    new lattice = M @ lattice) or by the smallest diagonal repetition whose
    perpendicular widths are all >= ``min_width`` [A].  Atoms are in
    cell-major order (atom k of cell c is c * N + k).
    """
    return _supercell(s, matrix, min_width)[0]


# ---------------------------------------------------------------------------
#  substitute, vacancy
# ---------------------------------------------------------------------------


def _counts(to: dict, n_from: int, what: str) -> dict:
    out = {}
    for sp, v in to.items():
        if isinstance(v, (bool, np.bool_)):
            raise ValueError(f"{what}: {sp}: a count or a fraction, got {v}.")
        if isinstance(v, (int, np.integer)):
            n = int(v)
        else:
            x = float(v)
            if not 0.0 <= x <= 1.0:
                raise ValueError(f"{what}: {sp}: fraction {x} not in [0, 1].")
            n = int(round(x * n_from))
        if n < 0:
            raise ValueError(f"{what}: {sp}: count {n} < 0.")
        out[str(sp)] = n
    if sum(out.values()) > n_from:
        raise ValueError(f"{what}: {out} asks for more than the {n_from} sites.")
    return out


def _substitute(s, from_, to, seed=None, indices=None):
    types = list(s["atom_types"])
    if indices is not None:
        if not isinstance(to, str):
            raise ValueError("substitute: with indices, to is one species.")
        idx = [int(i) for i in indices]
        if len(set(idx)) != len(idx) or any(not 0 <= i < len(types) for i in idx):
            raise ValueError(f"substitute: bad indices {idx}.")
        if from_ is not None and any(types[i] != from_ for i in idx):
            raise ValueError(f"substitute: not every index is a {from_} atom.")
        for i in idx:
            types[i] = str(to)
        chosen = {str(to): idx}
    else:
        if from_ is None or not isinstance(to, dict):
            raise ValueError(
                "substitute: give from and to: {species: count | fraction}."
            )
        cand = [i for i, t in enumerate(types) if t == from_]
        if not cand:
            raise ValueError(f"substitute: no {from_} atom.")
        counts = _counts(to, len(cand), "substitute")
        perm = _rng(seed, "substitute").permutation(cand)
        chosen, k = {}, 0
        for sp, n in counts.items():
            sel = sorted(int(i) for i in perm[k : k + n])
            for i in sel:
                types[i] = sp
            chosen[sp] = sel
            k += n
    new = _new(s, s["lattice"], s["positions"], types, s.get("fix_atoms"))
    return new, {"indices": chosen}


def substitute(s, from_=None, to=None, seed=None, indices=None) -> dict:
    """
    Random alloy: replace exactly ``to = {species: count | fraction}`` of
    the ``from_`` atoms (an int is a count, a float a fraction of the
    ``from_`` atoms, rounded), chosen with ``seed``; several targets draw
    disjoint sites.  Or replace the atoms ``indices`` by the one species
    ``to``.  No lattice scaling (use :func:`scale`).
    """
    return _substitute(s, from_, to, seed, indices)[0]


def _vacancy(s, species=None, count=None, seed=None, indices=None):
    types = list(s["atom_types"])
    if indices is not None:
        if species is not None or count is not None:
            raise ValueError("vacancy: give indices, or species and count.")
        idx = sorted(int(i) for i in indices)
        if len(set(idx)) != len(idx) or any(not 0 <= i < len(types) for i in idx):
            raise ValueError(f"vacancy: bad indices {idx}.")
    else:
        if species is None or count is None:
            raise ValueError("vacancy: give species and count (and seed), or indices.")
        cand = [i for i, t in enumerate(types) if t == species]
        n = int(count)
        if not 0 <= n <= len(cand):
            raise ValueError(f"vacancy: count {n} for {len(cand)} {species} atoms.")
        idx = sorted(
            int(i) for i in _rng(seed, "vacancy").choice(cand, n, replace=False)
        )
    keep = np.setdiff1d(np.arange(len(types)), idx)
    fix = s.get("fix_atoms")
    new = _new(
        s,
        s["lattice"],
        np.asarray(s["positions"])[keep],
        [types[i] for i in keep],
        None if fix is None else np.asarray(fix)[keep],
    )
    return new, {"removed": idx}


def vacancy(s, species=None, count=None, seed=None, indices=None) -> dict:
    """Remove ``count`` random ``species`` atoms (``seed``) or the atoms
    ``indices``.  No charge compensation."""
    return _vacancy(s, species, count, seed, indices)[0]


# ---------------------------------------------------------------------------
#  rattle, strain, scale
# ---------------------------------------------------------------------------


def rattle(s, sigma: float, seed=None) -> dict:
    """Add N(0, sigma^2) [A] to every Cartesian coordinate."""
    pos = np.asarray(s["positions"], dtype=float)
    d = _rng(seed, "rattle").normal(0.0, float(sigma), pos.shape)
    return _new(s, s["lattice"], pos + d, s["atom_types"], s.get("fix_atoms"))


def _strain(s, epsilon=None, volume=None, shear=None, seed=None):
    L = np.asarray(s["lattice"], dtype=float)
    if epsilon is not None:
        if volume is not None or shear is not None:
            raise ValueError("strain: give epsilon, or volume / shear.")
        eps = np.asarray(epsilon, dtype=float).reshape(3, 3)
        if not np.allclose(eps, eps.T, atol=1e-12):
            raise ValueError("strain: epsilon must be symmetric.")
        F = np.eye(3) + eps
    else:
        if volume is None and shear is None:
            raise ValueError("strain: give epsilon, volume or shear.")
        F = np.eye(3)
        if shear:
            g = _rng(seed, "strain (shear)").uniform(-float(shear), float(shear), 3)
            F = F + np.array([[0, g[2], g[1]], [g[2], 0, g[0]], [g[1], g[0], 0]])
        target = 1.0 + (0.0 if volume is None else float(volume))
        if not target > 0:
            raise ValueError(f"strain: volume change {volume} leaves no volume.")
        F = F * (target / np.linalg.det(F)) ** (1.0 / 3.0)  # volume exact
    if np.linalg.det(F) <= 0:
        raise ValueError("strain: the deformation inverts the cell.")
    frac = np.asarray(s["positions"], dtype=float) @ np.linalg.inv(L)
    new_L = L @ F.T
    new = _new(s, new_L, frac @ new_L, s["atom_types"], s.get("fix_atoms"))
    return new, {"deformation": F.tolist()}


def strain(s, epsilon=None, volume=None, shear=None, seed=None) -> dict:
    """
    Deform the cell by F = I + epsilon (symmetric), or by a random
    symmetric shear (off-diagonals from U(-shear, shear), ``seed``) scaled
    so that the volume changes by exactly ``volume`` (relative).  Each
    lattice vector a becomes F a; fractional coordinates are kept.
    """
    return _strain(s, epsilon, volume, shear, seed)[0]


def scale(s, factor=None, volume=None) -> dict:
    """Scale the cell and positions isotropically by ``factor`` (on
    lengths) or to the ``volume`` [A^3]."""
    if (factor is None) == (volume is None):
        raise ValueError("scale: give factor or volume.")
    L = np.asarray(s["lattice"], dtype=float)
    f = (
        float(factor)
        if factor is not None
        else (float(volume) / abs(np.linalg.det(L))) ** (1.0 / 3.0)
    )
    if not f > 0:
        raise ValueError(f"scale: factor must be > 0, got {f}.")
    return _new(
        s,
        L * f,
        np.asarray(s["positions"], dtype=float) * f,
        s["atom_types"],
        s.get("fix_atoms"),
    )


# ---------------------------------------------------------------------------
#  slab
# ---------------------------------------------------------------------------


def _miller(miller):
    m = [int(x) for x in miller]
    if len(m) == 4:  # Miller-Bravais (h k i l)
        if m[2] != -(m[0] + m[1]):
            raise ValueError(f"slab: (h k i l) needs i = -(h + k), got {m}.")
        m = [m[0], m[1], m[3]]
    if len(m) != 3 or not any(m):
        raise ValueError(
            f"slab: miller must be 3 (or 4) integers, not all 0, got {miller}."
        )
    g = np.gcd.reduce(np.abs(m))
    return [x // g for x in m]


def _plane_basis(hkl):
    """Unimodular W (det +1) whose columns v1, v2 satisfy hkl . v = 0 and
    v3 satisfies hkl . v3 = 1 (integer column reduction)."""
    r = list(hkl)
    W = np.eye(3, dtype=int)
    while sum(1 for x in r if x) > 1:
        p = min((i for i in range(3) if r[i]), key=lambda i: abs(r[i]))
        for j in range(3):
            if j != p and r[j]:
                q = r[j] // r[p]
                r[j] -= q * r[p]
                W[:, j] -= q * W[:, p]
    p = next(i for i in range(3) if r[i])
    if r[p] < 0:
        W[:, p] *= -1
    others = [i for i in range(3) if i != p]
    W = W[:, others + [p]]
    if round(np.linalg.det(W)) < 0:
        W[:, 0] *= -1
    return W


def _reduce_2d(v1, v2, L):
    """Lagrange-Gauss reduction of the integer in-plane basis (metric of L)."""

    def n2(v):
        x = v @ L
        return x @ x

    for _ in range(100):
        if n2(v1) > n2(v2):
            v1, v2 = v2, v1
        mu = int(np.round(((v1 @ L) @ (v2 @ L)) / n2(v1)))
        if mu == 0:
            break
        v2 = v2 - mu * v1
    return v1, v2


def _orient(s, hkl):
    """The cell of s re-based on the (hkl) plane (same atoms) and rotated so
    that a is along x, b in the xy plane; returns (dict, d_hkl)."""
    L = np.asarray(s["lattice"], dtype=float)
    W = _plane_basis(hkl)
    v1, v2, v3 = W[:, 0], W[:, 1], W[:, 2]
    v1, v2 = _reduce_2d(v1, v2, L)
    if np.linalg.det(np.array([v1, v2, v3]) @ L) < 0:
        v1, v2 = v2, v1
    # v3 as close to the normal as integer shifts by v1, v2 allow
    A = np.array([v1 @ L, v2 @ L])
    c = v3 @ L
    coef = np.linalg.lstsq(A.T, c, rcond=None)[0]
    v3 = v3 - int(np.round(coef[0])) * v1 - int(np.round(coef[1])) * v2
    M = np.array([v1, v2, v3])
    o, _ = _supercell(s, M)
    Lp, R = lammps_orientation(o["lattice"])
    o = _new(o, Lp, np.asarray(o["positions"]) @ R, o["atom_types"])
    return o, float(Lp[2, 2])


def _planes(z, d, tol=0.01):
    """Atomic planes of one repeat unit: sorted heights in [0, d) grouped
    within tol, periodic in d.  Returns (heights, label per atom)."""
    z = np.mod(z, d)
    order = np.argsort(z)
    groups, label = [], np.empty(len(z), int)
    for i in order:
        if groups and z[i] - groups[-1][-1] <= tol:
            groups[-1].append(z[i])
        else:
            groups.append([z[i]])
        label[i] = len(groups) - 1
    if len(groups) > 1 and groups[0][0] + d - groups[-1][-1] <= tol:
        last = len(groups) - 1
        label[label == last] = 0
        groups[0] = [x - d for x in groups[-1]] + groups[0]
        groups.pop()
    heights = np.array([np.mean(g) for g in groups])
    return heights, label


def _rect(Lab):
    """Smallest-area integer combinations (a', b') of the in-plane a, b with
    a'.b' = 0 (then the shorter pair); coefficients up to 8."""
    r = np.arange(-8, 9)
    C = np.array([(i, j) for i in r for j in r if (i, j) != (0, 0)])
    V = C @ Lab  # (K, 2) candidate vectors
    dot = V @ V.T
    det = C[:, 0][:, None] * C[:, 1][None, :] - C[:, 1][:, None] * C[:, 0][None, :]
    nrm = np.linalg.norm(V, axis=1)
    ok = (det > 0) & (np.abs(dot) <= 1e-8 * (nrm[:, None] ** 2 + nrm[None, :] ** 2))
    if not ok.any():
        raise ValueError("slab: no rectangular in-plane cell with coefficients up to 8.")
    ia, ib = np.nonzero(ok)
    key = np.lexsort((nrm[ia] + nrm[ib], det[ia, ib]))[0]
    (i, j), (m, n) = C[ia[key]], C[ib[key]]
    return [[int(i), int(j), 0], [int(m), int(n), 0], [0, 0, 1]]


def _slab(
    s,
    miller,
    layers,
    vacuum,
    termination=0,
    center="zero",
    orthogonal_ab=False,
    tol=0.01,
):
    hkl = _miller(miller)
    layers = int(layers)
    if layers < 1:
        raise ValueError(f"slab: layers must be >= 1, got {layers}.")
    vacuum = float(vacuum)
    if vacuum < 0:
        raise ValueError(f"slab: vacuum must be >= 0, got {vacuum}.")
    center = str(center).lower()
    if center not in ("zero", "middle"):
        raise ValueError(f"slab: center must be zero | middle, got '{center}'.")
    o, d = _orient(s, hkl)
    P = o["lattice"]
    pos = np.asarray(o["positions"])
    types = o["atom_types"]
    heights, label = _planes(pos[:, 2], d, tol)
    planes = [
        {
            "height": float(h),
            "composition": {
                sp: int(np.sum((label == k) & (np.array(types) == sp)))
                for sp in dict.fromkeys(types)
            },
        }
        for k, h in enumerate(heights)
    ]
    planes = [
        {
            "height": p["height"],
            "composition": {k: v for k, v in p["composition"].items() if v},
        }
        for p in planes
    ]
    k = int(termination)
    if not 0 <= k < len(heights):
        raise ValueError(
            f"slab: termination {k}; this surface has {len(heights)} planes: {planes}."
        )
    top = heights[k]
    # atoms of the stack of repeat units whose height lies in (top - layers d, top]
    frac3 = pos @ np.linalg.inv(P)
    base = pos - np.outer(np.floor(frac3[:, 2] + 1e-12), P[2])  # heights in [0, d)
    out_p, out_t = [], []
    for n in range(-layers - 1, 2):
        h = base[:, 2] + n * d
        sel = (h > top - layers * d + tol / 2) & (h <= top + tol / 2)
        if np.any(sel):
            out_p.append(base[sel] + n * P[2])
            out_t += [t for t, m in zip(types, sel) if m]
    out_p = np.vstack(out_p)
    if len(out_p) != layers * len(pos):
        raise RuntimeError(
            f"slab: cut {len(out_p)} atoms, expected {layers * len(pos)}; the "
            f"planes of this surface are closer than {tol} A apart."
        )
    zmin, zmax = out_p[:, 2].min(), out_p[:, 2].max()
    Lz = (zmax - zmin) + vacuum
    shift = -(zmin + zmax) / 2 + (Lz / 2 if center == "middle" else 0.0)
    out_p[:, 2] += shift
    lat = np.array([P[0], P[1], [0.0, 0.0, Lz]])
    # wrap in-plane
    f = out_p @ np.linalg.inv(lat)
    f[:, :2] -= np.floor(f[:, :2] + 1e-12)
    out_p = f @ lat
    new = _new(s, lat, out_p, out_t)
    rec = {
        "miller": hkl,
        "d_hkl": d,
        "planes": planes,
        "termination": k,
        "thickness": float(zmax - zmin),
        "Lz": float(Lz),
    }
    cosab = abs(lat[0] @ lat[1]) / (np.linalg.norm(lat[0]) * np.linalg.norm(lat[1]))
    if orthogonal_ab:
        if cosab > 1e-10:
            M = _rect(lat[:2, :2])
            new, _ = _supercell(new, M)
            Lp, R = lammps_orientation(new["lattice"])  # rotation about z
            new = _new(new, Lp, np.asarray(new["positions"]) @ R, new["atom_types"])
            rec["inplane_matrix"] = M
    elif cosab > 1e-10:
        warnings.warn(
            "slab: a and b are not orthogonal; ESM needs a rectangular cell "
            "(orthogonal_ab: true).",
            UserWarning,
            stacklevel=3,
        )
    return new, rec


def slab(
    s,
    miller,
    layers: int,
    vacuum: float,
    termination: int = 0,
    center: str = "zero",
    orthogonal_ab: bool = False,
) -> dict:
    """
    A slab of the (hkl) plane (``miller``: 3 indices of the cell of ``s``,
    or 4 Miller-Bravais indices for a hexagonal cell).

    layers      : repeat units along the normal (one unit is d_hkl thick,
                  possibly several atomic planes).
    vacuum      : total vacuum thickness [A].
    termination : index of the atomic plane (by height within one repeat
                  unit, 0.01 A tolerance) that ends the slab at the top;
                  :func:`slab_planes` lists them.
    center      : "zero" (slab centred on z = 0, as ESM needs) | "middle".
    orthogonal_ab : search the smallest rectangular in-plane cell.

    The cell has a along x, b in the xy plane and c = (0, 0, Lz).
    """
    return _slab(s, miller, layers, vacuum, termination, center, orthogonal_ab)[0]


def slab_planes(s, miller) -> list:
    """The atomic planes of one (hkl) repeat unit: height [A] and
    composition, in the order ``termination`` counts them."""
    o, d = _orient(s, _miller(miller))
    heights, label = _planes(np.asarray(o["positions"])[:, 2], d)
    types = np.array(o["atom_types"])
    return [
        {
            "height": float(h),
            "composition": {
                sp: int(np.sum((label == k) & (types == sp)))
                for sp in dict.fromkeys(o["atom_types"])
                if np.any((label == k) & (types == sp))
            },
        }
        for k, h in enumerate(heights)
    ]


# ---------------------------------------------------------------------------
#  summary
# ---------------------------------------------------------------------------


def summary(s) -> dict:
    """Composition, atom count, volume [A^3] and the shortest distance [A]
    per species pair (periodic images within one cell around)."""
    from scipy.spatial import cKDTree

    L = np.asarray(s["lattice"], dtype=float)
    pos = np.asarray(s["positions"], dtype=float)
    types = np.array(s["atom_types"])
    f = pos @ np.linalg.inv(L)
    pos = (f - np.floor(f)) @ L
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3))) @ L
    species = list(dict.fromkeys(s["atom_types"]))
    dmin = {}
    for ia, a in enumerate(species):
        pa = pos[types == a]
        for b in species[ia:]:
            pb = pos[types == b]
            img = (pb[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
            dist, _ = cKDTree(img).query(pa, k=2 if a == b else 1)
            dd = dist[:, 1] if a == b else dist
            dd = dd[np.isfinite(dd)]
            dmin[f"{a}-{b}"] = float(dd.min()) if dd.size else None
    return {
        "natoms": len(types),
        "composition": {sp: int(np.sum(types == sp)) for sp in species},
        "volume": float(abs(np.linalg.det(L))),
        "min_distance": dmin,
    }
