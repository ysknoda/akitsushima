"""Space-group operations for the symmetry reduction of phonon
displacements and elastic strains (ADR-0040).

spglib only supplies the operations (rotation and translation in the
fractional coordinates of the input cell).  Everything built on them --
the atom permutations, the Cartesian rotations, the choice of
displacement directions and strain modes -- is in this module and in the
two solvers that use it.  spglib is an optional extra
(``pip install akitsushima[symmetry]``).
"""

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DEFAULT_SYMPREC",
    "SymmetryOperations",
    "greedy_spanning",
    "require_spglib",
    "spglib_operations",
]

# spglib's own default distance tolerance [A].
DEFAULT_SYMPREC = 1e-5

# The atom mapping accepts images up to MAPPING_FACTOR * symprec away
# (ADR-0041 decision 6).  spglib's own overlap test does not bound the
# Cartesian distance of every atom by symprec: on 12 variable-cell-relaxed
# Si / AlN supercells at symprec 1e-5 to 1e-3 the largest residual of an
# operation spglib accepted was 1.53 symprec (1.08 in a second case), and
# a strict 1 * symprec check refused both.  3 leaves a factor ~2 of margin;
# the permutation must still be a bijection between atoms of one kind.
MAPPING_FACTOR = 3.0


def require_spglib():
    """Return the spglib module, or stop with the install hint."""
    try:
        import spglib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "symmetry: true needs spglib, which is not installed.  "
            "Install it with `pip install akitsushima[symmetry]` "
            "(or `pip install spglib`)."
        ) from exc
    return spglib


def _spglib_numbers(
    atom_types: Sequence[str], masses: np.ndarray | None
) -> np.ndarray:
    """One integer per distinct (species, mass) pair (ADR-0040 decision 4),
    so that atoms of one species with unequal masses are not equivalent."""
    keys = []
    for k, sp in enumerate(atom_types):
        m = None if masses is None else round(float(masses[k]), 10)
        keys.append((str(sp), m))
    table = {key: i + 1 for i, key in enumerate(dict.fromkeys(keys))}
    return np.array([table[key] for key in keys], dtype=np.int64)


@dataclass
class SymmetryOperations:
    """Space-group operations of one periodic cell.

    Attributes
    ----------
    lattice   : (3, 3) lattice vectors as rows [A].
    rot_frac  : (n_ops, 3, 3) integer rotations acting on fractional
                column vectors: f' = R f + t.
    trans     : (n_ops, 3) fractional translations.
    rot_cart  : (n_ops, 3, 3) the same rotations on Cartesian vectors.
    perm      : (n_ops, N) the image of each atom: op k maps atom i onto
                atom perm[k, i] ...
    shift     : (n_ops, N, 3) ... translated by the integer lattice
                vector shift[k, i]: R f_i + t = f_perm + shift.
    symbol, number : international symbol and space-group number.
    symprec   : the tolerance used [A].
    max_residual : largest distance between an atom's image and the atom
                it is mapped onto, over all operations [A].
    """

    lattice: np.ndarray
    rot_frac: np.ndarray
    trans: np.ndarray
    rot_cart: np.ndarray
    perm: np.ndarray
    shift: np.ndarray
    symbol: str
    number: int
    symprec: float
    n_found: int = field(default=0)
    max_residual: float = field(default=0.0)

    @property
    def n_ops(self) -> int:
        return len(self.rot_frac)

    @property
    def is_identity_only(self) -> bool:
        return self.n_ops == 1

    def keep(self, mask: np.ndarray) -> "SymmetryOperations":
        """The subset of operations selected by a boolean mask."""
        mask = np.asarray(mask, dtype=bool)
        return SymmetryOperations(
            self.lattice,
            self.rot_frac[mask],
            self.trans[mask],
            self.rot_cart[mask],
            self.perm[mask],
            self.shift[mask],
            self.symbol,
            self.number,
            self.symprec,
            self.n_found,
            self.max_residual,
        )

    def summary(self) -> dict:
        return {
            "spacegroup": self.symbol,
            "number": self.number,
            "n_operations": self.n_found,
            "n_operations_used": self.n_ops,
            "symprec_A": self.symprec,
            "max_mapping_residual_A": self.max_residual,
        }


def _atom_mapping(lattice, frac, numbers, R, t, tol, chunk=256):
    """perm, shift, largest residual of one operation; None when an atom
    has no image within ``tol`` [A] or the mapping is not a bijection.

    Distances are taken in chunks of rows so that a large cell (an
    elastic supercell of a few thousand atoms) needs no N x N x 3 array.
    """
    img = frac @ R.T + t                                  # (N, 3)
    n_at = len(frac)
    perm = np.empty(n_at, dtype=np.int64)
    shift = np.empty((n_at, 3), dtype=np.int64)
    worst = 0.0
    for a in range(0, n_at, chunk):
        d = img[a:a + chunk, None, :] - frac[None, :, :]
        n = np.round(d)
        dist = np.linalg.norm((d - n) @ lattice, axis=2)
        dist[numbers[a:a + chunk, None] != numbers[None, :]] = np.inf
        best = np.argmin(dist, axis=1)
        rows = np.arange(len(best))
        d_best = dist[rows, best]
        if np.any(d_best > tol):
            return None
        worst = max(worst, float(d_best.max()))
        perm[a:a + chunk] = best
        shift[a:a + chunk] = n[rows, best]
    if len(np.unique(perm)) != n_at:
        return None
    return perm, shift, worst


def spglib_operations(
    lattice,
    positions,
    atom_types,
    masses: np.ndarray | None = None,
    symprec: float = DEFAULT_SYMPREC,
    unique_rotations: bool = False,
) -> SymmetryOperations:
    """Space-group operations of a cell, with the atom permutation of each.

    The structure is not symmetrised: positions that are only nearly
    symmetric give a lower group.  An operation that spglib reports but
    whose atom mapping fails at ``symprec`` is an error (ADR-0040
    decision 4), not a silent skip.

    With ``unique_rotations`` only the first operation of each distinct
    rotation is kept.  That is enough when only the rotation acts on the
    quantity (strain and stress) and the permutation only has to be some
    valid one; it drops the pure translations of a supercell, which would
    otherwise multiply the work by the number of cells.
    """
    spglib = require_spglib()
    lattice = np.asarray(lattice, dtype=float)
    positions = np.asarray(positions, dtype=float)
    frac = positions @ np.linalg.inv(lattice)
    wrap = np.floor(frac).astype(np.int64)
    frac = frac - wrap
    numbers = _spglib_numbers(atom_types, masses)
    cell = (lattice, frac, numbers)
    with warnings.catch_warnings():
        # spglib >= 2.5 warns about its old error handling on every call.
        warnings.simplefilter("ignore", DeprecationWarning)
        sym = spglib.get_symmetry(cell, symprec=symprec)
        sg = spglib.get_spacegroup(cell, symprec=symprec)
    if sym is None or sg is None:
        raise ValueError(
            f"spglib found no symmetry for this cell at symprec = {symprec} A."
        )
    rots = np.asarray(sym["rotations"], dtype=np.int64)
    trans = np.asarray(sym["translations"], dtype=float)
    n_found = len(rots)
    if unique_rotations:
        _, first = np.unique(rots.reshape(len(rots), 9), axis=0,
                             return_index=True)
        first = np.sort(first)
        rots, trans = rots[first], trans[first]
    # Cartesian: x = L^T f, so R_cart = L^T R L^-T.
    LT = lattice.T
    LT_inv = np.linalg.inv(LT)
    rot_cart = np.einsum("ij,njk,kl->nil", LT, rots, LT_inv)
    perms, shifts, worst = [], [], 0.0
    tol = MAPPING_FACTOR * symprec
    for k in range(len(rots)):
        m = _atom_mapping(lattice, frac, numbers, rots[k], trans[k], tol)
        if m is None:
            raise ValueError(
                f"symmetry: spglib operation {k} (rotation "
                f"{rots[k].tolist()}, translation {np.round(trans[k], 6).tolist()}) "
                f"does not map every atom onto a distinct atom of the same kind "
                f"within {MAPPING_FACTOR:g} x symprec = {tol:g} A.  Tighten "
                f"symprec or symmetrise the structure."
            )
        worst = max(worst, m[2])
        # The mapping used wrapped coordinates; express the shift for the
        # positions as given: R (fw_i + w_i) + t = (fw_p + w_p) + shift.
        perms.append(m[0])
        shifts.append(m[1] - wrap[m[0]] + wrap @ rots[k].T)
    symbol, _, num = sg.partition(" (")
    return SymmetryOperations(
        lattice=lattice,
        rot_frac=rots,
        trans=trans,
        rot_cart=rot_cart,
        perm=np.array(perms, dtype=np.int64),
        shift=np.array(shifts, dtype=np.int64),
        symbol=symbol.strip(),
        number=int(num.rstrip(")") or 0),
        symprec=float(symprec),
        n_found=n_found,
        max_residual=worst,
    )


def greedy_spanning(
    candidates: Iterable[np.ndarray], images, dim: int, tol: float = 1e-8
) -> list[int]:
    """Indices of the candidates kept, in order, until the images span
    ``dim`` dimensions.

    ``images(c)`` returns an (m, dim) array: every symmetry image of the
    candidate vector c.  A candidate is kept when its images raise the
    rank of the set collected so far (ADR-0040 decisions 5 and 6).
    """
    kept, rows, rank = [], np.zeros((0, dim)), 0
    for k, c in enumerate(candidates):
        new = np.vstack([rows, images(np.asarray(c, dtype=float))])
        r = int(np.linalg.matrix_rank(new, tol=tol))
        if r > rank:
            kept.append(k)
            rows, rank = new, r
        if rank == dim:
            break
    if rank < dim:  # pragma: no cover - cannot happen with a basis
        raise RuntimeError(
            f"greedy_spanning: the candidates do not span {dim} dimensions."
        )
    return kept


def strain_vector(eps: np.ndarray) -> np.ndarray:
    """Engineering Voigt 6-vector of a symmetric strain tensor
    (xx, yy, zz, 2yz, 2xz, 2xy), the convention of the elastic solver."""
    return np.array(
        [
            eps[0, 0],
            eps[1, 1],
            eps[2, 2],
            2.0 * eps[1, 2],
            2.0 * eps[0, 2],
            2.0 * eps[0, 1],
        ]
    )


def stress_vector(sig: np.ndarray) -> np.ndarray:
    """Voigt 6-vector of a symmetric stress tensor (xx, yy, zz, yz, xz, xy)."""
    return np.array([sig[0, 0], sig[1, 1], sig[2, 2], sig[1, 2], sig[0, 2], sig[0, 1]])
