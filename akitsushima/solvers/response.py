"""Response models: the dipole and the electronic polarisability of a
structure, and their derivatives with respect to the atomic positions
(ADR-0046 decision 5, ADR-0049).

A response model plays the role for spectra that ``force_fn`` plays for
dynamics: from a structure (lattice, positions, atom types) it returns

    born_charges               Z*[k, a, b] = dP_a / du_kb       [e]
    dipole                     M_a = sum_k Z*_k (r_k - r_k^ref)  [e A]
    polarizability             alpha_ab                         [A^3]
    polarizability_derivative  A[k, a, b, c] = d alpha_ab / d u_kc  [A^2]

The harmonic Raman / IR path uses the derivatives at equilibrium; the MD
path of the nozuchi port will use the values frame by frame.  alpha is the
polarisability volume: alpha = V (eps_inf - 1) / (4 pi) for a cell of
volume V.  Response indices come first, displacement index last, as in
Z*.

Implemented: fixed tensors (``FixedResponse``: Z* from YAML or an OUTCAR,
A from VASP finite differences in a later stage) and the bond
polarisability model (``BondPolarizabilityModel``).  A model that cannot
give a quantity raises NotImplementedError; ``has_charges`` and
``has_polarizability`` say what it gives.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BondParams",
    "BondPolarizabilityModel",
    "FixedResponse",
    "ResponseModel",
    "polarizability_from_config",
]


class ResponseModel:
    """Interface of a response model (see the module docstring)."""

    has_charges = False
    has_polarizability = False

    def born_charges(self, lattice, positions, atom_types) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} has no charges")

    def dipole(self, lattice, positions, atom_types, reference_positions) -> np.ndarray:
        """Displacement form M = sum_k Z*_k (r_k - r_k^ref), the
        displacements taken as given (unwrapped coordinates)."""
        Z = self.born_charges(lattice, reference_positions, atom_types)
        u = np.asarray(positions, float) - np.asarray(reference_positions, float)
        return np.einsum("kab,kb->a", Z, u)

    def polarizability(self, lattice, positions, atom_types) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} has no polarisability")

    def polarizability_derivative(self, lattice, positions, atom_types) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} has no polarisability")


class FixedResponse(ResponseModel):
    """Tensors that do not depend on the structure: Z* (N, 3, 3) and / or
    A (N, 3, 3, 3), given for the atoms of one cell in their order."""

    def __init__(
        self,
        born_charges: np.ndarray | None = None,
        polarizability_derivative: np.ndarray | None = None,
    ):
        self._Z = (
            None if born_charges is None else np.asarray(born_charges, dtype=float)
        )
        self._A = (
            None
            if polarizability_derivative is None
            else np.asarray(polarizability_derivative, dtype=float)
        )
        self.has_charges = self._Z is not None
        self.has_polarizability = self._A is not None

    def _check(self, arr, positions, what):
        if arr is None:
            raise NotImplementedError(f"FixedResponse: no {what} given")
        if len(arr) != len(positions):
            raise ValueError(
                f"FixedResponse: {what} for {len(arr)} atoms, "
                f"structure has {len(positions)}"
            )
        return arr

    def born_charges(self, lattice, positions, atom_types):
        return self._check(self._Z, positions, "Born charges").copy()

    def polarizability_derivative(self, lattice, positions, atom_types):
        return self._check(self._A, positions, "polarisability derivative").copy()


# --------------------------------------------------------------------------
# Bond polarisability model
# --------------------------------------------------------------------------


@dataclass
class BondParams:
    """Parameters of one species pair.

    r_cut      : bonds are the pairs closer than this [A] (a sharp cutoff
                 between the first and second neighbour shells).
    r0         : reference bond length [A].
    alpha_diff : alpha_l - alpha_p at r0 [A^3].
    d_alpha_l  : d alpha_l / dR [A^2].
    d_alpha_p  : d alpha_p / dR [A^2].
    """

    r_cut: float
    r0: float
    alpha_diff: float
    d_alpha_l: float
    d_alpha_p: float


def _pair_key(a: str, b: str) -> tuple:
    return tuple(sorted((a, b)))


class BondPolarizabilityModel(ResponseModel):
    """Bond polarisability model (the form used by Umari, Pasquarello and
    Dal Corso, PRB 63, 094305 (2001)): every bond of length R along the
    unit vector n contributes

        alpha_bond = abar(R) I + D(R) (n n^T - I / 3),
        abar(R) = (d_alpha_l + 2 d_alpha_p) / 3 (R - r0),
        D(R)    = alpha_diff + (d_alpha_l - d_alpha_p) (R - r0),

    with D = alpha_l - alpha_p and abar = (alpha_l + 2 alpha_p) / 3.  The
    isotropic constant (alpha_l + 2 alpha_p) / 3 at r0 is not a parameter:
    alpha is defined up to that constant times the number of bonds, which
    no derivative sees.  Three parameters per species pair enter the
    derivative:

        d alpha_ab / d R_c = abar' n_c d_ab + D' n_c (n_a n_b - d_ab / 3)
                             + (D / R) (d_ac n_b + d_bc n_a - 2 n_a n_b n_c)

    and A_j += d alpha / d R, A_i -= d alpha / d R for the bond R = r_j - r_i
    (so sum_k A_k = 0).  Bonds are searched over all periodic images, so
    any cell size works; a bond of an atom with its own image contributes
    to alpha but not to A.
    """

    has_polarizability = True

    def __init__(self, bonds: dict):
        self.bonds = {}
        for key, p in bonds.items():
            pair = (
                key
                if isinstance(key, tuple)
                else tuple(s.strip() for s in str(key).split("-"))
            )
            if len(pair) != 2 or not all(pair):
                raise ValueError(f"polarizability.bonds: key {key!r} is not 'A-B'")
            if not isinstance(p, BondParams):
                unknown = set(p) - set(BondParams.__dataclass_fields__)
                missing = set(BondParams.__dataclass_fields__) - set(p)
                if unknown or missing:
                    raise ValueError(
                        f"polarizability.bonds[{key}]: unknown keys "
                        f"{sorted(unknown)}, missing keys {sorted(missing)}"
                    )
                p = BondParams(**{k: float(v) for k, v in p.items()})
            if p.r_cut <= 0.0 or p.r0 <= 0.0:
                raise ValueError(
                    f"polarizability.bonds[{key}]: r_cut and r0 must be positive"
                )
            k = _pair_key(*pair)
            if k in self.bonds:
                raise ValueError(f"polarizability.bonds: pair {k} given twice")
            self.bonds[k] = p
        if not self.bonds:
            raise ValueError("polarizability.bonds is empty")
        self.r_max = max(p.r_cut for p in self.bonds.values())

    def _bonds(self, lattice, positions, atom_types):
        """Every bond as an ordered pair (i, j, R vector, params), each
        unordered bond twice (i -> j and j -> i)."""
        lattice = np.asarray(lattice, dtype=float)
        pos = np.asarray(positions, dtype=float)
        types = list(atom_types)
        # Image range from the perpendicular widths d_i = V / |a_j x a_k|.
        vol = abs(np.linalg.det(lattice))
        widths = np.array(
            [
                vol
                / np.linalg.norm(np.cross(lattice[(i + 1) % 3], lattice[(i + 2) % 3]))
                for i in range(3)
            ]
        )
        nmax = np.ceil(self.r_max / widths).astype(int) + 1
        img = (
            np.array(np.meshgrid(*[np.arange(-m, m + 1) for m in nmax], indexing="ij"))
            .reshape(3, -1)
            .T
            @ lattice
        )
        frac = pos @ np.linalg.inv(lattice)
        wrapped = (frac - np.floor(frac)) @ lattice
        shift = pos - wrapped  # bond vectors use the given positions
        out_i, out_j, out_R, out_p = [], [], [], []
        species = sorted(set(types))
        tarr = np.array(types)
        for i in range(len(pos)):
            d = (wrapped[None, :, :] + img[:, None, :]) - wrapped[i]  # (n_img, N, 3)
            d = d + (shift[None, :, :] - shift[i])
            r = np.linalg.norm(d, axis=2)
            for sp in species:
                p = self.bonds.get(_pair_key(types[i], sp))
                if p is None:
                    continue
                mask = (r < p.r_cut) & (r > 1e-8) & (tarr == sp)[None, :]
                im, j = np.nonzero(mask)
                out_i.append(np.full(len(j), i))
                out_j.append(j)
                out_R.append(d[im, j])
                out_p.append(
                    np.tile([p.r0, p.alpha_diff, p.d_alpha_l, p.d_alpha_p], (len(j), 1))
                )
        if not out_i:
            return (
                np.zeros(0, int),
                np.zeros(0, int),
                np.zeros((0, 3)),
                np.zeros((0, 4)),
            )
        return (
            np.concatenate(out_i),
            np.concatenate(out_j),
            np.concatenate(out_R),
            np.concatenate(out_p),
        )

    @staticmethod
    def _bond_terms(R, p):
        r = np.linalg.norm(R, axis=1)
        n = R / r[:, None]
        r0, d0, dl, dp = p.T
        abar_p = (dl + 2.0 * dp) / 3.0
        D_p = dl - dp
        D = d0 + D_p * (r - r0)
        abar = abar_p * (r - r0)
        return r, n, abar, abar_p, D, D_p

    def polarizability(self, lattice, positions, atom_types) -> np.ndarray:
        i, j, R, p = self._bonds(lattice, positions, atom_types)
        if len(i) == 0:
            return np.zeros((3, 3))
        r, n, abar, _, D, _ = self._bond_terms(R, p)
        I = np.eye(3)
        nn = np.einsum("ba,bc->bac", n, n)
        alpha = abar[:, None, None] * I + D[:, None, None] * (nn - I / 3.0)
        return 0.5 * alpha.sum(axis=0)  # each bond listed twice

    def polarizability_derivative(self, lattice, positions, atom_types) -> np.ndarray:
        N = len(positions)
        A = np.zeros((N, 3, 3, 3))
        i, j, R, p = self._bonds(lattice, positions, atom_types)
        if len(i) == 0:
            return A
        r, n, _, abar_p, D, D_p = self._bond_terms(R, p)
        I = np.eye(3)
        nn = np.einsum("ba,bc->bac", n, n)
        g = (
            abar_p[:, None, None, None] * np.einsum("ab,kc->kabc", I, n)
            + D_p[:, None, None, None] * np.einsum("kab,kc->kabc", nn - I / 3.0, n)
            + (D / r)[:, None, None, None]
            * (
                np.einsum("ac,kb->kabc", I, n)
                + np.einsum("bc,ka->kabc", I, n)
                - 2.0 * np.einsum("kab,kc->kabc", nn, n)
            )
        )
        g *= 0.5  # each bond listed twice
        np.add.at(A, j, g)
        np.add.at(A, i, -g)
        return A


def polarizability_from_config(cfg: dict | None) -> ResponseModel | None:
    """The response model of a YAML ``polarizability:`` block, or None.

        polarizability:
          source: bpm
          bonds:                      # or params_file: bpm_params.json
            Si-Si: {r_cut: 2.8, r0: 2.35, alpha_diff: ...,
                    d_alpha_l: ..., d_alpha_p: ...}

    ``source: vasp_fd`` (VASP finite differences) and the BPM fit come in
    the next stage (ADR-0046, S2) and are refused here.
    """
    if not cfg:
        return None
    src = str(cfg.get("source", "bpm")).lower()
    if src == "vasp_fd" or "fit_to" in cfg:
        raise ValueError(
            "polarizability: source vasp_fd and fit_to come with "
            "the VASP finite-difference stage (ADR-0046, S2); "
            "give BPM parameters (bonds or params_file) for now"
        )
    if src != "bpm":
        raise ValueError(f"polarizability.source: unknown {src!r} (bpm)")
    unknown = set(cfg) - {"source", "bonds", "params_file"}
    if unknown:
        raise ValueError(f"polarizability: unknown keys {sorted(unknown)}")
    if ("bonds" in cfg) == ("params_file" in cfg):
        raise ValueError("polarizability: give exactly one of bonds and params_file")
    bonds = cfg.get("bonds")
    if "params_file" in cfg:
        import json

        with open(cfg["params_file"]) as fh:
            bonds = json.load(fh)["bonds"]
    return BondPolarizabilityModel(bonds)


def species_without_bonds(model: ResponseModel, atom_types: Sequence[str]) -> list:
    """Species of a BPM that appear in no bond pair (they then add nothing
    to alpha: worth a warning, not an error)."""
    if not isinstance(model, BondPolarizabilityModel):
        return []
    covered = {s for k in model.bonds for s in k}
    return sorted(set(atom_types) - covered)
