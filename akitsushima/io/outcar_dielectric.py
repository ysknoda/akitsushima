"""VASP OUTCAR readers for Born effective charges and eps_inf, and the
mapping of the atoms of a cell onto the sites of a smaller unit cell
(ADR-0046 decision 3, ADR-0049).

Ported from nozuchi 0.2.0a2 (nozuchi/charges.py: read_outcar_eps_inf,
read_outcar_born, map_to_unit_cell, rigid_shift), so that both packages
keep one implementation until nozuchi itself is ported.  Behaviour and
tolerances are unchanged; only the POSCAR reader is akitsushima's own.

Convention: Z[k, a, b] = dP_a / du_kb, the order printed in an OUTCAR
("BORN EFFECTIVE CHARGES" block: row a = field / polarisation direction,
column b = displacement direction), the same as solvers.dielectric.
"""

import re
import warnings
from pathlib import Path

import numpy as np

__all__ = ["map_to_unit_cell", "read_outcar_born", "read_outcar_eps_inf", "rigid_shift"]

_EPS_HEAD = (
    "MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)"
)
# LCALCEPS (finite field) prints the tensor without "in DFT" according to
# the VASP wiki LCALCEPS page (read 2026-11-25; ADR-0050 decision 7).  The
# DFPT header is preferred when both are present.
_EPS_HEAD_LCALCEPS = (
    "MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects)"
)


def _last_block(text: str, head: str, n_lines: int) -> list:
    i = text.rfind(head)
    if i < 0:
        raise ValueError(f"'{head}' not found")
    return text[i:].splitlines()[1 : 1 + n_lines]


def read_outcar_eps_inf(path) -> np.ndarray:
    """Electronic dielectric tensor (local field effects in DFT), last
    occurrence in the OUTCAR; the LCALCEPS header without "in DFT" is
    accepted when the DFPT one is absent."""
    text = Path(path).read_text(errors="replace")
    head = _EPS_HEAD if _EPS_HEAD in text else _EPS_HEAD_LCALCEPS
    try:
        lines = _last_block(text, head, 4)
    except ValueError:
        raise ValueError(
            f"{path}: no '{_EPS_HEAD}' block (LEPSILON or LCALCEPS run?)"
        ) from None
    # Skip the dashed separator, but not rows that start with a negative
    # number.
    rows = [ln for ln in lines if ln.strip() and set(ln.strip()) != {"-"}][:3]
    if len(rows) < 3:
        raise ValueError(f"{path}: incomplete eps_inf tensor")
    return np.array([[float(x) for x in r.split()[:3]] for r in rows])


def read_outcar_born(path) -> np.ndarray:
    """Born effective charges Z[k, a, b], shape (n_ions, 3, 3), last
    occurrence in the OUTCAR."""
    text = Path(path).read_text(errors="replace")
    i = text.rfind("BORN EFFECTIVE CHARGES (including local field effects)")
    if i < 0:
        i = text.rfind("BORN EFFECTIVE CHARGES")
    if i < 0:
        raise ValueError(f"{path}: no BORN EFFECTIVE CHARGES block")
    rx = re.compile(
        r"ion\s+(\d+)\s*\n((?:\s*[123]\s+[-+\d.Ee]+\s+[-+\d.Ee]+\s+"
        r"[-+\d.Ee]+\s*\n){3})"
    )
    z = []
    for m in rx.finditer(text, i):
        # Ions are numbered 1, 2, ...; stop at the first break (another
        # table may follow).
        if int(m.group(1)) != len(z) + 1:
            break
        z.append(
            [
                [float(v) for v in ln.split()[1:4]]
                for ln in m.group(2).strip().splitlines()
            ]
        )
    if not z:
        raise ValueError(f"{path}: empty BORN EFFECTIVE CHARGES block")
    return np.array(z)


def map_to_unit_cell(
    frac_sc,
    lattice_sc,
    symbols_sc,
    lattice_u,
    symbols_u,
    frac_u,
    supercell,
    tol: float = 0.6,
) -> np.ndarray:
    """Index of the unit-cell site of every atom of a (super)cell.

    frac_sc : fractional coordinates in the (super)cell.
    supercell : (3,) replication or 3x3 matrix S with L_sc = S L_u.

    The unit cell implied by the supercell lattice, inv(S) @ L_sc, is used
    for the matching, so that small differences between this cell and the
    DFT cell do not matter; a difference above 1 % is warned.  A rigid
    translation of the whole crystal is removed first (rigid_shift).
    """
    S = (
        np.diag(supercell)
        if np.ndim(supercell) == 1
        else np.asarray(supercell, dtype=float)
    )
    n_cells = round(abs(np.linalg.det(S)))
    lattice_u = np.asarray(lattice_u, dtype=float)
    lat_u_sc = np.linalg.inv(S) @ np.asarray(lattice_sc, dtype=float)
    rel = np.abs(lat_u_sc - lattice_u).max() / np.abs(lattice_u).max()
    if rel > 0.01:
        warnings.warn(f"MD unit cell differs from the DFPT cell by {rel:.1%}", stacklevel=2)
    if len(symbols_sc) != n_cells * len(symbols_u):
        raise ValueError(
            f"site mapping: {len(symbols_sc)} atoms != {n_cells} cells x "
            f"{len(symbols_u)} unit-cell atoms (check born_charges.supercell)"
        )
    frac_u = np.asarray(frac_u, dtype=float)
    g = (np.asarray(frac_sc, dtype=float) @ lattice_sc) @ np.linalg.inv(lat_u_sc)
    sym_sc, sym_u = np.array(symbols_sc), np.array(symbols_u)
    shift = rigid_shift(g, sym_sc, frac_u, sym_u, lat_u_sc)
    if np.linalg.norm(shift @ lat_u_sc) > 0.1:
        warnings.warn(
            f"site mapping: removed a rigid translation of the whole crystal "
            f"of {np.linalg.norm(shift @ lat_u_sc):.2f} A (e.g. from a Langevin NVT)",
            stacklevel=2,
        )
    g = g - shift
    site = np.full(len(symbols_sc), -1)
    for s in np.unique(sym_u):
        ia, iu = np.flatnonzero(sym_sc == s), np.flatnonzero(sym_u == s)
        d = g[ia, None, :] - frac_u[None, iu, :]
        d -= np.round(d)
        dist = np.linalg.norm(d @ lat_u_sc, axis=2)
        best = np.argmin(dist, axis=1)
        dmin = dist[np.arange(len(ia)), best]
        if dmin.max() > tol:
            raise ValueError(
                f"site mapping failed for {s}: an atom is {dmin.max():.3f} A "
                f"from every site (tolerance {tol} A)"
            )
        site[ia] = iu[best]
    missing = sorted(set(sym_sc) - set(sym_u))
    if missing:
        raise ValueError(f"site mapping: species {missing} are not in the unit cell")
    counts = np.bincount(site, minlength=len(symbols_u))
    if not np.all(counts == n_cells):
        raise ValueError(
            f"site mapping is not one-to-{n_cells}: counts per site {counts.tolist()}"
        )
    return site


def _residual(g, sym_sc, frac_u, sym_u, lat_u):
    """Displacement (unit-cell fractional) of every atom to its nearest
    same-species site."""
    res = np.zeros_like(g)
    for s in np.unique(sym_u):
        ia, iu = np.flatnonzero(sym_sc == s), np.flatnonzero(sym_u == s)
        d = g[ia, None, :] - frac_u[None, iu, :]
        d -= np.round(d)
        best = np.argmin(np.linalg.norm(d @ lat_u, axis=2), axis=1)
        res[ia] = d[np.arange(len(ia)), best]
    return res


def rigid_shift(g, sym_sc, frac_u, sym_u, lat_u) -> np.ndarray:
    """Rigid translation (unit-cell fractional) of the whole crystal
    relative to the unit cell.

    Every site of the rarest species is tried as the image of its first
    atom; among the shifts that fit as well as the best (mean residual
    within 0.05 A), the smallest is kept, so that a pure symmetry
    translation of a non-primitive cell is not introduced.  The shift is
    then refined by the mean residual.
    """
    rare = min(np.unique(sym_u), key=lambda s: np.sum(sym_u == s))
    a0 = np.flatnonzero(sym_sc == rare)
    if len(a0) == 0:
        return np.zeros(3)
    cands, score = [], []
    for j in np.flatnonzero(sym_u == rare):
        t = g[a0[0]] - frac_u[j]
        t -= np.round(t)
        r = _residual(g - t, sym_sc, frac_u, sym_u, lat_u)
        cands.append(t)
        score.append(np.linalg.norm(r @ lat_u, axis=1).mean())
    score = np.array(score)
    ok = np.flatnonzero(score <= score.min() + 0.05)
    t = min((cands[k] for k in ok), key=lambda v: np.linalg.norm(v @ lat_u))
    for _ in range(3):
        t = t + _residual(g - t, sym_sc, frac_u, sym_u, lat_u).mean(axis=0)
    return t
