"""Gamma harmonic modes for the mode-resolved estimator (ADR-0003 item 10, ADR-0007).

The supercell displacements are projected on the mass-weighted Gamma
eigenvectors e_m of the unit cell:
    Q_m = sum_k sqrt(m_k) e_m,s(k) . u_k / sqrt(N_c)            [sqrt(amu) A]
and the dipole is M_a = sum_m p_am Q_m (+ const) exactly for constant Z, with
    p_am = sum_k Z_k,ab e_m,s(k),b / sqrt(m_k N_c)               [e / sqrt(amu)].
Only modes that carry dipole (|p_m|^2 above `ir_share` of the total) are kept.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import charges as ch

EIG_HEAD = "Eigenvectors and eigenvalues of the dynamical matrix"


def has_outcar_modes(path: str | Path) -> bool:
    with open(path, errors="replace") as fh:
        return any(EIG_HEAD in ln for ln in fh)


def read_outcar_modes(path: str | Path, n_sites: int) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies [THz, imaginary < 0] and mass-weighted eigenvectors (3n, n, 3), last block."""
    lines = Path(path).read_text(errors="replace").splitlines()
    heads = [k for k, ln in enumerate(lines) if EIG_HEAD in ln]
    if not heads:
        raise ValueError(f"{path}: no '{EIG_HEAD}' block (DFPT with IBRION = 5-8 needed)")
    freqs, vecs, k = [], [], heads[-1]
    rx = re.compile(r"\s*\d+\s+f(/i)?\s*=\s*([\d.]+) THz")
    while len(freqs) < 3 * n_sites:
        if k >= len(lines):
            raise ValueError(f"{path}: eigenvector block ends after {len(freqs)} modes")
        m = rx.match(lines[k])
        if m:
            freqs.append(float(m.group(2)) * (-1 if m.group(1) else 1))
            vecs.append([[float(x) for x in lines[k + 2 + j].split()[3:6]] for j in range(n_sites)])
            k += 2 + n_sites
        k += 1
    return np.array(freqs), np.array(vecs)


def read_modes_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Generic .npz: frequencies_THz (3n,), eigenvectors (3n, n_sites, 3), mass-weighted."""
    d = np.load(path)
    f, e = np.asarray(d["frequencies_THz"], float), np.asarray(d["eigenvectors"], float)
    if e.ndim != 3 or e.shape[0] != len(f) or e.shape[2] != 3:
        raise ValueError(f"{path}: eigenvectors must have shape (n_modes, n_sites, 3)")
    return f, e


def read_phonopy_qpoints(path: str | Path, tol_thz: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Gamma frequencies [THz] and real mass-weighted eigenvectors (3n, n, 3) from a phonopy
    qpoints.yaml written with ``phonopy --qpoints="0 0 0" --eigvecs``.

    phonopy diagonalises a complex Hermitian matrix, so each Gamma eigenvector
    may carry a phase and a degenerate set may be a complex combination of
    real vectors.  Per degenerate set, the real and imaginary parts span the
    real eigenspace; its orthonormal basis (SVD) replaces the set.
    """
    import yaml

    with open(path) as fh:
        d = yaml.safe_load(fh)
    gamma = [ph for ph in d.get("phonon", []) if np.allclose(ph["q-position"], 0.0, atol=1e-8)]
    if not gamma:
        raise ValueError(
            f"{path}: no q = (0, 0, 0) entry (run phonopy --qpoints='0 0 0' --eigvecs)"
        )
    bands = gamma[0]["band"]
    if "eigenvector" not in bands[0]:
        raise ValueError(f"{path}: no eigenvectors (run phonopy with --eigvecs)")
    f = np.array([b["frequency"] for b in bands], float)
    v = np.array([b["eigenvector"] for b in bands], float)  # (3n, n, 3, 2) re / im
    n = v.shape[1]
    c = (v[..., 0] + 1j * v[..., 1]).reshape(len(f), 3 * n)
    e = np.zeros((len(f), 3 * n))
    for g in degenerate_groups(f, tol_thz):
        u, sv, _ = np.linalg.svd(np.concatenate([c[g].real, c[g].imag]).T, full_matrices=False)
        if np.sum(sv > 1e-6 * sv[0]) != len(g):
            raise ValueError(f"{path}: eigenvectors of modes {g + 1} do not span a real space")
        e[g] = u[:, : len(g)].T
    return f, e.reshape(len(f), n, 3)


def degenerate_groups(freq: np.ndarray, tol_thz: float = 1e-3) -> list[np.ndarray]:
    """Index groups of modes whose frequencies agree within tol_thz (symmetry degeneracy)."""
    order = np.argsort(freq)
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if abs(freq[i] - freq[cur[-1]]) <= tol_thz:
            cur.append(i)
        else:
            groups.append(np.array(cur))
            cur = [i]
    groups.append(np.array(cur))
    return groups


def build_projection(mcfg: dict, symbols, frac_mean, lattice, z, masses) -> dict:
    """Projection weights W (n_atoms, 3, n_ir), dipole coefficients p (3, n_ir) and mode data."""
    lat_u, sym_u, frac_u = ch.read_poscar(mcfg["unit_cell"])
    if mcfg["outcar"]:
        f, e = read_outcar_modes(mcfg["outcar"], len(sym_u))
    elif mcfg.get("phonopy"):
        f, e = read_phonopy_qpoints(mcfg["phonopy"])
    else:
        f, e = read_modes_file(mcfg["file"])
    if e.shape[1] != len(sym_u):
        raise ValueError(
            f"modes: {e.shape[1]} sites in the eigenvectors, {len(sym_u)} in the unit cell"
        )
    ortho = np.abs(np.einsum("mai,nai->mn", e, e) - np.eye(len(e))).max()
    if ortho > 1e-3:
        raise ValueError(f"modes: eigenvectors are not orthonormal (max deviation {ortho:.1e})")
    site = ch.map_to_unit_cell(
        frac_mean,
        lattice,
        list(symbols),
        lat_u,
        sym_u,
        frac_u,
        mcfg["supercell"],
        mcfg["map_tolerance"],
    )
    nc = len(symbols) // len(sym_u)
    ek = e[:, site, :]  # (modes, atoms, 3)
    sq = np.sqrt(np.asarray(masses, float))
    p_all = np.einsum("kab,mkb->am", z, ek / sq[None, :, None]) / np.sqrt(nc)
    w2 = (p_all**2).sum(axis=0)
    keep = (
        np.flatnonzero(w2 > float(mcfg["ir_share"]) * w2.sum())
        if w2.sum() > 0
        else np.array([], int)
    )
    return {
        "W": np.ascontiguousarray(
            np.transpose(ek[keep] * sq[None, :, None], (1, 2, 0)) / np.sqrt(nc)
        ),
        "p": p_all[:, keep],
        "freq_THz": f[keep],
        "index": keep + 1,  # 1-based mode numbers of the source
        # symmetry-degenerate sets among the kept modes (indices into the kept list): their mutual
        # cross terms are dropped by the estimator (zero in equilibrium, basis-dependent otherwise)
        "groups": [g for g in degenerate_groups(f[keep]) if len(g) > 1] if len(keep) else [],
        "n_ir_per_component": [
            int(np.sum(p_all[a, keep] ** 2 > float(mcfg["ir_share"]) * (p_all[a] ** 2).sum()))
            for a in range(3)
        ],
    }
