"""Force-based static Delta eps (ADR-0003 item 14, ADR-0009).

Atomic forces F_k are projected on the same Gamma modes as the positions,
    G_m = sum_k e_m,s(k) . F_k / sqrt(m_k N_c)           [eV / (A sqrt(amu))],
the generalised force conjugate to Q_m.  Per pair of Cartesian components
an effective harmonic matrix is fitted in the subspace of the modes that
carry M_a or M_b,
    D_eff = -<dG dQ^T> <dQ dQ^T>^-1   (symmetrised),
and Delta eps_ab = p_a D_eff^-1 p_b^T / (eps0 V).  It uses positions and
forces only: exact for a harmonic crystal whatever the mode energies,
includes the anharmonic renormalisation (TDEP-like, q = 0 only), and
needs no temperature.  Validated as the Tier 2a / 3a reference (ADR-0006).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import AMU, ANGSTROM, E_CHARGE, EPS0

HEAD = "TOTAL-FORCE (eV/Angst)"


def iter_outcar_forces(path: str | Path, n_atoms: int):
    """Positions [A] and forces [eV/A] of every ionic step of a VASP OUTCAR, streamed."""
    with open(path, errors="replace") as fh:
        for ln in fh:
            if HEAD in ln:
                next(fh)
                a = np.array([next(fh).split()[:6] for _ in range(n_atoms)], dtype=float)
                yield a[:, :3], a[:, 3:]


def weights(w: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Force projection weights e_m,s(k) / sqrt(m_k N_c) = W / m_k, shape (n_atoms, 3, n_ir)."""
    return w / np.asarray(masses, float)[:, None, None]


def project_outcar(path, n_atoms: int, wg: np.ndarray, every: int, chunk: int = 2000):
    """Generalised forces (n_kept, n_ir) at steps every, 2 every, ... and the first kept positions."""
    out, buf, first, step = [], [], None, 0
    for pos, f in iter_outcar_forces(path, n_atoms):
        step += 1
        if step % every:
            continue
        if first is None:
            first = pos
        buf.append(f)
        if len(buf) == chunk:
            out.append(np.einsum("tki,kim->tm", np.array(buf), wg))
            buf = []
    if buf:
        out.append(np.einsum("tki,kim->tm", np.array(buf), wg))
    if not out:
        raise ValueError(f"{path}: no '{HEAD}' blocks")
    return np.concatenate(out), first


def project_npy(path, n_atoms: int, wg: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """Generalised forces (n, n_ir) from an .npy of shape (n_frames, n_atoms, 3) in eV/A."""
    f = np.load(path, mmap_mode="r")
    if f.ndim != 3 or f.shape[1:] != (n_atoms, 3):
        raise ValueError(f"{path}: forces must have shape (n_frames, {n_atoms}, 3), got {f.shape}")
    return np.concatenate(
        [
            np.einsum("tki,kim->tm", np.asarray(f[i : i + chunk]), wg)
            for i in range(0, len(f), chunk)
        ]
    )


def check_alignment(path, pos_outcar: np.ndarray, cart_xdatcar: np.ndarray, lattice: np.ndarray):
    """Refuse forces whose first kept step does not hold the first trajectory frame."""
    d = (pos_outcar - cart_xdatcar) @ np.linalg.inv(lattice)
    d -= np.round(d)
    dev = float(np.abs(d @ lattice).max())
    if dev > 1e-3:
        raise ValueError(
            f"{path}: positions of the first kept force step differ from the first trajectory "
            f"frame by {dev:.3f} A (trajectory.frame_every = NBLOCK? same run?)"
        )


def d_eff(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Symmetrised effective force-constant matrix -<dG dQ^T> <dQ dQ^T>^-1 [eV / (amu A^2)]."""
    dq, dg = q - q.mean(axis=0), g - g.mean(axis=0)
    dm = -(dg.T @ dq) @ np.linalg.inv(dq.T @ dq)
    return 0.5 * (dm + dm.T)


def subspaces(p: np.ndarray, share: float = 1e-4) -> list[np.ndarray]:
    """Modes that carry M_a (|p_am|^2 above `share` of the component total), per component."""
    return [np.flatnonzero(p[a] ** 2 > share * max((p[a] ** 2).sum(), 1e-300)) for a in range(3)]


def static_tensor(q: np.ndarray, g: np.ndarray, p: np.ndarray, volume_a3: float) -> np.ndarray:
    """Delta eps_ab = p_a D_eff^-1 p_b^T / (eps0 V), D_eff fitted on the modes of M_a and M_b.

    Restricting the fit per component keeps IR-silent near-degenerate modes out
    (SiC: folded X modes 0.15 THz from the TO, beat period longer than the run,
    biased an all-optical fit by 2 %; ADR-0006).
    """
    pref = E_CHARGE * ANGSTROM**2 / (EPS0 * volume_a3 * ANGSTROM**3)  # e^2 A^2 / eV -> 1
    sub = subspaces(p)
    out = np.zeros((3, 3))
    for a in range(3):
        for b in range(a, 3):
            modes = np.union1d(sub[a], sub[b])
            if len(modes) == 0:
                continue
            dm = d_eff(q[:, modes], g[:, modes])
            out[a, b] = out[b, a] = p[a, modes] @ np.linalg.solve(dm, p[b, modes]) * pref
    return out


def frequencies(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Effective frequencies sqrt(D_eff,mm) [THz] over all given modes (negative: unstable)."""
    d2 = np.diag(d_eff(q, g)) * E_CHARGE / (AMU * ANGSTROM**2)  # s^-2
    return np.sign(d2) * np.sqrt(np.abs(d2)) / (2 * np.pi * 1e12)
