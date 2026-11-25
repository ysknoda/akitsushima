"""Optical quantities from eps(omega) (ADR-0010, F1).

With nozuchi's convention eps = eps' - i eps'' (eps'' > 0 absorbs), the
complex refractive index is N = n - i kappa = sqrt(eps), and for each
diagonal component (lab frame) and the isotropic average (trace / 3):
    R     = |(N - 1) / (N + 1)|^2        normal-incidence reflectivity
    alpha = 2 omega kappa / c            absorption coefficient [cm^-1]
    L     = eps'' / (eps'^2 + eps''^2)   energy-loss function -Im(1/eps)
TO frequencies are the peaks of eps'', LO frequencies those of L; the
Lyddane-Sachs-Teller relation prod(w_LO^2 / w_TO^2) = eps_s / eps_inf is
checked when both have the same number of peaks.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import find_peaks

from .constants import C_CM

PEAK_PROMINENCE = 0.05  # relative to the largest value of the curve
OFFDIAG_WARN = 0.05  # |Delta eps_ab| / mean diagonal Delta eps
LABELS = ("xx", "yy", "zz", "iso")


def _curves(eps_re, eps_im):
    """eps' and eps'' per label: (nfreq, 4) for xx, yy, zz, iso."""
    d_re = np.stack([eps_re[:, i, i] for i in range(3)], axis=1)
    d_im = np.stack([eps_im[:, i, i] for i in range(3)], axis=1)
    return (
        np.column_stack([d_re, d_re.mean(axis=1)]),
        np.column_stack([d_im, d_im.mean(axis=1)]),
    )


def quantities(freq_hz: np.ndarray, eps_re: np.ndarray, eps_im: np.ndarray) -> dict:
    """n, kappa, R, alpha [cm^-1] and loss for xx, yy, zz, iso; each (nfreq, 4)."""
    er, ei = _curves(eps_re, eps_im)
    nc = np.sqrt(er - 1j * ei)  # principal root: Re >= 0, Im <= 0 for ei >= 0
    n, kappa = nc.real, -nc.imag
    r = np.abs((nc - 1.0) / (nc + 1.0)) ** 2
    alpha = 4.0 * np.pi * freq_hz[:, None] * kappa / C_CM
    loss = ei / np.maximum(er**2 + ei**2, 1e-300)
    return {"n": n, "kappa": kappa, "R": r, "alpha_cm-1": alpha, "loss": loss}


def peaks(freq: np.ndarray, y: np.ndarray, prominence: float = PEAK_PROMINENCE) -> list[float]:
    """Frequencies of the local maxima of y with prominence above a fraction of max(y)."""
    top = float(np.nanmax(y)) if len(y) else 0.0
    if not top > 0:
        return []
    idx, _ = find_peaks(y, prominence=prominence * top)
    return freq[idx].tolist()


def analyse(freq, freq_hz, eps_re, eps_im, eps_static, eps_inf) -> tuple[dict, dict]:
    """Optical curves and a JSON summary (TO / LO peaks, LST check) in the frequency unit of freq."""
    q = quantities(freq_hz, eps_re, eps_im)
    _, ei = _curves(eps_re, eps_im)
    pos = freq > 0
    d_static = np.asarray(eps_static) - np.asarray(eps_inf)
    diag = np.diag(d_static)
    off = np.abs(d_static - np.diag(diag)).max()
    if np.mean(np.abs(diag)) > 0 and off > OFFDIAG_WARN * np.mean(np.abs(diag)):
        warnings.warn(
            f"off-diagonal Delta eps_s reaches {off / np.mean(np.abs(diag)):.0%} of the diagonal: "
            "principal axes are not the lab axes; optical quantities use the lab-frame diagonal",
            stacklevel=2,
        )
    es = np.append(np.diag(eps_static), np.trace(eps_static) / 3)
    ein = np.append(np.diag(eps_inf), np.trace(eps_inf) / 3)
    summary = {}
    for k, lab in enumerate(LABELS):
        to = peaks(freq[pos], ei[pos, k])
        lo = peaks(freq[pos], q["loss"][pos, k])
        lst = None
        if to and len(to) == len(lo) and ein[k] > 0:
            lhs = float(np.prod(np.square(lo)) / np.prod(np.square(to)))
            lst = {"prod_lo2_over_to2": lhs, "eps_s_over_eps_inf": float(es[k] / ein[k])}
            lst["ratio"] = lhs / lst["eps_s_over_eps_inf"]
        summary[lab] = {
            "to_peaks": to,
            "lo_peaks": lo,
            "lst": lst,
            "R_max": float(np.nanmax(q["R"][:, k])),
        }
    return q, summary


def write(path, freq, unit: str, q: dict) -> None:
    cols, names = [freq], [f"freq_{unit}"]
    for k, lab in enumerate(LABELS):
        for key in ("n", "kappa", "R", "alpha_cm-1", "loss"):
            cols.append(q[key][:, k])
            names.append(f"{key}_{lab}")
    np.savetxt(path, np.column_stack(cols), header=" ".join(names), fmt="%.8e")
