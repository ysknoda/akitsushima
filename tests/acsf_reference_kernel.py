"""The ACSF kernel of 0.2.0a19, kept unchanged as a reference (ADR-0030).

``_acsf_kernel`` here is a verbatim copy of the kernel in
``akitsushima/fitting/descriptors/acsf.py`` before ADR-0030 reorganised
its angular loop.  ``tests/test_acsf_kernel_reference.py`` checks that
the package kernel gives the same G and dGdv to relative 1e-10 for any
structure and descriptor grid.  Do not edit it to follow later changes
of the package kernel: that would remove the check.
"""

import math

import numpy as np

from akitsushima.utils.jit import _numba_njit, _numba_par
from akitsushima.fitting.descriptors.acsf import _prange


@_numba_njit
def _fcut(r, pi_rc, r_off, r_on):
    """Cutoff factor and its r-derivative: the cosine cutoff fc(r) times,
    when r_on > 0, the inner switch s(r) = 0 (r <= r_off), 1 (r >= r_on),
    t^3 (10 - 15 t + 6 t^2) between, t = (r - r_off) / (r_on - r_off)
    (C2; ADR-0014: pairs closer than r_off are invisible to the model)."""
    fc = 0.5 * (math.cos(pi_rc * r) + 1.0)
    dfc = -0.5 * pi_rc * math.sin(pi_rc * r)
    if r_on <= 0.0 or r >= r_on:
        return fc, dfc
    if r <= r_off:
        return 0.0, 0.0
    w = r_on - r_off
    t = (r - r_off) / w
    sw = t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
    dsw = 30.0 * t * t * (1.0 - t) * (1.0 - t) / w
    return fc * sw, dfc * sw + fc * dsw


@_numba_par
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
    r_off,
    r_on,
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
    # atoms are independent: atom i writes only G[i] and its own entries
    for i in _prange(N):
        s0 = ptr[i]
        s1 = ptr[i + 1]
        # ---- G2 ----
        for m in range(s0, s1):
            j = nbj[m]
            vx = vec[m, 0]
            vy = vec[m, 1]
            vz = vec[m, 2]
            r = math.sqrt(vx * vx + vy * vy + vz * vz)
            fc, dfc = _fcut(r, pi_rc, r_off, r_on)
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
            fa, dfa = _fcut(ra, pi_rc, r_off, r_on)
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
                fb, dfb = _fcut(rb, pi_rc, r_off, r_on)
                if in_jk:
                    fcc, dfcc = _fcut(rc_, pi_rc, r_off, r_on)
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
