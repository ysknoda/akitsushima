"""numba kernels for the EAM potential (ADR-0043).

The EAM functions of ``EAMPotential`` are Python callables.  Those whose
form is known carry a spec, ``fn._ak_spec``, set where they are built:

- ``("spline", x, c)``: a scipy ``CubicSpline`` (PPoly breakpoints x and
  coefficients c, shape (4, n - 1)), evaluated as scipy does (interval
  from ``searchsorted(x, r, side='right') - 1`` clipped to the end
  intervals, i.e. extrapolation by the end polynomials; Horner).
- ``("spline_over_r", x, c)``: the setfl r*phi table, phi = s(r) / r.
- ``("zhou_density", p, r_c, width)``, ``("zhou_pair", p, r_c, width)``,
  ``("zhou_cross", pa, pb, chi, r_c, width)``: Zhou, Johnson and Wadley,
  PRB 69, 144113 (2004), with the C2 quintic switch of eam_zhou.py.
- ``("zhou_embedding", p)``: eqs. (A6)-(A8).

``build_tables`` turns the specs into flat arrays; ``eam_kernel`` then does
the whole force call in one parallel pass:
  1. per half pair (prange): rho contributions, their derivatives, phi;
  2. per atom (prange, full two-way CSR): rho_bar, F, F';
  3. per pair (prange): the pair force;
  4. per atom (prange): the force sum; energy and virial reductions.
The arithmetic is that of the numpy path term by term; only the order of
the sums differs (results agree to rounding, checked to 1e-10).
"""

import numpy as np

from ..utils.jit import _HAS_NUMBA

# function codes
_SPLINE, _SPLINE_R, _Z_DENS, _Z_PAIR, _Z_CROSS, _Z_EMB = 1, 2, 3, 4, 5, 6
# parameter slots of a Zhou element row
_ZKEYS = (
    "r_e",
    "f_e",
    "rho_e",
    "rho_s",
    "alpha",
    "beta",
    "A",
    "B",
    "kappa",
    "lambda",
    "Fn0",
    "Fn1",
    "Fn2",
    "Fn3",
    "F0",
    "F1",
    "F2",
    "F3",
    "eta",
    "F_e",
)
_NZ = len(_ZKEYS)
_IDX = {k: i for i, k in enumerate(_ZKEYS)}


def spec_of(fn):
    """The ``_ak_spec`` of a function, or of a scipy PPoly; None if unknown."""
    sp = getattr(fn, "_ak_spec", None)
    if sp is not None:
        return sp
    try:
        from scipy.interpolate import PPoly
    except ImportError:  # pragma: no cover
        return None
    if isinstance(fn, PPoly) and fn.c.shape[0] == 4 and fn.extrapolate:
        return ("spline", np.asarray(fn.x, float), np.asarray(fn.c, float))
    return None


def _zrow(p):
    return np.array([float(p[k]) for k in _ZKEYS])


def build_tables(specs):
    """Flat arrays for a list of specs: codes (nf,), P (nf, 2*_NZ + 3),
    spline offsets xo (nf,), sizes nx (nf,), X (sum nx,), C (4, sum nx)."""
    nf = len(specs)
    codes = np.zeros(nf, np.int64)
    P = np.zeros((nf, 2 * _NZ + 3))
    xo = np.zeros(nf, np.int64)
    nx = np.zeros(nf, np.int64)
    Xs, Cs, off = [], [], 0
    for f, sp in enumerate(specs):
        kind = sp[0]
        if kind in ("spline", "spline_over_r"):
            codes[f] = _SPLINE if kind == "spline" else _SPLINE_R
            x, c = np.asarray(sp[1], float), np.asarray(sp[2], float)
            xo[f], nx[f] = off, len(x)
            Xs.append(x)
            cc = np.zeros((4, len(x)))
            cc[:, : len(x) - 1] = c
            Cs.append(cc)
            off += len(x)
        elif kind == "zhou_embedding":
            codes[f] = _Z_EMB
            P[f, :_NZ] = _zrow(sp[1])
        else:
            if kind == "zhou_density":
                codes[f] = _Z_DENS
                P[f, :_NZ] = _zrow(sp[1])
                rc, w = sp[2], sp[3]
            elif kind == "zhou_pair":
                codes[f] = _Z_PAIR
                P[f, :_NZ] = _zrow(sp[1])
                rc, w = sp[2], sp[3]
            elif kind == "zhou_cross":
                codes[f] = _Z_CROSS
                P[f, :_NZ] = _zrow(sp[1])
                P[f, _NZ : 2 * _NZ] = _zrow(sp[2])
                P[f, 2 * _NZ + 2] = float(sp[3])
                rc, w = sp[4], sp[5]
            else:
                raise ValueError(f"EAM kernel: unknown spec {kind!r}")
            P[f, 2 * _NZ] = rc
            P[f, 2 * _NZ + 1] = w
    X = np.concatenate(Xs) if Xs else np.zeros(1)
    C = np.concatenate(Cs, axis=1) if Cs else np.zeros((4, 1))
    return codes, P, xo, nx, X, C


# ----------------------------------------------------------------------
#  scalar evaluators (numba)
# ----------------------------------------------------------------------


def _spline_eval(X, C, o, n, r):
    # interval i with x_i <= r < x_{i+1}, clipped to [0, n - 2]
    lo, hi = 0, n
    while lo < hi:  # searchsorted side='right'
        mid = (lo + hi) // 2
        if X[o + mid] <= r:
            lo = mid + 1
        else:
            hi = mid
    i = lo - 1
    i = max(i, 0)
    i = min(i, n - 2)
    dx = r - X[o + i]
    c0, c1, c2, c3 = C[0, o + i], C[1, o + i], C[2, o + i], C[3, o + i]
    v = ((c0 * dx + c1) * dx + c2) * dx + c3
    dv = (3.0 * c0 * dx + 2.0 * c1) * dx + c2
    return v, dv


def _decay(r, re, pre, expo, shift):
    x = r / re
    e = pre * np.exp(-expo * (x - 1.0))
    u = x - shift
    den = 1.0 + u**20
    val = e / den
    dval = (-expo * e / den - e * 20.0 * u**19 / den**2) / re
    return val, dval


def _dens(P, f, o, r):
    # f_e exp[-beta (x - 1)] / (1 + (x - lambda)^20)
    return _decay(r, P[f, o + 0], P[f, o + 1], P[f, o + 5], P[f, o + 9])


def _pair(P, f, o, r):
    a, da = _decay(r, P[f, o + 0], P[f, o + 6], P[f, o + 4], P[f, o + 8])
    b, db = _decay(r, P[f, o + 0], P[f, o + 7], P[f, o + 5], P[f, o + 9])
    return a - b, da - db


def _switch(r, rc, w):
    s = (r - (rc - w)) / w
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0
    S = 1.0 - s**3 * (10.0 - 15.0 * s + 6.0 * s * s)
    dS = -30.0 * s * s * (1.0 - s) ** 2 / w
    return S, dS


def _embed(P, f, rho):
    re, rs, eta = P[f, 2], P[f, 3], P[f, 18]
    rn = 0.85 * re
    if rho < rn:
        z = rho / rn - 1.0
        F = P[f, 10] + P[f, 11] * z + P[f, 12] * z**2 + P[f, 13] * z**3
        dF = (P[f, 11] + 2 * P[f, 12] * z + 3 * P[f, 13] * z**2) / rn
    elif rho >= 1.15 * re:
        t = rho / rs
        lt = np.log(t)
        F = P[f, 19] * (1.0 - eta * lt) * t**eta
        dF = -P[f, 19] * eta * eta * lt * t ** (eta - 1.0) / rs
    else:
        z = rho / re - 1.0
        F = P[f, 14] + P[f, 15] * z + P[f, 16] * z**2 + P[f, 17] * z**3
        dF = (P[f, 15] + 2 * P[f, 16] * z + 3 * P[f, 17] * z**2) / re
    return F, dF


def _feval(f, codes, P, xo, nx, X, C, r):
    code = codes[f]
    if code == _SPLINE:
        return _spline_eval(X, C, xo[f], nx[f], r)
    if code == _SPLINE_R:
        s, ds = _spline_eval(X, C, xo[f], nx[f], r)
        return s / r, (ds * r - s) / (r * r)
    if code == _Z_EMB:
        return _embed(P, f, r)
    if code == _Z_DENS:
        v, dv = _dens(P, f, 0, r)
    elif code == _Z_PAIR:
        v, dv = _pair(P, f, 0, r)
    else:  # _Z_CROSS
        fa, dfa = _dens(P, f, 0, r)
        fb, dfb = _dens(P, f, _NZ, r)
        ua, dua = _pair(P, f, 0, r)
        ub, dub = _pair(P, f, _NZ, r)
        k = P[f, 2 * _NZ + 2]
        q = fb / fa
        dq = (dfb * fa - fb * dfa) / fa**2
        v = 0.5 * k * (q * ua + ub / q)
        dv = 0.5 * k * (dq * ua + q * dua + dub / q - ub * dq / q**2)
    S, dS = _switch(r, P[f, 2 * _NZ], P[f, 2 * _NZ + 1])
    return v * S, dv * S + v * dS


# ----------------------------------------------------------------------
#  the force call
# ----------------------------------------------------------------------


def _eam_kernel(N, ii, jj, rij, r, spc, RHO, FF, PHI, codes, P, xo, nx, X, C):
    """(energy, forces (N, 3), virial sum -sum f (x) r (3, 3)) from half
    pairs ii < jj with rij = r_j - r_i (minimum image) and r = |rij|."""
    npair = r.shape[0]
    rho_i = np.empty(npair)  # density at i from j (rho of j's species)
    drho_j = np.empty(npair)
    rho_j = np.empty(npair)  # density at j from i
    drho_i = np.empty(npair)
    phi = np.empty(npair)
    dphi = np.empty(npair)
    for p in _prange(npair):
        si = spc[ii[p]]
        sj = spc[jj[p]]
        rho_i[p], drho_j[p] = _feval(RHO[sj], codes, P, xo, nx, X, C, r[p])
        rho_j[p], drho_i[p] = _feval(RHO[si], codes, P, xo, nx, X, C, r[p])
        phi[p], dphi[p] = _feval(PHI[si, sj], codes, P, xo, nx, X, C, r[p])
    # two-way CSR: atom a -> its pairs, side +1 (a = i) or -1 (a = j)
    cnt = np.zeros(N + 1, np.int64)
    for p in range(npair):
        cnt[ii[p] + 1] += 1
        cnt[jj[p] + 1] += 1
    for a in range(N):
        cnt[a + 1] += cnt[a]
    fill = cnt[:-1].copy()
    lst = np.empty(2 * npair, np.int64)
    side = np.empty(2 * npair, np.int64)
    for p in range(npair):
        a = ii[p]
        lst[fill[a]] = p
        side[fill[a]] = 1
        fill[a] += 1
        b = jj[p]
        lst[fill[b]] = p
        side[fill[b]] = -1
        fill[b] += 1
    Fv = np.empty(N)
    dF = np.empty(N)
    for a in _prange(N):
        rb = 0.0
        for k in range(cnt[a], cnt[a + 1]):
            p = lst[k]
            rb += rho_i[p] if side[k] == 1 else rho_j[p]
        rb = max(rb, 0.0)
        Fv[a], dF[a] = _feval(FF[spc[a]], codes, P, xo, nx, X, C, rb)
    fij = np.empty((npair, 3))
    for p in _prange(npair):
        c = dphi[p] + dF[ii[p]] * drho_j[p] + dF[jj[p]] * drho_i[p]
        s = c / r[p]
        fij[p, 0] = s * rij[p, 0]
        fij[p, 1] = s * rij[p, 1]
        fij[p, 2] = s * rij[p, 2]
    forces = np.zeros((N, 3))
    for a in _prange(N):
        fx = 0.0
        fy = 0.0
        fz = 0.0
        for k in range(cnt[a], cnt[a + 1]):
            p = lst[k]
            sg = side[k]
            fx += sg * fij[p, 0]
            fy += sg * fij[p, 1]
            fz += sg * fij[p, 2]
        forces[a, 0] = fx
        forces[a, 1] = fy
        forces[a, 2] = fz
    vir = np.zeros((3, 3))
    for p in range(npair):
        for x in range(3):
            for y in range(3):
                vir[x, y] -= fij[p, x] * rij[p, y]
    energy = 0.0
    for a in range(N):
        energy += Fv[a]
    ephi = 0.0
    for p in range(npair):
        ephi += phi[p]
    return energy + ephi, forces, vir


if _HAS_NUMBA:
    import numba as _nb

    _prange = _nb.prange
    _jit = _nb.njit(cache=True)
    _spline_eval = _jit(_spline_eval)
    _decay = _jit(_decay)
    _dens = _jit(_dens)
    _pair = _jit(_pair)
    _switch = _jit(_switch)
    _embed = _jit(_embed)
    _feval = _jit(_feval)
    eam_kernel = _nb.njit(parallel=True, cache=True)(_eam_kernel)
else:  # pragma: no cover - numba is a core dependency
    _prange = range
    eam_kernel = _eam_kernel
