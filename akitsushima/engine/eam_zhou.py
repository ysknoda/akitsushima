"""Analytic alloy EAM of Zhou, Johnson and Wadley (ADR-0015).

X. W. Zhou, R. A. Johnson and H. N. G. Wadley, Phys. Rev. B 69, 144113
(2004), Appendix A, eqs. (A3)-(A8); parameters in
published_parameters.ZHOU_2004 (Table III, all 16 elements).

    phi_aa(r) = A exp[-alpha (x - 1)] / (1 + (x - kappa)^20)
              - B exp[-beta  (x - 1)] / (1 + (x - lambda)^20),  x = r / r_e
    f(r)      = f_e exp[-beta (x - 1)] / (1 + (x - lambda)^20)
    phi_ab(r) = 1/2 [ f_b/f_a phi_aa + f_a/f_b phi_bb ]            (A5)
    F(rho)    = sum_i Fn_i (rho/rho_n - 1)^i       rho <  rho_n = 0.85 rho_e
              = sum_i F_i  (rho/rho_e - 1)^i       rho_n <= rho < rho_0 = 1.15 rho_e
              = F_e [1 - ln (rho/rho_s)^eta] (rho/rho_s)^eta    rho_0 <= rho

Continuity.  The paper requires F and dF/drho to match at rho_n and rho_0.
With F_1 = 0 (every row of Table III) these four conditions fix Fn0, Fn1,
F0 and F_e from Fn2, Fn3, F2, F3, eta, rho_e, rho_s (``continuous``).
For Ni and Al the re-derived values differ from the printed ones by at
most 2e-5 eV (F0 is printed to two decimals); for Pt and W they do not
(printed Pt jumps 2.6 eV at rho_n; W's negative eta cannot meet rho_0),
see ADR-0015.

Cutoff.  The functions have none; they decay through the 20th-power
denominators.  ``zhou_cutoff`` returns the distance beyond which every
|phi_aa| / max(A, B) and f / f_e stay below ``tol`` (1e-6), and every
function is multiplied by a C2 quintic switch over the last ``width``
(0.5 A) before it, so energy, forces and their first derivatives are
continuous at r_c.  phi_ab is formed from the UNSWITCHED functions and then
switched, so the ratio f_b/f_a never divides two underflowed numbers.

Optional scaling (ADR-0015 decision Q8(b), not in the paper):
phi_ab -> chi_ab phi_ab for a != b, default 1.
"""
import math
from typing import Optional

import numpy as np

from .eam import EAMPotential

KEYS = ("r_e", "f_e", "rho_e", "rho_s", "alpha", "beta", "A", "B", "kappa",
        "lambda", "Fn0", "Fn1", "Fn2", "Fn3", "F0", "F1", "F2", "F3", "eta",
        "F_e")
DERIVED = ("Fn0", "Fn1", "F0", "F_e")


def continuous(p: dict) -> dict:
    """Copy of an element's parameters with Fn0, Fn1, F0, F_e re-derived so
    that F and dF/drho are continuous at rho_n and rho_0 (F1 as given)."""
    q = {k: float(v) for k, v in p.items()}
    re, rs, eta = q["rho_e"], q["rho_s"], q["eta"]
    F1, F2, F3 = q["F1"], q["F2"], q["F3"]
    y, x = 0.15, -0.15                      # rho_0/rho_e - 1, rho_n/rho_e - 1
    t0 = 1.15 * re / rs
    slope0 = (F1 + 2 * F2 * y + 3 * F3 * y * y) / re
    dg = -eta * eta * math.log(t0) * t0 ** (eta - 1.0) / rs  # d/drho of the A8 shape
    if dg == 0.0:
        raise ValueError("Zhou EAM: rho_0 = rho_s (or eta = 0) leaves F_e "
                         "undetermined by the slope condition")
    q["F_e"] = slope0 / dg
    q["F0"] = q["F_e"] * (1 - eta * math.log(t0)) * t0 ** eta - (F1 * y + F2 * y * y + F3 * y ** 3)
    q["Fn0"] = q["F0"] + F1 * x + F2 * x * x + F3 * x ** 3
    q["Fn1"] = 0.85 * (F1 + 2 * F2 * x + 3 * F3 * x * x)
    return q


def _switch(r, r_c, width):
    s = np.clip((r - (r_c - width)) / width, 0.0, 1.0)
    S = 1.0 - s ** 3 * (10.0 - 15.0 * s + 6.0 * s * s)
    dS = -30.0 * s * s * (1.0 - s) ** 2 / width
    return S, dS


def _decay(r, p, pre, expo, shift):
    """pre exp[-expo (x - 1)] / (1 + (x - shift)^20) and its r-derivative."""
    x = r / p["r_e"]
    e = pre * np.exp(-expo * (x - 1.0))
    u = x - shift
    den = 1.0 + u ** 20
    val = e / den
    dval = (-expo * e / den - e * 20.0 * u ** 19 / den ** 2) / p["r_e"]
    return val, dval


def density(r, p):
    return _decay(r, p, p["f_e"], p["beta"], p["lambda"])


def pair_aa(r, p):
    a, da = _decay(r, p, p["A"], p["alpha"], p["kappa"])
    b, db = _decay(r, p, p["B"], p["beta"], p["lambda"])
    return a - b, da - db


def embedding(rho, p):
    """F(rho), dF/drho of eqs. (A6)-(A8) (vectorised)."""
    rho = np.asarray(rho, float)
    rn, re, rs, eta = 0.85 * p["rho_e"], p["rho_e"], p["rho_s"], p["eta"]
    F = np.empty_like(rho)
    dF = np.empty_like(rho)
    lo = rho < rn
    hi = rho >= 1.15 * re
    mid = ~(lo | hi)
    z = rho[lo] / rn - 1.0
    F[lo] = p["Fn0"] + p["Fn1"] * z + p["Fn2"] * z ** 2 + p["Fn3"] * z ** 3
    dF[lo] = (p["Fn1"] + 2 * p["Fn2"] * z + 3 * p["Fn3"] * z ** 2) / rn
    z = rho[mid] / re - 1.0
    F[mid] = p["F0"] + p["F1"] * z + p["F2"] * z ** 2 + p["F3"] * z ** 3
    dF[mid] = (p["F1"] + 2 * p["F2"] * z + 3 * p["F3"] * z ** 2) / re
    t = rho[hi] / rs
    lt = np.log(t)
    F[hi] = p["F_e"] * (1.0 - eta * lt) * t ** eta
    dF[hi] = -p["F_e"] * eta * eta * lt * t ** (eta - 1.0) / rs
    return F, dF


def zhou_cutoff(p: dict, tol: float = 1e-6) -> float:
    """Distance beyond which |phi_aa| < tol max(A, B) and f < tol f_e."""
    r = np.linspace(p["r_e"], 5.0 * p["r_e"], 40001)
    ok = ((np.abs(pair_aa(r, p)[0]) < tol * max(p["A"], p["B"]))
          & (density(r, p)[0] < tol * p["f_e"]))
    bad = np.nonzero(~ok)[0]
    if len(bad) == len(r):
        raise ValueError("Zhou EAM: functions do not decay within 5 r_e")
    return float(r[bad[-1] + 1]) if len(bad) else float(r[0])


def zhou_eam(elements: dict, species: list, r_cut: Optional[float] = None,
             chi: Optional[dict] = None, width: float = 0.5,
             tol: float = 1e-6, enforce_continuity: bool = True) -> EAMPotential:
    """
    EAMPotential with exact derivatives from Zhou (2004) element parameters.

    elements : {symbol: Table III parameter dict}.
    species  : symbols to include.
    r_cut    : cutoff [A]; default: max of zhou_cutoff over the species.
    chi      : optional {"a-b": factor} on the cross pair (default 1).
    enforce_continuity : re-derive Fn0, Fn1, F0, F_e (``continuous``).
    """
    species = list(species)
    P = {}
    for s in species:
        if s not in elements:
            raise ValueError(f"Zhou EAM: no parameters for '{s}'")
        missing = [k for k in KEYS if k not in elements[s]]
        if missing:
            raise ValueError(f"Zhou EAM: '{s}' lacks {missing}")
        P[s] = continuous(elements[s]) if enforce_continuity else {
            k: float(v) for k, v in elements[s].items()}
    if r_cut is None:
        r_cut = max(zhou_cutoff(P[s], tol) for s in species)
    chi = chi or {}

    def cfac(a, b):
        return float(chi.get(f"{a}-{b}", chi.get(f"{b}-{a}", 1.0))) if a != b else 1.0

    def switched(fn):
        def val(r):
            S, _ = _switch(np.asarray(r, float), r_cut, width)
            return fn(np.asarray(r, float))[0] * S

        def der(r):
            r = np.asarray(r, float)
            v, dv = fn(r)
            S, dS = _switch(r, r_cut, width)
            return dv * S + v * dS
        return val, der

    def cross(a, b):
        pa, pb, k = P[a], P[b], cfac(a, b)

        def fn(r):
            fa, dfa = density(r, pa)
            fb, dfb = density(r, pb)
            ua, dua = pair_aa(r, pa)
            ub, dub = pair_aa(r, pb)
            q = fb / fa
            dq = (dfb * fa - fb * dfa) / fa ** 2
            v = 0.5 * k * (q * ua + ub / q)
            dv = 0.5 * k * (dq * ua + q * dua + dub / q - ub * dq / q ** 2)
            return v, dv
        return fn

    F_tab = {s: (lambda rho, p=P[s]: embedding(rho, p)[0],
                 lambda rho, p=P[s]: embedding(rho, p)[1]) for s in species}
    rho_tab = {s: switched(lambda r, p=P[s]: density(r, p)) for s in species}
    phi_tab = {}
    for i, a in enumerate(species):
        for b in species[i:]:
            fn = (lambda r, p=P[a]: pair_aa(r, p)) if a == b else cross(a, b)
            phi_tab[frozenset((a, b))] = switched(fn)
    pot = EAMPotential(F_tab, rho_tab, phi_tab, r_cut=r_cut, species=species)
    pot.zhou_params = P
    pot.zhou_chi = {f"{a}-{b}": cfac(a, b) for i, a in enumerate(species)
                    for b in species[i + 1:]}
    return pot


# -- setfl (DYNAMO multi-element funcfl) files ------------------------------

# Atomic numbers of the 16 Table III metals, for the setfl element lines.
ATOMIC_NUMBER = {"Mg": 12, "Al": 13, "Ti": 22, "Fe": 26, "Co": 27, "Ni": 28,
                 "Cu": 29, "Zr": 40, "Mo": 42, "Pd": 46, "Ag": 47, "Ta": 73,
                 "W": 74, "Pt": 78, "Au": 79, "Pb": 82}

def write_setfl(path: str, pot: EAMPotential, species: list,
                n_rho: int = 5000, rho_max: Optional[float] = None,
                n_r: int = 5000, comments=("", "", ""),
                lattice: Optional[dict] = None) -> None:
    """
    Tabulate an EAMPotential in the setfl format (LAMMPS pair_style
    eam/alloy): F(rho) on [0, rho_max], rho(r) and r*phi(r) on [0, r_cut].

    rho_max defaults to twice the largest rho_e (Zhou) or 100.
    lattice : optional {symbol: (a0, "fcc")} for the element header lines.
    """
    from ..constants import ATOMIC_MASS
    r_cut = float(pot.r_cut)
    if rho_max is None:
        zp = getattr(pot, "zhou_params", None)
        rho_max = 2.0 * max(p["rho_e"] for p in zp.values()) if zp else 100.0
    drho = rho_max / (n_rho - 1)
    dr = r_cut / (n_r - 1)
    rho = np.arange(n_rho) * drho
    r = np.arange(n_r) * dr        # every Zhou function is finite at r = 0

    def block(vals):
        vals = list(vals)
        return "\n".join(" ".join(f"{v: .16e}" for v in vals[i:i + 5])
                         for i in range(0, len(vals), 5)) + "\n"

    with open(path, "w") as fh:
        for c in (list(comments) + ["", "", ""])[:3]:
            fh.write(c.replace("\n", " ") + "\n")
        fh.write(f"{len(species)} " + " ".join(species) + "\n")
        fh.write(f"{n_rho} {drho:.16e} {n_r} {dr:.16e} {r_cut:.16e}\n")
        for s in species:
            a0, kind = (lattice or {}).get(s, (0.0, "fcc"))
            fh.write(f"{ATOMIC_NUMBER.get(s, 0)} {ATOMIC_MASS[s]:.6f} {a0:.6f} {kind}\n")
            fh.write(block(pot._F[s][0](rho)))
            fh.write(block(pot._rho[s][0](r)))
        for i, a in enumerate(species):
            for b in species[:i + 1]:
                fh.write(block(r * pot._phi[frozenset((a, b))][0](r)))


def read_setfl(path: str, species: Optional[list] = None) -> EAMPotential:
    """
    Read a setfl file into an EAMPotential.  Each table is interpolated by
    a cubic spline (r*phi as stored, phi = (r phi)/r), with the spline's
    exact derivative, as LAMMPS interpolates these files.
    """
    from scipy.interpolate import CubicSpline
    with open(path) as fh:
        lines = fh.read().splitlines()
    head = lines[3].split()
    n_el = int(head[0])
    names = head[1:1 + n_el]
    n_rho, drho, n_r, dr, r_cut = lines[4].split()[:5]
    n_rho, n_r = int(n_rho), int(n_r)
    drho, dr, r_cut = float(drho), float(dr), float(r_cut)
    pos = 5
    tables = {}

    def take(n):
        nonlocal pos
        out = []
        while len(out) < n:
            out.extend(float(v) for v in lines[pos].split())
            pos += 1
        return np.array(out[:n])

    for s in names:
        pos += 1                                   # element header line
        tables[("F", s)] = take(n_rho)
        tables[("rho", s)] = take(n_r)
    for i, a in enumerate(names):
        for b in names[:i + 1]:
            tables[("rphi", a, b)] = take(n_r)
    rho_g = np.arange(n_rho) * drho
    r_g = np.arange(n_r) * dr

    def spline(x, y):
        cs = CubicSpline(x, y)
        d = cs.derivative()
        return (lambda z: cs(z)), (lambda z: d(z))

    def over_r(x, y):
        cs = CubicSpline(x, y)
        d = cs.derivative()
        return ((lambda z: cs(z) / z),
                (lambda z: (d(z) * z - cs(z)) / (z * z)))

    F_tab = {s: spline(rho_g, tables[("F", s)]) for s in names}
    rho_tab = {s: spline(r_g, tables[("rho", s)]) for s in names}
    phi_tab = {}
    for i, a in enumerate(names):
        for b in names[:i + 1]:
            phi_tab[frozenset((a, b))] = over_r(r_g, tables[("rphi", a, b)])
    return EAMPotential(F_tab, rho_tab, phi_tab, r_cut=r_cut,
                        species=list(species) if species else names)
