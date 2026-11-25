"""Charges, Born effective charges, eps_inf, masses and temperatures (nozuchi ADR-0002).

Ported from nozuchi 0.2.0a2 (ADR-0050, ADR-0051).  The OUTCAR Born / eps_inf
readers and the site mapping are akitsushima.io.outcar_dielectric (ported
in ADR-0049), re-exported here.

Convention: Z[k, a, b] = dP_a / du_kb, the order printed in a VASP OUTCAR
("BORN EFFECTIVE CHARGES" block: row a = polarisation / field direction,
column b = displacement direction), so that M_a = sum_k sum_b Z[k, a, b] r_kb.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np

from ..io.outcar_dielectric import (  # one implementation (ADR-0050 decision 7)
    map_to_unit_cell,
    read_outcar_born,
    read_outcar_eps_inf,
    rigid_shift,
)
from .constants import ATOMIC_MASS


# --------------------------------------------------------------------------
# VASP OUTCAR / POSCAR and akitsushima thermo.log readers
# --------------------------------------------------------------------------





def _outcar_header_matches(path: str | Path, pattern: str) -> list[str]:
    """First group of every match of `pattern` in the OUTCAR header.

    Streams line by line and stops at the first electronic iteration, so
    multi-GB MD OUTCARs are not read into memory.
    """
    rx, out = re.compile(pattern), []
    with open(path, errors="replace") as fh:
        for ln in fh:
            if "Iteration" in ln:
                break
            m = rx.search(ln)
            if m:
                out.append(m.group(1))
    return out


def read_outcar_pomass(path: str | Path) -> list[float]:
    """POMASS of each POTCAR, in POTCAR (= POSCAR species) order."""
    vals = _outcar_header_matches(path, r"POMASS\s*=\s*([\d.]+)\s*;")
    if not vals:
        raise ValueError(f"{path}: no POMASS entries")
    return [float(v) for v in vals]


def read_outcar_temperatures(path: str | Path) -> np.ndarray:
    """Instantaneous temperature of every MD step (from '(temperature ... K)'), streamed."""
    rx, temps = re.compile(r"\(temperature\s+([-\d.]+)\s+K\)"), []
    with open(path, errors="replace") as fh:
        for ln in fh:
            if "(temperature" in ln:
                m = rx.search(ln)
                if m:
                    temps.append(float(m.group(1)))
    return np.array(temps)


def is_thermo_log(path: str | Path) -> bool:
    """True for an akitsushima thermo.log (first line holds 'Step' and 'T_inst...' columns).

    Decided by content, not by the file name (ADR-0011 A2); a VASP OUTCAR or
    OSZICAR never starts with such a header.
    """
    try:
        with open(path, errors="replace") as fh:
            head = fh.readline().split()
    except OSError:
        return False
    return "Step" in head and any(h.startswith("T_inst") for h in head)


def read_thermo_log(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """akitsushima thermo.log: (step, T_inst)."""
    with open(path) as fh:
        head = fh.readline().split()
    col = next((i for i, h in enumerate(head) if h.startswith("T_inst")), None)
    if col is None or "Step" not in head:
        raise ValueError(f"{path}: not an akitsushima thermo.log (no Step / T_inst columns)")
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    return data[:, head.index("Step")].astype(int), data[:, col]


def read_poscar(path: str | Path):
    """Minimal POSCAR reader: (lattice (3,3), symbols, fractional (n,3))."""
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1].split()[0])
    raw = np.array([[float(x) for x in ln.split()[:3]] for ln in lines[2:5]])
    # VASP: a negative scale is the target volume; the same factor scales Cartesian positions
    factor = (abs(scale) / abs(np.linalg.det(raw))) ** (1 / 3) if scale < 0 else scale
    lat = raw * factor
    species = lines[5].split()
    if all(s.isdigit() for s in species):
        raise ValueError(f"{path}: POSCAR without element names (VASP 4 format) is not supported")
    counts = [int(x) for x in lines[6].split()]
    k = 7
    if lines[k].strip()[0] in "sS":
        k += 1
    cart = lines[k].strip()[0] in "cCkK"
    n = sum(counts)
    pos = np.array([[float(x) for x in ln.split()[:3]] for ln in lines[k + 1 : k + 1 + n]])
    if cart:
        pos = (pos * factor) @ np.linalg.inv(lat)
    symbols = [s for s, c in zip(species, counts) for _ in range(c)]
    return lat, symbols, pos


# --------------------------------------------------------------------------
# Charge models
# --------------------------------------------------------------------------


def _tensor(value) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        return float(a) * np.eye(3)
    if a.shape == (3,):
        return np.diag(a)
    if a.shape == (3, 3):
        return a
    raise ValueError(f"charge / tensor must be scalar, 3 values or 3x3; got shape {a.shape}")


def point_charges(symbols: list[str], charges: dict, tol: float = 1e-6) -> np.ndarray:
    """Per-atom Z tensors (n, 3, 3) for scalar point charges; neutrality enforced."""
    missing = sorted(set(symbols) - set(charges))
    if missing:
        raise ValueError(f"dipole.charges has no value for {missing}")
    q = np.array([float(charges[s]) for s in symbols])
    if abs(q.sum()) > tol:
        raise ValueError(f"point charges are not neutral: sum = {q.sum():.3e} e (tolerance {tol})")
    return q[:, None, None] * np.eye(3)[None]





def apply_sum_rule(z: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove the mean so that sum_k Z_k = 0; return (Z, max |correction|)."""
    corr = z.mean(axis=0)
    return z - corr[None], float(np.abs(corr).max())


def born_charges(cfg_born: dict, symbols_sc, frac_mean, lattice_sc) -> np.ndarray:
    """Per-atom Born tensors (n, 3, 3) for the MD cell from the YAML 'dipole.born' block."""
    n = len(symbols_sc)
    z = np.full((n, 3, 3), np.nan)
    if cfg_born["outcar"]:
        zu = read_outcar_born(cfg_born["outcar"])
        lat_u, sym_u, frac_u = read_poscar(cfg_born["unit_cell"])
        if len(zu) != len(sym_u):
            raise ValueError(f"OUTCAR has {len(zu)} Born tensors, unit cell has {len(sym_u)} atoms")
        site = map_to_unit_cell(
            frac_mean,
            lattice_sc,
            symbols_sc,
            lat_u,
            sym_u,
            frac_u,
            cfg_born["supercell"],
            cfg_born["map_tolerance"],
        )
        z = zu[site].copy()
    for sp, val in (cfg_born["per_species"] or {}).items():
        mask = np.array(symbols_sc) == sp
        if not mask.any():
            raise ValueError(f"dipole.born.per_species: no atoms of {sp}")
        z[mask] = _tensor(val)
    for idx, val in (cfg_born["per_atom"] or {}).items():
        z[int(idx) - 1] = _tensor(val)
    if np.isnan(z).any():
        bad = sorted({symbols_sc[i] for i in np.flatnonzero(np.isnan(z).any(axis=(1, 2)))})
        raise ValueError(f"no Born charge for species {bad}")
    z, corr = apply_sum_rule(z)
    if corr > 0.1:
        warnings.warn(f"Born charge sum rule correction {corr:.3f} e is large", stacklevel=2)
    return z


def run_masses(cfg: dict, header):
    """Per-atom masses for a validated config: YAML > POMASS of the temperature_source OUTCARs > standard."""
    outcars = [s for s in (cfg["temperature_source"] or []) if not is_thermo_log(s)]
    masses, source = atom_masses(header.symbols, header.species, cfg["masses"], outcars)
    return masses, source, outcars


def atom_masses(symbols: list[str], species_order: list[str], override: dict, outcars=None):
    """Per-atom masses [amu] and their source: YAML > OUTCAR POMASS > standard weights."""
    table, source = {}, {}
    if outcars:
        for oc in outcars:
            try:
                pm = read_outcar_pomass(oc)
            except (ValueError, OSError):
                continue
            if len(pm) >= len(species_order):
                for s, m in zip(species_order, pm):
                    table[s], source[s] = m, "OUTCAR POMASS"
                break
    for s in species_order:
        if s in override:
            table[s], source[s] = override[s], "YAML"
        elif s not in table:
            if s not in ATOMIC_MASS:
                raise ValueError(f"no mass for {s}; give masses: {{{s}: ...}}")
            table[s], source[s] = ATOMIC_MASS[s], "standard atomic weight"
    return np.array([table[s] for s in symbols]), source
