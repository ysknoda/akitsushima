"""Shared helpers for the CLI sub-command handlers: config/POSCAR loading,
mass resolution, and console output formatting.
"""
import hashlib
import json
import os
import warnings

import numpy as np

from ..constants import ATOMIC_MASS
from ..io.poscar import parse_poscar
from ..solvers.config_bridge import _load_yaml
from ..utils.mpi import _mpi_rank

__all__ = [
    "_check_yaml",
    "_check_poscar",
    "_resolve_masses",
    "_fmt_time",
    "_print_separator",
    "_print_header",
    "_print_results_header",
    "_print_timing_summary",
    "_file_sizes",
    "_print_output_files",
    "_resolve_mu_from_cfg",
    "_input_hash",
]


def _check_yaml(path: str):
    """Load YAML file; return None and print error if file not found."""
    if not os.path.isfile(path):
        print(f"ERROR: YAML file not found: '{path}'")
        return None
    return _load_yaml(path)


def _check_poscar(path: str):
    """Parse POSCAR file; return None and print error if file not found."""
    if not os.path.isfile(path):
        if _mpi_rank == 0:
            print(f"ERROR: POSCAR not found: '{path}'")
        return None
    data = parse_poscar(path)
    if _mpi_rank == 0:
        print(f"Loaded structure : {path}  ({data['natoms']} atoms)")
    return data


def _resolve_masses(cfg: dict, atom_types: list) -> np.ndarray:
    """Return per-atom mass array from a YAML section (masses sub-dict)."""
    mass_map = cfg.get("masses", {})
    return np.array([mass_map.get(t, ATOMIC_MASS.get(t, 1.0))
                     for t in atom_types])


def _resolve_mu_from_cfg(section_cfg: dict, default: float = -0.1,
                        section_name: str = "gcmc") -> float:
    """
    Read the ``gcmc``/``osmc`` YAML section's chemical-potential key,
    preferring the current ``mu`` and falling back to the deprecated
    ``mu_ex`` key for config files that predate the rename (see
    GCMCMDSimulation's ``mu`` parameter note: it was always the full
    chemical potential, despite the old key's name).  Warns, like the
    equivalent Python-level ``mu_ex=`` constructor kwarg, when only the
    deprecated key is present, so a user updating their YAML sees the
    same guidance either way they hit the deprecated path.

    section_name : the YAML top-level section this key lives under
                  (``"gcmc"`` or ``"osmc"``), used only for the warning
                  message.
    """
    if "mu" in section_cfg:
        return float(section_cfg["mu"])
    if "mu_ex" in section_cfg:
        warnings.warn(
            f"'{section_name}.mu_ex' in the YAML config is deprecated; "
            f"use '{section_name}.mu' instead (it was always the full "
            f"chemical potential, not the excess).",
            UserWarning, stacklevel=2)
        return float(section_cfg["mu_ex"])
    return float(default)


def _input_hash(lattice, positions, atom_types, pot_cfg: dict) -> str:
    """SHA-256 of a structure (lattice, element order, fractional
    coordinates rounded to 1e-6) and its potential section.  A command
    that reuses another command's results (an IFC file, an
    elastic_results.json) refuses them when this does not match
    (ADR-0017, Q30)."""
    lattice = np.asarray(lattice, dtype=float)
    frac = np.asarray(positions, dtype=float) @ np.linalg.inv(lattice)
    frac -= np.floor(frac + 1e-9)
    key = {"lattice": np.round(lattice, 6).tolist(),
           "types": list(atom_types),
           "frac": np.round(frac, 6).tolist(),
           "potential": pot_cfg}
    blob = json.dumps(key, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _fmt_time(s: float) -> str:
    """Human-readable elapsed time."""
    if s < 1.0:    return f"{s*1e3:.1f} ms"
    if s < 60.0:   return f"{s:.3f} s"
    if s < 3600.0: return f"{s/60:.2f} min ({s:.1f} s)"
    return f"{s/3600:.2f} h ({s:.1f} s)"


def _print_separator(width: int = 64, char: str = "=") -> None:
    print(char * width)


def _print_header(command: str, cfg: dict, poscar: str = None,
                  pot_cfg: dict = None) -> None:
    """Print the input-settings header common to all sub-commands."""
    _print_separator()
    print(f"  akitsushima  --  {command}")
    _print_separator()
    if poscar:
        print(f"\n[Input structure]")
        print(f"  POSCAR : {poscar}")
    if pot_cfg and pot_cfg.get("type"):
        print(f"\n[Potential]")
        print(f"  Type   : {pot_cfg.get('type','?')}")
        sp = pot_cfg.get("species")
        if sp:
            print(f"  Species: {sp}")


def _print_results_header(command: str) -> None:
    print(f"\n{'='*64}")
    print(f"  Results summary  --  {command}")
    print(f"{'='*64}")


def _print_timing_summary(sections: list, t_total: float) -> None:
    """
    Print a timing table.

    sections : list of (label, wall_seconds) tuples
    t_total  : total wall-clock time in seconds
    """
    print(f"\n{'='*64}")
    print(f"  Timing summary")
    print(f"{'='*64}")
    print(f"  {'Stage':<40}  {'Wall time':>12}")
    print(f"  {'-'*54}")
    for label, t in sections:
        print(f"  {label:<40}  {_fmt_time(t):>12}")
    print(f"  {'-'*54}")
    print(f"  {'TOTAL':<40}  {_fmt_time(t_total):>12}")


def _file_sizes(output_dir: str, patterns=None) -> list:
    """Return list of (filename, kB) for files in output_dir."""
    results = []
    if not os.path.isdir(output_dir):
        return results
    for fp in sorted(os.listdir(output_dir)):
        full = os.path.join(output_dir, fp)
        if os.path.isfile(full):
            results.append((fp, os.path.getsize(full) / 1024))
    return results


def _print_output_files(output_dir: str) -> None:
    files = _file_sizes(output_dir)
    if not files:
        return
    print(f"\n[Output files]  ({output_dir})")
    for fname, kb in files:
        print(f"  {fname:<36}  {kb:8.1f} kB")
