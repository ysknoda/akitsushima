"""akitsushima YAML for ``mdspec`` and its translation to the internal
(nozuchi-format) configuration (ADR-0050 decision 8, ADR-0051).

akitsushima form::

    born:                         # shared with phonon / dielectric / raman
      epsilon_inf: 1.0 | [..] | 3x3 | {outcar: OUTCAR}
      born_charges:               # used with mdspec.dipole.model born_tensor
        outcar: dfpt/OUTCAR
        unit_cell: dfpt/POSCAR
        supercell: [3, 3, 2]
        map_tolerance: 0.6
        per_species: {...}
        per_atom: {...}
    mdspec:
      trajectory: {files, mode, timestep_fs, frame_every, skip_frames, stride, species}
      temperature, temperature_source, masses
      dipole: {model, charges, external_file, external_unit, neutrality_tol, map_frames}
      spectrum, modes, forces, fit, output, performance

Internal form: nozuchi 0.2.0a2's schema, where eps_inf is the top-level
``epsilon_inf`` and the Born charges are ``dipole.born`` (with
``map_frames``).  The mapping is one-to-one, so ``from_nozuchi`` (the
``--convert-nozuchi`` converter) and ``to_internal`` are inverses.
Relative paths are resolved against the directory of the YAML file, as
in nozuchi.
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import yaml

from .config import ConfigError, validate

__all__ = ["from_nozuchi", "load", "to_internal"]

_BORN_KEYS = {"epsilon_inf", "born_charges", "clamped_ion_piezo"}


def to_internal(cfg: dict) -> dict:
    """The nozuchi-format raw configuration of an akitsushima YAML dict."""
    if not isinstance(cfg, dict) or not isinstance(cfg.get("mdspec"), dict):
        raise ConfigError("mdspec: the input needs an 'mdspec:' section")
    raw = copy.deepcopy(cfg["mdspec"])
    if "epsilon_inf" in raw:
        raise ConfigError("mdspec.epsilon_inf: give eps_inf as born.epsilon_inf")
    dp = raw.setdefault("dipole", {})
    if not isinstance(dp, dict):
        raise ConfigError("mdspec.dipole must be a mapping")
    if "born" in dp:
        raise ConfigError("mdspec.dipole.born: give the Born charges as born.born_charges")
    if raw.get("forces") == "trajectory":  # forces of an extxyz / LAMMPS dump (ADR-0052)
        files = (raw.get("trajectory") or {}).get("files")
        raw["forces"] = {"files": [files] if isinstance(files, str) else list(files or [])}
    born = cfg.get("born") or {}
    if not isinstance(born, dict):
        raise ConfigError("born must be a mapping")
    unknown = set(born) - _BORN_KEYS
    if unknown:
        raise ConfigError(f"unknown key(s) in 'born': {sorted(unknown)}")
    if "epsilon_inf" in born:
        raw["epsilon_inf"] = copy.deepcopy(born["epsilon_inf"])
    map_frames = dp.pop("map_frames", None)
    bc = born.get("born_charges")
    if bc is not None:
        if not isinstance(bc, dict):
            raise ConfigError("born.born_charges must be a mapping")
        b = copy.deepcopy(bc)
        if map_frames is not None:
            b["map_frames"] = map_frames
        dp["born"] = b
        if dp.get("model") == "point_charge":
            warnings.warn(
                "born.born_charges is not used with mdspec.dipole.model "
                "point_charge (the charges are dipole.charges)",
                stacklevel=2,
            )
    elif map_frames is not None:
        raise ConfigError("mdspec.dipole.map_frames needs born.born_charges")
    return raw


def from_nozuchi(raw: dict) -> dict:
    """The akitsushima YAML dict of a nozuchi input (the converter)."""
    if not isinstance(raw, dict):
        raise ConfigError("the nozuchi input must be a mapping")
    md = copy.deepcopy(raw)
    born = {}
    if "epsilon_inf" in md:
        ei = md.pop("epsilon_inf")
        if ei is not None:
            born["epsilon_inf"] = ei
    dp = md.get("dipole")
    if isinstance(dp, dict) and dp.get("born") is not None:
        b = dp.pop("born")
        if "map_frames" in b:
            dp["map_frames"] = b.pop("map_frames")
        born["born_charges"] = b
    elif isinstance(dp, dict):
        dp.pop("born", None)
    out = {}
    if born:
        out["born"] = born
    out["mdspec"] = md
    return out


def load(path) -> dict:
    """Read an akitsushima YAML file and return the validated internal
    configuration (paths resolved against the file's directory)."""
    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return validate(to_internal(cfg), path.parent)


def convert_file(path) -> str:
    """The akitsushima YAML text of a nozuchi input file."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    text = yaml.safe_dump(from_nozuchi(raw), sort_keys=False, default_flow_style=None)
    return (
        "# Converted from the nozuchi input "
        + str(Path(path).name)
        + " by akitsushima mdspec --convert-nozuchi.\n"
        "# Relative paths are read against this file's directory: save it\n"
        "# next to the original.\n" + text
    )
