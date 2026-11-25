"""Internal (nozuchi-format) configuration: schema, defaults and validation
(nozuchi ADR-0007).  The user-facing akitsushima YAML is translated into
this form by yaml_io (ADR-0051).

Unknown keys are refused; required keys have no default; relative paths are
resolved against the directory of the YAML file.
"""

from __future__ import annotations

import copy
import math
import warnings
from pathlib import Path

import numpy as np
import yaml

REQUIRED = object()

SCHEMA = {
    "trajectory": {
        "files": REQUIRED,
        "mode": "concatenate",
        "timestep_fs": REQUIRED,
        "frame_every": REQUIRED,
        "skip_frames": 0,
        "stride": 1,
        "species": None,
    },
    "temperature": REQUIRED,
    "temperature_source": None,
    "masses": {},
    "dipole": {
        "model": REQUIRED,
        "charges": None,
        "born": None,
        "external_file": None,
        "external_unit": "e*Angstrom",
        "neutrality_tol": 1.0e-6,
    },
    "epsilon_inf": None,
    "spectrum": {
        "max_lag_ps": "auto",
        "window": "hann",
        "zero_pad_factor": 4,
        "freq_unit": "THz",
        "freq_max": None,
        "kk_check": True,
        "n_blocks": 5,
        "estimator": "auto",
    },
    "modes": None,
    "forces": None,
    "fit": None,
    "output": {
        "dir": "mdspec_output",
        "prefix": "mdspec",
        "plot": True,
        "dipole_cache": True,
    },
    "performance": {"chunk_mb": 64.0},
}

BORN_SCHEMA = {
    "outcar": None,
    "unit_cell": None,
    "supercell": None,
    "per_species": None,
    "per_atom": None,
    "map_tolerance": 0.6,
    "map_frames": 200,
}
MODES_SCHEMA = {
    "outcar": None,
    "file": None,
    "phonopy": None,
    "unit_cell": None,
    "supercell": None,
    "map_tolerance": 0.6,
    "ir_share": 1.0e-4,
}
FORCES_SCHEMA = {
    "files": REQUIRED,
    "stride_frames": 1,
}
FIT_SCHEMA = {
    "model": "additive",
    "terms": REQUIRED,
    "freq_range": None,
    "candidates": None,
    "use_component": "iso",
}
CHOICES = {
    ("trajectory", "mode"): {"concatenate", "independent"},
    ("dipole", "model"): {"point_charge", "born_tensor", "external"},
    ("dipole", "external_unit"): {"e*Angstrom", "Debye"},
    ("spectrum", "window"): {"hann", "boxcar", "none"},
    ("spectrum", "freq_unit"): {"THz", "cm-1", "meV"},
    ("spectrum", "estimator"): {"auto", "mode", "fsum", "raw"},
    ("output", "dipole_cache"): {True, False},
}
FIT_TERMS = {"debye", "cole_cole", "cole_davidson", "havriliak_negami", "lorentz", "drude"}


class ConfigError(ValueError):
    """Invalid mdspec input."""


def _merge(schema: dict, user: dict, where: str) -> dict:
    if not isinstance(user, dict):
        raise ConfigError(f"'{where}' must be a mapping")
    unknown = set(user) - set(schema)
    if unknown:
        raise ConfigError(f"unknown key(s) in '{where}': {sorted(unknown)}")
    out = {}
    for key, default in schema.items():
        name = f"{where}.{key}" if where else key
        if key in user:
            value = user[key]
            if isinstance(default, dict) and default and isinstance(value, dict):
                value = _merge(default, value, name)
            out[key] = value
        elif default is REQUIRED:
            raise ConfigError(f"required key '{name}' is missing")
        elif isinstance(default, dict) and default:
            out[key] = _merge(default, {}, name)
        else:
            out[key] = copy.deepcopy(default)
    return out


def _number(value, name: str, *, integer=False, minimum=None, strict=False, maximum=None):
    """A YAML number as float (or int), refusing booleans and non-numbers (ADR-0011 A6).

    Strings are converted because PyYAML reads '1e-6' (no dot) as a string.
    """
    kind = "an integer" if integer else "a number"
    if isinstance(value, bool) or value is None or isinstance(value, (list, dict)):
        raise ConfigError(f"{name} must be {kind}, got {value!r}")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be {kind}, got {value!r}") from None
    if not math.isfinite(x):
        raise ConfigError(f"{name} must be finite, got {value!r}")
    if integer:
        if x != int(x):
            raise ConfigError(f"{name} must be an integer, got {value!r}")
        x = int(x)
    if minimum is not None and (x <= minimum if strict else x < minimum):
        raise ConfigError(f"{name} must be {'>' if strict else '>='} {minimum}, got {value!r}")
    if maximum is not None and x > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value!r}")
    return x


def _strings(value, name: str) -> list[str]:
    """A string or a non-empty list of strings."""
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not items or not all(isinstance(x, str) for x in items):
        raise ConfigError(f"{name} must be a path or a non-empty list of paths, got {value!r}")
    return items


def _tensor_value(value, name: str):
    """Scalar, three diagonal values or a 3 x 3 matrix of numbers."""
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a number, 3 numbers or a 3x3 matrix") from None
    if a.shape not in ((), (3,), (3, 3)) or not np.all(np.isfinite(a)):
        raise ConfigError(f"{name} must be a number, 3 numbers or a 3x3 matrix, got {value!r}")
    return value


def _supercell(value, name: str):
    """Three positive integers (diagonal) or a 3 x 3 integer matrix with a non-zero determinant."""
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be 3 integers or a 3x3 integer matrix") from None
    if a.shape not in ((3,), (3, 3)) or not np.all(a == np.round(a)):
        raise ConfigError(f"{name} must be 3 integers or a 3x3 integer matrix, got {value!r}")
    if (a.ndim == 1 and np.any(a < 1)) or (a.ndim == 2 and round(np.linalg.det(a)) == 0):
        raise ConfigError(f"{name} must describe a supercell of positive volume, got {value!r}")
    return a.astype(int).tolist()


def _path(base: Path, p):
    return None if p is None else str((base / p).resolve()) if not Path(p).is_absolute() else str(p)


def validate(raw: dict, base_dir: str | Path = ".") -> dict:
    """Return the resolved configuration (defaults filled, paths absolute)."""
    base = Path(base_dir)
    cfg = _merge(SCHEMA, raw or {}, "")
    tr = cfg["trajectory"]
    if isinstance(tr["files"], list) and not tr["files"]:
        raise ConfigError("trajectory.files is empty")
    tr["files"] = [_path(base, f) for f in _strings(tr["files"], "trajectory.files")]
    tr["timestep_fs"] = _number(tr["timestep_fs"], "trajectory.timestep_fs", minimum=0, strict=True)
    for key in ("frame_every", "stride"):
        tr[key] = _number(tr[key], f"trajectory.{key}", integer=True, minimum=1)
    tr["skip_frames"] = _number(
        tr["skip_frames"], "trajectory.skip_frames", integer=True, minimum=0
    )
    if tr["species"] is not None:
        tr["species"] = _strings(tr["species"], "trajectory.species")
    perf = cfg["performance"]
    perf["chunk_mb"] = _number(perf["chunk_mb"], "performance.chunk_mb", minimum=0, strict=True)
    for key in ("plot", "dipole_cache"):
        if not isinstance(cfg["output"][key], bool):
            raise ConfigError(f"output.{key} must be true or false")
    for key in ("dir", "prefix"):
        if not isinstance(cfg["output"][key], str) or not cfg["output"][key]:
            raise ConfigError(f"output.{key} must be a non-empty string")
    if not isinstance(cfg["spectrum"]["kk_check"], bool):
        raise ConfigError("spectrum.kk_check must be true or false")
    for (sec, key), allowed in CHOICES.items():
        if cfg[sec][key] not in allowed:
            raise ConfigError(
                f"{sec}.{key} = {cfg[sec][key]!r}; allowed: {sorted(map(str, allowed))}"
            )

    t = cfg["temperature"]
    if t != "auto":
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not t > 0:
            raise ConfigError("temperature must be a positive number [K] or 'auto'")
        cfg["temperature"] = float(t)
    src = cfg["temperature_source"]
    if src is not None:
        cfg["temperature_source"] = [_path(base, s) for s in _strings(src, "temperature_source")]
    if t == "auto" and not cfg["temperature_source"]:
        raise ConfigError("temperature: auto needs temperature_source (OUTCAR files or thermo.log)")

    dp = cfg["dipole"]
    model = dp["model"]
    dp["neutrality_tol"] = _number(dp["neutrality_tol"], "dipole.neutrality_tol", minimum=0)
    if model == "point_charge":
        if not isinstance(dp["charges"], dict) or not dp["charges"]:
            raise ConfigError("dipole.model = point_charge needs dipole.charges {species: charge}")
        dp["charges"] = {
            str(k): _number(v, f"dipole.charges.{k}") for k, v in dp["charges"].items()
        }
    elif model == "born_tensor":
        if not isinstance(dp["born"], dict):
            raise ConfigError("dipole.model = born_tensor needs a dipole.born mapping")
        dp["born"] = _merge(BORN_SCHEMA, dp["born"], "dipole.born")
        b = dp["born"]
        if b["outcar"] is None and b["per_species"] is None and b["per_atom"] is None:
            raise ConfigError(
                "dipole.born needs outcar (with unit_cell, supercell) or per_species / per_atom"
            )
        if b["outcar"] is not None and (b["unit_cell"] is None or b["supercell"] is None):
            raise ConfigError(
                "dipole.born.outcar needs dipole.born.unit_cell and dipole.born.supercell"
            )
        b["outcar"], b["unit_cell"] = _path(base, b["outcar"]), _path(base, b["unit_cell"])
        if b["supercell"] is not None:
            b["supercell"] = _supercell(b["supercell"], "dipole.born.supercell")
        b["map_tolerance"] = _number(
            b["map_tolerance"], "dipole.born.map_tolerance", minimum=0, strict=True
        )
        b["map_frames"] = _number(
            b["map_frames"], "dipole.born.map_frames", integer=True, minimum=1
        )
        for key in ("per_species", "per_atom"):
            if b[key] is not None and not isinstance(b[key], dict):
                raise ConfigError(f"dipole.born.{key} must be a mapping")
        for sp, val in (b["per_species"] or {}).items():
            _tensor_value(val, f"dipole.born.per_species.{sp}")
        for idx, val in (b["per_atom"] or {}).items():
            _number(idx, "dipole.born.per_atom index", integer=True, minimum=1)
            _tensor_value(val, f"dipole.born.per_atom.{idx}")
    else:
        if dp["external_file"] is None:
            raise ConfigError("dipole.model = external needs dipole.external_file")
        dp["external_file"] = _path(base, dp["external_file"])
        if cfg["spectrum"]["estimator"] in ("fsum", "mode"):
            warnings.warn(
                "external dipole: no exact <Mdot^2> and no positions per mode; estimator set "
                "to 'raw' (ADR-0002)",
                stacklevel=2,
            )
        cfg["spectrum"]["estimator"] = "raw"
        if cfg["modes"] is not None:
            raise ConfigError("modes cannot be used with dipole.model = external")

    _modes(cfg, base)
    _forces(cfg, base)

    ei = cfg["epsilon_inf"]
    if ei is None:
        warnings.warn(
            "epsilon_inf not given: using 1 (eps' lacks the electronic part, ADR-0001)",
            stacklevel=2,
        )
        cfg["epsilon_inf"] = 1.0
    elif isinstance(ei, dict):
        if set(ei) != {"outcar"} or not isinstance(ei["outcar"], str):
            raise ConfigError("epsilon_inf as a mapping must be {outcar: path}")
        cfg["epsilon_inf"] = {"outcar": _path(base, ei["outcar"])}
    else:
        _tensor_value(ei, "epsilon_inf")

    sp = cfg["spectrum"]
    if sp["max_lag_ps"] != "auto":
        sp["max_lag_ps"] = _number(sp["max_lag_ps"], "spectrum.max_lag_ps", minimum=0, strict=True)
    sp["n_blocks"] = _number(sp["n_blocks"], "spectrum.n_blocks", integer=True, minimum=2)
    sp["zero_pad_factor"] = _number(
        sp["zero_pad_factor"], "spectrum.zero_pad_factor", integer=True, minimum=1
    )
    if sp["freq_max"] is not None:
        sp["freq_max"] = _number(sp["freq_max"], "spectrum.freq_max", minimum=0, strict=True)

    if cfg["fit"] is not None:
        fit = _merge(FIT_SCHEMA, cfg["fit"], "fit")
        if fit["model"] not in ("additive", "gervais_piriou"):
            raise ConfigError("fit.model must be 'additive' or 'gervais_piriou'")
        lists = [fit["terms"]] + list(fit["candidates"] or [])
        for terms in lists:
            if not isinstance(terms, list) or not terms:
                raise ConfigError("fit.terms (and each candidate) must be a non-empty list")
            for term in terms:
                if fit["model"] == "additive" and term.get("type") not in FIT_TERMS:
                    raise ConfigError(
                        f"unknown fit term type {term.get('type')!r}; allowed {sorted(FIT_TERMS)}"
                    )
                if fit["model"] == "gervais_piriou" and not {"f_to", "f_lo"} <= set(term):
                    raise ConfigError("gervais_piriou terms need f_to and f_lo initial values")
        if fit["use_component"] not in ("iso", "xx", "yy", "zz"):
            raise ConfigError("fit.use_component must be iso, xx, yy or zz")
        if fit["freq_range"] is not None:
            fr = fit["freq_range"]
            if not isinstance(fr, list) or len(fr) != 2:
                raise ConfigError(f"fit.freq_range must be [low, high], got {fr!r}")
            lo = _number(fr[0], "fit.freq_range low", minimum=0)
            hi = _number(fr[1], "fit.freq_range high", minimum=0)
            if not hi > lo:
                raise ConfigError(f"fit.freq_range must have low < high, got {fr!r}")
            fit["freq_range"] = [lo, hi]
        cfg["fit"] = fit
    cfg["output"]["dir"] = _path(base, cfg["output"]["dir"])
    if not isinstance(cfg["masses"] or {}, dict):
        raise ConfigError("masses must be a mapping {species: mass in amu}")
    cfg["masses"] = {
        str(k): _number(v, f"masses.{k}", minimum=0, strict=True)
        for k, v in (cfg["masses"] or {}).items()
    }
    return cfg


def _modes(cfg: dict, base: Path) -> None:
    """Resolve the 'modes' block (ADR-0007 amendment); None when no Gamma modes are given."""
    if cfg["dipole"]["model"] == "external":
        return
    m = cfg["modes"]
    born = cfg["dipole"]["born"] if cfg["dipole"]["model"] == "born_tensor" else None
    if m is None:
        cfg["modes"] = None
        if born and born["outcar"]:
            # filled from dipole.born at run time if that OUTCAR holds eigenvectors
            cfg["modes"] = {
                **MODES_SCHEMA,
                "outcar": born["outcar"],
                "unit_cell": born["unit_cell"],
                "supercell": born["supercell"],
                "map_tolerance": born["map_tolerance"],
                "implicit": True,
            }
        elif cfg["spectrum"]["estimator"] == "mode":
            raise ConfigError(
                "spectrum.estimator = mode needs a 'modes' block (or dipole.born.outcar)"
            )
        return
    m = _merge(MODES_SCHEMA, m, "modes")
    if sum(m[k] is not None for k in ("outcar", "file", "phonopy")) != 1:
        raise ConfigError("modes needs exactly one of outcar, file or phonopy")
    m["unit_cell"] = m["unit_cell"] or (born["unit_cell"] if born else None)
    m["supercell"] = m["supercell"] or (born["supercell"] if born else None)
    m["map_tolerance"] = _number(m["map_tolerance"], "modes.map_tolerance", minimum=0, strict=True)
    m["ir_share"] = _number(m["ir_share"], "modes.ir_share", minimum=0, strict=True, maximum=1)
    if m["unit_cell"] is None or m["supercell"] is None:
        raise ConfigError(
            "modes needs unit_cell and supercell (or a dipole.born block that has them)"
        )
    m["supercell"] = _supercell(m["supercell"], "modes.supercell")
    for key in ("outcar", "file", "phonopy", "unit_cell"):
        m[key] = _path(base, m[key])
    m["implicit"] = False
    cfg["modes"] = m


def _forces(cfg: dict, base: Path) -> None:
    """Resolve the 'forces' block (force-based static Delta eps, ADR-0009)."""
    f = cfg["forces"]
    if f is None:
        return
    if cfg["dipole"]["model"] == "external":
        raise ConfigError("forces cannot be used with dipole.model = external")
    f = _merge(FORCES_SCHEMA, f, "forces")
    files = _strings(f["files"], "forces.files")
    if len(files) != len(cfg["trajectory"]["files"]):
        raise ConfigError(
            f"forces.files: {len(files)} files for {len(cfg['trajectory']['files'])} trajectory "
            "files (one per trajectory file, same order)"
        )
    f["files"] = [_path(base, x) for x in files]
    k = _number(f["stride_frames"], "forces.stride_frames", integer=True, minimum=1)
    if k > 1 and not all(x.endswith(".npy") for x in f["files"]):
        raise ConfigError("forces.stride_frames > 1 is for .npy forces only (OUTCAR: every step)")
    if int(cfg["trajectory"]["stride"]) != 1:
        raise ConfigError("forces need trajectory.stride = 1 (use forces.stride_frames for .npy)")
    if cfg["modes"] is None:
        raise ConfigError("forces need Gamma modes (a modes block or dipole.born.outcar)")
    f["stride_frames"] = k
    cfg["forces"] = f


def load(path: str | Path) -> dict:
    """Read and validate a YAML input file."""
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return validate(raw, path.parent)


TEMPLATE = {
    "point_charge": """\
# nozuchi input: fixed point charges (e.g. an akitsushima Buckingham + Coulomb run)
trajectory:
  files: [md_output/XDATCAR_nve]
  mode: concatenate            # concatenate | independent
  timestep_fs: 1.0             # MD time step
  frame_every: 2               # MD steps between frames (akitsushima traj_every, VASP NBLOCK)
  skip_frames: 0
  stride: 1
temperature: auto              # float [K] or auto
temperature_source: [md_output/thermo.log]
masses: {}                     # optional override [amu]; default: standard atomic weights
dipole:
  model: point_charge
  charges: {Mg: 2.0, O: -2.0}
epsilon_inf: 1.0               # rigid ion: electronic part absent
spectrum:
  max_lag_ps: auto             # auto = a quarter of the run
  window: hann                 # hann | boxcar | none
  zero_pad_factor: 4
  freq_unit: THz               # THz | cm-1 | meV
  freq_max: 30
  kk_check: true
  n_blocks: 5
  estimator: auto              # auto (default: mode if Gamma modes are known, else fsum) | mode | fsum | raw
fit:
  model: additive
  terms: [{type: lorentz, f0: 12.0}]
  freq_range: [1.0, 30.0]
output:
  dir: nozuchi_out
  prefix: mgo
  plot: true
  dipole_cache: true
""",
    "born_tensor": """\
# nozuchi input: Born effective charges from a VASP DFPT OUTCAR
trajectory:
  files: [aimd/2_nve/XDATCAR]
  mode: concatenate
  timestep_fs: 1.0             # POTIM
  frame_every: 1               # NBLOCK
  skip_frames: 0
temperature: auto
temperature_source: [aimd/2_nve/OUTCAR]   # also supplies POMASS for the f-sum masses
dipole:
  model: born_tensor
  born:
    outcar: dfpt_equiv/OUTCAR
    unit_cell: dfpt_equiv/POSCAR
    supercell: [3, 3, 2]
epsilon_inf: {outcar: dfpt_equiv/OUTCAR}
spectrum:
  freq_unit: THz
  freq_max: 30
  estimator: auto              # mode if the DFPT OUTCAR has Gamma eigenvectors (IBRION 5-8)
# modes: {phonopy: qpoints.yaml, unit_cell: POSCAR_phonopy, supercell: [3, 3, 2]}  # other source
forces:                        # optional: force-based static Delta eps, reported only
  files: [aimd/2_nve/OUTCAR]   # one per trajectory file (OUTCAR or .npy in eV/A)
fit:
  model: gervais_piriou
  terms: [{f_to: 11.0, f_lo: 16.0}]
  use_component: xx
output:
  dir: nozuchi_out
  prefix: zno
""",
    "external": """\
# nozuchi input: dipole time series computed elsewhere (Berry phase, Wannier centres)
trajectory:
  files: [aimd/XDATCAR]        # still read for the cell volume and frame count
  timestep_fs: 0.5
  frame_every: 1
temperature: 300.0
dipole:
  model: external
  external_file: dipole.dat    # columns: t [fs] (ignored), Mx, My, Mz
  external_unit: e*Angstrom    # e*Angstrom | Debye
epsilon_inf: 1.78
spectrum:
  estimator: raw               # f-sum normalisation is not defined here
  freq_unit: cm-1
output:
  dir: nozuchi_out
  prefix: water
""",
}
