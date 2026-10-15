"""Bridge between YAML/JSON CLI configuration files and the engine's
force-field classes: load a config file and build a ready-to-use
``(force_fn, r_cut)`` pair from its ``potential:`` section.
"""
import json
import os
from typing import Callable, Optional, Tuple

from ..fitting import registry
from ..fitting.io_utils import _load_potential_json
from ..utils.optional_yaml import _HAS_YAML, _yaml

__all__ = ["force_fn_from_config", "_load_yaml", "_load_cfg", "_write_cfg",
           "_build_force_fn_from_yaml"]


def _load_yaml(path: str) -> dict:
    """
    Load a YAML config file.  Falls back to json.load if PyYAML is not
    installed (YAML is a superset of JSON so pure-JSON files always work).
    """
    with open(path) as fh:
        text = fh.read()
    if _HAS_YAML:
        return _yaml.safe_load(text)
    # Fallback: try JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"PyYAML is not installed and '{path}' is not valid JSON.\n"
            f"Install PyYAML:  pip install pyyaml\n"
            f"JSON error: {exc}") from exc


def _build_force_fn_from_yaml(
        pot_cfg: dict
) -> Tuple[Optional[Callable], float, Optional[dict], Optional[dict]]:
    """
    Construct a (force_fn, r_cut) pair from the 'potential' section of the
    YAML config.

    Supported types
    ---------------
    SW          Stillinger-Weber
    Tersoff     Tersoff T3
    LJ          Lennard-Jones 12-6
    Morse       Morse
    BornMayer   Born-Mayer repulsive
    EAM         Embedded Atom Method (requires tabulated functions --
                see fitting/registry.py's _build_from_yaml_eam for why a
                flat YAML ``params:`` dict cannot express this)
    Buckingham  Buckingham + Coulomb (legacy; uses potential_json path,
                special-cased below rather than going through the
                registry, since it returns a structurally different
                4-tuple with non-None buck_params/charges instead of a
                force_fn)

    Every other type's construction logic (species/params parsing,
    engine assembly, r_cut derivation) lives in exactly one place, next
    to that potential's ``fit --potential`` support:
    fitting/registry.py -- see its module docstring for why (Phase 1,
    potential registry consolidation).

    Returns
    -------
    (force_fn, r_cut, buck_params, charges) -- force_fn is always a
    complete force function; the last two are non-None only for
    Buckingham, for callers that still pass them to a solver's legacy
    keywords.
    """
    pot_type = str(pot_cfg.get("type", "")).strip().lower()

    if pot_type in ("buckingham", "buck"):
        # Legacy path: load from JSON + Ewald. Kept special-cased here
        # (not unified into the registry's build_from_yaml) because it
        # returns a structurally different result -- a (buck_params,
        # charges) pair for the caller's legacy constructor kwargs,
        # rather than a force_fn.
        pot_json_path = pot_cfg.get("potential_json",
                                    "./fitting_output/fitted_potential.json")
        if not os.path.isfile(pot_json_path):
            raise FileNotFoundError(
                f"Buckingham potential JSON not found: '{pot_json_path}'")
        buck_params, charges = _load_potential_json(pot_json_path)
        r_cut = float(pot_cfg.get("r_cut", 10.0))
        # ADR-0008: a complete force function (Buckingham + Coulomb by
        # ewald / pppm / esm), so every solver takes it; it used to be
        # None here, which only relax and md could handle (through their
        # legacy buck_params / charges keywords).  The pair is still
        # returned for those callers.
        from ..fitting.force_fn import make_force_fn
        coulomb = str(pot_cfg.get("coulomb_method", "ewald")).strip().lower()
        if coulomb not in ("ewald", "pppm", "esm"):
            raise ValueError(f"potential.coulomb_method must be ewald, pppm or esm, "
                             f"got '{coulomb}'.")
        if pot_cfg.get("esm") is not None and coulomb != "esm":
            raise ValueError("potential.esm is set but coulomb_method is not 'esm'.")
        force_fn = make_force_fn(
            "buckingham", buck_params=buck_params, charges=charges, r_cut=r_cut,
            ewald_alpha=pot_cfg.get("ewald_alpha"), coulomb_method=coulomb,
            pppm_accuracy=pot_cfg.get("pppm_accuracy", 1e-5),
            pppm_mesh=pot_cfg.get("pppm_mesh"),
            esm=pot_cfg.get("esm") if coulomb == "esm" else None)
        return force_fn, r_cut, buck_params, charges

    spec = registry.get_by_yaml_type(pot_type)
    if spec.build_from_yaml is None:
        raise ValueError(
            f"'{pot_type}' has no YAML bridge implemented "
            f"(fitting.registry entry '{spec.name}' has build_from_yaml=None).")

    force_fn, r_cut = spec.build_from_yaml(pot_cfg)
    return force_fn, r_cut, None, None


# Backward-compat wrappers (keep old JSON config path working)
def _load_cfg(path: str, default: dict) -> dict:
    """
    Load a JSON or YAML config (auto-detected by file extension),
    back-filling any keys missing from ``cfg`` with ``default``'s values.

    Bug-fix note (F11): the YAML branch used to return immediately from
    ``_load_yaml(path)`` without ever applying the default-fill loop
    below (which only ran for the JSON branch) -- so a YAML config
    omitting an optional key would raise KeyError the first time a
    caller indexed ``cfg[key]`` directly (e.g. ``cmd_fit``'s
    ``cfg["r_cut"]``), while the equivalent JSON config would not. Both
    formats now go through the same fill-in-missing-keys step.
    """
    if path.endswith((".yaml", ".yml")):
        cfg = _load_yaml(path)
    else:
        with open(path) as fh:
            cfg = json.load(fh)
    for k, v in default.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


def _write_cfg(path: str, default: dict) -> None:
    with open(path, "w") as fh:
        json.dump(default, fh, indent=2)
    print(f"Template config written to {path}")


def force_fn_from_config(potential: dict):
    """A force function from a ``potential:`` section, as the CLI builds it
    (ADR-0024, Q24): the same dict a relax / md / phonon YAML holds, e.g.
    ``{"type": "SW", "species": ["Si"], "params": {...}}``,
    ``{"type": "MLIP", "model": "out/atomic_elm_model.npz"}`` or
    ``{"type": "Buckingham", "potential_json": "fit.json", "r_cut": 8.0,
    "coulomb_method": "pppm"}``.

    The function is called as ``fn(lattice, positions, atom_types)`` ->
    (energy [eV], forces (N, 3) [eV/A], stress (3, 3) [eV/A^3,
    compressive-positive]); ``fn.r_cut`` is the cutoff the CLI hands the
    solvers, so passing ``fn`` to any solver reproduces the CLI.
    """
    fn, r_cut, _, _ = _build_force_fn_from_yaml(dict(potential))
    try:
        fn.r_cut = float(r_cut)
    except AttributeError:          # an object without a settable r_cut
        pass
    return fn

