"""akitsushima-side preparation of an mdspec run (ADR-0050 decisions 5, 9,
10; ADR-0052): the charge model ``auto``, forces taken from the trajectory,
and Gamma modes computed with the same potential as the MD.

Everything here produces an ordinary internal (nozuchi-format)
configuration: the computed Gamma modes are written as an .npz plus the
unit-cell POSCAR they belong to, and ``modes.file`` points to them, so the
ported pipeline runs unchanged.
"""

from __future__ import annotations

import contextlib
import copy
import io
import warnings
from pathlib import Path

import numpy as np

from . import charges as ch
from .config import ConfigError, validate
from .dipole import mean_positions
from .trajectory import read_header

__all__ = ["DIRECT_MODES_MAX_ATOMS", "prepare"]

# Gamma modes of the MD cell itself only up to this size (ADR-0050 decision
# 10; ADR-0046: dense eigh 42 s at N = 1024 on a loaded machine).
DIRECT_MODES_MAX_ATOMS = 1000
# Optical modes below this are reported as unstable [THz].
IMAGINARY_THZ = -0.05


def _potential(cfg: dict):
    """(force_fn, r_cut, point charges or None) of the YAML potential, or None."""
    pot = cfg.get("potential")
    if not pot:
        return None
    from ..cli.common import _cli_force_fn
    from ..solvers.config_bridge import _build_force_fn_from_yaml

    charges = _build_force_fn_from_yaml(pot)[3]
    force_fn, r_cut = _cli_force_fn(pot, "mdspec")
    return force_fn, float(r_cut), (dict(charges) if charges else None)


def _resolve_model(cfg: dict, raw: dict, pot, log) -> None:
    """dipole.model auto: born.born_charges > the potential's point charges >
    mdspec.dipole.charges (ADR-0050 decision 9)."""
    dp = raw.setdefault("dipole", {})
    model = dp.get("model", "auto")
    if model == "auto":
        born = (cfg.get("born") or {}).get("born_charges")
        pot_q = pot[2] if pot else None
        if born:
            dp["model"] = "born_tensor"
            src = "born.born_charges"
        elif pot_q:
            if dp.get("charges") and dp["charges"] != pot_q:
                warnings.warn(
                    "dipole.model auto: the potential's point charges are used; "
                    "mdspec.dipole.charges is ignored (set model: point_charge to use it)",
                    stacklevel=3,
                )
            dp["model"], dp["charges"] = "point_charge", pot_q
            src = f"the potential's point charges {pot_q}"
        elif dp.get("charges"):
            dp["model"] = "point_charge"
            src = "mdspec.dipole.charges"
        else:
            raise ConfigError(
                "dipole.model auto found no charges: give born.born_charges, a potential "
                "with point charges, or mdspec.dipole.charges (an MLIP, SW, Tersoff or EAM "
                "has no charges of its own)"
            )
        log(f"[dipole] model auto -> {dp['model']} from {src}")
    if dp["model"] in ("point_charge", "born_tensor"):
        warnings.warn(
            "fixed charges: fluctuations of the electronic polarisation (charge "
            "transfer, induced dipoles) are not in M(t); eps_inf is added as a constant",
            stacklevel=3,
        )


def _relaxed_ions(lattice, positions, types, force_fn, r_cut, out_dir):
    """Positions relaxed at a fixed cell (on a supercell valid for the
    potential); the first len(types) atoms are the cell (cell-major order)."""
    from ..solvers.geometry_optimiser import GeometryOptimiser
    from ..solvers.qha import build_supercell
    from ..solvers.raman import auto_supercell

    S = auto_supercell(lattice, r_cut)
    L, P, T = build_supercell(lattice, positions, types, S)
    with contextlib.redirect_stdout(io.StringIO()):
        r = GeometryOptimiser(
            L,
            P,
            T,
            force_fn=force_fn,
            relax_cell=False,
            fmax=1e-4,
            max_steps=5000,
            verbose=False,
            output_dir=out_dir,
        ).run()
    return np.asarray(r["positions"])[: len(types)], bool(r.get("converged", True))


def gamma_modes(
    lattice,
    positions,
    types,
    masses,
    force_fn,
    r_cut,
    relax: bool,
    work_dir: str,
    log=print,
):
    """Frequencies [THz] and mass-weighted eigenvectors (3n, n, 3) of the
    Gamma dynamical matrix of one cell, and the positions used."""
    from ..solvers.phonon import _THZ_TO_CM, PhononCalculator
    from ..solvers.raman import auto_supercell
    from ..solvers.raman import gamma_modes as _gm

    if relax:
        positions, ok = _relaxed_ions(
            lattice, positions, types, force_fn, r_cut, work_dir
        )
        if not ok:
            log(
                "  WARNING: the fixed-cell relaxation for the Gamma modes did not converge"
            )
    pc = PhononCalculator(
        lattice,
        positions,
        types,
        force_fn,
        supercell=auto_supercell(lattice, r_cut),
        masses=masses,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        pc.compute_ifc(verbose=False)
    w_cm, E = _gm(pc.build_dynamical_matrix(np.zeros(3)))
    n = len(types)
    order = np.argsort(w_cm)
    f = w_cm[order] / _THZ_TO_CM
    e = E[:, order].T.reshape(3 * n, n, 3)
    return f, e, positions


def _modes_spec(raw: dict):
    """'off', None (not given), or the user's modes mapping."""
    m = raw.get("modes")
    if isinstance(m, str):
        if m.lower() != "off":
            raise ConfigError(f"mdspec.modes: a mapping or 'off', got {m!r}")
        return "off"
    return m


def prepare(cfg: dict, base, log=print) -> dict:
    """The validated internal configuration of an akitsushima YAML dict."""
    from .yaml_io import to_internal

    base = Path(base)
    raw = to_internal(cfg)
    pot = _potential(cfg)
    _resolve_model(cfg, raw, pot, log)
    spec = _modes_spec(raw)
    if spec == "off":
        raw["modes"] = None
        return validate(raw, base)
    given = isinstance(spec, dict) and any(
        spec.get(k) for k in ("file", "outcar", "phonopy")
    )
    external = raw["dipole"]["model"] == "external"
    if given or external or pot is None:
        if isinstance(spec, dict) and not given:
            raise ConfigError(
                "mdspec.modes without file / outcar / phonopy needs a potential: block "
                "(the Gamma modes are then computed with it)"
            )
        return validate(raw, base)

    # In-house Gamma modes (ADR-0050 decisions 5, 10).
    pre = copy.deepcopy(raw)
    pre["modes"], pre["forces"] = None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pre = validate(pre, base)
    header = read_header(pre["trajectory"]["files"][0], pre["trajectory"]["species"])
    masses, _, _ = ch.run_masses(pre, header)
    frac_mean, lattice = mean_positions(pre, header)
    symbols = header.symbols
    out_dir = Path(pre["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = pre["output"]["prefix"]
    spec = dict(spec or {})
    relax = spec.pop("relax", True)
    if not isinstance(relax, bool):
        raise ConfigError("mdspec.modes.relax must be true or false")
    if spec.get("unit_cell"):
        if spec.get("supercell") is None:
            raise ConfigError("mdspec.modes.unit_cell needs mdspec.modes.supercell")
        lat_u0, sym_u, frac_u = ch.read_poscar(
            base / spec["unit_cell"]
            if not Path(spec["unit_cell"]).is_absolute()
            else spec["unit_cell"]
        )
        S = np.asarray(spec["supercell"])
        S = np.diag(S) if S.ndim == 1 else S
        # the unit cell implied by the MD cell: its volume and orientation
        lat_u = np.linalg.inv(S) @ lattice
        cell, types, pos = lat_u, list(sym_u), frac_u @ lat_u
        mass_of = {s: m for s, m in zip(symbols, masses)}
        missing = sorted(set(types) - set(mass_of))
        if missing:
            raise ConfigError(
                f"mdspec.modes.unit_cell: species {missing} are not in the MD cell"
            )
        cell_masses = np.array([mass_of[s] for s in types])
        sc = spec["supercell"]
        what = f"unit cell {spec['unit_cell']} at the MD cell's mean lattice"
    else:
        if len(symbols) > DIRECT_MODES_MAX_ATOMS:
            warnings.warn(
                f"mode estimator unavailable: {len(symbols)} atoms (> "
                f"{DIRECT_MODES_MAX_ATOMS}) and no mdspec.modes.unit_cell; fsum / raw are "
                "used (give modes: {unit_cell, supercell} for a crystal)",
                stacklevel=2,
            )
            return validate(raw, base)
        cell, types, pos, cell_masses = (
            lattice,
            list(symbols),
            frac_mean @ lattice,
            masses,
        )
        sc = [1, 1, 1]
        what = "the MD cell at its mean positions"
    force_fn, r_cut = pot[0], pot[1]
    log(
        f"[modes] Gamma modes of {what} with the potential "
        f"({len(types)} atoms, {'relaxed at fixed cell' if relax else 'as given'})"
    )
    f, e, pos = gamma_modes(
        cell,
        pos,
        types,
        cell_masses,
        force_fn,
        r_cut,
        relax,
        str(out_dir / f"{prefix}_gamma_relax"),
        log,
    )
    bad = f[3:][f[3:] < IMAGINARY_THZ]
    if len(bad):
        warnings.warn(
            f"Gamma modes: {len(bad)} unstable optical mode(s), lowest "
            f"{bad.min():.3f} THz",
            stacklevel=2,
        )
    from ..io.poscar import write_poscar

    npz = out_dir / f"{prefix}_gamma_modes.npz"
    unit = out_dir / f"{prefix}_gamma_unit.vasp"
    np.savez(npz, frequencies_THz=f, eigenvectors=e)
    write_poscar(str(unit), cell, pos, types, title="mdspec Gamma-mode cell (ADR-0052)")
    log(f"[modes] wrote {npz.name} and {unit.name}")
    raw["modes"] = {
        "file": str(npz),
        "unit_cell": str(unit),
        "supercell": list(np.asarray(sc).tolist()),
        **{k: v for k, v in spec.items() if k in ("map_tolerance", "ir_share")},
    }
    return validate(raw, base)


def load(path, log=print) -> dict:
    """Read an akitsushima YAML file and return the prepared internal
    configuration.  mdspec paths are read against the YAML file's directory
    (as in nozuchi); the potential section, as in every other command,
    against the working directory."""
    import yaml

    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return prepare(cfg, path.parent, log)
