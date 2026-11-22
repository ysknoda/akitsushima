"""``akitsushima bte`` -- lattice thermal conductivity from third-order force
constants in the relaxation-time approximation (ADR-0042).
"""

import os
import time as _time

import numpy as np

from ..constants import ATOMIC_MASS
from ..solvers.bte import BTECalculator, mass_variance
from ..solvers.phonon import PhononCalculator
from ..utils.mpi import _mpi_rank
from .common import (
    _born_setup,
    _check_structure,
    _check_yaml,
    _cli_force_fn,
    _print_header,
    _print_output_files,
    _print_results_header,
    _print_timing_summary,
    _resolve_masses,
    _structure_file,
)

__all__ = ["cmd_bte", "mass_variances_from_config"]


def mass_variances_from_config(sec: dict, atom_types: list, masses: np.ndarray):
    """(masses, g per atom) from the ``mass_variance`` / ``isotopes`` /
    ``alloy`` keys (ADR-0042 decision 16).

    - ``mass_variance: {Si: 2.0e-4}``: g of a species directly.
    - ``isotopes: {Si: {27.977: 0.9223, 28.976: 0.0468, ...}}``: mass [amu]
      -> fraction; g from the composition.
    - ``alloy: {host: Ge, Sn: 0.08}``: the host atoms get the virtual-
      crystal mass m_bar = sum f_i m_i and g = sum f_i (1 - m_i / m_bar)^2.
    Several keys add up per species."""
    masses = np.asarray(masses, dtype=float).copy()
    g = np.zeros(len(atom_types))
    species = set(atom_types)
    for sp, val in (sec.get("mass_variance") or {}).items():
        _check_species(sp, species, "mass_variance")
        g[[t == sp for t in atom_types]] += float(val)
    for sp, comp in (sec.get("isotopes") or {}).items():
        _check_species(sp, species, "isotopes")
        m = np.array([float(k) for k in comp])
        f = np.array([float(v) for v in comp.values()])
        g[[t == sp for t in atom_types]] += mass_variance(m, f)
    alloy = sec.get("alloy")
    if alloy:
        alloy = dict(alloy)
        host = alloy.pop("host", None)
        _check_species(host, species, "alloy.host")
        x = {k: float(v) for k, v in alloy.items()}
        if sum(x.values()) >= 1.0 or min(x.values(), default=0) < 0:
            raise ValueError("bte: alloy fractions must be >= 0 and sum to < 1")
        ms = [ATOMIC_MASS[host]] + [ATOMIC_MASS[k] for k in x]
        fs = [1.0 - sum(x.values())] + list(x.values())
        sel = [t == host for t in atom_types]
        masses[sel] = float(np.dot(fs, ms))
        g[sel] += mass_variance(ms, fs)
    return masses, g


def _check_species(sp, species, key):
    if sp not in species:
        raise ValueError(
            f"bte: {key} names '{sp}', which is not in the structure "
            f"({sorted(species)})"
        )


def cmd_bte(args) -> int:
    """RTA lattice thermal conductivity from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1
    pot_cfg = cfg.get("potential", {})
    sec = cfg.get("bte", {}) or {}
    poscar = _structure_file(cfg.get("structure", {}), "POSCAR")
    pos = _check_structure(cfg.get("structure", {}), "POSCAR")
    if pos is None:
        return 1
    is_root = _mpi_rank == 0
    verbose = sec.get("verbose", True) and not getattr(args, "quiet", False) and is_root
    out_dir = sec.get("output_dir", "./bte_output")
    os.makedirs(out_dir, exist_ok=True)
    fn, _ = _cli_force_fn(pot_cfg, "cmd_bte")
    if is_root:
        _print_header("Lattice thermal conductivity, RTA (bte)", cfg, poscar, pot_cfg)
    t_all = _time.perf_counter()
    sections = []
    types = pos["atom_types"]
    masses = (
        _resolve_masses(sec, types)
        if sec.get("masses")
        else np.array([ATOMIC_MASS.get(a, 1.0) for a in types])
    )
    masses, g = mass_variances_from_config(sec, types, masses)
    symmetry = sec.get("symmetry", True)
    if not isinstance(symmetry, bool):
        raise ValueError(f"bte.symmetry must be true or false, got {symmetry!r}")
    occupation = str(sec.get("occupation", "quantum")).lower()
    sc = sec.get("supercell", 2)
    sc2 = sec.get("supercell_fc2")
    fc2_calc = None
    ifc_path = sec.get("ifc_path")
    if ifc_path:
        fc2_calc = PhononCalculator.load_ifc(
            ifc_path,
            force_fn=fn,
            supercell=sc if sc2 is None else sc2,
            delta=sec.get("delta_fc2", 0.01),
        )
    bc = BTECalculator(
        pos["lattice"],
        pos["positions"],
        types,
        fn,
        supercell=sc,
        supercell_fc2=sc2,
        delta=sec.get("delta", 0.03),
        delta_fc2=sec.get("delta_fc2", 0.01),
        masses=masses,
        symmetry=symmetry,
        symprec=float(sec.get("symprec", 1e-5)),
        cutoff_pair=sec.get("cutoff_pair"),
        sym_fc3=bool(sec.get("sym_fc3", True)),
        fc2_calc=fc2_calc,
    )
    if np.any(g > 0):
        bc.set_mass_variance(g)
    bc.set_boundary_length(sec.get("boundary_length_um"))
    n_jobs = int(sec.get("n_jobs", 1))
    if verbose:
        print("\n[Settings]")
        print(
            f"  supercell  : fc3 {bc.ph3.supercell}, fc2 {bc.ph2.supercell}"
            f"{' (loaded ' + ifc_path + ')' if ifc_path else ''}"
        )
        print(
            f"  mesh       : {sec.get('mesh', [11, 11, 11])}   sigma {sec.get('sigma', 0.1)} THz"
        )
        print(
            f"  occupation : {occupation}   symmetry {symmetry}   sym_fc3 {bc.sym_fc3}"
        )
        if np.any(g > 0):
            print(f"  mass var.  : {dict(zip(types, g))}")
        if bc.boundary_length_m:
            print(f"  boundary L : {bc.boundary_length_m * 1e6:g} um")
    t0 = _time.perf_counter()
    bc.compute_fc2(n_jobs=n_jobs, verbose=verbose)
    sections.append(("fc2", _time.perf_counter() - t0))
    born = _born_setup(cfg, types, pot_cfg)
    if born is not None:
        bc.ph2.set_nac(born["Z"], born["epsilon_inf"])
        if verbose:
            print(f"  NAC        : Gonze-Lee, {bc.ph2._nac['n_G']} G-vectors")
    t0 = _time.perf_counter()
    bc.compute_fc3(n_jobs=n_jobs, verbose=verbose)
    sections.append(("fc3", _time.perf_counter() - t0))
    t0 = _time.perf_counter()
    temps = sec.get("temperatures", [100.0, 200.0, 300.0, 400.0, 500.0])
    res = bc.compute_kappa(
        sec.get("mesh", [11, 11, 11]),
        sigma=float(sec.get("sigma", 0.1)),
        temperatures=temps,
        occupation=occupation,
        verbose=verbose,
    )
    sections.append(("gamma + kappa", _time.perf_counter() - t0))
    if is_root:
        settings = {k: v for k, v in sec.items() if k != "verbose"}
        settings["nac"] = born is not None
        bc.save_results(out_dir, settings=settings)
        _print_results_header("bte")
        print(
            f"  {res['n_irreducible_q']} irreducible q points, fc3 "
            f"{bc.fc3_info['n_force_calls']} force calls"
        )
        print(
            f"  {'T (K)':>8} {'kappa_xx':>10} {'kappa_yy':>10} {'kappa_zz':>10}  (W/m/K)"
        )
        for T, k in zip(res["temperatures_K"], res["kappa_W_mK"]):
            print(f"  {T:8.1f} {k[0, 0]:10.3f} {k[1, 1]:10.3f} {k[2, 2]:10.3f}")
        _print_timing_summary(sections, _time.perf_counter() - t_all)
        _print_output_files(out_dir)
    return 0
