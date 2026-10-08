"""``akitsushima piezo`` -- relaxed-ion piezoelectric tensors e and d and
the electromechanical coupling k of a rigid-ion (fixed Z*) model
(ADR-0019).
"""
import json
import os
import time as _time

import numpy as np

from ..io.poscar import parse_poscar
from ..solvers.dielectric import piezoelectric_tensor, static_dielectric_by_field
from ..utils.mpi import _mpi_rank
from .cmd_elastic import cmd_elastic
from .common import (
    _born_setup,
    _check_poscar,
    _check_yaml,
    _cli_force_fn,
    _input_hash,
    _print_header,
    _print_output_files,
    _print_results_header,
    _print_timing_summary,
)

__all__ = ["cmd_piezo"]

# |e| below this is reported as consistent with a centrosymmetric cell
# (ADR-0019, Q15: judged numerically, no symmetry library).
_CENTRO_TOL_C_M2 = 1e-4

_V = ["xx", "yy", "zz", "yz", "xz", "xy"]


def cmd_piezo(args) -> int:
    """Piezoelectric response from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1
    pot_cfg = cfg.get("potential", {})
    sec = cfg.get("piezo", {}) or {}
    poscar = cfg.get("structure", {}).get("poscar", "POSCAR")
    pos_data = _check_poscar(poscar)
    if pos_data is None:
        return 1
    is_root = (_mpi_rank == 0)
    out_dir = sec.get("output_dir", "./piezo_output")
    if is_root:
        os.makedirs(out_dir, exist_ok=True)
        _print_header("Piezoelectric response (piezo)", cfg, poscar, pot_cfg)
    force_fn, _ = _cli_force_fn(pot_cfg, "cmd_piezo")
    types = pos_data["atom_types"]
    born = _born_setup(cfg, types, pot_cfg, required=True)
    in_hash = _input_hash(pos_data["lattice"], pos_data["positions"], types, pot_cfg)
    t_sections = []

    t0 = _time.perf_counter()
    el_dir = sec.get("load_elastic")
    if el_dir:
        el = json.load(open(os.path.join(el_dir, "elastic_results.json")))
        if el.get("input_hash") != in_hash:
            raise ValueError(f"piezo.load_elastic: {el_dir} was computed for "
                             "another structure or potential (input hash differs).")
    else:
        if cmd_elastic(args) != 0:
            return 1
        el_dir = (cfg.get("elastic", {}) or {}).get("output_dir", "./elastic_output")
        el = json.load(open(os.path.join(el_dir, "elastic_results.json")))
        t_sections.append(("Elastic constants + internal strain",
                           _time.perf_counter() - t0))
    ref = parse_poscar(os.path.join(el_dir, "POSCAR_reference"))
    if list(ref["atom_types"]) != list(types):
        raise ValueError("piezo: the elastic reference structure has another "
                         "atom order than the POSCAR (Born charges are per atom).")
    lat, pos = ref["lattice"], ref["positions"]
    vol = abs(float(np.linalg.det(lat)))

    t0 = _time.perf_counter()
    epsS = static_dielectric_by_field(
        force_fn, lat, pos, types, born["Z"], born["epsilon_inf"],
        field=float(sec.get("field", 5e-3)), relax_kw=sec.get("relax"),
        output_dir=os.path.join(out_dir, "field_relax"))
    t_sections.append(("eps^S by field relaxation (6 runs)", _time.perf_counter() - t0))
    pz = piezoelectric_tensor(born["Z"], np.array(el["internal_strain_A"]), vol,
                              np.array(el["S_per_GPa"]), epsS,
                              born["clamped_ion_piezo"])
    if not is_root:
        return 0
    emax = float(np.max(np.abs(pz["e_C_m2"])))
    out = {"command": "piezo", "input_hash": in_hash, "volume_A3": vol,
           "elastic_dir": os.path.abspath(el_dir),
           "rigid_ion_default": born["rigid_ion_default"],
           "epsilon_inf": born["epsilon_inf"].tolist(),
           "epsilon_static_clamped": epsS.tolist(),
           "epsilon_static_free": pz["epsilon_T_relative"].tolist(),
           "e_C_m2": pz["e_C_m2"].tolist(), "e_ionic_C_m2": pz["e_ionic_C_m2"].tolist(),
           "e_clamped_ion_C_m2": (None if born["clamped_ion_piezo"] is None
                                  else born["clamped_ion_piezo"].tolist()),
           "d_pC_N": pz["d_pC_N"].tolist(), "k": pz["k"].tolist(),
           "max_abs_e_C_m2": emax,
           "centrosymmetric_like": emax < _CENTRO_TOL_C_M2}
    with open(os.path.join(out_dir, "piezo_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(os.path.join(out_dir, "piezo_tensors.dat"), "w") as fh:
        for name, M, fmt in (("e [C/m^2]", pz["e_C_m2"], "{:11.5f}"),
                             ("d [pC/N]", pz["d_pC_N"], "{:11.4f}"),
                             ("k", pz["k"], "{:11.4f}")):
            fh.write(f"# {name}; rows x y z, columns " + " ".join(_V) + "\n")
            fh.write("\n".join(" ".join(fmt.format(v) for v in r) for r in M) + "\n")

    _print_results_header("piezo")
    print("\n[eps (relaxed-ion): strain-clamped / stress-free]")
    for a in range(3):
        print("  " + " ".join(f"{v:9.4f}" for v in epsS[a]) + "   |  "
              + " ".join(f"{v:9.4f}" for v in pz["epsilon_T_relative"][a]))
    for name, M, f in (("e (C/m^2)", pz["e_C_m2"], "{:9.4f}"),
                       ("d (pC/N)", pz["d_pC_N"], "{:9.3f}"),
                       ("k", pz["k"], "{:9.4f}")):
        print(f"\n[{name}]     " + " ".join(f"{v:>9s}" for v in _V))
        for a, r in zip("xyz", M):
            print(f"  {a}  " + " ".join(f.format(v) if np.isfinite(v) else f"{'-':>9s}"
                                       for v in r))
    if out["centrosymmetric_like"]:
        print(f"\n  NOTE: |e| < {_CENTRO_TOL_C_M2} C/m^2 everywhere -- consistent "
              f"with a centrosymmetric structure.")
    if born["clamped_ion_piezo"] is None:
        print("\n  Clamped-ion (electronic) e is zero for point charges; give "
              "born.clamped_ion_piezo for a model with electronic polarisation.")
    _print_output_files(out_dir)
    _print_timing_summary(t_sections, sum(t for _, t in t_sections))
    return 0
