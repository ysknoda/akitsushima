"""``akitsushima dielectric`` -- static dielectric tensor, LO-TO splitting
and infrared mode analysis from Born charges (ADR-0019).
"""
import json
import os
import time as _time

import numpy as np

from ..solvers import PhononCalculator
from ..solvers.dielectric import dielectric_response
from ..solvers.phonon import _THZ_TO_CM
from ..solvers.qha import phonons_at
from .cmd_qha import _volume_setup
from .common import _print_output_files, _print_results_header, _print_timing_summary

__all__ = ["cmd_dielectric"]


def _freqs(pc, q, qd=None):
    return np.sort(PhononCalculator._eig_to_freq(
        np.linalg.eigvalsh(pc.build_dynamical_matrix(q, qd))) / _THZ_TO_CM)


def _load_phonon(sec, st):
    """phonon output reused when ``load_ifc`` names a directory whose
    phonon_results.json has this run's input hash (POSCAR + potential)."""
    d = sec["load_ifc"]
    meta = json.load(open(os.path.join(d, "phonon_results.json")))
    if meta.get("input_hash") != st["input_hash"]:
        raise ValueError(f"dielectric.load_ifc: {d} was computed for another "
                         "structure or potential (input hash differs).")
    return PhononCalculator.load_ifc(os.path.join(d, "ifc.npz"),
                                     force_fn=st["force_fn"],
                                     supercell=meta["supercell"],
                                     delta=meta["delta_A"])


def cmd_dielectric(args) -> int:
    """Dielectric tensor and IR modes from YAML input."""
    st = _volume_setup(args, "dielectric", "Dielectric tensor (dielectric)")
    if st is None:
        return 1
    sec, born, is_root, out_dir = st["sec"], st["born"], st["is_root"], st["out_dir"]
    t0 = _time.perf_counter()
    if sec.get("load_ifc"):
        pc = _load_phonon(sec, st)
        pc.set_nac(born["Z"], born["epsilon_inf"])
    else:
        pc = phonons_at(st["lattice"], st["positions"], st["types"], st["force_fn"],
                        st["supercell"], delta=sec.get("delta", 0.01),
                        masses=st["masses"], n_jobs=sec.get("n_jobs", 1), born=born)
    st["t_sections"].append(("Force constants", _time.perf_counter() - t0))
    vol = abs(float(np.linalg.det(pc.lattice_prim)))
    D0 = pc.build_dynamical_matrix(np.zeros(3))          # analytic (TO)
    res = dielectric_response(D0, pc.masses, born["Z"], born["epsilon_inf"], vol)
    dirs = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    lo = {k: _freqs(pc, np.zeros(3), np.array(v, float)).tolist()
          for k, v in dirs.items()}
    to = _freqs(pc, np.zeros(3)).tolist()
    if not is_root:
        return 0
    out = {"command": "dielectric", "input_hash": st["input_hash"],
           "natoms_cell": pc.N_basis, "volume_A3": vol,
           "born_charges_e": born["Z"].tolist(),
           "born_sources": born["sources"],
           "sum_rule_correction_e": born["sum_rule_correction"],
           "rigid_ion_default": born["rigid_ion_default"],
           "epsilon_inf": res["epsilon_inf"].tolist(),
           "epsilon_0": res["epsilon_0"].tolist(),
           "gamma_TO_THz": to, "gamma_LO_THz": lo,
           "modes": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in m.items()} for m in res["modes"]]}
    with open(os.path.join(out_dir, "dielectric_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(os.path.join(out_dir, "ir_modes.dat"), "w") as fh:
        fh.write("# Gamma modes: frequency_THz IR_intensity_e2_amu "
                 "Zm_x Zm_y Zm_z d_eps_xx d_eps_yy d_eps_zz\n")
        for m in res["modes"]:
            fh.write(f"{m['frequency_THz']:12.5f} {m['ir_intensity_e2_amu']:12.6f} "
                     + " ".join(f"{v:9.4f}" for v in m["mode_effective_charge_e"]) + " "
                     + " ".join(f"{v:10.5f}" for v in np.diag(m["delta_epsilon"])) + "\n")

    _print_results_header("dielectric")
    print("\n[eps_inf]")
    print("\n".join("  " + " ".join(f"{v:9.4f}" for v in r) for r in res["epsilon_inf"]))
    print("\n[eps_0 (relaxed-ion, static)]")
    print("\n".join("  " + " ".join(f"{v:9.4f}" for v in r) for r in res["epsilon_0"]))
    print("\n[Gamma modes]   f (THz)   IR intensity (e^2/amu)   d_eps (xx yy zz)")
    for m in res["modes"]:
        tag = "acoustic" if m["acoustic"] else ""
        print(f"  {m['frequency_THz']:10.4f}  {m['ir_intensity_e2_amu']:12.5f}   "
              + " ".join(f"{v:8.4f}" for v in np.diag(m["delta_epsilon"])) + f"  {tag}")
    print("\n[LO frequencies at Gamma by direction of approach (THz)]")
    for k, f in lo.items():
        print(f"  q || {k} : " + " ".join(f"{v:.3f}" for v in f[3:]))
    _print_output_files(out_dir)
    _print_timing_summary(st["t_sections"], sum(t for _, t in st["t_sections"]))
    return 0
