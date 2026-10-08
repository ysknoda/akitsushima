"""``akitsushima gruneisen`` -- mode Grueneisen parameters along a q-path
and on a mesh, and their C_v-weighted mean gamma(T) (ADR-0018).
"""
import json
import os
import time as _time

import numpy as np

from ..solvers.qha import mean_gruneisen, mode_gruneisen, phonons_at, relax_at_volume
from .cmd_qha import _volume_setup
from .common import (
    _print_output_files,
    _print_results_header,
    _print_timing_summary,
)

__all__ = ["cmd_gruneisen"]


def _path_points(q_path, n_points):
    """Reduced q along the path: n_points per segment, the last corner
    once; labels are (name, index) of the corners."""
    pts = []
    for (_, q0), (_, q1) in zip(q_path[:-1], q_path[1:]):
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        pts += [q0 + (q1 - q0) * i / n_points for i in range(n_points)]
    pts.append(np.asarray(q_path[-1][1], float))
    return np.array(pts), [(lab, k * n_points) for k, (lab, _) in enumerate(q_path)]


def cmd_gruneisen(args) -> int:
    """Mode Grueneisen parameters from YAML input."""
    st = _volume_setup(args, "gruneisen", "Mode Grueneisen parameters (gruneisen)")
    if st is None:
        return 1
    sec, fn, types = st["sec"], st["force_fn"], st["types"]
    out_dir, is_root = st["out_dir"], st["is_root"]
    dv = float(sec.get("volume_step", 0.01))
    V0 = abs(float(np.linalg.det(st["lattice"])))
    if is_root:
        print(f"\n[Settings]")
        print(f"  V0         : {V0:.4f} A^3;  volumes V0 x (1 -+ {dv})")
        print(f"  supercell  : {st['supercell']}")

    t0 = _time.perf_counter()
    pcs, press = [], []
    for k, f in enumerate((1.0 - dv, 1.0, 1.0 + dv)):
        r = relax_at_volume(st["lattice"], st["positions"], types, fn, V0 * f,
                            st["supercell"], st["relax_kw"],
                            output_dir=os.path.join(out_dir, f"relax_v{k}"))
        press.append(r["pressure_GPa"])
        pcs.append(phonons_at(r["lattice"], r["positions"], types, fn,
                              st["supercell"], delta=sec.get("delta", 0.01),
                              masses=st["masses"], n_jobs=sec.get("n_jobs", 1),
                              born=st["born"]))
    st["t_sections"].append(("3 volumes: relax + IFC", _time.perf_counter() - t0))
    # Bulk modulus from the three pressures, for information.
    B = -V0 * (press[2] - press[0]) / (2 * dv * V0)

    out = {"command": "gruneisen",
           "input_hash": st["input_hash"], "nac": st["born"] is not None,
           "volumes_A3": [abs(float(np.linalg.det(p.lattice_prim))) for p in pcs],
           "pressures_GPa": press, "bulk_modulus_GPa": B}

    bs = sec.get("band_structure", {}) or {}
    if bs.get("q_path"):
        t0 = _time.perf_counter()
        q_path = [(e[0], e[1]) if isinstance(e, (list, tuple)) else (e["label"], e["q"])
                  for e in bs["q_path"]]
        qs, labels = _path_points(q_path, int(bs.get("n_points", 50)))
        g = mode_gruneisen(pcs, qs)
        rec = 2 * np.pi * np.linalg.inv(pcs[1].lattice_prim).T
        dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(qs @ rec, axis=0), axis=1))])
        n = g["gamma"].shape[1]
        np.savetxt(os.path.join(out_dir, "gruneisen_band.dat"),
                   np.column_stack([dist, qs, g["frequencies_THz"], g["gamma"]]),
                   fmt="%.8g", header="q_distance_invA q1 q2 q3 "
                   + " ".join(f"f{i}_THz" for i in range(n)) + " "
                   + " ".join(f"gamma{i}" for i in range(n))
                   + "\nlabels: " + " ".join(f"{l}@{i}" for l, i in labels))
        out["band_labels"] = labels
        st["t_sections"].append(("Band path", _time.perf_counter() - t0))

    mesh = sec.get("mesh", {}) or {}
    th = None
    if mesh:
        t0 = _time.perf_counter()
        m = int(mesh.get("q_mesh", 12))
        grid = np.array([[i / m, j / m, k / m] for i in range(m)
                         for j in range(m) for k in range(m)])
        grid -= np.round(grid)
        g = mode_gruneisen(pcs, grid)
        T = np.arange(float(mesh.get("t_min", 10.0)), float(mesh.get("t_max", 1000.0)) + 1e-9,
                      float(mesh.get("t_step", 10.0)))
        th = mean_gruneisen(g["frequencies_THz"], g["gamma"], T)
        np.savetxt(os.path.join(out_dir, "gruneisen_mean.dat"),
                   np.column_stack([T, th]), fmt="%.8g",
                   header=f"C_v-weighted mean mode Grueneisen parameter, mesh {m}^3\nT_K gamma")
        out.update(mesh_q=m, temperatures=T.tolist(), gamma_mean=th.tolist(),
                   gamma_mesh_min=float(np.nanmin(g["gamma"])),
                   gamma_mesh_max=float(np.nanmax(g["gamma"])))
        st["t_sections"].append(("Mesh", _time.perf_counter() - t0))
    if not is_root:
        return 0
    with open(os.path.join(out_dir, "gruneisen_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    _print_results_header("gruneisen")
    print(f"\n  Pressures at V0(1-dv), V0, V0(1+dv): "
          + ", ".join(f"{p:+.3f}" for p in press) + " GPa")
    print(f"  Bulk modulus from them: {B:.2f} GPa")
    if th is not None:
        print(f"\n  Mean gamma(T) on {out['mesh_q']}^3 (mode range "
              f"{out['gamma_mesh_min']:.3f} .. {out['gamma_mesh_max']:.3f}):")
        for t_show in (100.0, 300.0, 1000.0):
            k = int(np.argmin(np.abs(T - t_show)))
            print(f"    T = {T[k]:7.1f} K : {th[k]:.4f}")
    _print_output_files(out_dir)
    _print_timing_summary(st["t_sections"], sum(t for _, t in st["t_sections"]))
    return 0
