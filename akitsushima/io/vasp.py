"""VASP output readers: every ionic step of ``vasprun.xml`` and ``OUTCAR``.

Both readers return the same list of frame dicts (one per ionic step, in
run order), so either file can stand in for the other and the two can be
compared step by step (ADR-0021):

    lattice           : (3, 3) lattice vectors [A] (rows = a, b, c)
    positions         : (N, 3) Cartesian coordinates [A]
    atom_types        : list[str] of length N, in VASP's (POSCAR) order
    natoms            : int
    energy_free       : free energy TOTEN [eV]          (vasprun e_fr_energy)
    energy_wo_entropy : energy without entropy [eV]     (vasprun e_wo_entrp)
    energy_sigma0     : energy(sigma->0) [eV]           (vasprun e_0_energy)
    forces            : (N, 3) [eV/A]
    stress            : (3, 3) [eV/A^3], or None when the run wrote none

Stress convention: VASP's ``in kB`` line (OUTCAR) and ``<varray
name="stress">`` (vasprun.xml) carry the same numbers, positive under
compression -- the engine's own convention (ADR-0001) -- so they are only
scaled by ``VASP_KB_TO_EV_ANG3`` (+1/1602.176621), never sign-flipped.
The OUTCAR line is ordered xx yy zz xy yz zx and is placed into the
symmetric tensor accordingly.

Only the standard library (xml.etree, re) is used.
"""

import re
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from ..constants import VASP_KB_TO_EV_ANG3

__all__ = ["read_outcar", "read_vasprun"]


# ============================================================================
#  vasprun.xml
# ============================================================================


def _varray(elem) -> np.ndarray:
    return np.array([[float(x) for x in v.text.split()] for v in elem.findall("v")])


def _vasprun_atom_types(atominfo) -> list[str]:
    for arr in atominfo.findall("array"):
        if arr.get("name") == "atoms":
            return [
                rc.findall("c")[0].text.strip() for rc in arr.find("set").findall("rc")
            ]
    raise ValueError("vasprun.xml: <atominfo> has no 'atoms' array.")


def _vasprun_frame(calc, atom_types: list[str]) -> dict:
    struct = calc.find("structure")
    crystal = struct.find("crystal")
    lattice = None
    for va in crystal.findall("varray"):
        if va.get("name") == "basis":
            lattice = _varray(va)
    frac = None
    for va in struct.findall("varray"):
        if va.get("name") == "positions":
            frac = _varray(va)
    forces = stress = None
    for va in calc.findall("varray"):
        if va.get("name") == "forces":
            forces = _varray(va)
        elif va.get("name") == "stress":
            # symmetrised: vasprun prints the two triangles with
            # last-digit differences (~1e-6 kB)
            raw = _varray(va)
            stress = 0.5 * (raw + raw.T) * VASP_KB_TO_EV_ANG3
    # The <energy> child of <calculation> (not those of the <scstep>s)
    # holds the converged energies of this ionic step.
    energy = {i.get("name"): float(i.text) for i in calc.find("energy").findall("i")}
    if lattice is None or frac is None or forces is None:
        raise ValueError("vasprun.xml: <calculation> lacks basis, positions or forces.")
    return {
        "lattice": lattice,
        "positions": frac @ lattice,
        "atom_types": list(atom_types),
        "natoms": len(atom_types),
        "energy_free": energy["e_fr_energy"],
        "energy_wo_entropy": energy["e_wo_entrp"],
        "energy_sigma0": energy["e_0_energy"],
        "forces": forces,
        "stress": stress,
    }


def read_vasprun(path: str) -> list[dict]:
    """
    Read every ionic step of a ``vasprun.xml``.

    A file cut off mid-run (job killed) is read up to the last complete
    ``<calculation>``, with a warning; a file without any complete step
    raises ``ValueError``.
    """
    atom_types = None
    frames: list[dict] = []
    try:
        for _, elem in ET.iterparse(path, events=("end",)):
            if elem.tag == "atominfo":
                atom_types = _vasprun_atom_types(elem)
            elif elem.tag == "calculation":
                if atom_types is None:
                    raise ValueError(f"{path}: <calculation> before <atominfo>.")
                frames.append(_vasprun_frame(elem, atom_types))
                elem.clear()
    except ET.ParseError as exc:
        if not frames:
            raise ValueError(f"{path}: unreadable vasprun.xml ({exc}).") from exc
        warnings.warn(
            f"{path}: truncated vasprun.xml ({exc}); "
            f"kept the {len(frames)} complete ionic steps."
        )
    if not frames:
        raise ValueError(f"{path}: no ionic step found.")
    return frames


# ============================================================================
#  OUTCAR
# ============================================================================

_FLOAT = r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?"


def _outcar_atom_types(lines: list[str]) -> list[str]:
    species, counts = [], None
    for line in lines:
        if line.lstrip().startswith("VRHFIN"):
            # "VRHFIN =Ge: s2p2" -- one per POTCAR, in POSCAR species order
            species.append(line.split("=", 1)[1].split(":")[0].strip())
        elif "ions per type" in line:
            counts = [int(x) for x in line.split("=")[1].split()]
            break
    if counts is None or len(species) != len(counts):
        raise ValueError("OUTCAR: cannot determine species (VRHFIN / ions per type).")
    return [sp for sp, n in zip(species, counts) for _ in range(n)]


def read_outcar(path: str) -> list[dict]:
    """
    Read every ionic step of an ``OUTCAR``.

    Per step VASP writes, in this order: the ``in kB`` stress line, the
    ``VOLUME and BASIS-vectors are now`` block (the lattice of this step),
    the ``POSITION  TOTAL-FORCE`` table, and the ``FREE ENERGIE`` block.
    A step is kept once its energy block has been read, so an OUTCAR cut
    off mid-step yields the complete steps only.
    """
    with open(path, "r", errors="replace") as fh:
        lines = fh.readlines()
    atom_types = _outcar_atom_types(lines)
    n = len(atom_types)

    frames: list[dict] = []
    cur: dict = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("  in kB"):
            v = [float(x) for x in re.findall(_FLOAT, line)]
            xx, yy, zz, xy, yz, zx = v[:6]
            cur["stress"] = (
                np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]])
                * VASP_KB_TO_EV_ANG3
            )
        elif "VOLUME and BASIS-vectors are now" in line:
            # 4 lines on: "direct lattice vectors" header, then 3 rows
            j = i + 1
            while "direct lattice vectors" not in lines[j]:
                j += 1
            cur["lattice"] = np.array(
                [[float(x) for x in re.findall(_FLOAT, lines[j + k])[:3]] for k in (1, 2, 3)]
            )
            i = j + 3
        elif line.lstrip().startswith("POSITION") and "TOTAL-FORCE" in line:
            rows = np.array(
                [[float(x) for x in re.findall(_FLOAT, lines[i + 2 + k])[:6]] for k in range(n)]
            )
            cur["positions"] = rows[:, :3]
            cur["forces"] = rows[:, 3:]
            i += 2 + n
        elif "FREE ENERGIE OF THE ION-ELECTRON SYSTEM" in line:
            j = i + 1
            while "energy  without entropy" not in lines[j]:
                if "free  energy   TOTEN" in lines[j]:
                    cur["energy_free"] = float(re.findall(_FLOAT, lines[j])[0])
                j += 1
            e_wo, e_0 = (float(x) for x in re.findall(_FLOAT, lines[j])[:2])
            cur["energy_wo_entropy"], cur["energy_sigma0"] = e_wo, e_0
            if "positions" not in cur or "lattice" not in cur:
                raise ValueError(
                    f"{path}: energy block at line {j + 1} "
                    f"without a preceding lattice/force block."
                )
            frames.append(
                {
                    "lattice": cur["lattice"],
                    "positions": cur["positions"],
                    "atom_types": list(atom_types),
                    "natoms": n,
                    "energy_free": cur["energy_free"],
                    "energy_wo_entropy": e_wo,
                    "energy_sigma0": e_0,
                    "forces": cur["forces"],
                    "stress": cur.get("stress"),
                }
            )
            cur = {}
            i = j
        i += 1
    if not frames:
        raise ValueError(f"{path}: no complete ionic step found.")
    return frames
