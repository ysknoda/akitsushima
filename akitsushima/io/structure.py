"""Structure files beyond POSCAR: extxyz, LAMMPS data and CIF (ADR-0034).

``read_structure`` / ``write_structure`` dispatch on the format, which
``format="auto"`` takes from the file name:

    POSCAR*, CONTCAR*, *.vasp, *.poscar  -> poscar
    *.xyz, *.extxyz                      -> extxyz
    *.data, *.lmp                        -> lammps-data
    *.cif                                -> cif

Every reader returns the dict of :func:`parse_poscar` (``lattice``,
``positions`` Cartesian [A], ``atom_types``, ``species``, ``counts``,
``natoms``, ``fix_atoms``).  Fixed atoms come from POSCAR only; the other
formats give none.

extxyz, LAMMPS data and the LAMMPS dump frames of ``io.trajectory`` are
written and read with numpy only.  CIF is read through ASE (symmetry
expansion, occupancies and the dialects of the format are ASE's job) and
written in P1 by this module.

Conventions
-----------
* Stress: akitsushima's is positive under compression (ADR-0001), ASE's
  and extxyz's positive under tension; both in eV/A^3.  ``read_extxyz``
  returns akitsushima's sign, ``write_extxyz`` takes it and writes ASE's.
* LAMMPS: a cell is stored in the restricted triclinic orientation (a
  along x, b in the xy plane, c with cz > 0).  Writing rotates the
  structure into it (:func:`lammps_orientation`); reading back gives the
  same crystal in that orientation.  A left-handed cell cannot be
  written.
"""

from __future__ import annotations

import os
import re
import warnings

import numpy as np

from ..constants import ATOMIC_MASS
from .poscar import parse_poscar, write_poscar

__all__ = [
    "STRUCTURE_FORMATS",
    "detect_format",
    "lammps_orientation",
    "read_extxyz",
    "read_lammps_data",
    "read_structure",
    "structure_dict",
    "write_cif",
    "write_extxyz",
    "write_lammps_data",
    "write_structure",
]

STRUCTURE_FORMATS = ("poscar", "extxyz", "lammps-data", "cif")
ATOMIC_NUMBER = {el: z for z, el in enumerate(ATOMIC_MASS, start=1)}


def _sort_species(types) -> list:
    """Unique species by atomic number (unknown symbols last, by name)."""
    return sorted(set(types), key=lambda el: (ATOMIC_NUMBER.get(el, 10_000), el))


def structure_dict(lattice, positions, atom_types, fix_atoms=None) -> dict:
    """The dict :func:`parse_poscar` returns, for any structure."""
    lattice = np.asarray(lattice, dtype=float).reshape(3, 3)
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    atom_types = [str(t) for t in atom_types]
    if len(atom_types) != len(positions):
        raise ValueError(
            f"structure: {len(atom_types)} atom types for {len(positions)} positions."
        )
    species = list(dict.fromkeys(atom_types))
    n = len(atom_types)
    return {
        "lattice": lattice,
        "species": species,
        "counts": [atom_types.count(s) for s in species],
        "positions": positions,
        "atom_types": atom_types,
        "natoms": n,
        "fix_atoms": (
            np.zeros((n, 3), dtype=bool)
            if fix_atoms is None
            else np.asarray(fix_atoms, dtype=bool).reshape(n, 3)
        ),
    }


# ---------------------------------------------------------------------------
#  Format detection and dispatch
# ---------------------------------------------------------------------------


def detect_format(path: str) -> str:
    """Format from the file name (see the module docstring)."""
    name = os.path.basename(str(path))
    low = name.lower()
    if low.startswith(("poscar", "contcar")) or low.endswith((".vasp", ".poscar")):
        return "poscar"
    if low.endswith((".xyz", ".extxyz")):
        return "extxyz"
    if low.endswith((".data", ".lmp")):
        return "lammps-data"
    if low.endswith(".cif"):
        return "cif"
    raise ValueError(
        f"structure: cannot tell the format of '{path}' from its name; give "
        f"format: one of {list(STRUCTURE_FORMATS)}."
    )


def _resolve_format(path: str, fmt) -> str:
    fmt = "auto" if fmt is None else str(fmt).lower()
    if fmt == "auto":
        return detect_format(path)
    if fmt not in STRUCTURE_FORMATS:
        raise ValueError(
            f"structure: format must be auto or one of {list(STRUCTURE_FORMATS)}, "
            f"got '{fmt}'."
        )
    return fmt


def read_structure(path: str, format: str = "auto", index=-1, species=None) -> dict:
    """
    Read one structure from ``path``.

    format  : "auto" (from the name) | "poscar" | "extxyz" | "lammps-data" | "cif".
    index   : frame of a multi-frame extxyz or CIF file (default the last,
              as in ASE).  POSCAR and LAMMPS data hold one structure.
    species : LAMMPS data only -- element of each atom type, in type order,
              when the Masses section carries no element comments.
    """
    fmt = _resolve_format(path, format)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"structure file not found: '{path}'")
    index = -1 if index is None else int(index)
    if fmt in ("poscar", "lammps-data") and index not in (0, -1):
        raise ValueError(f"structure: a {fmt} file holds one structure; index {index}.")
    if fmt == "poscar":
        return parse_poscar(path)
    if fmt == "lammps-data":
        return read_lammps_data(path, species=species)
    if fmt == "extxyz":
        frames = read_extxyz(path)
        try:
            fr = frames[index]
        except IndexError:
            raise ValueError(
                f"structure: '{path}' has {len(frames)} frame(s); index {index}."
            ) from None
        return structure_dict(fr["lattice"], fr["positions"], fr["atom_types"])
    return _read_cif(path, index)


def write_structure(
    path: str,
    lattice,
    positions,
    atom_types,
    format: str = "auto",
    fix_atoms=None,
    title: str = "Written by akitsushima",
    **kw,
) -> None:
    """Write one structure; the format as in :func:`read_structure`.
    Extra keywords go to the format's writer (``species`` and ``masses``
    for LAMMPS data; ``energy``, ``forces`` and ``stress`` for extxyz)."""
    fmt = _resolve_format(path, format)
    if fix_atoms is not None and fmt != "poscar" and np.any(fix_atoms):
        warnings.warn(
            f"write_structure: fixed atoms are kept by POSCAR only; the {fmt} "
            "file will not carry them.",
            UserWarning,
            stacklevel=2,
        )
    if fmt == "poscar":
        write_poscar(
            path, lattice, positions, list(atom_types), title=title, fix_atoms=fix_atoms
        )
    elif fmt == "extxyz":
        write_extxyz(path, lattice, positions, atom_types, **kw)
    elif fmt == "lammps-data":
        write_lammps_data(path, lattice, positions, atom_types, title=title, **kw)
    else:
        write_cif(path, lattice, positions, atom_types, title=title)


# ---------------------------------------------------------------------------
#  extxyz
# ---------------------------------------------------------------------------

_KV = re.compile(r'([A-Za-z_][\w\-]*)\s*=\s*("([^"]*)"|\'([^\']*)\'|\S+)')


def _parse_comment(line: str) -> dict:
    """key=value pairs of an extxyz comment line (values as strings)."""
    out = {}
    for m in _KV.finditer(line):
        key, raw, dq, sq = m.groups()
        if dq is not None:
            val = dq
        elif sq is not None:
            val = sq
        else:
            val = raw
        out[key] = val
    return out


def _get(info: dict, key: str):
    """Case-insensitive lookup (ASE treats Lattice / Properties / pbc so)."""
    for k, v in info.items():
        if k.lower() == key.lower():
            return v
    return None


def _floats(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.replace(",", " ").split()])


def _stress_matrix(vals: np.ndarray) -> np.ndarray:
    if vals.size == 9:
        return vals.reshape(3, 3)
    if vals.size == 6:  # Voigt: xx yy zz yz xz xy
        xx, yy, zz, yz, xz, xy = vals
        return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    raise ValueError(f"extxyz: stress needs 6 or 9 values, got {vals.size}.")


def read_extxyz(
    path: str,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
) -> list:
    """
    Every frame of an extended-XYZ file, as dicts with ``lattice``,
    ``positions`` [A], ``atom_types``, ``natoms``, ``info`` (the comment
    line's key=value strings) and, when present, ``energy`` [eV],
    ``forces`` [eV/A] and ``stress`` [eV/A^3, akitsushima sign].

    The stress is read from ``stress_key`` (6 Voigt or 9 values, ASE sign),
    or else from ``virial`` (eV, virial = -V * stress in ASE's sign).
    Frames must be periodic in all three directions with a ``Lattice``.
    """
    frames = []
    with open(path) as fh:
        lines = fh.read().splitlines()
    i, n_lines = 0, len(lines)
    while i < n_lines:
        if not lines[i].strip():
            i += 1
            continue
        try:
            n = int(lines[i].split()[0])
        except ValueError:
            raise ValueError(
                f"extxyz '{path}' line {i + 1}: expected the atom count, got "
                f"'{lines[i].strip()}'."
            ) from None
        if i + 2 + n > n_lines:
            raise ValueError(f"extxyz '{path}': frame at line {i + 1} is truncated.")
        info = _parse_comment(lines[i + 1])
        frames.append(
            _extxyz_frame(
                path,
                i + 1,
                info,
                lines[i + 2 : i + 2 + n],
                n,
                energy_key,
                forces_key,
                stress_key,
            )
        )
        i += 2 + n
    if not frames:
        raise ValueError(f"extxyz '{path}': no frames.")
    return frames


def _extxyz_frame(path, lineno, info, body, n, energy_key, forces_key, stress_key):
    if len(body) != n:
        raise ValueError(f"extxyz '{path}': frame at line {lineno} is truncated.")
    lat = _get(info, "Lattice")
    if lat is None:
        raise ValueError(
            f"extxyz '{path}' line {lineno + 1}: no Lattice; akitsushima "
            "structures are periodic."
        )
    lattice = _floats(lat).reshape(3, 3)
    pbc = _get(info, "pbc")
    if pbc is not None and not all(
        t.upper() in ("T", "TRUE", "1") for t in pbc.split()
    ):
        raise ValueError(
            f"extxyz '{path}' line {lineno + 1}: pbc=\"{pbc}\"; akitsushima "
            "structures are periodic in all three directions."
        )
    props = _get(info, "Properties") or "species:S:1:pos:R:3"
    p = props.split(":")
    if len(p) % 3:
        raise ValueError(f"extxyz '{path}': bad Properties '{props}'.")
    cols, c0 = {}, 0
    for name, typ, ncol in zip(p[0::3], p[1::3], p[2::3]):
        cols[name] = (c0, int(ncol), typ.upper())
        c0 += int(ncol)
    if "species" not in cols or "pos" not in cols:
        raise ValueError(
            f"extxyz '{path}': Properties needs species and pos, got '{props}'."
        )
    rows = [ln.split() for ln in body]
    if any(len(r) < c0 for r in rows):
        raise ValueError(
            f"extxyz '{path}': frame at line {lineno} has short atom lines."
        )

    def column(name):
        start, ncol, _ = cols[name]
        return np.array([[float(x) for x in r[start : start + ncol]] for r in rows])

    s0 = cols["species"][0]
    fr = {
        "lattice": lattice,
        "positions": column("pos").reshape(n, 3),
        "atom_types": [r[s0] for r in rows],
        "natoms": n,
        "info": info,
    }
    if forces_key in cols:
        fr["forces"] = column(forces_key).reshape(n, 3)
    e = _get(info, energy_key)
    if e is not None:
        fr["energy"] = float(e)
    s = _get(info, stress_key)
    if s is not None:
        fr["stress"] = -_stress_matrix(_floats(s))
    elif _get(info, "virial") is not None:
        vol = abs(np.linalg.det(lattice))
        fr["stress"] = _stress_matrix(_floats(_get(info, "virial"))) / vol
    return fr


def _fmt_row(v, decimals: int) -> str:
    return " ".join(f"{x:.{decimals}f}" for x in v)


def extxyz_frame_text(
    lattice,
    positions,
    atom_types,
    energy=None,
    forces=None,
    stress=None,
    info: dict | None = None,
    decimals: int = 14,
    energy_decimals: int | None = None,
    stress_digits: int | None = None,
) -> str:
    """One extxyz frame as text.  ``stress`` in akitsushima's sign
    [eV/A^3]; it is written with ASE's (positive under tension)."""
    lattice = np.asarray(lattice, dtype=float)
    positions = np.asarray(positions, dtype=float)
    n = len(atom_types)
    ed = decimals if energy_decimals is None else energy_decimals
    sd = decimals + 2 if stress_digits is None else stress_digits
    props = "species:S:1:pos:R:3" + (":forces:R:3" if forces is not None else "")
    head = [
        f'Lattice="{_fmt_row(lattice.reshape(-1), decimals)}"',
        f"Properties={props}",
    ]
    if energy is not None:
        head.append(f"energy={float(energy):.{ed}f}")
    if stress is not None:
        s = -np.asarray(stress, dtype=float).reshape(-1)
        head.append('stress="' + " ".join(f"{x:.{sd}g}" for x in s) + '"')
    for k, v in (info or {}).items():
        head.append(f"{k}={v}")
    head.append('pbc="T T T"')
    out = [str(n), " ".join(head)]
    width = max(len(t) for t in atom_types) if n else 2
    F = None if forces is None else np.asarray(forces, dtype=float)
    for i in range(n):
        row = f"{atom_types[i]:<{width}s} {_fmt_row(positions[i], decimals)}"
        if F is not None:
            row += " " + _fmt_row(F[i], decimals)
        out.append(row)
    return "\n".join(out) + "\n"


def write_extxyz(
    path,
    lattice,
    positions,
    atom_types,
    energy=None,
    forces=None,
    stress=None,
    info=None,
    append: bool = False,
) -> None:
    """Write (or append) one extxyz frame with 14 decimals (ADR-0034 Q8).
    ``stress`` in akitsushima's sign; ASE's is written."""
    with open(path, "a" if append else "w") as fh:
        fh.write(
            extxyz_frame_text(
                lattice, positions, list(atom_types), energy, forces, stress, info
            )
        )


# ---------------------------------------------------------------------------
#  LAMMPS
# ---------------------------------------------------------------------------


def lammps_orientation(lattice):
    """
    The LAMMPS restricted-triclinic cell of ``lattice`` and the rotation R
    with ``r_lammps = r @ R`` (positions and forces; a tensor T becomes
    R^T T R).

    a = (ax, 0, 0), b = (bx, by, 0), c = (cx, cy, cz) with ax, by, cz > 0;
    the Gram matrix of the cell is unchanged, so R is a proper rotation.
    """
    L = np.asarray(lattice, dtype=float)
    if np.linalg.det(L) <= 0.0:
        raise ValueError(
            "LAMMPS: the cell is left-handed (det < 0); LAMMPS cannot hold "
            "it.  Swap two lattice vectors first."
        )
    a, b, c = L
    ax = np.linalg.norm(a)
    ahat = a / ax
    bx = b @ ahat
    by = np.linalg.norm(np.cross(ahat, b))
    cx = c @ ahat
    cy = (b @ c - bx * cx) / by
    cz = np.sqrt(max(c @ c - cx * cx - cy * cy, 0.0))
    Lp = np.array([[ax, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]])
    R = np.linalg.solve(L, Lp)
    return Lp, R


def _lammps_species(atom_types, species):
    if species is None:
        return _sort_species(atom_types)
    species = [str(s) for s in species]
    missing = sorted(set(atom_types) - set(species))
    if missing:
        raise ValueError(f"LAMMPS: species {species} lacks {missing}.")
    return species


def write_lammps_data(
    path,
    lattice,
    positions,
    atom_types,
    species=None,
    masses=None,
    title: str = "Written by akitsushima",
) -> None:
    """
    LAMMPS data file, atom_style atomic, metal units, 14 decimals.

    species : element of each atom type, in type order (default: the
              elements present, by atomic number).  The Masses section
              names them in comments, which :func:`read_lammps_data`
              reads back.
    masses  : {element: amu}; default ``constants.ATOMIC_MASS``.
    The structure is rotated to the LAMMPS orientation.
    """
    atom_types = [str(t) for t in atom_types]
    species = _lammps_species(atom_types, species)
    tid = {s: k + 1 for k, s in enumerate(species)}
    Lp, R = lammps_orientation(lattice)
    pos = np.asarray(positions, dtype=float) @ R
    masses = dict(masses or {})
    d = 14
    lines = [
        f"{title}",
        "",
        f"{len(atom_types)} atoms",
        f"{len(species)} atom types",
        "",
        f"{0.0:.{d}f} {Lp[0, 0]:.{d}f} xlo xhi",
        f"{0.0:.{d}f} {Lp[1, 1]:.{d}f} ylo yhi",
        f"{0.0:.{d}f} {Lp[2, 2]:.{d}f} zlo zhi",
    ]
    if np.any(np.abs([Lp[1, 0], Lp[2, 0], Lp[2, 1]]) > 0.0):
        lines.append(f"{Lp[1, 0]:.{d}f} {Lp[2, 0]:.{d}f} {Lp[2, 1]:.{d}f} xy xz yz")
    lines += ["", "Masses", ""]
    for s in species:
        m = masses.get(s, ATOMIC_MASS.get(s))
        if m is None:
            raise ValueError(f"LAMMPS: no mass for '{s}'; give masses.")
        lines.append(f"{tid[s]} {float(m):.8f} # {s}")
    lines += ["", "Atoms # atomic", ""]
    for i, (t, r) in enumerate(zip(atom_types, pos), start=1):
        lines.append(f"{i} {tid[t]} {_fmt_row(r, d)}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


_HEADER_KEYS = ("atom types", "atoms", "xlo xhi", "ylo yhi", "zlo zhi", "xy xz yz")
_SECTIONS = (
    "Atoms",
    "Velocities",
    "Masses",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
    "Pair Coeffs",
    "PairIJ Coeffs",
    "Bond Coeffs",
    "Angle Coeffs",
    "Dihedral Coeffs",
    "Improper Coeffs",
    "Ellipsoids",
    "Lines",
    "Triangles",
    "Bodies",
)


def read_lammps_data(path: str, species=None, atom_style: str | None = None) -> dict:
    """
    A LAMMPS data file (atom_style atomic or charge) as the dict of
    :func:`parse_poscar`, in the file's (LAMMPS) orientation, with the box
    origin moved to 0.

    The element of each type comes from the comment after its mass
    (``1 72.63 # Ge``), else from ``species`` (type order), else an error.
    The atom style comes from the ``Atoms # <style>`` comment, else
    ``atom_style``, else the column count (5 or 8: atomic).  Charges are
    dropped with a warning.
    """
    with open(path) as fh:
        lines = fh.read().splitlines()
    head, k = {}, 1  # line 0 is the title
    while k < len(lines):
        raw = lines[k]
        s = raw.split("#")[0].strip()
        if s in _SECTIONS or raw.strip().split("#")[0].strip() in _SECTIONS:
            break
        for key in _HEADER_KEYS:
            if s.endswith(key):
                head[key] = [float(x) for x in s[: -len(key)].split()]
                break
        k += 1
    for key in ("atoms", "atom types", "xlo xhi", "ylo yhi", "zlo zhi"):
        if key not in head:
            raise ValueError(f"LAMMPS data '{path}': header lacks '{key}'.")
    n_atoms, n_types = int(head["atoms"][0]), int(head["atom types"][0])
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = (
        head["xlo xhi"],
        head["ylo yhi"],
        head["zlo zhi"],
    )
    xy, xz, yz = head.get("xy xz yz", [0.0, 0.0, 0.0])
    lattice = np.array(
        [[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]], dtype=float
    )

    sections = {}
    while k < len(lines):
        raw = lines[k]
        name = raw.split("#")[0].strip()
        if name in _SECTIONS:
            comment = raw.split("#", 1)[1].strip() if "#" in raw else ""
            body, k = [], k + 1
            while k < len(lines) and not lines[k].strip():
                k += 1
            while (
                k < len(lines)
                and lines[k].strip()
                and lines[k].split("#")[0].strip() not in _SECTIONS
            ):
                body.append(lines[k])
                k += 1
            sections[name] = (comment, body)
        else:
            k += 1

    if "Atoms" not in sections:
        raise ValueError(f"LAMMPS data '{path}': no Atoms section.")
    # element of each type
    names = {}
    for ln in sections.get("Masses", ("", []))[1]:
        parts = ln.split("#", 1)
        t = int(parts[0].split()[0])
        if len(parts) > 1 and parts[1].split():
            names[t] = parts[1].split()[0]
    if len(names) < n_types:
        if species is None:
            raise ValueError(
                f"LAMMPS data '{path}': the Masses section does not name the "
                "element of every type (# <element> comments); give species "
                "in type order.  Elements are not guessed from masses."
            )
        species = [str(s) for s in species]
        if len(species) < n_types:
            raise ValueError(
                f"LAMMPS data '{path}': {n_types} atom types but species {species}."
            )
        names = {t + 1: species[t] for t in range(n_types)}

    comment, body = sections["Atoms"]
    style = (
        comment.split()[0].lower()
        if comment
        else (atom_style.lower() if atom_style else None)
    )
    ncol = len(body[0].split("#")[0].split()) if body else 0
    if style is None:
        if ncol in (5, 8):
            style = "atomic"
        else:
            raise ValueError(
                f"LAMMPS data '{path}': the Atoms section has {ncol} columns "
                "and no '# <style>' comment; give atom_style (atomic | charge)."
            )
    if style == "atomic":
        it, ix = 1, 2
    elif style == "charge":
        it, ix = 1, 3
        warnings.warn(
            f"LAMMPS data '{path}': atom_style charge; the charges are not "
            "used and are dropped.",
            UserWarning,
            stacklevel=2,
        )
    else:
        raise ValueError(
            f"LAMMPS data '{path}': atom_style '{style}' is not supported "
            "(atomic | charge)."
        )
    rows = [ln.split("#")[0].split() for ln in body]
    if len(rows) != n_atoms:
        raise ValueError(
            f"LAMMPS data '{path}': header says {n_atoms} atoms, Atoms has {len(rows)}."
        )
    rows.sort(key=lambda r: int(r[0]))
    types = [names[int(r[it])] for r in rows]
    pos = np.array([[float(x) for x in r[ix : ix + 3]] for r in rows])
    pos -= np.array([xlo, ylo, zlo])
    return structure_dict(lattice, pos, types)


def lammps_dump_frame_text(
    step: int,
    lattice,
    positions,
    atom_types,
    species,
    forces=None,
    decimals: int = 8,
) -> str:
    """One LAMMPS dump frame (``id type element x y z [fx fy fz]``, metal
    units) in the LAMMPS orientation."""
    Lp, R = lammps_orientation(lattice)
    pos = np.asarray(positions, dtype=float) @ R
    tid = {s: k + 1 for k, s in enumerate(species)}
    xy, xz, yz = Lp[1, 0], Lp[2, 0], Lp[2, 1]
    d = decimals
    out = [
        "ITEM: TIMESTEP",
        str(int(step)),
        "ITEM: NUMBER OF ATOMS",
        str(len(atom_types)),
    ]
    if xy == 0.0 and xz == 0.0 and yz == 0.0:
        out.append("ITEM: BOX BOUNDS pp pp pp")
        out += [f"{0.0:.{d}f} {Lp[k, k]:.{d}f}" for k in range(3)]
    else:
        xlo_b = min(0.0, xy, xz, xy + xz)
        xhi_b = Lp[0, 0] + max(0.0, xy, xz, xy + xz)
        out.append("ITEM: BOX BOUNDS xy xz yz pp pp pp")
        out.append(f"{xlo_b:.{d}f} {xhi_b:.{d}f} {xy:.{d}f}")
        out.append(f"{min(0.0, yz):.{d}f} {Lp[1, 1] + max(0.0, yz):.{d}f} {xz:.{d}f}")
        out.append(f"{0.0:.{d}f} {Lp[2, 2]:.{d}f} {yz:.{d}f}")
    cols = "id type element x y z" + (" fx fy fz" if forces is not None else "")
    out.append(f"ITEM: ATOMS {cols}")
    F = None if forces is None else np.asarray(forces, dtype=float) @ R
    for i, t in enumerate(atom_types):
        row = f"{i + 1} {tid[t]} {t} {_fmt_row(pos[i], d)}"
        if F is not None:
            row += " " + _fmt_row(F[i], d)
        out.append(row)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
#  CIF
# ---------------------------------------------------------------------------


def _read_cif(path: str, index) -> dict:
    try:
        import ase.io
    except ImportError:
        raise ImportError(
            "Reading CIF needs ASE (pip install 'akitsushima[ase]')."
        ) from None
    atoms = ase.io.read(path, index=index, format="cif")
    occ = atoms.info.get("occupancy")
    if occ:
        partial = {
            site: v
            for site, v in occ.items()
            if len(v) > 1 or any(abs(x - 1.0) > 1e-6 for x in v.values())
        }
        if partial:
            raise ValueError(
                f"CIF '{path}': partially occupied or mixed sites "
                f"{sorted(partial)[:5]}; akitsushima needs an ordered structure."
            )
    from .ase_adapter import from_ase_atoms

    return from_ase_atoms(atoms)


def write_cif(path, lattice, positions, atom_types, title: str = "akitsushima") -> None:
    """CIF in P1 (every atom listed), 14 decimals.  A CIF keeps the cell
    parameters, not its orientation."""
    L = np.asarray(lattice, dtype=float)
    a, b, c = np.linalg.norm(L, axis=1)

    def ang(u, v):
        return np.degrees(
            np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1.0, 1.0))
        )

    alpha, beta, gamma = ang(L[1], L[2]), ang(L[0], L[2]), ang(L[0], L[1])
    frac = np.asarray(positions, dtype=float) @ np.linalg.inv(L)
    name = re.sub(r"\s+", "_", str(title).strip()) or "akitsushima"
    d = 14
    out = [
        f"data_{name}",
        f"_cell_length_a {a:.{d}f}",
        f"_cell_length_b {b:.{d}f}",
        f"_cell_length_c {c:.{d}f}",
        f"_cell_angle_alpha {alpha:.{d}f}",
        f"_cell_angle_beta {beta:.{d}f}",
        f"_cell_angle_gamma {gamma:.{d}f}",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        "",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "'x, y, z'",
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    seen = {}
    for t, f in zip(atom_types, frac):
        seen[t] = seen.get(t, 0) + 1
        out.append(f"{t}{seen[t]} {t} {_fmt_row(f, d)} 1.0")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
