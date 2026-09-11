"""VASP POSCAR / CONTCAR structure file I/O."""
from typing import Optional

import numpy as np


def parse_poscar(path: str) -> dict:
    """
    Parse a VASP POSCAR / CONTCAR file (VASP 4 and 5 formats).

    Returns
    -------
    dict
        lattice    : (3,3) float array -- lattice vectors in A (rows = a, b, c)
        species    : list[str]         -- element symbols
        counts     : list[int]         -- atom counts per species
        positions  : (N,3) float array -- Cartesian coordinates in A
        atom_types : list[str]         -- element label for each atom
        natoms     : int
        fix_atoms  : (N,3) bool array  -- True = frozen DoF (Selective Dynamics).
                     All False when the POSCAR has no Selective Dynamics block.

    Selective Dynamics
    ------------------
    When the POSCAR contains a Selective Dynamics line (starting with 'S'
    or 's'), the sixth flag column of each coordinate line is parsed::

        T T F   ->  fix_atoms[i] = [False, False, True]  (z frozen)
        F F F   ->  fix_atoms[i] = [True,  True,  True]  (all frozen)

    The resulting ``fix_atoms`` array can be passed directly to
    :class:`GeometryOptimiser`, :class:`NVTSimulation`, etc.::

        struct = parse_poscar("POSCAR")
        go = GeometryOptimiser(
            struct["lattice"], struct["positions"], struct["atom_types"],
            fix_atoms = struct["fix_atoms"],
            force_fn  = engine,
        )
    """
    with open(path, "r") as fh:
        raw_lines = fh.readlines()

    scale = float(raw_lines[1].strip())
    lattice = np.array(
        [list(map(float, raw_lines[i].split())) for i in range(2, 5)]
    ) * scale

    token6 = raw_lines[5].split()
    if token6[0].isalpha():           # VASP 5+
        species = token6
        count_line = raw_lines[6].split()
        coord_line_idx = 7
    else:                              # VASP 4
        species = [f"X{i}" for i in range(len(token6))]
        count_line = token6
        coord_line_idx = 6

    counts = list(map(int, count_line))
    natoms = sum(counts)

    # Check for Selective Dynamics header
    selective_dynamics = False
    if raw_lines[coord_line_idx].strip()[0].upper() == 'S':
        selective_dynamics = True
        coord_line_idx += 1            # skip 'Selective dynamics' line

    coord_type = raw_lines[coord_line_idx].strip()[0].upper()
    coord_line_idx += 1

    raw = []
    fix_rows = []
    for i in range(natoms):
        tokens = raw_lines[coord_line_idx + i].split()
        raw.append(list(map(float, tokens[:3])))
        if selective_dynamics and len(tokens) >= 6:
            # 'F' -> True (frozen), 'T' -> False (free)
            fix_rows.append([t.upper() == 'F' for t in tokens[3:6]])
        else:
            fix_rows.append([False, False, False])

    raw_arr = np.array(raw, dtype=float)
    positions = raw_arr @ lattice if coord_type == "D" else raw_arr * scale
    fix_atoms = np.array(fix_rows, dtype=bool)

    atom_types = []
    for sp, cnt in zip(species, counts):
        atom_types.extend([sp] * cnt)

    return {
        "lattice": lattice,
        "species": species,
        "counts": counts,
        "positions": positions,
        "atom_types": atom_types,
        "natoms": natoms,
        "fix_atoms": fix_atoms,
    }


def write_poscar(path: str,
                  lattice: np.ndarray,
                  positions: np.ndarray,
                  atom_types: list,
                  title: str = "Written by akitsushima",
                  fix_atoms: Optional[np.ndarray] = None,
                  ) -> None:
    """
    Write a VASP 5 POSCAR file in Direct (fractional) coordinates.

    Parameters
    ----------
    path       : output file path.
    lattice    : (3, 3) lattice vectors [A].
    positions  : (N, 3) Cartesian coordinates [A].
    atom_types : list[str] of length N.
    title      : comment line (line 1 of POSCAR).
    fix_atoms  : (N, 3) bool array -- if provided, a Selective Dynamics
                 block is written.  True -> 'F' (frozen), False -> 'T' (free).
                 None -> no Selective Dynamics block (all atoms free).

    Example
    -------
    >>> struct = parse_poscar("POSCAR")          # reads fix_atoms
    >>> # ... modify positions ...
    >>> write_poscar("POSCAR_out",
    ...              struct["lattice"], struct["positions"],
    ...              struct["atom_types"],
    ...              fix_atoms=struct["fix_atoms"])  # round-trip
    """
    species_order = []
    seen = set()
    for at in atom_types:
        if at not in seen:
            species_order.append(at)
            seen.add(at)

    counts = [atom_types.count(sp) for sp in species_order]

    # Fractional coordinates
    inv_lat = np.linalg.inv(lattice)
    frac = positions @ inv_lat

    # Validate fix_atoms if provided
    N = len(atom_types)
    if fix_atoms is not None:
        fa = np.asarray(fix_atoms, dtype=bool)
        if fa.shape != (N, 3):
            raise ValueError(
                f"write_poscar: fix_atoms must have shape ({N}, 3), "
                f"got {fa.shape}.")
    else:
        fa = None

    with open(path, "w") as fh:
        fh.write(f"{title}\n")
        fh.write("1.0\n")
        for vec in lattice:
            fh.write(f"  {vec[0]:20.14f}  {vec[1]:20.14f}  {vec[2]:20.14f}\n")
        fh.write("  " + "  ".join(species_order) + "\n")
        fh.write("  " + "  ".join(str(c) for c in counts) + "\n")
        if fa is not None:
            fh.write("Selective dynamics\n")
        fh.write("Direct\n")
        # Bug-fix note (found in code-check): the coordinate block MUST be
        # grouped by species to match the header's `species_order`/`counts`
        # lines -- that grouping is what lets a reader (parse_poscar, VASP,
        # or any other POSCAR consumer) recover which atom is which
        # species purely from position-in-file, since POSCAR carries no
        # per-atom species label. This function's `atom_types` parameter
        # has never been documented or enforced to already be pre-grouped
        # (species_order above is explicitly built to tolerate an
        # arbitrary/interleaved order), and it is NOT guaranteed to be
        # grouped by every caller: e.g. GCMCMDSimulation._accept_insertion
        # appends each newly-inserted atom's species to the END of
        # self.atom_types regardless of species, so a multi-species GCMC/
        # OSMMC run's self.atom_types becomes interleaved after the first
        # insertion of a species that is not already the group at the end
        # of the array -- and solvers/gcmc.py writes the final structure
        # via `write_poscar(..., self.atom_types)` with no fix_atoms=.
        # Previously this branch (fix_atoms=None) wrote `frac` in raw
        # input order while the `fix_atoms` branch below already
        # (correctly) regrouped by species -- for interleaved input, the
        # two branches therefore produced files that decode to DIFFERENT,
        # silently wrong species-to-position assignments on re-read (no
        # crash, no warning). Reproduced directly: writing
        # atom_types=['Li','Li','O','O','Li'] and reading the result back
        # gave atom_types=['Li','Li','Li','O','O'] with the SAME
        # (now mislabelled) positions. Fixed by always grouping by species
        # here too, exactly like the fix_atoms branch.
        for sp in species_order:
            for i, at in enumerate(atom_types):
                if at != sp:
                    continue
                f = frac[i]
                if fa is None:
                    fh.write(
                        f"  {f[0]:20.14f}  {f[1]:20.14f}  {f[2]:20.14f}\n")
                else:
                    flags = "  ".join("F" if fa[i, alpha] else "T"
                                       for alpha in range(3))
                    fh.write(
                        f"  {f[0]:20.14f}  {f[1]:20.14f}  {f[2]:20.14f}"
                        f"  {flags}\n")
