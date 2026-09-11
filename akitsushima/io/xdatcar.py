"""VASP XDATCAR trajectory file writer."""
from typing import Tuple

import numpy as np


def _xdatcar_species_order_counts(atom_types: list) -> Tuple[list, list]:
    """
    Return (species_order, counts) for a VASP-style species header,
    using first-appearance order and total occurrence count -- the same
    convention already used by write_poscar(). This assumes atom_types
    is grouped by species (all atoms of one species contiguous), which
    is the standard VASP POSCAR/XDATCAR convention.
    """
    species_order = []
    seen = set()
    for at in atom_types:
        if at not in seen:
            species_order.append(at)
            seen.add(at)
    counts = [atom_types.count(sp) for sp in species_order]
    return species_order, counts


def _write_xdatcar_header(fh, lattice: np.ndarray, atom_types: list,
                           comment: str = "akitsushima MD trajectory") -> None:
    """Write one VASP XDATCAR header block: comment, scale, 3 lattice
    vectors, species names, species counts."""
    species_order, counts = _xdatcar_species_order_counts(atom_types)
    fh.write(f"{comment}\n")
    fh.write("1.0\n")
    for vec in lattice:
        fh.write(f"  {vec[0]:20.14f}  {vec[1]:20.14f}  {vec[2]:20.14f}\n")
    fh.write("  " + "  ".join(species_order) + "\n")
    fh.write("  " + "  ".join(str(c) for c in counts) + "\n")


def _write_xdatcar_frame(fh, positions: np.ndarray, lattice: np.ndarray,
                          atom_types: list, frame_idx: int,
                          variable_header: bool = False,
                          comment: str = "akitsushima MD trajectory") -> None:
    """
    Append one atomic configuration to an already-open XDATCAR file.

    Parameters
    ----------
    variable_header : if True, a FULL header (comment, scale, lattice,
        species, counts) is written immediately before this frame's
        coordinates.  This is required whenever the cell (NPT) or the
        atom count/composition (GCMC, OS-MC) can change from one frame
        to the next, since standard VASP XDATCAR format assumes a
        single fixed header applies to the whole file.  If False
        (the default, used for NVT/NVE), only the
        "Direct configuration=" line and fractional coordinates are
        written; the header must already have been written once, before
        the first call, via _write_xdatcar_header().
    frame_idx : configuration index (1-based, matching VASP's own
        "Direct configuration=" numbering convention).
    """
    if variable_header:
        _write_xdatcar_header(fh, lattice, atom_types, comment)
    inv_lat = np.linalg.inv(lattice)
    frac = positions @ inv_lat
    fh.write(f"Direct configuration={frame_idx:6d}\n")
    # Bug-fix note (found in code-check, same class as the write_poscar
    # fix): the header (species_order/counts, above) assumes atom_types
    # is grouped by species -- XDATCAR/POSCAR carry no per-atom species
    # label, so a reader recovers species purely from position-in-file.
    # This loop used to write `frac` in raw input order regardless, which
    # is silently wrong whenever atom_types is NOT already grouped --
    # exactly what happens for GCMCMDSimulation/OSMMCSimulation, whose
    # `_accept_insertion` appends each newly-inserted atom's species to
    # the END of self.atom_types regardless of species (both call this
    # function with variable_header=True every frame, so the header is
    # always fresh and correct, but the coordinates beneath it were not
    # reordered to match). Fixed by writing coordinates in the same
    # species-grouped order as the header, exactly like write_poscar.
    species_order, _ = _xdatcar_species_order_counts(atom_types)
    for sp in species_order:
        for i, at in enumerate(atom_types):
            if at != sp:
                continue
            p = frac[i]
            fh.write(f"  {p[0]:12.8f}  {p[1]:12.8f}  {p[2]:12.8f}\n")
