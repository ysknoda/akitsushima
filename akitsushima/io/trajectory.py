"""Trajectory output in three formats (ADR-0034 decisions 3, 4, 8, 10).

``traj_format``:

* ``xdatcar``     -- ``XDATCAR_<task>`` (the default, unchanged output);
* ``extxyz``      -- ``traj_<task>.extxyz``: lattice, positions, species,
  energy, forces and stress (ASE's sign) per frame;
* ``lammps-dump`` -- ``traj_<task>.lammpstrj``: ``id type element x y z
  fx fy fz``, metal units, in the LAMMPS orientation.

Positions and forces are written with 8 decimals, the energy with 10 and
the stress with 10 significant digits.  Every format is append-only, so the
restart of ADR-0031 (truncate to the recorded size, append) applies.
"""

from __future__ import annotations

import os

from .structure import _sort_species, extxyz_frame_text, lammps_dump_frame_text
from .xdatcar import _write_xdatcar_frame, _write_xdatcar_header

__all__ = ["TRAJ_FORMATS", "TrajectoryWriter", "check_traj_format", "traj_filename"]

TRAJ_FORMATS = ("xdatcar", "extxyz", "lammps-dump")


def check_traj_format(fmt) -> str:
    fmt = "xdatcar" if fmt is None else str(fmt).lower()
    if fmt not in TRAJ_FORMATS:
        raise ValueError(
            f"traj_format must be one of {list(TRAJ_FORMATS)}, got '{fmt}'."
        )
    return fmt


def traj_filename(task: str, fmt: str) -> str:
    """``XDATCAR_<task>``, ``traj_<task>.extxyz`` or ``traj_<task>.lammpstrj``."""
    fmt = check_traj_format(fmt)
    if fmt == "xdatcar":
        return f"XDATCAR_{task}"
    return f"traj_{task}." + ("extxyz" if fmt == "extxyz" else "lammpstrj")


class TrajectoryWriter:
    """
    Writes frames to an open file handle in one of ``TRAJ_FORMATS``.

    variable_cell : xdatcar only -- a full header before every frame
                    (NPT, or a changing atom count); otherwise one header
                    from :meth:`header`.
    species       : lammps-dump only -- the element of each atom type;
                    default the species of the first frame by atomic
                    number, extended when a new one appears (gcmc).
    """

    def __init__(self, fh, fmt: str, variable_cell: bool = False, species=None):
        self.fh = fh
        self.fmt = check_traj_format(fmt)
        self.variable_cell = bool(variable_cell)
        self.species = list(species) if species is not None else None
        self.frame_idx = 1  # xdatcar "Direct configuration=" numbering

    def header(self, lattice, atom_types, comment: str) -> None:
        """Once at the start of a fresh file: the XDATCAR header of a fixed
        cell.  The other formats need none."""
        if self.fmt == "xdatcar" and not self.variable_cell:
            _write_xdatcar_header(self.fh, lattice, atom_types, comment=comment)

    def frame(
        self,
        step,
        lattice,
        positions,
        atom_types,
        energy=None,
        forces=None,
        stress=None,
        comment: str = "",
    ) -> None:
        """Append one frame.  ``stress`` in akitsushima's sign [eV/A^3]."""
        if self.fmt == "xdatcar":
            _write_xdatcar_frame(
                self.fh,
                positions,
                lattice,
                atom_types,
                self.frame_idx,
                variable_header=self.variable_cell,
                comment=comment or "akitsushima trajectory",
            )
        elif self.fmt == "extxyz":
            self.fh.write(
                extxyz_frame_text(
                    lattice,
                    positions,
                    list(atom_types),
                    energy,
                    forces,
                    stress,
                    info={"step": int(step)},
                    decimals=8,
                    energy_decimals=10,
                    stress_digits=10,
                )
            )
        else:
            if self.species is None:
                self.species = _sort_species(atom_types)
            new = [s for s in _sort_species(atom_types) if s not in self.species]
            self.species += new
            self.fh.write(
                lammps_dump_frame_text(
                    step,
                    lattice,
                    positions,
                    list(atom_types),
                    self.species,
                    forces=forces,
                    decimals=8,
                )
            )
        self.frame_idx += 1


def traj_path(output_dir: str, task: str, fmt: str) -> str:
    return os.path.join(output_dir, traj_filename(task, fmt))
