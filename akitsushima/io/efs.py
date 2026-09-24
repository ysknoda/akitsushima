"""Plain-text energy/force/stress (EFS) file format.

Used to exchange results with external codes (VASP, QE, LAMMPS, ...) via
:class:`~akitsushima.fitting.external.ExternalFileEngine`.
"""
import os
from typing import List, Optional, Tuple

import numpy as np


def write_efs_file(path: str,
                    energy: float,
                    forces: np.ndarray,
                    stress: np.ndarray,
                    n_atoms: Optional[int] = None) -> None:
    """
    Write energy, forces, and stress in the akitsushima EFS plain-text format.

    Column order
    ------------
    ``<value>  <category>  <label>``

    The category and label columns are human-readable annotations only.
    The parser reads values **positionally** and does not rely on them.

    Energy unit
    -----------
    Energy is written as **eV/atom** (total energy divided by N).
    ``read_efs_file`` multiplies it back by N to recover eV.

    Parameters
    ----------
    path    : destination file path.
    energy  : total energy [eV].  Written as eV/atom (divided by N).
    forces  : (N, 3) force array [eV/A].
    stress  : (3, 3) stress tensor [eV/A^3] (symmetric; Voigt order).
    n_atoms : number of atoms N.  Inferred from ``forces`` when None.
    """
    N = n_atoms if n_atoms is not None else len(forces)
    alpha = ['x', 'y', 'z']
    voigt = [('xx', 0, 0), ('yy', 1, 1), ('zz', 2, 2),
             ('yz', 1, 2), ('zx', 2, 0), ('xy', 0, 1)]
    e_per_atom = energy / N if N > 0 else energy
    with open(path, 'w') as fh:
        # Line 1: energy per atom
        fh.write(f"{e_per_atom:.10f}  E  0\n")
        # Lines 2 .. 3N+1: forces  (atom 1 x/y/z, atom 2 x/y/z, ...)
        for i in range(N):
            for ai, a in enumerate(alpha):
                fh.write(f"{forces[i, ai]:.10f}  F  {i + 1}{a}\n")
        # Lines 3N+2 .. 3N+7: stress Voigt components
        for label, r, c in voigt:
            fh.write(f"{stress[r, c]:.10f}  S  {label}\n")


def read_efs_file(path: str,
                   n_atoms: int,
                   stress_to_eV_Ang3: float = 1.0,
                   ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Parse an EFS plain-text file and return (energy [eV], forces, stress).

    Parsing strategy -- **positional, category/label columns are ignored**
    -----------------------------------------------------------------------
    After stripping blank lines and ``#``-comment lines, data lines are
    counted in the order they appear:

    * Line  1            : energy per atom [eV/atom]  -> multiplied by N
    * Lines 2 .. 3N+1   : force components, row-major over atoms then axes
                           (atom 1 x, atom 1 y, atom 1 z,
                            atom 2 x, ..., atom N z)    [eV/A]
    * Lines 3N+2..3N+7  : stress Voigt components in the order
                           xx, yy, zz, yz, zx, xy     [eV/A^3]

    The second and third columns (category "E"/"F"/"S" and label) are
    read and stored for diagnostic messages but do **not** control where
    each value is placed.  This makes the parser robust to any labelling
    convention the external code uses.

    Parameters
    ----------
    path              : path to the EFS output file.
    n_atoms           : number of atoms N.
    stress_to_eV_Ang3 : unit conversion factor applied to all stress values.
                        Common values:

                        ==============================  ====================
                        Source unit                     Factor
                        ==============================  ====================
                        eV/A^3 (no conversion)           1.0
                        VASP kB (raw OUTCAR "in kB")     +1 / 1602.176621
                        QE Ry/bohr^3                     +91.8157677
                        ==============================  ====================

                        No sign flip for either: VASP's and QE's raw stress
                        are both positive under compression, which is this
                        codebase's convention too (see
                        ``akitsushima.constants.VASP_KB_TO_EV_ANG3`` for the
                        derivation, checked against ASE's production
                        readers).  Pass that constant rather than
                        re-deriving the VASP factor by hand.

                        Bug-fix notes: the VASP entry was first
                        ``1/160.2176621`` (the eV/A^3->GPa factor, 10x off),
                        then ``-1/1602.176621`` (right magnitude, wrong
                        sign).  The QE entry read ``14.7103``; 1 Ry/bohr^3
                        is 91.8157677 eV/A^3 (ase.units.Ry/ase.units.Bohr**3),
                        so that was 6.24x too small.

    Returns
    -------
    energy : float [eV]  (= value_in_file x N)
    forces : (N, 3) array [eV/A]
    stress : (3, 3) symmetric array [eV/A^3]

    Raises
    ------
    FileNotFoundError : if *path* does not exist.
    ValueError        : if the file contains fewer data lines than required
                        (1 + 3N + 6) or if a value cannot be parsed as float.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"read_efs_file: output file not found: '{path}'\n"
            "  Ensure the external command ran successfully and wrote this file.")

    # Collect all non-blank, non-comment lines
    data_lines: List[Tuple[int, str, float]] = []   # (lineno, raw, value)
    with open(path, 'r') as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # First token is the numeric value in the new format.
            # We try to parse parts[0] as float; if it fails (e.g. the
            # file uses the old format where category came first) we try
            # to locate the float among the tokens to give a clear error.
            try:
                val = float(parts[0])
            except (ValueError, IndexError):
                raise ValueError(
                    f"read_efs_file: cannot parse numeric value from first "
                    f"token '{parts[0] if parts else ''}' at line {lineno}: "
                    f"'{raw.rstrip()}'\n"
                    "  Expected format: <value> <category> <label>")
            data_lines.append((lineno, raw.rstrip(), val))

    # Check total line count
    n_required = 1 + 3 * n_atoms + 6
    n_found = len(data_lines)
    if n_found < n_required:
        raise ValueError(
            f"read_efs_file: expected {n_required} data lines "
            f"(1 energy + {3 * n_atoms} forces + 6 stress) "
            f"for N={n_atoms} atoms, but found {n_found} in '{path}'.")

    # -- Line 0: energy per atom -> total energy -----------
    energy = data_lines[0][2] * n_atoms   # eV/atom x N = eV

    # -- Lines 1 .. 3N: forces in row-major (atom, axis) order ----
    forces = np.zeros((n_atoms, 3))
    idx = 1
    for i in range(n_atoms):
        for ai in range(3):
            forces[i, ai] = data_lines[idx][2]
            idx += 1

    # -- Lines 3N+1 .. 3N+6: stress Voigt components -------
    voigt_rc = [(0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1)]
    stress = np.zeros((3, 3))
    for r, c in voigt_rc:
        v = data_lines[idx][2] * stress_to_eV_Ang3
        stress[r, c] = v
        stress[c, r] = v   # symmetrise
        idx += 1

    return float(energy), forces, stress
