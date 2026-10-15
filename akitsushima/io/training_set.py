"""Training sets in the fit layout (``structures.csv`` + ``<idx>/POSCAR``),
the layout ``akitsushima dataset`` writes and ``fit`` / ``mlip`` read
(ADR-0024, Q23; moved here from cli/cmd_fit.py so the Python API has a
public loader, ADR-0023 gap G2)."""

import os
from typing import Dict

import numpy as np

from .poscar import parse_poscar

__all__ = ["load_training_set"]


def load_training_set(csv_path: str, poscar_dir: str = None, verbose: bool = False) -> list:
    """Structure dicts of a fit-layout training set.

    ``csv_path``: the ``structures.csv`` (columns target, label, weight,
    struct_idx); ``poscar_dir``: where ``<struct_idx>/POSCAR`` live
    (default: the CSV's directory).

    Each dict has ``name``, ``lattice`` (3, 3) [A], ``positions`` (N, 3)
    [A, Cartesian], ``atom_types``, ``natoms``, ``volume`` [A^3],
    ``fix_atoms``, ``ref_energy`` [eV/atom], ``ref_forces`` (N, 3) [eV/A]
    and ``ref_stress`` (3, 3) [eV/A^3, compressive-positive: a compressed
    cell has positive diagonal stress] -- the dicts ``fit_potential``,
    ``build_table`` and the MLIP models' ``fit`` / ``fit_structures`` take.
    """
    from ..fitting.models.csv_io import load_mlip_csv

    targets, labels, _, X = load_mlip_csv(csv_path, verbose=verbose)
    if poscar_dir is None:
        poscar_dir = os.path.dirname(os.path.abspath(csv_path))
    return _structures_from_csv(targets, labels, X, poscar_dir, verbose=verbose)


def _parse_geometry_layout_b(unique_ids: list, poscar_dir: str) -> Dict:
    """
    Layout B (only supported layout for classical force-field fitting).

    Loads geometry from POSCAR files located at::

        poscar_dir/<struct_idx>/POSCAR

    Parameters
    ----------
    unique_ids : sorted list of integer structure indices from the CSV.
    poscar_dir : root directory containing numbered sub-directories.

    Returns
    -------
    dict  {struct_id: structure_dict}

    Raises
    ------
    FileNotFoundError
        When ``poscar_dir/<sid>/POSCAR`` does not exist.
    """
    struct_map: Dict = {}
    for sid in unique_ids:
        poscar_path = os.path.join(poscar_dir, str(sid), "POSCAR")
        if not os.path.isfile(poscar_path):
            raise FileNotFoundError(
                f"POSCAR not found for structure {sid}: '{poscar_path}'\n"
                f"  Expected: {poscar_dir}/<struct_idx>/POSCAR\n"
                f"  Ensure poscar_dir is set correctly in the config "
                f"(default: directory of training_csv).")
        pd = parse_poscar(poscar_path)
        struct_map[sid] = {
            "name"       : str(sid),
            "lattice"    : pd["lattice"],
            "positions"  : pd["positions"],
            "atom_types" : pd["atom_types"],
            "natoms"     : pd["natoms"],
            "volume"     : float(abs(np.linalg.det(pd["lattice"]))),
            "fix_atoms"  : pd["fix_atoms"],
            "ref_energy" : None,
            "ref_forces" : None,
            "ref_stress" : None,
        }
    return struct_map


def _fill_efs_from_csv(struct_map    : Dict,
                        targets       : np.ndarray,
                        categories    : np.ndarray,
                        struct_idx_col: np.ndarray) -> list:
    """
    Fill ref_energy / ref_forces / ref_stress into the struct_map.

    Runs in O(M) time (M = total number of CSV rows) by pre-grouping
    rows per structure rather than rescanning the full array per row.

    Returns a list of structure dicts that have at least ref_energy set.
    """
    # Pre-group row indices by (struct_id, category) in one O(M) pass
    from collections import defaultdict
    grp: dict = defaultdict(list)   # (sid, cat) -> [row_indices]
    for row_idx, (sid, cat) in enumerate(zip(struct_idx_col, categories)):
        grp[(int(sid), str(cat))].append(row_idx)

    for sid, s in struct_map.items():
        # Energy (1 row expected)
        e_rows = grp.get((sid, 'E'), [])
        if e_rows:
            s['ref_energy'] = float(targets[e_rows[-1]])  # last wins if dup

        # Forces (3N rows expected)
        f_rows = grp.get((sid, 'F'), [])
        if f_rows:
            N      = s['natoms']
            f_vals = targets[np.array(f_rows, dtype=int)]
            if len(f_vals) == N * 3:
                s['ref_forces'] = f_vals.reshape(N, 3)

        # Stress (6 Voigt or 9 full rows expected)
        s_rows = grp.get((sid, 'S'), [])
        if s_rows:
            s_vals = targets[np.array(s_rows, dtype=int)]
            if len(s_vals) == 6:        # Voigt order: xx yy zz yz zx xy
                xx, yy, zz, yz, xz, xy = s_vals
                s['ref_stress'] = np.array(
                    [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            elif len(s_vals) == 9:
                s['ref_stress'] = s_vals.reshape(3, 3)

    return [s for s in struct_map.values() if s['ref_energy'] is not None]


def _structures_from_csv(targets    : np.ndarray,
                          categories : np.ndarray,
                          X_desc     : np.ndarray,
                          poscar_dir : str,
                          verbose    : bool = True) -> list:
    """
    Load classical force-field training structures using Layout B.

    **Layout B is the only supported CSV format** for classical force-field
    fitting.  The CSV must have exactly **one** descriptor column (col 2)
    containing an integer structure index.  Geometry is read from the
    corresponding POSCAR file::

        poscar_dir/<struct_idx>/POSCAR

    CSV format
    ----------
    ::

        col 0  target value  [eV/atom | eV/A | eV/A^3]   (float)
        col 1  category      "E" | "F" | "S"             (str)
        col 2  struct_idx    integer structure index      (int)

    Row ordering within one structure (N atoms):

    * 1 E row  -- total energy [eV/atom]; multiplied by N internally
    * 3N F rows -- force components, row-major (atom 1 x/y/z, atom 2 ...) [eV/A]
    * 6 S rows  -- Voigt stress: xx yy zz yz zx xy [eV/A^3],
      compressive-positive (a compressed cell has positive diagonal
      stress -- the engine's own convention, docs/adr/0001)

    If S rows are derived from a real VASP OUTCAR "in kB" line, apply a
    unit conversion and a REORDERING, but NO sign flip.  VASP's raw value
    is positive under compression, exactly like the engine's.  Multiply
    by ``akitsushima.constants.VASP_KB_TO_EV_ANG3`` (= +1/1602.176621)
    and reorder VASP's xx yy zz xy yz zx into xx yy zz yz zx xy.  (This
    docstring used to require a sign flip and point at a constant of
    -1/1602.176621; see that constant's bug-fix note.)

    Note that the CSV has FOUR columns, with a header row:
    ``target, label, weight, struct_idx``.  The ``weight`` column is
    read by :func:`load_mlip_csv` but not used by ``fit``: classical-
    potential residuals are normalised by the config's
    ``weights: {energy, forces, stress, r0, eps0}`` instead.

    Parameters
    ----------
    targets     : (M,) float  -- target values from CSV col 0.
    categories  : (M,) str    -- "E", "F", or "S" from CSV col 1.
                  (Parameter named ``labels`` in :func:`load_mlip_csv`
                  return value; both names refer to the same array.)
    X_desc      : (M, 1) float -- struct_idx column (col 2 of the CSV).
    poscar_dir  : root directory containing numbered POSCAR sub-directories.
    verbose     : print a summary line when True.

    Returns
    -------
    list of structure dicts with keys:
        name, lattice, positions, atom_types, natoms, volume,
        fix_atoms, ref_energy, ref_forces, ref_stress.

    Raises
    ------
    ValueError
        When the CSV has zero or more than one descriptor column ---
        i.e. Layout A data or an ELM descriptor CSV was passed.
    FileNotFoundError
        When a required POSCAR file is missing.
    RuntimeError
        When no structures with energy rows are found.
    """
    if X_desc.shape[1] == 0:
        raise ValueError(
            "_structures_from_csv: CSV has no descriptor column (col 2).\n"
            "  Layout B requires exactly one column: the integer structure index.")

    if X_desc.shape[1] > 1:
        raise ValueError(
            f"_structures_from_csv: CSV has {X_desc.shape[1]} descriptor "
            f"columns (cols 2+), but Layout B requires exactly 1 column "
            f"(the integer structure index).\n"
            f"  - If you passed an ELM descriptor CSV by mistake, use "
            f"ELMModel.fit() instead.\n"
            f"  - If you intended Layout B, keep only col 2 (struct_idx) "
            f"and place geometry in poscar_dir/<struct_idx>/POSCAR.")

    struct_idx_col = X_desc[:, 0].astype(int)
    unique_ids     = sorted(set(struct_idx_col))

    struct_map = _parse_geometry_layout_b(unique_ids, poscar_dir)
    result     = _fill_efs_from_csv(struct_map, targets, categories,
                                     struct_idx_col)

    if not result:
        raise RuntimeError(
            "_structures_from_csv: no structures with energy rows found.\n"
            "  Check that at least one row per structure has category 'E'.")

    if verbose:
        n_f = sum(1 for s in result if s["ref_forces"] is not None)
        n_s = sum(1 for s in result if s["ref_stress"] is not None)
        print(f"  Loaded {len(result)} structures from Layout B CSV  "
              f"(forces: {n_f}  stress: {n_s})")
    return result
