"""CSV loaders shared by every ML-potential model (ELM, QuantizedELM, NN,
LightGBM): the common 4-column training-data format, the optional
block-wise user-weight file, and the structure-grouped train/val split
that every model's ``fit()`` uses to avoid leaking a structure into both
splits (see :func:`structure_grouped_split`).
"""
import csv as _csv
import warnings
from typing import Sequence, Tuple

import numpy as np

__all__ = ["load_mlip_csv", "load_weights_csv",
           "infer_structure_ids", "structure_grouped_split"]


def load_mlip_csv(csv_path: str,
                 verbose  : bool = True,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ML-potential training data from a CSV file.

    CSV format
    ----------
    ::

        Row 0         : header  (any column labels --- required, always skipped)
        Row 1+, col 0 : target  (float)
        Row 1+, col 1 : label   (str)    "E" | "F" | "S"
        Row 1+, col 2 : weight  (float)  eV-unit normalisation weight
        Row 1+, col 3+: descriptor features (float, D columns)

    Column definitions
    ------------------
    **col 0 --- target**
        Raw physical quantity:

        * ``E`` rows : total energy          [eV/atom]
        * ``F`` rows : force component        [eV/A]
        * ``S`` rows : stress component       [eV/A^3]  (Voigt: xx yy zz yz zx xy,
          compressive-positive -- a compressed cell has positive
          diagonal stress).  From a real VASP OUTCAR "in kB" line:
          multiply by ``akitsushima.constants.VASP_KB_TO_EV_ANG3``
          (= +1/1602.176621, no sign flip) AND reorder, since VASP writes
          xx yy zz xy yz zx.

    **col 1 --- label**
        ``"E"``, ``"F"``, or ``"S"`` (case-insensitive).

    **col 2 --- weight**
        Scaling coefficient that converts each squared residual to a
        common eV^2 unit so that E, F, and S terms are comparable in
        the loss function.  Correct values that achieve eV^2 units:

        * **E row** : ``N``  (number of atoms in the cell)
          ``(y [eV/atom] * N)^2 = (delta_E [eV])^2``

        * **F row** : ``r0``  (typical atomic displacement [A])
          ``(y [eV/A] * r0)^2 = (delta_F * r0 [eV])^2``
          Example: r0 = 0.1 A  ->  set ``weight = 0.1`` for all F rows.

        * **S row** : ``eps0 * V``  (typical strain * cell volume [A^3])
          ``(y [eV/A^3] * eps0 * V)^2 = (delta_sigma * eps0 * V [eV])^2``
          Example: eps0 = 0.01, V = 64.5 A^3  ->  ``weight = 0.645``.

        Any positive value is accepted (weight = 0 excludes the row).

    **col 3+ --- descriptor features**
        Pre-computed descriptor vector of dimension *D*.
        All rows must have the same *D*.

    Row ordering within one structure (N atoms)
    --------------------------------------------
    ::

        1 E row           total energy
        3N F rows         forces: atom 0 x/y/z, atom 1 x/y/z, ...
        6 S rows          Voigt stress: xx yy zz yz zx xy

    F rows and S rows are optional.
    Blank lines and lines starting with ``#`` are ignored.

    Parameters
    ----------
    csv_path : str   Path to the CSV file.
    verbose  : bool  Print a summary table when True.

    Returns
    -------
    targets : (M,)   float64 -- target values              (col 0)
    labels  : (M,)   str     -- "E" / "F" / "S"           (col 1)
    weights : (M,)   float64 -- per-sample weights         (col 2)
    X       : (M, D) float64 -- descriptor feature matrix  (col 3+)

    Raises
    ------
    ValueError
        Fewer than 4 columns on any data row; unknown label; negative
        weight; inconsistent descriptor width across rows.
    RuntimeError
        File contains no data rows.
    """
    rows_y   : list = []
    rows_lbl : list = []
    rows_w   : list = []
    rows_X   : list = []

    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = _csv.reader(fh)
        next(reader)        # skip header row

        for lineno, row in enumerate(reader, start=2):
            if not row or row[0].strip().startswith('#'):
                continue    # blank / comment

            if len(row) < 4:
                raise ValueError(
                    f"load_mlip_csv: line {lineno} has {len(row)} column(s); "
                    f"expected >= 4  (target, label, weight, >=1 descriptor).\n"
                    f"  row = {row}\n"
                    f"  The CSV format requires:\n"
                    f"    col 0 : target value (float)\n"
                    f"    col 1 : label  'E' | 'F' | 'S'\n"
                    f"    col 2 : weight (float >= 0)\n"
                    f"    col 3+: descriptor features (float)")

            try:
                y_val = float(row[0])
                lbl   = row[1].strip().upper()
                w_val = float(row[2])
                feats = [float(v) for v in row[3:]]
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"load_mlip_csv: parse error at line {lineno}: {exc}\n"
                    f"  row = {row[:7]}{'...' if len(row) > 7 else ''}") from exc

            if lbl not in ('E', 'F', 'S'):
                raise ValueError(
                    f"load_mlip_csv: unknown label '{lbl}' at line {lineno}. "
                    f"Expected 'E', 'F', or 'S'.")

            if w_val < 0.0:
                raise ValueError(
                    f"load_mlip_csv: negative weight {w_val} at line {lineno}. "
                    f"Weights must be >= 0.")

            if len(feats) == 0:
                raise ValueError(
                    f"load_mlip_csv: no descriptor features at line {lineno}. "
                    f"col 3+ must contain at least one float value.")

            rows_y.append(y_val)
            rows_lbl.append(lbl)
            rows_w.append(w_val)
            rows_X.append(feats)

    if not rows_X:
        raise RuntimeError(f"load_mlip_csv: no data rows found in '{csv_path}'.")

    D_first = len(rows_X[0])
    bad = [(i + 2, len(r)) for i, r in enumerate(rows_X) if len(r) != D_first]
    if bad:
        raise ValueError(
            f"load_mlip_csv: inconsistent descriptor width. "
            f"Expected D={D_first} (from row 2); "
            f"rows {bad[:5]} have different widths.")

    targets = np.array(rows_y,   dtype=np.float64)
    labels  = np.array(rows_lbl, dtype=object)
    weights = np.array(rows_w,   dtype=np.float64)
    X       = np.array(rows_X,   dtype=np.float64)

    if verbose:
        counts = {c: int(np.sum(labels == c)) for c in ('E', 'F', 'S')}
        print(f"  Loaded '{csv_path}':")
        print(f"    Rows       : {len(targets)}  "
              f"(E={counts['E']}  F={counts['F']}  S={counts['S']})")
        print(f"    Descriptor : D = {X.shape[1]}")
        for c in ('E', 'F', 'S'):
            m = labels == c
            if not m.any():
                continue
            t = targets[m]; w = weights[m]
            print(f"    [{c}]  target  mean={t.mean():.4g}  "
                  f"std={t.std():.4g}  "
                  f"range=[{t.min():.4g}, {t.max():.4g}]")
            print(f"         weight  mean={w.mean():.4g}  "
                  f"range=[{w.min():.4g}, {w.max():.4g}]")

    return targets, labels, weights, X


def load_weights_csv(path    : str,
                     n_total : int,
                     verbose : bool = True,
                     ) -> np.ndarray:
    """
    Load user-defined per-sample weights from a weights.csv file.

    This is the **third** of three weight factors combined in
    :meth:`~akitsushima.fitting.models.elm.ELMModel.fit`.  It lets the
    user manually adjust the relative learning emphasis of contiguous
    blocks of samples, independently of the unit-normalisation weights
    (col 2 of the training CSV) and the Uncertainty Weighting terms.

    File format
    -----------
    ::

        #samples, weights          <- header line (required, any text)
        10000     1.00             <- first  10 000 samples -> weight 1.00
        50000     0.01             <- next   50 000 samples -> weight 0.01
        20000     0.10             <- next   20 000 samples -> weight 0.10

    * **Col 0** : number of samples in this block (positive integer).
    * **Col 1** : weight coefficient (float, >= 0).
    * Separator : any whitespace or comma.
    * Lines beginning with ``#`` and blank lines are ignored.

    The blocks are applied sequentially from the top of the training
    dataset.  The sum of all block sizes must equal ``n_total``
    (the total number of training samples); otherwise a ``ValueError``
    is raised.

    Parameters
    ----------
    path    : str   Path to the weights.csv file.
    n_total : int   Expected total number of samples (= len(training data)).
                    Used to validate that the block sizes sum correctly.
    verbose : bool  Print a summary table when True.

    Returns
    -------
    w_user : (n_total,) float64 ndarray  -- per-sample user weights.

    Raises
    ------
    FileNotFoundError
        When ``path`` does not exist.
    ValueError
        When a block size is not a positive integer, a weight is
        negative, the number of columns is wrong, or the sum of block
        sizes does not equal ``n_total``.
    """
    blocks: list = []   # list of (n_samples: int, weight: float)

    with open(path, newline='', encoding='utf-8') as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue    # skip header / blank / comment lines

            # Accept comma or whitespace as delimiter
            parts = line.replace(',', ' ').split()
            if len(parts) < 2:
                raise ValueError(
                    f"load_weights_csv: line {lineno} has only "
                    f"{len(parts)} column(s); expected 2 "
                    f"(n_samples, weight).\n"
                    f"  line = {repr(raw_line.rstrip())}")

            try:
                n_blk = int(parts[0])
            except ValueError:
                raise ValueError(
                    f"load_weights_csv: cannot parse n_samples at "
                    f"line {lineno}: {repr(parts[0])!r} is not an integer.")

            try:
                w_blk = float(parts[1])
            except ValueError:
                raise ValueError(
                    f"load_weights_csv: cannot parse weight at "
                    f"line {lineno}: {repr(parts[1])!r} is not a float.")

            if n_blk <= 0:
                raise ValueError(
                    f"load_weights_csv: n_samples at line {lineno} is "
                    f"{n_blk}; must be a positive integer.")

            if w_blk < 0.0:
                raise ValueError(
                    f"load_weights_csv: weight at line {lineno} is "
                    f"{w_blk}; must be >= 0.")

            blocks.append((n_blk, w_blk))

    if not blocks:
        raise ValueError(
            f"load_weights_csv: no data rows found in '{path}'.")

    # Validate total sample count
    total_blk = sum(n for n, _ in blocks)
    if total_blk != n_total:
        raise ValueError(
            f"load_weights_csv: sum of block sizes = {total_blk}, "
            f"but n_total = {n_total}.\n"
            f"  The block sizes in '{path}' must sum to exactly "
            f"the total number of training samples ({n_total}).\n"
            f"  Blocks: {[(n, w) for n, w in blocks]}")

    # Build weight array
    w_user = np.empty(n_total, dtype=np.float64)
    offset = 0
    for n_blk, w_blk in blocks:
        w_user[offset:offset + n_blk] = w_blk
        offset += n_blk

    if verbose:
        print(f"  User weights <- '{path}':")
        print(f"    Total samples : {n_total}")
        print(f"    Blocks        : {len(blocks)}")
        for i, (n_blk, w_blk) in enumerate(blocks):
            print(f"      block {i+1:3d}  : n={n_blk:>8d}  weight={w_blk:.6g}")
        print(f"    weight range  : [{w_user.min():.4g}, {w_user.max():.4g}]")

    return w_user


# ===========================================================================
# Structure-grouped train/validation split
#
# Bug-fix note (F17): every model's fit() -- ELMModel, QuantizedELMModel
# (which inherits ELMModel.fit), NNModel, LightGBMModel -- used to split
# train/validation ROW-wise:
#
#     rng_split = np.random.default_rng(val_seed)
#     idx_all   = rng_split.permutation(N)      # N = number of CSV ROWS
#     idx_val   = idx_all[:n_val]
#     idx_train = idx_all[n_val:]
#
# That is a textbook group-leakage bug for this data format.  As
# load_mlip_csv documents, ONE structure of N atoms contributes
# 1 E row + 3N F rows + 6 S rows, and every one of those rows is derived
# from the SAME atomic configuration: the F and S rows are just spatial
# derivatives of the very energy the E row records, evaluated at the same
# descriptor neighbourhood.  They are not independent samples.
#
# Consequently a row-wise permutation puts rows from essentially EVERY
# structure on BOTH sides of the split.  Measured on a synthetic 8-structure
# set (128 rows, val_fraction=0.2): 8/8 structures appeared in both the
# training and the validation split at val_seed=0 and 1, and 7/8 at
# val_seed=2.  The reported "validation" error therefore measured
# interpolation within already-seen configurations, not generalisation to
# unseen ones, and was optimistically biased for every model in the package.
#
# The consequence was worst for LightGBMModel.fit, whose Optuna search
# *minimises* that leaked validation RMSE: hyperparameter selection was
# being steered by a corrupted signal, and both NNModel's early stopping
# and LightGBM's early stopping used it as their stopping criterion.
#
# The fix is to assign whole STRUCTURES to one side of the split.  This
# changes the meaning of val_fraction from "fraction of rows" to "fraction
# of structures" -- a deliberate semantic change, documented in every
# affected fit()/fit_csv() docstring.
# ===========================================================================

def infer_structure_ids(labels: Sequence) -> np.ndarray:
    """
    Map every CSV row to the index of the structure it belongs to.

    The training CSV has **no structure-ID column** (its columns are
    ``y, label, weight, descriptors...``), so structure boundaries have to
    be inferred from the label sequence.  Per the row layout documented in
    :func:`load_mlip_csv`, one structure of *N* atoms contributes

    ::

        1 E row           total energy
        3N F rows         forces: atom 0 x/y/z, atom 1 x/y/z, ...
        6 S rows          Voigt stress: xx yy zz yz zx xy

    with the F and S blocks both optional.  The ``E`` row is therefore the
    only reliable delimiter: **each ``E`` row starts a new structure**, and
    every following non-``E`` row belongs to that structure until the next
    ``E`` row.

    Parameters
    ----------
    labels : (M,) sequence of str
        The per-row ``'E'``/``'F'``/``'S'`` labels, i.e. the second return
        value of :func:`load_mlip_csv`.  Comparison is case-insensitive and
        whitespace-tolerant, matching load_mlip_csv's own normalisation.

    Returns
    -------
    structure_ids : (M,) int64 ndarray
        ``structure_ids[i]`` is the 0-based index of the structure that row
        *i* belongs to.  Indices are contiguous: ``0 .. n_structures - 1``.
        For an empty input an empty array is returned.

    Edge-case behaviour (all deliberate, none of them raise)
    --------------------------------------------------------
    * **Structures with no F and/or no S rows** (``E``-only, ``E``+``F``,
      ``E``+``S``, ``E``+``F``+``S``) are all handled identically: a new
      structure simply starts at each ``E`` row, whatever follows it.
    * **Rows before the first ``E`` row** cannot be attributed to any
      structure.  They are grouped together into a single leading
      pseudo-structure (index 0) and a ``UserWarning`` is issued.  Grouping
      them rather than dropping or scattering them is the conservative
      choice: as one indivisible group they can never straddle the split.
    * **A file with no ``E`` rows at all** (e.g. a force-only CSV) carries
      no structure-boundary information whatsoever.  All rows are then
      assigned to a single structure (index 0) and a ``UserWarning`` says
      so.  A grouped split of such a file can hold nothing out -- see
      :func:`structure_grouped_split`, which then trains on everything and
      warns rather than silently falling back to a leaky row-wise split.
      The remedy is to include the delimiting ``E`` rows (a ``weight`` of
      0 excludes an ``E`` row from the loss while still delimiting its
      structure).

    Examples
    --------
    >>> infer_structure_ids(['E', 'F', 'F', 'F', 'E', 'F', 'F', 'F', 'S'])
    array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    """
    labels_arr = np.asarray(labels, dtype=object).ravel()
    n_rows = labels_arr.shape[0]
    if n_rows == 0:
        return np.empty(0, dtype=np.int64)

    is_energy_row = np.fromiter(
        (str(lbl).strip().upper() == 'E' for lbl in labels_arr),
        dtype=bool, count=n_rows)

    # cumsum of the E-row indicator, minus one, numbers the structures from
    # 0; any row preceding the first E row gets -1 (no structure yet).
    structure_ids = np.cumsum(is_energy_row) - 1

    n_orphan_rows = int(np.count_nonzero(structure_ids < 0))
    if n_orphan_rows:
        if is_energy_row.any():
            warnings.warn(
                f"infer_structure_ids: {n_orphan_rows} row(s) appear before "
                f"the first 'E' row and cannot be attributed to a structure. "
                f"They are grouped into one leading pseudo-structure so that "
                f"they stay on a single side of any structure-grouped "
                f"train/validation split.  Per the documented CSV layout "
                f"each structure must begin with its 'E' row.",
                UserWarning, stacklevel=2)
        else:
            warnings.warn(
                f"infer_structure_ids: the dataset contains no 'E' rows, so "
                f"structure boundaries cannot be inferred; all {n_rows} rows "
                f"are treated as ONE structure.  A structure-grouped "
                f"train/validation split can then hold nothing out.  Include "
                f"the delimiting 'E' row of each structure (weight = 0 keeps "
                f"it out of the loss while still delimiting the structure) "
                f"to get a meaningful validation split.",
                UserWarning, stacklevel=2)
        # Shift so the leading orphan block becomes structure 0 and the
        # real structures follow contiguously (1, 2, ...).
        structure_ids = structure_ids + 1

    return structure_ids.astype(np.int64)


def structure_grouped_split(labels       : Sequence,
                            val_fraction : float = 0.2,
                            val_seed     : int   = 0,
                            caller       : str   = "structure_grouped_split",
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split CSV rows into training and validation sets **by structure**.

    Whole structures are assigned to one side of the split, so no structure
    ever contributes rows to both.  This is the leak-free replacement for
    the row-wise ``rng.permutation(n_rows)`` split every model used to do
    (see the F17 bug-fix note above this function).

    Semantics of ``val_fraction`` (changed by the F17 fix)
    ------------------------------------------------------
    ``val_fraction`` is the fraction of **structures** held out, not the
    fraction of rows.  Because different structures have different atom
    counts (and therefore different row counts), the resulting *row*
    fraction is only approximately ``val_fraction``.  That is the intended
    behaviour: the split unit has to be the structure for the validation
    score to mean anything.

    Parameters
    ----------
    labels       : (M,) sequence of str
        Per-row ``'E'``/``'F'``/``'S'`` labels; see
        :func:`infer_structure_ids`.
    val_fraction : float
        Fraction of structures held out for validation.  ``<= 0`` disables
        the validation split entirely (all rows go to training).
    val_seed     : int
        Seed for the structure-permutation RNG; the split is fully
        reproducible for a given ``(labels, val_fraction, val_seed)``.
    caller       : str
        Name used in warning messages (e.g. ``'ELMModel.fit'``) so a
        warning points at the model the user actually called.

    Returns
    -------
    idx_train : (M_train,) int64 ndarray -- row indices of the training set.
    idx_val   : (M_val,)   int64 ndarray -- row indices of the validation set.

    Both index arrays are sorted ascending (the original file order is
    preserved within each split).  Nothing downstream depends on the rows
    being shuffled: ELM solves one LSQR system over all training rows, NN
    re-shuffles minibatches every epoch with ``torch.randperm``, and
    LightGBM is order-independent.  Keeping file order makes a split
    reproducible and inspectable.

    Edge cases
    ----------
    * ``val_fraction <= 0``  -> all rows train, empty validation set.
    * **Fewer than 2 structures** (a single-structure file, or a file with
      no ``E`` rows at all, which :func:`infer_structure_ids` collapses to
      one structure): a grouped split cannot hold a structure out without
      emptying the training set, so all rows are used for training, the
      validation set is empty, and a ``UserWarning`` explains why.  There
      is deliberately **no** fall-back to a row-wise split here: that is
      exactly the leaking behaviour F17 removed, and a "validation" score
      computed from sibling rows of the same configuration would measure
      memorisation rather than generalisation.
    * ``val_fraction`` large enough to claim every structure -> same
      warn-and-train-on-everything behaviour, mirroring the pre-F17
      ``n_val >= N`` guard.
    * A ``val_fraction`` too small to round up to one structure still
      yields one validation structure (``max(1, ...)``), matching the
      convention of :func:`akitsushima.fitting.driver.split_train_val`.
    """
    structure_ids = infer_structure_ids(labels)
    n_rows = int(structure_ids.size)
    all_rows = np.arange(n_rows, dtype=np.int64)
    no_rows  = np.empty(0, dtype=np.int64)

    if n_rows == 0:
        return no_rows, no_rows

    if val_fraction <= 0.0:
        return all_rows, no_rows

    n_structures = int(structure_ids.max()) + 1

    if n_structures < 2:
        warnings.warn(
            f"{caller}: the dataset contains only {n_structures} structure(s) "
            f"({n_rows} rows), so a structure-grouped train/validation split "
            f"cannot hold anything out without emptying the training set. "
            f"Using all rows for training and NO validation set.  A row-wise "
            f"split is deliberately not used as a fall-back: every row of one "
            f"structure (1 E + 3N F + 6 S) derives from the same atomic "
            f"configuration, so such a 'validation' score would measure "
            f"memorisation, not generalisation.  Provide at least two "
            f"structures to get a meaningful validation score.",
            UserWarning, stacklevel=2)
        return all_rows, no_rows

    n_val_structures = max(1, int(round(n_structures * val_fraction)))

    if n_val_structures >= n_structures:
        warnings.warn(
            f"{caller}: val_fraction={val_fraction} would hold out all "
            f"{n_structures} structures, leaving no training data; using all "
            f"{n_rows} rows for training and no validation set.",
            UserWarning, stacklevel=2)
        return all_rows, no_rows

    rng = np.random.default_rng(val_seed)
    val_structures = rng.permutation(n_structures)[:n_val_structures]
    is_val_row = np.isin(structure_ids, val_structures)

    return all_rows[~is_val_row], all_rows[is_val_row]
