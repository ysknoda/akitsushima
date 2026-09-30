"""Unified fitting driver: train/val split, scoring, and the fit_potential
optimisation loop shared by every potential type (LJ, Morse, BornMayer,
Buckingham, Stillinger-Weber, Tersoff).

Consolidation note
-------------------
The original script kept a second, Buckingham-only copy of this whole
stack (``buckingham_energy_forces_stress``, ``loss_function``,
``run_fitting``, ``print_parameters``, ``evaluate_diagnostics``, plus a
module-level ``ParameterSet = BuckinghamParameterSet`` alias) that
predated the generic ``fit_potential`` interface. ``run_fitting``'s own
docstring already said it was nothing but a thin wrapper delegating to
``fit_potential``, so -- per the project decision to consolidate every
potential onto a single ``fit`` code path (no more special-cased
``fitpot``/Buckingham branch) -- that whole duplicate stack has been
removed here.  The one genuinely distinct feature it carried,
LAMMPS ``pair_coeff.txt`` export, is preserved as the explicit-argument
:func:`export_lammps_buckingham` helper, called from
:func:`save_fit_result_generic` when ``potential == "buckingham"``.
"""
import json
import os
import time
from typing import Optional

import numpy as np
from scipy.optimize import differential_evolution, least_squares, minimize

from ..constants import EV_ANG3_TO_GPA
from ..engine import compute_all_general
from ..utils.optional_matplotlib import HAS_MATPLOTLIB, plt
from ..utils.mpi import _mpi_allreduce_sum_scalar
from ..utils.timing import TimingCollector
from .io_utils import _pair_to_str, save_potential
from .loss import generic_loss_function, generic_residuals
from .parameter_sets import FittableParameterSet

__all__ = [
    "split_train_val",
    "ScoreCard",
    "compute_scorecard",
    "print_scorecard_comparison",
    "parity_plot_all",
    "fit_potential",
    "LOCAL_METHOD_OPTIONS",
    "DEFAULT_LOCAL_METHOD",
    "resolve_local_method",
    "print_fit_result",
    "evaluate_diagnostics_generic",
    "save_fit_result_generic",
    "export_lammps_buckingham",
]


def split_train_val(structures   : list,
                    val_fraction  : float = 0.2,
                    seed          : int   = 0,
                    stratify_by   : Optional[str] = "natoms",
                    ) -> tuple:
    """
    Split a structure list into training and validation subsets.

    Parameters
    ----------
    structures   : list of structure dicts (from the Buckingham fitting
                   workflow) or any list that can be indexed.
                   For ELM training data loaded from CSV, use the
                   ELMModel.fit() val_fraction argument instead.
    val_fraction : fraction of structures held out for validation [0, 1).
                   0.0 -> no validation set (all structures used for training).
    seed         : random seed for reproducibility.
    stratify_by  : None | 'natoms' | 'name_prefix'.
                   'natoms'       -- bin by atom count so that each bin is
                                   represented proportionally in both sets.
                   'name_prefix'  -- bin by the first directory component of
                                   the structure name (useful when structures
                                   are grouped by type: equilibrium, displaced).
                   None           -- pure random shuffle.

    Returns
    -------
    train : list of training structures
    val   : list of validation structures (empty when val_fraction == 0.0)

    Notes
    -----
    - val_fraction < 1/(n_structures) silently returns all structures as
      training and an empty validation set.
    - When stratify_by is set but the strata are too small to split, a
      warning is issued and the strata are merged into the random pool.
    """
    import warnings

    if val_fraction <= 0.0 or len(structures) == 0:
        return list(structures), []

    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(structures) * val_fraction)))

    if n_val >= len(structures):
        warnings.warn(
            f"split_train_val: val_fraction={val_fraction} leaves no training "
            f"data.  Returning all {len(structures)} structures as training.",
            UserWarning, stacklevel=2)
        return list(structures), []

    # ----- Stratified split ------------------------------------------------
    if stratify_by is not None:
        if stratify_by == "natoms":
            key_fn = lambda s: s["natoms"]
        elif stratify_by == "name_prefix":
            key_fn = lambda s: s["name"].split("/")[0] if "/" in s["name"] else s["name"]
        else:
            raise ValueError(f"Unknown stratify_by value: '{stratify_by}'")

        from collections import defaultdict
        bins: dict = defaultdict(list)
        for s in structures:
            bins[key_fn(s)].append(s)

        train_all, val_all = [], []
        unassigned = []
        for k, group in bins.items():
            rng.shuffle(group)
            n_v = max(0, round(len(group) * val_fraction))
            if n_v == 0 or n_v == len(group):
                # Group too small to split cleanly -> defer to random pool
                unassigned.extend(group)
            else:
                val_all.extend(group[:n_v])
                train_all.extend(group[n_v:])

        # Handle unassigned structures with plain random split
        if unassigned:
            rng.shuffle(unassigned)
            n_v2 = max(0, round(len(unassigned) * val_fraction))
            val_all.extend(unassigned[:n_v2])
            train_all.extend(unassigned[n_v2:])

        # Guarantee minimum sizes
        if len(val_all) == 0 or len(train_all) == 0:
            # Fall back to plain random
            idx = rng.permutation(len(structures))
            return ([structures[i] for i in idx[n_val:]],
                    [structures[i] for i in idx[:n_val]])

        rng.shuffle(train_all)
        rng.shuffle(val_all)
        return train_all, val_all

    # ----- Plain random split ----------------------------------------------
    idx = rng.permutation(len(structures))
    train = [structures[i] for i in idx[n_val:]]
    val   = [structures[i] for i in idx[:n_val]]
    return train, val


# ---------------------------------------------------------------------------
# ScoreCard: unified MAE/RMSE/MaxAE for E, F, sigma
# ---------------------------------------------------------------------------

class ScoreCard:
    """
    Collects prediction errors for energies, forces, and stresses and
    computes MAE, RMSE, and MaxAE for each quantity.

    Usage
    -----
    sc = ScoreCard("train")
    for s in structures:
        E_pred, F_pred, S_pred = model.compute(...)
        sc.add(s, E_pred, F_pred, S_pred)
    sc.finalize()
    print(sc)
    """

    def __init__(self, label: str = ""):
        self.label   = label
        self._e_ref  : list = []
        self._e_pred : list = []
        self._f_ref  : list = []
        self._f_pred : list = []
        self._s_ref  : list = []
        self._s_pred : list = []
        self._final  : Optional[dict] = None

    # ------------------------------------------------------------------
    def add(self,
            struct  : dict,
            e_pred  : float,
            f_pred  : Optional[np.ndarray],
            s_pred  : Optional[np.ndarray]) -> None:
        """Accumulate one structure's predictions."""
        N = struct["natoms"]
        # ref_energy is ALREADY eV/atom -- the convention of the training
        # CSV, of generic_loss_function (dE = e/N - ref_energy) and of
        # fit_potential's own docstring.  It used to be divided by N again
        # here, turning the reference into E/N^2 ~ 0, so the "Energy MAE"
        # every fit printed was just |E/atom| of the data: 2.19 eV/atom
        # for a GeSn fit whose held-out energy error, measured
        # independently, was 0.155 meV/atom.  The energy parity plots
        # were built from the same wrong numbers.
        self._e_ref.append(float(struct["ref_energy"]))
        self._e_pred.append(e_pred / N)

        if struct.get("ref_forces") is not None and f_pred is not None:
            self._f_ref.extend(struct["ref_forces"].ravel().tolist())
            self._f_pred.extend(np.asarray(f_pred).ravel().tolist())

        if struct.get("ref_stress") is not None and s_pred is not None:
            # Use Voigt 6 unique components (avoid double-counting off-diagonal)
            sig_r = struct["ref_stress"]
            sig_p = np.asarray(s_pred)
            for a, b in [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]:
                self._s_ref.append(float(sig_r[a, b]))
                self._s_pred.append(float(sig_p[a, b]))

    # ------------------------------------------------------------------
    def finalize(self) -> dict:
        """Compute and cache MAE / RMSE / MaxAE for all quantities."""
        def _metrics(ref_list, pred_list, gpa_factor=1.0):
            if not ref_list:
                return None
            r = np.array(ref_list)
            p = np.array(pred_list)
            d = np.abs(r - p) * gpa_factor
            return {
                "MAE"  : float(np.mean(d)),
                "RMSE" : float(np.sqrt(np.mean(d**2))),
                "MaxAE": float(np.max(d)),
                "N"    : len(r),
            }

        GPa = EV_ANG3_TO_GPA
        self._final = {
            "energy": _metrics(self._e_ref,  self._e_pred),          # eV/atom
            "forces": _metrics(self._f_ref,  self._f_pred),          # eV/A
            "stress": _metrics(self._s_ref,  self._s_pred, GPa),     # GPa
            "n_struct": len(self._e_ref),
        }
        return self._final

    @property
    def results(self) -> dict:
        if self._final is None:
            self.finalize()
        return self._final

    # ------------------------------------------------------------------
    def __str__(self) -> str:
        r = self.results
        lbl = f"[{self.label}]" if self.label else ""
        lines = [f"  {lbl} n_structures = {r['n_struct']}"]
        for qty, unit in [("energy","eV/atom"), ("forces","eV/A"),
                           ("stress","GPa")]:
            m = r[qty]
            if m is None:
                continue
            lines.append(
                f"  {lbl} {qty:8s}  MAE={m['MAE']:.4f}  "
                f"RMSE={m['RMSE']:.4f}  MaxAE={m['MaxAE']:.4f}  [{unit}]  "
                f"(n={m['N']})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unified score runner (shared by classical and ELM workflows)
# ---------------------------------------------------------------------------

def compute_scorecard(structures  : list,
                      predict_fn  ,
                      label       : str = "") -> ScoreCard:
    """
    Evaluate *predict_fn* on every structure and return a ScoreCard.

    Parameters
    ----------
    structures : list of structure dicts.
    predict_fn : callable(lattice, positions, atom_types)
                 -> (energy[eV], forces[eV/A], stress[eV/A^3])
    label      : descriptive label for the scorecard.

    Returns
    -------
    ScoreCard (already finalised)
    """
    import warnings

    sc = ScoreCard(label)
    for s in structures:
        try:
            e, f, sig = predict_fn(
                s["lattice"], s["positions"], s["atom_types"])
        except Exception as exc:
            warnings.warn(f"compute_scorecard: prediction failed for "
                          f"'{s['name']}': {exc}", UserWarning, stacklevel=2)
            continue
        sc.add(s, e, f, sig)
    sc.finalize()
    return sc


def print_scorecard_comparison(train_sc: ScoreCard,
                                val_sc  : Optional[ScoreCard] = None,
                                title   : str = "Prediction errors") -> None:
    """
    Print a formatted comparison table of train vs validation scorecards.

    Example output
    --------------
    ================================================================
      Prediction errors
    ================================================================
      Quantity     | Metric  |       Train |         Val | Unit
      -------------+---------+-------------+-------------+-----------
      Energy       | MAE     |      0.0012 |      0.0031 | eV/atom
      Energy       | RMSE    |      0.0018 |      0.0045 | eV/atom
      ...
    """
    sep = "=" * 64
    mid_parts = ["  " + "-" * 13 + "+" + "-" * 9 + "+" + "-" * 14]
    if val_sc is not None:
        mid_parts.append("+" + "-" * 14)
    mid_parts.append("+" + "-" * 11)
    mid = "".join(mid_parts)

    print("\n" + sep)
    print(f"  {title}")
    print(sep)

    quantities = [
        ("energy", "Energy  ", "eV/atom"),
        ("forces", "Forces  ", "eV/A   "),
        ("stress", "Stress  ", "GPa    "),
    ]
    hdr = f"  {'Quantity':12s} | {'Metric':7s} | {'Train':>12s}"
    if val_sc is not None:
        hdr += f" | {'Val':>12s}"
    hdr += " | Unit"
    print(hdr)
    print(mid)

    for qty, name, unit in quantities:
        tr = train_sc.results.get(qty)
        if tr is None:
            continue
        va = val_sc.results.get(qty) if val_sc is not None else None
        for metric in ("MAE", "RMSE", "MaxAE"):
            tr_v = f"{tr[metric]:12.4f}"
            row  = f"  {name:12s} | {metric:7s} | {tr_v}"
            if val_sc is not None:
                va_v = f"{va[metric]:12.4f}" if va else f"{'N/A':>12s}"
                row += f" | {va_v}"
            row += f" | {unit}"
            print(row)
        print(mid)

    print(f"  {'n_struct':12s} | {'':<7s} | {train_sc.results['n_struct']:>12d}", end="")
    if val_sc is not None:
        n_v = val_sc.results["n_struct"] if val_sc else 0
        print(f" | {n_v:>12d}", end="")
    print(" |")
    print(sep)


def parity_plot_all(train_sc  : ScoreCard,
                    val_sc    : Optional[ScoreCard] = None,
                    output_dir: str  = ".",
                    prefix    : str  = "parity") -> None:
    """
    Save 3-panel parity plots (E / F / sigma) for train and optionally val sets.

    Each panel shows:
      - Training points in blue
      - Validation points in orange (when val_sc is not None)
      - Perfect-prediction diagonal (dashed)
      - Per-set RMSE in the legend
    """
    if not HAS_MATPLOTLIB:
        return
    os.makedirs(output_dir, exist_ok=True)

    quantities = [
        ("energy", "Energy (eV/atom)",    "_e_ref",  "_e_pred"),
        ("forces", "Force comp (eV/A)",   "_f_ref",  "_f_pred"),
        ("stress", "Stress comp (GPa)",   "_s_ref",  "_s_pred"),
    ]
    GPa = EV_ANG3_TO_GPA

    for qty, xlabel, rattr, pattr in quantities:
        tr_ref  = getattr(train_sc, rattr, [])
        tr_pred = getattr(train_sc, pattr, [])
        if not tr_ref:
            continue

        fig, ax = plt.subplots(figsize=(5, 5))

        # Apply GPa conversion for stress
        scale = GPa if qty == "stress" else 1.0
        tr_r = np.array(tr_ref) * scale
        tr_p = np.array(tr_pred) * scale
        tr_rmse = train_sc.results[qty]["RMSE"] if train_sc.results.get(qty) else 0.0

        ax.scatter(tr_r, tr_p, s=8, alpha=0.5, color="steelblue",
                   label=f"Train  RMSE={tr_rmse:.4f}")

        if val_sc is not None:
            va_ref  = getattr(val_sc, rattr, [])
            va_pred = getattr(val_sc, pattr, [])
            if va_ref:
                va_r = np.array(va_ref) * scale
                va_p = np.array(va_pred) * scale
                va_rmse = val_sc.results[qty]["RMSE"] if val_sc.results.get(qty) else 0.0
                ax.scatter(va_r, va_p, s=8, alpha=0.7, color="darkorange",
                           marker="^", label=f"Val    RMSE={va_rmse:.4f}")

        all_v = np.concatenate([tr_r, tr_p])
        if val_sc is not None and getattr(val_sc, rattr, []):
            va_r  = np.array(getattr(val_sc, rattr)) * scale
            va_p  = np.array(getattr(val_sc, pattr)) * scale
            all_v = np.concatenate([all_v, va_r, va_p])
        lim = [all_v.min(), all_v.max()]
        margin = 0.05 * (lim[1] - lim[0] + 1e-10)
        lim = [lim[0] - margin, lim[1] + margin]
        ax.plot(lim, lim, "k--", lw=1, zorder=0)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"DFT {xlabel}"); ax.set_ylabel(f"Predicted {xlabel}")
        ax.set_title(f"Parity -- {xlabel}")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        fig.tight_layout()
        path = os.path.join(output_dir, f"{prefix}_{qty}.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  Parity plot saved : {path}")


# ---------------------------------------------------------------------------
#  Local optimisers accepted by fit_potential
# ---------------------------------------------------------------------------

# Every scipy.optimize.minimize method that honours BOUNDS and runs without
# an analytic gradient -- fit_potential supplies neither a Jacobian nor a
# Hessian -- each with the SAME stopping intent (at most 5000 iterations
# and 100000 loss evaluations; loss tolerance 1e-12 eV^2; gradient
# tolerance 1e-8; step tolerance 1e-8 in the optimisation variable, which
# is log(value) for most parameters, i.e. ~1e-8 relative) expressed in
# that method's own option names.
#
# Bug-fix note: local_method used to be passed to scipy unchecked, always
# with the L-BFGS-B option dict {maxiter, ftol, gtol, maxfun}.  Measured on
# a bounded problem with scipy 1.17.1: BFGS and CG accepted the bounds
# with only a warning and ended OUTSIDE them, 0.76 from the answer;
# trust-constr raised TypeError on the foreign option keys; Newton-CG and
# the four trust-region Newton methods raised for want of a Jacobian; and
# every other method silently ignored some of the keys (Nelder-Mead all
# three tolerances), stopping on scipy's defaults instead.  With the
# table below every listed method converges on that problem and on a
# 12-D bounded quadratic, inside the bounds, with no warnings.
#
# Powell satisfies the same two conditions and is still left out, on
# measurement against REAL fit losses (an LJ fit from three starting
# points, and a 6-variable GeSn Stillinger-Weber fit): with bounds it
# returned a point worse than one it had itself evaluated on all four --
# loss 2-3e-2 where every other method reached 1e-13 or below on LJ, and
# 0.58 on GeSn after evaluating 7.6e-3 -- and on one LJ start it returned
# a point worse than its STARTING point (4.42 vs 3.58).  The same LJ
# problem without bounds converges to 3.9e-27, so this is scipy's bounded
# Powell, not the loss.  (Also measured there, and worth knowing: from
# the worst LJ start, L-BFGS-B stalled at 14.2 while TNC, SLSQP,
# Nelder-Mead, COBYLA and COBYQA all reached 1e-14 or below; and on the
# GeSn problem adaptive Nelder-Mead was the most accurate, 5e-19.)
# Default local method: Nelder-Mead (adaptive), since 2026-09-26.
# L-BFGS-B, the former default, works from a finite-difference gradient
# and stopped far from the answer on three real fit losses where the loss
# at the reference parameters is ~1e-14 (measured 2026-09-26, scipy
# 1.17.1; loss after the local step, loss evaluations in brackets):
#
#                    Si SW from DE   Si SW from x0   GeSn SW (19 var) from x0
#   L-BFGS-B         6.6e-5 [19]     2.3e-3 [559]    1.9e+4 [61]  (no step)
#   Nelder-Mead      7.2e-15 [827]   7.2e-15 [1660]  2.7e-14 [31721]
#   TNC              2.8e-5 [247]    0.50 [1117]     not finished in 100 min
#   SLSQP            3.6e-6 [4699]   3.2e+5 [7] (no step)  7.5e-7 [7765]
#   COBYQA           1.7e-13 [1219]  4.9e-14 [4043]  1.1e-7 [100 001] (maxfev)
#
# L-BFGS-B stops after one line search in narrow curved valleys
# ("relative reduction <= factr*epsmch" with |g| ~ 8), and on GeSn its
# first trial step reaches a cutoff the training cells cannot host, which
# the loss scores as a failed evaluation (1e30) and ends the search.
# Nelder-Mead needs no gradient and treats such points as merely bad.
# Its maxiter is 100 000, the same as maxfev: at 5000 it stopped GeSn at
# 3.2e-3 on the iteration limit.  It costs more evaluations (17 min for
# GeSn), which is the price of reaching the minimum.
DEFAULT_LOCAL_METHOD = "Nelder-Mead"

LOCAL_METHOD_OPTIONS = {
    "L-BFGS-B":    {"maxiter": 5000, "maxfun": 100_000, "ftol": 1e-12, "gtol": 1e-8},
    "TNC":         {"maxfun": 100_000, "ftol": 1e-12, "gtol": 1e-8},
    "SLSQP":       {"maxiter": 5000, "ftol": 1e-12},
    # adaptive: dimension-dependent simplex coefficients (Gao & Han 2012);
    # the fixed textbook ones degrade badly beyond a handful of variables.
    # scipy stops only when BOTH the simplex's x spread <= xatol AND its
    # loss spread <= fatol, and fatol is absolute.  With fatol = 1e-12 a
    # fit whose minimum is not ~0 (any real data) never met it: the Si SW
    # control (minimum 2.12) ran the full 100 000 evaluations, ~50 min,
    # to 2.121594, a point reached in 1758 evaluations with fatol off.
    # On the three zero-residual cases above fatol off changes nothing
    # (same loss, same evaluation count), so the stop is xatol alone: a
    # simplex 1e-8 wide in log(value), ~1e-8 relative in each parameter.
    "Nelder-Mead": {"maxiter": 100_000, "maxfev": 100_000,
                    "fatol": float("inf"), "xatol": 1e-8, "adaptive": True},
    # COBYLA's maxiter counts loss evaluations, not iterations.
    "COBYLA":      {"maxiter": 100_000, "tol": 1e-10},
    "COBYQA":      {"maxiter": 5000, "maxfev": 100_000, "final_tr_radius": 1e-10},
    # ADR-0015: scipy.optimize.least_squares, method 'dogbox' (bounds
    # honoured), on the residual vector whose squares the loss sums
    # (loss.generic_residuals), 2-point finite-difference Jacobian,
    # x_scale='jac'.  On the Ni-Al Zhou-EAM fit the loss Hessian had a
    # condition number 1e11-1e14 in a curved valley where Nelder-Mead
    # crept for 1e5 evaluations; a Gauss-Newton type method uses the
    # curvature directly.  On an 8-variable zero-residual Ni3Al test
    # (2 % start): dogbox 5 iterations, 46 evaluations, loss 2e-25; 'lm'
    # (no bounds) 54 evaluations, 2e-32; 'trf' 177 iterations and still
    # 1e-10, 50 x slower (its reflective bound handling stalls here).
    # max_nfev counts residual evaluations of the least-squares iterations
    # (the Jacobian's own evaluations are not included by scipy).  The
    # Jacobian is fit_potential's own forward difference (jacobian_fn).
    "least_squares": {"max_nfev": 2000, "ftol": 1e-12, "xtol": 1e-12,
                      "gtol": 1e-12},
}

_REJECTED_LOCAL_METHODS = {
    "bfgs": "does not support bounds -- scipy warns and then steps outside them",
    "cg": "does not support bounds -- scipy warns and then steps outside them",
    "newton-cg": "requires an analytic gradient, which fit does not provide",
    "dogleg": "requires an analytic gradient and Hessian, which fit does not provide",
    "trust-ncg": "requires an analytic gradient and Hessian, which fit does not provide",
    "trust-exact": "requires an analytic gradient and Hessian, which fit does not provide",
    "trust-krylov": "requires an analytic gradient and Hessian, which fit does not provide",
    "trust-constr": "is not supported by fit (its option set is incompatible)",
    "powell": ("is not supported by fit: with bounds, scipy's Powell returned "
               "points worse than ones it had itself evaluated on real fit losses"),
}


def resolve_local_method(name: str):
    """Map a user-supplied local optimiser name (case-insensitive) to its
    canonical scipy spelling and option dict, or raise ValueError saying
    why it cannot be used and what can."""
    key = str(name).strip().lower()
    for canon, opts in LOCAL_METHOD_OPTIONS.items():
        if canon.lower() == key:
            return canon, dict(opts)
    allowed = ", ".join(LOCAL_METHOD_OPTIONS)
    reason = _REJECTED_LOCAL_METHODS.get(key, "is not a recognised method")
    raise ValueError(f"local_method '{name}' {reason}.  Choose one of: {allowed}.")


def _rotated_simplex(x, bounds, rng):
    """Nelder-Mead start simplex around x: scipy's step sizes (5 % of each
    component, 2.5e-4 for a zero one) along the columns of a random
    orthonormal matrix, clipped into the bounds."""
    x = np.asarray(x, float)
    n = len(x)
    step = np.where(x != 0.0, 0.05 * np.abs(x), 2.5e-4)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    sim = [x] + [np.clip(x + step * q[:, i], lo, hi) for i in range(n)]
    return np.array(sim)


def fit_potential(param_set      : FittableParameterSet,
                  structures     : list,
                  r_cut          : float,
                  ewald_alpha    : Optional[float] = None,
                  weights        : Optional[dict]  = None,
                  method         : str   = DEFAULT_LOCAL_METHOD,
                  use_de         : bool  = False,
                  de_maxiter     : int   = 500,
                  de_popsize     : int   = 15,
                  coulomb_method : str   = "ewald",
                  loss_type      : str   = "squared",
                  val_fraction   : float = 0.2,
                  val_seed       : int   = 0,
                  output_dir     : str   = ".",
                  verbose        : bool  = True,
                  timing         : bool  = False,
                  restarts       : int   = 0,
                  restart_rtol   : float = 1e-3,
                  local_maxiter  : Optional[int] = None,
                  restart_seed   : int   = 0,
                  time_limit_hours: Optional[float] = None) -> dict:
    """
    Unified parameter fitting for any pair potential.

    Supports LennardJonesPotential, MorsePotential, BornMayerPotential,
    BuckinghamPotential (via BuckinghamParameterSet), StillingerWeberPotential,
    and TersoffPotential through a common optimisation loop.

    Loss function
    -------------
    Energy term normalised by atom count; force and stress terms averaged
    over components within each structure.  Per-structure unit-conversion
    scale via ``structure["scale"]`` (default 1.0), and an independent
    per-structure trust weight via ``structure["weight"]`` (default 1.0):

    .. math::

        L = \\sum_s w_{\\rm trust}(s) \\, c_s^2 \\left[
              w_E \\left(\\frac{E_s^{\\rm pred} - E_s^{\\rm ref}}{N_s}\\right)^2
            + w_F \\,\\overline{\\|\\delta\\mathbf{F}\\|^2}
            + w_\\sigma \\,\\overline{|\\delta\\boldsymbol{\\sigma}|^2}
          \\right]

    Set the ``r0`` and ``eps0`` keys in the ``weights`` dict to bring
    all residuals to the same physical unit (eV^2); see
    :func:`~akitsushima.fitting.loss.generic_loss_function` for the
    exact formula, including the optional ``loss_type='huber'`` mode.

    Parameters
    ----------
    param_set      : FittableParameterSet instance.
    structures     : list of structure dicts (``lattice``, ``positions``,
                     ``atom_types``, ``natoms``, ``ref_energy``; optional
                     ``ref_forces``, ``ref_stress``, ``scale``).
    r_cut          : short-range cutoff [A].
    ewald_alpha    : Ewald alpha [A^-1]; auto if None.
    weights        : dict controlling the loss function.
                     Keys and defaults:

                     ======  =======  =================================================
                     Key     Default  Meaning
                     ======  =======  =================================================
                     energy  1.0      w_E  -- energy category weight
                     forces  1.0      w_F  -- force category weight
                     stress  0.1      w_S  -- stress category weight
                     r0      1.0      r_0 [A] -- atomic displacement for F->eV conversion
                     eps0    1.0      epsilon_0 [--] -- cell strain for S->eV conversion
                     ======  =======  =================================================

                     ``ref_energy`` in structure dicts must be in **eV/atom**.
                     The loss function converts each residual to eV:

                     - E term:  w_E x (N_s x deltaE_per_atom)^2  [eV^2]
                     - F term:  w_F x r_0^2 x mean(deltaF^2)       [eV^2]
                     - S term:  w_S x (V_s epsilon_0)^2 x mean(deltasigma^2) [eV^2]
    method         : local optimiser, one of LOCAL_METHOD_OPTIONS
                     (default 'Nelder-Mead'; case-insensitive).  Checked
                     BEFORE any loss evaluation, so a bad name fails in
                     seconds instead of after a long DE run.
    use_de         : run Differential Evolution before local refinement.
    de_maxiter     : max DE generations.
    de_popsize     : DE population size (x n_params).
    coulomb_method : 'ewald' | 'pppm'.
    loss_type      : 'squared' (default) | 'huber' -- see
                     :func:`~akitsushima.fitting.loss.generic_loss_function`.
                     When 'huber', ``weights`` must also contain
                     ``huber_delta_energy``/``huber_delta_forces``/
                     ``huber_delta_stress`` (eV).
    val_fraction   : fraction of structures held out for validation.
    val_seed       : RNG seed for the train/val split.
    output_dir     : directory for parity plots.
    verbose        : print progress.
    restarts       : further local runs, each from the best point so far;
                     default 0.  For Nelder-Mead each restart starts from a
                     new simplex: scipy's step sizes (5 % of each variable,
                     2.5e-4 for a zero) along a random orthonormal basis
                     (``restart_seed``).  A restart from scipy's own
                     axis-aligned simplex retraces the previous run
                     exactly and cannot improve on it (ADR-0015: SiC stuck
                     at 2.9e-5; the same point reached 3.5e-15 from a
                     start differing only in the last digits).  Stops
                     after two consecutive restarts that each improve the
                     loss by less than ``restart_rtol`` (relative).
    time_limit_hours : wall-clock budget [h] for the whole search (DE,
                     local runs and restarts), measured with time.time():
                     a machine asleep in between counts, as the user's
                     deadline does (time.monotonic stops during sleep on
                     macOS, which let a run with an 8.8 h limit go on for
                     12.5 h of wall time, ADR-0015).  When it is exceeded the
                     search stops at the next loss evaluation and the
                     best point evaluated so far is kept and saved as
                     usual (``result["time_limited"]`` is True).  Under
                     MPI the decision is reduced over all ranks, so they
                     stop at the same evaluation.
    local_maxiter  : cap on the iterations (and evaluations) of each
                     local run, so that a slow run is restarted instead of
                     spending the whole budget in one valley; default: the
                     method's own (LOCAL_METHOD_OPTIONS).

    A note on cross-validation
    --------------------------
    This function only ever performs a single stratified train/val
    holdout (via :func:`split_train_val`), not full k-fold
    cross-validation. Given the objective is already called "tens of
    thousands of times" per fit (see
    :func:`~akitsushima.fitting.loss.generic_loss_function`'s own module
    docstring), k-fold would multiply total fitting wall-clock time by
    k -- disproportionate for the typical training-set sizes this
    project's classical-potential fitting targets (few to a few dozen
    structures, a handful of fitted parameters). For a cheap variance
    estimate instead of true k-fold, call this function several times
    with different ``val_seed`` values and compare the resulting
    ``val_sc``/``val_loss`` spread across runs -- ``split_train_val``'s
    existing ``seed`` parameter already supports exactly this "repeated
    holdout" recipe with no new code needed.

    Returns
    -------
    dict with keys: ``params``, ``engine``, ``x``, ``loss``,
    ``train_structs``, ``val_structs``, ``train_sc``, ``val_sc``.
    """
    method, local_options = resolve_local_method(method)
    if local_maxiter is not None:
        local_options = dict(local_options)
        for key in ("maxiter", "maxfev", "maxfun", "max_nfev"):
            if key in local_options:
                local_options[key] = int(local_maxiter)
    if int(restarts) < 0:
        raise ValueError("fit_potential: restarts must be >= 0")

    _tc_fit = TimingCollector(enabled=timing)
    _tc_fit.start("total")
    if weights is None:
        weights = {"energy": 1.0, "forces": 1.0, "stress": 0.1}

    # ---- Train / val split ------------------------------------------------
    if val_fraction > 0.0:
        train_structs, val_structs = split_train_val(
            structures, val_fraction=val_fraction, seed=val_seed,
            stratify_by="natoms")
        if verbose:
            print(f"\nTrain/val split (val_fraction={val_fraction}):")
            print(f"  Train: {len(train_structs)} structures")
            print(f"  Val  : {len(val_structs)} structures")
    else:
        train_structs = list(structures)
        val_structs   = []

    if not train_structs:
        raise RuntimeError("fit_potential: training set is empty after split.")

    # ---- Objective -------------------------------------------------------
    call_count = [0]
    t_start = time.time()
    tracked = {"x": None, "loss": np.inf}

    class _TimeUp(Exception):
        pass

    def objective(x):
        call_count[0] += 1
        loss = generic_loss_function(
            x, param_set, train_structs, weights,
            r_cut, ewald_alpha, coulomb_method, loss_type)
        if loss < tracked["loss"]:
            tracked["x"], tracked["loss"] = np.array(x, float), float(loss)
        if verbose and call_count[0] % 50 == 0:
            print(f"    iter {call_count[0]:6d}  loss = {loss:.6e}")
        if time_limit_hours is not None and armed[0]:
            over = (time.time() - t_start) > 3600.0 * float(time_limit_hours)
            if _mpi_allreduce_sum_scalar(1.0 if over else 0.0) > 0.0:
                raise _TimeUp
        return loss

    time_limited = [False]
    armed = [False]         # the initial evaluation is never cut

    def residual_fn(x):
        r = generic_residuals(x, param_set, train_structs, weights,
                              r_cut, ewald_alpha, coulomb_method)
        call_count[0] += 1
        loss = float(r @ r)
        if loss < tracked["loss"]:
            tracked["x"], tracked["loss"] = np.array(x, float), loss
        if verbose and call_count[0] % 50 == 0:
            print(f"    eval {call_count[0]:6d}  loss = {loss:.6e}")
        if time_limit_hours is not None and armed[0]:
            over = (time.time() - t_start) > 3600.0 * float(time_limit_hours)
            if _mpi_allreduce_sum_scalar(1.0 if over else 0.0) > 0.0:
                raise _TimeUp
        return r

    lo_b = np.array([b[0] for b in param_set.bounds], float)
    hi_b = np.array([b[1] for b in param_set.bounds], float)

    def jacobian_fn(x, *args, **kwargs):
        """Forward differences, step 1e-7 max(1, |x_i|) (backward at an
        upper bound).  scipy's own relative step, diff_step |x_i|, is zero
        at x_i = 0 -- e.g. log(chi) at chi = 1 -- where it returned a zero
        column and least_squares stopped on gtol at a wrong point."""
        r0 = residual_fn(x)
        J = np.empty((len(r0), len(x)))
        for i in range(len(x)):
            h = 1e-7 * max(1.0, abs(x[i]))
            if x[i] + h > hi_b[i]:
                h = -h
            xp = np.array(x, float)
            xp[i] += h
            J[:, i] = (residual_fn(xp) - r0) / h
        return J

    def local(x0, options):
        """One local run from x0; returns an object with .x and .fun."""
        if method == "least_squares":
            if loss_type != "squared":
                raise ValueError("local_method 'least_squares' needs loss_type "
                                 "'squared'")
            opts = {k: v for k, v in options.items()
                    if k not in ("initial_simplex", "diff_step")}
            res = run(least_squares, residual_fn, np.clip(x0, lo_b, hi_b),
                      jac=jacobian_fn, bounds=(lo_b, hi_b), method="dogbox",
                      x_scale="jac", **opts)
            if res is None:
                return None
            res.fun = float(2.0 * res.cost)     # scipy's cost is 1/2 sum r^2
            return res
        return run(minimize, objective, x0=x0, method=method,
                   bounds=param_set.bounds, options=options)

    def run(fn, *args, **kwargs):
        """Run an optimiser; on the time limit return None (best point
        so far stays in ``tracked``)."""
        if time_limited[0]:
            return None
        armed[0] = True
        try:
            return fn(*args, **kwargs)
        except _TimeUp:
            time_limited[0] = True
            if verbose:
                print(f"\nTime limit of {time_limit_hours:g} h reached after "
                      f"{call_count[0]} evaluations; keeping the best point "
                      f"(loss {tracked['loss']:.6e})")
            return None

    best_x    = param_set.x0.copy()
    best_loss = objective(best_x)
    if verbose:
        print(f"\nInitial loss : {best_loss:.6e}")

    # ---- Differential Evolution (optional) --------------------------------
    if use_de:
        if verbose:
            print("\nRunning Differential Evolution ...")
        de_res = run(differential_evolution,
            objective, bounds=param_set.bounds,
            maxiter=de_maxiter, popsize=de_popsize,
            tol=1e-8, seed=42, workers=1, polish=False,
            disp=verbose)
        if de_res is not None and de_res.fun < best_loss:
            best_x, best_loss = de_res.x, de_res.fun
        if verbose:
            print(f"DE finished  : loss = {best_loss:.6e}")

    # ---- Local refinement -------------------------------------------------
    if verbose:
        print(f"\nLocal refinement with {method} ...")
    opt = local(best_x, local_options)
    if opt is not None and opt.fun < best_loss:
        best_x, best_loss = opt.x, opt.fun
    loss_history = [float(best_loss)]
    rng = np.random.default_rng(restart_seed)
    stalled = 0
    for k in range(int(restarts)):
        if time_limited[0]:
            break
        if verbose:
            print(f"\nRestart {k + 1}/{restarts} from loss {best_loss:.6e} ...")
        prev = best_loss
        options = dict(local_options)
        if method == "Nelder-Mead":
            options["initial_simplex"] = _rotated_simplex(
                best_x, param_set.bounds, rng)
        opt = local(best_x, options)
        if opt is not None and opt.fun < best_loss:
            best_x, best_loss = opt.x, opt.fun
        if time_limited[0] and tracked["loss"] < best_loss:
            best_x, best_loss = tracked["x"], tracked["loss"]
        loss_history.append(float(best_loss))
        if verbose:
            print(f"  -> loss {best_loss:.6e}")
        stalled = stalled + 1 if (prev <= 0.0 or (prev - best_loss) < restart_rtol * prev) else 0
        if stalled >= 2:
            if verbose:
                print(f"  two restarts below restart_rtol = {restart_rtol:g}; stopping")
            break

    if tracked["x"] is not None and tracked["loss"] < best_loss:
        best_x, best_loss = tracked["x"], tracked["loss"]

    params_fit  = param_set.decode(best_x)
    engine_fit  = param_set.build_engine(params_fit)
    charges_fit = params_fit.get("charges")

    if verbose:
        print(f"\nFinal   loss : {best_loss:.6e}")

    # ---- ScoreCards -------------------------------------------------------
    def _predict(lattice, positions, atom_types):
        return compute_all_general(
            lattice, positions, atom_types,
            engine_fit,
            charges        = charges_fit,
            r_cut          = r_cut,
            ewald_alpha    = ewald_alpha,
            coulomb_method = coulomb_method,
        )

    train_sc = compute_scorecard(train_structs, _predict, label="train")
    val_sc   = (compute_scorecard(val_structs, _predict, label="val ")
                if val_structs else None)

    print_scorecard_comparison(train_sc, val_sc,
                               title=f"{type(param_set).__name__} -- prediction errors")

    parity_plot_all(train_sc, val_sc, output_dir=output_dir,
                    prefix=type(param_set).__name__.replace("ParameterSet", "").lower())

    # Compute validation loss for diagnostic output
    if val_structs:
        val_loss = generic_loss_function(
            best_x, param_set, val_structs, weights,
            r_cut=r_cut, ewald_alpha=ewald_alpha, loss_type=loss_type)
        if verbose:
            print(f"  val_loss  = {val_loss:.6e}  "
                  f"(train_loss = {best_loss:.6e}  "
                  f"ratio = {val_loss/best_loss:.3f})")
    else:
        val_loss = None

    _tc_fit.stop("total")
    if timing:
        _tc_fit.print_summary(label="fit_potential")

    return {
        "params"        : params_fit,
        "engine"        : engine_fit,
        "x"             : best_x,
        "loss"          : best_loss,
        "val_loss"      : val_loss,
        "opt_result"    : opt,
        "loss_history"  : loss_history,
        "time_limited"  : bool(time_limited[0]),
        "train_structs" : train_structs,
        "val_structs"   : val_structs,
        "train_sc"      : train_sc,
        "val_sc"        : val_sc,
        "timing"        : _tc_fit.summary(),
    }


# ---------------------------------------------------------------------------
#  Reporting / persistence of fit_potential() results (any potential type)
# ---------------------------------------------------------------------------

def print_fit_result(result: dict, potential: str) -> None:
    """
    Pretty-print fitted parameters for any potential type.
    """
    params = result.get("params", {})
    print("\n" + "=" * 62)
    print(f"  Fitted Parameters  [{potential.upper()}]")
    print("=" * 62)

    if potential == "buckingham":
        buck    = params.get("buck", {})
        charges = params.get("charges", {})
        print(f"  {'Pair':<15}  {'A (eV)':>14}  {'rho (A)':>10}  {'C (eV.A^6)':>12}")
        print("  " + "-" * 58)
        for pair, p in buck.items():
            # Bug-fix note: a like-species pair is frozenset({'O'}), one
            # element, so `s1, s2 = sorted(pair)` raised ValueError -- after
            # the optimisation, before save_fit_result_generic, losing the
            # fit.  Any Buckingham fit with an O-O term (i.e. every oxide)
            # crashed here.
            names  = sorted(pair)
            s1, s2 = names[0], names[-1]
            print(f"  {s1+'-'+s2:<15}  {p['A']:14.4f}  "
                  f"{p['rho']:10.6f}  {p['C']:12.4f}")
        if charges:
            print(f"\n  {'Species':<12}  {'Charge (e)':>12}")
            print("  " + "-" * 28)
            for sp, q in charges.items():
                print(f"  {sp:<10}  {q:+.4f} e")

    elif potential in ("lj",):
        lj = params.get("lj", {})
        print(f"  {'Pair':<15}  {'epsilon (eV)':>14}  {'sigma (A)':>12}")
        print("  " + "-" * 45)
        for pair, p in lj.items():
            sp = "-".join(sorted(pair))
            print(f"  {sp:<15}  {p['epsilon']:>14.6f}  {p['sigma']:>12.6f}")

    elif potential == "morse":
        mo = params.get("morse", {})
        print(f"  {'Pair':<15}  {'D_e (eV)':>10}  {'alpha (A^-1)':>12}  {'r0 (A)':>10}")
        print("  " + "-" * 55)
        for pair, p in mo.items():
            sp = "-".join(sorted(pair))
            print(f"  {sp:<15}  {p['D_e']:>10.6f}  {p['alpha']:>12.6f}  {p['r0']:>10.6f}")

    elif potential == "bornmayer":
        bm      = params.get("bm", {})
        charges = params.get("charges", {})
        print(f"  {'Pair':<15}  {'A (eV)':>14}  {'rho (A)':>12}")
        print("  " + "-" * 45)
        for pair, p in bm.items():
            sp = "-".join(sorted(pair))
            print(f"  {sp:<15}  {p['A']:>14.4f}  {p['rho']:>12.6f}")
        if charges:
            print(f"\n  {'Species':<12}  {'Charge (e)':>12}")
            print("  " + "-" * 28)
            for sp, q in charges.items():
                print(f"  {sp:<12}  {q:>+12.4f}")

    elif potential in ("sw", "tersoff"):
        # Two shapes reach here.  The single-species parameter sets decode
        # to a FLAT {name: value} dict; the multi-species SW set decodes
        # to {(i, j, k): {name: value}}.
        #
        # Bug-fix note: this branch used to assume the nested shape only,
        # so on every flat result it iterated ('epsilon', 2.1683) as if
        # it were (triple, sub-dict) and raised "AttributeError: 'float'
        # object has no attribute 'items'".  print_fit_result runs AFTER
        # the optimisation and BEFORE save_fit_result_generic, so
        # `akitsushima fit --potential sw` and `--potential tersoff`
        # completed the whole fit and then lost it without writing the
        # result.  No test ran either potential through cmd_fit.
        #
        # Triples are printed as "i;j,k" in their stored ORDER.  They
        # used to be joined after sort(), which folds (Ge,Ge,Sn) and
        # (Ge,Sn,Ge) -- two distinct rows with distinct gamma in the
        # multi-species format -- onto the same label.
        key = "sw" if potential == "sw" else "tersoff"
        sub = params.get(key, {})
        if sub and all(not isinstance(v, dict) for v in sub.values()):
            print()
            for k, v in sub.items():
                print(f"    {k:<20s} = {float(v):>15.8g}")
        else:
            for trio, p in sub.items():
                if isinstance(trio, tuple) and len(trio) == 3:
                    label = f"{trio[0]};{trio[1]},{trio[2]}"
                else:
                    label = str(trio)
                print(f"\n  [{label}]")
                for k, v in p.items():
                    print(f"    {k:<20s} = {float(v):>15.8g}")

    else:
        # Generic fallback: dump the params dict, recursively converting
        # any frozenset-keyed pair dict (e.g. Mie/Yukawa's own
        # {pair: {...}} shape, identical to LJ/Morse/BornMayer's) into
        # "Sp1-Sp2" string keys first.
        #
        # Bug-fix note (found adding Mie/Yukawa, Phase 2): this used to
        # stringify only the OUTER dict's keys (here, just the single
        # potential-name key, e.g. "mie" -- already a string, a no-op),
        # leaving any frozenset keys nested one level down inside
        # completely unconverted, so json.dumps immediately raised
        # ``TypeError: keys must be str, int, float, bool or None, not
        # frozenset`` for every potential type that isn't one of the 5
        # branches explicitly special-cased above. Every potential
        # added after those 5 originally-ported ones would have hit this
        # same crash -- the fallback needed to be genuinely generic, not
        # almost-generic, given the whole point of Phase 1's registry is
        # that adding a potential should not require finding every
        # remaining hardcoded special case by trial and error. Mirrors
        # the recursive frozenset-key conversion already used by
        # save_potential's own _serialise helper (io_utils.py).
        def _serialise(p):
            if isinstance(p, dict):
                return {(_pair_to_str(k) if isinstance(k, frozenset) else str(k)):
                        _serialise(v) for k, v in p.items()}
            return p

        print(json.dumps(_serialise(params), indent=2, default=str))

    print("=" * 62)


def evaluate_diagnostics_generic(structures    : list,
                                   param_set,
                                   fitted_params : dict,
                                   r_cut         : float,
                                   output_dir    : str,
                                   ewald_alpha   : Optional[float] = None,
                                   coulomb_method: str = "ewald",
                                   val_structs   : Optional[list]  = None,
                                   ) -> dict:
    """
    Compute energy / force / stress MAE and RMSE for any ParameterSet.

    Builds an engine from the **fitted** parameters and evaluates all
    structures through :func:`~akitsushima.engine.compute_all_general`
    (so that any Coulomb/Ewald contribution driven by ``fitted_params
    ["charges"]`` -- e.g. Buckingham or Born-Mayer + Coulomb -- is
    included, not just the short-range PotentialEngine part).

    Bug-fix note (F5): the original implementation ignored the fitted
    result entirely and re-evaluated ``param_set.decode(param_set.x0)``
    -- i.e. always the *initial guess*, never the optimised parameters
    -- so every "diagnostics" report was silently meaningless.  It also
    built the force function via ``make_force_fn(engine)`` on the bare
    short-range PotentialEngine, which for an ionic (Buckingham/
    Born-Mayer + Coulomb) potential drops the electrostatic energy,
    forces, and stress entirely; that gap was latent in the original
    code (the CLI never routed Buckingham through this function) but
    would become live and silently wrong once Buckingham is unified
    onto the same generic ``fit`` path as every other potential.  Both
    issues are fixed here: the caller must now pass the fitted
    parameter dict explicitly, and evaluation goes through
    ``compute_all_general(..., charges=...)`` directly.

    Returns a diagnostics dict compatible with ``save_fit_result_generic``.
    """
    import math

    engine = param_set.build_engine(fitted_params)
    if engine is None:
        return {}
    charges = fitted_params.get("charges")

    def _eval_set(structs, label):
        e_err, f_err, s_err = [], [], []
        for s in structs:
            e_pred, f_pred, s_pred = compute_all_general(
                s["lattice"], s["positions"], s["atom_types"], engine,
                charges        = charges,
                r_cut          = r_cut,
                ewald_alpha    = ewald_alpha,
                coulomb_method = coulomb_method,
            )
            n = s["natoms"]
            if s.get("ref_energy") is not None:
                e_err.append(e_pred / n - s["ref_energy"])
            if s.get("ref_forces") is not None:
                f_err.extend((f_pred - s["ref_forces"]).ravel().tolist())
            if s.get("ref_stress") is not None:
                s_err.extend((s_pred - s["ref_stress"]).ravel().tolist())

        def _stats(arr):
            if not arr:
                return {"mae": None, "rmse": None}
            a = [abs(x) for x in arr]
            mae  = sum(a) / len(a)
            rmse = math.sqrt(sum(x**2 for x in arr) / len(arr))
            return {"mae": mae, "rmse": rmse}

        return {
            "label"  : label,
            "n"      : len(structs),
            "energy" : _stats(e_err),
            "forces" : _stats(f_err),
            "stress" : _stats(s_err),
        }

    diag: dict = {"train": _eval_set(structures, "train")}
    if val_structs:
        diag["val"] = _eval_set(val_structs, "val")

    # Print table
    print("\n  Diagnostics")
    print("  " + "-" * 54)
    for key, d in diag.items():
        print(f"  [{d['label']:5s}]  N={d['n']}")
        for qty in ("energy", "forces", "stress"):
            st = d[qty]
            if st["mae"] is None:
                continue
            print(f"    {qty:<8s}  MAE={st['mae']:.4e}  RMSE={st['rmse']:.4e}")

    return diag


def export_lammps_buckingham(buck_params: dict,
                              charges    : dict,
                              output_dir : str) -> None:
    """
    Write a LAMMPS ``pair_coeff.txt`` snippet for a fitted Buckingham
    potential (``pair_style buck/coul/long``).

    Kept as a standalone, explicit-argument function -- rather than
    bundled into a Buckingham-only diagnostics/fitting stack -- so that
    ``cmd_fit`` can call it for ``potential == "buckingham"`` without
    branching onto a separate code path for the rest of the fit.
    """
    os.makedirs(output_dir, exist_ok=True)
    lammps_path = os.path.join(output_dir, "pair_coeff.txt")
    with open(lammps_path, "w") as fh:
        fh.write("# Buckingham pair coefficients for LAMMPS\n")
        fh.write("# pair_style buck/coul/long <r_cut>\n\n")
        for pair, p in buck_params.items():
            parts = sorted(pair)
            s1 = parts[0];  s2 = parts[-1]   # same for homoatomic pairs
            fh.write(f"pair_coeff  {s1}  {s2}  "
                     f"{p['A']:.4f}  {p['rho']:.6f}  {p['C']:.4f}\n")
        fh.write("\n# Formal charges\n")
        for sp, q in charges.items():
            fh.write(f"# {sp}  {q:+.4f} e\n")
    print(f"LAMMPS snippet saved to {lammps_path}")


def save_fit_result_generic(result: dict,
                              potential : str,
                              output_dir: str,
                              diag      : dict,
                              extra_meta: Optional[dict] = None,
                              r_cut     : Optional[float] = None,
                              species   : Optional[list]  = None) -> None:
    """
    Save fitted parameters for any potential type to JSON.

    Produces ``fitted_potential.json`` (via :func:`~akitsushima.fitting.
    io_utils.save_potential`, so the file uses exactly the same schema
    that :func:`~akitsushima.fitting.io_utils.load_potential` /
    ``_load_potential_json`` expect -- see the bug-fix note below) and
    prints a short diagnostic.  For Buckingham, additionally writes the
    LAMMPS ``pair_coeff.txt`` snippet via
    :func:`export_lammps_buckingham`.

    Bug-fix note (schema consistency, found alongside S13): the original
    implementation hand-rolled a *second*, incompatible JSON schema here
    (top-level key ``"potential"`` instead of ``"potential_type"``, and
    ``charges`` left nested inside ``params`` instead of hoisted to the
    top level).  Every YAML config's default
    ``potential_json: ./fitting_output/fitted_potential.json`` -- the
    exact file this function produces -- is read back through
    ``_load_potential_json`` -> :func:`load_potential`, which requires
    the *other* schema (the one :func:`save_potential` writes).  Once
    S13's missing ``_load_potential_json`` is implemented, reloading a
    file saved by the original ``save_fit_result_generic`` would have
    raised ``KeyError('potential_type')`` immediately.  Delegating to
    ``save_potential`` here makes the write and read sides consistent.

    ``r_cut``, ``species`` : the cutoff and elements the fit used, stored
    so the file reloads to the same potential.

    ``extra_meta`` : fixed (not-fit) construction hyperparameters that
    are not part of the fitted ``params`` dict but are still needed to
    reconstruct an identical potential on load -- e.g. Mie's ``n``/``m``
    exponents or Yukawa's ``kappa`` screening length. Persisted verbatim
    into the JSON so :func:`~akitsushima.fitting.io_utils.load_potential`
    can read them back instead of silently falling back to a default
    that might not match what was actually fit.
    """
    out_path = os.path.join(output_dir, "fitted_potential.json")
    extra = {"diagnostics": diag}
    if extra_meta:
        extra.update(extra_meta)
    # r_cut and species go into the file: without them load_potential
    # fell back to r_cut = 10 A, and `potential_json:` silently built a
    # different potential from the one fitted (ADR-0003, "fit output
    # JSON"; a Born-Mayer / Buckingham fit at r_cut 5 A).
    save_potential(result, out_path, potential_type=potential,
                   r_cut=r_cut, species=species, extra=extra)

    if potential == "eam":
        # ADR-0015: the fitted Zhou EAM tabulated for LAMMPS (pair_style
        # eam/alloy) and for `potential: {type: EAM, setfl: ...}`.
        from ..engine.eam_zhou import write_setfl, zhou_eam
        p = result["params"]["eam"]
        sp = list(species) if species else list(p["elements"])
        pot = zhou_eam(p["elements"], sp, r_cut=p["r_cut"], chi=p["chi"])
        path = os.path.join(output_dir, "fitted.eam.alloy")
        write_setfl(path, pot, sp, comments=(
            "akitsushima fit: Zhou et al. (2004) form (ADR-0015)",
            "species " + " ".join(sp), ""))
        print(f"  setfl table   -> {path}")

    if potential == "buckingham":
        params = result.get("params", {})
        export_lammps_buckingham(
            params.get("buck", {}), params.get("charges", {}), output_dir)
