"""Potential-agnostic weighted sum-of-squared-residuals loss function.

``generic_loss_function`` is called tens of thousands of times per
``scipy.optimize`` run, so it is the highest-priority target for MPI
parallelisation in the fitting workflow: the training-structure list is
split across ranks (:func:`_mpi_domain_split`) and partial sums are
combined with :func:`_mpi_allreduce_sum_scalar`.  When MPI is inactive
(``_HAS_MPI`` is False) both helpers are no-ops and behaviour is
identical to the original single-process implementation.
"""
from typing import Optional

import numpy as np

from ..engine import compute_all_general
from ..utils.mpi import (
    _mpi_allgather_concat,
    _mpi_allreduce_sum_scalar,
    _mpi_domain_split,
    local_only,
)
from .parameter_sets import FittableParameterSet

__all__ = ["generic_loss_function"]


def _huber_of_squared(sq_value: float, delta: float) -> float:
    """
    Huber-transform a value that is ALREADY a squared residual (or a
    mean of squared residuals), treating ``r = sqrt(sq_value)`` (>= 0
    by construction) as the effective residual magnitude:

        rho(r) = r^2                  if r <= delta   (identical to the
                                                        plain-squared loss)
        rho(r) = 2 delta r - delta^2  if r >  delta   (grows linearly,
                                                        not quadratically)

    Used by :func:`generic_loss_function`'s optional ``loss_type="huber"``
    mode to temper the influence of a single badly-off structure on the
    total loss, applied at the same per-structure-per-category
    granularity the loss already reduces to (one number per structure
    per energy/forces/stress category), not per individual residual
    component.
    """
    r = np.sqrt(max(sq_value, 0.0))
    if r <= delta:
        return sq_value
    return 2.0 * delta * r - delta ** 2


def generic_loss_function(x            : np.ndarray,
                           param_set   : FittableParameterSet,
                           structures  : list,
                           weights     : dict,
                           r_cut       : float,
                           ewald_alpha : Optional[float],
                           coulomb_method: str = "ewald",
                           loss_type   : str = "squared") -> float:
    """
    Weighted sum-of-squared-residuals loss (eV^2-consistent formulation).

    All three terms are brought to the same physical unit (eV^2):

    .. math::

        L = \\sum_s w_{\\rm trust}(s) \\, c_s^2 \\left[
              w_E \\, N_s^2 \\,
                  \\left(E_s^{\\rm pred} - E_s^{\\rm ref}\\right)^2
            + w_F \\, r_0^2 \\,
                  \\overline{\\|\\mathbf{F}_s^{\\rm pred}-\\mathbf{F}_s^{\\rm ref}\\|^2}
            + w_S \\, (V_s\\,\\varepsilon_0)^2 \\,
                  \\overline{|\\boldsymbol{\\sigma}_s^{\\rm pred}
                               -\\boldsymbol{\\sigma}_s^{\\rm ref}|^2}
          \\right]

    Unit conversion
    ---------------
    ===========  ===========  ============  ===========================
    Quantity     Ref. unit    Scale factor  Resulting term unit
    ===========  ===========  ============  ===========================
    Energy       eV/atom      N_s           (N_s x deltaE_per_atom)^2 eV^2
    Force        eV/A         r_0 [A]       (r_0 x deltaF)^2 eV^2
    Stress       eV/A^3        V_s epsilon_0 [A^3]  (V_s epsilon_0 x deltasigma)^2 eV^2
    ===========  ===========  ============  ===========================

    Parameters
    ----------
    x             : optimisation vector.
    param_set     : any FittableParameterSet subclass.
    structures    : list of structure dicts.  Each dict must contain
                    ``ref_energy`` *in eV/atom*, ``natoms``, and optionally
                    ``ref_forces`` [eV/A] and ``ref_stress`` [eV/A^3].
                    ``structure["volume"]`` is used for the stress scale; if
                    absent it is computed from ``structure["lattice"]``.
                    Two further, OPTIONAL, independent per-structure keys:

                    - ``structure["scale"]`` (``c_s`` above, default 1.0)
                      -- a per-structure physical unit-bridging factor
                      (e.g. from :func:`~akitsushima.fitting.io_utils.
                      apply_scales_to_structures`), applied SQUARED,
                      matching :func:`~akitsushima.fitting.driver.
                      fit_potential`'s own documented loss formula.
                      Bug-fix note: this key existed (written by
                      ``make_scale_csv``/``apply_scales_to_structures``,
                      whose own docstring already claimed it was "then
                      squared inside generic_loss_function()") but was
                      never actually read here -- the whole per-structure
                      scaling mechanism was dead code. ``.get(..., 1.0)``
                      keeps every existing caller's behaviour identical.
                    - ``structure["weight"]`` (``w_trust`` above, default
                      1.0) -- a distinct, SUBJECTIVE per-structure trust
                      multiplier (e.g. down-weighting a lower-level-
                      theory reference structure relative to a DFT one),
                      applied LINEARLY, matching conventional weighted
                      least squares (``sum_i w_i . residual_i^2``) rather
                      than a squared unit-bridging factor. Do not confuse
                      the two: "scale" corrects units, "weight" expresses
                      trust.
    weights       : dict with keys ``'energy'`` (w_E), ``'forces'`` (w_F),
                    ``'stress'`` (w_S), ``'r0'`` (r_0 [A], default 1.0),
                    ``'eps0'`` (epsilon_0 dimensionless, default 1.0).
                    When ``loss_type='huber'``, must additionally contain
                    ``'huber_delta_energy'``, ``'huber_delta_forces'``,
                    ``'huber_delta_stress'`` (all in eV, the unit each
                    term is already converted to above) -- required
                    explicitly rather than silently defaulted, since a
                    wrong delta silently changes the loss landscape.
    r_cut         : short-range cutoff [A].
    ewald_alpha   : Ewald alpha (None = auto).
    coulomb_method: ``'ewald'`` | ``'pppm'``.
    loss_type     : ``'squared'`` (default; exact original behaviour) or
                    ``'huber'`` -- see :func:`_huber_of_squared`. Tames a
                    single badly-off structure (e.g. one non-converged
                    reference calculation) from dominating the sum over
                    potentially hundreds of structures; opt-in, fully
                    backward-compatible default.

    Returns
    -------
    float : loss value (unit eV^2, non-negative).

    MPI
    ---
    ``structures`` is partitioned across ranks via
    :func:`_mpi_domain_split`; each rank evaluates only its shard and the
    partial sums are combined with :func:`_mpi_allreduce_sum_scalar`. A
    failure (unphysical parameters raising inside the potential
    evaluation) on any single rank is propagated to all ranks so that
    every rank returns the same ``1e30`` sentinel, matching the
    fail-fast behaviour of the original single-process code.

    Bug-fix note: ``compute_all_general`` itself ALSO unconditionally
    splits its own internal pair/atom work across every rank on the same
    global communicator (see every engine potential's ``compute()``).
    Without :func:`~akitsushima.utils.mpi.local_only` below, each rank's
    per-structure call would try to Allreduce with every OTHER rank's
    call for a DIFFERENT structure in its own shard -- silently mixing
    together partial sums from unrelated structures (discovered via a
    bit-for-bit divergence between 1-rank and multi-rank runs of an
    analogous per-item MPI split elsewhere in the solvers). Wrapping the
    whole per-shard loop in ``local_only()`` makes each
    ``compute_all_general`` call fully local to its own rank, so the
    single, explicit reduction below is the ONLY cross-rank
    communication for the structure list.
    """
    if loss_type not in ("squared", "huber"):
        raise ValueError(
            f"generic_loss_function: unknown loss_type '{loss_type}'. "
            "Choose 'squared' or 'huber'.")
    if loss_type == "huber":
        for key in ("huber_delta_energy", "huber_delta_forces", "huber_delta_stress"):
            if key not in weights:
                raise ValueError(
                    f"generic_loss_function: loss_type='huber' requires "
                    f"weights['{key}'] (eV) -- no silent default is guessed.")

    try:
        params = param_set.decode(x)
    except Exception:
        return 1e30

    qs     = params.get("charges")
    engine = param_set.build_engine(params)
    w_E  = weights.get("energy", 1.0)
    w_F  = weights.get("forces", 1.0)
    w_S  = weights.get("stress", 0.1)
    r0   = float(weights.get("r0",   1.0))
    eps0 = float(weights.get("eps0", 1.0))

    i0, i1 = _mpi_domain_split(len(structures))
    local_total  = 0.0
    local_failed = False

    with local_only():
        for s in structures[i0:i1]:
            N   = s["natoms"]
            vol = float(s.get("volume",
                              abs(np.linalg.det(s["lattice"]))))
            try:
                e, f, st = compute_all_general(
                    s["lattice"], s["positions"], s["atom_types"],
                    engine,
                    charges        = qs,
                    r_cut          = r_cut,
                    ewald_alpha    = ewald_alpha,
                    coulomb_method = coulomb_method,
                )
            except Exception:
                local_failed = True
                break

            struct_total = 0.0

            # E: ref is eV/atom -> multiply by N_s to get eV
            dE_per_atom = (e / N) - s["ref_energy"]   # ref already eV/atom
            e_sq = (N * dE_per_atom) ** 2
            if loss_type == "huber":
                e_sq = _huber_of_squared(e_sq, weights["huber_delta_energy"])
            struct_total += w_E * e_sq

            # F: eV/A x r_0[A] -> eV
            if s["ref_forces"] is not None and f is not None:
                f_sq = (r0 ** 2) * float(np.mean((f - s["ref_forces"]) ** 2))
                if loss_type == "huber":
                    f_sq = _huber_of_squared(f_sq, weights["huber_delta_forces"])
                struct_total += w_F * f_sq

            # S: eV/A^3 x V_s[A^3] x epsilon_0 -> eV
            if s["ref_stress"] is not None and st is not None:
                c_s = vol * eps0
                s_sq = (c_s ** 2) * float(np.mean((st - s["ref_stress"]) ** 2))
                if loss_type == "huber":
                    s_sq = _huber_of_squared(s_sq, weights["huber_delta_stress"])
                struct_total += w_S * s_sq

            # Fix (scale-wiring): per-structure unit-bridging factor,
            # squared -- see the docstring's "structure['scale']" note.
            c_scale = float(s.get("scale", 1.0))
            # Fix (per-structure trust weight): distinct from "scale",
            # applied linearly -- see the docstring's "structure['weight']" note.
            w_trust = float(s.get("weight", 1.0))
            local_total += w_trust * (c_scale ** 2) * struct_total

    failed_anywhere = _mpi_allreduce_sum_scalar(1.0 if local_failed else 0.0)
    if failed_anywhere > 0.0:
        return 1e30

    return _mpi_allreduce_sum_scalar(local_total)


def residual_length(structures: list) -> int:
    """Length of the vector generic_residuals returns for these structures."""
    n = 0
    for s in structures:
        n += 1
        if s.get("ref_forces") is not None:
            n += 3 * int(s["natoms"])
        if s.get("ref_stress") is not None:
            n += 9
    return n


def generic_residuals(x            : np.ndarray,
                      param_set   : FittableParameterSet,
                      structures  : list,
                      weights     : dict,
                      r_cut       : float,
                      ewald_alpha : Optional[float],
                      coulomb_method: str = "ewald") -> np.ndarray:
    """
    Residual vector r with sum(r**2) == generic_loss_function(...) for the
    squared loss (ADR-0015; used by local_method 'least_squares').

    Per structure, in order: sqrt(w_E) N dE_per_atom; sqrt(w_F / 3N) r0
    dF_ia (all 3N components); sqrt(w_S / 9) V eps0 dsigma_ab (all 9) --
    each multiplied by sqrt(w_trust) c_scale, exactly the terms of the
    loss.  Under MPI every rank evaluates its shard and the pieces are
    concatenated in rank order, so all ranks hold the same vector.  If any
    structure fails to evaluate, every entry is 1e15 (sum of squares
    ~1e30 per entry, the loss's failure value).
    """
    n_total = residual_length(structures)
    try:
        params = param_set.decode(x)
    except Exception:
        return np.full(n_total, 1e15)
    qs = params.get("charges")
    engine = param_set.build_engine(params)
    w_E = weights.get("energy", 1.0)
    w_F = weights.get("forces", 1.0)
    w_S = weights.get("stress", 0.1)
    r0 = float(weights.get("r0", 1.0))
    eps0 = float(weights.get("eps0", 1.0))
    i0, i1 = _mpi_domain_split(len(structures))
    local, failed = [], False
    with local_only():
        for s in structures[i0:i1]:
            N = s["natoms"]
            vol = float(s.get("volume", abs(np.linalg.det(s["lattice"]))))
            try:
                e, f, st = compute_all_general(
                    s["lattice"], s["positions"], s["atom_types"], engine,
                    charges=qs, r_cut=r_cut, ewald_alpha=ewald_alpha,
                    coulomb_method=coulomb_method)
            except Exception:
                failed = True
                break
            c = np.sqrt(float(s.get("weight", 1.0))) * float(s.get("scale", 1.0))
            local.append([c * np.sqrt(w_E) * N * ((e / N) - s["ref_energy"])])
            if s.get("ref_forces") is not None:
                local.append(c * np.sqrt(w_F / (3.0 * N)) * r0
                             * (np.asarray(f) - s["ref_forces"]).ravel())
            if s.get("ref_stress") is not None:
                local.append(c * np.sqrt(w_S / 9.0) * vol * eps0
                             * (np.asarray(st) - s["ref_stress"]).ravel())
    if _mpi_allreduce_sum_scalar(1.0 if failed else 0.0) > 0.0:
        return np.full(n_total, 1e15)
    vec = np.concatenate(local) if local else np.zeros(0)
    return _mpi_allgather_concat(vec)
