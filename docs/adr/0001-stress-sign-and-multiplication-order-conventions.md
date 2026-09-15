---
status: accepted
---

# Stress sign, cell-gradient order, and instantaneous-pressure sign

This package defines stress as `sigma_ab = -(1/V) dE/dT_ab` under the affine
deformation `lattice' = lattice @ T` (right multiplication, no transpose) with
`positions' = frac @ lattice'`. Three consequences of that choice look like sign
or transpose errors to a reader applying the more common continuum-mechanics
convention, and each has already been either mis-flagged or wrongly "corrected"
at least once, so they are recorded here deliberately.

1. **A compressed cell has POSITIVE diagonal stress.** For `E(s) ~ (s-1)^2`,
   compression gives `dE/ds < 0`, so `sigma = -(1/V) dE/dT` is positive. The
   intuition "compression means negative stress" belongs to the opposite
   convention and does not apply here.
2. **`NPTSimulation._pressure_inst` uses `Pv = +trace(sigma)/3`**, not the
   textbook-looking `-trace(sigma)/3`. The negative form is a real bug that was
   shipped once: it made the MTTK barostat drive a compressed cell to collapse
   further (volume 543 -> 1.72e6) and an expanded cell to expand further,
   because `G_eps = 3V(P_inst - P_ext)` was fed a pressure of the wrong sign.
3. **The cell gradient is `f_cell = vol * (h_inv_T @ stress)` with `h_inv_T` on
   the LEFT** (`akitsushima/engine/cell_gradient.py`). The transposed order
   `stress @ h_inv_T` was the original implementation and is wrong: finite
   differencing all nine lattice components against `-dE/dh` gives a maximum
   discrepancy of 17.0 for the transposed order versus 1.24e-7 for this one.
   The two agree for orthogonal cells, which is why the error survived.

By contrast `cell_update_scheme='vasp'` uses `f_cell = stress @ lattice` with no
sign flip, and that IS correct: it was checked experimentally by compressing and
expanding a relaxed FCC Al cell and confirming the cell moves back toward
equilibrium in both directions. Reasoning indirectly from VASP's own opposite
stress-sign convention predicted a flip was needed, and that prediction was
wrong -- which is the reason this ADR exists rather than a code comment.

## Consequences

Every future change to a force, stress, cell-gradient, or barostat expression is
validated against a finite-difference oracle on a randomly distorted,
low-symmetry cell before it is believed. Symmetric equilibrium structures cannot
discriminate: the forces are zero there, so a sign error is invisible.
`tests/test_engine_forces.py` holds the oracle (`finite_diff_stress`), and
`tests/test_solvers.py` holds the physical-direction regressions for the `vasp`
scheme and for the NPT barostat.
