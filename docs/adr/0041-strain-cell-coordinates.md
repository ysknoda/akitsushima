---
status: proposed (2026-11-22)
---

# Strain coordinates for variable-cell relaxation

Follow-up of ADR-0040.  The variable-cell relaxations in the ADR-0040
work broke the symmetry of perfect crystals by about 1e-5 A.  This ADR
records the diagnosis and the fix.  Agreed with the user on 2026-11-22
(grilling, two rounds, Q1-Q12, all as recommended).

## Problem

`GeometryOptimiser(relax_cell=True)` uses the 9 raw lattice components
h (rows = lattice vectors) as the cell coordinates.

- The default `generalized_force` returns f_cell = V h^-T sigma.  This
  is the true dE/dh.  But a step in raw-h space depends on the basis.
  For a non-orthogonal basis, an isotropic stress gives an anisotropic
  step:
  - fcc primitive cell: 60 deg -> 67.8 deg in one test step;
  - hexagonal cell: gamma 120 deg -> 107.5 deg.
- The runs still converge, but they leave a symmetry-breaking strain
  inside the stress tolerance (pmax):
  - Si fcc primitive 4x4x4, FIRE: angle 60.000783 deg.  Through the
    internal strain the atoms move about 5e-6 A along [111], and spglib
    finds R-3m at symprec 1e-5.
  - AlN wurtzite, FIRE: gamma 120.000192 deg, found as Cmc2_1.
  - Orthogonal cubic cells are unaffected (3e-15 A).
- **The `vasp` scheme has the wrong operand order.**  It uses
  `stress @ lattice`.  VASP's update A <- A + sigma A has lattice
  vectors as columns, so with rows it is `lattice @ stress`.  As
  written, the scheme is symmetric only for an isotropic stress.
  AlN with L-BFGS and `vasp` gave Cmc2_1.
- **RFO + relax_cell does not converge** in 5000 steps (Si fcc
  primitive and AlN), with either scheme.
- **The ADR-0040 mapping check is too strict.**  On a slightly
  distorted AlN, `spglib_operations` raised "does not map every atom":
  spglib accepted an operation at symprec 1e-5 that the strict 1e-5
  atom mapping rejected.

These matter because symmetric relaxed structures are what the
ADR-0040 savings need, and they are the clean starting structures for
phonons and elastic constants.

## Fix experiment (2026-11-22)

The force was patched to a covariant `h @ sigma` scaled by V^(1/3):

- FIRE keeps Fd-3m and P6_3mc, and Si converges in 8 steps instead
  of 29.
- L-BFGS breaks (P1, 105 steps).  The patched force is no longer the
  gradient of the h coordinates, so L-BFGS pairs coordinate changes and
  gradient changes that do not belong together.

So the fix is a change of coordinates, not of the force alone.

## Decisions

1. **Strain coordinates become the default (Q1).**
   - New key `cell_coordinates: strain | lattice`, default `strain`.
   - `lattice` keeps the old raw-h path.  With `generalized_force` it is
     bit for bit the old code.
   - This changes the default `relax_cell` results within the
     convergence tolerances: an exception to ADR-0028 decision 5.  The
     qha and elastic-CLI pre-relaxations shift slightly too.

2. **Linear deformation gradient (Q2).**
   - Reference cell h0 = the starting cell.  h = h0 F^T, and the ions
     are carried as r = r0 F^T.
   - Coordinates: x = [r0, s (F - I)].
   - Gradient: dE/dr0 = -f F, and dE/dF = -V sigma F^-T (from dE/dh at
     fixed fractional coordinates).  It is computed from the shared
     `cell_force_from_stress` as f_F = f_h^T h0.
   - sigma and F are both invariant under the crystal's point group, so
     the gradient stays in the symmetric subspace, for every optimiser.
   - `_set_x` becomes a pure function of x.
   - log F (ASE FrechetCellFilter) is not used: the strains in a
     relaxation are a few per cent at most, and log F needs Frechet
     derivatives of the matrix logarithm.

3. **Constraints keep their raw-h meaning (Q3).**
   - `lattice_constraints` (9 components) and `fix_cell` (3x3) still
     freeze raw lattice components.
   - In F space a frozen h_ab is the linear constraint
     sum_c h0[a, c] dF[b, c] = 0.  It is constant because h0 is fixed,
     so it is consistent with L-BFGS.  It is applied by orthogonal
     projection of the cell gradient and of every cell step.
   - Frozen components are reset to their h0 values after every update,
     so they stay bit for bit unchanged.
   - The ESM requirement `'1 0 0 0 1 0 0 0 0'` keeps its meaning.

4. **Scale s = V0^(1/3) (Q4).**
   - x_cell then has units of length.  The 0.2 A step cap
     (`max_displacement`) acts on it as it did on raw h.
   - No new key.

5. **RFO is part of this ADR (Q5).**  It is re-measured with strain
   coordinates.  If it still does not converge, the cause is found and
   fixed, or `method: rfo` with `relax_cell` is refused with a clear
   error.

6. **ADR-0040 mapping tolerance (Q6).**
   - The atom mapping accepts distances up to c * symprec.  c is set
     from the measured residual of the failing AlN case (about 2-3).
   - The mapping must be a bijection.
   - The largest residual is reported in the symmetry summary.
   - This goes in a separate commit, with an addendum to ADR-0040.

7. **Packaging (Q7).**  This ADR, the implementation in a worktree,
   release 0.3.0a5.  Push and tag only after asking.

8. **All 9 components of F; no rotation projection (Q8).**
   - E(QF) = E(F), so the gradient is always orthogonal to the rotation
     direction, and nothing drives a rotation.
   - The residual rotation is measured and recorded.  Old path: 0.003
     to 0.008 deg.

9. **`cell_update_scheme` (Q9).**
   - On the `lattice` path, `vasp` is fixed to `lattice @ stress`.
     This is a bug fix of an opt-in path.
   - With strain coordinates the scheme has no effect.  `vasp` there
     gives a DeprecationWarning, and it is removed at the next minor
     release.
   - Note that the corrected `vasp` step at F = I points the same way
     as the strain-coordinate gradient.

10. **`fix_volume` (Q10).**
    - The cell gradient is projected off the volume normal
      d(det F)/dx = det F F^-T / s, jointly with the constraint rows.
    - After each update, F is rescaled by (V0/V)^(1/3).
    - The stress test uses the deviatoric stress, as before.

11. **Checkpoint restart (Q11).**
    - On a restart h0 is the restored cell (F = I).  The optimiser state
      starts afresh anyway, and the constraints do not depend on h0.
    - `cell_coordinates` is added to the settings a restart must match.
      A checkpoint without it counts as `lattice`.
    - A mismatch is refused, with a message that suggests
      `cell_coordinates: lattice`.

12. **Gates (Q12).**
    - **Gradient:** finite differences of E(x) on a low-symmetry,
      non-orthogonal cell, also with constraints and `fix_volume`;
      relative error about 1e-6.
    - **Symmetry:** Si fcc primitive and AlN wurtzite, relaxed from a
      strained lattice with all 6 methods (fire, lbfgs, sd, diis,
      rmmdiis, rfo).  spglib must still find Fd-3m and P6_3mc at
      symprec 1e-5.
    - **Old path:** `cell_coordinates: lattice` reproduces the previous
      results bit for bit.
    - **Existing tests:** changed references are updated if the new
      value is within the convergence tolerance, and each one is listed
      here.  Anything else is investigated.
    - **Step counts:** a table, old against new.  Every case the old
      path converged must still converge.
    - **ADR-0040 end to end:** the qha and gruneisen CLI tests go back
      to the default symprec 1e-5.
    - **Constraints:** frozen components stay bit for bit, also with
      ESM.
    - **Recording:** `cell_coordinates` is written in the relax.log
      banner and in the result dict.
    - **Test budget:** new slow tests take 60 s or less in total.

## Consequences

- Default variable-cell relaxations of non-orthogonal cells keep their
  space group, so symmetry reduction (ADR-0040) works on relaxed
  structures without rebuilding them by hand.
- Default `relax_cell` numbers change at the tolerance level.  Anyone
  who needs the old numbers sets `cell_coordinates: lattice`.
- Old checkpoints continue only with `cell_coordinates: lattice`.
