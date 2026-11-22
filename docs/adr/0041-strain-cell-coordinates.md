---
status: accepted (2026-11-22, 0.3.0a5)
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

## Result (2026-11-22, 0.3.0a5)

### Implementation (`akitsushima/solvers/geometry_optimiser.py`)

- **New argument and key:**
  - `cell_coordinates='strain' | 'lattice'` (default `strain`);
  - YAML `relax.cell_coordinates` in `relax` and `anneal`; `qha`,
    `gruneisen` and the elastic pre-relaxation use the default.
- **Reference and constraints:**
  - `_set_reference()` stores h0, h0^-1, s = |det h0|^(1/3) and the
    constraint projector P (SVD of the frozen-component rows).
  - It is called in `__init__`, at the start of `run()` (after a
    checkpoint restore), and so h0 is the cell the run starts from.
- **Coordinates and gradient:**
  - `_get_x`: r0 = r h^-1 h0, x_cell = s (F - I).
  - `_set_x`: projects x_cell with P, builds h = h0 F^T, resets frozen
    entries to h0, carries the fractional coordinates, then applies the
    `fix_volume` retraction.
  - `_generalised_force`: [f F, P(f_h^T h0) / s] from the shared
    `cell_force_from_stress`, followed by the `fix_volume` projection.
  - `_apply_generalised_displacement` adds the step in x.
  - In strain mode the cell block of `_free_mask` is all free, because
    constraints act by projection.
- **RFO:** `_rotation_basis()` gives the infinitesimal rotations allowed
  by the constraints.  `_run_rfo` replaces them in the model Hessian by a
  positive curvature, removes them from the gradient and from the step.
  This was the cause of the RFO failure (finding 1).
- **`vasp`:** `lattice @ stress` on the lattice path.  With strain
  coordinates it has no effect and gives a DeprecationWarning.
- **Checkpoint:**
  - `cell_coordinates` is in the restart settings when `relax_cell` is
    on; a missing key counts as `lattice`.
  - A mismatch raises a ValueError that names `cell_coordinates:
    lattice`.
- **Records:**
  - the result dict has `cell_coordinates` (None without relax_cell);
  - relax.log ends with `Cell coordinates: strain`;
  - the verbose console banner prints it.
  - It is written at the end of relax.log, not at the top: the first
    line of relax.log is the column header that the restart code and
    several readers skip.
- **Version:** 0.3.0a5.

### Gates

| Gate | Result |
|---|---|
| Gradient vs finite differences of E(x), 64-atom sheared rattled Si, free / constraints / fix_volume / fix_volume + fix_cell | max rel. error 3e-8 .. 4e-7 |
| `_set_x` then `_get_x` | exact (0.0) |
| Symmetry after relaxation from a strained lattice, 6 methods | Si fcc-prim 3x3x3: Fd-3m; AlN 3x3x2: P6_3mc (all 12) |
| fix_volume (AlN, FIRE) | P6_3mc, volume to 1e-12 |
| Old path (`cell_coordinates: lattice`), 16-case matrix | bit for bit, except `vasp` (intended fix) |
| Frozen components (constraints, fix_cell, ESM a_x b_y) | bit for bit |
| Every case the old path converged still converges | yes (table below, GeSn table) |
| qha / gruneisen CLI tests at default symprec 1e-5 | Fd-3m (were R-3m) |
| New tests | 22, ~2 s in total, none marked slow (RMM-DIIS symmetry case at fmax 1e-4 / pmax 1e-3, finding 4) |

**Space groups at symprec 1e-5, same matrix, old path:**

| Method | Si | AlN |
|---|---|---|
| fire | R-3m | Cmc2_1 |
| lbfgs | R-3m | Cm |
| sd | R-3m | Cmc2_1 |
| diis | R-3m | Cm |
| rmmdiis | R-3m | Cmc2_1 |
| rfo | R-3m | P1 (not converged) |

**Step counts.**  Tersoff, fmax 1e-3 eV/A, pmax 1e-2 GPa, max 400
steps.  The starting cells are strained: Si a = 5.50 A; AlN a = 3.30,
c = 5.40, u = 0.38.

| Case | Old | Strain | dE (eV, strain - old) |
|---|---|---|---|
| Si fire | 52 | 32 | -3.6e-6 |
| AlN fire | 61 | 66 | -6.0e-6 |
| Si lbfgs | 21 | 4 | +5.0e-7 |
| AlN lbfgs | 17 | 7 | -4.0e-6 |
| Si sd | 21 | 7 | -5.4e-6 |
| AlN sd | 61 | 14 | -8.7e-7 |
| Si diis | 16 | 5 | -8.8e-7 |
| AlN diis | 33 | 24 | -3.1e-7 |
| Si rmmdiis | 36 | 17 | -2.0e-7 |
| AlN rmmdiis | 48 | 21 | -1.5e-6 |
| Si rfo | 9 | 3 | -4.8e-6 |
| AlN rfo | 400 (not conv.) | 4 | -5.4e-2 |
| AlN fire + fix_volume | 47 | 32 | -5.8e-7 |
| AlN lbfgs + vasp | 27 | 7 | -1.9e-6 |
| Si fire + constraints `1 1 0 1 1 0 0 0 1` | 400 (not conv.) | 400 (not conv.) | 0 |
| Si lbfgs + fix_cell (anti-diagonal) | 400 (not conv.) | 400 (not conv.) | +4.0e-6 |

The two constrained cases never meet pmax on either path.  The stress
along frozen components is not relaxed, and the run ends at the same
constrained minimum.

**GeSn 63-atom vacancy cell** (fitted SW, `tests/data/fire_gesn`,
nearly cubic): fmax 1e-3, pmax 1e-2, max 1000 steps.

| Method | Old | Strain |
|---|---|---|
| fire | 234 | 247 |
| lbfgs | 106 | 74 |
| sd | 405 | 463 |
| diis | 909 | 486 |
| rmmdiis | 397 | 390 |
| rfo | 1000 (not conv.) | 57 |

Some runs end in a different local minimum of the vacancy cell
(-251.673 vs -251.628 eV).  The old path splits the same way between
methods.

### Findings

1. **RFO: the cell rotations have negative curvature.**
   - In linear-F coordinates a move along W F is a rotation only to
     first order; (I + eps W)^T (I + eps W) = I - eps^2 W^2 is a stretch.
     The curvature along the three rotations is proportional to the
     stress: -0.92, -0.72, -0.72 eV for the strained AlN.
   - RFO follows these modes.  The stretch then broke the symmetry
     (AlN converged in 18 steps, but to P1, off by 1e-4 A), independent
     of `hessian_delta`.
   - With the rotations removed: 4 steps, P6_3mc at 1e-5.  The residual
     gamma - 120 deg (3.6e-7, 3.6e-9, 3.6e-11 for hessian_delta 1e-2,
     1e-3, 1e-4) scales as delta^2: ordinary FD truncation.
   - The first-order methods need nothing, because the gradient is
     orthogonal to rotations.
   - The same mechanism (raw-h rotations under stress) is the likely
     reason RFO + relax_cell never converged on the old path.  That
     path is left unchanged, as decided (bit for bit).
2. **Residual rigid rotation.**  Relaxing the 64-atom sheared Si cell
   stretches it by +21 / -10 / -12 %.  The polar rotation of
   h_new^T h_old^-T is:
   - 0.255 deg (fire) and 0.290 deg (lbfgs) with strain coordinates;
   - 0.228 / 0.210 deg on the old path.

   It is second order in the strain along a curved path, harmless, and
   of the same size as before.  For a symmetric crystal F stays in the
   symmetric subspace and no rotation arises.  The "0.003-0.008 deg"
   quoted in decision 8 was for a smaller shear.
4. **RMM-DIIS amplifies symmetry-breaking noise.**
   - The full suite runs with 10 numba threads.  There Si + RMM-DIIS
     ended in P1 at 1e-5, with 3.4e-6 of shear in F.  With one thread it
     ended in Fd-3m, with 4e-9 of shear.
   - The 10-thread force sums are deterministic but slightly
     asymmetric (about 1e-15).
   - A 1e-10 shear seed on a cubic start (one thread) peaked at:

     | Method | Strain | Old path |
     |---|---|---|
     | rmmdiis | 3.9e-6 | 1.2e-2 |
     | sd | 2.6e-9 | 1.2e-2 |
     | fire, lbfgs, diis | no growth | 3.6e-3 to 1.2e-2 |

   - RMM-DIIS accepts any step that lowers the energy, and the soft
     modes dominate that energy change.  A stiff shear mode can then
     overshoot (|1 - alpha k| > 1) and grow until the soft modes have
     converged.  At fmax 1e-3 / pmax 1e-2 GPa the run stopped
     mid-growth: 3.4e-6 of shear is about 1.7e-4 GPa, inside pmax.
   - At fmax 1e-4 / pmax 1e-3 the 10-thread run gives Fd-3m (AlN
     P6_3mc at every tolerance).  The symmetry test runs RMM-DIIS at
     those tolerances.
   - The cause is in the method, not in the coordinates: the old path
     amplified the seed 10^4 times more.  A curvature-aware step or a
     cell preconditioner would remove it.  That is left for later.
5. **Mapping tolerance (decision 6).**  Measured residuals up to 1.53
   symprec; `MAPPING_FACTOR = 3`.  See the ADR-0040 addendum.

### Existing tests changed

- `tests/test_solvers.py`: five tests of the raw-lattice internals now
  pass `cell_coordinates="lattice"`.
  - The two `_generalised_force` dE/dh tests;
  - `test_set_x_round_trips_only_when_the_cell_does_not_move`;
  - `test_rfo_variable_cell_rereads_its_state_vector...` (it re-injects
    the non-pure `_set_x` defect);
  - `test_vasp_cell_scheme_drives_volume...`.

  Their reference numbers are unchanged.
- `tests/test_symmetry_reduction.py`: the qha / gruneisen CLI tests no
  longer set symprec 1e-4.  The mapping error message match was updated,
  and one tolerance test was added (ADR-0040 addendum).
- No test needed a new reference value.  Every other default
  `relax_cell` test passed with strain coordinates within its existing
  tolerance.

## Consequences

- Default variable-cell relaxations of non-orthogonal cells keep their
  space group, so symmetry reduction (ADR-0040) works on relaxed
  structures without rebuilding them by hand.
- Default `relax_cell` numbers change at the tolerance level.  Anyone
  who needs the old numbers sets `cell_coordinates: lattice`.
- Old checkpoints continue only with `cell_coordinates: lattice`.
