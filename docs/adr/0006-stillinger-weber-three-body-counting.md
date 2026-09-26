---
status: accepted
---

# Stillinger-Weber: each three-body angle counted once, legs from their pair records

Until 2026-09-25 every Stillinger-Weber code path summed the three-body
term over both orders (j, k) and (k, j) of each neighbour pair around a
centre, with no factor 1/2, and a comment claimed that literature SW
parameters assume this convention.  They do not.  Stillinger & Weber
(PRB 31, 5262 (1985), eq. 5) write Phi3 = sum_{i<j<k} of three h terms,
one per centre, so each (centre, unordered neighbour pair) enters once;
LAMMPS `pair_sw` loops `jj < kk` the same way.  The engine's three-body
term was therefore twice Stillinger-Weber's (an effective lambda of 42
for Si).  Found during ADR-0005 when the elastic constants of the
reference potential were computed.

## Evidence

64-atom diamond Si at a = 5.430950 A, SW Si parameters (epsilon 2.1683,
sigma 2.0951, a 1.80, lambda 21.0, gamma 1.20, A 7.049556277,
B 0.6022245584, p 4, q 0):

| | C11 | C12 | C44 (ions relaxed) | B [GPa] |
|---|---|---|---|---|
| engine before the fix | 201.4 | 51.4 | 90.5 | 101.4 |
| engine before the fix, lambda halved | 151.4 | 76.4 | 56.5 | 101.4 |
| engine after the fix | 151.4 | 76.4 | 56.4 | 101.4 |
| published SW Si (Cowley, PRL 60, 2379 (1988); LAMMPS examples/ELASTIC) | 151.4 | 76.4 | 56.4 | 101.4 |

The published values are quoted from memory of those sources and were
not re-read here; the agreement to four digits is itself the check.
Energy finite differences give the same C11 and C12 as
`ElasticConstantsCalculator`, so the calculator is not at fault.  The bulk
modulus could never have caught the error: hydrostatic strain keeps every
angle tetrahedral, where the three-body term vanishes to second order.
A brute-force energy written from the paper's definition (no neighbour
list, no kernels) differed from the engine by exactly one extra
three-body sum on a rattled cell, and now agrees to 1e-10.

## Decisions

1. **Single species** (numpy kernel, numba parallel kernel, Python path):
   the inner loop runs over k > j.  h is symmetric in j and k for one
   species, so this is the paper's definition and halves the cost.
2. **Multi-species** (numba kernel, Python path): both orders are still
   visited, each with its own (i, j, k) record, and each weighs 1/2, so
   an unordered pair contributes (h_ijk + h_ikj) / 2.  That equals
   Stillinger-Weber's h (and LAMMPS) when the two records agree, and does
   not depend on neighbour order when they do not (LAMMPS would use the
   record of whichever neighbour comes first).
3. `tests/test_sw_three_body_counting.py` pins it: brute-force energies
   for single species and for the Tomita GeSn set, two labels with
   identical records equal to single species, forces and stress against
   finite differences (both numba and numpy), and SW Si C11 / C12 / C44
   = 151.4 / 76.4 / 56.4 GPa (+/- 0.5).
4. **A behaviour change**: every SW energy, force and stress changes
   except those of perfect tetrahedral structures; listed in the next
   release notes.

5. **Multi-species legs follow LAMMPS** (decided 2026-09-25 from Tomita
   et al., Jpn. J. Appl. Phys. 57, 04FB04 (2018), PDF in `references/`).
   Their eq. (2) sums the three-body term over sum_i sum_{j != i} sum_{k > j}
   (each unordered pair once), eq. (4) gives each leg its own pair's
   gamma, sigma, a, and the text states that Table I was "optimized for
   LAMMPS", whose `pair_sw` takes leg i-j's sigma, a, gamma from
   (i, j, j) and only lambda, epsilon, cos theta0 from (i, j, k).  The
   engine used to apply a mixed row's own sigma, a, gamma to both legs,
   which is a different potential for this table (mixed rows have sigma
   2.337 A against Ge-Ge 2.181 A, and gamma 1.8136 / 0.7232 that LAMMPS
   never reads).  Now: legs from the pair records; a mixed row supplies
   lambda, epsilon, cos theta0.  The (i, j, k) and (i, k, j) lambdas are
   equal in every row pair of Table I, so the (h_ijk + h_ikj) / 2 of
   decision 2 is exactly LAMMPS for this set.  (The paper's eq. (4)
   prints (cos theta + cos theta0)^2 with cos theta0 = -1/3, which would
   favour 109.5 deg only with a minus sign; LAMMPS's (cos theta - cos
   theta0)^2 is kept.)
   `MultiSpeciesStillingerWeberParameterSet` fits accordingly: a mixed
   row contributes one lambda variable per pair of rows (only their mean
   is determined); its sigma, a, gamma are no longer variables
   (29 -> 19 variables for GeSn).
   Tests: brute force with per-leg parameters, and a mixed row's sigma,
   a, gamma have no effect while its lambda does.

## What must be re-run or re-labelled

Results computed with the engine before this fix describe a potential
with twice SW's three-body term.  Structures that stay perfectly
tetrahedral (ideal diamond; its lattice constant 5.430950 A, set by the
two-body term) are unaffected.

| result | effect | action |
|---|---|---|
| ADR-0003 T2 `relax` GeSn SW; `md` GeSn SW (NVT/NVE/NPT) | internal-consistency checks (relax vs scipy on the same potential, T, drift, P) remain valid logic; the numbers change | re-run `example/check_relax.py`, `example/check_md.py` (GeSn rows) |
| `example/Si` relax (six optimisers, +/-5% volume) | ideal diamond, uniformly scaled: three-body zero; unaffected | none (spot-check) |
| ADR-0003 T5 `neb` / `tsa`, SW Si vacancy (barrier 0.949 / 0.909 eV) | barrier, modes and prefactor change; the checks are internal | re-run `example/neb/check_neb_tsa.py`; re-state the barrier |
| ADR-0003 T8 / `example/fit/Si_sw`, `example/fit/GeSn_sw` | training labels in `training_set/structures.csv` were made with the old engine; fitting them with the new engine would "recover" lambda ~ 2x; the GeSn fit config fits mixed-row sigma/gamma, which are no longer variables | regenerate the training sets (`make_training_set.py`), re-fit, re-check |
| `example/fit/GeSnVac_SW` | VASP data (OUTCAR, vasprun.xml), not labelled by the engine | none |
| ADR-0004 SW-Si MLIP verification (`example/mlip/adr0004`) | re-run 2026-09-26 (data regenerated); decisions confirmed; exposed and fixed a Bayesian-ridge degeneracy | done: ADR-0004 appendix, "Re-run after ADR-0006" |
| ADR-0005 comparison (`example/mlip/adr0005`) | re-run 2026-09-26 with both fixes; the ELM out-of-distribution gap of the first run closed | done: ADR-0005 appendix rewritten |
| `tests/` | no existing test pinned a three-body-dependent SW number (all 470 pass unchanged; FD force/stress tests included) | new regression tests added |

## Re-run of the SW examples (2026-09-26)

`example/rerun_sw_examples.sh`; outputs under each example directory.

| example | result | reading |
|---|---|---|
| GeSn relax (T2) | PASS (E -313.4005 eV, was -312.98; within 5e-6 eV/atom of scipy, RMSD 1e-3 A) | |
| GeSn md (T2), 20 ps examples | NVE PASS (drift 0.022 meV/atom/ns); NVT 900 K PASS; NVT 300 / 600 K Var(K) ratio 1.52 / 1.32 (FAIL, 20% band); NPT 0 GPa PASS (B ratio 1.11); NPT 5 GPa B ratio 0.39 (FAIL) | the 20 ps NVT runs repeat ADR-0003's AlN finding (insufficient equilibration, not a thermostat defect) until shown otherwise; per-run B has little power (ADR-0003) |
| GeSn NPT 200 ps (T2 powered test) | PASS: B_T 62.0 +/- 6.8 GPa vs static 60.1, ratio 1.03 (was 1.23 +/- 0.06, marginal) | |
| GeSn fit (T8, 19 variables) | PASS: unseen compositions E MAE 0.019 meV/atom, F RMSE 1.6e-3 eV/A, S MAE 1.4e-3 GPa; P = 0 lattice constants within 0.004%; two-body parameters within 0.15%; angular prefactors within 2.8% (lambda and leg gamma trade off individually, up to 12%) | |
| Si fit (T8 SW), first re-run | **FAIL**: differential evolution settled at sigma 7.70 A (cutoff 13.9 A in a 10.5 A cell), loss 771 (was 3.3e-6) | the engine's O(N^2) path evaluated a cutoff past half the cell width without error (minimum image drops the other images), so the global search found a spurious minimum there; loss at the true parameters is 7e-15 |
| Si fit (T8 SW), after the minimum-image check (below) | PASS with L-BFGS-B: parameters within 7.7e-3, loss 6.6e-5 | DE alone now reaches 6.6e-5 (it stalled at 6.47 in the first fit before this ADR) and takes 6 min instead of 27, since oversized cutoffs fail at once; L-BFGS-B then stopped after one iteration ("relative reduction <= factr*epsmch", gradient norm ~8) in a narrow, curved valley |
| Si fit (T8 SW), Nelder-Mead local step (new default, below) | PASS: loss 7.2e-15 (7e-15 at the reference); sigma / A / B / lambda / gamma within 7e-11 / 3e-10 / 4e-10 / 1.5e-8 / 4e-9; held-out \|dE\| max 2.9e-7 meV/atom, F RMSE 2.9e-9 eV/A; a_eq and B exact to the printed digits. Control (old stress sign) loss 2.1216, parameters 1-6% off, as intended; 6.5 min for both | |
| GeSn fit (T8, 19 variables), Nelder-Mead local step | PASS: loss 2.7e-14; every sigma, A, B, lambda and leg gamma equal to the reference to the printed 4-5 digits (the L-BFGS-B row above had lambda and gamma trading off by up to 12%); unseen compositions E MAE 8e-8 meV/atom; 23 min | with the minimum-image check, L-BFGS-B no longer moves at all on this fit: its first trial step reaches a cutoff past half a training cell, the loss scores that as a failed evaluation (1e30) and the line search ends at the start point (loss 1.9e4) |
| Si vacancy neb / tsa (T5), first re-run | **FAIL**: 2x2x2 profile has an image 1.2 eV below both endpoints; 3x3x3 profile breaks down | see next row |
| Si vacancy neb / tsa (T5), endpoints rebuilt | PASS all 8 checks: barrier 0.517 eV (63 atoms) / 0.513 eV (215 atoms), imaginary mode -178.1 / -171.8 cm^-1 (independent Hessian within 0.01%), IRC ends within 0.004 A | the working hypothesis (Td vacancy a saddle) was wrong: at the Td vacancy the lowest non-zero Hessian eigenvalue is +5.9e26 s^-2 and four rattles re-relax to it within 1e-6 eV.  It is a metastable minimum (E_f 4.336 eV = 2 epsilon, the unrelaxed value) with a basin reaching past a 0.59 A displacement of a neighbour; the ground state is a C3v vacancy, the hop atom 24% of the way toward the vacant site, E_f 2.835 / 2.694 eV, 1.50 / 1.64 eV lower.  The doubled three-body term had made Td the lowest.  The old NEB between two Td endpoints ran through the C3v basin (its -270.09 eV image relaxes to -270.358 eV, the C3v energy).  The 3x3x3 run's divergence (image energies to +600 eV after ~700 FIRE steps) did not recur on re-running the same Td setup, which converged at step 602 with an image in the C3v basin: a chaotic trajectory of an ill-posed band, not reproduced; the solver was not changed.  Endpoints now start the hop atom 40% along the hop (25% still returns to Td).  Literature, for information: 0.43 eV (Maroudas & Brown 1993, as quoted by El-Mellouhi et al., arXiv cond-mat/0308289) |

## Minimum-image check for SW and Tersoff (2026-09-26)

`StillingerWeberPotential` and `TersoffPotential` (single- and
multi-species, numpy and numba) now call `_check_min_image` on their
neighbour-list-free path, as `BasePairPotential` already did: a cutoff
(a * sigma, the largest a * sigma of a multi-species set, R + D) at or
past half the smallest perpendicular cell width raises `ValueError`
instead of silently dropping periodic images.  `allow_minimum_image_violation()`
still suppresses it.  The fitting loss already turns the exception into
a failed evaluation, which is what removed the spurious sigma = 7.7 A
minimum above.  Tests that labelled SW or Tersoff on the 8-atom diamond
cell (half-width 2.715 A against 3.771 / 3.0 A cutoffs) moved to the
2x2x2 cell; the MLIP tests keep their 8-atom structures and take the
labels from the 2x2x2 replica (E / 8, the first 8 forces, the stress:
equal to a 3x3x3 replica to 1e-11).  `tests/test_min_image_many_body.py`
pins the refusal, including a sheared cell whose lattice vectors are
long enough but whose perpendicular width is not.

`EAMPotential` had the same unguarded path and now makes the same check
(r_cut).  Its tests ran r_cut = 6.0 A on the 4-atom FCC Al cell
(half-width 2.025 A) and moved to 3x3x3 (108 atoms, 6.075 A).  Two tests
had passed only because images were dropped: the tabulated-spline test's
rho grid ended at 0.97 while the correct rho_bar is 3.32 (the grid now
reaches 5.0), and the MPI worker's LJ fit loss on that cell was the
failed-evaluation value 1e30 on every rank, so its cross-rank comparison
compared nothing (it is now a real loss on the 3x3x3 cell).  With every
image counted, EAM energy per atom agrees between 3x3x3 and 4x4x4 to
1e-12 (1.507 eV/atom, against 0.250 in the bare cell with the check
suppressed).  No `example/` script uses EAM (the CLI cannot build one
from YAML; registry S19).

## Default local optimiser: Nelder-Mead (2026-09-26)

Decided by the user after the Si fit row above.  `fit`'s local step now
defaults to adaptive Nelder-Mead instead of L-BFGS-B, in the library
(`DEFAULT_LOCAL_METHOD`), the CLI and every `--write-config` template.
Measured on the real losses (driver.py keeps the table): Nelder-Mead was
the only method to reach the minimum (~1e-14) on all three of Si from
the DE result, Si from the 10-25% perturbed start, and GeSn (19
variables) from its start; L-BFGS-B stopped at 6.6e-5, 2.3e-3 and 1.9e4.
Two option changes came with it: maxiter 100 000 (= maxfev; at 5000 it
stopped GeSn at 3.2e-3), and no absolute loss tolerance (scipy requires
both tolerances, and with fatol 1e-12 the Si control, minimum 2.12, ran
all 100 000 evaluations for what 1758 evaluations reach).  It stops on a
simplex ~1e-8 relative in every parameter.

## Consequences

- SW now matches the published potential; the Si elastic constants are
  a regression test.
- Earlier SW numbers in the ADRs describe a different potential and are
  marked so until re-run.
- The three-body kernels do half the work for a single species.
