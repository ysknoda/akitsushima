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
| Si fit (T8 SW) | **FAIL**: differential evolution settled at sigma 7.70 A (cutoff 13.9 A in a 10.5 A cell), loss 771 (was 3.3e-6) | the engine's O(N^2) path evaluates a cutoff past half the cell width without error (minimum image drops the other images), so the global search found a spurious minimum there; loss at the true parameters is 7e-15 and a local-only fit reaches 2.3e-3.  Adding the minimum-image check that pair potentials already have fails ~25 tests that label SW on 8-atom cells -- a separate change (open) |
| Si vacancy neb / tsa (T5) | **FAIL**: 2x2x2 profile has an image 1.2 eV below both endpoints; 3x3x3 profile breaks down | with SW's own three-body strength the symmetric vacancy the endpoints relax to is no longer a minimum (the doubled term held it); the check's endpoint construction must break the symmetry before relaxing (open) |

## Consequences

- SW now matches the published potential; the Si elastic constants are
  a regression test.
- Earlier SW numbers in the ADRs describe a different potential and are
  marked so until re-run.
- The three-body kernels do half the work for a single species.
