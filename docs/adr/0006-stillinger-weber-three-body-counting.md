---
status: proposed
---

# Stillinger-Weber: each three-body angle counted once

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

## Open question for the user (not decided here)

The Tomita et al. GeSn records (Jpn. J. Appl. Phys. 57, 04FB04 (2018),
`example/GeSn/relax/gesn_sw_relax.yaml`) have (i, j, k) and (i, k, j)
rows that differ in gamma (e.g. Ge-Ge-Sn 1.8136, Ge-Sn-Ge 0.7232), and
the engine applies a mixed row's sigma, a and gamma to both legs of the
angle.  LAMMPS instead takes each leg's sigma, a and gamma from the
diagonal records (i, j, j) and (i, k, k) and only lambda and cos theta0
from (i, j, k).  The two readings give different energies for this
parameter set.  Which one Tomita et al. used should be settled from the
paper (the PDF is needed); the counting fix above does not depend on it.

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
| ADR-0003 T8 / `example/fit/Si_sw`, `example/fit/GeSn_sw` | training labels in `training_set/structures.csv` were made with the old engine; fitting them with the new engine would "recover" lambda ~ 2x | regenerate the training sets (`make_training_set.py`), re-fit, re-check |
| `example/fit/GeSnVac_SW` | VASP data (OUTCAR, vasprun.xml), not labelled by the engine | none |
| ADR-0004 SW-Si MLIP verification (`example/mlip/adr0004`, `sw_si_data*.npz`, all tables in its appendix) | the MLIPs learned the doubled potential; the learning / conservation / lattice results stay valid as "learning a known three-body potential", but the reference is not SW | re-label as "SW with a doubled three-body term" or regenerate the caches and re-run `sw_si_verify.py` (long) |
| ADR-0005 comparison (`example/mlip/adr0005`, `data/*.npz`, ledgers) | same as ADR-0004; its reference elastic constants (201.8 / 51.6 / 90.6 GPa) are those of the doubled potential | re-label, or regenerate data and re-run `compare.py` (several hours) |
| `tests/` | no existing test pinned a three-body-dependent SW number (all 470 pass unchanged; FD force/stress tests included) | new regression tests added |

## Consequences

- SW now matches the published potential; the Si elastic constants are
  a regression test.
- Earlier SW numbers in the ADRs describe a different potential and are
  marked so until re-run.
- The three-body kernels do half the work for a single species.
