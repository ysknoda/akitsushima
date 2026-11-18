---
status: accepted (2026-11-18; gate passed, 2.1-2.8 times -- see Result)
---

# ACSF kernel: fewer operations in the angular loop

Tier A step 5 of ADR-0028.  Agreed with the user on 2026-11-18
(short grilling, Q1-Q8).  Same descriptor, same models, same stored
Jacobian; only the arithmetic of `_acsf_kernel`
(`fitting/descriptors/acsf.py`) changes.

## Starting point (2026-11-18, 0.2.0a19)

One force call of an atom-centred ELM on the ACSF grid of
`ac_elm_iso_r2`: r_cut 5 A, 15 G2 (eta 12.5), 12 G5 (eta 0 and 0.05,
zeta 1, 2, 4, lambda +-1).  Diamond cell, 4096 atoms, 1 numba thread,
loaded machine (load average about 7), best of 3.

| Stage | Si (1 species, Dc 27) | GeSn (2 species, Dc 66) |
|---|---|---|
| `_neighbours` (cell list) | 22 ms | 21 ms |
| kernel, G only | 691 ms | 656 ms |
| kernel, G and dGdv | 1201 ms | 1137 ms |
| `_contract` (forces and virial) | 4 ms | 10 ms |
| ELM, 400 hidden units | 44 ms | - |
| whole force call | 1224 ms | - |
| dGdv | 74 MB | 177 MB |

- **The angular loop is 95 % of the kernel.**  The GeSn kernel takes
  35 ms (G only) and 53 ms (with dGdv) without the G5 terms.  With 27
  neighbours per atom there are 351 pairs per atom and 12 angular terms
  per pair.
- **Each term repeats work that does not depend on the term.**
  - `base_c**zeta` is a general power, although zeta is 1, 2 or 4.
  - `exp(-eta * rsum)` is computed 12 times per pair, for two values
    of eta.
  - `2**(1 - zeta)` is computed per term.
  - The cutoff function of the second neighbour is evaluated again for
    every pair it belongs to.
  - The derivative adds `base_c**(zeta - 1)`, a second general power.
- With 4 threads the whole call takes 447 ms (2.7 times faster).
- `atomic_jacobian` is also what the fitting uses, so fitting time
  scales with the same kernel.

## Decisions

1. **What changes (Q1).**  Inside the kernel, with the same result:
   - the cutoff factor and its derivative per neighbour entry, computed
     once per atom instead of once per pair;
   - one exponential per distinct eta per pair, shared by the terms
     with that eta;
   - integer zeta raised by repeated multiplication;
   - `2**(1 - zeta)` precomputed per term.

   The derivative path (dGdv) uses the same shared quantities.  A
   non-integer zeta keeps the general power.  The descriptor, its
   feature order, the stored `(M, 3, Dc)` Jacobian, `_contract`,
   `_aggregate` and every model are unchanged.

2. **Taken out of the plan of ADR-0028 step 5 (Q1, Q2).**
   - Contracting forces and virial without storing dGdv saves memory,
     not time, on a CPU.  The contraction needs g = d eps / dG, which
     needs G first, so a fused kernel computes G twice.  The G pass is
     about 60 % of the call, and `_contract` itself takes 4-10 ms.
     dGdv takes 0.6 GB at the largest size of ADR-0025 (13608 atoms)
     on a 24 GB machine.  Processing atoms in blocks would bound the
     memory without recomputation, because the ELM energy is per atom.
     That is left to ADR-0016 decision 6 (tiled GPU inference).
   - Reusing the ACSF neighbour list within a skin: `_neighbours` is
     2 % of the call, and about 5 % even after a threefold faster
     kernel.  That is far below the gate.

3. **Equality with the current kernel (Q3, Q6).**  The current kernel
   is copied unchanged into `tests/acsf_reference_kernel.py` as the
   reference.  A test compares G, dGdv and the E, F, S of an ELM
   between the reference and the new kernel, to relative 1e-10 (the
   default-path rule of ADR-0028).  The cases are:
   - Si and GeSn;
   - G4 and G5;
   - with and without `inner_cutoff`;
   - a non-integer zeta;
   - a triclinic cell.

   The existing finite-difference and chain-rule tests stay as they are.

4. **Gate (Q4, Q7).**  The new kernel is adopted if one force call is
   at least 1.5 times faster.  The old and new code are timed
   alternately in the same session, at 1 and 4 threads, for Si and
   GeSn at 512 and 4096 atoms.  The provisional `baseline.jsonl` is not
   used, because the load varies between 7 and 10.  If the gate is not
   met, the change is reverted and this ADR is marked rejected, with
   the measurements kept.

5. **Recording (Q5, Q8).**  This ADR is written before the
   implementation.  The result is appended and the status set to
   accepted or rejected.  ADR-0028 step 5 links here for the two items
   taken out.  If the change is adopted, the version goes to 0.2.0a20.

## Result (2026-11-18, 0.2.0a20)

**Implementation.**  `_acsf_kernel` keeps its signature.  It now
prepares the per-term constants:
- the index of each term's exponential among the distinct
  (eta, G4/G5) pairs;
- `2**(1 - zeta)`;
- zeta - 1 as an integer where zeta is integral.

It then calls `_acsf_kernel_impl`.  That kernel stores |v|, fc and
dfc of each neighbour entry in the G2 loop and reads them back in the
angular loop.  The other changes in the angular loop:
- it evaluates one exponential per distinct (eta, G4/G5) per pair;
- it raises integer zeta with `_pow_int` (repeated squaring) and forms
  base^zeta as base^(zeta-1) * base, so the derivative reuses
  base^(zeta-1);
- it skips the cutoff of r_jk when the grid has no G4 term.

**Equality.**  `tests/test_acsf_kernel_reference.py` (19 tests) passes
to relative 1e-10 against the copied kernel.  The actual differences
are far smaller.  On GeSn with 4096 atoms, G differs by at most
1.2e-16 relative and dGdv by 2.8e-16; 87 % and 78 % of the elements are
bit-identical.  The existing ACSF and MLIP tests (finite differences,
chain rule, thin cells) pass with and without the JIT.  The two
comparison tests are `slow`: without the JIT they take 2-12 s.

**Gate.**  One ACSF(ac_elm_iso_r2 grid) + ELM(400) force call, old and
new kernel alternated in one process: 3 rounds, best of 3 each, load
average about 8.

| Case | N | threads | old [ms] | new [ms] | speed-up |
|---|---|---|---|---|---|
| Si | 512 | 1 | 145.3 | 53.1 | 2.74 |
| Si | 512 | 4 | 56.5 | 23.1 | 2.44 |
| Si | 4096 | 1 | 1217.3 | 437.5 | 2.78 |
| Si | 4096 | 4 | 435.8 | 181.2 | 2.41 |
| GeSn | 512 | 1 | 147.4 | 58.1 | 2.53 |
| GeSn | 512 | 4 | 54.3 | 25.0 | 2.17 |
| GeSn | 4096 | 1 | 1207.9 | 472.0 | 2.56 |
| GeSn | 4096 | 4 | 447.0 | 212.1 | 2.11 |

Every case passes the 1.5x gate.  On the kernel alone (GeSn, 4096
atoms, 1 thread), G takes 178 ms against 660 ms (3.7 times) and G with
dGdv takes 410 ms against 1164 ms (2.8 times).  Writing dGdv is now the
larger part of the kernel.  The share of the whole call that is not
kernel grows with the thread count, which is why the 4-thread
speed-ups are lower.

Full suite: 855 passed, 1 skipped (14.1 min, loaded machine).
