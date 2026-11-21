---
status: accepted (2026-11-21; implemented in 0.3.0a2 -- see Result)
---

# G5 from Cartesian moment tensors: an angular kernel linear in the neighbours

Tier B item 3 of ADR-0028 (addendum 5, Q10 of the roadmap grilling).
Agreed with the user on 2026-11-21 (grilling, Q1-Q7).  The item was
planned as a new descriptor family with retrained models; a finding
made during the grilling (below) re-scoped it to the same descriptor and
the same models.

## Starting point (2026-11-21, 0.3.0a1)

- `fitting/descriptors/acsf.py`: G2, G4 and G5 (Behler 2011, eqs. 4, 8
  and 9) with analytic derivatives `dGdv` (M, 3, Dc) per neighbour entry.
  ADR-0030 removed repeated work inside the angular loop (2.1-2.8 times
  per force call), but the loop still runs over every unordered pair of
  neighbours of an atom: 351 pairs per atom with 27 neighbours, times 12
  angular terms for the grid of `ac_elm_iso_r2`.
- Profile at 0.3.0a1 (`benchmarks/after_tierA.jsonl`, `bench_force.py`,
  1 numba thread, machine loaded):

  | case | atoms | force call | `_acsf_kernel` self time |
  |---|---|---|---|
  | `acsf27_elm400_si` | 512 | 55 ms | 72 % |
  | `acsf27_elm400_si` | 4096 | 441 ms | 83 % |
  | `mlip_file_ac_elm_iso_r2` | 512 | 58 ms | 75 % |
  | `mlip_file_ac_elm_iso_r2` | 4096 | 496 ms | 82 % |

  The kernel stayed below the 90 % at which addendum 5 would have asked
  whether items 2 and 3 swap.
- Every GeSn model in use (`ac_elm_iso_r2`, the ensembles of ADR-0033,
  the models of ADR-0037) uses G2 and G5 only, with integer zeta (1, 2,
  4) and lambda = +-1.

## Finding: G5 with integer zeta is a sum of tensor contractions

For atom i and a neighbour entry j (vector v_j, length r_j, unit vector
u_j, species s_j) let

    w_j = exp(-eta r_j^2) fc(r_j)      (fc includes the inner switch of
                                        ADR-0014 when it is set)
    T_n^b = sum_{j: s_j = b} w_j u_j (x) u_j (x) ... (x) u_j   (n factors)

With cos theta_jk = u_j . u_k and an integer zeta,
(1 + lambda cos)^zeta = sum_n C(zeta, n) lambda^n cos^n, and
sum_{j in b, k in c} w_j w_k (u_j . u_k)^n = T_n^b : T_n^c (full
contraction).  The unordered-pair convention of `acsf.py` then gives

    G5_i[b, c; eta, zeta, lambda] = 2^(1-zeta) sum_{n=0}^{zeta}
        C(zeta, n) lambda^n D_n^{bc}

    D_n^{bc} = T_n^b : T_n^c                         b != c
    D_n^{bb} = (T_n^b : T_n^b - sum_{j in b} w_j^2) / 2

(the subtracted term removes j = k, where cos = 1).  The cost per atom
is linear in the number of neighbours: one accumulation of the tensors
per neighbour entry and eta, then a contraction per species pair.  This
is the device behind the moment tensor potentials (Shapeev, Multiscale
Model. Simul. 14, 1153 (2016)) and the atomic cluster expansion
(Drautz, Phys. Rev. B 99, 014104 (2019)).

Numerical check (2026-11-21, scratch script, not in the suite): 64-atom
Ge54Sn10 diamond cell, rattled 0.08 A, slightly sheared, r_c 5 A, 27.7
neighbours per atom, the G5 grid of `ac_elm_iso_r2` (eta 0 and 0.05,
zeta 1, 2, 4, lambda +-1), with and without `inner_cutoff` (1.8, 2.2):
the largest |G5(tensor) - G5(acsf.py)| over all atoms and columns is
9.4e-16 of the largest |G| (6e-18 absolute at an exactly zero Sn-Sn
column).

G4 does not factorise: fc(r_jk) and exp(-eta r_jk^2) depend on the pair
(j, k) and not on j and k separately.

## Decisions

1. **What item 3 becomes (Q1).**  An angular kernel that computes G5
   with integer zeta from the moment tensors above.  The descriptor,
   its feature order, the stored Jacobian `dGdv`, `_aggregate`,
   `_contract`, the tabular `structure_table` path and every model file
   are unchanged; no model is retrained.  The new descriptor family
   (ACE or MTP type, body order above three, retrained models) moves to
   tier C, to be taken up when accuracy rather than speed calls for it;
   so far the accuracy problems of the GeSn models have come from data
   coverage (ADR-0027, ADR-0037), not from the descriptor.

2. **Which terms go where (Q2).**  Per term:
   - G5 with integer zeta: moment tensors;
   - G4, and G5 with non-integer zeta: the pair loop of ADR-0030,
     which is not entered when no such term is present.

   The tensors are stored in symmetric form, (n+1)(n+2)/2 components
   for rank n, each weighted by its multinomial coefficient in the
   contraction.  Every integer zeta goes to the tensors; zeta 16 (153
   components at rank 16) is part of the measurement (decision 5), and
   a cap is set only if it measures slower than the pair loop.

   Derivatives per neighbour entry m of species s, with
   w'_m = dw/dr, t_n^c = u_m^{(x)n} : T_n^c (a scalar) and
   q_n^c = T_n^c contracted n-1 times with u_m (a vector):

       d(T_n^s : T_n^c)/dv_m = w'_m u_m t_n^c
                               + (n w_m / r_m)(q_n^c - t_n^c u_m)

   for c != s (and the same with the roles of the two species swapped);
   for D_n^{ss} the factor 2 of the square cancels the 1/2 and
   w_m w'_m u_m is subtracted.  Checked against central differences
   (h = 1e-6 A) on 12 random neighbours of two species, n = 0-4, all
   three species pairs: largest relative difference 7e-8, the level of
   the difference formula itself.

3. **Switch (Q3).**  `ACSF(..., angular_kernel="auto" | "pairs" |
   "moments")`, default `auto` (tensors for every term that qualifies).
   `pairs` keeps the ADR-0030 kernel for every angular term, as the
   reference and for comparisons; `moments` raises if a G4 or a
   non-integer zeta is present.  The choice is how the descriptor is
   computed, not what it is, so it is not written to the model file,
   the descriptor config or the YAML.

4. **Equality and derivatives (Q4).**  Against
   `tests/acsf_reference_kernel.py` (the pre-ADR-0030 kernel kept as the
   reference):
   - G, dGdv and the energy, forces and stress of an atom-centred ELM;
   - tolerance 1e-10 of the largest |value| of each column (a column of
     G or of dGdv), because columns that are exactly zero (a species
     pair with no neighbour of one species) make an element-wise
     relative test meaningless;
   - cases: Si and GeSn; with and without `inner_cutoff`; zeta 1, 2, 4
     and 16; lambda +-1; G4 and G5 together; a non-integer zeta; a cell
     smaller than 2 r_c (several images of one atom in the list); a
     compressed and a stretched cell, where the alternating sum over n
     with lambda = -1 cancels most.
   - Forces and stress against finite differences of the energy, and
     the sum of the forces against zero.
   - At most 60 s of `slow` tests in total (addendum 5, Q9).

5. **Speed gate and measurements (Q5).**  The 1.5x gate of ADR-0028 on
   `bench_force.py` case `mlip_file_ac_elm_iso_r2`, 4096 atoms, one
   numba thread, best of three, whole force call (496 ms at 0.3.0a1).
   It runs on the second core next to the size ladder and is tagged
   `loaded`, like the 0.3.0a1 rows.  Recorded but not gated:
   - `acsf27_elm400_si` at 512 and 4096 atoms, and
     `mlip_file_ac_elm_iso_r2` at 512;
   - the fitting time of an atom-centred ELM (`atomic_jacobian` uses the
     same kernel);
   - r_c 5, 6 and 7 A on one synthetic grid, to show the linear growth
     with the number of neighbours against the quadratic one of `pairs`;
   - zeta 16 (decision 2).

   Measurements with more than one thread wait for the user's word,
   like the parallel matrix of addendum 5.

6. **Where the work happens (Q6).**  akitsushima is an editable
   install, and the size ladder of addendum 5 starts a new
   `python -m akitsushima` for every unit; its last nine units use the
   `ac_elm_iso_r2` model.  The implementation is therefore done in a git
   worktree (`../akitsushima-adr0038`, branch `adr-0038-moment-g5`),
   where `python -m pytest` imports the worktree's package, and merged
   into main after the ladder has finished.  No third concurrent CPU job
   is started.

7. **Process (Q7).**  This ADR is committed as Proposed; the
   implementation follows, and the ADR is accepted with its result.
   Version 0.3.0a2.  Local commits only.

## Consequences

- Existing models, ensembles and active-learning runs become faster
  without retraining, and the result is the same to rounding.
- The cost of G5 grows linearly with the cutoff volume instead of
  quadratically, which makes a longer r_c affordable for later fits.
- ADR-0016 decision 6 (tiled ACSF on the GPU) would tile the tensor
  accumulation instead of the pair loop; noted there when the GPU work
  starts.
- G4 keeps the quadratic cost.  A model that needs G4 for speed would
  need a different descriptor (tier C).

## Implementation notes

- `fitting/descriptors/acsf.py`: `_g5_moments` prepares, per call, the
  distinct etas, the coefficients 2^(1-zeta) C(zeta, n) lambda^n of every
  tensor term and the symmetric components (`_moment_components`);
  `_g5_moments_impl` (numba, `prange` over atoms like the pair kernel)
  accumulates T_n^b per atom, contracts it into D_n^{bc} and G, and adds
  the derivative of decision 2 to `dGdv`.  A term of zeta z uses ranks
  n <= z only.
- `_acsf_kernel` keeps its former positional signature and takes an
  optional `moments` mask; `ACSF._run` passes it only when a term is on
  the tensors, so the pre-ADR-0030 reference kernel can still stand in
  for it in `tests/test_acsf_kernel_reference.py` (its fixture switches
  the descriptor to `pairs` meanwhile).
- **Cap on zeta (decision 2 applied).**  Measured on 4096 GeSn atoms
  (15 % Sn, rattled diamond), G2 grid of `ac_elm_iso_r2`, G5 with eta 0
  and 0.05, zeta 1 and z_max, lambda +-1, one thread, G and dGdv:

  | z_max | r_c 5 A: pairs | tensors | r_c 7 A: pairs | tensors |
  |---|---|---|---|---|
  | 4 | 0.313 s | 0.202 s | 0.785 s | 0.316 s |
  | 6 | 0.308 s | 0.281 s | 0.805 s | 0.484 s |
  | 8 | 0.300 s | 0.441 s | 0.800 s | 0.764 s |
  | 10 | 0.314 s | 0.672 s | 0.800 s | 1.128 s |
  | 12 | 0.305 s | 1.024 s | 0.793 s | 1.698 s |
  | 16 | 0.344 s | 2.122 s | - | - |

  The tensors win up to zeta 6 and lose from zeta 8 at r_c 5 A, and
  break even at 8 at r_c 7 A.  `auto` therefore puts G5 terms with
  integer zeta <= 6 (`MOMENT_ZETA_MAX`) on the tensors and leaves larger
  zeta to the pair loop; `moments` keeps every integer zeta on the
  tensors.  A grid mixing zeta 1, 2, 4 and 16 under `auto` (the zeta-16
  terms on the pair loop) still takes 0.372 s against 0.510 s with
  `pairs`.

## Result

Accepted on 2026-11-21.  Measured on the second core next to the size
ladder (tag `loaded`, load average 6-8), one numba thread, best of
three; ledgers `benchmarks/adr0038.jsonl` (gate) and
`benchmarks/adr0038_extra.jsonl` (the rest).  The gate run records
version 0.3.0a1 and commit a0ede27 because it ran on the uncommitted
working tree of this ADR.

**Gate (decision 5): passed.**  Whole force call, against
`benchmarks/after_tierA.jsonl` (0.3.0a1):

| case | atoms | 0.3.0a1 | 0.3.0a2 | old / new |
|---|---|---|---|---|
| `mlip_file_ac_elm_iso_r2` (gate) | 4096 | 496 ms | 284 ms | **1.74** |
| `mlip_file_ac_elm_iso_r2` | 512 | 58 ms | 31 ms | 1.90 |
| `acsf27_elm400_si` | 4096 | 441 ms | 180 ms | 2.45 |
| `acsf27_elm400_si` | 512 | 55 ms | 25 ms | 2.24 |

The energies per atom are unchanged (-2.50928 and -2.56008 eV/atom for
the GeSn model).  In the gate case the self time is now 42 % in
`_g5_moments`, 27 % in `_acsf_kernel` (now G2 only), 8 % in
`_neighbours` and the rest in the ELM.

**Growth with the cutoff.**  Descriptor with derivatives, 4096 GeSn
atoms, `ac_elm_iso_r2` grid with r_c changed:

| r_c | neighbours per atom | pairs | tensors | pairs / tensors |
|---|---|---|---|---|
| 5 A | 27.2 | 0.427 s | 0.228 s | 1.87 |
| 6 A | 33.5 | 0.616 s | 0.270 s | 2.28 |
| 7 A | 46.0 | 1.061 s | 0.377 s | 2.82 |

From 5 to 7 A the neighbours grow 1.69 times; the tensor path grows
1.65 times (linear), the pair loop 2.49 times.

**Fitting.**  An atom-centred ELM (400 hidden units) on 40 rattled
64-atom GeSn cells: 1.90 s with `pairs`, 1.54 s with `auto` (1.24
times); at this size the ELM solve and the Python overhead weigh more
than the descriptor.

**Equality (decision 4).**  Largest difference between `auto` (or
`moments`) and `pairs`, over Si and GeSn and cubic, triclinic, 15 %
compressed, 15 % stretched and 8-atom cells, as a fraction of each
column's largest value:

| grid | G | dGdv |
|---|---|---|
| `ac_elm_iso_r2` G5 (zeta 1, 2, 4) | 4.5e-15 | 8.4e-15 |
| the same with `inner_cutoff` | 4.2e-15 | 5.6e-15 |
| G4 and G5 together | 4.5e-15 | 8.4e-15 |
| zeta 2 and 2.5 (one term each way) | 3.0e-15 | 9.7e-16 |
| zeta 1 and 16 on the tensors (`moments`) | 1.1e-13 | 1.1e-13 |

The worst case is always the stretched cell, where the alternating sum
over n cancels most; even at zeta 16 it stays three orders below the
1e-10 rule.  Forces and stress of E = sum_i c_i . G_i agree with central
differences (1e-6 eV/A, 1e-7 in the stress) and the forces sum to zero
to 1e-10 for the zeta-16, G4 + G5, mixed-zeta and `inner_cutoff` grids.
The existing tests of the reference kernel now run with both `auto` and
`pairs`, and the finite-difference tests of `test_mlip_chain_rule.py` and
`test_mlip_atom_centred.py` (three species, G4 and G5 with integer zeta)
exercise the mixed path.

**Tests.**  `tests/test_acsf_moments.py`: 60 comparisons, 4
finite-difference checks and the validation of `angular_kernel`; about
17 s of `slow` tests, most of it the first numba compilation.

**What is left.**
- The G2 loop (27 % of the call) and the tensor derivative pass are the
  largest parts now; neither is planned.
- Multi-thread scaling of the tensor kernel waits for the parallel
  matrix of addendum 5.
- The new descriptor family stays in tier C (decision 1).
