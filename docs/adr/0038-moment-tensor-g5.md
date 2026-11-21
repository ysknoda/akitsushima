---
status: proposed (2026-11-21)
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

## Result

To be written when the implementation is measured.
