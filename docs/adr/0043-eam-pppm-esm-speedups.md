---
status: accepted (2026-11-22, 0.3.0a7)
---

# EAM, PPPM and ESM speed-ups; `need_stress`

Tier B item 8 of ADR-0028 (addendum 5; Q12 and addendum 4).  Item 7, the
point-defect workflow, was skipped by the user on 2026-11-22.  Agreed
with the user on 2026-11-22 (grilling, two rounds, Q1-Q13, all as
recommended).

## Starting point (0.3.0a6)

ADR-0028 profiles (one force call, one numba thread):

| Case | Largest shares |
|---|---|
| Zhou EAM Ni, 4000 atoms | `eam.py:compute` 55 %, `_decay` 20 % |
| Buckingham + PPPM ZnO, 3600 atoms | FFT 30 %, `_mesh_stress` 28 % |
| ESM bc2, 96x96x384 grid | `_compute_stress` 69 % |

- **EAM** (`engine/eam.py`): numpy, `np.add.at`, no numba.  The
  functions come in three forms:
  - Zhou analytic closures (`eam_zhou.py`);
  - scipy `CubicSpline` tables (setfl, arrays);
  - bare callables.
- **PPPM** (`engine/coulomb/pppm.py`): complex `np.fft.fftn`; Python
  charge assignment; `_mesh_stress` on every call.
- **ESM** (`engine/coulomb/esm.py`): every call builds one Nz x Nz
  Green-function matrix per distinct |G_par|, and the dG/dG_par
  matrices for the stress.  The stress is computed on every call.
- **No caller can say that the stress is not needed.**  Every potential
  computes it on every call.

## Decisions

1. **One ADR and one version (Q1).**  0.3.0a7.  EAM, PPPM and ESM are
   implemented in turn and committed separately.

2. **`need_stress` (Q2, Q8, Q9, Q10).**
   - A keyword `need_stress=True` on engine `compute` and on the force
     functions.  With False the stress is returned as None, not zeros, so
     that a caller that uses it fails loudly.
   - It is passed only to callables that accept it, checked from the
     signature as `neigh_list` is.  Other callables return the stress as
     before.
   - Callers that pass False:
     - MD NVE / NVT / Langevin / CSVR on steps that write neither a
       thermo row nor a trajectory frame.  NPT always needs the stress.
     - The displacement force calls of phonon, BTE and dielectric.
     - NEB, TSA and the GCMC energy evaluations.
   - Relaxation is unchanged: relax.log writes Pmax on every step.
   - Engines that skip work under False:
     - PPPM `_mesh_stress`;
     - ESM `_compute_stress`;
     - the separable reciprocal-space stress of Ewald.
   - The Ewald JIT stays in tier C.  EAM and pair-potential virials are
     cheap and are always computed.

3. **EAM in numba (Q3, Q12).**
   - Kernels for the Zhou analytic functions, and for `CubicSpline`
     tables using scipy's PPoly coefficients as they are.
   - Bare callables, and MPI with more than one rank, keep the numpy
     path.
   - One rank: `prange` over atoms on a full (two-way) neighbour list,
     so no writes collide.
   - The summation order changes; results stay within 1e-10.

4. **PPPM (Q4).**
   - Real FFT (`rfftn` / `irfftn`).
   - Charge assignment in numba.
   - `_mesh_stress` only when the stress is needed.

5. **ESM without Green-function matrices (Q5, Q11).**
   - The kernels are sums of exponentials: exp(-g|z - z'|) and
     exp(-g(L_z -/+ (z + z'))).
   - They are applied in O(Nz) per G_par by forward and backward
     recursions, with every exponent <= 0.
   - The stress kernels (dG/dG_par, terms such as |z - z'| e^{-g|z-z'|})
     are applied the same way.
   - If 1e-10 against the matrix path is not met, the fallback is a
     memory-capped cache of the matrices.

6. **Gates (Q6).**
   - The default path within 1e-10 (relative) of 0.3.0a6 (ADR-0028
     decision 5).
   - Each changed path at least 1.5x faster on the `bench_force.py`
     cases:
     - Zhou EAM Ni, 500 and 4000 atoms;
     - Buckingham + PPPM MgO, 512 and 4096 atoms;
     - ESM bc1 / bc2 / bc3 at 64x64x256.
   - One thread; multi-thread measurements only on the user's word.

7. **References in the tests (Q13).**
   - The EAM numpy path stays, and is compared with the numba path.
   - The ESM matrix construction stays as a private reference.
   - PPPM is compared with the complex FFT in the tests.
   - The existing finite-difference force and stress tests must pass.

8. **Process (Q7).**
   - Worktree; slow tests at most 60 s in total.
   - Ask before every push and tag.

## Result (2026-11-22, 0.3.0a7)

### Implementation

- **`need_stress`:**
  - Added to `compute_all_general`, the `make_force_fn` closures
    (Buckingham, engine, engine + Coulomb), `PPPMSolver.compute`,
    `ewald_sum` and `ESMCalculator.__call__` / `ESMCoulomb.compute`.
  - `fitting.force_fn.call_force_fn` and `accepts_need_stress` (an
    explicit parameter only, cached).
  - Callers:
    - MD NVT (NHC) / Langevin / CSVR on steps without a trajectory frame
      (their thermo rows carry no pressure);
    - NVE on steps with neither a thermo row (it shows the pressure) nor
      a frame;
    - phonon (both IFC paths), BTE, NEB, TSA, the GCMC energy
      evaluations.
  - NPT and relaxation are unchanged.
- **EAM** (`engine/_eam_kernels.py`):
  - Functions of known form carry `_ak_spec`:
    - Zhou density / pair / cross / embedding, set in `zhou_eam`;
    - setfl splines and r*phi / r, set in `read_setfl`;
    - array tables as scipy PPoly.
  - One numba call does the work in parallel passes:
    - per pair: rho, rho' and phi;
    - per atom, through a two-way CSR: rho_bar and F;
    - per pair: the pair force;
    - per atom: the force sum.
  - Bare callables and MPI splits keep the numpy path.  Its species-pair
    masks are now built from integer codes instead of a per-pair Python
    comprehension over frozensets.
- **PPPM:**
  - One `rfftn` of the charge mesh (it was transformed twice, complex)
    and three `irfftn` for the field.
  - B-spline weights are computed once and shared by the numba
    assignment and gather.
  - The half-spectrum factors are built per lattice and averaged with
    the partner index of -k (finding 1).
  - The stress is a single einsum.
- **ESM:**
  - `_esm_apply` (numba, parallel over G_par columns) applies G and
    dG/dG_par in O(Nz): forward / backward recursions for
    exp(-g|z - z'|), (triangular) rank-one sums for the image terms, and
    every exponent <= 0.
  - The |G_par| rounded to 8 significant digits, as the matrix path used
    for its kernels and prefactors, is kept, so that the results agree.
  - The matrix methods remain as `_solve_poisson_esm_matrix` and
    `_compute_stress_matrix`.
- **Benchmark:** `benchmarks/bench_force.py` gained the
  `buck_esm_bc{1,2,3}_mgo_slab` cases (288 atoms, 64x64x256).  The ledger
  is `benchmarks/adr0043.jsonl`.

### Compatibility (default path against 0.3.0a6, relative max difference)

| Case | E | F | stress |
|---|---|---|---|
| Zhou EAM Ni (256 atoms) | 4e-16 | 7e-16 | 9e-16 |
| Zhou EAM NiAl random | 4e-16 | 1e-15 | 2e-16 |
| setfl EAM NiAl | 1.2e-15 | 9e-16 | 2e-16 |
| Buckingham + PPPM MgO (216) | 0 | 5e-16 | 2.2e-14 |
| Buckingham + Ewald MgO | 0 | 0 | 0 |
| ESM bc1 / bc2 / bc3 / bc2 with field | 0 | <= 1.9e-15 | <= 4.5e-15 |

All are within the 1e-10 of ADR-0028 decision 5.

### Speed gate (`bench_force.py`, one numba thread, call without a list)

| Case | N | 0.3.0a6 (s) | 0.3.0a7 (s) | Ratio |
|---|---|---|---|---|
| Zhou EAM Ni | 500 | 0.0192 | 0.0043 | 4.4 |
| Zhou EAM Ni | 4000 | 0.156 | 0.037 | 4.2 |
| Buckingham + PPPM MgO | 512 | 0.0380 | 0.0231 | 1.65 |
| Buckingham + PPPM MgO | 4096 | 0.336 | 0.213 | 1.58 |
| Buckingham + ESM bc1 slab | 288 | 1.658 | 0.238 | 7.0 |
| Buckingham + ESM bc2 slab | 288 | 2.545 | 0.308 | 8.3 |
| Buckingham + ESM bc3 slab | 288 | 1.850 | 0.249 | 7.4 |

- Every case passes the 1.5x gate.
- With `need_stress=False` the ESM call drops further to 0.19-0.21 s
  (`ref043.py` timings).
- The PPPM ratio is held down by the Buckingham short-range part:
  `_build_pair_groups`, a per-pair Python dict loop, is 0.175 of the
  0.28 s profile at 4096 atoms.  The PPPM part itself now takes about
  0.08 s.  See finding 2.

### Tests (`tests/test_speedups_adr0043.py`, 18 tests, about 3 s, none slow)

- EAM numba against the numpy path (Zhou Ni, Zhou NiAl, a setfl written
  from it): 1e-10.
- Bare callables keep the numpy path.
- PPPM half spectrum against the full-FFT reference:
  - cubic, auto and odd meshes;
  - hexagonal, M = 16 and 17.
- ESM recursions against the matrix path:
  - the potential and the stress, bc1 / bc2 / bc3, bc2 with a field;
  - the dense kernels and their derivatives for three g, all boundary
    conditions, to 1e-12.
- `need_stress=False`: the same E and F bit for bit and stress None
  (PPPM, Ewald, ESM, EAM).  `call_force_fn` does not pass the flag
  through `**kwargs`.
- NVT with PPPM:
  - the same trajectory whether the flag is honoured or ignored;
  - the stress is asked for only on the trajectory steps.

Full suite: 1153 passed, 1 skipped in 12.8 min (17.9-23.6 min before;
the ESM finite-difference tests are the main gain).

### Findings

1. **Even PPPM meshes are not Hermitian at the Nyquist index.**
   - m = -M/2 is its own negative, so k(-k) != -k(k) there.  The old
     Re(ifftn) silently projected the summand; a plain irfftn did not
     (forces off by 4.6e-6 at M = 16).
   - Averaging every factor with its value at the index of -k
     reproduces the full-mesh sums exactly.  This is the form now used.
2. **The Buckingham short-range pair grouping
   (`pair_potentials._build_pair_groups`) is the next PPPM bottleneck.**
   It is a Python dict loop over pairs (181684 `setdefault` calls at
   4096 atoms).  It is not in this item's scope.
3. **`test_npt_aniso.py::test_z_mode_matches_the_constrained_relax` is
   marginal.**
   - With `NUMBA_NUM_THREADS=1` it fails on 0.3.0a6 too: 0.002254
     against a bound of 0.002221.  It passes with the default threads.
   - It is independent of this change (SW potential, NPT and relax
     paths untouched): the summation order of the threaded SW kernel
     moves the 3000-step NPT mean by a few 1e-5.

## Consequences

- MD with EAM, PPPM or ESM gets faster; the MLIP paths are unchanged.
- Callers may get a None stress where they asked for none.  The relax
  and NPT paths never ask for none.
