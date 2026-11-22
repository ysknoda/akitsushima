---
status: proposed (2026-11-22)
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

## Consequences

- MD with EAM, PPPM or ESM gets faster; the MLIP paths are unchanged.
- Callers may get a None stress where they asked for none.  The relax
  and NPT paths never ask for none.
