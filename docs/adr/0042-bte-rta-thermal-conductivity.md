---
status: accepted (2026-11-22, 0.3.0a6)
---

# Lattice thermal conductivity from third-order force constants (RTA)

Tier B item 6 of ADR-0028 (addendum 5).  Agreed with the user on
2026-11-22: grilling in three rounds, Q1-Q22, all as recommended.

## Purpose

- **Intrinsic lattice thermal conductivity of ordered crystals** from the
  phonon Boltzmann transport equation in the relaxation-time
  approximation (RTA).
  - Inputs: the harmonic (fc2) and third-order (fc3) force constants of
    any akitsushima potential, including the MLIPs.
  - It is cheaper and less noisy than Green-Kubo MD.
  - It gives mode-resolved lifetimes and mean free paths.
- **Quantum statistics.**  Below the Debye temperature this matters; the
  classical MD methods (`kappa` GK, `nemd`) cannot give it.
- **Reuse of ADR-0040.**  The space-group operations, the atom mapping
  and the site-symmetry reconstruction reduce the displaced pairs.

## Starting point (0.3.0a5)

- `kappa` (Green-Kubo MD, Slack / Debye estimate) and `nemd` exist.  They
  were validated on LJ Ar (ADR-0020): GK 0.68 +- 0.04, NEMD 0.64 +- 0.07
  W/m/K.
- `PhononCalculator` provides:
  - fc2 by finite displacements on a diagonal supercell;
  - the dynamical matrix with minimum-image translations;
  - NAC (`set_nac`) and the D-ASR correction;
  - symmetry reduction (ADR-0040).
- Nothing computes fc3, group velocities or phonon-phonon scattering.
- phono3py 4.1.0, phonopy 4.1.0 and spglib 2.7.0 are installed in the
  development environment.

## Decisions

### Scope

1. **Scattering (Q1).**
   - Three-phonon scattering in the RTA.
   - Optional mass-disorder scattering (Tamura): isotopes, and an alloy
     treated as a virtual crystal.
   - Optional boundary scattering, 1/tau_b = |v| / L (Casimir length L).
   - Both options are off by default.
   - The iterative or direct solution of the full BTE goes to tier C.
2. **Own implementation (Q2).**
   - fc3, its symmetry reduction and the RTA are written here, on the
     ADR-0040 operations.
   - phono3py is a test oracle only, as phonopy was for ADR-0040.  It
     is never imported by the package.
3. **Occupation (Q3).**
   - Bose-Einstein by default.
   - Classical, n = kT / (hbar omega), as an option, for comparison with
     classical MD.
4. **New command (Q5).**  `akitsushima bte bte.yaml`, a command of its
   own like `nemd`.
5. **NAC (Q6).**  When Born charges and eps_inf are given, the existing
   `set_nac` enters the harmonic part: frequencies, eigenvectors and
   group velocities.

### Method

6. **Delta function (Q8).**
   - A Gaussian of fixed width `sigma` (THz), the same definition as
     phono3py's `--sigma`.
   - Convergence in sigma and mesh is checked in the validation.
   - The tetrahedron method and adaptive smearing go to tier C.
7. **fc3 displacements (Q9).**
   - Two-step finite differences: displace a first atom, then a second
     atom, and difference the forces.  This is the same formula as
     phono3py.
   - Amplitude 0.03 A.
   - `cutoff_pair` (A), default off: pairs farther apart are not
     computed, and their fc3 is zero.
8. **fc3 symmetrisation (Q10).**
   - On by default: index-permutation symmetrisation and translational
     invariance (acoustic sum rule) by projection.
   - The oracle comparison is made with both off, and with both on
     (phono3py `--sym-fc`).
9. **Supercells (Q11).**
   - fc2 and fc3 may use different diagonal supercells; the default is
     the same one.
   - fc2 can be read from the `ifc.npz` that `phonon` saves.
10. **Symmetry (Q12).**  `symmetry: true` is the default in `bte`, since
    it is a new command; it needs `akitsushima[symmetry]`.  With
    `false` every displacement is computed.
11. **q points and triplets (Q13).**
    - A Gamma-centred `mesh`.  The irreducible q points come from spglib.
    - All momentum-conserving triplets (q + q' + q'' = G) are kept.  The
      little-group reduction is added only if speed requires it.
    - The interaction is computed with numba `prange`.
    - No speed gate: the time for Si on a 19^3 mesh is recorded next to
      phono3py's.
12. **Group velocity (Q14).**
    - The short-range part comes from the analytic dD/dq.
    - Degenerate modes are handled in their degenerate subspace (as
      phonopy does).
    - The NAC term is differentiated by finite differences in q.
13. **Outputs (Q15).**
    - `bte_results.json`: the kappa tensor per temperature, in W/m/K,
      and the settings.
    - `bte_modes.npz`: per mode, the frequency, group velocity,
      linewidth, lifetime, heat capacity and kappa contribution.
    - Cumulative kappa against frequency and against mean free path, as
      CSV.
    - Plots: kappa(T), Gamma(omega) and the cumulative kappa.
    - No HDF5, which would be a new dependency.
14. **Parallelism (Q16).**
    - `n_jobs` for the displaced force calls, as in `compute_ifc`.
    - numba `prange` for the interaction.
    - Multi-thread measurements only on the user's word.
15. **GeSn (Q17).**
    - The host (Ge) force constants with the virtual-crystal mass, plus
      the Sn mass variance through Tamura's formula.
    - Force-constant disorder is ignored, and this limitation is stated.
    - SQS alloy supercells go to tier C.

### Inputs, gates and tests

16. **Mass variance (Q18).**  Three forms in the YAML:
    - `mass_variance: {Ge: g}`, the g factor directly;
    - `isotopes: {Ge: {mass: fraction, ...}}`;
    - `alloy: {host: Ge, Sn: x}`.

    For the last two, g = sum_i f_i (1 - m_i / m_bar)^2 with
    m_bar = sum_i f_i m_i.  Template example values carry their source
    or are marked UNCERTAIN.
17. **Gate (Q4, Q19), on Si (Tersoff).**
    - **BTE part:** the own fc2 / fc3 are given to phono3py, with the
      same mesh and sigma.  kappa must agree within 0.5 % and the
      linewidths within 1 %.
    - **Whole pipeline:** displacements, fc3 and kappa, against
      phono3py's own pipeline with the same `force_fn`.  kappa must
      agree within 2 %.
    - Recorded only, not gates:
      - LJ Ar classical RTA against the GK of ADR-0020;
      - literature values for Si and AlN.
18. **Template (Q20).**
    - Si Tersoff (the qha template's potential);
    - 2x2x2 conventional supercell (64 atoms);
    - mesh 11^3, sigma 0.1 THz, 100-500 K;
    - `symmetry: true`, quantum occupation.
19. **Tests (Q21).**
    - **In the suite** (at most 60 s of slow tests):
      - group velocity against finite differences in q;
      - fc3 permutation symmetry and ASR;
      - the classical limit;
      - Tamura on an analytic case;
      - a CLI smoke test;
      - a small-mesh phono3py oracle (slow, skipped without phono3py).
    - **Outside the suite**, under nohup:
      - Si on a 19^3 mesh: convergence in sigma and mesh, and timing;
      - LJ Ar RTA against GK;
      - AlN ELM MLIP kappa along a and c.
20. **AlN without NAC (Q22).**
    - The AlN validation runs without NAC.  NAC is checked by unit tests
      and by the oracle.
    - A NAC run is added only if the user supplies VASP LEPSILON
      values.  No literature Born charges are used without a source.
21. **Process (Q7).**
    - This ADR, the implementation in a worktree, version 0.3.0a6.
    - New dependencies only as optional extras.
    - Ask before every push and tag.

## Result (2026-11-22, 0.3.0a6)

### Implementation

- **`akitsushima/solvers/bte.py`, `BTECalculator`:**
  - `compute_fc2` (`PhononCalculator`), `compute_fc3`,
    `set_mass_variance`, `set_boundary_length`, `compute_kappa`,
    `save_results`.
  - `symmetrize_fc3` and `mass_variance` are module functions.
- **fc3:**
  - The first displacements are the ADR-0040 plan of the fc3 supercell
    (amplitude `delta`).
  - Every supercell atom is displaced second by +/- delta (central),
    giving the fc2 of each first-displaced cell.
  - The images of every slab under the supercell-compatible operations
    enter per-atom normal equations, as in ADR-0040's fc2 path.
  - The storage is compact: (n_basis, N_sc, N_sc, 3, 3, 3).
- **Symmetrisation:**
  - Average over the six index permutations, translating so that the
    first atom is in the origin cell.
  - The sum-rule residual sum_k Phi(i, j, k) is put on the k = i block,
    which keeps the support of fc3, and the two steps alternate until
    the residual is below 1e-10.  It falls geometrically: Si 1.7e-7
    after 20 rounds, 8e-14 after 100.
- **Linewidth kernel (numba `prange` over q'):**
  - fc3 in reciprocal space from the non-zero blocks.  Blocks below
    1e-10 of max |fc3| are dropped: Si keeps 872 of 32768.
  - The phases are <exp(i q.R)> over the minimum images seen from the
    first atom.
  - Three BLAS contractions with the eigenvectors; only q' <= q'' is
    computed, with weight 2.
  - In the classical occupation the spontaneous-decay "+1" term is left
    out.  It is O(hbar), so the classical gamma is proportional to T and
    kappa to 1/T exactly.
- **Group velocities:**
  - The analytic dD/dq; the NAC term by finite differences in q.
  - Degenerate sets as phonopy: the basis is fixed by dD along (1, 2, 3).
  - The velocity is averaged over the little group {R : R q = q} with q
    taken in the zone as q - rint(q).  An operation mapping q onto
    q + G is not used, as in phonopy (finding 1).
- **kappa:**
  - The sum over the irreducible points with their star sizes,
    symmetrised over the point group.
  - Cumulative kappa against frequency and against mean free path.
  - `bte_results.json`, `bte_modes.npz`, two CSVs, `bte_kappa.png`.
- **`akitsushima/cli/cmd_bte.py`:**
  - `mass_variance` / `isotopes` / `alloy` (virtual-crystal mass + g),
    `boundary_length_um`, `born` (NAC), `ifc_path`.
  - Template `bte` (Si Tersoff, 8-atom cell, 2x2x2, mesh 11^3).
- **Version:** 0.3.0a6.

### Gate: phono3py 4.1.0 oracle

| Case | Mesh | sigma | kappa rel. diff | gamma rel. diff (max) |
|---|---|---|---|---|
| Si, own fc to phono3py, sym_fc3 off | 5^3 | 0.2 | 1.4e-4 | 7.1e-4 |
| Si, own fc to phono3py, sym_fc3 on | 5^3 | 0.2 | 1.1e-5 | 1.8e-5 |
| Si, own fc, sym_fc3 on | 6^3 | 0.2 | 1.1e-5 | 1.5e-5 (per q point) |
| Si, own fc, sym_fc3 on | 19^3 | 0.1 | 1.3e-5 (297.905 vs 297.909) | - |
| AlN Tersoff, own fc | 5x5x3 / 6x6x3 | 0.2 | 1.2e-5 / 1.3e-5 | 3.9e-4 / 1.2e-4 |
| Si isotope g = 2e-4, own fc | 5^3 | 0.2 | 1.1e-5 | Tamura gamma 1.7e-5 |
| **Si whole pipeline** (phono3py's own 111 supercells, same force_fn), sym off / on | 5^3 | 0.2 | **7.3e-4 / 4.1e-4** | - |

Gate (0.5 % / 1 % / 2 %): passed.

### Suite (`tests/test_bte.py`, 9 tests)

- Group velocity against finite differences in q: 1e-5, and 1e-4 with
  NAC (AlN, model Born charges).
- fc3 permutation symmetry 1e-12 and ASR 1e-8 after symmetrisation;
  O(delta^2) asymmetry before it.
- Classical kappa(300 K) = 2 kappa(600 K) to 1e-10, C = k_B; quantum
  within 2 % of classical at 4000 K.
- Tamura on fcc LJ Ar against the isotropic identity: 1e-10.
- The `mass_variance` / `isotopes` / `alloy` inputs (natural Si g =
  2.01e-4).
- Boundary scattering = |v| / L.
- Slow tests, about 15 s in total:
  - symmetry-reduced against full displacements for AlN: fc3 within
    2e-3, a quarter of the force calls;
  - the phono3py oracle on a 4^3 mesh.
- The template test (`test_template_runs_as_shipped[bte]`, 2.2 s).
- Full suite: 1135 passed, 1 skipped (23.6 min, with the validation
  running at the same time).

### Validation outside the suite (`example/bte/validate_adr0042.py`, one thread)

**Si Tersoff** (qha template; a = 5.432 A, not relaxed), kappa at 300 K
(W/m/K):

| Mesh | sigma 0.05 | sigma 0.1 | sigma 0.2 |
|---|---|---|---|
| 11^3 | | 286.1 | |
| 15^3 | | 293.0 | |
| 19^3 | 304.8 | 297.9 | 287.8 |

- Not converged at 19^3: +1.7 % from 15^3 to 19^3, and 5 % between
  sigma 0.05 and 0.1.  A converged value needs a denser mesh or the
  tetrahedron method (tier C).
- Timing of gamma + kappa at 19^3, one thread:
  - own: 627 s (220 irreducible q; the full suite ran at the same time
    for part of it);
  - phono3py: 1258 s with the same fc2 / fc3.
  - Not a gate.  fc3: 384 force calls, 4 s.

**LJ Ar at 40 K** (a = 5.30 A, the ADR-0020 GK set-up; fcc primitive,
6^3 supercell, 1296 fc3 force calls):

| Mesh | sigma (THz) | Classical RTA | Quantum RTA |
|---|---|---|---|
| 12^3 | 0.05 | 0.660 | 0.621 |
| 16^3 | 0.05 | 0.670 | 0.635 |
| 20^3 | 0.05 | 0.676 | 0.641 |
| 20^3 | 0.025 | 0.694 | 0.658 |
| 20^3 | 0.1 | 0.639 | 0.609 |

- GK (ADR-0020, classical MD): 0.682 +- 0.035 W/m/K.  The classical RTA
  agrees within 1 SE at every converged setting.
- Recorded only: the RTA leaves out the four-phonon processes and the
  normal-process repopulation that GK contains.

**AlN ELM MLIP** (relaxed a = 3.1143, c = 4.9645 A, u = 0.3827):
- fc3 on 3x3x2 (2592 force calls, 35 s with fc2), fc2 on 4x4x3, no NAC.
- kappa (W/m/K) at mesh 14x14x9, sigma 0.05:

| T (K) | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| kappa_a | 246 | 118 | 73.0 | 53.0 | 41.8 |
| kappa_c | 281 | 126 | 77.5 | 56.1 | 44.1 |

- At 300 K this is about a quarter of the measured AlN value (about
  285 W/m/K for high-purity single crystals; literature value from
  memory, UNCERTAIN, not checked against a source).
- The likely causes are not resolved here (finding 3).

### Findings

1. **Degenerate group velocities define kappa on a finite mesh.**
   - Without symmetry, the phonopy (1, 2, 3) basis alone gave Si kappa
     259 / 262 / 263 against 219 with little-group averaging, on a 5^3
     mesh.
   - phonopy averages only over R with R q = q exactly; operations with
     R q = q + G are excluded.  That changes the zone-boundary points of
     even meshes: 154 against phono3py's 223 on 4^3 before the change,
     equal after it.
   - Both conventions converge with the mesh.  The phonopy one is used
     so that the oracle holds.
   - With `symmetry: false` no averaging is done, so kappa on a coarse
     mesh depends on the basis convention.  The test therefore compares
     fc3, not kappa.
2. **phono3py's default finite differences are O(delta).**  AlN Tersoff
   kappa_xx (W/m/K), no symmetrisation:

   | | delta 0.03 A | delta 0.01 A |
   |---|---|---|
   | phono3py | 322.5 | 347.3 |
   | own (central in both steps, O(delta^2)) | 358.3 | 359.2 |

   - Extrapolating phono3py linearly to delta = 0 gives 359.7, within
     0.15 % of the own value.
   - So the 4 % whole-pipeline difference on AlN (2 % gate defined on
     Si) is phono3py's truncation error, not the reconstruction.
3. **The AlN ELM MLIP gives a low kappa.**
   - The fc3 supercell (3x3x2, sides 9.3-9.9 A) is thinner than 2
     r_cut = 10 A, so fc3 is truncated.  PhononCalculator warns about
     this.
   - The model was fitted to a small data set, and its third derivatives
     are not validated.
   - A larger fc3 supercell, and a comparison with DFT fc3, would
     separate the two.  Not done here.
4. **No little-group triplet reduction.**  Speed was already
   comparable to phono3py on one thread, so decision 11's option was not
   taken.

## Consequences

- akitsushima gets a quantum, mode-resolved lattice thermal
  conductivity for ordered crystals, for any of its potentials.
- For alloys it gives only a virtual-crystal estimate: force-constant
  disorder and local relaxations are missing.
- The RTA underestimates kappa where normal processes dominate, as in
  diamond and similar materials.  The full BTE is in tier C.
