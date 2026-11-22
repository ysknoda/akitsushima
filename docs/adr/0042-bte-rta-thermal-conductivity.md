---
status: proposed (2026-11-22)
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

## Consequences

- akitsushima gets a quantum, mode-resolved lattice thermal
  conductivity for ordered crystals, for any of its potentials.
- For alloys it gives only a virtual-crystal estimate: force-constant
  disorder and local relaxations are missing.
- The RTA underestimates kappa where normal processes dominate, as in
  diamond and similar materials.  The full BTE is in tier C.
