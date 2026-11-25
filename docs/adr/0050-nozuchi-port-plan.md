---
status: accepted (plan agreed 2026-11-25); P1 done (ADR-0051, 0.4.0a1); each later stage waits for the user's go-ahead
---

# Port of nozuchi into akitsushima: MD spectra (`mdspec`)

Plan agreed with the user in the grilling of 2026-11-25 (Q1-Q21, all as
recommended).  It takes the place, for spectra from MD time
correlations, that ADR-0046 holds for harmonic spectra.  Each stage gets
its own ADR (ADR-0051 onwards).

## Context

- The user wants akitsushima to be usable with little dependence on
  external first-principles codes (VASP included).  S2 of ADR-0046
  (VASP finite-difference Raman tensors, BPM fit) was therefore moved
  down, and the port was brought forward.
- nozuchi 0.2.0a2 (/Users/yusuke/Claude_Code/dielectric_analysis,
  commit 84dec4c): 3749 lines in 16 modules, 138 tests (about 70 s).
  Dependencies: numpy, scipy, matplotlib, PyYAML, numba (optional).
  - It computes eps(omega) = eps' - i eps'' from the total dipole M(t)
    by the fluctuation-dissipation theorem.
  - Dipole models: point charges, Born tensors, or an external series.
  - Static estimators: mode, fsum and raw, plus a force-based value.
  - Also: mode-resolved spectra with damped-oscillator fits, optics,
    seven fit models and `check`.
  - Validation: Tier 1 (analytic), Tier 2/2a (akitsushima classical MD
    of MgO, ZnO and TiO2) and Tier 3/3a (VASP AIMD of SiC, ZnO, ice XI
    and urea), all within 1.5 % of the force-based reference.  The data
    are in its local, untracked `validation/` (67 GB).
  - Its ADR-0011 0.3.0a1 items B1-B8 are not implemented.
- akitsushima side:
  - `md` writes xdatcar / extxyz (with forces) / lammps-dump trajectories
    without velocities.
  - `DiffusionAnalyzer.compute_vacf_dos` runs its own MD (API only).
  - Gamma modes can be computed in-house (ADR-0049); S1 already ported
    the OUTCAR readers and the site mapping.

## Decisions

### Form and scope

1. **Full port (Q1).**  A subpackage `akitsushima/mdspec/`.  nozuchi is
   frozen at 0.2.0a2: bug fixes only, and its README points to
   akitsushima once P1 has passed.  That README change is made in the
   nozuchi repository, after asking.
2. **nozuchi 0.3.0a1 is folded into the port (Q2).**  Items B1-B8 are
   done in akitsushima, after the equivalence stage.  B4 (`check` reads
   TEBEG / POTIM) and B8 (POMASS) are VASP-specific and are rebuilt
   around akitsushima's MD output.
3. **Trajectory formats (Q3).**
   - akitsushima's three formats: xdatcar, extxyz and lammps-dump.
   - VASP AIMD XDATCAR and OUTCAR (temperatures, forces) stay readable.
     That is analysis of existing data, not a dependency, and Tier 3a
     needs it.
   - extxyz with forces is the standard input for MLIP MD.
4. **Charge models; no ML charge model now (Q6).**
   - Fixed tensors only, through the response models of ADR-0049: point
     charges, Z* from `born:` (YAML or OUTCAR), and an external dipole
     series.
   - The ML charge model goes to tier C: its labels need extra DFT runs.
     This revises ADR-0046 decision 5.
5. **Gamma eigenvectors for the mode estimator (Q7).**  Computed by
   default with the same potential as the MD.  OUTCAR (IBRION 5-8),
   phonopy qpoints.yaml and .npz remain optional inputs.
6. **MD Raman (Q9)** (alpha(t) from the BPM) is the last stage.  Peak
   positions and widths come earlier, from VDOS and SED, without alpha;
   only the intensities wait for BPM parameters.

### Structure and input

7. **Layout (Q10).**
   - nozuchi's modules are kept: trajectory, dipole, spectrum, modes,
     modespec, optics, fitting, check, output and config.
   - The OUTCAR readers are one implementation, akitsushima's
     io/outcar_dielectric.py.  It is extended to the LCALCEPS header
     "... (including local field effects)", without "in DFT", from the
     VASP wiki LCALCEPS page (read 2026-11-25).
   - XDATCAR uses nozuchi's streaming reader (O(N_atoms) memory,
     optional numba).  extxyz and lammps-dump get streaming readers too.
   - Under MPI, rank 0 only.
8. **Command and YAML (Q4, Q11).**
   - One post-processing command, `akitsushima mdspec input.yaml`, which
     reads the trajectory once for every requested analysis.
   - The YAML follows akitsushima: `born:` and `potential:` as in other
     commands, and the analysis in an `mdspec:` section that keeps
     nozuchi's trajectory / spectrum / modes / forces / fit / check
     content.
   - `akitsushima mdspec --convert-nozuchi in.yaml` translates an existing
     nozuchi input.
   - Python API: `akitsushima.mdspec.run(cfg)`.
9. **Charge model default (Q16).**  `dipole.model: auto` takes, in
   order, `born:` (per_atom > per_species > outcar), then the potential's
   point charges, then `mdspec.dipole.charges`; otherwise it is an
   error.
   - Neutrality is checked.
   - With fixed charges, a warning says that electronic-polarisation
     fluctuations are missing.
   - point_charge, born_tensor and external remain explicit choices.
10. **Gamma modes of large MD cells (Q17).**
    - With `mdspec.modes: {unit_cell, supercell}`, the unit cell's Gamma
      modes are computed with the potential (supercell auto,
      pre-relaxed) and tiled onto the MD cell by site mapping.
    - Without a unit cell (alloys, disorder), the MD cell itself is
      diagonalised at its mean positions when N <= 1000.  Above that the
      mode estimator is not available: fsum / raw, with a warning.
11. **Velocities (Q5, Q18).**
    - `md` gets `traj_velocities: true` (default false).
    - extxyz: `velocities:R:3` in A/fs.
    - lammps-dump: `vx vy vz` in A/ps (LAMMPS metal units).
    - XDATCAR: no velocities.
    - Without velocities, central differences of the positions are used,
      with a Nyquist warning.

### New analyses

12. **VDOS (Q14).**
    - |FFT v|^2 averaged over atoms (Wiener-Khinchin; equal to going
      through the VACF), mass-weighted.
    - Normalised to 3N modes; partial per species and per site set.
    - Window and resolution are options.  Classical, no quantum
      correction.
    - Optional overlay of the harmonic `phonon` DOS of the same
      potential.
13. **`DiffusionAnalyzer.compute_vacf_dos` (Q15)** keeps its API and
    uses the common VDOS function.  Any numerical change is recorded in
    the stage ADR.
14. **SED (Q13).**
    - Spectral energy density without eigenvectors (Thomas et al., PRB
      81, 081411 (2010)), at q-points commensurate with the MD
      supercell, along a given path.
    - Lorentzian fits give each peak's frequency and linewidth.
    - Alloys use the reference (virtual-crystal) lattice positions.
    - nozuchi's eigenvector projection at Gamma is kept.
15. **Raman temperature shift (Q20).**
    - Perfect crystals: the spectra projected on the Raman-active Gamma
      modes (activity by characters, ADR-0049), with damped-oscillator
      fits of frequency and width against T.
    - Alloys: the q = 0 SED as a proxy, labelled as not Raman-weighted.
      The Raman weighting comes with P6.

### Verification

16. **Equivalence first (Q8).**  P1 ports without behaviour changes.
    - Gate: agreement with nozuchi 0.2.0a2 on the same inputs, to 1e-10
      relative.
    - Coverage: the 138 ported tests, Tier 2a and Tier 3a (the local
      validation/ data).
    - Improvements come only after this gate.
17. **VDOS / SED (Q19), no DFT.**
    - Automatic tests:
      - harmonic-potential MD: VDOS peaks = `phonon` DOS;
      - low-T Tersoff Si: SED peaks = `phonon` dispersion at
        commensurate q, within 1 %;
      - normalisation: 3N, and the partials sum to the total;
      - finite-difference and true velocities agree below Nyquist.
    - example/: Si SED linewidths at 300 K against `bte` 2 gamma, as
      information only (classical MD against quantum occupations).

### Stages, versions and documents

18. **Stages (Q12).**  Each stage gets its own ADR, an alpha bump and a
    worktree; push and tag are asked each time.

    | stage | content | version |
    |---|---|---|
    | P1 | equivalence port: `mdspec` with XDATCAR and external modes, nozuchi YAML converter, 138 tests, Tier 2a/3a equivalence; CONTEXT.md terms merged | 0.4.0a1 |
    | P2 | integration: in-house Gamma modes (decisions 5, 10), extxyz / lammps-dump readers, `md` velocities, `born:` and response models | 0.4.0a2 |
    | P3 | nozuchi 0.3.0a1 items B1-B8 | 0.4.0a3 |
    | P4 | VDOS and SED | 0.4.0a4 |
    | P5 | Raman temperature shift | 0.4.0a5 |
    | P6 | MD Raman (BPM alpha(t)) | 0.4.0a6 |

    S3 of ADR-0046 (harmonic IR eps(omega), using the ported optics)
    comes after P1.  S4 and S5 fit between stages.  S2 waits (decision
    19).
19. **Documents (Q21).**
    - ADR-0046 decisions 5, 15 and 17 are revised, and S2 is lowered.
    - ADR-0028 addendum 9 points here.
    - nozuchi's CONTEXT.md terms are merged into akitsushima's in P1.
      On a clash akitsushima's term wins, and the mapping is recorded in
      the P1 ADR.

## Self-critique

| value | source | confidence |
|---|---|---|
| 1e-10 relative equivalence | same code and inputs; only reader / config plumbing changes | ~90 %; summation-order changes (e.g. a shared reader) may give ~1e-14 relative, still inside |
| N <= 1000 for direct Gamma modes | ADR-0046 eigh timings (42 s at 1024) | ~80 % as a guide |
| SED-dispersion agreement within 1 % at low T | harmonic limit; resolution set by run length | ~80 %; UNCERTAIN until measured (run length decides) |
| LCALCEPS header without "in DFT" | VASP wiki LCALCEPS page; the LRPA page differs | ~70 %, UNCERTAIN; a real LCALCEPS OUTCAR resolves it (both headers are accepted) |
| version 0.4.0a1 for P1 | a new command family of nozuchi's size | convention, user agreed |

## Consequences

- akitsushima gains a second analysis family (MD spectra), with
  nozuchi's validation carried over by the equivalence gate.
- nozuchi is frozen after P1.
