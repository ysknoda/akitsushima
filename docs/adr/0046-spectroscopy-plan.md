---
status: accepted (plan agreed 2026-11-24; decisions 5, 15 and 17 revised 2026-11-25 by ADR-0050); S1 done (ADR-0049, 0.3.0a11); S2 lowered; later stages wait for the user's go-ahead
---

# Displacement-based spectroscopy: Raman, IR eps(omega), linewidths, two-phonon DOS and ADPs

Plan agreed with the user in the grilling of 2026-11-24 (Q1-Q17, all as
recommended).  It covers spectra that follow from the harmonic
(eigenmode) picture.  Spectra from time correlations along an MD
trajectory belong to the planned port of nozuchi (below).  Each stage
gets its own ADR (ADR-0049 onwards; ADR-0047 and 0048 went to other
work), like the property-command plan of
ADR-0017.

## Starting point (found by reading the code, 2026-11-24, 0.3.0a8)

- **No engine supplies the Raman input.**  A Raman intensity needs the
  derivative of the electronic polarisability with respect to atomic
  displacement.  Every engine here gives a constant eps_inf: the
  classical potentials, the energy-only MLIPs and the rigid-ion `born:`
  model of ADR-0019.  The Gamma frequencies and the selection rules are
  already available from `phonon`.
- **IR is half there.**  `dielectric` (ADR-0019) gives, per Gamma mode,
  the IR intensity |p_m|^2, the mode effective charge, the eps_0
  contribution, and the LO frequencies along x, y and z.  eps(omega) was
  deferred (ADR-0019 decision 5).
- **The MD VDOS is API-only.**  `DiffusionAnalyzer.compute_vacf_dos()`
  has no CLI.  The `md` trajectories (xdatcar / extxyz / lammps-dump)
  hold no velocities.
- **Linewidths exist.**  `bte` (ADR-0042) gives the three-phonon gamma
  of every mode on its Gamma-centred mesh.  The normalisation is
  phono3py's: gamma is a half width in THz, so the FWHM is 2 gamma.
- **Symmetry is limited.**  spglib supplies the operations only
  (ADR-0040; optional extra).  There are no character tables.
- **Biaxial strain is possible.**  `relax` has `fix_cell`, so the
  in-plane cell can be held fixed.
- **Dense eigensolver cost.**  `numpy.linalg.eigh` of a random symmetric
  3N x 3N matrix took 5.5 s at N = 512, 42 s at N = 1024 and 249 s at
  N = 2048.  The machine was loaded by the ADR-0045 runs, so these times
  are indicative.  The cost scales as about N^3.
- **nozuchi** (/Users/yusuke/Claude_Code/dielectric_analysis, 0.2.0a1)
  is to be ported into akitsushima later.  Its main use there will be
  orientational polarisation from MLIP MD.  It already has:
  - FDT eps(omega) from M(t), with three dipole models (point_charge,
    born_tensor, external);
  - Gamma normal-coordinate mode spectra with damped-oscillator line
    fits;
  - optics (n, kappa, R, absorption, loss function, LO, LST);
  - OUTCAR Born-charge and eps_inf readers;
  - unit-cell -> supercell site mapping;
  - dielectric fits and `nozuchi check`.

  Its ADR-0011 roadmap (0.2.0a2 fixes, 0.3.0a1 improvements) is accepted
  but not implemented.

## Decisions

### Scope (Q1-Q3, Q6)

1. **Level (Q1).**  Stage 1: relative Raman intensities, from Raman
   tensors, polarisation geometries and the powder average.  Then:
   linewidths from the existing BTE gamma at Gamma.  The
   temperature-dependent peak shift needs MD and waits for the port.
2. **Systems (Q2).**
   - Main target: GeSn alloy supercells, that is, how the Ge-Ge, Ge-Sn
     and Sn-Sn modes depend on Sn content and strain.
   - Verification: perfect Si, Ge, AlN, ZnO and MgO.
   - Polar crystals: TO intensities and LO frequencies (the existing
     NAC).  The electro-optic term of the LO intensity is not included.
3. **Split with the nozuchi port (Q3, Q6).**
   - This plan: harmonic Raman, harmonic IR eps(omega) and optics, BTE
     linewidths, the two-phonon DOS, and ADPs U_ij(T).
   - Moved to the port plan (its own grilling): the MD VDOS on the CLI
     with velocity output, MD IR, the spectral energy density (SED, which
     nozuchi's mode projection already is at Gamma), MD Raman, and the
     ML charge model.
   - Tier C: inelastic neutron scattering (neutron-weighted DOS,
     one-phonon S(Q, omega)).
   - Out of scope: powder XRD (not displacement-based) and electronic
     spectra (impossible with classical potentials).
   - Ported from nozuchi now, with their tests: the OUTCAR
     Born-charge / eps_inf readers, the site mapping and the optics
     conversion.  This keeps one implementation of each from the start.
     The conventions follow nozuchi: eps = eps' - i eps'', eps'' > 0 for
     absorption.  The Z* index convention is already the same.
4. **Dependencies (Q5).**
   - Only numpy and scipy are required; spglib stays optional.
   - Raman and IR activity is decided numerically, from the characters
     of the mode representation against those of the vector and
     symmetric rank-2 representations.  There are no character tables
     and no Mulliken labels.
   - phonopy and Phonopy-Spectroscopy are used only as verification
     oracles in example/, and are installed only after asking.

### Physics and models (Q7-Q13)

5. **Response model (Q9).**  A common interface, in the same role as
   `force_fn`: from a structure it returns the dipole M and the
   polarisability alpha, and their displacement derivatives Z* and A.
   - The harmonic path uses the derivatives at equilibrium; the MD path
     of the port uses the values per frame.
   - Implemented here: fixed tensors (Z* from YAML or OUTCAR, A from
     VASP finite differences), point charges, and the bond polarisability
     model.
   - The ML charge model, for orientational polarisation from MLIP MD,
     is the first new item of the port plan.  An ML polarisability is
     tier C.
   - **Revised 2026-11-25 (ADR-0050 decision 4):** the ML charge model
     goes to tier C as well; its labels need extra DFT runs, and the user
     wants little dependence on external first-principles codes.  The
     port uses fixed tensors only (point charges, Z* from `born:`, an
     external dipole series).
6. **Conventions (Q16).**
   - Polarisability derivative A[k, a, b, c] = d alpha_ab / d u_kc, in
     A^2.  The response indices come first, as in
     Z*[k, a, b] = dP_a / du_kb.
   - alpha is the polarisability volume in A^3; from DFT,
     alpha = V (eps_inf - 1) / (4 pi).
   - Mode Raman tensor R_m = sum_k A_k . e_m(k) / sqrt(M_k).  Only
     relative intensities are reported.
7. **Raman-tensor sources (Q7).**
   - (a) **VASP finite differences.**  akitsushima writes +-delta
     structures, the user runs VASP, and akitsushima reads eps_inf back
     from each OUTCAR.  Ge and alpha-Sn have about zero gap in PBE/LDA,
     so they need a hybrid functional.  LEPSILON (DFPT) does not support
     hybrids, so LCALCEPS (finite field) is needed; hybrid + LCALCEPS
     has a known crash on GPUs.  Si (PBE) can use LEPSILON.
   - (b) **Bond polarisability model (BPM).**  Three parameters per
     species pair (nine for GeSn), with analytic derivatives, usable at
     any cell size and later for alpha(t) in MD.  The parameters are
     given, or fitted to (a).
   - For GeSn: (b) fitted to (a) on small cells (Ge, alpha-Sn, ordered
     GeSn), then (b) on the large supercells.
   - Without a `polarizability:` block, `raman` reports activities and
     frequencies only.  For the diamond structure the tensor has one
     constant d, so relative intensities follow from symmetry alone.
8. **Geometry and intensity (Q8).**
   - Single crystal: named (e_i, e_s) pairs, in Cartesian or [hkl]
     directions.  Porto notation is not parsed.
   - Powder: I_par = 45 a^2 + 4 g^2 and I_perp = 3 g^2, from the
     rotational invariants a and g; the total; and rho = I_perp / I_par.
   - Stokes factor (n(omega, T) + 1) / omega, with T = 300 K by default.
     The laser factor (omega_L - omega)^4 is applied only when a
     wavelength is given.  Anti-Stokes is optional.
   - Lorentzian broadening with a constant FWHM (about 3 cm^-1 by
     default), until the BTE stage.
   - Degenerate partners are summed.  Outputs are normalised and raw:
     a mode table (frequency, activity, rho, intensity per geometry),
     .dat, .json and a figure.
9. **Alloy supercells (Q10).**
   - The Gamma dynamical matrix of the N-atom cell itself, from 6N
     displacements; images fold exactly at Gamma.  Whether
     `PhononCalculator` with supercell [1, 1, 1] does this exactly is
     checked in stage 1.
   - Dense diagonalisation, with a guide of about 1000 atoms and a
     warning above it.
   - Several random configurations (`build` seeds) are given as input,
     and the mean and standard deviation are reported.
   - Strain through `relax` with `fix_cell`.
   - Lanczos (Haydock) / KPM spectra without eigenvectors: tier C.
     Very large cells: MD after the port.
10. **Harmonic IR (Q11).**
    - eps_ab(omega) = eps_inf,ab + (4 pi K_E / V) sum_m p_m,a p_m,b /
      (w_m^2 - w^2 + i g_m w), with the analytic (TO) Gamma modes and
      the nozuchi sign.
    - Damping: constant or per mode; BTE gamma in the linewidth stage.
    - Outputs: the eps tensor, n, kappa, R (normal incidence, per
      diagonal component), absorption, and the loss function with its LO
      peaks, through the ported nozuchi optics.  The files are in
      nozuchi's format, so MD results can be overlaid directly.
    - Checks: eps(0) equals the existing eps_0 exactly; LST; KK.
    - A `spectrum:` block of `dielectric`, not a new command.
11. **Two-phonon DOS and ADPs (Q13), as `phonon` options.**
    - Two-phonon DOS: only (q, -q) pairs, since the photon momentum is
      about zero.  g2(omega) = (1/N_q) sum_q sum_nu,nu'
      delta(omega - w_q,nu - w_-q,nu'), weighted by (n + 1)(n' + 1) at
      T.  Overtone and combination are separate columns; difference
      processes are optional.  No intensities and no selection rules,
      and the output says so.
    - U_k = hbar / (2 N_q M_k) sum_q,nu (2n + 1) / w Re[e e*^T] in A^2,
      with U_iso and B = 8 pi^2 U_iso.  Gamma acoustic modes are
      excluded, and a mesh that is not converged is warned about.
      phonopy's thermal displacements are the oracle.

### Input, workflow and verification (Q12, Q14, Q15)

12. **YAML (Q14).**
    - `born:` gains `born_charges: {outcar, unit_cell, supercell}` and
      `epsilon_inf: {outcar}`, with nozuchi's site mapping.  The
      priority is per_atom > per_species > outcar > the potential's
      point charges.
    - A new top-level `polarizability:` block (`source: bpm | vasp_fd`),
      readable later by the MD path.
    - A new `raman` command and section:
      - `supercell`, or `gamma_only: true` for alloy cells, or
        `load_ifc`;
      - geometries, `powder`, `temperature`, `laser_nm`, `fwhm_cm`;
      - a `structures:` list, averaged.
    - `dielectric.spectrum`, `phonon.two_phonon_dos` and `phonon.adp`.
    - `--dump-default raman`, and the existing templates annotated.
13. **VASP finite differences and the BPM fit (Q15).**
    - One `raman` command, run twice, like the auto-continue of `nemd`.
      - First run: write `raman_fd/disp_XXX/POSCAR`,
        `displacements.json` (input hash, atom, direction and delta of
        each displacement) and `NOTE.txt` (the INCAR tags needed, as in
        decision 7; no INCAR is written).  Then stop with exit code 0.
      - Second run: read the OUTCARs when all are present and the hash
        matches.  Otherwise list the missing ones and stop.
    - Displacements: +-0.01 A along each Cartesian direction of each
      atom, reduced by symmetry when spglib is present.
    - BPM: `polarizability: {source: bpm, fit_to: [dirs]}` fits several
      finite-difference sets jointly inside `raman` and writes
      `bpm_params.json`, which `params_file:` reuses.  The BPM fit is not
      part of `fit`.
14. **Verification (Q12).**
    - Automatic tests:
      1. Selection-rule counts.  Diamond: one triply degenerate
         Raman-active, IR-inactive set.  Wurtzite: A1 + E1 + 2E2 Raman,
         A1 + E1 IR, 2B1 silent.  Rock salt: no first-order Raman,
         T1u IR.
      2. Tensor form and invariants.  Diamond off-diagonal d only;
         powder rho = 0.75 exactly for that mode; rotating the structure
         rotates the tensors; permuting the atoms changes nothing.
      3. BPM: dalpha/du against central differences of alpha, and the
         translation sum sum_k A_k = 0 (necessary, not sufficient).
      4. Pure Ge 2x2x2 (16 atoms): the Gamma spectrum equals the
         primitive one, and the folded modes have zero intensity.
    - In example/:
      5. Si with PBE + LEPSILON, the same OUTCARs through
         Phonopy-Spectroscopy: relative intensities within 1 %, plus a
         delta scan.
      6. GeSn Ge-Ge shift against Sn content and strain, compared with
         experiment as information only (no gate).

### Plan, versions and process (Q4, Q17)

15. **Stages.**  All start after ADR-0045 is settled.  Each stage gets
    its own ADR, one alpha bump and a worktree; push and tag are asked
    each time.

    | stage | content | version |
    |---|---|---|
    | S1 | nozuchi shared parts (OUTCAR readers, site mapping), response-model interface, activity from characters, `raman` (BPM with given parameters, geometries, powder, constant FWHM, configuration average), checks 1-4 | 0.3.0a11 (done, ADR-0049) |
    | S2 | VASP finite differences (write / read back), BPM fit; check 5 when the user's VASP runs exist (does not block later stages) | 0.3.0a12 |
    | S3 | `dielectric.spectrum` (eps(omega), optics via the ported nozuchi optics) | 0.3.0a13 |
    | S4 | BTE linewidths for Raman and IR (`load_bte`) | 0.3.0a14 |
    | S5 | two-phonon DOS and U_ij in `phonon` | 0.3.0a15 |

    The versions were planned one lower; 0.3.0a10 went to ADR-0047
    (classical force-field speed-ups), so every stage moved up by one.

    **Revised 2026-11-25 (ADR-0050):** the user wants akitsushima to be
    usable with little dependence on external first-principles codes.
    S2 (VASP finite differences, BPM fit) is lowered and waits; the
    nozuchi port (ADR-0050, versions 0.4.0aN) comes first.  S3 follows
    P1 of the port, since it uses the ported optics; S4 and S5 fit
    between the port stages.  The version numbers above are void; each
    stage takes the next free alpha when it starts.  For S2, if resumed:
    the VASP wiki gives the LCALCEPS header without "in DFT", which the
    reader must accept (ADR-0050 decision 7).
    | application | GeSn Raman (BPM fitted to hybrid data, composition and strain), runs and report in example/ | -- |

    - A short grilling per stage only when open decisions remain.
    - README, templates, CONTEXT.md and example/ are updated in each
      stage.
16. **CONTEXT.md terms (Q16), added in S1.**
    - Response model.  _Avoid_: charge model, dielectric model.
    - Polarisability derivative.  _Avoid_: Raman tensor, Raman
      susceptibility.
    - Raman tensor: the mode 3x3 only.
    - Activity: the symmetry yes/no.  The scalar 45 a^2 + 7 g^2 (what
      Gaussian and Phonopy-Spectroscopy call "Raman activity") equals
      I_par + I_perp and is called the **powder intensity** here.
    - Polarisation geometry.  _Avoid_: Porto notation.
    - BPM.
    - Two-phonon DOS.  _Avoid_: joint DOS, which in phono3py means
      general q' + q'' = q.
    - ADP U_ij.  _Avoid_: Debye-Waller factor, which is the exponential.
17. **Order relative to nozuchi's own roadmap (asked after Q17).**
    Recommended order:
    1. nozuchi 0.2.0a2 (its ADR-0011 fixes) first, in its own
       repository.  S1 and S3 copy `charges.py` and `optics.py`, which
       its items A8 and D4 change.  Its A1 (the improper NPT dipole term)
       is a correctness bug in the code that will be ported.
       akitsushima cannot start S1 before ADR-0045 is settled anyway.
       Its Tier 2a / 3a regression competes for CPU with the ADR-0045
       runs, so it is best run after them.
    2. This plan, S1-S5.
    3. The port grilling, which also decides whether nozuchi 0.3.0a1 is
       done in nozuchi or folded into the port.

    **Revised 2026-11-25 (ADR-0050 decision 2):** nozuchi 0.2.0a2 was
    done first, and S1 followed.  nozuchi 0.3.0a1 (B1-B8) is folded into
    the port, after its equivalence stage, and nozuchi is frozen at
    0.2.0a2.

## Self-critique of the chosen values

| value | source | confidence |
|---|---|---|
| I_par = 45 a^2 + 4 g^2, I_perp = 3 g^2; activity 45 a^2 + 7 g^2 | rotational invariants of a symmetric tensor (D. A. Long, The Raman Effect, Wiley 2002) | > 95 % |
| (q, -q) pairs only in the second-order DOS | photon momentum << zone size | > 95 % |
| LEPSILON has no hybrid support; LCALCEPS works with hybrids; GPU crash with hybrid + LCALCEPS | VASP wiki LEPSILON and Known issues pages, read 2026-11-24 | > 95 % |
| Ge about zero gap in PBE/LDA; alpha-Sn a semimetal | common DFT knowledge | ~90 % |
| BPM with three parameters per pair | form of Umari, Pasquarello and Dal Corso, PRB 63, 094305 (2001) | ~85 % (recalled); UNCERTAIN whether reliable Ge-Sn literature values exist -- the fit to (a) avoids depending on them |
| default FWHM about 3 cm^-1 | room-temperature Si / Ge line widths as commonly quoted | ~70 %, UNCERTAIN; a measured linewidth table (e.g. Balkanski, Wallis and Haro, PRB 28, 1928 (1983), Si) resolves it; only a display default |
| displacement 0.01 A for d eps / du | the `phonon` default; not yet checked for eps | ~60 %, UNCERTAIN; the delta scan of check 5 resolves it |
| about 1000 atoms for dense Gamma | eigh timings above (loaded machine) | ~80 % as a guide; MLIP force calls (O(N^2) in total) may be the real limit |
| 1 % agreement with Phonopy-Spectroscopy | the same OUTCARs; only the contraction and averaging differ | ~85 % as attainable |

## Consequences

- ADR-0028 addendum 8 points to this plan.
- S1 (ADR-0049, 0.3.0a11): `gamma_only` became `supercell: auto`, since
  the Gamma matrix of a cell is exact from any supercell the potential
  accepts (ADR-0049 decision 3).
- The nozuchi port plan (its own grilling) receives the VDOS, MD IR,
  SED, MD Raman and ML charge-model items, and the decision on nozuchi
  0.3.0a1.

Sources: [VASP wiki LEPSILON](https://vasp.at/wiki/LEPSILON),
[VASP wiki Known issues](https://www.vasp.at/wiki/Known_issues),
[VASP forum, LCALCEPS with a hybrid](https://www.vasp.at/forum/viewtopic.php?p=18672).
