---
status: accepted (2026-11-25, 0.3.0a11)
---

# Raman spectra in the harmonic approximation (ADR-0046 stage S1)

First stage of the spectroscopy plan of ADR-0046.  Started on the user's
go-ahead (2026-11-25) once ADR-0045 was settled (0.3.0a9, pushed and
tagged) and nozuchi 0.2.0a2 was done (nozuchi commit 84dec4c), as
ADR-0046 decision 17 asked.

## Numbering

ADR-0046 planned S1 as ADR-0047, version 0.3.0a10.  Both went to the
classical force-field speed-ups (ADR-0047, 0.3.0a10), and ADR-0048 is
reserved for the thermo labels of that plan.  S1 is therefore ADR-0049,
0.3.0a11, and the later stages move up by one version: S2 0.3.0a12,
S3 0.3.0a13, S4 0.3.0a14, S5 0.3.0a15.

## Decisions

1. **Ported from nozuchi 0.2.0a2** (akitsushima/io/outcar_dielectric.py):
   `read_outcar_eps_inf`, `read_outcar_born`, `map_to_unit_cell` and
   `rigid_shift`, unchanged in behaviour and tolerances, with their tests
   and test data (tests/data/nozuchi).  akitsushima's own POSCAR reader is
   used.  `born:` gains `epsilon_inf: {outcar}` and `born_charges:
   {outcar, unit_cell, supercell, map_tolerance}`; per atom the priority
   is per_atom > per_species > outcar > the potential's point charges.
   Every command with a `born:` block (phonon, gruneisen, qha, bte,
   dielectric, piezo, raman) passes its structure for the site mapping.
2. **Response model** (solvers/response.py), the interface of ADR-0046
   decision 5: `born_charges`, `dipole` (displacement form, the
   displacements as given), `polarizability`, `polarizability_derivative`,
   with `has_charges` / `has_polarizability`.  Implemented:
   - `FixedResponse`: Z* and / or A given per atom (A from VASP finite
     differences in S2);
   - `BondPolarizabilityModel`: per bond alpha = abar(R) I + D(R) (n n^T -
     I/3), abar = (d_alpha_l + 2 d_alpha_p)/3 (R - r0), D = alpha_diff +
     (d_alpha_l - d_alpha_p)(R - r0), the form of Umari, Pasquarello and
     Dal Corso (PRB 63, 094305 (2001)).  The keys per species pair are
     r_cut, r0, alpha_diff [A^3], d_alpha_l, d_alpha_p [A^2].  alpha is
     defined up to an isotropic constant per bond, which no derivative
     sees.  Bonds are searched over all periodic images, so any cell size
     works.  The bond cutoff is sharp: fine at equilibrium, but the MD path
     of the port needs a smooth one (left to the port).
   - `polarizability: {source: vasp_fd}` and `fit_to` are refused with a
     pointer to S2.
3. **The Gamma matrix: `supercell: auto` instead of `gamma_only`.**
   ADR-0046 decision 12 planned `gamma_only: true` for alloy cells.  But
   the Gamma dynamical matrix of a cell is exact from any supercell that
   the potential accepts: the images of a displaced atom fold onto the
   same atom with phase 1.  `PhononCalculator` with any supercell gives
   it.  So `raman` has `supercell: auto` (the default): the smallest
   replication whose perpendicular widths exceed 2 (r_cut + 0.5 A skin).
   Alloy supercells of 64 atoms or more get [1, 1, 1].  An explicit
   `supercell` and `load_ifc` remain.  Measured on Tersoff Si: the Gamma
   frequencies from 2x2x2 to 5x5x5 agree to 1.2e-11 cm^-1.
4. **Activity from characters** (solvers/raman.py).
   - Degenerate sets are frequencies within `degeneracy_cm` (0.1 cm^-1).
   - chi_S(g) = sum_m e_m^T G(g) e_m over the spglib operations of the
     cell; n_IR = <chi_S, tr R>, n_Raman = <chi_S, (tr(R)^2 + tr(R^2))/2>.
   - The largest deviation of a multiplicity from an integer is printed:
     a split degenerate set, or a structure that is not symmetric to
     symprec, shows there.  Merging two irreps into one set is harmless.
   - The three modes of smallest |w| are acoustic (warned above
     1 cm^-1).
   - `activity: auto` computes it when spglib is installed; `true`
     requires spglib; `false` skips it.
5. **Intensities.**
   - R_m = sum_k A_k e_m(k) / sqrt(M_k) (symmetrised).
   - Single crystal: (e_s . R . e_i)^2.  Powder: I_par = 45 a^2 + 4 g^2,
     I_perp = 3 g^2, rho, and the powder intensity 45 a^2 + 7 g^2.
   - The sum is over the degenerate set.
   - Stokes factor (n + 1) / w, with c2 = hc/k_B = 1.438776877 cm K
     (CODATA 2018), or anti-Stokes n / w.  (w_L -+ w)^4 is applied only
     with `laser_nm`.
   - Area-normalised Lorentzians of constant FWHM (`fwhm_cm`, 3 by
     default).
   - Normalisation: the powder columns are divided by the powder maximum,
     and all geometry columns by the largest geometry maximum.  Found
     while checking the first figure: per-column normalisation drew the
     forbidden Si z(xx) geometry (1e-30) as a full peak.
6. **Configurations** (`raman.structures`): each is relaxed (unless
   `pre_relax: false`) and analysed on its own.  The spectra are averaged
   per atom, with the standard deviation (ddof 1), on one common grid.
7. **Polar crystals:** with `born:`, the LO frequencies along x, y and z
   are listed (NAC).  The electro-optic LO intensity term is not included
   (ADR-0046 decision 2).
8. **Output.**
   - Files: raman_modes.dat (one per structure), raman_spectrum.dat
     (raw), raman_spectrum_normalised.dat, raman_results.json (with the
     mode Raman tensors) and raman_spectrum.png.
   - Above 40 sets the console prints the 20 strongest.
   - Above 1000 atoms the cost of the dense diagonalisation is warned
     about (ADR-0046 decision 9).

## Verification (tests/test_raman.py, 18 tests; full suite 1208 passed, 1 skipped)

- **Check 3, BPM.**
  - dalpha/du against central differences (h = 1e-5 A), on a rattled
    8-atom Ge/Sn cell with three pairs and on a 2-atom primitive cell
    whose bonds all cross the boundary: agreement to 1e-10 (test at
    1e-8).
  - The translation sum is zero to 2e-16, and A is symmetric in (a, b).
  - The bond counts are 16 and 4.
- **Mode tensor.**
  - R_m equals the central difference of the BPM alpha along the normal
    coordinate u_k = e_m(k) Q / sqrt(M_k).
  - Tested on every optical mode of wurtzite AlN (two masses) to 1e-8.
- **Check 1, selection rules.**
  - Diamond (Tersoff Si): one triplet, Raman yes and IR no, at
    535.96 cm^-1.
  - Wurtzite (Tersoff AlN, u = 0.382 and c = 5.45 A, off the ideal
    tetrahedron): 2 E2 Raman only, A1 and E1 both active, 2 B1 silent.
    The multiplicity deviation is below 1e-6.  At the ideal u and c/a,
    A1 and E1 (TO) lie 0.007 cm^-1 apart and merge into one set (still
    correct counts), and the BPM gives the A1 tensor no trace.
  - Rock salt (a Born-Mayer model, used only for its symmetry): one
    triplet, IR yes and Raman no, with intensity below 1e-20.
  - Raman-inactive sets carry below 1e-10 of the strongest intensity.
    This is the O(delta^2) symmetry breaking of Cartesian displacements
    (ADR-0040).
- **Check 2, tensor form.**
  - The Si mode tensors have no diagonal (below 1e-12 relative).
  - rho = 0.75 to 1e-12, and z(xx) is zero.
  - Rotating the crystal (two angles) leaves the powder intensity and
    the rotated geometries unchanged to 1e-8.  The frequency moves by
    4e-3 cm^-1, because the displacement axes do not turn with the
    crystal (O(delta^2)).
  - Reversing the atom order leaves everything unchanged to 1e-10.
- **Check 4, folding.**
  - The 16-atom 2x2x2 replica of the Si primitive cell: one Raman-active
    triplet at the primitive frequency (1e-3), with the same intensity
    per atom (1e-6); 2x2x2 and 4x4x4 supercells of the primitive cell give
    its optical Gamma frequencies equal to 1e-8 (measured 1e-11).
  - The 42 folded modes are inactive by characters, with below 1e-12 of
    the intensity.
  - Tersoff Si is used instead of Ge (ADR-0046 named Ge): the structure
    and the test are the same, and the tests carry no cited Ge parameter
    set.
- **Ported readers.** OUTCAR readers, site mapping (shuffled, displaced,
  rigidly shifted 2x2x2), `born_from_config` with the OUTCAR on a 2x1x1
  cell: the priority, the sum rule (correction (3.0 - 2.6898) / 16) and
  eps_inf from the OUTCAR.
- **CLI** (`slow`).
  - Two identical configurations average with zero std.
  - A [110] direction given in the lattice basis equals its Cartesian
    form.
  - The normalised z(xx) column stays at 0.
  - Unknown keys are refused.
  - The template runs as shipped (test_cli_templates).
- **By hand.**
  - Si template: 0.2 s; the figure was inspected.
  - 64-atom rattled Si, two configurations: P1, all modes active,
    multiplicity deviation 2e-14, mean +- std figure inspected.
- **Not part of S1.** Checks 5 (Si PBE LEPSILON against
  Phonopy-Spectroscopy) and 6 (GeSn against experiment) need the VASP
  stage (S2) and the application run.

## Self-critique of the chosen values

| value | source | confidence |
|---|---|---|
| I_par = 45 a^2 + 4 g^2, I_perp = 3 g^2, rho = 3/4 for a traceless tensor | rotational invariants (Long, The Raman Effect, 2002); checked by the diamond rho and by hand on diag(1, 0, 0) (rho 1/3) | > 95 % |
| (n + 1)/w, c2 = 1.438776877 cm K | Bose factor of a harmonic normal coordinate; CODATA 2018 | > 95 % |
| degeneracy 0.1 cm^-1 | FD symmetry breaking seen here: 4e-3 cm^-1 (rotated Si); a merged pair still gives integer multiplicities | ~85 %; a dense alloy spectrum merges some near-degenerate modes (intensities are summed, nothing is lost) |
| acoustic warning at 1 cm^-1 | relaxed cells give 1e-3 cm^-1 here | ~80 % as a threshold |
| auto supercell, skin 0.5 A | the relax neighbour-list skin; exactness at Gamma measured (1.2e-11 cm^-1) | > 95 % |
| map_tolerance 0.6 A | nozuchi's default | ~80 % (for thermal MD frames; here only relaxed cells are mapped) |
| template BPM values (alpha_diff 1.0 A^3, d_alpha_l 2.0, d_alpha_p 0.5 A^2) | ILLUSTRATIVE, not physical; diamond's relative intensities do not depend on them | UNCERTAIN by design; S2 fits them to VASP finite differences |
| FWHM 3 cm^-1 default | as ADR-0046 (display default) | ~70 %, UNCERTAIN |

## Consequences

- New: akitsushima/io/outcar_dielectric.py, solvers/response.py,
  solvers/raman.py, cli/cmd_raman.py; `--dump-default raman`; the
  dielectric template lists the OUTCAR keys.  README and CONTEXT.md
  (spectroscopy terms of ADR-0046 decision 16) updated.
- ADR-0046: S1 done; the version column moves up by one (above).
- Next: S2 (VASP finite differences, BPM fit), on the user's go-ahead.
