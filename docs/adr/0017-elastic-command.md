---
status: accepted (implemented and verified 2026-10-08)
---

# `akitsushima elastic`: elastic tensor, polycrystalline averages and internal strain

First stage of the property-command plan agreed with the user on
2026-10-08 (grilling Q1-Q35).  Stage order: elastic (this ADR) ->
phonon thermodynamics, Grueneisen and QHA (ADR-0018) -> dielectric, NAC
and piezoelectric response (ADR-0019) -> NEMD and the decomposition of
the Green-Kubo conductivity (ADR-0020).  Every new command reads one YAML
file, has a `--dump-default` template and writes `<cmd>_results.json`
(keys carry their units), `.dat` tables and a printed summary (Q2, Q25).

## Starting point (found by reading the code, 2026-10-08)

`solvers/elastic.py` already had `ElasticConstantsCalculator` (one +-delta
central difference of the stress per Voigt mode, ions relaxed at each
strained cell) returning C and the Voigt bulk modulus, and
`fit_equation_of_state` (third-order Birch-Murnaghan).  No CLI command
reached either.

## Decisions (agreed with the user, 2026-10-08)

1. **Command (Q2).**  New sub-command `elastic`.  The calculator stays in
   `solvers/elastic.py`; the moduli are a separate pure function
   `elastic_moduli(C, ...)` so they can be checked on their own.
2. **Pre-relaxation (Q11).**  Cell and ions are relaxed first
   (`elastic.pre_relax: true`, settings under `elastic.relax`); a
   residual stress above 0.1 GPa after it is reported as a warning.  A
   cutoff (plus the optimiser's skin) at or past half the smallest
   perpendicular width of any strained cell is refused before any force
   call; there is no automatic supercell (the user gives the supercell,
   as for `phonon`).
3. **Strains (Q4).**  Each of the six Voigt modes at +-0.25 %, +-0.5 %
   and +-1 % (engineering shear strain for modes 4-6), plus the unstrained
   cell; each column of C is the slope of a straight-line least-squares
   fit of the six stress components against the strain, and the RMS fit
   residual per mode is reported (GPa).  The fitted intercept is the
   residual stress.  A non-linear response shows as a residual that grows
   with the largest amplitude.
   *Source and confidence.*  The Materials Project elastic workflow (de
   Jong et al., Sci. Data 2, 150009 (2015)) uses +-0.5 % and +-1 %
   normal and shear strains with a linear stress-strain fit.  The 0.25 %
   point is added here to see the linear regime from below.  Confidence
   about 85 % that these are the published amplitudes (UNCERTAIN, see
   the self-critique below).  They are defaults, not fixed: `elastic.strains`.
4. **Clamped- and relaxed-ion tensors (Q4, Q29).**  The clamped-ion C^0
   is evaluated at affinely strained cells (fractional coordinates fixed);
   the relaxed-ion C after relaxing the ions in each strained cell
   (`fmax` 1e-4 eV/A by default).  With one atom per cell they coincide.
5. **Internal strain (Q29).**  du/d(eta_j) [A per unit Voigt strain],
   shape (N, 3, 6), from the relaxed positions: displacement from the
   affine position (minimum image in the strained cell, mean removed),
   fitted against strain like the stress.  It feeds the relaxed-ion
   piezoelectric tensor of ADR-0019.  The analytic route
   du/deta = Phi^+ Lambda (Lambda_{ia,j} = dF_ia/deta_j at clamped ions,
   Phi the force-constant matrix of the cell, acoustic modes removed by the
   pseudo-inverse) is used only as a test, together with
   C = C^0 - (1/V) Lambda^T Phi^+ Lambda.
6. **Moduli (Q4, Q10, Q21).**  S = C^-1; Voigt and Reuss bounds of B and
   G (Hill, Proc. Phys. Soc. A 65, 349 (1952)):
   - B_V = [(C11+C22+C33) + 2(C12+C23+C31)] / 9
   - G_V = [(C11+C22+C33) - (C12+C23+C31) + 3(C44+C55+C66)] / 15
   - B_R = 1 / [(S11+S22+S33) + 2(S12+S23+S31)]
   - G_R = 15 / [4(S11+S22+S33) - 4(S12+S23+S31) + 3(S44+S55+S66)]
   - Hill: B_H = (B_V+B_R)/2, G_H = (G_V+G_R)/2.
   For each of V, R, H: E = 9BG/(3B+G), nu = (3B-2G)/(2(3B+G)) from that
   scheme's own B and G (E_H is not the mean of E_V and E_R; this is also
   what pymatgen's `ElasticTensor` does).
   Also: Pugh ratio B_H/G_H (Pugh, Phil. Mag. 45, 823 (1954); 1.75 is
   the usual ductile/brittle divide, printed for information only);
   universal anisotropy A^U = 5 G_V/G_R + B_V/B_R - 6 (Ranganathan and
   Ostoja-Starzewski, Phys. Rev. Lett. 101, 055504 (2008); zero for a
   locally isotropic crystal); Born stability as "all six eigenvalues of
   the Voigt C are positive", the necessary and sufficient condition for
   an unstressed crystal of any symmetry (Mouhat and Coudert, Phys. Rev.
   B 90, 224104 (2014)).  Directional E(n), G(n, m), nu(n, m) are deferred
   (Q10).
7. **Sound velocities and Debye temperature (Q4).**  From the Hill moduli
   and the density rho (masses from the YAML or the built-in table):
   v_l = sqrt((B_H + 4 G_H / 3) / rho), v_t = sqrt(G_H / rho),
   v_m = [(2 / v_t^3 + 1 / v_l^3) / 3]^(-1/3),
   Theta_D = (h / k_B) (3 n / (4 pi))^(1/3) v_m, with n the number density
   of atoms (Anderson, J. Phys. Chem. Solids 24, 909 (1963), eq. (6),
   written per atom instead of per formula unit and molar mass, which is
   the same quantity).
8. **Refusals (Q32).**  ESM electrostatics is refused (straining the cell
   moves the ESM boundaries, as for NPT in ADR-0008).  An MLIP must know
   every element of the structure (ADR-0007).  A force function whose
   stress is identically zero at every strain (an external engine that
   returns none) ends with an error instead of C = 0.
9. **Parallelism (Q20).**  `n_jobs` worker threads over the strained
   cells, as in `phonon`; each cell's relaxation writes its scratch files
   in its own directory.
10. **Verification (Q8).**  pymatgen and ASE are test-only oracles
    (skipped when absent), never dependencies.

## Self-critique of the chosen values

| value | source | confidence |
|---|---|---|
| strains 0.25 / 0.5 / 1 % | de Jong et al. 2015 (0.5 / 1 %), 0.25 % added | ~85 %, UNCERTAIN: the exact amplitude list of the paper's workflow; resolved by its Methods section |
| residual-stress warning 0.1 GPa | the `relax` default `pmax` (0.1 GPa) | a convention, not a physical bound |
| ion fmax 1e-4 eV/A | existing `ElasticConstantsCalculator` default | the internal-strain test (decision 5) measures whether it is tight enough |
| Hill / Reuss / Voigt formulas | Hill 1952; identical in pymatgen | > 95 %, checked numerically against pymatgen |
| A^U | Ranganathan and Ostoja-Starzewski 2008 | > 95 % |
| Born criterion | Mouhat and Coudert 2014 | > 95 % |
| Theta_D formula | Anderson 1963 | > 90 %, checked by hand below |
| Pugh 1.75 | Pugh 1954 | ~80 %; printed for information only |

## Verification results (2026-10-08)

`example/elastic/check_elastic.py`: 14 / 14 checks pass.

- **Tersoff Si** (64-atom cell, jittered by 0.03 A, pre-relaxed by the
  CLI): C11 142.53, C12 75.41, C44 69.01, C44^0 118.82 GPa against
  Tersoff's quoted 142.5, 75.4, 69, 118.8; Kleinman zeta 0.675 (quoted
  0.67).  Those quoted values are recalled, not re-read (~85 %), so the
  independent checks below carry the weight.
- **ASE's own Tersoff** (independent engine and stress), same +-0.5 %
  central difference: clamped-ion C agrees to 1.8e-8 GPa.  The default
  multi-amplitude fit differs from that single difference by up to
  0.025 GPa (C12), the stress-strain curvature seen at +-1 %.
- **Internal strain** against Phi^+ Lambda from finite differences of the
  forces: < 2e-3 A per unit strain; C^0 - Lambda^T Phi^+ Lambda / V
  against the relaxed-ion C: < 0.1 GPa (tests/test_elastic.py).
- **pymatgen ElasticTensor**: B, G (V, R, H), E_H, nu_H and A^U agree to
  rel 1e-10 on cubic, hexagonal and random positive-definite tensors.
- **Debye temperature**: the isotropic closed form exactly; Al from its
  0 K elastic constants (Kamm and Alers 1964) 430.7 K against the
  tabulated 428 K (UNCERTAIN ~80 %: inputs recalled; 2 % tolerance).
- **ZnO** (Binks-Grimes Buckingham + PPPM, 300 atoms, 25 s): C11 = C22
  to 0.26 GPa, C66 = (C11 - C12)/2 to 0.05 GPa, C13 = C23, C44 = C55,
  no normal-shear coupling; Born stable (lowest eigenvalue 68.7 GPa);
  under the axial strain the Zn and O sublattices move against each
  other along z only (the u-parameter response, 1.0 A per unit strain).
  For information, C11 232, C33 211, C44 75, C12 95, C13 86 GPa (relaxed
  ion); this rigid-ion model is not expected to reproduce experiment's
  C44 (about 43 GPa).

## Side changes

- `GeometryOptimiser` with `verbose=False` no longer prints its
  end-of-run summary (each strained cell printed one, 37 per run).  Files
  are written as before.
- `cli/common._input_hash`: SHA-256 of lattice, element order, wrapped
  fractional coordinates (1e-6) and the potential section, written to
  every results JSON that a later command may reuse (Q30).
- Version 0.2.0a6.
