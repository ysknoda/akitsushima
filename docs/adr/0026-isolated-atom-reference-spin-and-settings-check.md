---
status: accepted (plan, 2026-10-30)
---

# Isolated-atom reference: same settings as the training data, and a check

Agreed with the user on 2026-10-30 (grilling, rounds 1-4, Q1-Q15).

## Finding

The GeSn and AlN tutorials subtract isolated-atom energies that were
computed spin-polarised, while every training structure is
non-spin-polarised:

| | isolated atom (`example/*/vasp_ref`) | value used (eV) | training data |
|---|---|---|---|
| GeSn | ISPIN = 2, NUPDOWN = 2 (Ge, Sn) | Ge -0.76635249 (`Ge_atom_int`, integer occupation, ISMEAR = -2); Sn -0.65677602 (`Sn_atom`, fractional, ISMEAR = 0, SIGMA = 0.01, energy without entropy) | ISPIN = 1 (all 50 V01 OUTCARs) |
| AlN | ISPIN = 2, NUPDOWN = 1 (Al), 3 (N) | Al -0.28070402, N -3.18704194 (both `_int`) | ISPIN = 1 (192 V01 + 802 round OUTCARs) |

ENCUT (500 eV), GGA = PS, LASPH, PREC = Normal and the POTCARs
(PAW_PBE Ge 05Jan2001, Sn 08Apr2002, Al 04Jan2001, N 08Apr2002) agree;
only ISPIN (and, for the atom, the smearing) differs.  GeSn also mixes
an integer-occupation Ge with a fractional-occupation Sn (Sn_atom_int is
-0.65142214, 5.4 meV apart).  The same Ge/Sn values are in
`example/GeSn/mlip/dataset_round*.yaml` and `dataset_short_range.yaml`,
so the ADR-0010 A1 round-2 model uses them too.

## Why it matters

`dataset` subtracts sum_s n_s E_atom,s, and the MLIP is then fitted with
`isolated_atom_energy: 0`, i.e. eps_s(G = 0) = 0 exactly (ADR-0014 B).
Without that constraint a per-species constant would be absorbed by the
model.  With it, the constant has to be bridged by the shape of eps_s(G)
between the last training data and G = 0.  The ISPIN = 1 data describe
the non-spin-polarised potential-energy surface, whose dissociation limit
is the non-spin-polarised atom.  A spin-polarised reference asks the
model for a step Delta E that is not in the data, in the region without
data (surfaces, vacancies, atoms leaving in an anneal).  SW and Tersoff
fits also vanish at dissociation and see the same step.  The physics of
an atom spin-polarising as it leaves cannot be learned from ISPIN = 1
data either way.

## Decisions

1. **Reference (Q1, Q3).**  Training and classical fits use the
   isolated atom computed like the training data: ISPIN = 1, ISMEAR = 0,
   SIGMA = 0.01, fractional (spherical) occupation, energy without
   entropy (`U-sph`).  The spin-polarised integer-occupation atom
   (`P-int`) is kept for reporting only:
   Delta E_s = E(U-sph) - E(P-int), the "isolated-atom correction".  It
   contains both the spin polarisation and the loss of spherical
   symmetry.  Cohesive energies are reported against both references.
2. **No new package key (Q7).**  Only the tutorials report cohesive
   energies, so the correction lives there.
3. **Settings check (Q2, Q6).**
   - `isolated_atom_energy` (dataset) takes, per species, a number (as
     now) or a path to a VASP run (a directory with OUTCAR, or the
     OUTCAR itself).  For a path the energy is read with the dataset's
     own `energy` key from the last ionic step.
   - For a path, ISPIN, ENCUT, GGA / METAGGA, LASPH, PREC, ISMEAR,
     SIGMA and that species' POTCAR TITEL are compared with every
     training run's OUTCAR.  A mismatch is a warning with a
     per-setting table (not an error: a deliberate ISPIN = 2 reference
     must stay possible).  For a number, one line notes that the
     settings cannot be checked.
   - The library function returns the comparison in its summary and
     raises a `UserWarning`; the CLI prints the table.
4. **Version (Q15).**  0.2.0a16.

## Calculations (Q4, Q9, Q11 (ii))

Run locally by Claude with VASP 6.6.1, **one core** (`mpirun -np 1`;
Python with OMP / NUMBA / OPENBLAS / MKL / VECLIB threads = 1), one job
at a time (another session uses 6 cores).

- Isolated atoms, same box (15 x 15.5 x 16 A), ENCUT and POTCAR as now:

  | label | ISPIN | occupation | symmetry | status |
  |---|---|---|---|---|
  | U-sph | 1 | fractional, p shared evenly | spherical | new; **training reference** |
  | U-axial | 1 | p filled orbital by orbital (FERWE) | non-spherical | new |
  | P-sph | 2 | fractional per spin | spherical | existing `*_atom` |
  | P-int | 2 | integer (Hund) | non-spherical (N p3-up spherical) | existing `*_atom_int`; **reporting reference** |

  With ISPIN = 1 a band holds two electrons, so U-axial of N (p3) and
  Al (p1) keeps a 0.5 occupation.  N P-sph and P-int are the same
  configuration and must agree (a check).
- Uniform expansion, ISPIN = 1: alpha-Ge and wurtzite AlN at about six
  scale factors in 1.0 - 1.7 a0.

## Tutorials (Q5, Q8, Q10 - Q12)

Both rebuilt from `run_all.sh` with the U-sph reference; the old models
are kept for the comparison first.

- Dataset chapter: three or four lines on why the ISPIN = 1 atom is the
  reference, pointing to a new appendix.
- Appendix "isolated-atom calculations": per element the four variants
  (energy without entropy and sigma -> 0, difference to U-sph, total
  moment, s / p1 / p2 / p3 occupations per spin, symmetry); a level
  diagram per element splitting Delta E into spin polarisation and
  symmetry breaking.
- Impact: cohesive energies of alpha-Ge, alpha-Sn and wurtzite AlN
  against U-sph and P-int, with experiment and the published SW /
  Tersoff; old (P-int) and new (U-sph) ELM energy along the uniform
  expansion against the ISPIN = 1 VASP points.
- Troubleshooting: the ISPIN = 2 atom and the mixed Ge / Sn occupations,
  with the new warning as the example output.

## A1 re-check (Q5, Q13, Q14)

After the tutorials, in the same session and on one core: retrain the
GeSn round-2 model with the U-sph reference and re-judge it (relax
10/10, anneal 10/10, DFT check 3/3).  On a failure, stop and report.
ADR-0025's large-scale runs (on hold) will then use the new model.

## Commits (Q15)

Local, in stages: ADR-0026; check and tests; VASP results; tutorials;
A1 re-check.  Push and tag are asked for at the end.
