---
status: accepted
---

# An MLIP in every solver: capabilities, refusals and an exact many-body heat flux

Until 2026-09-26 an MLIP (`potential: {type: MLIP, model: ...}`, ADR-0004
decision 6) was verified only in `relax` and `md`.  Every force-field
command already builds its force function through the same YAML bridge
(`_build_force_fn_from_yaml` -> registry), so `gcmc`, `osmc`, `neb`, `tsa`,
`phonon` and `kappa` accepted an MLIP too -- unverified, and in two cases
silently approximate: `osmc` took the ghost particle's soft-core LJ
parameters from the engine and, finding none, fell back to a Gaussian;
`kappa` handed the whole virial term of the heat flux to a centre-of-mass
approximation that is ~0 in a momentum-conserving run.  The same kappa
approximation applied to Stillinger-Weber, Tersoff and EAM.

## Decisions (agreed with the user, 2026-09-26, Q1-Q7)

1. **Scope.** The six CLI solvers above, plus `ElasticConstantsCalculator`
   (Python API).  `esm` computes electrostatics only; it refuses an MLIP
   (an MLIP trained on total energies would also count the electrostatics
   twice).
2. **One connection, declared capabilities.** No per-solver entry point.
   `MLIPForceField` states what it can do (`per_atom_energy`, `species`,
   `site_terms`), and each solver checks at construction what it needs
   (`check_mlip_for_solver`): a refusal is a start-up error with the
   reason, never a silent fallback.  The CLI prints it as `ERROR: ...`.
3. **Model form per solver.** `relax`, `md`, `neb`, `tsa`, `phonon`: any
   force-field MLIP (tabular or atom-centred).  `gcmc` (the atom count
   changes) and `kappa` (per-atom energies): atom-centred only; elements a
   solver will place must be among the model's.
4. **Exact many-body heat flux.** A site-energy decomposition
   (`akitsushima/engine/site_terms.py`): E = sum_i U_i with U_i a function
   of the vectors v_m from atom i to its neighbours; per atom U_i and per
   neighbour entry dU_i/dv_m.  Forces, stress and the heat flux
   J V = sum_i E_i v_i - sum_m v_m (dU_centre/dv_m . v_nbr) (Fan et al.,
   Phys. Rev. B 92, 094301 (2015), eq. 21) follow from it alone.
   Implemented for pair potentials (energy split equally), Stillinger-Weber
   (single and multi-species, the ADR-0006 convention), Tersoff (single and
   multi-species, analytic), EAM, and the atom-centred MLIP (from the ACSF
   Jacobians).  `kappa` uses it for every term that has one; only a Coulomb
   term keeps the centre-of-mass approximation, with the warning.
5. **osmc** refuses an MLIP until its own muVT failure (ADR-0003 T7,
   O1-O2) is settled.
6. **Verification.** Internal consistency through the CLI (tests), and an
   MLIP trained on the reference potential against that potential through
   the same solver (`example/mlip/adr0007/verify_solvers.py`), tolerances
   sized by the model's force error.
7. **Open CLI items handled here:** T8 Tersoff re-judged with the
   Nelder-Mead default; the fit output JSON (Born-Mayer nesting, r_cut);
   the tsa rate label; the ADR-0003 table (T3a).
8. **Tersoff fit template without DE** (decided 2026-09-26): the
   `--write-config` Tersoff template sets `use_de: false`; SW keeps it.

## Evidence

### Site terms against the engines (tests/test_site_terms_heat_flux.py)

Forces and stress rebuilt from the site terms equal each engine's own
output: SW (single, GeSn Tomita) 1e-10 eV/A, EAM and LJ 1e-15.  Tersoff
differs by up to 2.4e-7 eV/A -- its engine differentiates zeta by finite
differences; against central differences of the engine's energy the
analytic site terms agree to ~1e-8, the engine to ~1e-7.

Heat flux against J V = d/dt sum_i r_i E_i (central differences over one
velocity-Verlet step each way, isolated 32-64-atom clusters, 300 K):
relative error 1.9e-7 (SW), 6.9e-7 (Tersoff, two species), 2.2e-9 (EAM),
and within the test's 1e-5 for the atom-centred MLIP.  The replaced
centre-of-mass approximation gave a flux 24x (SW), 66x (Tersoff) and ~300x
(EAM) smaller than the true one on the same clusters.

### MLIP-driven solvers against SW Si (example/mlip/adr0007)

Model: AtomicELMModel, 400 hidden units, ADR-0004 ACSF; trained on 320
SW-labelled cells (bulk 64-atom rattled and strained; 63-atom vacancy cells
along the hop from ideal sites, and around SW's relaxed endpoints and NEB
images).  Test RMSE: E 0.29 meV/atom, F 0.040 eV/A, S 0.046 GPa.  One
active-learning check (the geometries the MLIP's own relax/neb reach,
labelled by SW) agreed to 0.066 meV/atom, so no further round was needed.

| solver | SW | MLIP | difference | tolerance | verdict |
|---|---|---|---|---|---|
| elastic (ions relaxed) | C11 / C12 / C44 151.42 / 76.43 / 56.40 GPa | 150.43 / 75.90 / 56.44 | -0.99 / -0.52 / +0.05 GPa | 5 GPa | PASS |
| phonon (8-atom cell x 2^3; Gamma incl. X, R, M) | | | max 0.023 THz | 0.2 THz | PASS |
| neb + tsa, vacancy hop (63 atoms) | Ea 0.5172 eV, nu_i -178.09 cm^-1, E_f 2.835 eV | 0.5188 eV, -177.33 cm^-1, 2.831 eV | 1.6 meV, -0.43% | 20 meV, 3% | PASS; the MLIP run passes all 8 internal checks |
| kappa (300 K, 64 atoms; one 100 ps SW trajectory, heat flux from each potential's site terms) | 14.60 W/(m K) | 14.55 W/(m K) | -0.33%; J correlation 0.99983 | 15% | PASS |
| gcmc (LJ Ar teacher, T7 NIST point: T* 0.90, p/p_sat 0.80, 5000 cycles, last 80%) | <N> 42.88 +/- 1.41 | 41.75 +/- 1.74 | -1.13 +/- 2.23 | 3 SE | PASS (NIST 41.90) |

The LJ model: AtomicELMModel, 200 hidden units, on G2 Gaussians every
0.2 A from 2.0 A plus three G2 centred at r = 0; trained on 300 random
configurations (N 10-80, nearest distance >= 0.9 sigma, half with one pair
at 0.65-0.95 sigma); test RMSE 3.6e-5 meV/atom.  A pair closer than the
data -- which a trial insertion meets in ~2% of attempts at 0.4 sigma --
comes out repulsive (0.3 sigma: 5507 eV; LJ 7.8e4) only because of the
r = 0 functions: without them the model returned ~0 eV there, and such
insertions would be accepted.  An MLIP used in gcmc must be trained, or
built, to be repulsive at overlaps; nothing in gcmc can supply that.
(A first run, whose data generator could drop an atom onto a third one,
trained a useless model and let N run away; archived in `gcmc_iter0/`.)

A first attempt without relaxed vacancy geometries in the training data
(`results_iter0.jsonl`) passed elastic and phonon but put the vacancy
formation energy at 1.96 eV and the barrier 272 meV high: its relax ran
into a configuration it had never seen.  The solver was not at fault -- all
of its internal checks passed -- the training set was.

### CLI (tests/test_mlip_solvers.py)

neb, tsa, phonon, kappa and gcmc run end to end on an atom-centred MLIP;
neb on a tabular one; kappa on the MLIP issues no centre-of-mass warning;
gcmc and kappa refuse a tabular model; gcmc refuses an element the model
lacks; osmc and esm refuse an MLIP; the CLI's `main` turns a refusal into
`ERROR: ...` and exit status 1.

### Items from ADR-0003

- T8 Tersoff (FT1): with Nelder-Mead the local fit from +10% PASSES
  (E0 / V0 / B0 within 3e-6 / 3e-6 / 1.6e-4); with the template's DE first
  it fails (DE settles at loss 210 in another basin).  The template now
  leaves DE off (decision 8).
- fit output JSON and the tsa rate label: fixed (ADR-0003 findings).

## Consequences

- `kappa` for SW, Tersoff and EAM changes: their virial contribution was
  ~0 before.  Earlier kappa numbers for those potentials are
  underestimates.
- Every refusal is a start-up error; configurations that used to run with a
  silent approximation (osmc on an MLIP, kappa on a tabular MLIP) now stop.
- `gcmc` on an MLIP recomputes the full energy per trial move (correct,
  not fast); a local-energy fast path from the site terms is possible
  later.
