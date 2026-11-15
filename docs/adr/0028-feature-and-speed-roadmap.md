---
status: accepted (plan, 2026-11-12)
---

# Roadmap: features to add and code to speed up

Agreed with the user on 2026-11-12 (grilling, rounds 1-3, Q1-Q25), at
0.2.0a16.  The question was which features to add and which code to
improve, with the weight on breadth and speed.  This ADR ranks the
candidates, together with the items already deferred in ADR-0010,
ADR-0016 and ADR-0025.  It is a plan: nothing here is implemented, and
the user decides when each item starts.

## Starting point (2026-11-12, 0.2.0a16)

Found by reading the code:

- numba kernels exist for the pair potentials, Stillinger-Weber,
  Tersoff, ACSF, the neighbour list and phonon.  EAM (`engine/eam.py`,
  `use_numba=False`) and the Coulomb methods (`engine/coulomb/`) are
  numpy only.
- `md` is NVE, NVT (Nose-Hoover chain) and NPT with an isotropic
  barostat.  Trajectories are written as XDATCAR only.
- `md` and `relax` cannot restart (ADR-0025); `anneal` can.
- The MLIP has no MPI path and holds the full `dGdv (M, 3, Dc)` array
  at inference (ADR-0016).
- The repository has no benchmark suite.  The full test suite takes 15
  to 31 minutes, and the user's post-edit hook runs it after every edit
  of a `.py` file.

Measured (one force call, warm, one process, **one numba thread**,
Apple M4 while other jobs used about seven cores, so the absolute times
are indicative; the ratios and the profile shares are the result):

| System | N | no neighbour list | with neighbour list | largest share (profile) |
|---|---|---|---|---|
| SW GeSn (Tomita 2018) | 4096 | 1.39 s | 0.069 s | `_compute_multi_species` 88 % |
| Tersoff Si | 4096 | 1.26 s | 0.011 s | `_compute_single_species` 63 % |
| Zhou EAM Ni | 4000 | 1.45 s | 0.238 s | `eam.py:compute` 55 %, `_decay` 20 % |
| Buckingham + PPPM ZnO | 3600 | 3.57 s | 0.81 s | FFT 30 %, `_mesh_stress` 28 % |
| Buckingham + Ewald ZnO | 600 | 2.40 s | not measured | `ewald_sum` 96 % |
| `ac_elm_iso_r2` GeSn | 4096 | 1.24 s | (own cell list) | ACSF kernel 92 % |

Three findings:

1. **A force call without a neighbour list is O(N^2).**  The classical
   engines then build every pair with `np.triu_indices`
   (`pair_potentials.py`, `stillinger_weber.py`, `tersoff.py`, `eam.py`,
   `site_terms.py`, `coulomb/esm.py`).  For SW at 4096 atoms this is 20
   times slower than the call with a list, and at 512 atoms 3 times
   (27 ms against 8.3 ms).  `md` and `relax` pass a list.  `phonon`,
   `elastic`, `tsa` and the Green-Kubo MD loop of `kappa`
   (`thermal_conductivity.py`) call the force function without one.
   This comes from the code and the profile; `kappa` itself was not run.
2. **`NeighbourList.build` is slow.**  With r_cut 4.46 A (list radius
   4.96 A) and 4096 atoms: 2.08 s for the non-orthogonal GeSn cell,
   0.41 s for a cubic cell (both cKDTree), 8.0 s with the `cll`
   back-end.  One build of the GeSn cell costs 30 SW force calls.  The
   numba cell list of ACSF (ADR-0025) takes about 0.025 s at the same
   size (2 % of the MLIP call).  The share of a whole MD run depends on
   the rebuild interval and was not measured.
3. **The MLIP force call is linear, at 0.30 ms per atom**, about 18
   times SW with a neighbour list (0.017 ms per atom).  The ACSF kernel
   is 92 to 94 % of it at 512 and 4096 atoms.

## Decisions

### Frame

1. **Record (Q1).**  This ADR holds the ranked list.  Each item is
   implemented later, when the user says so.
2. **Target (Q2).**  First: covalent semiconductors, that is building
   MLIPs for GeSn and AlN and running large MD with them.  Second: ionic
   crystals and oxides.  Features that widen the package as a general
   tool are also considered.  Surfaces and electrochemistry (ESM, GCMC)
   are not a target of this roadmap.
3. **CPU first (Q3).**  Speed work goes to the numba CPU path before the
   GPU (ADR-0016).
4. **Pain points (Q4).**  The MLIP force call in `md`, `relax` and
   `anneal`, and the time of the test suite.
5. **Numerical compatibility (Q5).**  The default path stays within a
   relative 1e-10 of the present results.  fastmath, float32 and
   approximations are opt-in.
6. **Dependencies (Q6).**  numpy and scipy stay the only required ones.
   Anything new (ase, spglib, ...) is an optional extra.
7. **One table (Q7).**  The deferred items of ADR-0010, ADR-0016 and
   ADR-0025 are ranked together with the new candidates.  Ranking an
   item does not start it.
8. **Adoption gate (Q11).**  A speed-up is adopted when it is at least
   1.5 times faster on a fixed case of `benchmarks/`.  The value is a
   judgement call (UNCERTAIN); it is lower than the 3x of ADR-0016
   because the CPU work is a series of small changes that keep the
   results.

### Tier A: the next milestone, in this order (Q21)

1. **`benchmarks/` and a baseline (Q11).**  The force-call benchmark of
   this ADR (potentials and sizes of the table above, a JSONL ledger)
   and its baseline values.  The baseline is measured again on an idle
   machine, with 1 and 4 numba threads.
2. **One numba cell list for `NeighbourList` (Q9, finding 2).**  The
   cell list of ACSF becomes the back-end of the engine's neighbour
   list.
3. **No O(N^2) path without a neighbour list (Q9, Q20, finding 1).**
   The force function owns a neighbour list: when none is passed it
   builds one and reuses it within the skin.  The all-pairs path stays
   as the reference for small cells without numba.  Step 2 comes first
   so that callers that rebuild on every call (`phonon` displacements)
   do not lose time.
4. **Test suite (Q13).**  Measure once with `--durations`, mark the slow
   tests, and let the post-edit hook run `-m "not slow"`; the full suite
   is run by hand.  The hook is the user's global setting, so the change
   is shown before it is made.
5. **ACSF kernel (Q10, finding 3).**  Same descriptor and same models:
   contraction of force and virial without storing `dGdv`, radial terms
   computed once per pair, reuse of the neighbour list within a skin.
   This is also the CPU form of ADR-0016 decision 6.
6. **Restart of `md` and `relax` (Q14).**  The limit recorded in
   ADR-0025: a run longer than 12 h is not possible today.
7. **Langevin thermostat in `md` (Q14; moved from tier B by the user on
   2026-11-12).**  NVT for the whole system; `nemd` already integrates
   Langevin baths (ADR-0020), but only as its two heat-bath regions.  It
   follows step 6 so that the random-number state is designed into the
   checkpoint from the start.
8. **Ensemble ELM with an uncertainty output (Q15; ADR-0010 item 7).**
   ADR-0027 rounds 3-5 found that the spurious anneal minima differ from
   one ELM seed to the next, which more data does not remove.
9. **Structure and trajectory I/O, ASE adapter (Q14, Q17).**  Reading
   and writing extxyz, LAMMPS data and CIF; extxyz and LAMMPS dump as
   trajectory output; an ASE Calculator that wraps an akitsushima force
   function.  `ase` is an optional extra.
10. **Structure builders (Q16).**  Supercell, random alloy, vacancy,
   slab.  SQS generation is not part of it (tier C, Q19).
11. **Anisotropic NPT (Q14).**  Full cell or per axis.  The in-plane
    barostat of ADR-0010 item 2 is part of this step (Q24).  It is last
    because its verification (conserved quantity, each pressure
    component) is the heaviest.

### Tier B: after tier A

- A descriptor family whose angular part is linear in the number of
  neighbours (Q10); models have to be retrained.
- EAM kernel in numba; PPPM: no `_mesh_stress` on steps that do not need
  the stress, real FFT, charge assignment in numba (Q12).
- CSVR thermostat (Q14).
- Active-learning command driven by the ensemble uncertainty (Q15).
- Lattice thermal conductivity from third-order force constants in the
  relaxation-time approximation; point-defect formation energy and
  migration barrier as one workflow (Q16).
- phonopy force constants and `FORCE_SETS` output; spglib symmetry to
  reduce the phonon displacements and elastic strains; reading LAMMPS
  potential files (`.sw`, `.tersoff`) (Q17).
- GPU, ADR-0016 (Q23): stage 1 starts after step 5 and once access to
  the H100 is known.
- `pytest-xdist` (Q13): decided after the durations of step 4; it needs
  a new dev dependency.

### Tier C: not now

- SQS generation (Q19).  Considered and set aside by the user.  For the
  record: in a 64-atom Ge0.875Sn0.125 cell a random draw has an expected
  1.78 Sn-Sn first-neighbour pairs out of 128, with a spread of about
  75 % in a Poisson estimate, against about 9 % at 4096 atoms; so it
  would matter for small DFT cells, not for large MD cells.  Cao et al.
  (arXiv:2010.02991) report short-range order in GeSn (fewer Sn-Sn first
  neighbours than random), so a later version should take a target
  correlation rather than only zero.
- An MPI path for the MLIP (Q10); Ewald in numba (Q12).
- OS-ELM, GAP, external descriptors at inference, export of potentials
  for LAMMPS (Q15; ADR-0010 items 8, 6, 9).
- Diffusion coefficient and ionic conductivity; thermodynamic
  integration (Q16).
- New potential families: MEAM, core-shell, D3 dispersion, variable
  charge (Q17); chosen when a system needs one.
- DFT-free calculation of any system (added 2026-11-15, ADR-0029
  follow-up): an existing universal MLIP as an external engine, or a
  package-native multi-element atom-centred model.  Notes in ADR-0029.
- ADR-0010 items 1 (MLIP + ESM), 3 (multi-species CFCMC), 4
  (constant-potential ESM), 5 (atom-centred linear and quadratic
  models), 10 (further linear methods), 11 (second-derivative data)
  (Q24).

### Not goals (Q18)

Domain-decomposition MPI for one structure; message-passing MLIPs
(NequIP / MACE class; reachable from outside through the ASE adapter);
ReaxFF; a GUI.

### Deferred measurements of ADR-0025 (Q22)

The ladder and the parallel runs are done once, after step 5, as the
"after" measurement; the table above and ADR-0025 are the "before"
record.  The durability runs follow step 6.  Each still waits for the
user's go.

### Recording and versions (Q25)

Steps 1 to 4 are recorded as addenda to this ADR.  From step 5 on, each
item gets its own ADR and a short grilling before it starts.  Each
finished item raises the alpha version by one; 0.2.0b1 follows when all
of tier A is done.  Commits are local; every push and tag is asked for.

## Addendum 1: benchmark and baseline (tier A step 1, 2026-11-12)

`benchmarks/bench_force.py` times one warm force call per potential
family at about 500 and 4000 atoms and appends one JSON line per case to
a ledger: the time of one `NeighbourList.build`, of the call with that
list, and of the call without a list, each the best of three.
`--compare` prints old / new against another ledger for the same case,
size, thread count and tag.  `example/` is not tracked, so the script
builds everything itself from parameters in the package or cited in its
docstring: SW Si (Stillinger and Weber 1985), Tersoff Si (1988), Zhou
EAM Ni (2004), Buckingham MgO (Lewis and Catlow 1985) with PPPM and
Ewald, and an atom-centred ELM (400 hidden units) trained on SW silicon
with the ACSF grid of `ac_elm_iso_r2` (27 features per atom for one
species).  `--mlip-model` adds a model file.  It also prints the energy
per atom, the summed force, and the difference between the call with a
list and the all-pairs call.  `tests/test_benchmarks.py` runs it on
small cells (marked slow).

`benchmarks/baseline.jsonl`, tag `loaded`: one numba thread, 0.2.0a16
at 392b2aa, while other jobs held about seven of the ten cores (load
average 14).  It is **provisional**: the baseline on an idle machine,
with 1 and 4 threads, is still to be measured and will be appended with
an empty tag.

| Case | N | build [s] | call with list [s] | call without list [s] |
|---|---|---|---|---|
| SW Si | 512 | 0.032 | 0.0033 | 0.019 |
| SW Si | 4096 | 0.250 | 0.028 | 1.58 |
| Tersoff Si | 512 | 0.0093 | 0.0018 | 0.025 |
| Tersoff Si | 4096 | 0.072 | 0.015 | 1.64 |
| Zhou EAM Ni | 500 | 0.182 | 0.043 | 0.059 |
| Zhou EAM Ni | 4000 | 1.44 | 0.325 | 1.95 |
| Buckingham + PPPM MgO | 512 | 0.264 | 0.087 | 0.144 |
| Buckingham + PPPM MgO | 4096 | 2.07 | 0.684 | 3.33 |
| Buckingham + Ewald MgO | 512 | 0.206 | 0.724 | 0.698 |
| ACSF(27) + ELM(400) Si | 512 | - | - | 0.141 |
| ACSF(27) + ELM(400) Si | 4096 | - | - | 1.03 |
| `ac_elm_iso_r2` (Ge, Sn at random on the Si geometry) | 512 | - | - | 0.135 |
| `ac_elm_iso_r2` | 4096 | - | - | 1.23 |

The three findings above hold on these self-contained cases too: the
all-pairs call is 57 times (SW) and 112 times (Tersoff) the call with a
list at 4096 atoms, and one neighbour-list build costs 3 to 9 force
calls.  The call with a list and the all-pairs call agree to 4.9e-11
eV/A (SW) and below 1e-13 eV/A (the others).

Full suite with the new test: 802 passed (22.7 min, 2 numba threads,
loaded machine).  The version is not raised until the idle baseline is in.
