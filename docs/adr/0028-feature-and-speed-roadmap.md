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
   *Done as ADR-0030 (0.2.0a20): only the arithmetic of the angular
   loop changed, for 2.1-2.8 times per force call.  The fused
   contraction and the skin reuse were taken out after measurement
   (ADR-0030 decision 2).*
6. **Restart of `md` and `relax` (Q14).**  The limit recorded in
   ADR-0025: a run longer than 12 h is not possible today.
   *Done as ADR-0031 (0.2.0a21): md continues bit for bit, relax from
   the checkpointed structure with the optimiser restarted.*
7. **Langevin thermostat in `md` (Q14; moved from tier B by the user on
   2026-11-12).**  NVT for the whole system; `nemd` already integrates
   Langevin baths (ADR-0020), but only as its two heat-bath regions.  It
   follows step 6 so that the random-number state is designed into the
   checkpoint from the start.
   *Done as ADR-0032 (0.2.0a22): BAOAB, `thermostat: langevin` under
   `ensemble: nvt`; restarts bit for bit with its random-number state.*
8. **Ensemble ELM with an uncertainty output (Q15; ADR-0010 item 7).**
   ADR-0027 rounds 3-5 found that the spurious anneal minima differ from
   one ELM seed to the next, which more data does not remove.
   *Done as ADR-0033 (0.2.0a23): ensemble mean as force field, spread
   as uncertainty; GeSn round-5 evaluation in its Result.*
9. **Structure and trajectory I/O, ASE adapter (Q14, Q17).**  Reading
   and writing extxyz, LAMMPS data and CIF; extxyz and LAMMPS dump as
   trajectory output; an ASE Calculator that wraps an akitsushima force
   function.  `ase` is an optional extra.
   *Done as ADR-0034 (0.2.0a24): extxyz / LAMMPS data / CIF input,
   `traj_format` for md, relax and gcmc, `reader: extxyz` for training
   sets, `AkitsushimaCalculator`.*
10. **Structure builders (Q16).**  Supercell, random alloy, vacancy,
   slab.  SQS generation is not part of it (tier C, Q19).
   *Done as ADR-0035 (0.2.0a25): `akitsushima build` / `ak.build` with
   bulk, supercell, substitute, vacancy, slab, rattle, strain, scale.*
11. **Anisotropic NPT (Q14).**  Full cell or per axis.  The in-plane
    barostat of ADR-0010 item 2 is part of this step (Q24).  It is last
    because its verification (conserved quantity, each pressure
    component) is the heaviest.

### Tier B: after tier A

- A descriptor family whose angular part is linear in the number of
  neighbours (Q10); models have to be retrained.
- EAM kernel in numba; PPPM: no `_mesh_stress` on steps that do not need
  the stress, real FFT, charge assignment in numba (Q12).
- ESM (`ESMCalculator`, also behind `ESMCoulomb`): no `_compute_stress`
  on calls that do not need the stress (69 % of a bc2 call at 96x96x384),
  and the Green-function matrices cached while the cell and grid are
  unchanged (rebuilt on every call today).  Found in step 4; added to
  tier B by the user on 2026-11-18 (addendum 4).
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
- `pytest-xdist` (Q13): not added after the durations of step 4
  (addendum 4); reconsidered if the full suite exceeds 20 min again.

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

## Addendum 2: compiled cell list as the default neighbour list (tier A step 2, 2026-11-17)

**Change from the plan.**  Decision step 2 said the ACSF cell list would
become the back-end of `NeighbourList`.  Measured first (one thread,
4096 atoms): the cell-linked list already in `neighborlist.py`, compiled
with numba, builds as fast as the ACSF one (0.022 s against 0.026 s at
r_list 4.96 A on the sheared GeSn cell; 0.14 s against 0.18 s at 32768
atoms) and finds the same pairs.  It was never used because `use_numba`
defaulted to False and `'auto'` picked the KD-tree.  So no code is
shared with ACSF; the ACSF list stays where it is, and the engine does
not import from `fitting/`.

What changed (`engine/neighborlist.py`, 0.2.0a17):

- `use_numba=None` (default) means on whenever numba is importable and
  not switched off with `NUMBA_ENABLE=0`, the rule of the potentials'
  kernels.  `backend='auto'` then runs the compiled cell list; with the
  JIT off it runs the KD-tree as before.  `active_backend` reports
  which one a build runs; an unknown backend name is refused.
- The compiled cell-list kernel is built without fastmath, so it gives
  the NumPy version's pairs, order and displacements bit for bit
  (new test on three skewed cells).
- The KD-tree's per-pair Python loop is vectorised; it was most of the
  KD-tree time (2.3 s of 2.3 s for the sheared 4096-atom cell).

Checks:

- Energy, forces and stress with a neighbour list, before against after,
  for SW, Tersoff, Zhou EAM and Buckingham + PPPM on a cubic and a
  sheared triclinic cell (512-864 atoms): the same number of candidate
  pairs everywhere; largest relative difference 4.5e-11 (SW forces, the
  summation order changes with the pair order; the all-pairs and the
  listed SW call already differ by 4.9e-11), all others 1.5e-11 or below,
  inside the 1e-10 of decision 5.
- New tests in `tests/test_neighborlist.py`: back-end selection;
  default list against the brute-force pairs on the hexagonal and
  triclinic cases; compiled against NumPy cell list bitwise; KD-tree and
  cell list give the same candidate set with a skin.

Benchmark, one thread, before and after measured back to back on the
same loaded machine (load average 5.6-7.8), ratio old / new:

| Case | N | build | call with list | call without list |
|---|---|---|---|---|
| SW Si | 512 | 22.0 | 1.35 | 1.17 |
| SW Si | 4096 | 13.8 | 1.22 | 0.97 |
| Tersoff Si | 512 | 7.2 | 0.81 | 0.86 |
| Tersoff Si | 4096 | 6.3 | 1.26 | 1.07 |
| Zhou EAM Ni | 500 | 25.9 | 0.87 | 1.09 |
| Zhou EAM Ni | 4000 | 21.6 | 1.00 | 0.78 |
| Buckingham + PPPM MgO | 512 | 42.4 | 1.30 | 1.14 |
| Buckingham + PPPM MgO | 4096 | 27.9 | 1.12 | 1.03 |

The build passes the 1.5x gate on every case; the force calls do not
change (0.8-1.35 is the noise of the loaded machine).  After the change
one build at 4096 atoms costs about one force call (SW 0.016 s against
0.019 s) instead of 3 to 9.  The non-orthogonal GeSn cell of finding 2
goes from 2.3 s to 0.022 s (about 100x).  Finding 1 (the all-pairs call)
is unchanged; it is step 3.

Full suite: 809 passed (27.1 min, 2 numba threads, loaded machine),
including the 7 new tests.

## Addendum 3: no all-pairs enumeration without a neighbour list (tier A step 3, 2026-11-18)

A force function called without a neighbour list now takes one from
`owned_neighbour_list(lattice, positions, r_cut)` in
`engine/neighborlist.py` (0.2.0a18) and enumerates its candidate pairs
instead of all N(N-1)/2.  The call sites are the former all-pairs
branches: Stillinger-Weber and Tersoff (single- and multi-species paths),
EAM, `BasePairPotential` (LJ, Buckingham, Morse and the other pair
forms), `real_space_coulomb` (the real-space part of Ewald and PPPM,
which built (N, N, 3) arrays) and the site terms of the Green-Kubo heat
flux (`site_terms._pairs`).

**Changes from the plan.**  Decision step 3 said "the force function
owns a neighbour list".

- The lists live in a per-thread cache in `neighborlist.py`, keyed by
  (r_cut, N), at most 4 per thread, not on each potential object.  Three
  reasons:
  - `compute_all` (the Buckingham force function) builds a new engine on
    every call, so a list on the object would never be reused.
  - `phonon` and `elastic` call one potential object from a
    `ThreadPoolExecutor`, so a shared list could be rebuilt for one
    displaced cell while another thread reads it.
  - Potential objects are pickled for the MPI and process paths.
- Reuse is decided by `NeighbourList.check_and_rebuild` (cell
  deformation and minimum-imaged non-affine displacement).  That
  criterion guarantees a complete list whatever was evaluated before.
  Alternating unrelated structures of the same N therefore only costs
  rebuilds, never wrong pairs.
- After every build the candidates are put in canonical order (`i < j`,
  sorted by `(i, j)`, the order of `np.triu_indices`).  Masked to
  r_cut they are then the all-pairs path's pairs in the same order.
  This was not in the plan.  The first version kept the cell list's
  order, and the full suite caught the effect:
  `test_nemd.py::test_restart_continues_bit_for_bit` failed.  The pair
  order, and so the summation order, depended on when the list was last
  built.  A restarted run rebuilds at a different step, so it left the
  uninterrupted one at the rounding level, and the difference then grew
  chaotically.  With the canonical order a call returns the same bits
  however and whenever its list was built.
- The owned list's skin is 0.3 A, not the solvers' 0.5 A.  This lets
  cells down to 2 (r_cut + 0.3) across use it.  Phonon steps
  (0.01-0.03 A) and elastic strains (about 1 %) stay well inside it.
- The all-pairs path stays in two cases.
  - Below 100 atoms, whether or not numba is in use.  The plan tied
    this to "without numba", but the measured crossover does not depend
    on it (see below).
  - In cells too thin for r_cut + 0.3 (half the smallest perpendicular
    width).  There the old minimum-image error message and
    `allow_minimum_image_violation` behave exactly as before.
- The ESM real-space complement is not changed.  It applies the
  minimum image in x and y only, and `NeighbourList` is periodic in
  three directions.

Crossover, one numba thread (all-pairs / owned list with the build /
owned list reused):

| Potential | N | all-pairs (ms) | owned + build (ms) | owned, reused (ms) |
|---|---|---|---|---|
| SW Si | 64 | 0.88 | 0.94 | 1.04 |
| SW Si | 216 | 5.7 | 1.9 | 2.0 |
| SW Si | 1000 | 83 | 10.3 | 6.3 |
| Tersoff Si | 64 | 1.36 | 0.85 | 0.56 |
| Tersoff Si | 216 | 3.8 | 2.4 | 0.94 |
| LJ Al | 256 | 13.5 | 11.4 | 8.3 |
| EAM Al | 500 | 46 | 29 | 18 |

With `NUMBA_ENABLE=0`, the SW and Tersoff NumPy kernels dominate, so
the list neither gains nor loses (0.9-1.2x at 64-1000 atoms).  LJ and
EAM gain 1.6-2x at 500 atoms.  So the threshold is a plain 100 atoms.

Checks:

- Energy, forces, stress and site terms from calls without a list,
  before against after, were compared on a cubic and a sheared cell
  (216-512 atoms), each called twice so the second call reuses the
  list.  Potentials: SW (Si and multi-species GeSn), Tersoff (Si,
  nested, two-species X-Y), EAM, LJ, and Buckingham MgO with PPPM and
  with Ewald.  SW, EAM, LJ and the site terms are bit-identical.
  Tersoff differs by at most 6.5e-17 relative and Buckingham by
  8.2e-15.  The Buckingham difference is in the real-space Coulomb sum,
  whose list branch computes distances with `np.linalg.norm` instead of
  the dense branch's `sqrt(einsum)`.  Both are inside the 1e-10 of
  decision 5.  (Before canonical ordering the largest difference was
  3.4e-11, in the SW forces; that showed the owned list was in use in
  every case.)
- New `tests/test_owned_neighbour_list.py` (26 tests) checks:
  - every potential, the site terms and the real-space Coulomb sum
    against the all-pairs path on cubic and sheared cells;
  - a phonon/strain sequence that must reuse the list (displacement
    +-0.02 A, 1 % strain) or rebuild it (-10 % strain, 0.3 A noise),
    exact at each step;
  - small and thin cells keep the all-pairs path, its error and its
    escape hatch;
  - concurrent calls from a 4-thread pool;
  - the cache bound and its key;
  - bit-identical results through a reused list, a fresh list and the
    all-pairs path.

Benchmark, one thread, before and after measured back to back on the
same loaded machine (load average 7.6-9.7), call without a list, ratio
old / new; "first" is the first call, which includes the build:

| Case | N | call | first call |
|---|---|---|---|
| SW Si | 512 | 5.9 | 4.2 |
| SW Si | 4096 | 70 | 56 |
| Tersoff Si | 512 | 5.9 | 4.2 |
| Tersoff Si | 4096 | 85 | 57 |
| Zhou EAM Ni | 500 | 2.2 | 1.7 |
| Zhou EAM Ni | 4000 | 5.9 | 6.2 |
| Buckingham + PPPM MgO | 512 | 1.7 | 1.0 |
| Buckingham + PPPM MgO | 4096 | 8.6 | 7.3 |
| Buckingham + Ewald MgO | 512 | 1.3 | 1.3 |

A call without a list now costs about the same as one with a list (SW
4096: 0.025 s against 0.024 s), which closes finding 1.  Two cases are
below the 1.5x gate.
- Ewald at 512 atoms: the reciprocal sum dominates and is untouched
  (tier C, "Ewald JIT").
- The first PPPM call at 512 atoms: the mesh dominates.
Every all-pairs-dominated case passes the gate.  `bench_force.py`
now also records `call_first_s`.

Full suite: 835 passed (31.5 min, loaded machine), including the 26 new
tests.  The first run, before canonical ordering, had the one NEMD
restart failure described above and 830 passed.

## Addendum 4: slow marks and the post-edit hook (tier A step 4, 2026-11-18)

Tests that take 0.5 s or more are marked `@pytest.mark.slow` (0.2.0a19).
The post-edit hook (the user's global
`~/.claude/hooks/post_edit_python_check.sh`) now runs
`pytest tests -q --tb=short -m "not slow"`.  The full suite is still run
by hand before each commit.  The user approved the hook change on
2026-11-18.  It is one line, and in a project without a `slow` marker
`-m "not slow"` still selects every test.

**What the measurement found.**  The hook had never reported a result
for akitsushima, for two reasons.
- Its timeout is 60 s, and the full suite takes 26.6 min with the JIT
  and 2 h 45 min without it.
- The hook sets `NUMBA_DISABLE_JIT=1`.  Under that setting the suite
  aborted on its seventh test (`Fatal Python error: Aborted`, libomp
  initialised twice).  The cause was in `utils/jit.py`.
  - The warm-up kernel that starts numba's threading layer on import
    (0.2.0a11) runs as plain Python when the JIT is disabled, so it
    started nothing.
  - `numba.get_thread_id()` inside the pair, SW and Tersoff kernels
    still launches the layer.  It did so after collection had imported
    torch, and that is the order that aborts.
  - `_start_threading_layer` now also calls `numba.get_num_threads()`,
    which launches the layer in both modes.
  - `tests/test_openmp_order.py` now runs both import orders with the
    JIT on and off.  Before the fix, only the case "akitsushima first,
    JIT off" failed; after it, all four pass.

The run without the JIT then had one failure:
`test_atomic_option_rows_equal_engine_and_engine_is_conservative[AtomicELMModel-kw0]`.
Its fourth-order force check misses by 1.2e-7 against a 1e-7 tolerance.
- This is rounding noise, not a wrong derivative.  The force error takes
  either sign and grows as the step shrinks: with the JIT it is 3e-9 at
  h = 1e-4 and -1.2e-7 at 7.5e-6, and the same holds without it.
- The PCA-whitened energy carries about 1e-12 eV of noise, and
  h = 3e-5 is too small for that.
- The fit itself differs between the two modes (different rounding in
  an ill-conditioned solve), so the noise falls differently.
- `_fd4_check` now takes a separate strain step.  The ELM case uses a
  displacement step of 1e-4 (error 3e-9 with the JIT, 6e-9 without)
  and keeps 3e-5 for strain: at 1e-4 the stress error is 6e-8, above
  `tol_s` = 1e-8.

**Rule for `slow`.**  A test is marked if either holds:
- it took 0.5 s or more (setup + call + teardown) with or without the
  JIT;
- it shares a costly module fixture with slow tests.
A parametrised test is marked as a whole if any case is slow.

The fixture rule was found on the first try of the fast set.  It took
172 s although its tests had summed to 19 s, because a module fixture's
cost moves to whichever test uses it first.
- The MPI fixtures cost 119 s, 19 s and 12 s.  Every test in
  `test_mpi.py` and `test_mpi_cli.py` is now slow.
- Three single fast users of 1-5 s fixtures are slow: the `si` fixture
  in `test_qha.py`, `aln` in `test_phonon_regressions.py` and `mlip_fn`
  in `test_mlip_thin_cells.py`.
- Two fixtures keep their fast users, because the cost is spread over
  several refusal and format tests: `models` in `test_mlip_solvers.py`
  (1.7 s, 4 tests) and `si_table` in `test_elm_options.py` (1.6 s, 8
  tests).

Candidate thresholds were compared on the measured times (both modes,
whole functions marked):

| Threshold | Slow functions | Fast tests | Fast set without JIT | with JIT |
|---|---|---|---|---|
| 0.3 s | 245 / 554 | 476 | 10.4 s | 7.6 s |
| 0.5 s | 223 / 554 | 500 | 19.1 s | 11.1 s |
| 0.75 s | 198 / 554 | 538 | 37.1 s | 20.5 s |
| 1.0 s | 176 / 554 | 570 | 57.2 s | 28.6 s |

With 0.5 s, the fast set is about half of the hook's budget once the
~9 s of collection is added.  Collection imports every test module,
torch included.  The 0.75 s set would leave little margin on a loaded
machine.

**Result**, with the machine loaded (load average 8-9):

| Run | Tests | Time |
|---|---|---|
| `NUMBA_DISABLE_JIT=1 pytest -m "not slow"` | 480 passed, 1 skipped, 356 deselected | 25.5 s (30 s wall) |
| the hook end to end, ruff included | same | 34 s wall |
| `pytest -m slow` | 356 | the rest of the full suite |

The fast set leaves out most finite-difference force and stress checks
of the classical potentials, all MPI tests and the end-to-end CLI runs.
It is a trip-wire, not a substitute for the full suite.

Full suite after the change: 837 passed (27.7 min, loaded machine),
including the two new JIT-off cases of `test_openmp_order.py`.

**Where the full suite spent its time** (with the JIT, 1595 s):
- The three ESM finite-difference tests took 670 s, 42 % of the suite
  (`test_esm_force_matches_finite_difference[bc1-3]`).
- Next came the MPI GCMC run (89 s), `test_msd_vacf_smoke_and_equipartition`
  (37 s) and the multi-species SW fit (36 s).
Without the JIT, the multi-species SW fit alone takes 3327 s.

**ESM finite-difference test (follow-up grilling, 2026-11-18).**  The
test ran 13 ESM evaluations per boundary condition on a 96x96x384 grid
(11 s per call for bc1, 22 s for bc2) with a tolerance of atol 3e-2.
Its comment said the forces were interpolated trilinearly, so a fine
grid and a loose tolerance were needed.  That is no longer so: the
analytic forces are the exact derivative of the discretised energy.

| Grid | bc1 | bc2 | bc3 | max \|F - F_fd\| |
|---|---|---|---|---|
| 32x32x128 | 1.9 s | 3.4 s | 2.1 s | 5.3-5.4e-10 |
| 48x48x192 | 6.4 s | 15.0 s | 11.4 s | 5.4-5.5e-10 |
| 64x64x256 | 27.6 s | 52.0 s | 37.1 s | 5.6-5.7e-10 |
| 96x96x384 | 163 s | 297 s | 198 s | 5.8e-10 (bc1) |

The residual is the central difference's own error on every grid.  The
test now uses 32x32x128 and atol 1e-6 (rtol 0), and the three cases take
6.9 s together, with or without the JIT.  A force error planted in
`_compute_forces` shows the gain: a relative error of 1e-6 now fails
all three cases, while the old tolerance let even 1e-3 pass.
Convergence to the continuum limit is covered by other tests:
- `test_esm_gpar0_poisson_solve_converges_quadratically`
- `test_esm_electrostatics.py` (image-Ewald references)
- `test_esm_matches_ewald_for_a_symmetric_slab_in_a_long_cell`

Full suite after the change: 837 passed in 872 s (14.5 min, loaded
machine), against 1595-1661 s before.  The slowest entries are now the
MPI GCMC fixture (100 s), then the multi-species SW fit and the MSD/VACF
test (32 s each).

The test stays `slow` (bc2 takes 3.3 s, above the 0.5 s rule).  The
profile also showed two costs in ESM itself, which the user put in
tier B rather than fixing now:
- every call computes the stress, whether it is needed or not;
- every call rebuilds the Green-function matrices.

**`pytest-xdist` (Q13).**  Not added.  After the ESM change no single
test bounds a parallel run.  But xdist is a new dependency, and its
workers would compete with numba threads and MPI subprocesses on a
machine that is already loaded.  It will be reconsidered if the full
suite exceeds 20 min again.

**Candidates for the same review**, left for later.  Each needs its own
check that its size buys accuracy, unlike ESM, where the residual did
not depend on the grid:
- the PPPM finite-difference tests (8.4 s);
- `test_esm_gpar0_poisson_solve_converges_quadratically` (14 s);
- the multi-species SW fit (36 s, 3327 s without the JIT);
- `test_gcmc_mu_is_the_full_chemical_potential` (10 s, 847 s without
  the JIT).

**Re-measuring.**
```
pytest tests -q -p no:cacheprovider --durations=0 --junitxml=jit.xml
NUMBA_DISABLE_JIT=1 pytest tests -q -p no:cacheprovider --durations=0 --junitxml=nojit.xml
NUMBA_DISABLE_JIT=1 pytest tests -q -m "not slow" --durations=25
```
The last command shows module-fixture setups that have moved into the
fast set.  A new test is marked `slow` by its author when it takes
0.5 s or more.
