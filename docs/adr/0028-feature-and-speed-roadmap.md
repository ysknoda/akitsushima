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
7. **Ensemble ELM with an uncertainty output (Q15; ADR-0010 item 7).**
   ADR-0027 rounds 3-5 found that the spurious anneal minima differ from
   one ELM seed to the next, which more data does not remove.
8. **Structure and trajectory I/O, ASE adapter (Q14, Q17).**  Reading
   and writing extxyz, LAMMPS data and CIF; extxyz and LAMMPS dump as
   trajectory output; an ASE Calculator that wraps an akitsushima force
   function.  `ase` is an optional extra.
9. **Structure builders (Q16).**  Supercell, random alloy, vacancy,
   slab.  SQS generation is not part of it (tier C, Q19).
10. **Anisotropic NPT (Q14).**  Full cell or per axis.  The in-plane
    barostat of ADR-0010 item 2 is part of this step (Q24).  It is last
    because its verification (conserved quantity, each pressure
    component) is the heaviest.

### Tier B: after tier A

- A descriptor family whose angular part is linear in the number of
  neighbours (Q10); models have to be retrained.
- EAM kernel in numba; PPPM: no `_mesh_stress` on steps that do not need
  the stress, real FFT, charge assignment in numba (Q12).
- Langevin and CSVR thermostats (Q14).
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
