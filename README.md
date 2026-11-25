# akitsushima

> **Beta release.** This project is feature-complete for its intended
> scope but still being validated; bugs may still be found and fixed
> before a stable release. Use in production with caution, and pin the
> exact version you tested against. Computed values are not comparable
> across the 0.2.0 alpha pre-releases listed below. 0.2.0a1 stopped halving Tersoff energies,
> forces and stresses, gave Buckingham and Born-Mayer the force-shifted
> truncation their sibling pair potentials already had, and chose the
> Ewald and PPPM cutoffs from the accuracy you ask for. 0.2.0a2 makes the
> single-species Stillinger-Weber format honour `gamma` (it was ignored;
> the default is now SW's own 1.20), removes a spurious sign flip from
> `VASP_KB_TO_EV_ANG3`, fixes variable-cell RFO, and now refuses
> multi-species SW parameters and `fit` optimiser choices it used to
> accept and get wrong. 0.2.0a3 fixes phonon frequencies (minimum-image
> force constants, the ASR and the unit conversions), the GCMC deletion
> acceptance, and the Green-Kubo heat flux and seeding (ADR-0003); and
> changes every MLIP fit (ADR-0004): ELM / QELM / NN now learn one energy
> whose derivatives give the F and S rows (chain rule; `row_independent:
> true` restores the old per-row form), all models weight the squared
> residual by w_cat x w_unit^2 x w_user (the ELM Uncertainty Weighting is
> gone). ELM / QELM now add a direct linear term by default
> (`direct_link`), and NN defaults to softplus with a direct linear term.
> LightGBM has since been removed (ADR-0012): trees give no conservative
> force, so it could never drive a solver. 0.2.0a4 makes Tersoff forces
> analytic (they change by ~1e-7 eV/A) and fits multi-species Tersoff and
> the Zhou (2004) alloy EAM (ADR-0015). 0.2.0a5 makes an MPI `fit` print
> and write its files on rank 0 only. 0.2.0a6 adds the `elastic` command
> (ADR-0017): clamped- and relaxed-ion elastic tensors from a
> multi-amplitude stress-strain fit, B / G / E / nu as Voigt, Reuss and
> Hill averages, Born stability, sound velocities, Debye temperature and
> the internal strain. 0.2.0a7 adds harmonic thermodynamics to `phonon`
> (ADR-0018): F, U, S, C_v and Theta_D(T) by a Gamma-centred q-mesh mode
> sum (F_vib too), and fixes the sound-velocity Debye temperature, which
> lacked the 1/3 of the Debye mean and was 0.69x too low. 0.2.0a8 adds
> constant-volume cell relaxation (`relax.fix_volume`), the `gruneisen`
> command (mode Grueneisen parameters from +-1 % volume force constants)
> and the `qha` command (thermal expansion, B(T), C_p, thermodynamic
> gamma), and the Vinet equation of state. 0.2.0a9 adds Born charges
> (`born:` block), the Gonze-Lee non-analytic correction (LO-TO splitting)
> in phonon / gruneisen / qha, and the `dielectric` (eps_0, IR modes) and
> `piezo` (relaxed-ion e, d, k) commands (ADR-0019). 0.2.0a10 adds the
> `nemd` command (Langevin heat baths, temperature profile, local heat
> flux, kappa and Kapitza resistance, checkpoint / restart) and the
> species / region decomposition of the Green-Kubo kappa (ADR-0020).
> 0.2.0a11 adds VASP readers (vasprun.xml / OUTCAR, every ionic step) and
> the `dataset` command, which turns VASP runs into a `fit` / `mlip`
> training set split by run (ADR-0021). It also changes `relax` with FIRE,
> the default method. FIRE no longer shrinks its velocity when a step is
> clamped: that had defeated its uphill reset and trapped variable-cell
> runs in limit cycles. Runs whose steps are never clamped are unchanged.
> `write_table` now writes 17 significant digits plus a test-set table,
> and numba no longer aborts when torch is imported first.
> 0.2.0a12 adds the `anneal` command (ADR-0022): MD-quench annealing on a
> temperature schedule, or basin hopping, with every quench a `relax`,
> optional pressure (enthalpy), checkpoint / restart and a list of
> distinct minima. `md`'s NVT / NPT classes take `initial_velocities`.
> The NN models (tabular and atom-centred) take the ELM's input
> standardisation keys (`standardize: zscore | pca`, `pca_eps`,
> `pca_drop`). The NN default stays z-score.
> 0.2.0a13 adds `min_feature_std` to every MLIP model (linear, ELM /
> QELM, NN, tabular and atom-centred): input columns whose training std
> is below it are masked (stored as an infinite std, so they add nothing
> to E, F or S).  The default 1e-12 masks nothing; 1e-4 replaces the
> external column reduction needed for near-constant columns of external
> descriptor tables (ADR-0010, open finding A4).
> 0.2.0a14: a machine-learning potential (and an external code) is no
> longer held to the solvers' neighbour-list minimum-image check, which
> it never used: relax, md, neb, gcmc and elastic refused cells thinner
> than 2 (r_cut + skin) although the ACSF handles them (ADR-0024).  The
> CLI's `--version` prints the version; the Python API is verified
> against the CLI (ADR-0023).
> 0.2.0a15 (ADR-0024): every MLIP model takes `baseline=` and
> `isolated_atom_energy=`; tabular models have `fit_structures()`;
> `akitsushima.io.load_training_set()` / `build_training_set()` and
> `akitsushima.force_fn_from_config()` are public.  API defaults follow
> the CLI's (tabular ELM `eps0` 0.01, relax `output_dir` ./relax_output,
> ...); the CLI's NVT `tau_nhc` falls back to 100 fs like the API.
> 0.2.0a16 (ADR-0026): `dataset.isolated_atom_energy` also takes the
> path of the atom's VASP run (the energy of its last step, read with
> `dataset.energy`).  Which atom to use is the user's choice; for
> ISPIN = 1 training data the lowest spin-polarised atom is recommended
> (ADR-0026 shows why the ISPIN = 1 atom is the worse choice for a
> finite-cutoff MLIP).
> 0.2.0a17 (ADR-0028 step 2): `NeighbourList` builds with the
> numba-compiled cell list by default (`backend='auto'` with
> `use_numba=None`, i.e. on unless `NUMBA_ENABLE=0`), 6-42x faster than
> the former KD-tree default at 500-4000 atoms; the same pairs, so results
> change only by summation order (<= 5e-11 relative).  An unknown
> `backend` is now refused; `active_backend` says which one a build runs.
> 0.2.0a18 (ADR-0028 step 3): a force function called without a
> neighbour list (phonon, elastic, tsa, Green-Kubo kappa, one-off
> evaluations) builds and reuses its own instead of enumerating all
> N(N-1)/2 pairs: 6-85x faster at 500-4096 atoms for SW, Tersoff, EAM and
> the PPPM real-space sum.  The pairs are kept in the all-pairs order, so
> results are bit-identical for the short-range potentials (real-space
> Coulomb <= 1e-14 relative).  Systems under 100 atoms and cells thinner than
> 2 (r_cut + 0.3 A) keep the all-pairs path.
>
> 0.2.0a19 (ADR-0028 step 4): tests that take 0.5 s or more (with or
> without `NUMBA_DISABLE_JIT=1`) are marked `slow`; `pytest -m "not slow"`
> runs the other 481 in about 30 s even without JIT.  With
> `NUMBA_DISABLE_JIT=1`, importing torch after akitsushima no longer aborts
> the process at the first engine kernel.
>
> 0.2.0a20 (ADR-0030): the angular loop of the ACSF kernel computes
> each cutoff factor once per neighbour and one exponential per distinct
> eta, and raises integer zeta by multiplication.  An atom-centred MLIP
> force call is 2.1-2.8 times faster, and so is fitting.  The descriptor
> agrees with the former kernel to 3e-16 relative.
>
> 0.2.0a21 (ADR-0031): `md` and `relax` write a checkpoint and continue
> from it when run again (`restart: false` starts afresh).  md continues
> bit for bit and can be extended by raising `n_steps`.  relax continues
> from the checkpointed structure with the optimiser restarted.
>
> 0.2.0a22 (ADR-0032): `md` with `ensemble: nvt` takes `thermostat:
> langevin` (BAOAB, `damping` in fs, centre of mass kept at rest;
> `ak.LangevinSimulation` in the API).  Its random-number state is part
> of the checkpoint, so it restarts bit for bit too; `thermo.log` shows
> E_tot - Q (Q the heat the thermostat added) as the integrator check.
>
> 0.2.0a23 (ADR-0033): `mlip` with `ensemble: 5` (atom-centred ELM / QELM)
> fits seed-varied members on shared descriptors.  Their mean is the force
> field and their spread the uncertainty: `sigF_max` in md / relax logs,
> sigma in the anneal ledger, and `mlip.evaluate` writes sigma_E,
> sigma_F_max and sigma_S of a list of structures to CSV.
>
> 0.2.0a24 (ADR-0034): structures are read from POSCAR, extxyz, LAMMPS
> data or CIF (`structure: {file, format, index}`); md, relax and gcmc
> write `traj_format: xdatcar | extxyz | lammps-dump`; `dataset` reads
> labelled extxyz (`reader: extxyz`); `ak.AkitsushimaCalculator` wraps a
> force function for ASE.
>
> 0.2.0a25 (ADR-0035): `akitsushima build` and `ak.build` make structures:
> bulk crystals, supercells (any integer matrix or `min_width`), random
> alloys with exact counts, vacancies, (hkl) slabs with chosen
> termination (ESM-ready cells), rattle, strain and scale; every random
> step takes a seed and the choices are recorded in `<output>_build.json`.
>
> 0.2.0a26 (ADR-0036): NPT `cell_dof: iso | aniso | tri | xy | xy_iso | z`
> -- per-axis, full-cell (with shear), in-plane and out-of-plane barostats
> (MTTK fully flexible cell); `xy` / `xy_iso` run with ESM at P = 0;
> `baro_thermostat: langevin` (Langevin piston) for strain fluctuations.
>
> 0.2.0b1: first beta.  No code change from 0.2.0a26; tier A of the
> ADR-0028 roadmap (steps 1-11, ADR-0030 to ADR-0036) is complete.
>
> 0.3.0a1 (ADR-0037): `akitsushima active` -- active learning driven by
> the ensemble uncertainty: md / anneal exploration, a trust window on
> sigma_F,max with farthest-point thinning, VASP inputs or an external
> code as labeller, retraining; resumable.  md and anneal take
> `stop_sigma`.  First alpha of tier B (ADR-0028 addendum 5).
>
> 0.3.0a2 (ADR-0038): G5 terms with integer zeta up to 6 are computed
> from Cartesian moment tensors, at a cost linear in the number of
> neighbours.  Same descriptor and models (no retraining; equal to the
> pair loop to 1e-13 of each column).  The GeSn atom-centred ELM force
> call is 1.7-1.9 times faster on one thread, and the gain grows with
> r_cut.  `ACSF(..., angular_kernel="pairs")` keeps the former kernel.
>
> 0.3.0a3 (ADR-0039): `md` NVT gets the CSVR thermostat (`thermostat:
> csvr`, Bussi, Donadio and Parrinello 2007), and `anneal` md_quench gets
> `thermostat: nhc | langevin | csvr` with the mean MD temperature of
> every chunk (`T_md`) in its ledger.  The anneal template's md_quench no
> longer fails every MD chunk on its NPT-only `tau_baro`.  The parallel
> pair, SW and Tersoff kernels no longer overrun their per-thread buffers
> when a numba cache built with one thread is loaded with more.
>
> 0.4.0a3 (ADR-0053): FIRE gives the 9 cell coordinates a mass,
> `relax.fire_cell_mass` (default `auto` = N^(1/3)), so the cell no longer
> sets FIRE's time step in large cells: a 108864-atom GeSn SW cell that
> stopped at 2000 steps now converges in 816.  Every FIRE variable-cell
> relaxation takes a new path; `fire_cell_mass: 1` restores the old one
> (and continues 0.4.0a2 checkpoints).
>
> 0.4.0a2 (ADR-0052): `mdspec` reads extxyz and LAMMPS-dump trajectories
> (forces too: `forces: trajectory`), computes the Gamma modes of the
> mode estimator with the MD's own potential, and takes `dipole.model:
> auto`; `md` writes velocities with `traj_velocities: true`.
>
> 0.4.0a1 (ADR-0050, ADR-0051): new `mdspec` command -- spectra from MD
> trajectories, starting with the complex permittivity eps(omega) from the
> total dipole M(t) by the fluctuation-dissipation theorem (point charges,
> Born tensors or an external dipole series; raw / f-sum / mode
> estimators; optics, mode table, fits, `--check`).  Ported from nozuchi
> 0.2.0a2 without behaviour changes; `--convert-nozuchi` translates a
> nozuchi input.
>
> 0.3.0a11 (ADR-0049): new `raman` command -- first-order Raman spectra
> in the harmonic approximation: Gamma modes of any cell (primitive or
> alloy supercell), Raman / IR activity from characters (spglib), and with
> a `polarizability:` block (bond polarisability model) relative
> intensities for named polarisation geometries and the powder average,
> several configurations averaged.  `born:` reads Z* and eps_inf from a
> VASP OUTCAR (site mapping ported from nozuchi).
>
> 0.3.0a10 (ADR-0047): Stillinger-Weber three-body forces are analytic
> instead of finite differences (energies unchanged, forces within
> ~1e-9 eV/A of before), the multi-species SW and Tersoff kernels run on
> numba's threads, and pair potentials group their pairs with numpy (bit
> for bit as before).  One force call, one thread: SW GeSn 1.7-2.6 times
> and Buckingham MgO 3.8 times faster; SW GeSn at 13608 atoms on ten
> threads 5 times faster than 0.3.0a9 on one.
>
> 0.3.0a9 (ADR-0045): `anneal` md_quench uses CSVR by default at nvt
> (damping 50 fs; npt keeps the Nose-Hoover chain).  After each step down
> of the schedule T relaxes with a time constant of 2 damping whatever the
> system size, while the Nose-Hoover chain with its standard masses slows
> down with the number of atoms and with tau_nhc.  An anneal input that
> sets `tau_nhc` or `nhc_chain` without `thermostat` is refused (add
> `thermostat: nhc` to keep the former behaviour), and a restart refuses a
> thermostat other than the checkpoint's.  The `md` NVT template gives the
> response times of the three thermostats.
>
> 0.3.0a8 (ADR-0044): the Nose-Hoover thermostat of NVT and NPT is a
> chain of `nhc_chain` links, three by default (`md`, the `anneal` md
> section and the Python API).  With the former two links a run started
> at a minimum took about ln(1.9) tau sqrt(n_dof) to reach its target
> (14.7 ps at 13608 atoms with tau 100 fs); three links take about 1 ps,
> and anneal segments now follow the schedule within 1 % in their second
> half.  `nhc_chain: 2` reproduces 0.3.0a7 bit for bit, and 0.3.0a7
> checkpoints load as two links.
>
> 0.3.0a7 (ADR-0043): EAM (Zhou and setfl tables) in numba, PPPM on the
> real FFT with numba charge assignment, and ESM Green functions applied
> in O(Nz) without matrices: one force call, one thread, Zhou Ni 4000
> atoms 4.2x, Buckingham + PPPM MgO 1.6x, ESM 64x64x256 7-8x, results
> within 1e-13 of 0.3.0a6.  Force functions take `need_stress=False`
> (stress returned as None); MD skips the Coulomb stress on steps that
> write no pressure or trajectory frame, and phonon / BTE / NEB / TSA /
> GCMC energy calls never compute it.
>
> 0.3.0a6 (ADR-0042): `akitsushima bte` computes the lattice thermal
> conductivity from third-order force constants in the relaxation-time
> approximation, for any akitsushima potential (MLIPs included): fc3 by
> two-step central differences reduced by the space group, three-phonon
> linewidths with a Gaussian delta, optional isotope / alloy (Tamura) and
> boundary scattering, quantum or classical occupation, NAC for polar
> crystals.  Checked against phono3py with the same force constants
> (kappa and linewidths to ~1e-5 on Si and AlN).
>
> 0.3.0a5 (ADR-0041): variable-cell relaxation (`relax_cell: true`) now
> moves the cell by a deformation gradient (`cell_coordinates: strain`,
> the new default), so a symmetric crystal keeps its space group: Si (fcc
> primitive) and AlN (wurtzite) stay Fd-3m / P6_3mc at symprec 1e-5 for
> all six methods, where the raw lattice components left ~1e-5 A of
> symmetry-breaking strain.  Default `relax_cell` results change within
> the convergence tolerances; `cell_coordinates: lattice` restores the
> old path bit for bit.  `method: rfo` with `relax_cell` now converges;
> `cell_update_scheme: vasp` is corrected (`lattice @ stress`) and
> deprecated with strain coordinates.
>
> 0.3.0a4 (ADR-0040): `symmetry: true` in `phonon`, `qha`, `gruneisen`,
> `kappa` (Slack section) and `elastic` uses the space group (spglib,
> `pip install akitsushima[symmetry]`) to displace only inequivalent
> atoms along a minimal set of directions and to strain only the modes
> the symmetry needs.  Diamond Si: 12 -> 1 force calls for the phonons,
> 37 -> 10 strained cells for C, equal to the full method to round-off.
> Off by default; a P1 cell gives the full method bit for bit.

A classical force-field and machine-learning potential toolkit for atomistic
simulation. `akitsushima` fits interatomic potentials, relaxes structures,
runs molecular dynamics, and computes phonon / thermal-transport properties
for crystalline and molecular systems — from the command line or as a Python
library.

## Features

### Force fields

**Pair potentials** (ionic / van der Waals)
- Buckingham — `A exp(-r/rho) - C/r^6`
- Born-Mayer — `A exp(-r/rho)`
- Lennard-Jones — `4epsilon [(sigma/r)^12 - (sigma/r)^6]`
- Morse — `D_e [1 - exp(-alpha(r-r0))]^2`

**Many-body potentials** (covalent / metallic)
- Stillinger-Weber — 2-body + 3-body angle term (Si, Ge, H2O)
- Tersoff — bond-order (`b_ij`) with coordination (Si, C, Ge, GaAs);
  multi-species as a LAMMPS-style (i, j, k) table or through the
  Tersoff (1989) mixing rules
- EAM — embedding `F(rho_i)` + pair `phi(r)` (metals): the analytic
  alloy form of Zhou, Johnson & Wadley (2004), or a setfl table
  (LAMMPS `eam/alloy`)

**Long-range Coulomb**
- Ewald summation — vectorised, `O(N^1.5)` for N ≲ 2,000
- PPPM / SPME — FFT mesh, `O(N log N)` for N ≳ 500
- ESM — Effective Screening Medium (open / metallic boundary conditions)

All three are chosen in the potential section and reach every solver
through the same force function (ADR-0008):

```yaml
potential:
  type: buckingham
  potential_json: ./fitting_output/fitted_potential.json
  r_cut: 5.5
  coulomb_method: esm          # ewald | pppm | esm
  esm:
    bc: bc1                    # bc1 vacuum|slab|vacuum, bc2 metal|slab|metal,
                               # bc3 vacuum|slab|metal (Otani-Sugino / QE)
    sigma: 0.5
    n_grid_xy: 32
    n_grid_z: 512
    external_field: 0.0        # V/A (bc2)
```

With ESM the slab is centred at fractional z = 0 (the medium starts at
fractional z = 1/2), as in Quantum ESPRESSO.  relax may relax only a_x and
b_y; NPT only with `cell_dof: xy` or `xy_iso` at P = 0 (ADR-0036) and
kappa refuses ESM; phonon needs supercell [N, N, 1]; gcmc
inserts charged species only with bc2 / bc3.  The z-grid error is
O(dz^2 q^2): ~20 meV per +-1 ion at dz = 0.125 A, 1.3 at dz = 0.031 A.

**Machine-learning potentials**
- `ELMModel` — Extreme Learning Machine (single hidden layer)
- `QuantizedELMModel` — ELM with quantised hidden-layer weights
- `LinearModel` — linear model: OLS, ridge, smoothness-penalised ridge, LASSO, Elastic Net, PLS, Bayesian ridge, ARD
- `NNModel` — PyTorch feed-forward network (lazily imported)
- `AtomicELMModel` / `AtomicQELMModel` / `AtomicNNModel` — atom-centred (Behler-Parrinello) models E = sum_i eps_s(G_i), one function per species (ADR-0005)
- `ACSF` — atom-centred symmetry functions G2 / G4 / G5 with analytic position and cell derivatives

### Simulation capabilities

| Task | Description |
|---|---|
| `fit`   | Fit classical or ML potential parameters against DFT reference data (POSCAR geometry + CSV energy/force/stress). |
| `relax` | Geometry optimisation: FIRE, L-BFGS, DIIS, RMM-DIIS, RFO — ionic or variable-cell, with transition-state refinement. |
| `md`    | Molecular dynamics: NVE, NVT (Nosé-Hoover chain), NPT (isotropic, per-axis, full cell, in-plane or out-of-plane; ADR-0036). |
| `anneal` | Global structure search: MD-quench annealing on a temperature schedule, or basin hopping (ADR-0022). |
| `gcmc` / `osmc` | Grand-canonical (μVT) MC/MD: discrete insertion/deletion (`gcmc`) or a continuous fractional atom, CFCMC (`osmc`, ADR-0009). |
| `elastic` | Elastic tensor (clamped- and relaxed-ion), B / G / E / ν as Voigt, Reuss and Hill averages, Born stability, sound velocities, Debye temperature, internal strain (ADR-0017). |

### Post-processing

- **CI-NEB** — Climbing-image nudged elastic band
- **`TransitionStateAnalyser`** — Hessian, IRC, saddle-point verification, TST reaction rate
- **`PhononCalculator`** — Finite-displacement force constants, band structure, DOS, Debye temperature, `thermal_properties` (F, U, S, C_v, Theta_D(T); ADR-0018)
- **`ThermalConductivityCalculator`** — Green-Kubo MD + Slack-Debye estimate
- **`ElasticConstantsCalculator`, `elastic_moduli`** — elastic tensor and polycrystalline moduli (ADR-0017)

### Performance

- **`NeighbourList`** — cell-linked list / SciPy `cKDTree` back-end, with a Verlet skin for lazy rebuilds during MD.
- **Numba JIT** — optional `@njit` compilation of the pair / many-body loops,
  used automatically whenever numba is importable, with a transparent NumPy
  fallback when it isn't. Measured on 64-atom silicon: Stillinger-Weber
  66 → 4.2 ms/step, Tersoff 117 → 2.5 ms/step. Set `NUMBA_ENABLE=0` to force
  the NumPy path.
- **MPI (mpi4py)** — domain-split + `Allreduce` parallelism across ranks, available for every potential, for NEB image evaluation, phonon finite displacements, and the fitting loss function — with a transparent single-process fallback when mpi4py isn't installed.

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .
```

The only hard dependencies are NumPy and SciPy. Everything else is an
optional extra, installed as needed:

```bash
pip install -e ".[numba]"       # JIT-compiled potential kernels
pip install -e ".[mpi]"         # multi-rank parallelism (requires mpi4py/mpirun)
pip install -e ".[torch]"       # NNModel
pip install -e ".[yaml]"        # YAML config files for the CLI
pip install -e ".[plotting]"    # matplotlib output (NEB/phonon/relax plots)
pip install -e ".[ase]"         # CIF input and the ASE calculator (ADR-0034)
pip install -e ".[all]"         # everything above
```

`NNModel` / `AtomicNNModel` are imported lazily — `import akitsushima`,
and every other class in the package, never pulls in `torch` unless you
actually reference one of them.

## Quick start

### Command line

Every task reads a single YAML input file (`fit` instead reads a training
CSV plus a JSON/YAML fitting config). Generate a starting template with
`--dump-default`:

```bash
akitsushima --dump-default relax > relax.yaml
akitsushima relax relax.yaml

akitsushima --dump-default nvt   > md_nvt.yaml
akitsushima md md_nvt.yaml
```

Run `akitsushima --help` for the full list of tasks and options.

### Python API

```python
import numpy as np
from akitsushima import (
    PotentialEngine,
    LennardJonesPotential,
    NeighbourList,
    make_force_fn,
)

# Lennard-Jones argon: sigma = 3.405 A, epsilon = 0.0103 eV
# (Verlet, Phys. Rev. 159, 98 (1967)); r_cut = 10 A = 2.94 sigma.
lj_params = {frozenset({"Ar"}): {"epsilon": 0.0103, "sigma": 3.405}}

# The cell edge must exceed 2 * (r_cut + skin) = 2 * (10.0 + 0.5) = 21 A.
lattice = np.eye(3) * 24.0
# Two atoms 4.0 A apart, just beyond the minimum at 2^(1/6) sigma = 3.82 A.
positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
atom_types = ["Ar", "Ar"]

engine = PotentialEngine().add(LennardJonesPotential(lj_params, r_cut=10.0))
fn = make_force_fn(engine)

nl = NeighbourList(r_cut=10.0)
nl.build(lattice, positions)

energy, forces, stress = fn(lattice, positions, atom_types, nl)
# energy = -9.4170e-03 eV (force-shifted LJ, the default shift=True)
```

### Python API: a machine-learning potential

Training data, an atom-centred ELM on ACSF descriptors, and the model as a
force field (the same steps as `akitsushima mlip` and `akitsushima relax`):

```python
import numpy as np
import akitsushima as ak
from akitsushima.fitting import AtomicELMModel
from akitsushima.fitting.descriptors import ACSF

# Training data.  Real data come from DFT: akitsushima.io.build_training_set
# turns VASP runs into a training set and akitsushima.io.load_training_set
# reads it back as the structure dicts below.  Here Stillinger-Weber silicon
# (Stillinger & Weber, PRB 31, 5262 (1985)) stands in for DFT.
sw = ak.force_fn_from_config({"type": "SW", "species": ["Si"], "params": {
    "epsilon": 2.1683, "sigma": 2.0951, "a": 1.80, "gamma": 1.20, "lambda": 21.0,
    "A": 7.049556277, "B": 0.6022245584, "p": 4.0, "q": 0.0, "costheta0": -1 / 3}})
diamond = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                    [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])
cells = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])
frac = (diamond[None] + cells[:, None]).reshape(-1, 3) / 2       # 64-atom 2x2x2 cell
rng = np.random.default_rng(0)
structures = []
for _ in range(8):
    lattice = np.eye(3) * 10.862 * rng.uniform(0.98, 1.02)
    positions = frac @ lattice + rng.normal(0.0, 0.05, frac.shape)
    types = ["Si"] * len(positions)
    e, f, s = sw(lattice, positions, types)
    structures.append({"lattice": lattice, "positions": positions, "atom_types": types,
                       "ref_energy": e / len(types),        # eV/atom
                       "ref_forces": f,                     # eV/A, (N, 3)
                       "ref_stress": s})                    # eV/A^3, compressive-positive

# Descriptor and model.  A pair baseline (e.g. ZBL) and isolated-atom
# energies go in the constructor: AtomicELMModel(baseline=..., isolated_atom_energy=...).
acsf = ACSF.from_grid(["Si"], 3.771, g2={"eta": [4.0], "rs": [2.2, 2.6, 3.0, 3.4]},
                      g4={"eta": [0.05], "zeta": [1, 2], "lambda": [-1, 1]})
model = AtomicELMModel(n_hidden=64, seed=42, verbose=False).fit(structures, acsf)
print(model.score_structures(structures, acsf)["F"]["RMSE"], "eV/A training force RMSE")
model.save("si_model.npz")

# The model is a force field for every solver.
fn = ak.make_force_fn(model)
result = ak.GeometryOptimiser(structures[0]["lattice"], structures[0]["positions"],
                              structures[0]["atom_types"], force_fn=fn, relax_cell=True,
                              fmax=0.01, verbose=False).run()
print(result["energy"] / 64, "eV/atom after relaxation")
```

## CLI reference

| Command | Purpose |
|---|---|
| `relax`  | Geometry optimisation (FIRE / L-BFGS / DIIS / RMM-DIIS / RFO); restartable |
| `md`     | Molecular dynamics (NVT with Nose-Hoover chain or Langevin BAOAB / NVE / NPT); restartable, bit for bit |
| `anneal` | MD-quench annealing or basin hopping; distinct minima, restartable |
| `gcmc`   | Grand-canonical Monte Carlo MD (μVT) |
| `neb`    | Climbing-image NEB transition-state search |
| `tsa`    | Transition-state analysis (Hessian / IRC / TST rate) |
| `osmc`   | CFCMC grand-canonical MC/MD (fractional atom) |
| `esm`    | Effective screening medium |
| `phonon` | Phonon band structure, DOS, Debye temperature, F / U / S / C_v (mode sum) |
| `kappa`  | Thermal conductivity (Green-Kubo + optional Slack-Debye) |
| `elastic` | Elastic tensor, B / G / E / ν (Voigt / Reuss / Hill), Born stability, Debye temperature |
| `gruneisen` | Mode Grüneisen parameters (q-path, mesh, C_v-weighted mean γ(T)) |
| `qha`    | Quasi-harmonic thermal expansion, B(T), C_p, thermodynamic γ(T) |
| `dielectric` | Static dielectric tensor, LO-TO splitting, IR modes (Born charges) |
| `raman`  | Raman spectra: activity from characters, polarisation geometries, powder average (ADR-0049) |
| `mdspec` | MD spectra: permittivity eps(omega) from M(t) by FDT, optics, fits, run diagnostics (ADR-0051) |
| `piezo`  | Relaxed-ion piezoelectric e, d and coupling k (rigid ion) |
| `nemd`   | Non-equilibrium MD: T profile, local heat flux, kappa, Kapitza resistance |
| `mlip`   | Fit a machine-learning potential (ELM / QELM / NN / linear) |
| `fit`    | Fit a classical potential (LJ / Mie / Morse / Born-Mayer / Yukawa / Buckingham / Stillinger-Weber / Tersoff / EAM) |
| `dataset` | `fit` / `mlip` training set from VASP runs (vasprun.xml / OUTCAR), split by run |
| `active` | Active learning: explore with an ensemble MLIP, select uncertain structures, label (VASP inputs / external code), retrain |

### Elastic constants (ADR-0017)

```bash
akitsushima --dump-default elastic > elastic.yaml   # Tersoff Si template
akitsushima elastic elastic.yaml
```

The cell and ions are relaxed first (`pre_relax`), then each of the six
Voigt strains is applied at +-0.25, 0.5 and 1 % (`strains`).  Each column
of C is the slope of a straight-line fit of the stress (RMS residual
reported per mode); the clamped-ion C^0 uses affinely strained cells, the
relaxed-ion C relaxes the ions in each.  The POSCAR must already be a
supercell with r_cut + skin below half its smallest perpendicular width
at every strain -- this is checked and refused, never extended
automatically.  ESM electrostatics is refused (straining moves its
boundaries).

Output (`elastic_output/`): `elastic_results.json` (C, C^0, S, eigenvalues
and Born stability, B / G / E / nu for Voigt, Reuss and Hill -- E and nu
from each scheme's own B and G, as pymatgen -- Pugh ratio, universal
anisotropy A^U, density, v_l / v_t / v_m, Debye temperature, internal
strain du/deta in A per unit strain, and an input hash),
`elastic_tensor.dat`, `internal_strain.dat` and `POSCAR_reference`.
`example/elastic/check_elastic.py` checks Tersoff Si against Tersoff's
published C11 / C12 / C44 / C44^0 and Kleinman zeta, ASE's own Tersoff
and pymatgen, and ZnO (Buckingham + PPPM) against the hexagonal relations.

### Grüneisen parameters and the quasi-harmonic approximation (ADR-0018)

```bash
akitsushima --dump-default gruneisen > g.yaml   # Tersoff Si, primitive POSCAR
akitsushima gruneisen g.yaml
akitsushima --dump-default qha > q.yaml
akitsushima qha q.yaml
```

Both start from the primitive cell (as `phonon`), relax it fully
(`pre_relax`), then relax shape and ions at each fixed volume in the
phonon supercell (`GeometryOptimiser(fix_volume=True)`, also available as
`relax.fix_volume`).  `gruneisen` uses V0 (1 -+ 0.01): gamma =
-(V/2w^2)<e|dD/dV|e> along the q-path and on a mesh, with the
C_v-weighted mean gamma(T).  `qha` scans 7 volumes over +-5 %, fits
E(V) + F_vib(V, T) per temperature (Birch-Murnaghan or Vinet) and reports
V(T), B(T), alpha(T), C_v, C_p and gamma(T); it stops on an imaginary mode
(below -0.05 THz) unless `skip_unstable_volumes: true`, and reports
nothing above the temperature where V(T) leaves the scanned volumes.
Both match phonopy's own Grueneisen and QHA (tests/test_qha.py).

### Born charges, dielectric tensor and piezoelectric response (ADR-0019)

```yaml
born:                         # top level; also turns on NAC in phonon/gruneisen/qha
  epsilon_inf: 3.7            # scalar, [xx, yy, zz] or 3x3; default 1 (warned)
  born_charges:               # default: the potential's point charges
    per_species: {Zn: 2.1, O: -2.1}       # scalar, diagonal or 3x3
    per_atom: {4: [[...], [...], [...]]}  # 1-based POSCAR index, wins
  clamped_ion_piezo: [[...6...], [...], [...]]   # C/m^2, piezo only
```

Z* and eps_inf can also come from a VASP LEPSILON / LCALCEPS run
(ADR-0049): `epsilon_inf: {outcar: OUTCAR}` and `born_charges: {outcar:
OUTCAR, unit_cell: POSCAR_of_that_run, supercell: [2, 2, 2]}`, which maps
every atom onto a site of the DFT cell by position (rigid translations
removed).  Per atom the priority is per_atom > per_species > outcar >
the potential's point charges.

Z[k, a, b] = dP_a / du_kb (phonopy's and VASP's order); the sum rule is
imposed.  `akitsushima dielectric` gives eps_0 = eps_inf + (4 pi / V)
sum_m S_m / w_m^2 with the per-mode oscillator strengths, mode effective
charges, IR intensities and LO frequencies along x, y, z.  `akitsushima
piezo` takes the internal strain of `elastic` (or `load_elastic`) and
gives e (C/m^2), d (pC/N) and k.  The model is rigid ion: with point
charges only, eps_inf = 1 and the electronic (clamped-ion) piezoelectric
term is zero unless given.  Without charges or a `born:` block (MLIP,
EAM, Tersoff, SW) both commands refuse.

### Raman spectra (ADR-0049)

```yaml
polarizability:               # top level; omit for frequencies + activity only
  source: bpm                 # bond polarisability model, per species pair:
  bonds:                      #   alpha_l - alpha_p = alpha_diff
    Si-Si: {r_cut: 2.8, r0: 2.35, alpha_diff: 1.0,   #   + (d_alpha_l - d_alpha_p)(R - r0)
            d_alpha_l: 2.0, d_alpha_p: 0.5}
raman:
  supercell: auto             # Gamma is exact from any valid supercell
  structures: [cfg1.vasp, cfg2.vasp]   # optional: averaged per atom, with std
  geometries:
    - {name: "z(xy)-z", incident: [1, 0, 0], scattered: [0, 1, 0]}
    - {name: "110", incident: [1, 1, 0], scattered: [0, 0, 1], basis: lattice}
  temperature: 300.0          # (n + 1) / w;  laser_nm: 532 adds (w_L - w)^4
  fwhm_cm: 3.0
```

`akitsushima raman` diagonalises the Gamma dynamical matrix of each cell,
groups degenerate modes and decides Raman and IR activity from the
characters of the space-group operations (spglib; no character tables).
With a polarizability derivative A[k, a, b, c] = d alpha_ab / d u_kc it
gives the mode Raman tensors R_m = sum_k A_k e_m(k) / sqrt(M_k), the
intensity (e_s . R . e_i)^2 for each named geometry, the powder
I_par = 45 a^2 + 4 g^2, I_perp = 3 g^2 and rho, and Lorentzian spectra
(raw and normalised; the geometries share one normalisation).  Only
relative intensities are meaningful.  Outputs: raman_modes.dat,
raman_spectrum.dat, raman_spectrum_normalised.dat, raman_results.json and
raman_spectrum.png.  With `born:` the LO frequencies along x, y, z are
listed (the electro-optic LO intensity term is not included).  VASP
finite-difference Raman tensors and the BPM fit follow in the next stage
(ADR-0046).

### MD spectra: permittivity from M(t) (ADR-0050, ADR-0051)

```yaml
born:
  epsilon_inf: 1.0                      # or {outcar: dfpt/OUTCAR}
  # born_charges: {outcar: dfpt/OUTCAR, unit_cell: dfpt/POSCAR, supercell: [3, 3, 2]}
mdspec:
  trajectory: {files: [md_output/XDATCAR_nvt], timestep_fs: 1.0, frame_every: 2}
  temperature: auto
  temperature_source: [md_output/thermo.log]
  dipole: {model: point_charge, charges: {Mg: 2.0, O: -2.0}}   # | born_tensor | external
  spectrum: {freq_unit: THz, freq_max: 30, estimator: auto}  # auto | mode | fsum | raw
  # modes: {unit_cell: POSCAR_unit, supercell: [8, 8, 8]}  # computed with potential:
  # forces: trajectory                  # force-based static value, reported only
  output: {dir: mdspec_output, prefix: mgo}
```

`akitsushima mdspec input.yaml` reads the trajectory once (streaming
XDATCAR reader, optional numba) and gives eps_ab(omega) = eps' - i eps''
with eps_inf + [<dM_a dM_b> - i w int <dM_a(0) dM_b(t)> e^{-iwt} dt] /
(eps0 V kB T), the static value by the raw, f-sum and mode estimators
(the last needs the Gamma eigenvectors), optics (n, kappa, R, absorption,
loss function, TO / LO, LST), the mode table, fits (Debye ... Lorentz,
Gervais-Piriou) and a Kramers-Kronig check.  `--check` diagnoses the runs
(equilibration, drift, statistics) and exits 1 on any WARN.  Relative
paths are read against the YAML file's directory.  Validated in nozuchi
(Tier 2a / 3a within 1.5 % of a force-based reference); the port
reproduces nozuchi 0.2.0a2 on those inputs (ADR-0051).  VDOS, SED and the
Raman temperature shift follow (ADR-0050 stages P3-P6).

Since 0.4.0a2 (ADR-0052): trajectories may be XDATCAR, extxyz or LAMMPS
dump (detected from the file; atoms of one species must be contiguous, as
akitsushima writes them; a dump is analysed in the LAMMPS orientation and
Cartesian inputs for another orientation are refused).  With a
`potential:` block the Gamma modes are computed with it: for `modes:
{unit_cell, supercell}` on the unit cell implied by the MD lattice, or on
the MD cell itself up to 1000 atoms (relaxed at fixed cell; written as
`<prefix>_gamma_modes.npz` and `<prefix>_gamma_unit.vasp`).  `dipole.model:
auto` (the default) takes born.born_charges, then the potential's point
charges, then `dipole.charges`.  `md` writes velocities (extxyz A/fs,
LAMMPS dump A/ps) with `traj_velocities: true`.

### NEMD and the decomposition of kappa (ADR-0020)

`akitsushima nemd` places a hot and a cold Langevin bath (fractional slabs
along a lattice vector) in a periodic cell, so heat flows both ways
round: J = Qdot / (2A).  It reports the temperature profile, the local
heat flux per bin (from the per-atom heat currents), kappa = J / |dT/dz|
from each of the two segments with block errors, the Kapitza resistance
of interfaces you name, and steady-state indicators (bath balance,
profile drift, gradient sign).  kappa depends on the length: run several
and extrapolate 1/kappa against 1/L (example/nemd).  Long runs: launch
detached; the checkpoint (`checkpoint_every`) lets a re-run continue bit
for bit.

`md` and `relax` write a checkpoint too (ADR-0031): `md_checkpoint.npz`
every `checkpoint_every` steps (default 1000) and `relax_checkpoint.npz`
(default every 10 steps) in `output_dir`.  Running the same input again
continues from it -- resubmitting a job after its time limit is enough --
and `restart: false` discards it and starts afresh.
- An md run continues bit for bit: the state, the averages and the
  random-number state are restored, and `thermo.log` and the XDATCAR
  are cut back to the checkpoint before new lines are appended.
  Raising `n_steps` extends a finished run.
- relax continues from the checkpointed structure with the optimiser's
  own state started afresh; raising `max_steps` continues a run that
  stopped there, and a converged run is not redone.
- A restart whose settings or potential differ from the checkpoint's
  (the potential section and the bytes of the files it names are
  hashed) stops with the list of differences.

`kappa.decompose: {species: true, regions: {axis: c, bounds: [0, 0.5, 1]}}`
splits the Green-Kubo tensor into kappa_g = (V/k_B T^2) int <J_g(0) J(t)>,
which add up to kappa exactly.

### Structure and trajectory files, ASE (ADR-0034)

Every command reads its structure from `structure: poscar:` as before or
from `structure: file:`, whose format follows the name (`format: auto`):
POSCAR / CONTCAR / `*.vasp`, `*.xyz` / `*.extxyz`, `*.data` / `*.lmp`
(LAMMPS data, atom_style atomic or charge) and `*.cif` (read through ASE;
partial occupancies are refused).  `index` picks a frame of a multi-frame
file (default the last); `species` names the LAMMPS atom types when the
Masses section has no `# <element>` comments.  Fixed atoms come from
POSCAR only, and output structures stay POSCAR.

`md`, `relax` and `gcmc` / `osmc` take `traj_format`: `xdatcar` (default),
`extxyz` (`traj_<task>.extxyz` with energy, forces and stress in ASE's
sign; gcmc frames carry the energy only) or `lammps-dump`
(`traj_<task>.lammpstrj`, `id type element x y z fx fy fz`, metal units).
LAMMPS files hold the cell in LAMMPS's orientation (a along x, b in the xy
plane): writing rotates the structure, so it reads back as the same
crystal turned.  `dataset` with `reader: extxyz` reads labelled extxyz
files, one run per file (`extxyz_keys` renames the labels).

```python
import akitsushima as ak
s = ak.read_structure("GeSn.cif")                    # dict as parse_poscar
ak.write_structure("GeSn.data", s["lattice"], s["positions"], s["atom_types"])
atoms = ak.to_ase_atoms(s["lattice"], s["positions"], s["atom_types"],
                        calculator=ak.AkitsushimaCalculator(force_fn))
atoms.get_stress()      # ASE's sign; sigma_E etc. in calc.results for an ensemble
```

### Building structures (ADR-0035)

`akitsushima build build.yaml` (template: `--dump-default build`) starts
from `bulk` (diamond, fcc, bcc, hcp, zincblende, wurtzite, rocksalt) or a
file and applies `steps` in order, writing `output` in any format above:

| Step | Options |
|---|---|
| `supercell` | `[n1, n2, n3]`, a 3x3 integer matrix, or `{min_width: A}` |
| `substitute` | `{from: Ge, to: {Sn: 8}, seed}` -- exact counts (a float is a fraction) |
| `vacancy` | `{species, count, seed}` or `{indices: [...]}` |
| `slab` | `{miller, layers, vacuum, termination, center: zero / middle, orthogonal_ab}` |
| `rattle` | `{sigma, seed}` |
| `strain` | `{epsilon: 3x3}` or `{volume, shear, seed}` (volume exact) |
| `scale` | `{factor}` or `{volume}` |

`repeat: n` builds n structures with every seed offset by k (an extxyz
output holds them as frames, other formats get `_000`, `_001`, ...).  The
record `<output>_build.json` keeps the input, the chosen indices and, per
structure, the composition, volume and shortest distance per species pair.
A slab's `layers` counts repeat units of thickness d_hkl; its atomic planes
are listed and `termination: k` puts plane k on top.  The slab cell has c
along z and is centred on z = 0 by default, as ESM needs; `orthogonal_ab`
finds a rectangular in-plane cell.  The same functions are in the API:

```python
import akitsushima as ak
s = ak.build.bulk("diamond", "Ge", 5.7624)
s = ak.build.substitute(ak.build.supercell(s, [2, 2, 2]), "Ge", {"Sn": 8}, seed=1)
z = ak.build.slab(ak.build.bulk("wurtzite", ["Zn", "O"], 3.25), (0, 0, 0, 1),
                  layers=4, vacuum=15.0, orthogonal_ab=True)
```

### Anisotropic NPT (ADR-0036)

`md` with `ensemble: npt` takes `cell_dof` (also `NPTSimulation(cell_dof=)`
and anneal's md section).  x, y, z are Cartesian; the target is the scalar
`pressure`:

| `cell_dof` | Cell degrees of freedom |
|---|---|
| `iso` (default) | volume only, shape kept (unchanged, bit for bit) |
| `aniso` | Lx, Ly, Lz independently; no shear |
| `tri` | all six strain components |
| `xy` | Lx, Ly independently; Lz and shear fixed (ESM slabs) |
| `xy_iso` | Lx, Ly scaled together (ESM slabs) |
| `z` | Lz only (a film held at its substrate's in-plane cell) |

The integrator is the MTTK fully flexible cell restricted to these strain
generators: each has the iso barostat mass, one Nose-Hoover chain drives
them all, and the conserved quantity H_NPT is reported as before.  Other
modes add Pxx, Pyy, Pzz (kinetic part included) and a, b, c to thermo.log
(`tri` also the shear components and the angles) and return the mean
lattice, the mean pressure tensor and a, b, c with their spread.  With ESM
only `xy` / `xy_iso` at `pressure: 0` are allowed: Lz, which places the
boundaries, stays fixed.  Cells are not flipped; `tri` warns when an angle
leaves [45, 135] degrees.

`baro_thermostat: nhc` (default) thermostats the barostat with one
Nose-Hoover chain and conserves H_NPT.  In a crystal the strain degrees of
freedom barely exchange energy with each other or with the atoms, so one
chain fixes only their total: means (pressure, volume, lattice) are right,
but volume and strain FLUCTUATIONS are not (SW Si, 300 K: C44 from strain
fluctuations 85 GPa against 56).  `baro_thermostat: langevin` (Langevin
piston) puts an Ornstein-Uhlenbeck step with friction time `tau_baro` on
every barostat momentum; use it for compressibility or elastic constants
from fluctuations.  H_NPT then subtracts the heat the noise added, so it
remains a drift check.

### Global structure search (ADR-0022)

`akitsushima anneal anneal.yaml` (template: `--dump-default anneal`).
Cycle 0 relaxes the input, which is the relax-only reference. Then:

- `method: md_quench` runs one continuous NVT or NPT trajectory from that
  relaxed structure. The trajectory follows a temperature schedule
  (constant, linear, geometric or a list), and a copy of the structure
  is relaxed every `quench_every` steps.
- `method: basin_hopping` (Wales & Doye 1997) displaces every atom
  inside a sphere and, with `relax_cell`, strains the cell. It relaxes
  the trial and accepts it by the Metropolis rule on the relaxed energy.
  The step adapts towards 50 % acceptance.

The `anneal.relax` and `anneal.md` blocks take the `relax` and `md` key
names. With `pressure` set, every quench minimises H = E + PV.

A cycle fails when the solver raises, the energy is not finite, or the
energy drops below `min_energy_per_atom`, which guards against an MLIP
that collapses under extrapolation. A failed cycle is logged and
skipped, and `max_consecutive_failures` failures in a row stop the run.

Every cycle is appended to `anneal_log.jsonl` and checkpointed, so
`restart: true` continues an interrupted run and reproduces the
uninterrupted one. Distinct minima go to `POSCAR_rank01..` and
`minima.csv`, the lowest to `POSCAR_best`, and the history to
`anneal.png`.

### Fitting a classical potential

```bash
akitsushima fit --potential lj --write-config fit_lj.yaml
# edit fit_lj.yaml, then:
akitsushima fit --potential lj --config fit_lj.yaml --training-csv training.csv
```

The training CSV has one row per energy/force/stress component:

```
col 0  target value  [eV/atom | eV/A | eV/A^3]
col 1  category       "E" | "F" | "S"
col 2  struct_idx     integer structure index
```

Geometry for each `struct_idx` is read from `poscar_dir/<struct_idx>/POSCAR`.

#### Optimisation

Fitting runs in two stages, both driven from the config file:

1. **Global search (optional)** -- `use_de: true` runs scipy's
   Differential Evolution over the parameter bounds first
   (`de_maxiter`, `de_popsize`; `--use-de` on the command line does the
   same).  The `--write-config` templates switch it on for `sw` only:
   for `tersoff` it settled in a wrong basin along the beta / c / d
   directions the data cannot resolve, where the local search alone
   recovers the potential (ADR-0007).  Its cost is `de_popsize x n_params` loss evaluations
   per generation, so it is practical for a handful of parameters, not
   for tens.
2. **Local refinement (always)** -- `local_method` picks a bounded
   scipy optimiser (default `Nelder-Mead`, case-insensitive):

| `local_method` | uses a gradient | notes |
|---|---|---|
| `L-BFGS-B`    | finite-difference | former default; stalls in narrow valleys |
| `TNC`         | finite-difference | truncated Newton |
| `SLSQP`       | finite-difference | |
| `Nelder-Mead` | none | default; adaptive simplex; many evaluations |
| `COBYLA`      | none | linear trust-region model |
| `COBYQA`      | none | quadratic trust-region model |
| `least_squares` | finite-difference Jacobian of the residuals | scipy `least_squares` (`dogbox`) on the residual vector the loss sums; squared loss only |

All six respect the parameter bounds and receive the same stopping
intent in their own option names (at most 5000 iterations -- 100 000
for Nelder-Mead -- and 100 000 loss evaluations, loss tolerance 1e-12
eV^2, gradient and step tolerances 1e-8).  Nelder-Mead stops on the step
tolerance alone (a simplex ~1e-8 relative in every parameter): scipy's
loss tolerance is absolute, so a fit whose minimum is not zero never
met it and ran all 100 000 evaluations.  Any other name is refused
before any data is loaded: `BFGS` and `CG` ignore bounds, `Newton-CG`,
`trust-*` and `dogleg` need an analytic gradient, which `fit` does not
provide, and scipy's bounded `Powell` returned points worse than ones
it had itself evaluated on real fit losses.

`least_squares` (ADR-0015) minimises the vector of E / F / S residuals
whose squares the loss sums, with a Jacobian from forward differences and
the variables rescaled by it.  It is the method for ill-conditioned fits:
on the Ni-Al Zhou-EAM test (loss Hessian condition number ~1e11 in a
curved valley) it reached loss 7e-14 in about 3 minutes, where
Nelder-Mead with restarts stopped at 2.7e-4 after 8.8 hours; on the
multi-species Tersoff tests it reached ~1e-15.  The multi-species
`--write-config` templates use it.

Three settings apply to any method:

- `restarts: N` reruns the local search from the best point (Nelder-Mead
  from a randomly rotated simplex, `restart_seed`); it stops after two
  consecutive restarts that improve the loss by less than `restart_rtol`
  (default 1e-3, relative).
- `local_maxiter` caps each local run, so a slow run is restarted.
- `time_limit_hours` ends the search on wall-clock time (sleep included)
  and keeps the best point evaluated so far.

`Nelder-Mead` is the default because it was the only method that
reached the minimum (loss ~1e-14) on every real fit loss measured: SW Si
from the DE result and from a 10-25% perturbed start, and the 19-variable
GeSn SW fit.  `L-BFGS-B`, the former default, stopped at 6.6e-5, 2.3e-3
and 1.9e4 on those (it also stalled at 14 on an LJ test where the others
reached 1e-14): its finite-difference gradient and line search end early
in narrow curved valleys and at points the loss scores as failed.  The
price is evaluations: 800-1700 for 5 SW parameters, ~32 000 for 19.
`COBYQA` is the next choice for a handful of parameters.

#### Multi-species Tersoff and EAM (ADR-0015)

The pair potentials take one record per `A-B` pair and were always
multi-species.  Tersoff and EAM now are too:

```bash
akitsushima fit -p tersoff --write-config t.json --template tersoff1989  # Si-C
akitsushima fit -p tersoff --write-config t.json --template records      # AlN
akitsushima fit -p eam     --write-config e.json                         # Ni-Al
```

- **Tersoff, 1989 mixing** (`initial_tersoff: {mixing: tersoff1989,
  elements: {...}, chi: {...}}`): per-element A, B, lambda, mu, c, d, h
  and one chi per pair are fitted (Tersoff, Phys. Rev. B 39, 5566
  (1989), eqs. 1a-1e); `fit_keys: [chi]` is the paper's own procedure.
- **Tersoff, records** (a list of `i, j, k` records, the LAMMPS table):
  the pair terms of (i, j, j) and (j, i, i) are one variable; c, d, h per
  record, or per centre element with `tie: centre`.
- Train on more than one coordination: Tersoff's angular parameters c
  and d are only determined through c / d^2 when every angle is near
  tetrahedral, and an unseen rocksalt or sheet phase then fails; adding
  rocksalt cells to the training data fixed it (ADR-0015).  Likewise an
  EAM needs expanded and compressed cells: Zhou's low-density embedding
  branch (rho < 0.85 rho_e) is otherwise left undetermined.
- In both forms **beta and n are fixed by default**: with near-tetrahedral
  training angles they trade off almost exactly against c and d, and
  fitted they drifted by orders of magnitude without improving the loss
  (ADR-0015).  Add them to `fit_keys` to fit them anyway.
- **EAM** (`initial_eam: {form: zhou2004, elements: {...} | [Ni, Al]}`):
  A, B, alpha, beta and the embedding coefficients Fn2, Fn3, F2, F3, eta
  per element; Fn0, Fn1, F0, F_e follow from the continuity of F and
  dF/drho, which the paper requires.  Each fit also writes
  `fitted.eam.alloy` (setfl), usable as `potential: {type: EAM, setfl:
  fitted.eam.alloy}`.
- Published tables: Tersoff (1989) Table I and Zhou (2004) Table III in
  full (`akitsushima.engine.published_parameters`), with two misprints
  corrected (carbon d, Al A) and the rows whose embedding function is
  discontinuous as printed flagged (Pt, W; ADR-0015).

### Fitting a machine-learning potential

`akitsushima mlip mlip.yaml` (template: `akitsushima mlip --dump-default`)
trains on a table of E, F and S rows (`load_mlip_csv`):
`target, label, weight, <descriptors>`, where an E row holds the structure
descriptor d (target E/N), an F row X = -dd/dr and an S row
X = -(1/V) dd/dT.  Per `docs/adr/0004`:

- **ELM / QELM / NN** learn E/N = f(d) and fit every F and S row as
  N grad f(d) . X of its own structure (chain rule), so one energy function
  gives E, F and S.  `row_independent: true` restores the pre-0.2.0a3
  per-row regression, which is not a force field.
- **linear** is E/N = w . d + b; with d the per-atom average of atomic
  descriptors it is an atom-centred linear potential.
- Every model weights the squared residual by w_cat x w_unit^2 x w_user.

**Direct linear term (`direct_link`, on by default for ELM / QELM / NN).**
Besides the hidden layer, the input descriptor enters the output linearly:

    ELM / QELM :  f(x) = beta . g(W x + b) + gamma . x + beta_0
    NN         :  f(x) = MLP(x) + w . x

with x the standardised descriptor (d for the tabular form, the atom's
G_i for the atom-centred one, one set of weights per species).  For the
ELM this is the random-vector functional-link (RVFL) network: W and b are
random and fixed, and beta, gamma and beta_0 are solved together in one
least-squares system.  The linear part the data contain is then fitted
exactly, and the hidden layer only has to learn the non-linear remainder.
Outside the training data the model falls back towards that linear
model instead of towards the saturated hidden units.  That is gentle for
small excursions, but the linear term is unbounded: for close contacts
(a bond pushed to 1.2 A) it runs away by hundreds of eV, more so with
the PCA whitening (ADR-0014).  Regularisation differs by model:

- ELM / QELM: `reg` (and `reg_search`) acts on the hidden-layer weights
  beta only; gamma and beta_0 get only a relative jitter (1e-12 of the
  mean diagonal) that keeps the normal equations definite.  A large reg
  therefore shrinks the model towards the linear model it contains, not
  towards zero.
- NN: the linear weights w start at zero and are trained with the rest;
  Adam's `weight_decay` applies to all parameters alike.

`direct_link: false` gives the plain ELM (reg on every output weight) and
the plain MLP.  With the direct term the ELM has D + 1 more fitted
weights per species (D descriptor components), the NN D more.

**ELM / QELM input standardisation and options (`docs/adr/0013`).**
By default (since ADR-0013) the descriptor is z-scored and then whitened
with the training PCA (`standardize: pca`; all directions kept,
`pca_eps` bounds the gain; the map is stored in the model and applied to
F / S rows without the mean) -- in the ADR-0013 comparison this lowered
the force error in 22 of 24 cases (by 7 % on average, up to 30 %).
`standardize: zscore` gives the former per-feature standardisation;
models saved before ADR-0013 load as z-score models. Since 0.2.0a12 the NN
(tabular and atom-centred) takes the same keys, with `zscore` as its
default: `standardize: pca` feeds the network the whitened descriptor,
and `pca_drop` narrows its input.  `min_feature_std` (all models, default
1e-12) masks input columns that hardly vary in the training data.  Options without a
measurable gain there: `w_init_scale_search` (factors on the Glorot
scale of W, b, picked on the validation split), `bias_init: data` (each
unit centred on a training input), `w_correlation: radial` with
`w_corr_length(_search)` (G2 weights correlated along r_s; in-package
ACSF only; implies z-score).  QELM:
`levels: pow2 | ternary | uniform`, `max_exponent` (formerly `n_bits`),
`n_levels`, `p_zero`, `matmul: dense | sparse`, `w_memory: int8 | float64`
(default int8: W held and saved as integer levels plus column scales,
files and memory 1.7-5.2x smaller than the ELM's); accuracy matches the ELM,
W storage shrinks, the sparse product is not faster in numpy.

**Physics-informed parts (`docs/adr/0014`).**
`mlip.baseline: {type: zbl, r_inner, r_outer}` (or `born_mayer` with
per-pair `A`, `rho`; a list of terms) is a fixed short-range repulsion,
switched off smoothly between r_inner and r_outer.  It is subtracted
from the labels and added back by the engine (Delta-learning).  Choose
r_outer below the shortest distance in the data; it is printed.
`mlip.isolated_atom_energy: {Si: 0.0}` makes every atom-centred energy
equal the isolated-atom energy when an atom has no neighbour within the
cutoff: exactly for ELM / QELM (a constrained solve; tabular form for
one species) and the atom-centred NN.  In ADR-0014's tests the isolated-
atom limit was exact at little cost.
`mlip.descriptor.inner_cutoff: [r_off, r_on]` switches pairs closer
than r_off out of the ACSF, so the model cannot run away at close
contacts (the ELM / QELM direct link otherwise does, see above) and the
baseline alone acts there.  Put r_on at or below the data's shortest
distance and let the baseline's r_outer reach r_on.  For example, for Si
with data from 2.0 A: `inner_cutoff: [1.8, 2.0]`, `baseline: {type: zbl,
r_inner: 1.2, r_outer: 2.0}`, `isolated_atom_energy: {Si: 0.0}`.  All
models then gave a repulsive bond-compression curve with unchanged
accuracy inside the data.
ZBL is universal, not fitted, and was 20-50 % too soft at 1.0-1.2 A.
With a few dozen short-range reference points (one atom pushed towards
a neighbour, from about 1.1 A to the data's edge, in the same `fit`
layout), a Born-Mayer term can be fitted instead:
`baseline: [{type: born_mayer, r_inner: 1.8, r_outer: 2.0, fit_to:
sr/structures.csv}]`.  After training, A and rho (per species pair) are
fitted to the residual E_ref - E_model and stored in the model.  If the
term reaches the training data, the model is refitted once.  In
ADR-0014 this matched the reference within 2 % at 1.0-1.5 A (SW Si)
and 1.2 A (Tersoff).

With `mlip.descriptor` set (ACSF), `training_csv` is a structure list in the
`fit` layout above and the descriptors are computed in-package and stored
in the model file.  Such a model then drives `relax` / `md` / ... with

```yaml
potential:
  type:  MLIP
  model: ./mlip_output/elm_model.npz
```

`mlip.form: atom_centred` (ADR-0005) trains the conventional atom-centred
form instead, E = sum_i eps_s(G_i) on each atom's own ACSF vector G_i
(models `elm`, `qelm`, `nn`), from the same structure list.  The energy is
a sum of local terms, so a model trained on one cell size runs on others;
the model file (`atomic_<model>_model.npz` / `.pt`) records the form and
is used with the same `potential: {type: MLIP, model: ...}` entry.

### Active learning (ADR-0037)

`akitsushima active active.yaml` (`ak.active(...)` in Python) runs rounds
of explore -> select -> label -> train with an atom-centred ELM / QELM
ensemble (ADR-0033); `akitsushima --dump-default active` writes a
commented input.

- **explore**: md and / or anneal sections with start structures, run
  with the current model in cells DFT can label; a trajectory stops when
  sigma_F,max stays above hi (`stop_sigma`, also usable in `md` and
  `anneal` on their own).
- **select**: every saved frame is re-evaluated.  sigma_F,max < lo:
  accurate; above hi: failed; in between: candidate, thinned to `k` by
  farthest-point sampling.  lo and hi default to the 99th percentile of
  sigma_F,max on the training structures and 10 times it.  Frames with
  atoms closer than 0.75 times the training set's shortest distance are
  never labelled.
- **label**: `mode: vasp_inputs` writes POSCAR / INCAR / KPOINTS /
  `make_potcar.sh` per structure and exits with code 2; run VASP and
  rerun the command to read `vasprun.xml`.  `mode: external` runs an
  ADR-0011 external potential for each structure.
- **train**: the ensemble is refitted on the start data plus every label
  so far; added structures stay out of the validation split.

Everything is recorded in `active_ledger.jsonl`; rerunning resumes.  On a
synthetic GeSn oracle (SW as "DFT", six start sets) the selection gave a
lower force RMSE than random picks from the same explorers on every start
set (0.02-0.22 eV/A), not a lower energy RMSE; see ADR-0037, Result.

### External codes (ADR-0011)

Any code that can be run from a shell drives every E/F/S solver through

```yaml
potential:
  type:     external
  cmd:      "mpirun -np 8 vasp_std > vasp.out && python -m akitsushima.tools.ase_efs --read vasprun.xml"
  work_dir: ./external_run
```

Each call writes `work_dir/POSCAR`, runs `cmd` in `work_dir` and reads
`work_dir/result.efs` (akitsushima EFS format: energy per atom, forces,
stress in eV/A^3, positive under compression).  Wrappers:
`python -m akitsushima.tools.ase_efs --calc emt|lj` (an ASE calculator),
`--read FILE` (any ASE-readable output with energy, forces and stress --
unverified on real VASP / QE files), and
`python -m akitsushima.tools.akitsushima_efs pot.yaml` (akitsushima itself,
for testing).  `relax`, `md`, `neb`, `tsa`, `phonon` and elastic constants
take it; `gcmc` / `osmc` warn (one external run per trial move); `kappa`
and `coulomb_method` / `charges` / `esm` are refused.

Which solvers take an MLIP (ADR-0007):

| command | tabular | atom-centred | note |
|---|---|---|---|
| `relax`, `md`, `neb`, `tsa`, `phonon`, `elastic`, `gruneisen`, `qha` | yes | yes | need E, F, S only |
| `dielectric`, `piezo` | with `born:` | with `born:` | Z* and eps_inf must be given (no charges of its own) |
| `raman` | yes | yes | frequencies from the MLIP; intensities from `polarizability:` |
| `mdspec` | -- | -- | post-processing of a trajectory (charges from `mdspec.dipole` / `born:`) |
| `gcmc` | refused | yes | the atom count changes: needs a sum of site energies; inserted elements must be among the model's |
| `kappa`, `nemd` (local flux) | refused | yes | exact heat flux from the site terms |
| `osmc` | refused | yes | CFCMC couples the fractional atom through total energies (ADR-0009): as `gcmc` |
| `esm` | refused | refused | electrostatics only |

A refusal is an error at start-up, not a silent approximation.  The heat
flux of `kappa` is exact for every potential with a site-energy
decomposition -- pair potentials, Stillinger-Weber, Tersoff, EAM and an
atom-centred MLIP (`akitsushima/engine/site_terms.py`); only a Coulomb
term still uses the centre-of-mass approximation, with a warning.

## Project layout

```
akitsushima/
├── constants.py        # physical constants
├── utils/              # timing, MPI helpers, numba JIT setup
├── io/                 # POSCAR, extxyz, LAMMPS, CIF, XDATCAR, EFS; ASE adapter
├── engine/              # energy / force / stress: pair, many-body, and Coulomb potentials
├── fitting/             # parameter sets, loss function, fitting driver, ML models
├── solvers/             # geometry optimisation, MD, GCMC, NEB, TSA, phonons, kappa
└── cli/                 # argument parsing and one module per sub-command
tests/
```

Dependencies form a one-directional DAG: `cli → solvers/fitting → engine →
utils/io`.

## Testing

```bash
pip install -e ".[all,dev]"
pytest tests/
```

The full suite takes about 15 min.  For a quick check while editing:

```bash
pytest tests/ -m "not slow"
```

`slow` marks every test that takes 0.5 s or more with or without
`NUMBA_DISABLE_JIT=1`, or that shares a costly module fixture
(docs/adr/0028-feature-and-speed-roadmap.md, addendum 4).  The rest, 481
tests, run in about 30 s, also with the JIT disabled.

`tests/test_mpi.py` spawns real `mpirun -np 1/2/4` subprocesses to check
that results agree across rank counts; it is skipped automatically if
`mpirun` or `mpi4py` is not available.

torch ships its own OpenMP runtime (`libomp`), and so does numba's
default `omp` threading layer. When torch's copy is initialised first,
the first numba `parallel=True` kernel aborts the process with
`OMP: Error #15` and then `Fatal Python error: Aborted`. This was
observed on macOS, also under miniconda with pip's torch wheel. The
reverse order works. Since 0.2.0a11, akitsushima starts numba's
threading layer when it is imported, and every torch import inside the
package happens after that. If torch was imported before akitsushima,
the `workqueue` layer is chosen instead, which loads no OpenMP. So the
order of imports, and of test files, no longer matters
(tests/test_openmp_order.py). An explicit `NUMBA_THREADING_LAYER` is
respected. Since 0.2.0a19 this also holds with `NUMBA_DISABLE_JIT=1`,
where the warm-up kernel runs as plain Python and used to start nothing.

## License

MIT — see [LICENSE](LICENSE).
