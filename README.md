# akitsushima

> **Beta release.** This project is feature-complete for its intended
> scope but still being validated; bugs may still be found and fixed
> before a stable release. Use in production with caution, and pin the
> exact version you tested against.

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
- Tersoff — bond-order (`b_ij`) with coordination (Si, C, Ge, GaAs)
- EAM — embedding `F(rho_i)` + pair `phi(r)`, tabulated (metals)

**Long-range Coulomb**
- Ewald summation — vectorised, `O(N^1.5)` for N ≲ 2,000
- PPPM / SPME — FFT mesh, `O(N log N)` for N ≳ 500
- ESM — Effective Screening Medium (open / metallic boundary conditions)

**Machine-learning potentials**
- `ELMModel` — Extreme Learning Machine (single hidden layer)
- `QuantizedELMModel` — ELM with quantised hidden-layer weights
- `NNModel` — PyTorch feed-forward network (lazily imported)
- `LightGBMModel` — Gradient-boosted trees + Optuna search (lazily imported)

### Simulation capabilities

| Task | Description |
|---|---|
| `fit`   | Fit classical or ML potential parameters against DFT reference data (POSCAR geometry + CSV energy/force/stress). |
| `relax` | Geometry optimisation: FIRE, L-BFGS, DIIS, RMM-DIIS, RFO — ionic or variable-cell, with transition-state refinement. |
| `md`    | Molecular dynamics: NVE, NVT (Nosé-Hoover chain), NPT. |
| `gcmc` / `osmc` | Grand-canonical (μVT) MD, including soft-core lambda-dynamics insertion/deletion. |

### Post-processing

- **CI-NEB** — Climbing-image nudged elastic band
- **`TransitionStateAnalyser`** — Hessian, IRC, saddle-point verification, TST reaction rate
- **`PhononCalculator`** — Finite-displacement force constants, band structure, DOS, Debye temperature
- **`ThermalConductivityCalculator`** — Green-Kubo MD + Slack-Debye estimate

### Performance

- **`NeighbourList`** — cell-linked list / SciPy `cKDTree` back-end, with a Verlet skin for lazy rebuilds during MD.
- **Numba JIT** — optional `@njit` compilation of the pair / many-body loops (roughly 10–100x speed-up), with a transparent NumPy fallback when numba isn't installed.
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
pip install -e ".[lightgbm]"    # LightGBMModel + Optuna hyperparameter search
pip install -e ".[yaml]"        # YAML config files for the CLI
pip install -e ".[plotting]"    # matplotlib output (NEB/phonon/relax plots)
pip install -e ".[all]"         # everything above
```

`NNModel` and `LightGBMModel` are imported lazily — `import akitsushima`,
and every other class in the package, never pulls in `torch` or `lightgbm`
unless you actually reference one of those two classes.

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

lattice = np.eye(3) * 15.0
positions = np.array([[5.0, 5.0, 5.0], [7.5, 5.0, 5.0]])
atom_types = ["Ar", "Ar"]
lj_params = {frozenset({"Ar"}): {"epsilon": 0.0103, "sigma": 3.405}}

engine = PotentialEngine().add(LennardJonesPotential(lj_params, r_cut=5.0))
fn = make_force_fn(engine)

nl = NeighbourList(r_cut=5.0)
nl.build(lattice, positions)

energy, forces, stress = fn(lattice, positions, atom_types)
```

## CLI reference

| Command | Purpose |
|---|---|
| `relax`  | Geometry optimisation (FIRE / L-BFGS / DIIS / RMM-DIIS / RFO) |
| `md`     | Molecular dynamics (NVT / NVE / NPT) |
| `gcmc`   | Grand-canonical Monte Carlo MD (μVT) |
| `neb`    | Climbing-image NEB transition-state search |
| `tsa`    | Transition-state analysis (Hessian / IRC / TST rate) |
| `osmc`   | OS-MC soft-core lambda-integration GCMC |
| `esm`    | Effective screening medium |
| `phonon` | Phonon band structure, DOS, Debye temperature, free energy |
| `kappa`  | Thermal conductivity (Green-Kubo + optional Slack-Debye) |
| `mlip`   | Fit a machine-learning potential (ELM / QELM / NN / LightGBM) |
| `fit`    | Fit a classical potential (LJ / Morse / Born-Mayer / Buckingham / Stillinger-Weber / Tersoff) |

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

## Project layout

```
akitsushima/
├── constants.py        # physical constants
├── utils/              # timing, MPI helpers, numba JIT setup
├── io/                 # POSCAR / XDATCAR / EFS file I/O
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

`tests/test_mpi.py` spawns real `mpirun -np 1/2/4` subprocesses to check
that results agree across rank counts; it is skipped automatically if
`mpirun` or `mpi4py` is not available.

On macOS, a pure-pip environment with both `[numba]` and `[torch]`
installed can abort at import time (`Fatal Python error: Aborted`):
numba and pip's torch wheel each bundle their own OpenMP runtime, and
loading both into one process is not supported. This is an ecosystem
packaging conflict, not an akitsushima bug -- it does not occur under
conda, which coordinates a single shared `libomp`. Work around it with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).
