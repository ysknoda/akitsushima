"""
akitsushima
===========
Classical force-field and machine-learning potential toolkit for
atomistic simulation.  Fits, relaxes, runs molecular dynamics, and
computes phonon / thermal-transport properties for crystalline and
molecular systems.

Force fields
------------
  Pair potentials  (ionic / van der Waals)
    Buckingham      -- A exp(-r/rho) - C/r^6
    Born-Mayer      -- A exp(-r/rho)
    Lennard-Jones   -- 4epsilon [(sigma/r)^12 - (sigma/r)^6]
    Morse           -- D_e [1 - exp(-alpha(r-r0))]^2

  Many-body potentials  (covalent / metallic)
    Stillinger-Weber -- 2-body + 3-body angle term              (Si, Ge, H2O)
    Tersoff          -- bond-order (b_ij) with coordination     (Si, C, Ge, GaAs)
    EAM              -- embedding F(rho_i) + pair phi(r), tabulated  (metals)

  Long-range Coulomb
    Ewald summation  -- vectorised, O(N^(3/2))  for N <~ 2 000
    PPPM / SPME      -- FFT mesh,   O(N log N)  for N >~ 500
    ESM              -- Effective Screening Medium (open/metallic boundaries)

  Machine-learning potentials
    ELMModel         -- Extreme Learning Machine (single hidden layer)
    QuantizedELMModel -- ELM with quantised hidden-layer weights
    LinearModel      -- linear E/F/S model (OLS, ridge, LASSO, PLS, Bayesian, ...)
    NNModel          -- PyTorch feedforward network (lazy import)
    AtomicELMModel, AtomicQELMModel, AtomicNNModel
                     -- atom-centred (Behler-Parrinello) models (ADR-0005)

Simulation capabilities
-----------------------
  fit    -- Fit classical or ML potential parameters to DFT reference
           data (VASP POSCAR + CSV format; energies, forces, stresses).
  relax  -- Geometry optimisation: FIRE, L-BFGS, DIIS, RMM-DIIS, RFO
           (ionic or variable-cell; RFO supports TS refinement).
  md     -- Molecular dynamics: NVE, NVT (Nose-Hoover chain), NPT.
  gcmc/osmc -- Grand-canonical (muVT) MD: discrete moves / CFCMC.

Post-processing
---------------
  CI-NEB                    -- Climbing-image nudged elastic band
  TransitionStateAnalyser   -- Hessian, IRC, saddle-point verification
  PhononCalculator          -- Finite-displacement IFC, dispersion, DOS
  ThermalConductivityCalculator -- Green-Kubo MD + Slack-Debye estimate

Performance features
--------------------
  NeighbourList  -- Cell-linked list / scipy cKDTree back-end;
                   Verlet skin for lazy rebuild during MD.
  Numba JIT      -- Optional @njit compilation of the pair/many-body loops
                   (~10--100x speed-up); transparent fall-back without numba.
  MPI (mpi4py)   -- Domain-split + Allreduce parallelism across ranks for
                   every potential and for the fitting loss function;
                   transparent fall-back to single-process operation.

Quick start
-----------
  # Command-line (see `akitsushima --help` for every sub-command)
  akitsushima --dump-default relax  > relax.yaml
  akitsushima relax relax.yaml

  # API example: LJ engine + neighbour list.  Self-contained: paste and run.
  import numpy as np
  from akitsushima import (PotentialEngine, LennardJonesPotential,
                            NeighbourList, make_force_fn)

  # Standard Lennard-Jones argon: sigma = 3.405 A, epsilon = 0.0103 eV
  # (= 119.8 K * k_B).  Verlet, Phys. Rev. 159, 98 (1967).
  lj_params = {frozenset({"Ar"}): {"epsilon": 0.0103, "sigma": 3.405}}

  # r_cut = 10.0 A = 2.94 sigma, inside the conventional 2.5-3.5 sigma
  # range for LJ argon.  NeighbourList.build() enforces the minimum-image
  # convention, so the cell must then satisfy
  #     cell edge > 2 * (r_cut + skin_width) = 2 * (10.0 + 0.5) = 21.0 A.
  # 24.0 A clears that with margin; a smaller box raises ValueError.
  lattice = np.eye(3) * 24.0

  # Two argon atoms 4.0 A apart, just beyond the LJ minimum
  # r_min = 2^(1/6) sigma = 3.82 A, so the pair attracts weakly.
  positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
  atom_types = ["Ar", "Ar"]

  engine = PotentialEngine().add(LennardJonesPotential(lj_params, r_cut=10.0))
  fn = make_force_fn(engine)
  nl = NeighbourList(r_cut=10.0)
  nl.build(lattice, positions)
  energy, forces, stress = fn(lattice, positions, atom_types, nl)
  # energy = -9.4170e-03 eV  (force-shifted LJ, the LennardJonesPotential
  # default shift=True; the unshifted 12-6 value would be -9.7116e-03 eV)
"""
from .constants import ATOMIC_MASS, EV_ANG_TO_AMU_ANG_FS2, K_B, K_E
from .engine import (
    BasePairPotential,
    BornMayerPotential,
    BuckinghamPotential,
    EAMPotential,
    ESMCalculator,
    LennardJonesPotential,
    MiePotential,
    MorsePotential,
    NeighbourList,
    PotentialEngine,
    PPPMSolver,
    StillingerWeberPotential,
    TersoffPotential,
    YukawaPotential,
    compute_all_general,
    ewald_sum,
)
from .fitting import (
    AtomicELMModel,
    AtomicQELMModel,
    BornMayerParameterSet,
    BuckinghamParameterSet,
    ELMModel,
    ExternalFileEngine,
    FittableParameterSet,
    LinearModel,
    LJParameterSet,
    MieParameterSet,
    MorseParameterSet,
    QuantizedELMModel,
    ScaledForceEngine,
    StillingerWeberParameterSet,
    TersoffParameterSet,
    YukawaParameterSet,
    compute_all,
    fit_potential,
    load_potential,
    make_force_fn,
    save_potential,
)
from .io import parse_poscar, write_poscar
from .solvers import (
    CFCMCSimulation,
    DiffusionAnalyzer,
    ElasticConstantsCalculator,
    GCMCMDSimulation,
    GeometryOptimiser,
    NEBChain,
    NEBOptimiser,
    NPTSimulation,
    NVESimulation,
    NVTSimulation,
    PhononCalculator,
    RadialDistributionFunction,
    ThermalConductivityCalculator,
    TransitionStateAnalyser,
    elastic_moduli,
    fit_equation_of_state,
)

__version__ = "0.2.0a9"

__all__ = [
    "__version__",
    # constants
    "ATOMIC_MASS", "K_B", "K_E", "EV_ANG_TO_AMU_ANG_FS2",
    # engine
    "PotentialEngine", "NeighbourList", "BasePairPotential",
    "LennardJonesPotential", "MorsePotential", "BornMayerPotential",
    "BuckinghamPotential", "MiePotential", "YukawaPotential",
    "StillingerWeberPotential", "TersoffPotential",
    "EAMPotential", "ewald_sum", "PPPMSolver", "ESMCalculator",
    "compute_all_general",
    # fitting
    "make_force_fn", "compute_all", "fit_potential",
    "save_potential", "load_potential",
    "FittableParameterSet", "BuckinghamParameterSet", "LJParameterSet",
    "MorseParameterSet", "BornMayerParameterSet",
    "MieParameterSet", "YukawaParameterSet",
    "StillingerWeberParameterSet", "TersoffParameterSet",
    "ExternalFileEngine", "ScaledForceEngine",
    "ELMModel", "QuantizedELMModel", "LinearModel", "NNModel",
    "AtomicELMModel", "AtomicQELMModel", "AtomicNNModel",
    # solvers
    "GeometryOptimiser", "NVTSimulation", "NVESimulation", "NPTSimulation",
    "GCMCMDSimulation", "CFCMCSimulation",
    "NEBChain", "NEBOptimiser", "TransitionStateAnalyser",
    "PhononCalculator", "ThermalConductivityCalculator",
    "RadialDistributionFunction",
    "ElasticConstantsCalculator", "elastic_moduli", "fit_equation_of_state",
    "DiffusionAnalyzer",
    # io
    "parse_poscar", "write_poscar",
]

# NNModel / AtomicNNModel are resolved lazily, all the way up from
# fitting/models/__init__.py through fitting/__init__.py to here -- see
# fitting/models/__init__.py for why (avoids an eager torch import -- and
# the OpenMP runtime conflict with numba that motivated it, see engine's
# own F15 bug-fix note -- for every caller who never touches them,
# including everyone who just does `import akitsushima` for the
# classical/engine functionality above).
_LAZY = {"NNModel": "fitting", "AtomicNNModel": "fitting"}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import fitting as _fitting
    value = getattr(_fitting, name)
    globals()[name] = value
    return value
