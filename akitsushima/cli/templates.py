"""Default YAML/JSON configuration templates, printed by
``akitsushima --dump-default <task>`` and used as defaults when no
``--config``/``INPUT.yaml`` is given.
"""
from ..fitting import registry

__all__ = [
    "_DEFAULT_FIT_CFG",
    "_DEFAULT_FIT_CFG_LJ",
    "_DEFAULT_FIT_CFG_MORSE",
    "_DEFAULT_FIT_CFG_BORNMAYER",
    "_DEFAULT_FIT_CFG_SW",
    "_DEFAULT_FIT_CFG_TERSOFF",
    "_FIT_CFG_TEMPLATES",
    "_YAML_RELAX_DEFAULT",
    "_YAML_MD_NVT_DEFAULT",
    "_YAML_MD_NVE_DEFAULT",
    "_YAML_MD_NPT_DEFAULT",
    "_YAML_GCMC_DEFAULT",
    "_YAML_NEB_DEFAULT",
    "_YAML_PHONON_DEFAULT",
    "_YAML_KAPPA_DEFAULT",
    "_YAML_ELM_DEFAULT",
    "_YAML_TSA_DEFAULT",
    "_YAML_OSMC_DEFAULT",
    "_YAML_ESM_DEFAULT",
    "_DUMP_MAP",
]

# ============================================================================
#  `fit` JSON/YAML config templates (one per potential type)
# ============================================================================

# Consolidation note (Phase 1, potential registry): every `fit`-related
# default config used to be defined directly in this module, in 6 nearly
# duplicate dicts (one per potential type) plus a _FIT_CFG_TEMPLATES
# dict collecting them -- one of 4 locations that had to be kept
# manually in sync whenever a potential was added (see
# fitting/registry.py's module docstring for the other 3). Each
# per-potential dict has moved to fitting/registry.py, next to that
# potential's build_param_set/build_from_yaml functions; the names below
# are kept as backward-compatible aliases derived from the registry, so
# any existing code importing them from here (including this project's
# own tests) keeps working unchanged.
_DEFAULT_FIT_CFG            = registry.get("buckingham").default_cfg
_DEFAULT_FIT_CFG_LJ         = registry.get("lj").default_cfg
_DEFAULT_FIT_CFG_MORSE      = registry.get("morse").default_cfg
_DEFAULT_FIT_CFG_BORNMAYER  = registry.get("bornmayer").default_cfg
_DEFAULT_FIT_CFG_SW         = registry.get("sw").default_cfg
_DEFAULT_FIT_CFG_TERSOFF    = registry.get("tersoff").default_cfg

_FIT_CFG_TEMPLATES = {name: registry.get(name).default_cfg
                     for name in registry.names()}


# ============================================================================
#  Per-task YAML templates (printed by --dump-default)
# ============================================================================

_YAML_RELAX_DEFAULT = """\
# akitsushima  --  geometry relaxation input
# Usage: akitsushima relax relax.yaml

structure:
  poscar: POSCAR                   # path to input POSCAR

potential:
  type: SW                         # SW | Tersoff | LJ | Morse | BornMayer | EAM | Buckingham
  species: [Si]
  params:                          # Stillinger-Weber Si (Stillinger & Weber 1985)
    epsilon: 2.1682
    sigma:   2.0951
    A:       7.049556
    B:       0.6022246
    p:       4
    q:       0
    a:       1.80
    lambda:  21.0
    gamma:   1.20
    costheta0: -0.333333

relax:
  output_dir:    ./relax_output
  method:        fire               # fire | lbfgs | sd | diis | rmmdiis | rfo
  relax_cell:    false

  # -- convergence criteria (each independently on/off) ----------------
  # ALL enabled criteria must be satisfied simultaneously to converge.
  check_forces:  true                # require max atomic force < fmax
  check_stress:  null                # null = auto (follows relax_cell);
                                     # true/false to override explicitly
  check_energy:  false               # require |E_new - E_prev| < emax

  fmax:          0.01                # eV/A   force convergence threshold
  pmax:          0.1                 # GPa    stress convergence threshold
  emax:          1.0e-6              # eV     energy convergence threshold

  max_steps:     2000
  skin_width:    0.5

  # Step control and method-specific knobs.  These reach
  # GeometryOptimiser but were absent from this template, so the RFO
  # mode it offers above had none of its own settings reachable.
  max_displacement: 0.2              # A   per-component cap on one step
  # memory:      20                  # (s,y) pairs for lbfgs/diis/rmmdiis.
                                     # Left unset, the default is chosen
                                     # per method: 3 for rmmdiis, 20 for
                                     # the other two.  rmmdiis projects
                                     # its residual out of the saved
                                     # subspace, so the step length
                                     # collapses as the subspace fills --
                                     # measured 5-7x more steps at 20
                                     # than at 3 -- while diis is
                                     # slightly better at 20 and lbfgs is
                                     # insensitive.
  uphill:        false               # true: Partitioned RFO, i.e. climb to
                                     # a saddle point instead of descending.
                                     # method: rfo only.
  trust_radius:  0.3                 # A   initial RFO trust region
  hessian_delta: 0.01                # A   RFO finite-difference Hessian step.
                                     # That Hessian costs 2*(3N+9) force
                                     # evaluations before step 0.
  # fix_cell:    [1, 1, 1, 0, 0, 0]  # Voigt mask, 1 = frozen
  flush_every_step: true             # flush relax.log/XDATCAR each step
  fsync_every_step: false            # fsync too (slow; for crash safety)
  timing:        true

  save_best:     false               # true: save lowest-energy structure seen
                                     # during the run as POSCAR_best, and
                                     # return it as the final result.
                                     # Useful when FIRE/LBFGS overshoots the
                                     # minimum due to inertia or poor curvature.
  # FIRE timestep control (only used when method: fire)
  # dt_start: initial fictitious timestep [fs].
  #   Default: 0.1 fs.  For large initial forces or stresses (Pmax > 10 GPa),
  #   a smaller value (e.g. 0.01 fs) reduces the risk of overshooting small
  #   energy barriers on the first few steps.  Note: FIRE resets dt to
  #   dt_start * F_DEC (= 0.05 fs by default) after the very first step
  #   because the initial velocity is zero (P = v.g = 0 <= 0 triggers a
  #   reset), so dt_start is effectively used for step 0 only.
  # dt_max:  upper bound on the adaptive timestep [fs].
  #   Default: 1.0 fs.  Reduce to limit maximum step size in later stages.
  dt_start:      null                # null = use default (0.1 fs)
  dt_max:        null                # null = use default (1.0 fs)

  # -- Cell update scheme (relax_cell: true only) -----------------------
  # 'generalized_force' (default): physically-derived Parrinello-Rahman-
  #   style force f_cell = +volume * (h^-T @ stress) (exact dE/dh with
  #   correct sign; '@' is a matrix product and h^-T multiplies stress
  #   from the LEFT). Can amplify small stress asymmetries into large
  #   cell-shape changes for strongly non-orthogonal (hexagonal,
  #   monoclinic, ...) cells, since h^-T depends sensitively on the
  #   current cell shape.
  # 'vasp': direct update f_cell = stress @ lattice (no lattice-inverse
  #   factor), mirroring VASP's own cell-relaxation formula (dyna.F).
  #   Not a strict energy gradient, but an empirically well-behaved,
  #   numerically gentler update rule -- worth trying if
  #   'generalized_force' behaves erratically for a non-orthogonal cell.
  cell_update_scheme: generalized_force   # generalized_force | vasp

  # -- Lattice component constraints (relax_cell: true only) -------------
  # Mask over all 9 raw lattice-matrix components, in row-major order
  # matching the lattice matrix (rows = lattice vectors):
  #   [a1x, a1y, a1z, a2x, a2y, a2z, a3x, a3y, a3z]
  # Each is 1 (free to relax) or 0 (frozen). null = all 9 free (default,
  # fully unconstrained cell).
  #
  # NOTE: this mask acts on the RAW (unsymmetrized) components of the
  # cell force, so it does not by itself guarantee suppression of
  # spurious cell-rotation drift (an unsymmetrized cell force can, over
  # long relaxations, slowly rotate the whole cell relative to the lab
  # frame even though the physical shape/volume converges normally). If
  # you do not need independent raw-component control, leave this at
  # its default (null, fully free).
  #
  # Examples:
  #   "1 0 0 0 1 0 0 0 1"  -- diagonal components only (a1x, a2y, a3z
  #                           free; all off-diagonal components frozen).
  #   "1 1 1 0 0 0 0 0 0"  -- only lattice vector a1 is free to move (in
  #                           any direction); a2 and a3 fully frozen.
  #   "1 1 1 1 1 1 1 1 1"  -- fully unconstrained (equivalent to null).
  lattice_constraints: null          # null = "1 1 1 1 1 1 1 1 1" (all free)

  verbose:       true
"""

_YAML_MD_NVT_DEFAULT = """\
# akitsushima  --  NVT molecular dynamics input
# Usage: akitsushima md md_nvt.yaml

structure:
  poscar: POSCAR

potential:
  type: Tersoff                    # see the timestep note in the md section below
  species: [Si]
  params:                          # Tersoff T3 Si (Tersoff 1988)
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 1.7322
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.218
    h:       -0.59826
    R:       2.85
    D:       0.15

md:
  ensemble:     nvt
  output_dir:   ./nvt_output
  temperature:  300.0              # K
  timestep:     1.0                # fs
  # This template used to claim "SW requires dt<=0.05 fs; Tersoff allows
  # dt=0.1-0.5 fs".  Measured instead -- 64-atom silicon, NVE, 5 ps per
  # row, secular drift from a linear fit rather than an endpoint value,
  # in meV/atom/ps:
  #
  #   dt [fs]     0.05    0.1    0.25     0.5     1.0     2.0     4.0
  #   SW          0.000  0.000  0.00002 0.00009 0.00029 0.00136 0.00146
  #   Tersoff     0.000  0.00001 0.00006 0.00017 0.00033 0.00211 0.00653
  #
  # Both scale as dt^2, as a correct velocity Verlet must, and <T> holds
  # to 0.03% from 0.05 fs all the way to 2.0 fs.  For scale, (3/2)kT at
  # 300 K is 38.8 meV/atom, so at dt = 1.0 fs a 100 ps run drifts by
  # 0.03 meV/atom -- around 0.08% of the thermal energy.  Neither
  # potential needs anything like 0.05 fs; silicon has no light atoms
  # and its fastest optical mode has a period near 64 fs.
  #
  # 1.0 fs is the default on that evidence.  n_steps below is unchanged,
  # so the same wall-clock cost now buys 10 ps of simulated time instead
  # of 1 ps -- at a drift four orders of magnitude below the thermal
  # energy.  It also matches the default the NVT/NVE/NPT classes
  # themselves have always had, which this template was silently
  # overriding with a tenth of it.
  #
  # Check the H_NHC (or E_tot) drift column in thermo.log for YOUR
  # system before trusting it: the numbers above are for silicon, which
  # has no light atoms.  A system containing hydrogen is the real
  # exception -- an X-H stretch has a period near 10 fs -- and the
  # solvers warn when the timestep is large for the lightest mass
  # present.
  n_steps:      10000
  equil_steps:  1000               # steps with velocity rescaling active
  tau_nhc:      20.0               # fs  NHC thermostat time constant
  velocity_rescale_steps: 1000     # = equil_steps: VS active during equilibration
  traj_every:   10                 # write trajectory every N steps
  thermo_every: 50                 # write thermo log every N steps
  skin_width:   0.5
  seed:         42
  timing:       true
  verbose:      true
"""

_YAML_MD_NVE_DEFAULT = """\
# akitsushima  --  NVE molecular dynamics input
# Usage: akitsushima md md_nve.yaml

structure:
  poscar: POSCAR

potential:
  type: Tersoff
  species: [Si]
  params:
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 1.7322
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.218
    h:       -0.59826
    R:       2.85
    D:       0.15

md:
  ensemble:     nve
  output_dir:   ./nve_output
  temperature:  300.0
  timestep:     1.0                # fs -- see the NVT template's
                                   # timestep note for the measured
                                   # drift-vs-dt table
  n_steps:      10000
  equil_steps:  0
  traj_every:   10
  thermo_every: 50
  skin_width:   0.5
  seed:         42
  timing:       true
  verbose:      true
"""

_YAML_MD_NPT_DEFAULT = """\
# akitsushima  --  NPT molecular dynamics input
# Usage: akitsushima md md_npt.yaml

structure:
  poscar: POSCAR

potential:
  type: Tersoff
  species: [Si]
  params:
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 1.7322
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.218
    h:       -0.59826
    R:       2.85
    D:       0.15

md:
  ensemble:    npt
  output_dir:  ./npt_output
  temperature: 300.0
  pressure:    0.0                 # GPa
  timestep:    1.0                 # fs -- see the NVT template's
                                   # timestep note for the measured
                                   # drift-vs-dt table
  n_steps:     20000
  equil_steps: 2000
  tau_nhc:     20.0                # fs
  tau_baro:    200.0               # fs
  velocity_rescale_steps: 2000
  traj_every:  10
  thermo_every: 100
  skin_width:  0.5
  seed:        42
  timing:      true
  verbose:     true
"""

_YAML_GCMC_DEFAULT = """\
# akitsushima  --  Grand-Canonical Monte Carlo MD (muVT)
# Usage: akitsushima gcmc gcmc.yaml

structure:
  poscar: POSCAR

potential:
  type: LJ
  species: [Ar]
  # Bug-fix note (S24, found while assembling this template): this used
  # to read epsilon_Ar_Ar/sigma_Ar_Ar, which do not match the flat
  # {epsilon, sigma[, r_cut]} keys make_force_fn/_build_force_fn_from_yaml
  # actually looks up (see S19) -- using this template as-is would raise
  # KeyError inside LennardJonesPotential.pair_energy_force.
  params:
    epsilon: 0.01032
    sigma:   3.405
    r_cut:   10.0

gcmc:
  output_dir:              ./gcmc_output
  temperature:             300.0
  # mu is the FULL chemical potential [eV].  The acceptance rule carries
  # the ideal-gas factor V/(lambda^3 (N+1)) explicitly, so supplying an
  # EXCESS chemical potential here instead would double-count it (9.5 kT
  # for supercritical Ar at 200 K, rho* = 0.4).  Verified against the
  # exact ideal-gas result <N> = V exp(beta mu) / lambda^3.  See
  # GCMCMDSimulation's mu parameter note.  (The old key name 'mu_ex' is
  # still accepted for backward compatibility, but is deprecated -- it
  # was always the full chemical potential, not the excess, despite the
  # name.)
  mu:                       -0.1
  species:                 Ar
  timestep:                1.0
  n_cycles:                1000
  md_steps_per_cycle:      20
  gcmc_attempts_per_cycle: 50
  ins_del_ratio:           0.5
  max_disp:                0.2
  tau_nhc:                 100.0
  traj_every:              10
  thermo_every:            1
  skin_width:              0.5
  seed:                    42
  verbose:                 true
  timing:                  false
"""

_YAML_NEB_DEFAULT = """\
# akitsushima  --  Climbing-Image NEB
# Usage: akitsushima neb neb.yaml

structure:
  poscar_initial: POSCAR_initial
  poscar_final:   POSCAR_final

potential:
  type: Tersoff
  species: [Si]
  params:
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 1.7322
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.218
    h:       -0.59826
    R:       2.85
    D:       0.15

neb:
  output_dir:  ./neb_output
  n_images:    5
  k_spring:    5.0
  climbing:    true
  fmax:        0.05
  max_steps:   500
  verbose:     true
  timing:      false
"""

_YAML_PHONON_DEFAULT = """\
# akitsushima  --  Phonon calculation
# Usage: akitsushima phonon phonon.yaml

structure:
  poscar: POSCAR

potential:
  type: Tersoff
  species: [Si]
  params:
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 1.7322
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.218
    h:       -0.59826
    R:       2.85
    D:       0.15

phonon:
  output_dir:        ./phonon_output
  supercell:         [4, 4, 4]
  delta:             0.01
  n_jobs:            1
  band_structure:
    q_path:
      - ["G", [0.0, 0.0, 0.0]]
      - ["X", [0.5, 0.0, 0.5]]
      - ["K", [0.375, 0.375, 0.75]]
      - ["G", [0.0, 0.0, 0.0]]
      - ["L", [0.5, 0.5, 0.5]]
    n_points:     50
    coordinates:  reduced
  dos:
    q_mesh:    10
    n_dos:     200
    smearing:  5.0
  debye_temperature: false
  free_energy:       null
  verbose: true
"""

_YAML_KAPPA_DEFAULT = """\
# akitsushima  --  Thermal conductivity (Green-Kubo)
# Usage: akitsushima kappa kappa.yaml

structure:
  poscar: POSCAR

potential:
  type: Tersoff
  species: [Si]
  params:
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 1.7322
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.218
    h:       -0.59826
    R:       2.85
    D:       0.15

kappa:
  output_dir:      ./kappa_output
  temperature:     300.0
  n_steps:         100000
  timestep:        0.5
  equil_steps:     10000
  heat_flux_every: 5
  traj_every:      100
  thermo_every:    1000
  tau_max:         20.0             # ps, HCACF integration window
  n_blocks:        5                # block-averaging for uncertainty
  seed:            42
  slack_debye:
    gruneisen:  1.7
    use_vsound: false
    supercell:  2
    ifc_path:   null                # reuse a saved ifc.npz if available
  verbose:         true
"""

_YAML_ELM_DEFAULT = """\
# akitsushima  --  Machine-learning potential fitting
# Usage: akitsushima mlip mlip.yaml
#
# Available mlip.model choices: elm | qelm | nn | lgbm
# The section below shows the default (ELM) configuration in full,
# followed by commented-out examples for the other 3 model types.
# Keys not relevant to the selected model are ignored.

structure:
  training_csv: training_data.csv   # see load_mlip_csv for the CSV format

potential:
  type:  ELM
  r_cut: 5.0
  species: [Si]

mlip:
  output_dir: ./mlip_output
  model:      elm            # elm | qelm | nn | lgbm
  n_hidden:   512
  activation: tanh
  w_energy:   1.0
  w_forces:   0.1
  w_stress:   10.0
  reg:        1.0e-6
  r0:         1.0
  eps0:       0.01
  seed:       42
  verbose:    true
  test_csv:   null

# ----------------------------------------------------------------------
# Example: model: nn  (PyTorch feedforward neural network)
# ----------------------------------------------------------------------
# mlip:
#   output_dir: ./mlip_output
#   model:      nn
#   # Number of hidden layers = length of this list; each entry is that
#   # layer's width. [128, 64] -> 2 hidden layers; [64] -> 1; etc.
#   hidden_layers: [128, 64]
#   activation:    silu       # tanh | relu | gelu | silu | mish | softplus
#   w_energy:      1.0
#   w_forces:      0.1
#   w_stress:      10.0
#   r0:            1.0
#   eps0:          0.01
#   learning_rate: 1.0e-3
#   epochs:        500
#   batch_size:    256
#   weight_decay:  1.0e-6
#   early_stopping_patience: 30
#   device:        auto      # auto | cpu | cuda
#   seed:          42
#   verbose:       true
#   test_csv:      null

# ----------------------------------------------------------------------
# Example: model: lgbm  (LightGBM, with Optuna hyperparameter search
#                        always run as a standard part of fitting)
# ----------------------------------------------------------------------
# mlip:
#   output_dir: ./mlip_output
#   model:      lgbm
#   w_energy:   1.0
#   w_forces:   0.1
#   w_stress:   10.0
#   r0:         1.0
#   eps0:       0.01
#   # Fixed (non-optimised) LightGBM parameters, merged with the
#   # Optuna-selected values (learning_rate, num_leaves, max_depth,
#   # min_data_in_leaf, bagging_fraction, feature_fraction,
#   # min_gain_to_split, lambda_l1, lambda_l2, max_bin) before training.
#   lgbm_params:
#     objective:       regression
#     boosting:        gbdt
#     num_iterations:  10000
#     bagging_freq:    1
#     min_data_in_bin: 3
#     metric:          rmse
#     seed:            0
#     verbosity:       -1
#   optuna_trials:  100
#   optuna_timeout: null      # seconds, null = no timeout
#   early_stopping_rounds: 100
#   seed:    42
#   verbose: true
#   test_csv: null
"""

_YAML_TSA_DEFAULT = """\
# akitsushima  --  Transition State Analysis
# Usage: akitsushima tsa tsa.yaml

structure:
  poscar: POSCAR_saddle

potential:
  type: Tersoff
  species: [Si]
  params:
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 1.7322
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.218
    h:       -0.59826
    R:       2.85
    D:       0.15

tsa:
  output_dir:    ./tsa_output
  delta:         0.01
  mass_weighted: true
  threshold:     -1.0e-3          # eigenvalue cutoff for is_first_order_saddle
  irc_delta:     0.1
  irc_fmax:      0.05
  irc_max_steps: 500
  temperature:   300.0
  wigner:        false
  run_irc:       true
  E_init:        null              # only used when run_irc is false
  verbose:       true
"""

_YAML_OSMC_DEFAULT = """\
# akitsushima  --  OS-MC (soft-core lambda-integration GCMC)
# Usage: akitsushima osmc osmc.yaml

structure:
  poscar: POSCAR

potential:
  type: LJ
  species: [Ar]
  # Bug-fix note (S24): see the identical fix in _YAML_GCMC_DEFAULT above.
  params:
    epsilon: 0.01032
    sigma:   3.405
    r_cut:   10.0

osmc:
  output_dir:              ./osmc_output
  temperature:             300.0
  # mu is the FULL chemical potential [eV].  The acceptance rule carries
  # the ideal-gas factor V/(lambda^3 (N+1)) explicitly, so supplying an
  # EXCESS chemical potential here instead would double-count it (9.5 kT
  # for supercritical Ar at 200 K, rho* = 0.4).  Verified against the
  # exact ideal-gas result <N> = V exp(beta mu) / lambda^3.  See
  # GCMCMDSimulation's mu parameter note.  (The old key name 'mu_ex' is
  # still accepted for backward compatibility, but is deprecated -- it
  # was always the full chemical potential, not the excess, despite the
  # name.)
  mu:                       -0.1
  species:                 Ar
  timestep:                1.0
  n_cycles:                1000
  md_steps_per_cycle:      20
  gcmc_attempts_per_cycle: 50
  tau_nhc:                 100.0
  traj_every:              10
  thermo_every:            1
  seed:                    42
  verbose:                 true
  m_lam:          50.0
  sc_alpha:       0.5
  sc_n:           4
  lam_dt_frac:    1.0
  insertion_mode: random
  n_probe:        20
"""

_YAML_ESM_DEFAULT = """\
# akitsushima  --  Effective Screening Medium (ESM)
# Usage: akitsushima esm esm.yaml

structure:
  poscar: POSCAR_slab

potential:
  type: Buckingham
  charges:
    O:  -2.0
    Mg:  2.0
  potential_json: ./fitting_output/fitted_potential.json

esm:
  output_dir:     ./esm_output
  bc:             bc1
  sigma:          0.5
  n_grid_xy:      64
  n_grid_z:       256
  external_field: 0.0
  # r_cut drives a MINIMUM CELL SIZE via the 'relax' section below.
  # GeometryOptimiser builds a NeighbourList with skin_width=0.5, so it
  # requires r_list = r_cut + 0.5 < min_i(d_i) / 2, where d_i is the
  # PERPENDICULAR width (interplanar spacing) of the cell along axis i --
  # not the lattice vector length, which is larger for a sheared cell.
  # For a slab the vacuum direction is generous and the IN-PLANE width
  # is what binds.  At r_cut = 15.0 that means every perpendicular width
  # must exceed 31.0 A: an MgO slab (a = 4.21 A) needs an 8x8 in-plane
  # supercell (33.7 A); 4x4 (16.8 A) and 6x6 (25.3 A) are REJECTED with a
  # ValueError from NeighbourList.  Either build a large enough slab or
  # lower r_cut to suit yours (r_cut < in-plane width / 2 - 0.5).
  # Removing the 'relax' section drops the constraint entirely, since
  # ESMCalculator itself does not build a NeighbourList.
  r_cut:          15.0
  relax:
    fmax:      0.01
    max_steps: 200
    method:    fire
  verbose: true
"""

_DUMP_MAP = {
    "relax":  _YAML_RELAX_DEFAULT,
    "md":     _YAML_MD_NVT_DEFAULT,
    "nvt":    _YAML_MD_NVT_DEFAULT,
    "nve":    _YAML_MD_NVE_DEFAULT,
    "npt":    _YAML_MD_NPT_DEFAULT,
    "gcmc":   _YAML_GCMC_DEFAULT,
    "neb":    _YAML_NEB_DEFAULT,
    "tsa":    _YAML_TSA_DEFAULT,
    "osmc":   _YAML_OSMC_DEFAULT,
    "esm":    _YAML_ESM_DEFAULT,
    "phonon": _YAML_PHONON_DEFAULT,
    "kappa":  _YAML_KAPPA_DEFAULT,
    "mlip":   _YAML_ELM_DEFAULT,
    # No "fitpot" entry: the fitpot sub-command has been removed and
    # consolidated into "fit" (see cmd_fit.py's module docstring).
}
