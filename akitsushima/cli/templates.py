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
    "_YAML_ELASTIC_DEFAULT",
    "_YAML_GRUNEISEN_DEFAULT",
    "_YAML_QHA_DEFAULT",
    "_YAML_DIELECTRIC_DEFAULT",
    "_YAML_PIEZO_DEFAULT",
    "_YAML_NEMD_DEFAULT",
    "_YAML_ANNEAL_DEFAULT",
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
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

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
# External code instead (ADR-0011): each call writes work_dir/POSCAR, runs
# cmd there and reads work_dir/result.efs (E per atom, F, S in eV, eV/A,
# eV/A^3, stress positive under compression).  A wrapper converts the
# code's output, e.g. `python -m akitsushima.tools.ase_efs --read vasprun.xml`.
# potential:
#   type:     external
#   cmd:      "vasp_std > vasp.out && python -m akitsushima.tools.ase_efs --read vasprun.xml"
#   work_dir: ./external_run
#   timeout:  null             # seconds
#   stress_to_eV_Ang3: 1.0     # factor applied to the EFS stress column

relax:
  output_dir:    ./relax_output
  method:        fire               # fire | lbfgs | sd | diis | rmmdiis | rfo
  relax_cell:    false
  fix_volume:    false               # with relax_cell: keep the volume, relax
                                     # shape + ions; pmax tests the deviatoric
                                     # stress (ADR-0018)

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
  # -- checkpoint and restart (ADR-0031) -----------------------------
  checkpoint_every: 10               # steps; relax_checkpoint.npz in output_dir
  restart:       true                # continue from an existing checkpoint
                                     # (optimiser state starts afresh; raise
                                     # max_steps to go on after it was reached);
                                     # false: discard it and start afresh

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
  # fix_cell:    [[0, 0, 0], [0, 0, 0], [1, 1, 1]]  # (3, 3) per lattice-vector
  #                                  # component, 1 = frozen (here: c fixed)
  flush_every_step: true             # flush relax.log/XDATCAR each step
  traj_format: xdatcar               # xdatcar | extxyz | lammps-dump (ADR-0034)
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

  # -- Cell coordinates (relax_cell: true only, ADR-0041) ----------------
  # 'strain' (default): the cell moves by a deformation gradient F
  #   relative to the starting cell (h = h0 F^T, atoms carried along).
  #   The step is the same for any choice of lattice vectors, so a
  #   symmetric crystal stays symmetric (spglib finds the same space
  #   group at symprec 1e-5 after the relaxation).
  # 'lattice': the 9 raw lattice components are the coordinates (the
  #   behaviour before 0.3.0a5).  For non-orthogonal cells (fcc
  #   primitive, hexagonal, ...) an isotropic stress gives an anisotropic
  #   step, and the relaxed crystal keeps ~1e-5 A of symmetry-breaking
  #   strain.  Use it only to reproduce older results or to continue an
  #   older checkpoint.
  cell_coordinates: strain           # strain | lattice

  # -- Cell update scheme (cell_coordinates: lattice only) --------------
  # 'generalized_force' (default): f_cell = +volume * (h^-T @ stress),
  #   the exact dE/dh ('@' is a matrix product; h^-T multiplies stress
  #   from the LEFT).
  # 'vasp': f_cell = lattice @ stress, VASP's update (dyna.F) for
  #   lattice vectors as rows (stress @ lattice before 0.3.0a5).  Not an
  #   exact energy gradient.
  # With cell_coordinates: strain the scheme has no effect; 'vasp' then
  # warns and is deprecated (removed at the next minor release).
  cell_update_scheme: generalized_force   # generalized_force | vasp

  # -- Lattice component constraints (relax_cell: true only) -------------
  # Mask over all 9 raw lattice-matrix components, in row-major order
  # matching the lattice matrix (rows = lattice vectors):
  #   [a1x, a1y, a1z, a2x, a2y, a2z, a3x, a3y, a3z]
  # Each is 1 (free to relax) or 0 (frozen). null = all 9 free (default,
  # fully unconstrained cell).
  #
  # A frozen component keeps its starting value exactly (with
  # cell_coordinates: strain the mask becomes a linear constraint on F;
  # the meaning is the same).  The stress along frozen components is not
  # relaxed, so pmax may never be met: use check_stress: false then.
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
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

potential:
  type: Tersoff                    # see the timestep note in the md section below
  species: [Si]
  params:                          # Tersoff Si(C) / T3, PRB 38, 9902 (1988); lambda3 = 0
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 0.0
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.217
    h:       -0.59825
    R:       2.85
    D:       0.15

md:
  ensemble:     nvt
  thermostat:   nhc                # nhc (Nose-Hoover chain) | langevin (BAOAB,
                                   # ADR-0032) | csvr (Bussi velocity rescaling,
                                   # ADR-0039).  langevin and csvr use damping
                                   # (fs) instead of tau_nhc, and need
                                   # velocity_rescale_steps: 0
  # damping:    100.0              # fs, langevin (1/gamma; K relaxes in
                                   # damping/2) and csvr (K relaxes in damping)
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
  tau_nhc:      100.0              # fs  NHC thermostat time constant (50-200 typical)
  velocity_rescale_steps: 1000     # = equil_steps: VS active during equilibration
  traj_every:   10                 # write trajectory every N steps
  traj_format:  xdatcar            # xdatcar | extxyz | lammps-dump (ADR-0034)
  thermo_every: 50                 # write thermo log every N steps
  skin_width:   0.5
  seed:         42
  timing:       true
  verbose:      true  # -- checkpoint and restart (ADR-0031) -------------------------------
  checkpoint_every: 1000           # steps; md_checkpoint.npz in output_dir
  restart:      true               # continue from an existing checkpoint
                                   # (bit for bit; raise n_steps to extend);
                                   # false: discard it and start afresh
"""

_YAML_MD_NVE_DEFAULT = """\
# akitsushima  --  NVE molecular dynamics input
# Usage: akitsushima md md_nve.yaml

structure:
  poscar: POSCAR
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

potential:
  type: Tersoff
  species: [Si]
  params:   # Tersoff Si(C) / T3, PRB 38, 9902 (1988); lambda3 = 0
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 0.0
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.217
    h:       -0.59825
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
  traj_format:  xdatcar            # xdatcar | extxyz | lammps-dump
  thermo_every: 50
  skin_width:   0.5
  seed:         42
  timing:       true
  verbose:      true  # -- checkpoint and restart (ADR-0031) -------------------------------
  checkpoint_every: 1000           # steps; md_checkpoint.npz in output_dir
  restart:      true               # continue from an existing checkpoint
                                   # (bit for bit; raise n_steps to extend);
                                   # false: discard it and start afresh
"""

_YAML_MD_NPT_DEFAULT = """\
# akitsushima  --  NPT molecular dynamics input
# Usage: akitsushima md md_npt.yaml

structure:
  poscar: POSCAR
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

potential:
  type: Tersoff
  species: [Si]
  params:   # Tersoff Si(C) / T3, PRB 38, 9902 (1988); lambda3 = 0
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 0.0
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.217
    h:       -0.59825
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
  tau_nhc:     100.0               # fs
  tau_baro:    200.0               # fs
  cell_dof:    iso                 # ADR-0036; x, y, z are Cartesian:
                                   #   iso    volume only (cell shape kept)
                                   #   aniso  Lx, Ly, Lz independent, no shear
                                   #   tri    all six strain components
                                   #   xy     Lx, Ly; Lz and shear fixed (ESM)
                                   #   xy_iso Lx, Ly scaled together (ESM)
                                   #   z      Lz only (epitaxial film)
                                   # With ESM only xy / xy_iso at pressure 0.
  baro_thermostat: nhc             # nhc (deterministic, H_NPT conserved) |
                                   # langevin (Langevin piston: each strain
                                   # DOF thermalised; use it for volume /
                                   # strain fluctuations, e.g. elastic
                                   # constants or compressibility)
  velocity_rescale_steps: 2000
  traj_every:  10
  traj_format: xdatcar             # xdatcar | extxyz | lammps-dump
  thermo_every: 100
  skin_width:  0.5
  seed:        42
  timing:      true
  verbose:     true  # -- checkpoint and restart (ADR-0031) -------------------------------
  checkpoint_every: 1000           # steps; md_checkpoint.npz in output_dir
  restart:      true               # continue from an existing checkpoint
                                   # (bit for bit; raise n_steps to extend);
                                   # false: discard it and start afresh
"""

_YAML_GCMC_DEFAULT = """\
# akitsushima  --  Grand-Canonical Monte Carlo MD (muVT)
# Usage: akitsushima gcmc gcmc.yaml

structure:
  poscar: POSCAR
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

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
  traj_format:             xdatcar  # xdatcar | extxyz | lammps-dump
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
  params:   # Tersoff Si(C) / T3, PRB 38, 9902 (1988); lambda3 = 0
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 0.0
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.217
    h:       -0.59825
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
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

potential:
  type: Tersoff
  species: [Si]
  params:   # Tersoff Si(C) / T3, PRB 38, 9902 (1988); lambda3 = 0
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 0.0
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.217
    h:       -0.59825
    R:       2.85
    D:       0.15

phonon:
  output_dir:        ./phonon_output
  supercell:         [4, 4, 4]
  delta:             0.01
  n_jobs:            1
  symmetry:          false          # true: displace only inequivalent atoms
                                    # (spglib, ADR-0040; pip install
                                    # akitsushima[symmetry])
  symprec:           1.0e-5         # A, spglib tolerance
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
  thermal:                       # ADR-0018: q-mesh mode sum
    t_min:      0.0              # K
    t_max:      1000.0
    t_step:     10.0
    q_mesh:     20               # Gamma-centred; low-T Cv and Theta_D(T)
                                 # need finer meshes (acoustic modes)
    cutoff_THz: 1.0e-3           # modes at or below are left out
  verbose: true
"""

_YAML_KAPPA_DEFAULT = """\
# akitsushima  --  Thermal conductivity (Green-Kubo)
# Usage: akitsushima kappa kappa.yaml

structure:
  poscar: POSCAR
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

potential:
  type: Tersoff
  species: [Si]
  params:   # Tersoff Si(C) / T3, PRB 38, 9902 (1988); lambda3 = 0
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 0.0
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.217
    h:       -0.59825
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
    symmetry:   false               # spglib displacement reduction (ADR-0040)
  verbose:         true
"""

_YAML_NEMD_DEFAULT = """\
# akitsushima  --  Non-equilibrium MD with Langevin heat baths (ADR-0020)
# Usage: akitsushima nemd nemd.yaml
# The cell is periodic: heat flows from the hot to the cold slab both ways
# round, so the flux is Qdot / (2 A).  kappa depends on the length L along
# the axis: run several lengths and extrapolate 1/kappa against 1/L.
# Long runs: launch detached (nohup ... &); the checkpoint makes a re-run
# continue where the last one stopped.

structure:
  poscar: POSCAR             # e.g. fcc Ar, 4 x 4 x 16 conventional cells
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

potential:
  # Lennard-Jones argon, eps/k_B 119.8 K = 10.32 meV, sigma 3.405 A (the
  # values of the gcmc template); cutoff 2.5 sigma
  type: LJ
  species: [Ar]
  params:
    epsilon: 0.01032
    sigma:   3.405
    r_cut:   8.5

nemd:
  output_dir:  ./nemd_output
  axis:        c             # transport along lattice vector a | b | c
  hot:  {temperature: 50.0, region: [0.0, 0.0625]}    # fractional slab
  cold: {temperature: 30.0, region: [0.5, 0.5625]}    # (or atoms: [1-based])
  damping:     500.0         # fs, Langevin relaxation time of the baths
  timestep:    5.0           # fs
  equil_steps: 20000
  n_steps:     200000
  sample_every: 10
  n_bins:      16
  local_flux:  true          # per-bin heat flux from per-atom currents
  fit: {fraction: 0.5}       # central part of each segment for dT/dz
  kapitza: {interfaces: [], window: 0.1, skip_bins: 1}
  checkpoint_every: 1000     # steps; nemd_checkpoint.npz in output_dir
  restart:     true          # false: discard an existing checkpoint
  seed:        42
"""

_YAML_ELM_DEFAULT = """\
# akitsushima  --  Machine-learning potential fitting
# Usage: akitsushima mlip mlip.yaml
#
# Available mlip.model choices: elm | qelm | nn | linear
# The section below shows the default (ELM) configuration in full,
# followed by commented-out examples for the other model types.
# Keys not relevant to the selected model are ignored.
#
# Model form (ADR-0004): elm / qelm / nn learn E/N = f(d) and fit every
# F and S row as N grad f(d) . X (chain rule), so E, F and S come from one
# energy function; linear is E/N = w . d + b (exact without the chain
# rule).
# Row weight on the squared residual: w_cat * w_unit^2 * w_user.
#
# Training data -- either
#   (a) training_csv = a descriptor table (target,label,weight,<features>;
#       see load_mlip_csv), e.g. from an external descriptor script; or
#   (b) with mlip.descriptor set: training_csv = a structure list in the
#       `fit` layout (target,label,weight,struct_idx; E in eV/atom, forces,
#       Voigt stress xx yy zz yz zx xy compressive-positive) with
#       <poscar_dir>/<k>/POSCAR.  ACSF descriptors are computed in-package
#       and stored in the model file; the model can then run relax / md as
#           potential: {type: MLIP, model: ./mlip_output/elm_model.npz}

structure:
  training_csv: training_data.csv
  # poscar_dir: ./training_set      # (b) only; default: training_csv's dir

potential:
  species: [Si]                     # (b): element list (Z order is applied)

mlip:
  output_dir: ./mlip_output
  model:      elm            # elm | qelm | nn | linear
  n_hidden:   512
  activation: tanh
  solver:     auto           # auto | direct | lsqr
  row_independent: false     # true = pre-0.2.0a3 per-row form (not a force field)
  direct_link: true          # RVFL: + linear term in d (reg then acts
                             # on the hidden weights only)
  reg_search: null           # e.g. [1.0e2, 1.0e4, 1.0e6, 1.0e8]: pick reg on
                             # the validation split (direct solver)
  w_energy:   1.0
  w_forces:   0.1
  w_stress:   10.0
  reg:        1.0e-6
  r0:         1.0            # (b): F-row unit weight r0 [A]
  eps0:       0.01           # (b): S-row unit weight eps0 * V
  val_fraction: 0.2
  seed:       42
  verbose:    true
  test_csv:   null           # same kind of file as training_csv
  # ELM / QELM options (ADR-0013; also for form: atom_centred).  Default:
  # PCA whitening of the descriptor; the other options are off:
  # w_init_scale_search: [0.5, 1, 2, 4, 8]  # factors on the Glorot scale of
  #                            # W, b, picked on the validation split
  # bias_init:     normal      # normal | data (b = -W . x_k, unit centred
  #                            # on a training input)
  # standardize:   pca         # pca (default: whitening, pca_eps added to
  # pca_eps:       1.0e-3      # the correlation-matrix eigenvalues) | zscore
  # pca_drop:      0.0         # drop directions below pca_drop * max
  #                            # eigenvalue (0: keep all, recommended)
  # min_feature_std: 1.0e-12   # mask columns whose training std is below
  #                            # it (all models; 1e-4 suits external tables
  #                            # with near-constant columns, ADR-0010 A4)
  # w_correlation: none        # none | radial: G2 weights correlated along
  # w_corr_length: 0.2         # r_s [A] (in-package ACSF only)
  # w_corr_length_search: [0.1, 0.2, 0.4, 0.8]   # (radial implies zscore)
  # qelm only:
  # levels:        pow2        # pow2 | ternary | uniform
  # max_exponent:  3           # pow2: {0, +-1, ..., +-2^n} (formerly n_bits)
  # n_levels:      3           # uniform: odd number of levels {-k..k}
  # p_zero:        0.5         # fraction of each column set to zero
  # matmul:        dense       # dense | sparse (CSC product)
  # w_memory:      int8        # int8 (W as integer levels) | float64
  # Physics-informed parts (ADR-0014), with mlip.descriptor only:
  # baseline: {type: zbl, r_inner: 1.2, r_outer: 2.0}   # fixed repulsion,
  #                            # r_outer at the data's shortest distance,
  #                            # with descriptor inner_cutoff: [1.8, 2.0]
  # baseline: [{type: born_mayer, r_inner: 1.8, r_outer: 2.0,
  #             fit_to: sr/structures.csv}]   # A, rho fitted after training
  #                            # to short-range data (fit layout, POSCARs)
  # isolated_atom_energy: {Si: 0.0}   # eps(G = 0) = E_atom exactly
  # (b) in-package descriptor.  Example for Si with Stillinger-Weber
  # labels: r_cut = a * sigma = 1.80 * 2.0951 = 3.771 A (SW's own cutoff,
  # Stillinger & Weber, PRB 31, 5262 (1985)); radial G2 centred every
  # 0.2 A from 2.0 to 3.6 A with width 0.2 A (eta = 1/(2 * 0.2^2)).
  # descriptor:
  #   type:  acsf
  #   r_cut: 3.771
  #   g2: {eta: [12.5], rs: [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6]}
  #   # G5 (no r_jk term): with r_c = 3.771 A two neighbours of a Si atom
  #   # are 3.84 A apart, so G4's fc(r_jk) would drop the bond angles.
  #   g5: {eta: [0.0, 0.05], zeta: [1, 2, 4], lambda: [-1, 1]}
  #   inner_cutoff: [1.8, 2.0]  # pairs < 1.8 A leave G (ADR-0014); use with
  #                             # mlip.baseline covering that range
  # write_table: false       # also write the descriptor table CSV
  #                          # (and descriptor_table_test.csv with test_csv)

# ----------------------------------------------------------------------
# Example: form: atom_centred  (Behler-Parrinello, ADR-0005)
# ----------------------------------------------------------------------
# E = sum_i eps_s(i)(G_i), one function per species of the atom's own
# ACSF vector G_i, so the model is local and transfers to other cell
# sizes.  Needs (b): mlip.descriptor and a `fit`-layout structure list.
# Models: elm | qelm (closed-form solve) | nn (torch).  Model file:
# ./mlip_output/atomic_<model>_model.npz (.pt for nn), usable as
#     potential: {type: MLIP, model: ./mlip_output/atomic_elm_model.npz}
# mlip:
#   form:           atom_centred
#   model:          elm       # elm | qelm | nn
#   n_hidden:       400       # elm / qelm, per species
#   hidden_layers:  [64, 64]  # nn, per species
#   reg_search:     [1.0e2, 1.0e4, 1.0e6, 1.0e8]   # elm / qelm
#   descriptor: {type: acsf, r_cut: 3.771, g2: ..., g5: ...}
#   ensemble:       5         # elm / qelm (ADR-0033): mean of 5 seed-varied
#                             # members is the force field; their spread is
#                             # the uncertainty (sigF_max in md / relax logs)
#   ensemble_seeds: [42, 1, 2, 3, 4]   # optional; default seed + 1000 m
#
# Spread of a model on a list of structures (candidates for the next DFT
# batch), instead of a fit:
# mlip:
#   evaluate:
#     model:   ./mlip_output/atomic_elm_model.npz
#     poscars: ./candidates/*/POSCAR      # and / or a labelled list:
#     csv:     null                       # structures.csv (+ poscar_dir)
#     output:  ./mlip_uncertainty.csv

# ----------------------------------------------------------------------
# Example: model: linear  (E/N = w . d + b; methods ADR-0004 decision 9)
# ----------------------------------------------------------------------
# mlip:
#   model:        linear
#   method:       ridge     # ols | ridge | smooth_ridge | lasso |
#                           # elastic_net | pls | bayes_ridge | ard
#   alpha:        1.0e-6    # ridge / smooth_ridge / lasso / elastic_net
#   l1_ratio:     0.5       # elastic_net
#   n_components: 20        # pls
#   alpha_search: null      # e.g. [1.0e-8, 1.0e-6, 1.0e-4]; pls: [5, 10, 20]
#   min_feature_std: 1.0e-12  # mask near-constant columns (ADR-0010 A4)

# ----------------------------------------------------------------------
# Example: model: nn  (PyTorch feedforward neural network)
# ----------------------------------------------------------------------
# mlip:
#   output_dir: ./mlip_output
#   model:      nn
#   # Number of hidden layers = length of this list; each entry is that
#   # layer's width. [128, 64] -> 2 hidden layers; [64] -> 1; etc.
#   hidden_layers: [128, 64]
#   activation:    softplus   # softplus | tanh | gelu | silu | mish (relu: not
#                             # for MD -- its forces are discontinuous)
#   row_independent: false
#   direct_link:   true       # + linear term in d (see ELM above)
#   deterministic: false      # true = bit-reproducible training (1 thread)
#   batch_structures: 30      # structures per mini-batch (overrides batch_size)
#   optimizer:     adam       # adam | lbfgs (full batch; use learning_rate 1.0)
#   w_energy:      1.0
#   w_forces:      0.1
#   w_stress:      10.0
#   learning_rate: 1.0e-3
#   epochs:        1500
#   batch_size:    256        # rows per batch (row_independent, or batch_structures: null)
#   weight_decay:  1.0e-6
#   early_stopping_patience: 150
#   standardize:   zscore     # zscore (default for nn) | pca: the ELM's
#   pca_eps:       1.0e-3     # whitening map (also form: atom_centred,
#   pca_drop:      0.0        # per species; ADR-0013 addendum)
#   min_feature_std: 1.0e-12  # mask near-constant columns (ADR-0010 A4)
#   device:        auto      # auto | cpu | cuda
#   seed:          42
#   verbose:       true
#   test_csv:      null
"""

_YAML_TSA_DEFAULT = """\
# akitsushima  --  Transition State Analysis
# Usage: akitsushima tsa tsa.yaml

structure:
  poscar: POSCAR_saddle
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

potential:
  type: Tersoff
  species: [Si]
  params:   # Tersoff Si(C) / T3, PRB 38, 9902 (1988); lambda3 = 0
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 0.0
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.217
    h:       -0.59825
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
# akitsushima  --  osmc: CFCMC grand-canonical MC/MD (ADR-0009)
# Usage: akitsushima osmc osmc.yaml

structure:
  poscar: POSCAR
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

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
  traj_format:             xdatcar  # xdatcar | extxyz | lammps-dump
  thermo_every:            1
  seed:                    42
  verbose:                 true
  # CFCMC (ADR-0009): one fractional atom, U = U_N + g(lambda, U_N+1 - U_N)
  max_disp:          0.2    # displacement amplitude [A]
  lambda_move_ratio: 0.5    # fraction of MC moves that change lambda
  max_dlambda:       0.2    # lambda step uniform in [-max_dlambda, +max_dlambda]
  n_lambda_bins:     10     # bias w(lambda) and estimator bins
  equil_cycles:      200    # discarded; w(lambda) adapted during them
  adapt_weights:     true
  softcore_kT:       10.0   # E_c / kT of the energy-space soft core (0 = linear)
"""

_YAML_ESM_DEFAULT = """\
# akitsushima  --  Effective Screening Medium (ESM)
# Usage: akitsushima esm esm.yaml

structure:
  poscar: POSCAR_slab
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

potential:
  type: Buckingham
  charges:
    O:  -2.0
    Mg:  2.0
  potential_json: ./fitting_output/fitted_potential.json

esm:
  output_dir:     ./esm_output
  # bc1 vacuum|slab|vacuum, bc2 metal|slab|metal (external_field applies),
  # bc3 vacuum|slab|metal -- Otani-Sugino / Quantum ESPRESSO naming
  # (bc2 and bc3 were swapped in akitsushima before 2026-09-26)
  bc:             bc1
  sigma:          0.5
  n_grid_xy:      64
  n_grid_z:       256
  external_field: 0.0
  # esm is analysis-only (ADR-0008): single point, planar-averaged
  # potential, work function, charge density.  To relax or run MD with ESM
  # electrostatics use relax / md with potential.coulomb_method: esm.
  verbose: true
"""

_YAML_ELASTIC_DEFAULT = """\
# akitsushima  --  Elastic constants (ADR-0017)
# Usage: akitsushima elastic elastic.yaml
#
# The POSCAR must already be a supercell wide enough for the potential:
# r_cut + skin_width < half the smallest perpendicular cell width at every
# strain (refused otherwise).  For this Tersoff Si, r_cut 3.0 + 0.5 needs
# the 64-atom 2x2x2 conventional cell, not the 8-atom one.

structure:
  poscar: POSCAR
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

potential:
  # Tersoff, PRB 38, 9902 (1988), Si(C): C11 142.5, C12 75.4, C44 69 GPa
  type: Tersoff
  species: [Si]
  params:
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 0.0
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.217
    h:       -0.59825
    R:       2.85
    D:       0.15

elastic:
  output_dir:  ./elastic_output
  pre_relax:   true          # relax cell + ions first
  relax:                     # pre-relaxation settings
    method:    fire
    fmax:      1.0e-3        # eV/A
    pmax:      0.01          # GPa
    max_steps: 3000
  strains:     [0.0025, 0.005, 0.01]   # each applied with + and -
  relax_ions:  true          # relaxed-ion C and internal strain
  fmax:        1.0e-4        # eV/A, ionic relaxation at each strain
  max_steps:   1000
  skin_width:  0.5           # A
  n_jobs:      1             # threads over strained cells (-1: all cores)
  symmetry:    false         # true: strain only the modes symmetry needs
                             # (spglib, ADR-0040)
  symprec:     1.0e-5        # A, spglib tolerance
  masses:      {}            # amu overrides, e.g. {Si: 28.0855}
  verbose:     true
"""

_YAML_GRUNEISEN_DEFAULT = """\
# akitsushima  --  Mode Grueneisen parameters (ADR-0018)
# Usage: akitsushima gruneisen gruneisen.yaml
# POSCAR: the PRIMITIVE cell (as for phonon).  Three structures relaxed at
# V0 (1 - dv), V0, V0 (1 + dv); gamma = -(V/2w^2) <e|dD/dV|e>.

structure:
  poscar: POSCAR
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

potential:
  # Tersoff Si(C) / T3, PRB 38, 9902 (1988); lambda3 = 0
  type: Tersoff
  species: [Si]
  params:
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 0.0
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.217
    h:       -0.59825
    R:       2.85
    D:       0.15

gruneisen:
  output_dir:  ./gruneisen_output
  volume_step: 0.01           # dv, fractional volume step
  supercell:   [4, 4, 4]      # phonon supercell; also the relaxation cell,
                              # so r_cut + 0.5 A skin < half its width
  delta:       0.01           # A, finite displacement
  symmetry:    false          # spglib displacement reduction (ADR-0040)
  pre_relax:   true           # full cell + ion relaxation first -> V0
  relax:                      # every relaxation (pre and fixed-volume)
    method:    fire
    fmax:      1.0e-4         # eV/A
    pmax:      1.0e-3         # GPa (deviatoric at fixed volume)
    max_steps: 5000
  masses:      {}
  n_jobs:      1
  verbose:     true
  band_structure:
    q_path:
      - ["G", [0.0, 0.0, 0.0]]
      - ["X", [0.5, 0.0, 0.5]]
      - ["K", [0.375, 0.375, 0.75]]
      - ["G", [0.0, 0.0, 0.0]]
      - ["L", [0.5, 0.5, 0.5]]
    n_points: 50
  mesh:                       # C_v-weighted mean gamma(T)
    q_mesh: 12
    t_min:  10.0
    t_max:  1000.0
    t_step: 10.0
"""

_YAML_QHA_DEFAULT = """\
# akitsushima  --  Quasi-harmonic approximation (ADR-0018)
# Usage: akitsushima qha qha.yaml
# POSCAR: the PRIMITIVE cell.  n_volumes structures relaxed at fixed volume
# (shape + ions), F(V,T) = E(V) + F_vib(V,T), EOS fit at each T.

structure:
  poscar: POSCAR
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

potential:
  # Tersoff Si(C) / T3, PRB 38, 9902 (1988); lambda3 = 0
  type: Tersoff
  species: [Si]
  params:
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 0.0
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.217
    h:       -0.59825
    R:       2.85
    D:       0.15

qha:
  output_dir:   ./qha_output
  volume_range: 0.05          # volumes V0 x [1 - r, 1 + r]
  n_volumes:    7             # >= 5
  eos:          birch-murnaghan   # birch-murnaghan | vinet
  t_min:        0.0
  t_max:        1000.0
  t_step:       10.0
  q_mesh:       20
  imaginary_tol_THz: 0.05     # stop if any mode is below -tol ...
  skip_unstable_volumes: false   # ... unless true: drop that volume
  supercell:   [4, 4, 4]      # phonon supercell; also the relaxation cell,
                              # so r_cut + 0.5 A skin < half its width
  delta:       0.01           # A, finite displacement
  symmetry:    false          # spglib displacement reduction (ADR-0040)
  pre_relax:   true           # full cell + ion relaxation first -> V0
  relax:                      # every relaxation (pre and fixed-volume)
    method:    fire
    fmax:      1.0e-4         # eV/A
    pmax:      1.0e-3         # GPa (deviatoric at fixed volume)
    max_steps: 5000
  masses:      {}
  n_jobs:      1
  verbose:     true
"""

_YAML_DIELECTRIC_DEFAULT = """\
# akitsushima  --  Static dielectric tensor, LO-TO splitting, IR modes
# Usage: akitsushima dielectric dielectric.yaml
# POSCAR: the primitive (or conventional) cell, as for phonon.

structure:
  poscar: POSCAR
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

potential:
  # Rigid-ion MgO: Buckingham of Lewis and Catlow, J. Phys. C 18, 1149
  # (1985), formal charges +-2, PPPM.  mgo_buckingham.json:
  #   {"potential_type": "buckingham", "r_cut": 5.0, "species": ["Mg", "O"],
  #    "charges": {"Mg": 2.0, "O": -2.0},
  #    "params": {"Mg-O": {"A": 1428.5, "rho": 0.2945, "C": 0.0},
  #               "O-O":  {"A": 22764.0, "rho": 0.149, "C": 27.88},
  #               "Mg-Mg": {"A": 0.0, "rho": 0.1, "C": 0.0}}}
  type: Buckingham
  potential_json: ./mgo_buckingham.json
  r_cut: 5.0
  coulomb_method: pppm

born:                       # ADR-0019; this block also turns on NAC in
                            # phonon, gruneisen and qha
  epsilon_inf: 2.95         # MgO, experiment (n^2 at optical frequencies);
                            # omit for the rigid-ion default 1 (warned)
  # born_charges:           # default: the potential's point charges
  #   per_species: {Mg: 2.0, O: -2.0}      # scalar, [xx,yy,zz] or 3x3
  #   per_atom: {1: [[...],[...],[...]]}  # 1-based POSCAR index, wins
  # clamped_ion_piezo: 3x6 C/m^2           # electronic e (piezo only)

dielectric:
  output_dir: ./dielectric_output
  supercell:  [3, 3, 3]     # r_cut + 0.5 A skin < half its width
  delta:      0.01
  pre_relax:  true          # cell + ions; ignored with load_ifc
  relax: {method: fire, fmax: 1.0e-4, pmax: 1.0e-3, max_steps: 5000}
  # load_ifc: ./phonon_output   # reuse a phonon run of the same POSCAR +
  #                             # potential (checked by input hash)
  n_jobs:     1
  masses:     {}
"""

_YAML_PIEZO_DEFAULT = """\
# akitsushima  --  Relaxed-ion piezoelectric tensors (rigid ion, ADR-0019)
# Usage: akitsushima piezo piezo.yaml
# POSCAR: a supercell wide enough for the potential (as for elastic).
# e = e0 + (e/V) sum_k Z_k du_k/deta  from the elastic internal strain;
# d = e S; k from d, S and the stress-free permittivity.

structure:
  poscar: POSCAR
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

potential:
  # Rigid-ion MgO: Buckingham of Lewis and Catlow, J. Phys. C 18, 1149
  # (1985), formal charges +-2, PPPM.  mgo_buckingham.json:
  #   {"potential_type": "buckingham", "r_cut": 5.0, "species": ["Mg", "O"],
  #    "charges": {"Mg": 2.0, "O": -2.0},
  #    "params": {"Mg-O": {"A": 1428.5, "rho": 0.2945, "C": 0.0},
  #               "O-O":  {"A": 22764.0, "rho": 0.149, "C": 27.88},
  #               "Mg-Mg": {"A": 0.0, "rho": 0.1, "C": 0.0}}}
  type: Buckingham
  potential_json: ./mgo_buckingham.json
  r_cut: 5.0
  coulomb_method: pppm

born:                       # ADR-0019; this block also turns on NAC in
                            # phonon, gruneisen and qha
  epsilon_inf: 2.95         # MgO, experiment (n^2 at optical frequencies);
                            # omit for the rigid-ion default 1 (warned)
  # born_charges:           # default: the potential's point charges
  #   per_species: {Mg: 2.0, O: -2.0}      # scalar, [xx,yy,zz] or 3x3
  #   per_atom: {1: [[...],[...],[...]]}  # 1-based POSCAR index, wins
  # clamped_ion_piezo: 3x6 C/m^2           # electronic e (piezo only)

elastic:                    # used when piezo.load_elastic is not given
  output_dir: ./elastic_output
  pre_relax: true
  relax: {method: fire, fmax: 1.0e-3, pmax: 0.01, max_steps: 3000}
  strains: [0.0025, 0.005, 0.01]
  fmax: 1.0e-4

piezo:
  output_dir: ./piezo_output
  # load_elastic: ./elastic_output   # reuse an elastic run (input hash checked)
  field:      5.0e-3        # V/A, for the strain-clamped eps by relaxation
  relax: {method: fire, fmax: 1.0e-5, max_steps: 10000}
"""

_YAML_ANNEAL_DEFAULT = """\
# akitsushima anneal -- global structure search (ADR-0022)
# Usage: akitsushima anneal anneal.yaml
#
# Cycle 0 relaxes the input (the relax-only reference).  Then:
#   md_quench     : one continuous MD run from that relaxed structure, cut
#                   into chunks of quench_every steps; a copy of the
#                   structure is relaxed after every chunk.  The target
#                   temperature follows `schedule`, one value per segment.
#   basin_hopping : random displacement -> relax -> Metropolis on the
#                   relaxed energy (enthalpy when pressure != 0).
# Every cycle is logged to anneal_log.jsonl and checkpointed; set
# restart: true to continue an interrupted run.  Distinct minima go to
# POSCAR_rank01.. and minima.csv, the lowest to POSCAR_best.

structure:
  poscar: POSCAR
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

potential:
  type: SW
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

anneal:
  method:     md_quench            # md_quench | basin_hopping
  output_dir: ./anneal_output
  seed:       42
  pressure:   0.0                  # GPa: NPT target and H = E + PV in every quench
  restart:    false
  max_consecutive_failures: 5      # stop after this many failed cycles in a row
  min_energy_per_atom: null        # eV/atom; lower = failed (MLIP collapse guard)
  keep_best:  10                   # distinct minima written as POSCAR_rankNN
  dedup_energy: 1.0e-4             # eV/atom  } same minimum when both
  dedup_volume: 1.0e-3             # relative } differences are this small

  # -- md_quench -----------------------------------------------------
  # 1500 K is below the melting point of SW silicon (1691 K, Broughton &
  # Li, PRB 35, 9120 (1987)): the crystal anneals instead of melting.
  schedule:
    type:       geometric          # constant | linear | geometric | list
    n_segments: 8
    T_start:    1500.0             # K
    T_end:      300.0              # K
    # temperatures: [1500, 1200, 900, 600, 300]   # for type: list
  segment_steps: 2000              # MD steps per segment
  quench_every:  500               # MD steps between quenches
  md:
    ensemble:     nvt              # nvt | npt (npt uses `pressure`)
    thermostat:   nhc              # nhc | langevin | csvr (nvt only; ADR-0039)
    # damping:    100.0            # fs, langevin / csvr (instead of tau_nhc)
    timestep:     1.0              # fs
    tau_nhc:      20.0             # fs
    tau_baro:     1000.0           # fs (npt)
    cell_dof:     iso              # npt: iso | aniso | tri | xy | xy_iso | z
    baro_thermostat: nhc           # npt: nhc | langevin
    traj_every:   50
    thermo_every: 50
    skin_width:   0.5

  # -- basin_hopping -------------------------------------------------
  basin_hopping:
    n_trials:          100
    temperature:       1000.0      # K, Metropolis on the relaxed energy
    step:              0.3         # A, radius of the per-atom displacement
    cell_step:         0.02        # max symmetric strain (relax_cell only)
    adapt_every:       10          # trials between step updates
    adapt_factor:      0.9         # step *= 0.9 below target, /= 0.9 above
    target_acceptance: 0.5

  # -- every quench (same keys as the relax command) -------------------
  relax:
    method:     fire               # fire | lbfgs | sd | diis | rmmdiis | rfo
    relax_cell: false
    fmax:       0.05               # eV/A
    pmax:       0.1                # GPa (relax_cell)
    max_steps:  2000
    max_displacement: 0.2          # A
"""

_YAML_BTE_DEFAULT = """\
# akitsushima  --  Lattice thermal conductivity, RTA (ADR-0042)
# Usage: akitsushima bte bte.yaml
# POSCAR: the unit cell (here the 8-atom conventional diamond cell).  fc2 and
# fc3 come from finite displacements on diagonal supercells of it; gamma is
# the three-phonon linewidth (Togo et al., PRB 91, 094306 (2015)), kappa
# = 1/(N V) sum C v v tau with tau = 1 / (4 pi gamma).

structure:
  poscar: POSCAR
  # or file: <path> -- POSCAR, *.extxyz, *.data / *.lmp (LAMMPS), *.cif
  # format: auto      # poscar | extxyz | lammps-data | cif; index: -1 (frame)

potential:
  # Tersoff Si(C) / T3, PRB 38, 9902 (1988); lambda3 = 0
  type: Tersoff
  species: [Si]
  params:
    A:       1830.8
    B:       471.18
    lambda1: 2.4799
    lambda2: 1.7322
    lambda3: 0.0
    beta:    1.1e-6
    n:       0.78734
    c:       100390.0
    d:       16.217
    h:       -0.59825
    R:       2.85
    D:       0.15

bte:
  output_dir:   ./bte_output
  supercell:    [2, 2, 2]      # fc3 supercell (64 atoms for this POSCAR)
  supercell_fc2: null          # null = supercell; larger is usual
  ifc_path:     null           # reuse fc2 from the phonon command's ifc.npz
  delta:        0.03           # A, fc3 displacements (both steps)
  delta_fc2:    0.01           # A, fc2 displacements
  cutoff_pair:  null           # A, skip second displacements farther away
  symmetry:     true           # spglib reduction (pip install akitsushima[symmetry])
  symprec:      1.0e-5
  sym_fc3:      true           # index-permutation symmetry + acoustic sum rule
  mesh:         [11, 11, 11]   # Gamma-centred q mesh; converge it
  sigma:        0.1            # THz, Gaussian width of the delta function
  temperatures: [100, 200, 300, 400, 500]
  occupation:   quantum        # quantum (Bose-Einstein) | classical (kT / hf)
  n_jobs:       1              # threads for the displaced force calls
  # -- optional scattering (off when absent) ---------------------------
  # mass_variance: {Si: 2.01e-4}   # Tamura g per species
  # isotopes:                      # or g from the isotopic composition:
  #   Si: {27.97693: 0.92223, 28.97649: 0.04685, 29.97377: 0.03092}
  #   # IUPAC representative abundances and AME masses; gives g = 2.01e-4,
  #   # the value of Tamura, PRB 27, 858 (1983)
  # alloy: {host: Ge, Sn: 0.08}    # virtual crystal: host mass averaged,
  #                                # g from the composition; force-constant
  #                                # disorder is ignored (ADR-0042)
  # boundary_length_um: 1.0        # Casimir length: 1 / tau_b = |v| / L
  # born: ...                      # Born charges + eps_inf for NAC (polar)
  verbose:      true
"""

_YAML_DATASET_DEFAULT = """\
# akitsushima dataset -- fit / mlip training set from VASP runs (ADR-0021)
#
# Every ionic step becomes one structure: <output_dir>/{train,test}/
# structures.csv (target,label,weight,struct_idx; E in eV/atom, F in eV/A,
# Voigt stress xx yy zz yz zx xy in eV/A^3, compressive-positive) and
# <k>/POSCAR, plus manifest.csv (run, step, composition, raw energy).
# Runs are split as a whole: the steps of one relaxation never end up on
# both sides.

dataset:
  sources:                  # a run directory, or a parent whose sub-directories are runs
    - ./vasp_runs
  reader:      vasprun      # vasprun (vasprun.xml) | outcar (OUTCAR) | extxyz
  #                         # extxyz (ADR-0034): each file is one run; labels
  #                         # energy / forces / stress (ASE sign), renamed by
  # extxyz_keys: {energy: energy, forces: forces, stress: stress}
  cross_check: true         # also read the other file; stop if any step disagrees
  energy:      wo_entropy   # wo_entropy | free | sigma0
  frames:      all          # all | last | first_last | <stride>
  # isolated_atom_energy: {Si: -0.8}   # eV; target = (E - sum n_s E_s) / N
  # isolated_atom_energy: {Si: ./vasp_atoms/Si}   # or the atom's VASP run
  #                           # (its last step, read with `energy`; ADR-0026)
  split:
    test_fraction: 0.2      # fraction of RUNS held out (0 = train only)
    stratify:      composition   # composition | none
    seed:          0
  r_cut_check: null         # A; warn when half the smallest cell width <= this
  output_dir:  ./training_set
"""

_YAML_ACTIVE_DEFAULT = """\
# akitsushima active -- active learning driven by the ensemble uncertainty
# (ADR-0037).  Usage: akitsushima active active.yaml
# Running it again resumes from output_dir/active_ledger.jsonl.  With
# label mode vasp_inputs it stops (exit code 2) until the VASP runs of the
# round are done, then continues.

active:
  output_dir: ./active_output
  rounds: 5                       # label-and-retrain rounds at most
  seed: 42
  model:
    train_dir: training_set/train  # fit layout (structures.csv + <k>/POSCAR)
    test_dir:  training_set/test   # optional; RMSE per round
    # file: mlip_output/atomic_elm_model.npz   # an ensemble; else trained here
  train:                          # the mlip settings of the ensemble (ADR-0033)
    model: elm
    form: atom_centred
    ensemble: 5
    n_hidden: 400
    descriptor:                   # the GeSn ACSF of the CLI tests (ADR-0021)
      type: acsf
      r_cut: 5.0
      g2: {eta: [12.5], rs: [2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8]}
      g5: {eta: [0.0, 0.05], zeta: [1, 2, 4], lambda: [-1, 1]}
    # isolated_atom_energy / baseline / reg_search / ...: as in mlip
  explore:                        # in cells DFT can label (<= 64 atoms)
    - name: md900
      kind: md
      structures: [start/POSCAR_*]
      md: {ensemble: nvt, temperature: 900.0, timestep: 1.0, n_steps: 2000,
           traj_every: 50, thermo_every: 10}
    - name: anneal
      kind: anneal
      structures: [start/POSCAR_*]
      anneal:
        method: md_quench
        schedule: {type: geometric, n_segments: 4, T_start: 1500.0, T_end: 300.0}
        segment_steps: 1000
        quench_every: 500
        md: {ensemble: nvt, traj_every: 50, thermo_every: 10}
        relax: {method: fire, fmax: 0.05, max_steps: 500}
  select:
    percentile: 99.0              # lo = this percentile of sigma_F,max on the
    hi_factor: 10.0               #   training structures; hi = hi_factor * lo
    # lo: 0.05                    # absolute values [eV/A] override them
    # hi: 0.5
    k: 10                         # structures per round (farthest-point sampling)
    strategy: uncertainty         # uncertainty | random (baseline)
    stop_sigma: true              # stop explorer trajectories above hi ...
    stop_consecutive: 3           # ... on this many logged steps in a row
    min_distance_factor: 0.75     # frames with atoms closer than this times the
    # min_distance: 1.7           #   training set's shortest distance are never
                                  #   labelled (absolute [A] overrides)
  label:
    mode: vasp_inputs             # vasp_inputs | external
    incar: INCAR                  # template, copied into every directory
    potcar: {Ge: /path/to/potpaw/Ge_d/POTCAR, Sn: /path/to/potpaw/Sn_d/POTCAR}
    kpoint_density: 44.0          # A: k_i = ceil(K / L_i), Gamma-centred
    energy: wo_entropy            # wo_entropy | free | sigma0
    # isolated_atom_energy: {Ge: -0.78, Sn: -0.65}   # eV, as in dataset
    # mode: external              # ADR-0011, e.g. a reference potential:
    # potential: {type: external, cmd: "python -m akitsushima.tools.akitsushima_efs /abs/ref.yaml",
    #             input_file: POSCAR, output_file: result.efs}
    # n_jobs: 1
  converge:
    accurate_fraction: 0.99       # stop when this share is below lo and none above hi
"""


_YAML_BUILD_DEFAULT = """\
# akitsushima build -- write a structure (ADR-0035)
# Usage: akitsushima build build.yaml
#
# Ge(1-x)Sn(x), x = 8/64 = 0.125, 64-atom cubic diamond cell.  Lattice
# constant from Vegard's law between experimental a(Ge) = 5.658 A and
# a(alpha-Sn) = 6.493 A: a = 0.875 * 5.658 + 0.125 * 6.493 = 5.7624 A
# (Vegard's law is an approximation for GeSn; check against measured
# lattice constants for the composition you build).

build:
  start:
    bulk:
      type:    diamond     # diamond | fcc | bcc | hcp | zincblende | wurtzite | rocksalt
      species: [Ge]        # two for zincblende / wurtzite / rocksalt (first on the origin)
      a:       5.7624      # A
      cell:    conventional   # conventional | primitive (hcp / wurtzite: hexagonal)
    # file: POSCAR         # or start from a file (POSCAR, extxyz, LAMMPS data, CIF)
  steps:                   # applied in order
    - supercell: [2, 2, 2]           # or a 3x3 matrix, or {min_width: 12.0}
    - substitute: {from: Ge, to: {Sn: 8}, seed: 1}   # exact count (or a fraction 0.125)
    # - vacancy:  {species: Ge, count: 1, seed: 3}   # or {indices: [0]}
    # - rattle:   {sigma: 0.05, seed: 2}             # A, every coordinate
    # - strain:   {volume: 0.03, shear: 0.02, seed: 4}   # or {epsilon: 3x3}
    # - scale:    {factor: 1.01}                     # or {volume: A^3}
    # - slab:     {miller: [1, 1, 1], layers: 3, vacuum: 15.0, termination: 0,
    #              center: zero, orthogonal_ab: true}  # center zero: ESM's
  output: POSCAR_GeSn      # POSCAR*, *.extxyz, *.data, *.cif; or give format:
  repeat: 1                # n > 1: n structures, every seed + k (k = 0..n-1)
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
    "elastic": _YAML_ELASTIC_DEFAULT,
    "gruneisen": _YAML_GRUNEISEN_DEFAULT,
    "qha":    _YAML_QHA_DEFAULT,
    "dielectric": _YAML_DIELECTRIC_DEFAULT,
    "piezo":  _YAML_PIEZO_DEFAULT,
    "nemd":   _YAML_NEMD_DEFAULT,
    "bte":    _YAML_BTE_DEFAULT,
    "mlip":   _YAML_ELM_DEFAULT,
    "dataset": _YAML_DATASET_DEFAULT,
    "anneal": _YAML_ANNEAL_DEFAULT,
    "build": _YAML_BUILD_DEFAULT,
    "active": _YAML_ACTIVE_DEFAULT,
    # No "fitpot" entry: the fitpot sub-command has been removed and
    # consolidated into "fit" (see cmd_fit.py's module docstring).
}
