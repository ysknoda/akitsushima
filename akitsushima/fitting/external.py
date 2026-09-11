"""External-code and force/stress-scaling force_fn adapters.

ExternalFileEngine bridges akitsushima's solver layer (GeometryOptimiser,
NVTSimulation, GCMCMDSimulation, ...) with any external electronic-structure
or force-field code that reads/writes plain text files.

One call to engine.compute() does four things in sequence:
 1. Write the current structure as a VASP 5 POSCAR-format text file.
 2. Run the user-supplied shell command (e.g. 'vasp_std', 'pw.x < espresso.in').
 3. Wait for the command to exit.
 4. Parse the output file in the akitsushima EFS format (see
    :func:`~akitsushima.io.efs.read_efs_file`).

ScaledForceEngine wraps any force callable and applies independent
proportional scaling and hard clamping to forces and stresses before
they reach the solver.  Energy is forwarded unchanged.
"""
import os
import subprocess as _sp
import time as _time
from typing import Callable, Optional, Tuple

import numpy as np

from ..engine.cell_gradient import cell_force_from_stress
from ..io.efs import read_efs_file
from ..io.poscar import write_poscar

__all__ = ["ExternalFileEngine", "ScaledForceEngine"]


class ExternalFileEngine:
    """
    Generic file-based external force engine.

    Each call to :meth:`compute` executes the following pipeline:

    1. **Write** the current structure to a file.
       - Default: VASP 5 POSCAR format via :func:`write_poscar`.
       - Custom: supply ``input_writer(path, lattice, positions, atom_types)``.

    2. **Run** the shell command (``cmd``) inside ``work_dir``.
       The command may reference ``{input}`` and ``{output}`` placeholders:

           cmd = "vasp_std"                         # simple case
           cmd = "mpirun -np 8 pw.x < pw.in > pw.out"
           cmd = "python wrap.py {input} {output}"  # explicit paths

    3. **Parse** the output file in the akitsushima EFS plain-text format
       via :func:`read_efs_file`.
       - Default output file: ``result.efs`` in ``work_dir``.
       - Custom: supply ``output_reader(path, n_atoms) -> (E, F, S)``.

    Parameters
    ----------
    cmd              : shell command to execute.  Run inside ``work_dir``.
                       Supports ``{input}`` / ``{output}`` format placeholders.
    work_dir         : directory in which the command runs and files are written.
                       Created automatically if it does not exist.
    input_file     : name of the structure input file written to ``work_dir``.
                       Default: ``'POSCAR'``.
    output_file       : name of the EFS result file expected in ``work_dir``.
                       Default: ``'result.efs'``.
    input_writer     : optional callable with signature
                       ``(path, lattice, positions, atom_types) -> None``.
                       When supplied, replaces the default POSCAR writer.
                       Use this for Quantum ESPRESSO, CP2K, LAMMPS, etc.
    output_reader    : optional callable with signature
                       ``(path, n_atoms) -> (energy, forces, stress)``.
                       When supplied, replaces the built-in EFS parser.
    mpi_prefix       : prefix prepended to ``cmd``, e.g. ``'mpirun -np 8'``.
    timeout          : maximum wall-clock seconds for the external call.
                       ``None`` = no limit (default).
    stress_to_eV_Ang3: unit conversion factor for stress values in the EFS
                       output (applied by the built-in parser).
                       Common values:
                       - 1.0                              -- already eV/A^3
                       - constants.VASP_KB_TO_EV_ANG3      -- VASP kB (raw
                         OUTCAR "in kB") -> eV/A^3, i.e. -1/1602.176621.
                         Bug-fix note: an earlier version of this docstring
                         said ``1/160.218``, which is the eV/A^3->GPa factor
                         (not eV/A^3->kbar -- 1 GPa = 10 kbar exactly) and
                         also omitted a required sign flip: VASP's OUTCAR
                         reports its "Total"/"in kB" stress with the
                         OPPOSITE sign from this codebase's tensile-positive
                         convention (confirmed via VASP's own forum
                         documentation -- "FORCE on cell = -STRESS" -- and
                         via ASE's production
                         ``ase.calculators.vasp.vasp::read_stress``, which
                         negates the raw value). Use the ready-made
                         ``VASP_KB_TO_EV_ANG3`` constant rather than
                         re-deriving this factor by hand.
                       - 14.7103                          -- QE Ry/bohr^3 -> eV/A^3
    env              : optional dict of extra environment variables to pass to
                       the subprocess (merged with the current environment).
    cache            : if True (default), skip re-computation when the
                       structure is identical to the previous call
                       (max atom displacement < ``cache_tol`` A, same lattice).
    cache_tol        : position-change threshold for cache hit [A].  Default 1e-6.
    r_cut            : short-range cutoff [A] stored as an attribute so that
                       ``make_force_fn(engine)`` and solver constructors can
                       auto-detect it.  Has no effect on the computation
                       itself (the external code manages its own cutoffs).
    verbose          : if True, print each call's command, timing, and E/F/S
                       summary.

    Security note
    -------------
    ``cmd`` is executed via ``subprocess.run(..., shell=True)`` so that
    pipes, redirections (``pw.x < in > out``), and MPI wrappers work
    without extra quoting logic.  This mirrors how ASE-style external
    calculators are normally used: ``cmd`` is a trusted, user-authored
    string (a local executable name or shell pipeline), never data
    derived from an untrusted source.  Do not build ``cmd`` from
    external/untrusted input.

    Examples
    --------
    **VASP (simplest)**::

        engine = ExternalFileEngine(
            cmd      = 'vasp_std',
            work_dir = 'vasp_run',
        )
        fn = make_force_fn(engine, r_cut=10.0)
        go = GeometryOptimiser(lat, pos, types, force_fn=fn, method='lbfgs')
        go.run()

    **Quantum ESPRESSO with custom writer**::

        def write_qe_input(path, lattice, positions, atom_types):
            ...  # write pw.x input file

        engine = ExternalFileEngine(
            cmd          = 'mpirun -np 4 pw.x < pw.in > pw.out',
            work_dir     = 'qe_run',
            input_writer = write_qe_input,
            input_file = 'pw.in',
            output_file   = 'result.efs',   # wrapper script writes this
        )

    **Custom output reader (LAMMPS, etc.)**::

        def read_lammps(path, n_atoms):
            ...  # parse LAMMPS log/dump -> (E, F, S)

        engine = ExternalFileEngine(
            cmd           = 'lammps -in in.lammps',
            work_dir      = 'lammps_run',
            output_reader = read_lammps,
        )

    **Read-back self-test (no external code required)**::

        # Write EFS file manually and verify round-trip.
        write_efs_file('test.efs', energy=-28.68, forces=np.zeros((2,3)),
                       stress=np.zeros((3,3)))
        # File content (2-atom system, energy = -28.68 eV -> -14.34 eV/atom):
        #   -14.3400000000  E  0
        #    0.0000000000  F  1x
        #    ...
        E, F, S = read_efs_file('test.efs', n_atoms=2)
        # E == -28.68  (restored: -14.34 x 2)
    """

    def __init__(self,
                 cmd              : str   = '',
                 work_dir         : str   = '.',
                 input_file     : str   = 'POSCAR',
                 output_file       : str   = 'result.efs',
                 input_writer     : Optional[Callable] = None,
                 output_reader    : Optional[Callable] = None,
                 mpi_prefix       : str   = '',
                 timeout          : Optional[float] = None,
                 stress_to_eV_Ang3: float = 1.0,
                 env              : Optional[dict] = None,
                 cache            : bool  = True,
                 cache_tol        : float = 1e-6,
                 r_cut            : float = 0.0,
                 verbose          : bool  = True):

        self.cmd               = cmd
        self.work_dir          = work_dir
        self.input_file      = input_file
        self.output_file        = output_file
        self.input_writer      = input_writer
        self.output_reader     = output_reader
        self.mpi_prefix        = mpi_prefix.strip()
        self.timeout           = timeout
        self.stress_to_eV_Ang3 = float(stress_to_eV_Ang3)
        self.env               = env
        self.cache             = cache
        self.cache_tol         = float(cache_tol)
        self.r_cut             = float(r_cut)
        self.verbose           = verbose

        # Call counter and cache
        self._n_calls    : int              = 0
        self._n_cache    : int              = 0
        self._cached_pos : Optional[np.ndarray]  = None
        self._cached_lat : Optional[np.ndarray]  = None
        self._cached_EFS : Optional[tuple]       = None   # (E, F, S)

    # ------------------------------------------------------------------ #
    #  Core compute method                                                 #
    # ------------------------------------------------------------------ #

    def compute(self,
                lattice    : np.ndarray,
                positions  : np.ndarray,
                atom_types : list,
                ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate energy, forces, and stress via the external code.

        Parameters
        ----------
        lattice    : (3, 3) lattice vectors [A].
        positions  : (N, 3) Cartesian positions [A].
        atom_types : list of element symbols, length N.

        Returns
        -------
        energy : float [eV]
        forces : (N, 3) array [eV/A]
        stress : (3, 3) symmetric array [eV/A^3]

        Raises
        ------
        RuntimeError  : if the external command fails (non-zero exit code).
        FileNotFoundError : if the EFS output file is not produced.
        ValueError    : if the EFS file has incorrect format or atom count.
        """
        lattice   = np.asarray(lattice,   dtype=float)
        positions = np.asarray(positions, dtype=float)
        N         = len(positions)

        # -- Cache check ---------------------------------------
        if self.cache and self._cached_EFS is not None:
            lat_match = np.allclose(lattice, self._cached_lat, atol=1e-12)
            if lat_match and self._cached_pos.shape == positions.shape:
                max_disp = float(np.max(np.abs(positions - self._cached_pos)))
                if max_disp < self.cache_tol:
                    self._n_cache += 1
                    if self.verbose:
                        print(f"  ExternalEngine: cache hit "
                              f"(call #{self._n_calls}, "
                              f"max_disp={max_disp:.2e} A)")
                    return self._cached_EFS

        self._n_calls += 1
        os.makedirs(self.work_dir, exist_ok=True)

        # -- Step 1: Write input structure ---------------------
        in_path = os.path.join(self.work_dir, self.input_file)
        if self.input_writer is not None:
            self.input_writer(in_path, lattice, positions, atom_types)
        else:
            write_poscar(in_path, lattice, positions, atom_types,
                         title=f"akitsushima external call #{self._n_calls}")

        # -- Step 2: Build and run shell command ---------------
        out_path = os.path.join(self.work_dir, self.output_file)
        full_cmd  = self._build_command(in_path, out_path)
        if self.verbose:
            print(f"\n  ExternalEngine call #{self._n_calls}: "
                  f"N={N}  cmd='{full_cmd}'")

        t0 = _time.perf_counter()
        result = self._run_command(full_cmd)
        elapsed = _time.perf_counter() - t0

        if result.returncode != 0:
            raise RuntimeError(
                f"ExternalFileEngine: command failed (exit {result.returncode}).\n"
                f"  Command : {full_cmd}\n"
                f"  work_dir: {self.work_dir}\n"
                f"  stdout  :\n{result.stdout[-2000:] if result.stdout else '(none)'}\n"
                f"  stderr  :\n{result.stderr[-2000:] if result.stderr else '(none)'}")

        # -- Step 3: Parse EFS output --------------------------
        if self.output_reader is not None:
            E, F, S = self.output_reader(out_path, N)
        else:
            E, F, S = read_efs_file(out_path, N,
                                    stress_to_eV_Ang3=self.stress_to_eV_Ang3)

        if self.verbose:
            fmax = float(np.max(np.abs(F))) if N > 0 else 0.0
            print(f"  -> E={E:.6f} eV  Fmax={fmax:.4f} eV/A  "
                  f"t={elapsed:.1f}s  (total calls={self._n_calls})")

        # -- Update cache --------------------------------------
        self._cached_pos = positions.copy()
        self._cached_lat = lattice.copy()
        self._cached_EFS = (E, F, S)

        return E, F, S

    # ------------------------------------------------------------------ #
    #  make_force_fn compatibility                                         #
    # ------------------------------------------------------------------ #

    def __call__(self,
                 lattice    : np.ndarray,
                 positions  : np.ndarray,
                 atom_types : list,
                 neigh_list = None,
                 ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Direct callable interface -- identical to :meth:`compute`.

        This allows passing the engine instance directly where a force
        function is expected, without going through :func:`make_force_fn`::

            fn = ExternalFileEngine(cmd='vasp_std', work_dir='run')
            go = GeometryOptimiser(lat, pos, types, force_fn=fn, r_cut=10.)
        """
        return self.compute(lattice, positions, atom_types)

    # ------------------------------------------------------------------ #
    #  Diagnostic properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def n_calls(self) -> int:
        """Total number of external code invocations."""
        return self._n_calls

    @property
    def n_cache_hits(self) -> int:
        """Number of times the cached result was returned."""
        return self._n_cache

    def summary(self) -> str:
        """Return a one-line diagnostic string."""
        return (f"ExternalFileEngine: {self._n_calls} calls, "
                f"{self._n_cache} cache hits, "
                f"cmd='{self.cmd}', work_dir='{self.work_dir}'")

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_command(self, in_path: str, out_path: str) -> str:
        """
        Construct the full shell command string.

        Substitutes ``{input}`` and ``{output}`` placeholders if present,
        then prepends the MPI prefix.
        """
        cmd = self.cmd
        if '{input}' in cmd or '{output}' in cmd:
            cmd = cmd.format(input=in_path, output=out_path)
        if self.mpi_prefix:
            cmd = f"{self.mpi_prefix} {cmd}"
        return cmd

    def _run_command(self, cmd: str):
        """
        Execute the shell command inside work_dir and return the result.

        Uses subprocess.run with shell=True so that pipes, redirections,
        and MPI wrappers work correctly.  The process is killed after
        ``self.timeout`` seconds if specified.
        """
        env = None
        if self.env:
            env = dict(os.environ)
            env.update(self.env)

        return _sp.run(
            cmd,
            shell   = True,
            cwd     = self.work_dir,
            capture_output = True,
            text    = True,
            timeout = self.timeout,
            env     = env,
        )


class ScaledForceEngine:
    """
    Force / stress scaling adapter for any force callable.

    Wraps a ``force_fn`` or any object with a
    ``(lattice, positions, atom_types) -> (energy, forces, stress)``
    interface and applies proportional scaling and optional hard clamping
    to the returned forces and stresses.  Energy is forwarded unchanged.

    Two-level ionic displacement control
    ----------------------------------------
    force_scale (c_f)
        Scalar multiplied to every force component::

            effective_force = c_f x F_raw

        Default 1.0 (no rescaling).
        Used together with ``max_displacement`` in :class:`GeometryOptimiser`
        for full two-level ionic control:

        - ``force_scale``     : proportional scaling of step amplitude
        - ``max_displacement``: absolute safety cap [A] per step

    Two-level cell / strain control
    ---------------------------------
    stress_scale (c_s)
        Scalar multiplied to every stress component::

            effective_stress = c_s x sigma_raw

        Reduces cell oscillations during variable-cell relaxation.
        Default 1.0 (no rescaling).

    max_strain
        Hard ceiling on the Voigt strain increment per optimisation step
        (dimensionless, > 0).

        **Physical meaning** -- the fractional change in a lattice parameter
        per step is bounded by ``max_strain``.  For example
        ``max_strain = 0.02`` means that no lattice vector changes by more
        than 2 % per step regardless of how large the stress is.

        **Derivation** -- the cell gradient produced by GeometryOptimiser is

        .. code-block:: text

            f_cell = -c_s . sigma . V . h^{-T}

        The Voigt strain estimate for row ``i`` (the gradient contribution
        associated with lattice vector ``i``, in this codebase's
        row-vectors convention: ``lattice[i, :]`` is lattice vector ``i``)
        is

        .. code-block:: text

            epsilon_ij ~= |f_cell[i, j]| / |h[i, :]|      (row-normalised)

        If ``max(|epsilon_ij|) > max_strain``, sigma is uniformly rescaled by
        ``max_strain / max(|epsilon_ij|)``.  The direction of sigma is preserved;
        only its magnitude is reduced.

        Default None (no clamping).

    Parameters
    ----------
    base_fn      : force callable.
    force_scale  : c_f >= 0.  Default 1.0.
    stress_scale : c_s >= 0.  Default 1.0.
    max_strain   : float or None.  Voigt strain ceiling per step.

    Attributes
    ----------
    r_cut        : forwarded from base_fn when present.

    Example
    -------
    >>> engine = ExternalFileEngine(cmd='vasp_std', work_dir='run', r_cut=10.)
    >>> fn = ScaledForceEngine(engine,
    ...          force_scale  = 0.3,
    ...          stress_scale = 0.1,
    ...          max_strain   = 0.02)
    >>> go = GeometryOptimiser(lat, pos, types, force_fn=fn,
    ...          max_displacement=0.05, relax_cell=True)
    >>> result = go.run()
    """

    def __init__(self,
                 base_fn,
                 force_scale  : float           = 1.0,
                 stress_scale : float           = 1.0,
                 max_strain   : Optional[float] = None):

        if float(force_scale) < 0.0:
            raise ValueError(
                f"ScaledForceEngine: force_scale must be >= 0, "
                f"got {force_scale}.")
        if float(stress_scale) < 0.0:
            raise ValueError(
                f"ScaledForceEngine: stress_scale must be >= 0, "
                f"got {stress_scale}.")
        if max_strain is not None and float(max_strain) <= 0.0:
            raise ValueError(
                f"ScaledForceEngine: max_strain must be > 0, "
                f"got {max_strain}.")

        self._base_fn     = base_fn
        self.force_scale  = float(force_scale)
        self.stress_scale = float(stress_scale)
        self.max_strain   = float(max_strain) if max_strain is not None else None

        if hasattr(base_fn, 'r_cut'):
            self.r_cut = base_fn.r_cut

    # ------------------------------------------------------------------ #

    def __call__(self,
                 lattice    : np.ndarray,
                 positions  : np.ndarray,
                 atom_types : list,
                 **kwargs,
                 ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Call the wrapped engine and apply force / stress scaling.

        Returns
        -------
        energy : float [eV]              -- unchanged
        forces : (N, 3) ndarray [eV/A]  -- multiplied by force_scale
        stress : (3, 3) ndarray [eV/A^3] -- multiplied by stress_scale,
                                           then uniformly rescaled so that
                                           max Voigt strain <= max_strain
        """
        energy, forces, stress = self._base_fn(lattice, positions, atom_types)

        if self.force_scale != 1.0:
            forces = np.asarray(forces, dtype=float) * self.force_scale

        stress = np.asarray(stress, dtype=float) * self.stress_scale

        if self.max_strain is not None:
            stress = self._clamp_by_strain(stress, lattice)

        return float(energy), forces, stress

    # ------------------------------------------------------------------ #

    def _clamp_by_strain(self,
                         stress  : np.ndarray,
                         lattice : np.ndarray,
                         ) -> np.ndarray:
        """
        Uniformly rescale *stress* so that the maximum Voigt strain
        increment implied by the cell gradient does not exceed
        ``self.max_strain``.

        The cell gradient is exactly the ``f_cell`` that
        :class:`~akitsushima.solvers.geometry_optimiser.GeometryOptimiser`
        itself computes for its default ``'generalized_force'``
        cell-relaxation scheme -- see
        :func:`~akitsushima.engine.cell_gradient.cell_force_from_stress`,
        the single shared implementation both classes now call. The
        Voigt strain estimate is

            epsilon_ij = |f_cell[i, j]| / |h[i, :]|

        If ``max(epsilon_ij) > max_strain``, sigma is rescaled by the factor
        ``max_strain / max(epsilon_ij)``, preserving the direction of sigma.

        Bug-fix note (F10): the original normalised by
        ``np.linalg.norm(h, axis=0)`` (column norms) broadcast across
        columns.  This codebase's lattice convention is row-vectors
        (``lattice[i, :]`` is lattice vector ``i`` -- see e.g.
        ``parse_poscar``/``write_poscar`` and every other stress/strain
        routine in ``engine/``), so the length of lattice vector ``i``
        is ``norm(h[i, :])`` (an ``axis=1`` row norm), not
        ``norm(h[:, j])``.  For a general (non-orthogonal) cell the two
        differ, so the strain estimate -- and hence the clamp threshold
        -- was wrong for triclinic/monoclinic/hexagonal cells (it
        happened to coincide for a diagonal/orthogonal ``h``).  Fixed by
        normalising row ``i`` of ``f_cell`` by the length of lattice
        vector ``i``.

        Bug-fix note (Bug C, cell-force formula): this method used to
        compute its own, independent copy of the cell-force formula as
        ``-(stress * vol) @ h_inv_T``, i.e. ``-vol*(stress @ h_inv_T)``.
        Direct numerical finite-difference verification (see
        ``cell_gradient.py``'s module docstring) showed this disagreed
        with the true ``f_cell`` by max|diff|=500.0 for a non-orthogonal
        test cell -- both the matrix-multiplication order AND the sign
        were wrong (worse than the closely-related bug independently
        found in ``GeometryOptimiser._generalised_force``, which had the
        same wrong order but the correct sign). Since this method's
        entire purpose is to predict what ``GeometryOptimiser`` will do
        with this stress, it must use the identical formula, not a
        re-derived one -- fixed by calling the shared
        ``cell_force_from_stress`` helper directly.
        """
        h   = np.asarray(lattice, dtype=float)
        vol = float(np.abs(np.linalg.det(h)))
        if vol < 1e-30:
            return stress                        # degenerate cell -- skip

        f_cell = cell_force_from_stress(stress, h)

        # Row norms: length of each lattice vector  |h[i, :]|
        row_norms = np.linalg.norm(h, axis=1)    # (3,)
        row_norms = np.where(row_norms > 1e-12, row_norms, 1.0)

        # Voigt strain estimate: epsilon_ij = |f_cell[i,j]| / |h[i,:]|
        strain  = np.abs(f_cell) / row_norms[:, None]   # (3, 3)
        max_eps = float(strain.max())

        if max_eps > self.max_strain:
            stress = stress * (self.max_strain / max_eps)

        return stress

    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        base = (self._base_fn.summary()
                if hasattr(self._base_fn, 'summary') else repr(self._base_fn))
        ms = f"{self.max_strain}" if self.max_strain is not None else "none"
        return (f"ScaledForceEngine("
                f"c_f={self.force_scale}, c_s={self.stress_scale}, "
                f"max_strain={ms})  wrapping: {base}")
