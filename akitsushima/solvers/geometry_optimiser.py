"""Geometry optimisation: FIRE / L-BFGS / DIIS / RMM-DIIS / RFO / steepest
descent, with optional variable-cell (Parrinello-Rahman-style) relaxation.

FIRE (Fast Inertial Relaxation Engine) reference:
  Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006).

Cell relaxation: the supercell matrix h and atomic positions are jointly
optimised by treating the stress as the gradient of the enthalpy with
respect to the strain tensor.
"""
import inspect
import os
from collections import deque
from typing import Callable, List, Optional

import numpy as np

from ..constants import EV_ANG3_TO_GPA
from ..engine.cell_gradient import cell_force_from_stress
from ..engine.neighborlist import (NeighbourList,
                                   wrap_positions_into_cell)
from ..fitting.force_fn import _resolve_force_fn
from ..io.poscar import write_poscar
from ..io.xdatcar import _write_xdatcar_frame, _write_xdatcar_header
from ..utils.mpi import _mpi_rank
from ..utils.timing import TimingCollector

__all__ = ["GeometryOptimiser"]


def _accepts_neigh_list(fn) -> bool:
    """True when ``fn`` can be called with a ``neigh_list=`` keyword.

    Every closure make_force_fn returns, and every potential's compute(),
    takes one.  ESMCalculator.__call__ does not -- it resolves periodic
    images through its own Green-function machinery, so it must be called
    with the three positional arguments only.  Anything accepting **kwargs
    is assumed to take it.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):       # C-implemented or unintrospectable
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "neigh_list" in params


class GeometryOptimiser:
    """
    Geometry optimiser with five interchangeable algorithms.

    Algorithms
    ----------
    ``'fire'``     (default)
        Fast Inertial Relaxation Engine -- velocity-Verlet with adaptive
        time step and direction quench.  Very robust for any starting point.

    ``'lbfgs'``
        Limited-memory BFGS (Nocedal 1980).  Quasi-Newton with implicit
        inverse-Hessian approximation from the last *memory* (s,y) pairs.
        Typically 2 to 5x fewer steps than FIRE.

    ``'diis'``
        Direct Inversion in the Iterative Subspace (Pulay 1980).
        Solves a Lagrangian system over the last *memory* residuals.
        Best used for polishing when the force is already small.

    ``'rmmdiis'``
        Residual Minimisation Method DIIS (Kresse & Furthmuller 1996).
        Projects the residual onto the DIIS subspace and applies a line
        search.  More stable than DIIS for ill-conditioned starting points.

    ``'rfo'``
        Rational Function Optimisation (Simons et al. 1983; Baker 1986).
        Solves the augmented Hessian eigenproblem.  With ``uphill=True``
        implements Partitioned RFO for saddle-point (TS) refinement.

    relax_cell = False  -> ionic positions only (fixed cell shape / volume)
    relax_cell = True   -> ions + cell jointly (variable-cell / NPT-like)

    Two-level displacement control with :class:`ScaledForceEngine`
    ---------------------------------------------------------------
    When calling an expensive external code (DFT, hybrid functional),
    combine ``ScaledForceEngine`` with ``max_displacement`` for precise
    step-size control::

        from akitsushima.fitting import ExternalFileEngine, ScaledForceEngine
        engine = ExternalFileEngine(cmd='vasp_std', work_dir='run',
                                    r_cut=10.)
        fn = ScaledForceEngine(engine,
                 force_scale      = 0.3,   # proportional scaling of delta_r
                 stress_scale     = 0.1,   # proportional scaling of delta_h
                 max_strain       = 0.02)  # hard cap 2 % per step
        go = GeometryOptimiser(lat, pos, types, force_fn=fn,
                 max_displacement = 0.05,  # hard cap 0.05 A per step
                 relax_cell       = True)

    Constraint syntax (VASP Selective Dynamics equivalent)
    -------------------------------------------------------
    fix_atoms : (N, 3) bool array -- True = frozen.
        Example -- fix all coordinates of atom 0::

            fix = np.zeros((N, 3), dtype=bool); fix[0] = True

    fix_cell  : (3, 3) bool array -- True = frozen (relax_cell=True only).
        Example -- fix the c-axis length (lattice[2, 2])::

            fc = np.zeros((3, 3), dtype=bool); fc[2, 2] = True

    Combining both::

        struct = parse_poscar("POSCAR")     # fix_atoms from Selective Dynamics
        go = GeometryOptimiser(
            struct["lattice"], struct["positions"], struct["atom_types"],
            fix_atoms = struct["fix_atoms"],
            force_fn  = fn,
        )

    Output files (written to *output_dir*)
    ----------------------------------------
    relax.log        -- step / energy / Fmax / Pmax / algorithm-specific metric
    POSCAR_relaxed   -- final structure in VASP 5 format
    """

    # -- FIRE constants ------------------------------------
    _DT_START    = 0.1    # fs  -- initial time step
    _DT_MAX      = 1.0    # fs
    _N_MIN       = 5      # steps before dt can grow
    _F_INC       = 1.1
    _F_DEC       = 0.5
    _ALPHA_START = 0.1
    _F_ALPHA     = 0.99
    # -- common displacement limit -------------------------
    _MAX_STEP    = 0.2    # A  -- max ionic displacement per step
    # -- DIIS extrapolation-coefficient bound ----------------
    # If any Pulay coefficient exceeds this magnitude, the linear system
    # is judged too ill-conditioned to trust (signals near-collinear
    # history vectors); fall back to a steepest-descent step instead.
    _DIIS_C_MAX  = 10.0

    def __init__(self,
                 lattice         : np.ndarray,
                 positions       : np.ndarray,
                 atom_types      : list,
                 # -- universal interface (A) ---------------------------
                 force_fn        : Optional[Callable] = None,
                 r_cut           : Optional[float] = None,
                 # -- legacy Buckingham kwargs (B) ----------------------
                 buck_params     : Optional[dict]  = None,
                 charges         : Optional[dict]  = None,
                 ewald_alpha     : Optional[float] = None,
                 coulomb_method  : str   = "ewald",
                 pppm_accuracy   : float = 1e-5,
                 pppm_mesh       : Optional[int] = None,
                 pppm_order      : int   = 6,
                 # -- optimiser settings --------------------------------
                 method          : str   = "fire",
                 relax_cell      : bool  = False,
                 # -- convergence criteria (on/off + thresholds) ---------
                 check_forces    : bool  = True,
                 check_stress    : Optional[bool] = None,
                 check_energy    : bool  = False,
                 fmax            : float = 0.01,
                 pmax            : float = 0.1,
                 emax            : float = 1e-6,
                 max_steps       : int   = 2000,
                 max_displacement: float = 0.2,
                 save_best       : bool  = False,
                 dt_start        : Optional[float] = None,
                 dt_max          : Optional[float] = None,
                 # -- cell relaxation control (relax_cell=True only) ----
                 cell_update_scheme : str = "generalized_force",
                 lattice_constraints : Optional[object] = None,
                 # -- real-time output control ---------------------------
                 flush_every_step : bool = True,
                 fsync_every_step : bool = False,
                 # -- L-BFGS / DIIS / RMM-DIIS history size -------------
                 memory          : int   = 20,
                 # -- RFO-specific settings -----------------------------
                 uphill          : bool  = False,
                 hessian_delta   : float = 0.01,
                 trust_radius    : float = 0.3,
                 # -- output --------------------------------------------
                 output_dir      : str   = ".",
                 skin_width      : float = 0.5,
                 verbose         : bool  = True,
                 # -- constraints ---------------------------------------
                 fix_atoms       : Optional[np.ndarray] = None,
                 fix_cell        : Optional[np.ndarray] = None,
                 # -- timing --------------------------------------------
                 timing          : bool  = False):

        self.lattice     = lattice.copy()
        self.positions   = positions.copy()
        self.atom_types  = list(atom_types)
        self.N           = len(atom_types)
        self.method      = method.lower()
        self.relax_cell  = relax_cell
        # -- Convergence criteria ------------------------------------
        # Each criterion can be independently enabled/disabled.
        # When relax_cell=True, check_stress defaults to True unless the
        # user explicitly set it (handled by the caller passing False);
        # this preserves the historical default behaviour (forces AND
        # stress must both converge whenever the cell is relaxed) while
        # making it possible to opt out (e.g. converge on forces only,
        # accepting a non-zero residual stress, as is appropriate when
        # the E(V) curve is very flat near its minimum).
        self.check_forces = bool(check_forces)
        # check_stress=None (default) -> follow legacy behaviour:
        #   stress convergence is required automatically when relax_cell
        #   is True, and skipped when relax_cell is False (there is no
        #   stress feedback on a fixed cell anyway).
        # check_stress=True/False -> explicit user override, always honoured.
        self.check_stress = (relax_cell if check_stress is None
                             else bool(check_stress))
        self.check_energy = bool(check_energy)
        if not (self.check_forces or self.check_stress or self.check_energy):
            print("  WARNING: GeometryOptimiser -- all convergence criteria "
                  "(check_forces, check_stress, check_energy) are disabled. "
                  "The optimiser will run for the full max_steps regardless "
                  "of how well-converged the structure is.")
        self.fmax        = float(fmax)
        self.pmax        = float(pmax)     # GPa
        self.emax        = float(emax)     # eV -- |E_new - E_prev| threshold
        self.max_steps   = max_steps
        self.save_best   = bool(save_best)
        # FIRE timestep overrides (None = use class-level defaults)
        # Stored as instance attributes so _run_fire picks them up.
        if dt_start is not None:
            self._DT_START = float(dt_start)
        if dt_max is not None:
            self._DT_MAX = float(dt_max)

        # -- Cell update scheme (relax_cell=True only) -----------------
        # 'generalized_force' (default): physically-derived Parrinello-
        #   Rahman-style force, f_cell = +volume * (h^-T @ stress).
        #   Mind the operand order: h^-T LEFT-multiplies stress. The
        #   right-multiplied ``stress @ h^-T`` is the historical S1
        #   defect (finite differences: max|diff| = 17.0 for it, vs
        #   1.24e-7 for the correct order) -- do not "simplify" it back.
        #   This is the exact gradient dE/dh (with correct sign, see
        #   _generalised_force docstring) but can amplify small stress
        #   asymmetries into large cell-shape changes for strongly
        #   non-orthogonal (e.g. hexagonal, monoclinic) cells, since h^-T
        #   depends sensitively on the current cell shape.
        # 'vasp': VASP-style direct update, f_cell = stress @ lattice
        #   (no lattice-inverse factor). This mirrors the update formula
        #   used in VASP's dyna.F (A_new = A_old + step*(stress @ A_old)),
        #   which tends to be numerically gentler / less prone to
        #   amplifying stress-tensor noise into large cell distortions,
        #   at the cost of no longer being an exact energy gradient (it
        #   is an empirically well-behaved update rule rather than a
        #   strict dE/dh).
        cell_update_scheme = str(cell_update_scheme).lower()
        if cell_update_scheme not in ("generalized_force", "vasp"):
            raise ValueError(
                f"GeometryOptimiser: unknown cell_update_scheme "
                f"'{cell_update_scheme}'. Choose 'generalized_force' or 'vasp'.")
        self.cell_update_scheme = cell_update_scheme

        # -- Lattice component constraints (QE cell_dofree-style, but ---
        # -- generalized to all 9 raw lattice-matrix components) --------
        # 9 independently controllable components, in row-major order
        # matching the lattice matrix (rows = lattice vectors):
        #   [a1x, a1y, a1z, a2x, a2y, a2z, a3x, a3y, a3z]
        # Each is 1/True (free to relax) or 0/False (frozen). Accepts a
        # string ("1 0 0 0 1 0 0 0 1"), a list/tuple of 9 values, or
        # None (all free, the default).
        #
        # Examples:
        #   "1 0 0 0 1 0 0 0 1"  -- diagonal only (a1x, a2y, a3z free;
        #                           all off-diagonal components frozen).
        #   "1 1 1 0 0 0 0 0 0"  -- only lattice vector a1 is free to
        #                           move (in any direction); a2 and a3
        #                           are completely frozen.
        #   "1 1 1 1 1 1 1 1 1"  -- fully unconstrained (equivalent to
        #                           the default None).
        #
        # This mask is applied DIRECTLY to the raw (unsymmetrized) 9
        # components of the cell force, in the SAME order as the
        # lattice matrix's own row-major layout, so it exactly matches
        # the semantics the components are named after (a1x is truly
        # the x-component of lattice vector a1, etc.) -- unlike a
        # strain-based (Voigt, 6-component) formulation, this allows
        # constraints that are NOT expressible as a symmetric strain
        # (e.g. freezing a2 and a3 while leaving a1 fully free).
        #
        # NOTE: because this mask acts on RAW lattice components rather
        # than on a symmetrized strain tensor, it does not by itself
        # guarantee suppression of the spurious cell-rotation drift
        # described elsewhere in this file (an unsymmetrized cell force
        # can, over long relaxations, slowly rotate the whole cell
        # relative to the lab frame even though the physical shape/
        # volume converges normally). If you do not need the extra
        # flexibility of independent raw-component control and want the
        # older automatic anti-rotation-drift behaviour, keep this at
        # its default (None, fully free) rather than supplying a
        # constraint pattern that only touches diagonal components.
        self._lattice_constraints_mask = \
            self._parse_lattice_constraints(lattice_constraints)

        # -- Real-time output control -----------------------------------
        # flush_every_step=True (default): after writing each step's row
        #   to relax.log and each frame to XDATCAR_relax, call
        #   fh.flush() to push the data from Python's internal buffer to
        #   the OS. This makes the current progress visible to other
        #   processes (e.g. `tail -f relax.log` while the job is still
        #   running) almost immediately, at a very small performance
        #   cost (a flush() call is cheap -- it does not wait for the
        #   physical storage device, only for the OS to accept the
        #   data). Without this, the OS's own buffering (typically a
        #   few KB) means the file may appear empty or stale until
        #   enough data accumulates or the file is closed at the end of
        #   the run.
        # fsync_every_step=False (default): if True, additionally call
        #   os.fsync() after each flush, forcing the OS to physically
        #   write the data to the storage device before continuing.
        #   This gives a much stronger durability guarantee (the data
        #   survives a power loss or OS crash, not just an application
        #   crash), but is SIGNIFICANTLY slower -- especially on
        #   network filesystems commonly used on HPC clusters (NFS,
        #   Lustre, GPFS), where a single fsync() can take tens to
        #   hundreds of milliseconds. For a fast force field (e.g.
        #   Numba-JIT-accelerated) where a single optimisation step
        #   might otherwise take only a few milliseconds, enabling
        #   fsync_every_step can make I/O the dominant cost and slow
        #   the overall run down by a large factor. Leave this False
        #   unless you specifically need crash-durable logs (e.g. for
        #   audit/compliance reasons) rather than just live monitoring.
        self.flush_every_step = bool(flush_every_step)
        self.fsync_every_step = bool(fsync_every_step)

        # Running best-energy tracker (populated during run() if save_best=True)
        self._best_e        = float('inf')
        self._best_pos      = None
        self._best_lat      = None
        self._best_step     = -1
        self._MAX_STEP   = float(max_displacement)   # A -- overrides class default
        self.memory      = int(memory)
        self.uphill      = uphill
        self.hessian_delta = float(hessian_delta)
        self.trust_radius  = float(trust_radius)
        self.output_dir  = output_dir
        self.verbose     = verbose

        self._force_fn, r_cut = _resolve_force_fn(
            force_fn, r_cut, buck_params, charges, ewald_alpha,
            coulomb_method, pppm_accuracy, pppm_mesh, pppm_order,
            "GeometryOptimiser")
        self.r_cut = float(r_cut)

        # Verlet neighbour list.  This matters for more than speed:
        # NeighbourList.build() is the only place the minimum-image
        # precondition (r_cut + skin_width < half the smallest perpendicular
        # cell width) is actually enforced, and check_and_rebuild() re-runs
        # that check on every lattice change -- so a variable-cell relax
        # that shrinks the cell past the limit now raises instead of
        # silently returning wrong energies, forces and stresses.
        #
        # A force function that does not take a neigh_list argument
        # (ESMCalculator is the one in this package) is driven without one:
        # it does its own spatial bookkeeping, and imposing this list's
        # precondition on it would reject cells it handles correctly.
        self._nl = (NeighbourList(r_cut=self.r_cut, skin_width=skin_width)
                    if _accepts_neigh_list(self._force_fn) else None)

        if self.method not in ('fire', 'lbfgs', 'diis', 'rmmdiis', 'rfo', 'sd'):
            raise ValueError(
                f"GeometryOptimiser: unknown method '{method}'. "
                "Choose from 'fire', 'lbfgs', 'diis', 'rmmdiis', 'rfo', 'sd'.")

        # -- Constraint masks ----------------------------------
        #
        # fix_atoms : (N, 3) bool array
        #   True  = that Cartesian DoF is frozen (never displaced).
        #   Equivalent to VASP Selective Dynamics flag 'F'.
        #   Example -- fix all three coordinates of atom 0:
        #     fix = np.zeros((N, 3), dtype=bool); fix[0] = True
        #
        # fix_cell  : (3, 3) bool array
        #   True  = that lattice-vector component is frozen.
        #   Only meaningful when relax_cell=True.
        #   Components map to lattice[i, alpha]:
        #     fix_cell[0] -> a-vector (a_x, a_y, a_z)
        #     fix_cell[1] -> b-vector (b_x, b_y, b_z)
        #     fix_cell[2] -> c-vector (c_x, c_y, c_z)
        #   Example -- fix c-axis length (lattice[2, 2]):
        #     fc = np.zeros((3, 3), dtype=bool); fc[2, 2] = True
        #
        # Both are stored as flat 1-D bool masks aligned with the
        # generalised coordinate vector [ ions(3N) | cell(9) ].

        N = self.N
        if fix_atoms is not None:
            fa = np.asarray(fix_atoms, dtype=bool)
            if fa.shape != (N, 3):
                raise ValueError(
                    f"GeometryOptimiser: fix_atoms must have shape ({N}, 3), "
                    f"got {fa.shape}.")
            self._fix_ions = fa.ravel()          # (3N,) bool
        else:
            self._fix_ions = np.zeros(N * 3, dtype=bool)

        if fix_cell is not None:
            fc = np.asarray(fix_cell, dtype=bool)
            if fc.shape != (3, 3):
                raise ValueError(
                    f"GeometryOptimiser: fix_cell must have shape (3, 3), "
                    f"got {fc.shape}.")
            self._fix_cell_mask = fc.ravel()     # (9,) bool
        else:
            self._fix_cell_mask = np.zeros(9, dtype=bool)

        # Rebuild generalised free-DoF mask whenever relax_cell changes
        self._rebuild_free_mask()

        # Timing
        self._tc = TimingCollector(enabled=timing)

    # ------------------------------------------------------------------ #
    #  Force evaluation helpers                                            #
    # ------------------------------------------------------------------ #

    def _get_forces_and_stress(self):
        if self._nl is None:
            return self._force_fn(self.lattice, self.positions,
                                  self.atom_types)
        self._nl.check_and_rebuild(self.lattice, self.positions)
        return self._force_fn(self.lattice, self.positions, self.atom_types,
                              neigh_list=self._nl)

    # ------------------------------------------------------------------ #
    #  Constraint helpers                                                  #
    # ------------------------------------------------------------------ #

    def _rebuild_free_mask(self) -> None:
        """Rebuild the flat free-DoF mask from fix_ions / fix_cell_mask."""
        if self.relax_cell:
            self._free_mask = np.concatenate(
                [~self._fix_ions, ~self._fix_cell_mask])  # (3N+9,)
        else:
            self._free_mask = ~self._fix_ions

    @staticmethod
    def _parse_lattice_constraints(lattice_constraints) -> np.ndarray:
        """
        Parse a lattice_constraints specification into a length-9 bool
        mask, in row-major order matching the lattice matrix (rows =
        lattice vectors):
            [a1x, a1y, a1z, a2x, a2y, a2z, a3x, a3y, a3z]

        Accepted formats
        ----------------
        None                          -> all 9 components free (default).
        "1 0 0 0 1 0 0 0 1"           -> space-separated string of 0/1.
        [1, 0, 0, 0, 1, 0, 0, 0, 1]   -> list/tuple/array of 0/1 or bool.

        Examples
        --------
        "1 0 0 0 1 0 0 0 1"  -- diagonal components only (a1x, a2y, a3z
                                free; all off-diagonal components frozen).
        "1 1 1 0 0 0 0 0 0"  -- only lattice vector a1 is free to move
                                (in any direction); a2 and a3 completely
                                frozen.
        "1 1 1 1 1 1 1 1 1"  -- fully unconstrained (equivalent to None).
        """
        if lattice_constraints is None:
            return np.ones(9, dtype=bool)
        if isinstance(lattice_constraints, str):
            parts = lattice_constraints.split()
        else:
            parts = list(lattice_constraints)
        if len(parts) != 9:
            raise ValueError(
                "lattice_constraints must specify exactly 9 components "
                "(a1x a1y a1z a2x a2y a2z a3x a3y a3z), "
                "e.g. '1 0 0 0 1 0 0 0 1'; "
                f"got {len(parts)} value(s): {lattice_constraints!r}")
        return np.array([bool(int(v)) for v in parts], dtype=bool)

    def _generalised_force(self, forces, stress):
        """
        Build a generalised force vector [f_ions | f_cell].

        f_cell is computed by one of two schemes (see cell_update_scheme):

        'generalized_force' (default): the force conjugate to the lattice
        matrix h (rows = lattice vectors, matching parse_poscar's
        convention), derived from dE/dh = -volume * h^-T @ stress
        (Parrinello-Rahman-style cell dynamics), so the force for a
        gradient-descent-style minimiser is f_cell = -dE/dh
        = +volume * (h^-T @ stress). See
        :func:`~akitsushima.engine.cell_gradient.cell_force_from_stress`
        for the single, shared, numerically-verified implementation of
        this formula (also used by
        :class:`~akitsushima.fitting.external.ScaledForceEngine`).

        'vasp': a direct, VASP-style update f_cell = stress @ lattice
        (no lattice-inverse factor), mirroring the update rule used in
        VASP's own source (dyna.F: A_new = A_old + step*(stress @ A_old)).
        This is not a strict energy gradient but an empirically
        well-behaved update that avoids the h^-T amplification factor,
        which can behave poorly for strongly non-orthogonal cells.

        In both cases, f_cell's 9 raw components [a1x, a1y, a1z, a2x,
        a2y, a2z, a3x, a3y, a3z] (row-major, matching the lattice
        matrix's own layout) are then masked by lattice_constraints:
        any component disabled there is zeroed out directly, with no
        symmetrization step. This gives independent control over each
        raw lattice-matrix component (see _parse_lattice_constraints for
        examples), at the cost of no longer automatically guaranteeing
        suppression of spurious cell-rotation drift the way a
        symmetric-strain-only formulation would (see the notes in
        __init__ next to lattice_constraints).

        Bug-fix note (S1, historical): the 'generalized_force' branch
        used to compute ``stress * vol * h_inv_T`` -- an ELEMENTWISE
        product, not a matrix product at all. S1's fix replaced this
        with a matrix product, but picked the wrong operand order,
        ``vol * (stress @ h_inv_T)`` (h_inv_T on the right) -- this
        coincides with the truly correct ``vol * (h_inv_T @ stress)``
        (h_inv_T on the LEFT) only on the diagonal entries of a diagonal
        (orthogonal-cell) lattice with a diagonal stress tensor, since
        matrix multiplication of two diagonal matrices commutes; for any
        off-diagonal (shear) stress component, or any non-orthogonal
        cell, the two orderings diverge. This second-level bug was
        confirmed via direct finite-difference verification: perturbing
        each of the 9 raw lattice-matrix components independently for a
        real LJ potential on a deliberately non-orthogonal, low-symmetry
        cell and comparing the resulting numerical dE/dh against both
        orderings -- ``stress @ h_inv_T`` disagreed with -dE/dh by
        max|diff|=17.0 (not a rounding-level mismatch), while
        ``h_inv_T @ stress`` agreed to 1.24e-7 (finite-difference
        precision). The existing S1 regression test had computed its own
        "expected" value using the same wrong ordering as the production
        code, so it passed by construction rather than by an independent
        check -- see test_solvers.py for the corrected, finite-difference
        oracle version of that test.

        Frozen degrees of freedom (fix_cell / fix_atoms) are zeroed so
        the optimiser never generates a displacement for them.
        """
        if not self.relax_cell:
            g = forces.ravel().copy()
        else:
            if self.cell_update_scheme == "vasp":
                f_cell_raw = stress @ self.lattice
            else:
                f_cell_raw = cell_force_from_stress(stress, self.lattice)

            # Apply the lattice_constraints mask directly to the 9 raw
            # components, in the same row-major order as the lattice
            # matrix itself (a1x, a1y, a1z, a2x, a2y, a2z, a3x, a3y, a3z).
            f_cell = (f_cell_raw.ravel() * self._lattice_constraints_mask
                     ).reshape(3, 3)
            g = np.concatenate([forces.ravel(), f_cell.ravel()])

        g[~self._free_mask] = 0.0
        return g

    def _apply_generalised_displacement(self, g_disp):
        """
        Unpack generalised displacement into ionic + cell parts.

        Frozen DoF components are zeroed before application (defence
        against numerical noise in the optimiser step).
        """
        g_disp = g_disp.copy()
        g_disp[~self._free_mask] = 0.0

        n_ion = self.N * 3
        self.positions += g_disp[:n_ion].reshape(self.N, 3)
        if self.relax_cell:
            d_cell = g_disp[n_ion:].reshape(3, 3)
            frac   = self.positions @ np.linalg.inv(self.lattice)
            self.lattice   += d_cell
            self.positions  = frac @ self.lattice
        self.positions = wrap_positions_into_cell(self.positions,
                                                  self.lattice)

    def _clamp_step(self, step: np.ndarray) -> np.ndarray:
        """
        Clamp the generalised displacement so that:
          - no atom moves more than _MAX_STEP Angstrom, AND
          - no cell-vector component changes by more than _MAX_STEP Angstrom.

        Both ionic and cell sub-vectors are scaled by the SAME factor so the
        search direction is preserved; clamping only the ionic part (as in
        earlier versions) left the cell part unbounded, which could cause a
        single step to multiply the cell volume several-fold and collapse
        all interatomic distances beyond the potential cutoff (E -> 0).
        """
        n_ion = self.N * 3
        d_max_ion = np.max(np.abs(step[:n_ion])) if n_ion > 0 else 0.0
        d_max_cell = (np.max(np.abs(step[n_ion:]))
                     if self.relax_cell and step.size > n_ion else 0.0)
        d_max = max(d_max_ion, d_max_cell)
        if d_max > self._MAX_STEP:
            step = step * (self._MAX_STEP / d_max)
        return step

    def _get_x(self) -> np.ndarray:
        """Flatten current positions (+ cell) into a 1-D vector."""
        if self.relax_cell:
            return np.concatenate([self.positions.ravel(),
                                   self.lattice.ravel()])
        return self.positions.ravel().copy()

    def _set_x(self, x: np.ndarray) -> None:
        """Unpack a 1-D coordinate vector back into positions (+ cell)."""
        n_ion = self.N * 3
        self.positions = x[:n_ion].reshape(self.N, 3).copy()
        if self.relax_cell:
            frac = self.positions @ np.linalg.inv(self.lattice)
            self.lattice   = x[n_ion:].reshape(3, 3).copy()
            self.positions = frac @ self.lattice
        self.positions = wrap_positions_into_cell(self.positions,
                                                  self.lattice)

    # ------------------------------------------------------------------ #
    #  Common run entry-point                                              #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """
        Execute the geometry optimisation.

        Returns
        -------
        dict with keys: energy, lattice, positions, atom_types, forces,
        converged, n_steps.
        """
        dispatch = {
            'fire'    : self._run_fire,
            'lbfgs'   : self._run_lbfgs,
            'diis'    : self._run_diis,
            'rmmdiis' : self._run_rmmdiis,
            'rfo'     : self._run_rfo,
            'sd'      : self._run_sd,
        }
        self._tc.start("total")
        result = dispatch[self.method]()
        self._tc.stop("total")
        timing_summary = self._tc.summary()
        if timing_summary.get("enabled") and _mpi_rank == 0:
            self._tc.print_summary(label=f"GeometryOptimiser ({self.method.upper()})")
        result["timing"] = timing_summary
        return result

    # ------------------------------------------------------------------ #
    #  Shared logging / finalisation                                       #
    # ------------------------------------------------------------------ #

    def _open_log(self, extra_col: str = "metric"):
        os.makedirs(self.output_dir, exist_ok=True)
        log_path = os.path.join(self.output_dir, "relax.log")
        header = (f"{'Step':>6}  {'Energy(eV)':>16}  "
                  f"{'Fmax(eV/A)':>12}  {'Pmax(GPa)':>10}  "
                  f"{extra_col:>12}")
        if self.verbose:
            print(f"\n  GeometryOptimiser [{self.method.upper()}]")
            print("\n" + header)
            print("  " + "-" * len(header))

        # -- XDATCAR_relax trajectory (VASP XDATCAR-style) --------------
        # relax_cell=True  : cell can change every step, so (like an NPT
        #                     MD run) a full header (comment, scale,
        #                     lattice, species, counts) is written before
        #                     EVERY step's coordinates.
        # relax_cell=False : cell is fixed throughout, so (like an NVT
        #                     MD run) the header is written ONCE here,
        #                     before the optimisation loop starts, and
        #                     each step only appends a
        #                     "Direct configuration=" block.
        xdatcar_path = os.path.join(self.output_dir, "XDATCAR_relax")
        self._xdatcar_fh = open(xdatcar_path, "w")
        self._xdatcar_frame_idx = 1
        # Bug-fix note (code-check follow-up): the S9 fix wraps every
        # _run_* method's LOOP BODY in try/finally to guarantee
        # self._xdatcar_fh / log_fh get closed on an exception -- but that
        # try/finally only starts after this method already returns, so a
        # failure in between opening self._xdatcar_fh (above) and this
        # method's own return (the XDATCAR header write, or the log file
        # open below) was still outside any cleanup and would leak the
        # XDATCAR handle. Narrow window (would need e.g. a malformed
        # lattice/atom_types or the log file open itself failing, such as
        # a full disk or a permissions error) but cheap to close off: wrap
        # the remaining steps in their own try/except that closes
        # self._xdatcar_fh and re-raises on any failure.
        try:
            if not self.relax_cell:
                _write_xdatcar_header(
                    self._xdatcar_fh, self.lattice, self.atom_types,
                    comment=f"akitsushima.py relax [{self.method.upper()}]")
            log_fh = open(log_path, "w")
        except Exception:
            self._xdatcar_fh.close()
            raise

        return log_fh, header, log_path

    def _log_row(self, log_fh, step, e, forces, stress, extra,
                 GPa=EV_ANG3_TO_GPA):
        fmax_val = float(np.max(np.abs(forces)))
        pmax_val = float(np.max(np.abs(stress))) * GPa
        row = (f"{step:6d}  {float(e):16.6f}  "
               f"{fmax_val:12.4f}  {pmax_val:10.4f}  {str(extra):>12}")
        log_fh.write(row + "\n")
        if self.verbose and step % 10 == 0:
            print("  " + row)

        # Append this step's configuration to XDATCAR_relax.
        comment = (f"akitsushima.py relax [{self.method.upper()}]  "
                  f"step={step}  E={float(e):.6f}eV  "
                  f"Fmax={fmax_val:.4f}eV/A  Pmax={pmax_val:.4f}GPa")
        _write_xdatcar_frame(
            self._xdatcar_fh, self.positions, self.lattice, self.atom_types,
            self._xdatcar_frame_idx, variable_header=self.relax_cell,
            comment=comment)
        self._xdatcar_frame_idx += 1

        # Push this step's output to disk immediately so it can be
        # monitored in real time (e.g. `tail -f relax.log`) during a
        # long-running calculation.  See the flush_every_step /
        # fsync_every_step docstring in __init__ for the tradeoffs.
        if self.flush_every_step:
            log_fh.flush()
            self._xdatcar_fh.flush()
            if self.fsync_every_step:
                os.fsync(log_fh.fileno())
                os.fsync(self._xdatcar_fh.fileno())

        return fmax_val, pmax_val

    def _check_converged(self, fmax_val, stress, e_curr=None, e_prev=None):
        """
        Evaluate convergence against the enabled criteria.

        Each of the three criteria -- forces, stress, energy -- can be
        independently switched on/off via check_forces / check_stress /
        check_energy, with thresholds fmax (eV/A), pmax (GPa), and
        emax (eV, |E_new - E_prev|).

        ALL enabled criteria must be satisfied simultaneously for the
        step to be considered converged (logical AND). A criterion that
        is disabled is treated as automatically satisfied and does not
        block convergence.

        Energy convergence requires both e_curr and e_prev; if either is
        None (e.g. on the very first iteration, before a previous energy
        exists), the energy criterion is treated as NOT satisfied so that
        the optimiser always takes at least one step before it can stop
        on energy alone.
        """
        GPa = EV_ANG3_TO_GPA
        ok_forces = (not self.check_forces) or (fmax_val < self.fmax)
        ok_stress = (not self.check_stress) or (
            float(np.max(np.abs(stress))) * GPa < self.pmax)
        if self.check_energy:
            ok_energy = (e_curr is not None and e_prev is not None
                        and abs(float(e_curr) - float(e_prev)) < self.emax)
        else:
            ok_energy = True
        return ok_forces and ok_stress and ok_energy

    def _update_best(self, e: float, step: int) -> None:
        """
        Record the current positions/lattice if e < E_best so far.

        Called once per outer iteration inside each _run_* method when
        save_best=True.  The overhead is a single float comparison per
        step plus an O(N) array copy only when a new minimum is found,
        which is negligible relative to the force_fn cost.
        """
        if not self.save_best:
            return
        if float(e) < self._best_e:
            self._best_e    = float(e)
            self._best_pos  = self.positions.copy()
            self._best_lat  = self.lattice.copy()
            self._best_step = step

    def _finalise(self, e, forces, stress, converged, n_steps):
        status = "CONVERGED" if converged else "NOT CONVERGED (max steps reached)"
        print(f"\n  Geometry optimisation -- {status}")

        # Save the true final positions/lattice BEFORE any restoration.
        # This is what POSCAR_relaxed will contain regardless of save_best.
        final_pos = self.positions.copy()
        final_lat = self.lattice.copy()
        e_final   = float(e)

        # If save_best=True, write POSCAR_best with the lowest-energy
        # structure seen during the run, then restore it into
        # self.positions/self.lattice for the returned result dict.
        if self.save_best and self._best_pos is not None:
            best_is_different = (self._best_step != n_steps - 1)
            if best_is_different:
                print(f"  Best energy found at step {self._best_step}: "
                      f"{self._best_e:.6f} eV")
                print(f"  (Final step energy was {e_final:.6f} eV; "
                      f"best structure saved to POSCAR_best)")
                # Restore best into self for the force re-evaluation and
                # for the returned result dict (callers receive best).
                self.positions = self._best_pos
                self.lattice   = self._best_lat
                e, forces, stress = self._get_forces_and_stress()
            else:
                print(f"  Best energy coincides with final step "
                      f"({self._best_e:.6f} eV)")

            best_out = os.path.join(self.output_dir, "POSCAR_best")
            fa_out = (self._fix_ions.reshape(self.N, 3)
                      if np.any(self._fix_ions) else None)
            write_poscar(best_out, self.lattice, self.positions,
                         self.atom_types,
                         title="Best-energy structure (akitsushima.py)",
                         fix_atoms=fa_out)
            print(f"  Best structure written to {best_out}")

        print(f"  Final energy : {e_final:.6f} eV")
        print(f"  Final Fmax   : {np.max(np.abs(forces)):.4f} eV/A")
        # POSCAR_relaxed always contains the LAST step structure,
        # regardless of whether save_best is active.
        out = os.path.join(self.output_dir, "POSCAR_relaxed")
        fa_out = (self._fix_ions.reshape(self.N, 3)
                  if np.any(self._fix_ions) else None)
        write_poscar(out, final_lat, final_pos,
                     self.atom_types, title="Relaxed by akitsushima.py",
                     fix_atoms=fa_out)
        print(f"  Relaxed structure written to {out}")
        return {
            "energy"    : float(e),
            "lattice"   : self.lattice.copy(),
            "positions" : self.positions.copy(),
            "atom_types": self.atom_types,
            "forces"    : forces.copy(),
            "converged" : converged,
            "n_steps"   : n_steps,
            "fix_atoms" : self._fix_ions.reshape(self.N, 3).copy(),
            "fix_cell"  : self._fix_cell_mask.reshape(3, 3).copy(),
            "best_energy": self._best_e if self.save_best else None,
            "best_step" : self._best_step if self.save_best else None,
        }

    def _run_fire(self) -> dict:
        """Fast Inertial Relaxation Engine (Bitzek et al., PRL 97, 2006)."""
        log_fh, header, _ = self._open_log("dt(fs)")
        # Bug-fix note (S9): every _run_* method used to close log_fh /
        # self._xdatcar_fh only in the statement immediately after the
        # loop, so an exception anywhere inside the loop (e.g. force_fn
        # raising on a pathological structure) left both file handles
        # open for the rest of the process's lifetime. Wrapping the body
        # in try/finally guarantees they are always closed.
        try:
            log_fh.write(header + "\n")

            dt = self._DT_START; alpha = self._ALPHA_START; n_pos = 0
            ndof = self.N * 3 + (9 if self.relax_cell else 0)
            v    = np.zeros(ndof)

            e, forces, stress = self._get_forces_and_stress()
            g = self._generalised_force(forces, stress)
            e_prev = None

            converged = False; step = 0
            for step in range(self.max_steps):
                self._tc.start("step")  # timing
                fmax_v, _ = self._log_row(log_fh, step, e, forces, stress,
                                           f"{dt:.4f}")
                self._tc.lap("step")   # timing
                self._update_best(e, step)
                if self._check_converged(fmax_v, stress, e, e_prev):
                    converged = True; break
                e_prev = e

                P = np.dot(v, g)
                g_norm = np.linalg.norm(g); v_norm = np.linalg.norm(v)
                if g_norm > 1e-12:
                    v = (1.0 - alpha) * v + alpha * (v_norm / g_norm) * g
                if P > 0.0:
                    n_pos += 1
                    if n_pos > self._N_MIN:
                        dt = min(dt * self._F_INC, self._DT_MAX)
                        alpha *= self._F_ALPHA
                else:
                    v = np.zeros(ndof); n_pos = 0
                    dt *= self._F_DEC; alpha = self._ALPHA_START

                v += 0.5 * dt * g
                disp = self._clamp_step(dt * v)
                self._apply_generalised_displacement(disp)
                e, forces, stress = self._get_forces_and_stress()
                g = self._generalised_force(forces, stress)
                v += 0.5 * dt * g

            log_fh.write(f"\nConverged: {converged}\n")
        finally:
            log_fh.close()
            self._xdatcar_fh.close()
        return self._finalise(e, forces, stress, converged, step + 1)

    # ------------------------------------------------------------------ #
    #  2. L-BFGS                                                           #
    # ------------------------------------------------------------------ #

    def _run_lbfgs(self) -> dict:
        """
        Limited-memory BFGS (Nocedal 1980).

        Maintains the last *memory* (s, y) displacement/gradient pairs to
        implicitly approximate H^-1.  The two-loop recursion gives the
        quasi-Newton direction p = H^-1 g; Armijo backtracking ensures
        sufficient energy decrease.
        """
        log_fh, header, _ = self._open_log("||g||")
        try:
            log_fh.write(header + "\n")

            e, forces, stress = self._get_forces_and_stress()
            g = -self._generalised_force(forces, stress)   # gradient (grad E)
            x = self._get_x()

            s_hist : deque = deque(maxlen=self.memory)
            y_hist : deque = deque(maxlen=self.memory)

            converged = False; step = 0
            step_size_prev = 1.0   # warm-start guess, updated after each accepted step
            e_prev = None
            for step in range(self.max_steps):
                self._tc.start("step")  # timing
                fmax_v, _ = self._log_row(log_fh, step, e, forces, stress,
                                           f"{np.linalg.norm(g):.3e}")
                if self._check_converged(fmax_v, stress, e, e_prev):
                    self._tc.lap("step")   # timing
                    converged = True; break
                self._update_best(e, step)
                e_prev = e

                p = self._lbfgs_direction(g, s_hist, y_hist)   # H^-1 g

                # Pre-scale p so that step_size=1.0 corresponds exactly to the
                # clamped (_MAX_STEP) displacement.  Without this, _clamp_step
                # saturates the displacement to _MAX_STEP for every step_size
                # above _MAX_STEP/||p||_inf, making the Armijo halving loop
                # repeat the SAME displacement many times (up to ~20 wasted
                # force_fn calls per outer step) before step_size finally drops
                # below that threshold and backtracking can take effect.
                p_inf = float(np.max(np.abs(p)))
                if p_inf > self._MAX_STEP:
                    p = p * (self._MAX_STEP / p_inf)

                # Armijo backtracking line search.
                # Warm-start from the step_size that worked last iteration
                # (capped at 1.0 and floored at a small value), then expand by
                # 2x once before backtracking.  This avoids re-discovering a
                # tiny step_size (e.g. 2^-10) from scratch every outer step,
                # which is the dominant cost when the line search is far from
                # quadratic (e.g. early L-BFGS iterations with poor curvature).
                step_size = min(1.0, step_size_prev * 2.0)
                accepted = False
                for _ in range(30):
                    dx = self._clamp_step(-step_size * p)
                    self._set_x(x + dx)
                    e_new, f_new, st_new = self._get_forces_and_stress()
                    if float(e_new) <= float(e) - 1e-4 * step_size * float(np.dot(g, p)):
                        accepted = True
                        break
                    step_size *= 0.5
                else:
                    dx = self._clamp_step(-g)
                    self._set_x(x + dx)
                    e_new, f_new, st_new = self._get_forces_and_stress()
                if accepted:
                    step_size_prev = step_size
                self._tc.lap("step")   # timing (covers full line search)

                g_new = -self._generalised_force(f_new, st_new)
                s = self._get_x() - x
                y = g_new - g
                sy = float(np.dot(s, y))
                if sy > 1e-14 * np.linalg.norm(s) * np.linalg.norm(y):
                    s_hist.append(s.copy()); y_hist.append(y.copy())
                    # deque(maxlen=memory) auto-evicts oldest entry -- no pop(0) needed

                x = self._get_x()
                e, forces, stress, g = e_new, f_new, st_new, g_new

            log_fh.write(f"\nConverged: {converged}\n")
        finally:
            log_fh.close()
            self._xdatcar_fh.close()
        return self._finalise(e, forces, stress, converged, step + 1)

    @staticmethod
    def _lbfgs_direction(g: np.ndarray,
                          s_hist: list, y_hist: list) -> np.ndarray:
        """L-BFGS two-loop recursion -> p ~= H^-1 g."""
        if not s_hist:
            return g.copy()
        m = len(s_hist)
        rho = [1.0 / (np.dot(y, s) + 1e-30)
               for s, y in zip(s_hist, y_hist)]
        q = g.copy(); alpha = np.zeros(m)
        for i in range(m - 1, -1, -1):
            alpha[i] = rho[i] * np.dot(s_hist[i], q)
            q        -= alpha[i] * y_hist[i]
        sy = np.dot(s_hist[-1], y_hist[-1])
        yy = np.dot(y_hist[-1], y_hist[-1])
        r  = q * (sy / yy if yy > 1e-30 else 1.0)
        for i in range(m):
            beta = rho[i] * np.dot(y_hist[i], r)
            r   += s_hist[i] * (alpha[i] - beta)
        return r

    # ------------------------------------------------------------------ #
    #  3. DIIS (Pulay 1980)                                               #
    # ------------------------------------------------------------------ #

    def _run_diis(self) -> dict:
        """
        Direct Inversion in the Iterative Subspace (Pulay, CPL 73, 1980).

        Stores the last *memory* (position, force) pairs.  At each step,
        the Lagrangian linear system
            B c = [0, ..., 0, 1]^T  s.t.  sum c_i = 1
        is solved, where B_ij = <e_i, e_j> and e_i = -force_i (residual).
        The new trial position is the linear combination sum c_i x_i.
        A steepest-descent seed is used while the subspace is too small.
        """
        log_fh, header, _ = self._open_log("subspace")
        try:
            log_fh.write(header + "\n")

            e, forces, stress = self._get_forces_and_stress()
            x_buf : deque = deque(maxlen=self.memory)
            e_buf : deque = deque(maxlen=self.memory)   # error = residual = -force

            converged = False; step = 0
            e_prev = None
            for step in range(self.max_steps):
                self._tc.start("step")  # timing
                g     = self._generalised_force(forces, stress)
                fmax_v, _ = self._log_row(log_fh, step, e, forces, stress,
                                           f"{len(x_buf)}")
                if self._check_converged(fmax_v, stress, e, e_prev):
                    self._tc.lap("step")   # timing
                    converged = True; break
                self._update_best(e, step)
                e_prev = e

                x_cur = self._get_x()
                x_buf.append(x_cur.copy()); e_buf.append(-g.copy())
                # deque(maxlen) auto-evicts oldest entry

                m = len(x_buf)
                if m < 2:
                    dx = self._clamp_step(g * 0.05)
                    self._set_x(x_cur + dx)
                    e, forces, stress = self._get_forces_and_stress()
                else:
                    # Pulay B-matrix via NumPy matrix multiply (O(m*ndof))
                    # instead of a Python double-loop (O(m^2*ndof)).
                    E_mat = np.array(e_buf)             # (m, ndof)
                    B     = np.zeros((m + 1, m + 1))
                    B[:m, :m] = E_mat @ E_mat.T         # single BLAS call
                    B[m, :m]  = 1.0; B[:m, m] = 1.0
                    rhs   = np.zeros(m + 1); rhs[m] = 1.0

                    # Plain np.linalg.solve fails catastrophically when B is
                    # (near-)singular -- which happens whenever two stored
                    # residual vectors are nearly parallel (e.g. consecutive
                    # steepest-descent-like seed steps).  The unregularised
                    # solve then returns wildly oscillating coefficients
                    # (e.g. c = [-2.18, 3.18, ~0, ~0, ...]) that extrapolate
                    # FAR outside the convex hull of the stored positions,
                    # producing a huge ||dx|| that gets clamped to _MAX_STEP
                    # but still moves in a direction that worsens the energy
                    # every single step -- a silent, deterministic divergence.
                    #
                    # Fix 1: solve via least-squares (effectively an SVD
                    # pseudo-inverse with small-singular-value damping)
                    # instead of a raw direct solve.
                    try:
                        c = np.linalg.lstsq(B, rhs, rcond=1e-10)[0]
                    except np.linalg.LinAlgError:
                        c = np.zeros(m + 1); c[m - 1] = 1.0

                    # Fix 2: reject pathological extrapolations outright.
                    use_sd_fallback = np.max(np.abs(c[:m])) > self._DIIS_C_MAX

                    if not use_sd_fallback:
                        x_arr = np.array(x_buf)             # (m, ndof)
                        x_new = c[:m] @ x_arr               # weighted sum
                        dx    = self._clamp_step(x_new - x_cur)
                        self._set_x(x_cur + dx)
                        e_trial, f_trial, st_trial = self._get_forces_and_stress()

                        # Fix 3: DIIS has no built-in energy-decrease guarantee
                        # (unlike a line-search method).  If the trial step
                        # makes the energy worse, discard it and fall back to
                        # steepest descent -- this is what prevents the
                        # monotonic divergence previously observed
                        # (E: -1185 -> -354 -> +10068 over 30 steps).
                        if float(e_trial) > float(e):
                            use_sd_fallback = True
                        else:
                            e, forces, stress = e_trial, f_trial, st_trial

                    if use_sd_fallback:
                        dx = self._clamp_step(g * 0.05)
                        self._set_x(x_cur + dx)
                        e, forces, stress = self._get_forces_and_stress()
                self._tc.lap("step")   # timing (covers trial energy evaluation)

            log_fh.write(f"\nConverged: {converged}\n")
        finally:
            log_fh.close()
            self._xdatcar_fh.close()
        return self._finalise(e, forces, stress, converged, step + 1)

    # ------------------------------------------------------------------ #
    #  4. RMM-DIIS (Kresse & Furthmuller 1996)                           #
    # ------------------------------------------------------------------ #

    def _run_rmmdiis(self) -> dict:
        """
        Residual Minimisation Method DIIS (Kresse & Furthmuller,
        Comput. Mater. Sci. 6, 1996).

        Projects the current residual r onto the subspace spanned by the
        last (m-1) residual differences delta_r_i = r_i - r_{i-1} via
        least-squares and performs a backtracking line search along the
        resulting preconditioned direction p = r - sum c_i delta_r_i.

        Key advantages over plain DIIS:
        - No Lagrange constraint: the projection is always well-defined.
        - Line search guarantees monotone energy decrease.
        - Numerically stable for large or ill-conditioned starting residuals.
        """
        log_fh, header, _ = self._open_log("step_size")
        try:
            log_fh.write(header + "\n")

            e, forces, stress = self._get_forces_and_stress()
            x_buf : List[np.ndarray] = []
            r_buf : List[np.ndarray] = []

            converged = False; step = 0
            e_prev = None
            alpha_prev = 1.0   # warm-start guess, updated after each accepted step
            for step in range(self.max_steps):
                self._tc.start("step")  # timing
                g = self._generalised_force(forces, stress)   # residual
                fmax_v, _ = self._log_row(log_fh, step, e, forces, stress, "---")
                if self._check_converged(fmax_v, stress, e, e_prev):
                    self._tc.lap("step")   # timing
                    converged = True; break
                self._update_best(e, step)
                e_prev = e

                x_cur = self._get_x()
                x_buf.append(x_cur.copy()); r_buf.append(g.copy())
                if len(x_buf) > self.memory:
                    x_buf.pop(0); r_buf.pop(0)

                m = len(r_buf)
                if m < 2:
                    p = g.copy()
                else:
                    # Build delta_r matrix and project r onto it
                    dr = np.array([r_buf[i] - r_buf[i-1] for i in range(1, m)])
                    A  = dr @ dr.T + 1e-12 * np.eye(m - 1)
                    b  = dr @ g
                    try:
                        c = np.linalg.lstsq(A, b, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        c = np.zeros(m - 1)
                    p = g - sum(c[i] * dr[i] for i in range(m - 1))

                # Pre-scale p so that alpha_ls=1.0 corresponds exactly to the
                # clamped (_MAX_STEP) displacement.  Without this, _clamp_step
                # saturates the displacement to _MAX_STEP for every alpha_ls
                # above _MAX_STEP/||p||_inf, making the backtracking loop
                # repeat the SAME displacement many times (each repeat costs
                # one full force_fn evaluation) before alpha_ls finally drops
                # below that threshold and backtracking can take effect.
                # This is the same fix applied to L-BFGS and SD.
                p_inf = float(np.max(np.abs(p)))
                if p_inf > self._MAX_STEP:
                    p = p * (self._MAX_STEP / p_inf)

                # Backtracking line search, warm-started from the alpha_ls
                # that worked on the previous outer iteration.
                alpha_ls = min(1.0, alpha_prev * 2.0); e0 = float(e)
                accepted = False
                for _ in range(30):
                    dx = self._clamp_step(p * alpha_ls)
                    self._set_x(x_cur + dx)
                    e_new, f_new, st_new = self._get_forces_and_stress()
                    if float(e_new) < e0:
                        accepted = True
                        break
                    alpha_ls *= 0.5
                else:
                    dx = self._clamp_step(g * 0.01)
                    self._set_x(x_cur + dx)
                    e_new, f_new, st_new = self._get_forces_and_stress()
                if accepted:
                    alpha_prev = alpha_ls
                self._tc.lap("step")   # timing (covers full line search)

                e, forces, stress = e_new, f_new, st_new

            log_fh.write(f"\nConverged: {converged}\n")
        finally:
            log_fh.close()
            self._xdatcar_fh.close()
        return self._finalise(e, forces, stress, converged, step + 1)

    # ------------------------------------------------------------------ #
    #  5. RFO (Rational Function Optimisation)                            #
    # ------------------------------------------------------------------ #

    def _run_rfo(self) -> dict:
        """
        Rational Function Optimisation (Simons et al. 1983; Baker 1986).

        Solves the augmented Hessian eigenproblem for the step direction.

        **Minimum search** (``uphill=False``, default):
        The step follows the lowest eigenvalue of the augmented Hessian,
        ensuring descent toward the nearest local minimum.

        **Saddle-point / TS refinement** (``uphill=True``):
        Partitioned RFO (Banerjee et al. 1985) -- maximises along the
        softest Hessian mode (the imaginary mode at a TS candidate) and
        minimises along all perpendicular modes.  This refines a
        transition-state guess (e.g. from CI-NEB) to the exact saddle point.

        The Hessian is computed by central-difference FD at step 0 and
        updated with symmetric BFGS (minimisation) or SR1 (TS refinement)
        at each subsequent step.

        Integration with TransitionStateAnalyser
        -----------------------------------------
        Typical TS refinement workflow::

            neb_result = NEBOptimiser(chain, fmax=0.05).run()

            go = GeometryOptimiser(
                lat, neb_result['climbing_image_pos'], types,
                force_fn=fn, method='rfo', uphill=True,
                fmax=0.01, trust_radius=0.2)
            ts_result = go.run()

            analyser = TransitionStateAnalyser(
                ts_result['lattice'], ts_result['positions'], types, fn)
            analyser.analyse_modes()
            print(analyser.is_first_order_saddle())   # True -> valid TS
        """
        log_fh, header, _ = self._open_log("trust_r")
        try:
            log_fh.write(header + "\n")

            e, forces, stress = self._get_forces_and_stress()
            g = -self._generalised_force(forces, stress)   # gradient grad E
            x = self._get_x()
            ndof = len(g)

            if self.verbose:
                print(f"  Computing initial Hessian ({2*ndof} FD evaluations) ...")
            H = self._fd_hessian(ndof)

            trust = self.trust_radius
            x_prev = x.copy(); g_prev = g.copy()
            converged = False; step = 0
            e_prev = None

            for step in range(self.max_steps):
                self._tc.start("step")  # timing
                fmax_v, _ = self._log_row(log_fh, step, e, forces, stress,
                                           f"{trust:.4f}")
                self._tc.lap("step")   # timing
                if self._check_converged(fmax_v, stress, e, e_prev):
                    converged = True; break
                e_prev = e

                # RFO or P-RFO step direction
                if self.uphill:
                    p, _ = self._prfo_step(g, H)
                else:
                    p, _ = self._rfo_step(g, H)

                # Trust-region clamp
                p_ion_norm = np.linalg.norm(p[:self.N*3])
                if p_ion_norm > trust:
                    p = p * (trust / (p_ion_norm + 1e-16))

                dx = self._clamp_step(p)
                x_new = x + dx
                self._set_x(x_new)
                e_new, f_new, st_new = self._get_forces_and_stress()
                g_new = -self._generalised_force(f_new, st_new)

                # Trust-radius update (rho = actual / predicted energy change)
                pred_de  = float(np.dot(g, dx) + 0.5 * float(dx @ H @ dx))
                actual_de = float(e_new) - float(e)
                rho = actual_de / pred_de if abs(pred_de) > 1e-14 else 1.0
                if rho > 0.75:
                    trust = min(trust * 2.0, self.trust_radius * 8)
                elif rho < 0.25:
                    trust = max(trust * 0.5, 1e-3)

                # Accept step
                if actual_de <= 0.0 or self.uphill:
                    x, e, forces, stress, g = x_new, e_new, f_new, st_new, g_new
                else:
                    self._set_x(x); trust *= 0.5; continue

                # Hessian update
                s_vec = x - x_prev; y_vec = g - g_prev
                sy = float(np.dot(s_vec, y_vec))
                sn = np.linalg.norm(s_vec) * np.linalg.norm(y_vec)
                if sy > 1e-14 * sn:
                    Hs = H @ s_vec; sHs = float(np.dot(s_vec, Hs))
                    if self.uphill:
                        # SR1 (preserves negative curvature)
                        diff = y_vec - Hs
                        denom = float(np.dot(diff, s_vec))
                        if abs(denom) > 1e-14:
                            H = H + np.outer(diff, diff) / denom
                    else:
                        # Symmetric BFGS
                        H = (H - np.outer(Hs, Hs) / sHs
                               + np.outer(y_vec, y_vec) / sy)

                x_prev, g_prev = x.copy(), g.copy()

            log_fh.write(f"\nConverged: {converged}\n")
        finally:
            log_fh.close()
            self._xdatcar_fh.close()
        return self._finalise(e, forces, stress, converged, step + 1)

    def _run_sd(self) -> dict:
        """
        Plain Steepest Descent with Armijo backtracking line search.

        The simplest possible optimiser: move along -gradient at each
        step, halving the step size until the energy decreases
        sufficiently (Armijo condition).  No curvature information is
        used, so convergence is typically much slower than FIRE or
        L-BFGS, but the algorithm is maximally robust and easy to reason
        about -- useful as a fallback or for debugging force fields.
        """
        log_fh, header, _ = self._open_log("step_size")
        try:
            log_fh.write(header + "\n")

            e, forces, stress = self._get_forces_and_stress()
            g = self._generalised_force(forces, stress)   # descent direction = +g

            converged = False; step = 0
            step_size_prev = 1.0
            e_prev = None
            for step in range(self.max_steps):
                self._tc.start("step")  # timing
                fmax_v, _ = self._log_row(log_fh, step, e, forces, stress,
                                           f"{step_size_prev:.3e}")
                if self._check_converged(fmax_v, stress, e, e_prev):
                    self._tc.lap("step")   # timing
                    converged = True; break
                self._update_best(e, step)
                e_prev = e

                x = self._get_x()

                # Pre-scale direction so step_size=1.0 lands exactly on the
                # _MAX_STEP-clamped displacement (see L-BFGS for rationale).
                g_inf = float(np.max(np.abs(g)))
                p = g / g_inf * self._MAX_STEP if g_inf > self._MAX_STEP else g.copy()

                step_size = min(1.0, step_size_prev * 2.0)
                accepted = False
                for _ in range(30):
                    dx = self._clamp_step(step_size * p)
                    self._set_x(x + dx)
                    e_new, f_new, st_new = self._get_forces_and_stress()
                    if float(e_new) <= float(e) - 1e-4 * step_size * float(np.dot(g, p)):
                        accepted = True
                        break
                    step_size *= 0.5
                else:
                    dx = self._clamp_step(g * 1e-3)
                    self._set_x(x + dx)
                    e_new, f_new, st_new = self._get_forces_and_stress()
                if accepted:
                    step_size_prev = step_size
                self._tc.lap("step")   # timing

                e, forces, stress = e_new, f_new, st_new
                g = self._generalised_force(forces, stress)

            log_fh.write(f"\nConverged: {converged}\n")
        finally:
            log_fh.close()
            self._xdatcar_fh.close()
        return self._finalise(e, forces, stress, converged, step + 1)

    def _fd_hessian(self, ndof: int) -> np.ndarray:
        """Central-difference Hessian via force finite differences."""
        delta  = self.hessian_delta
        x_save = self._get_x()
        H      = np.zeros((ndof, ndof))
        for j in range(ndof):
            xp = x_save.copy(); xp[j] += delta; self._set_x(xp)
            _, fp, sp = self._get_forces_and_stress()
            gp = -self._generalised_force(fp, sp)
            xm = x_save.copy(); xm[j] -= delta; self._set_x(xm)
            _, fm, sm = self._get_forces_and_stress()
            gm = -self._generalised_force(fm, sm)
            H[:, j] = (gp - gm) / (2.0 * delta)
        H = 0.5 * (H + H.T)
        self._set_x(x_save)
        return H

    @staticmethod
    def _rfo_step(g: np.ndarray, H: np.ndarray):
        """
        RFO step for minimum search.

        Augmented Hessian:  A = [[H, g], [g^T, 0]]
        Returns step p = +v[:n] / v[n] for the lowest eigenvector v of A.

        Derivation.  Write the lowest eigenpair as v = (v_x, v_n) with
        eigenvalue lam.  The two blocks of ``A v = lam v`` read

            H v_x + g v_n = lam v_x   ->   (H - lam I) v_x = -g v_n
            g^T v_x       = lam v_n

        so the RFO step is

            p = v_x / v_n = -(H - lam I)^{-1} g

        which reduces to the Newton step -H^{-1} g as lam -> 0.  The
        second block gives the exact identity ``g . p = lam``; because
        lam is the LOWEST eigenvalue of A it is <= 0 for any non-zero g
        (A has a zero on its diagonal, so lam <= 0 by the variational
        principle), i.e. p is guaranteed to be a DESCENT direction.

        Bug-fix note (RFO sign): this routine used to return
        ``p = -v[:n] / v[n]``, i.e. exactly ``+(H - lam I)^{-1} g`` --
        the NEGATIVE of the correct step, and therefore an ASCENT
        direction for a minimum search.  Because the augmented-Hessian
        eigenproblem hides the Newton limit, the wrong sign was not
        visible by inspection; it was confirmed numerically by comparing
        against the analytic Newton step for a random well-conditioned
        SPD Hessian with a small generic gradient, where the ratio
        p_code / p_newton came out at -1.0000 componentwise and
        g . p came out POSITIVE (+4.12e-4) instead of negative.
        End-to-end the defect was silent rather than loud: ``_run_rfo``
        rejects any step that raises the energy (``if actual_de <= 0.0``),
        so a minimisation with method='rfo' rejected *every* step and sat
        at its starting point forever -- on an analytic Rosenbrock test
        the energy stayed at exactly 1040.600000 and Fmax at exactly
        792.0000 for all 400 steps, while the sign-corrected step
        converged in 81 steps to E=2.2e-13 with Fmax=1.5e-05.
        See test_solvers.py::test_rfo_step_matches_newton_step and
        ::test_rfo_relaxation_converges_rosenbrock.
        """
        n = len(g)
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = H; A[:n, n] = g; A[n, :n] = g
        A = 0.5 * (A + A.T)
        eigvals, eigvecs = np.linalg.eigh(A)

        # `eigh` fixes each eigenvector only up to a global sign.  The regular
        # branch p = v_x / v_n is invariant under that flip (numerator and
        # denominator flip together), but the degenerate fallback below is
        # NOT, so pin a canonical sign here: make the largest-magnitude
        # component of v positive.  This costs O(n) next to an O(n^3) eigh
        # and is what makes the fallback reproducible rather than a step
        # whose direction depends on the LAPACK build.
        v = eigvecs[:, 0]
        if v[int(np.argmax(np.abs(v)))] < 0.0:
            v = -v
        denom = v[n]

        if abs(denom) < 1e-12:
            # Degenerate: v_n ~ 0, so the exact step v_x / v_n diverges.  Fall
            # back to the unit-norm direction v_x and let the caller's
            # trust-radius clamp set the length.  Orient it downhill where
            # that is decidable -- but note g . v_x = lam * v_n exactly (second
            # block above), so this test degenerates in precisely the limit
            # that triggers the fallback; the canonical sign pin above, not
            # this flip, is what guarantees determinism.
            p = v[:n]
            if float(np.dot(g, p)) > 0.0:
                p = -p
        else:
            p = v[:n] / denom
        return p, float(eigvals[0])

    @staticmethod
    def _prfo_step(g: np.ndarray, H: np.ndarray):
        """
        Partitioned RFO for saddle-point search (Banerjee et al. 1985).

        Maximises along the softest Hessian mode (mode 0, imaginary mode)
        via a 1-D RFO uphill step; minimises along all other modes.

        In the Hessian eigenbasis {lam_k, u_k}, with modal gradient
        b_k = u_k . g, every mode takes the same RFO form

            p_k = -b_k / (lam_k - nu_k)

        and the two partitions differ only in which root nu_k of the 1-D
        RFO secular equation is used:

        * mode 0 (MAXIMISE) takes the upper root
          nu_0 = (lam_0 + sqrt(lam_0^2 + 4 b_0^2)) / 2 >= lam_0, so
          lam_0 - nu_0 <= 0 and p_0 carries the SAME sign as b_0 --
          the step goes UPHILL along the mode being maximised.
        * modes k >= 1 (MINIMISE) take the lower root
          nu_k = (lam_k - sqrt(lam_k^2 + 4 b_k^2)) / 2 <= lam_k, so
          lam_k - nu_k >= 0 and p_k carries the OPPOSITE sign to b_k --
          the step goes DOWNHILL along every mode being minimised.

        Bug-fix note (RFO sign): both denominators used to be written with
        the operands REVERSED -- ``nu_0 - lam_0`` for mode 0 and
        ``nu_k - lam_k`` for the rest -- which negates every modal
        component and so returned exactly minus the correct step.  The
        two partitions therefore both did the opposite of their job:
        mode 0 walked DOWNHILL along the mode it is supposed to ascend
        and modes k >= 1 walked UPHILL along the modes they are supposed
        to minimise, so P-RFO could not converge on a saddle point.  This
        was confirmed numerically on an indefinite Hessian with a single
        negative eigenvalue: the old code produced modal gradient
        [-0.0166, +0.0063, +0.0048, +0.0616] with modal step
        [+0.0083, +0.0014, +0.0008, +0.0050] -- mode 0 opposed in sign
        and all higher modes aligned in sign, i.e. every mode backwards.
        The fix restores the operand order rather than negating the
        return value, so the code visibly matches the published formula.
        See test_solvers.py::test_prfo_step_mode_signs.
        """
        n = len(g)
        eigvals, eigvecs = np.linalg.eigh(H)
        g_modal = eigvecs.T @ g
        p_modal = np.zeros(n)

        # Mode 0: RFO maximisation (follow imaginary/softest mode uphill).
        # Upper root nu_0 >= lam_0  =>  denom0 <= 0  =>  p_0 has the same
        # sign as b_0, i.e. the step ascends this mode.
        lam0 = eigvals[0]; b0 = g_modal[0]
        disc0 = lam0**2 + 4.0 * b0**2
        nu_0 = (lam0 + np.sqrt(disc0)) / 2.0
        denom0 = lam0 - nu_0
        p_modal[0] = -b0 / denom0 if abs(denom0) > 1e-14 else 0.0

        # Modes 1..n-1: standard RFO minimisation.
        # Lower root nu_k <= lam_k  =>  denom_k >= 0  =>  p_k has the
        # opposite sign to b_k, i.e. the step descends these modes.
        for k in range(1, n):
            lk = eigvals[k]; bk = g_modal[k]
            disc_k = lk**2 + 4.0 * bk**2
            nu_k = (lk - np.sqrt(disc_k)) / 2.0
            denom_k = lk - nu_k
            p_modal[k] = -bk / denom_k if abs(denom_k) > 1e-14 else 0.0

        return eigvecs @ p_modal, float(eigvals[0])
