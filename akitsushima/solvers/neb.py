"""Climbing-Image Nudged Elastic Band (CI-NEB) reaction-path search.

References
----------
NEB              : Jonsson, Mills & Jacobsen (1998)
Improved tangent : Henkelman & Jonsson, JCP 113, 9901 (2000)
CI-NEB           : Henkelman, Uberuaga & Jonsson, JCP 113, 9978 (2000)
FIRE             : Bitzek et al., PRL 97, 170201 (2006)
"""
import os
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..engine.neighborlist import NeighbourList
from ..io.poscar import write_poscar
from ..utils.mpi import _mpi_allreduce_sum, _mpi_domain_split, _mpi_rank, local_only
from ..utils.optional_matplotlib import HAS_MATPLOTLIB, plt
from ..utils.timing import TimingCollector

__all__ = ["NEBChain", "NEBOptimiser"]


class NEBChain:
    """
    Chain of images for the Nudged Elastic Band (NEB) method.

    Manages N_img intermediate images between two fixed endpoint structures,
    evaluates true forces on each image using make_force_fn(), and returns
    the CI-NEB force array for use by NEBOptimiser.

    Parameters
    ----------
    images_pos  : list of (N_atoms, 3) arrays, length = N_img + 2
                  (first = initial, last = final, rest = intermediates).
    lattice     : (3,3) lattice vectors [A].  Same for all images.
    atom_types  : list of element symbols, length N_atoms.
    force_fn    : unified force callable from make_force_fn().
    k_spring    : spring constant [eV/A^2].  Default 5.0.
    climbing    : enable Climbing-Image modification (default True).
    r_cut       : NeighbourList cutoff [A].  Auto-detected from force_fn.r_cut.
    skin_width  : Verlet skin width [A].

    Notes
    -----
    Periodic boundary conditions (PBC) are handled via the minimum-image
    convention throughout.  The lattice must be the same for all images;
    variable-cell NEB is not supported.
    """

    def __init__(self,
                 images_pos : List[np.ndarray],
                 lattice    : np.ndarray,
                 atom_types : list,
                 force_fn   : Callable,
                 k_spring   : float = 5.0,
                 climbing   : bool  = True,
                 r_cut      : Optional[float] = None,
                 skin_width : float = 0.5):

        if len(images_pos) < 3:
            raise ValueError(
                "NEBChain: need at least 3 images (init + 1 intermediate + final).")

        self.images     = [p.copy() for p in images_pos]
        self.lattice    = lattice.copy()
        self.atom_types = list(atom_types)
        self.force_fn   = force_fn
        self.k_spring   = float(k_spring)
        self.climbing   = climbing
        self.N          = len(atom_types)
        self.n_images   = len(images_pos) - 2   # intermediate images only

        rc = float(r_cut if r_cut is not None
                   else getattr(force_fn, 'r_cut', 10.0))
        self.r_cut = rc
        self._nls = [NeighbourList(r_cut=rc, skin_width=skin_width)
                     for _ in range(self.n_images)]

        self.energies    : List[float] = [0.0] * (self.n_images + 2)
        self.true_forces : List[np.ndarray] = [
            np.zeros((self.N, 3)) for _ in range(self.n_images)]
        self._ci_index   : int = -1   # among intermediates (0-indexed)

    # ------------------------------------------------------------------ #
    #  Factory: linear interpolation with minimum-image convention         #
    # ------------------------------------------------------------------ #

    @classmethod
    def linear_interpolation(cls,
                              pos_init  : np.ndarray,
                              pos_final : np.ndarray,
                              lattice   : np.ndarray,
                              atom_types: list,
                              force_fn  : Callable,
                              n_images  : int = 7,
                              **kwargs) -> 'NEBChain':
        """
        Build a NEBChain with linearly interpolated intermediate images.

        Minimum-image convention is applied so interpolation always takes
        the shortest path across periodic boundaries.

        Parameters
        ----------
        n_images : number of intermediate images (endpoints excluded).
        **kwargs : passed to NEBChain.__init__ (k_spring, climbing, r_cut, ...).
        """
        inv_lat = np.linalg.inv(lattice)
        ds = (pos_final - pos_init) @ inv_lat   # fractional (N,3)
        ds -= np.round(ds)                             # min-image wrap
        d_cart = ds @ lattice                          # Cartesian (N,3)

        images = [pos_init.copy()]
        for k in range(1, n_images + 1):
            t = k / (n_images + 1)
            images.append(pos_init + t * d_cart)
        images.append(pos_final.copy())

        return cls(images, lattice, atom_types, force_fn, **kwargs)

    # ------------------------------------------------------------------ #
    #  Minimum-image Cartesian displacement                                #
    # ------------------------------------------------------------------ #

    def _mic(self, dr: np.ndarray) -> np.ndarray:
        """Apply minimum-image convention to a Cartesian displacement array."""
        inv = np.linalg.inv(self.lattice)
        ds  = dr @ inv
        ds -= np.round(ds)
        return ds @ self.lattice

    # ------------------------------------------------------------------ #
    #  Improved tangent (Henkelman & Jonsson, JCP 113 9901, 2000)         #
    # ------------------------------------------------------------------ #

    def _tangent(self, k: int) -> np.ndarray:
        """
        Improved tangent vector for intermediate image k (0-indexed).

        Avoids kinks at local energy maxima/minima by switching to the
        uphill-side displacement direction.
        """
        pos_m = self.images[k]       # k-th image (0 = initial endpoint)
        pos_c = self.images[k + 1]   # (k+1)-th = current intermediate
        pos_p = self.images[k + 2]   # (k+2)-th image

        E_m = self.energies[k]
        E_c = self.energies[k + 1]
        E_p = self.energies[k + 2]

        tau_p = self._mic(pos_p - pos_c)
        tau_m = self._mic(pos_c - pos_m)

        if E_p > E_c > E_m:
            tau = tau_p
        elif E_m > E_c > E_p:
            tau = tau_m
        else:
            # Weighted average for local extrema
            dE_max = max(abs(E_p - E_c), abs(E_m - E_c)) + 1e-30
            dE_min = min(abs(E_p - E_c), abs(E_m - E_c))
            if E_p > E_m:
                tau = tau_p * dE_max + tau_m * dE_min
            else:
                tau = tau_p * dE_min + tau_m * dE_max

        norm = np.linalg.norm(tau)
        return tau / (norm + 1e-30)

    # ------------------------------------------------------------------ #
    #  Force evaluation                                                    #
    # ------------------------------------------------------------------ #

    def evaluate(self) -> np.ndarray:
        """
        Evaluate true forces on all intermediate images and return the
        CI-NEB force array of shape (n_images, N_atoms, 3).

        Steps
        -----
        1. Evaluate (E, F) for endpoint images (energy only needed).
        2. Evaluate (E, F) for intermediate images via force_fn.
        3. Identify climbing image (highest energy among intermediates).
        4. Build improved tangent that per image.
        5. Apply NEB projection:
           - Standard: F_perp(true) + F_spring_parallel
           - Climbing: F_true - 2(F_true.tau)tau  (no spring, inverted parallel)
        """
        all_pos = self.images   # length n_images + 2

        # Step 1: endpoint energies (forces not needed)
        e0, _, _ = self.force_fn(self.lattice, all_pos[0],  self.atom_types)
        en, _, _ = self.force_fn(self.lattice, all_pos[-1], self.atom_types)
        self.energies[0]  = float(e0)
        self.energies[-1] = float(en)

        # Step 2: intermediate images. Each image's force evaluation
        # depends only on its own current positions (the tangent/spring
        # coupling to neighbouring images happens afterwards, in Python,
        # from the already-collected energies/forces), so this is an
        # ideal coarse-grained MPI split: every rank evaluates only its
        # own slice of images into zero-initialised buffers, and a single
        # SUM-allreduce assembles the complete, identically-replicated
        # result on every rank (each image contributes from exactly one
        # rank, so summing zeros everywhere else is exact -- not an
        # approximation).
        #
        # Bug-fix note: force_fn's underlying engine potentials ALSO
        # unconditionally split their own internal pair-list work across
        # every rank on the same global communicator and Allreduce the
        # result (see every engine potential's compute()). Without
        # local_only() here, each rank's force_fn call for its OWN image
        # would try to Allreduce with every other rank's call for a
        # DIFFERENT image -- silently mixing together partial force/energy
        # sums from unrelated images (confirmed via a bit-for-bit
        # divergence between a 1-rank and a 3-rank run of this exact
        # pattern, applied to PhononCalculator.compute_ifc, before this
        # fix: no crash or hang, just silently wrong numbers). local_only()
        # makes each force_fn call fully local to its own rank, so the
        # explicit Allreduce below is the ONLY cross-rank communication
        # for the image loop.
        i_start, i_end = _mpi_domain_split(self.n_images)
        local_e = np.zeros(self.n_images)
        local_f = np.zeros((self.n_images, self.N, 3))
        with local_only():
            for k in range(i_start, i_end):
                nl = self._nls[k]
                nl.check_and_rebuild(self.lattice, all_pos[k + 1])
                e, f, _ = self.force_fn(
                    self.lattice, all_pos[k + 1], self.atom_types,
                    neigh_list=nl)
                local_e[k] = float(e)
                local_f[k] = f

        all_e = _mpi_allreduce_sum(local_e)
        all_f = _mpi_allreduce_sum(local_f)
        for k in range(self.n_images):
            self.energies[k + 1] = float(all_e[k])
            self.true_forces[k]  = all_f[k]

        # Step 3: find climbing image (highest energy intermediate)
        self._ci_index = int(np.argmax(self.energies[1:-1]))

        # Steps 4-5: build CI-NEB force for each intermediate image
        neb_forces = []
        for k in range(self.n_images):
            tau_n  = self._tangent(k)   # unit tangent (N,3)
            f_true = self.true_forces[k]

            if self.climbing and k == self._ci_index:
                # CI-NEB: invert parallel component, remove spring force
                f_par_coeff = float(np.dot(f_true.ravel(), tau_n.ravel()))
                f_ci = f_true - 2.0 * f_par_coeff * tau_n
                neb_forces.append(f_ci)
            else:
                # Standard NEB: perpendicular true + parallel spring
                f_par_coeff = float(np.dot(f_true.ravel(), tau_n.ravel()))
                f_perp = f_true - f_par_coeff * tau_n

                r_next = self._mic(all_pos[k + 2] - all_pos[k + 1])
                r_prev = self._mic(all_pos[k + 1] - all_pos[k])
                spring_par = self.k_spring * (
                    float(np.dot(r_next.ravel(), tau_n.ravel())) -
                    float(np.dot(r_prev.ravel(), tau_n.ravel()))
                ) * tau_n
                neb_forces.append(f_perp + spring_par)

        return np.array(neb_forces)   # (n_images, N, 3)

    # ------------------------------------------------------------------ #
    #  Accessors                                                           #
    # ------------------------------------------------------------------ #

    @property
    def climbing_image_index(self) -> int:
        """0-indexed position of the climbing image among intermediates."""
        return self._ci_index

    @property
    def climbing_image_pos(self) -> np.ndarray:
        """Atomic positions of the current climbing image."""
        return self.images[self._ci_index + 1].copy()

    @property
    def activation_energy(self) -> float:
        """Forward barrier Ea = E_TS - E_initial [eV]."""
        return float(self.energies[self._ci_index + 1] - self.energies[0])

    @property
    def reverse_barrier(self) -> float:
        """Reverse barrier Ea' = E_TS - E_final [eV]."""
        return float(self.energies[self._ci_index + 1] - self.energies[-1])

    def reaction_path(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (arc_lengths [A], energies [eV]) along the full path
        including both endpoint images.
        """
        s = [0.0]
        for k in range(1, len(self.images)):
            dr = self._mic(self.images[k] - self.images[k - 1])
            s.append(s[-1] + float(np.linalg.norm(dr)))
        return np.array(s), np.array(self.energies)


class NEBOptimiser:
    """
    FIRE-based optimiser for a NEBChain.

    Applies the FIRE algorithm to the concatenated position/force vector
    of all intermediate images simultaneously.  The same FIRE constants
    as GeometryOptimiser are used for consistency.

    Parameters
    ----------
    chain       : NEBChain instance.
    fmax        : convergence criterion [eV/A] on max NEB force component.
    max_steps   : maximum FIRE iterations.
    dt_start    : initial FIRE time step [fs] (default 0.05).
    dt_max      : maximum FIRE time step [fs] (default 0.5).
    output_dir  : directory for neb.log, neb_path.csv, POSCAR_TS, neb_profile.png.
    verbose     : print progress every ``print_every`` steps.
    print_every : log interval (default 10).

    Examples
    --------
    >>> chain = NEBChain.linear_interpolation(
    ...     pos_init, pos_final, lattice, atom_types,
    ...     force_fn=make_force_fn(engine),
    ...     n_images=7, k_spring=5.0, climbing=True)
    >>> opt = NEBOptimiser(chain, fmax=0.05, max_steps=500)
    >>> result = opt.run()
    >>> print(f"Ea = {result['activation_energy']:.3f} eV")
    """

    # FIRE parameters (shared with GeometryOptimiser)
    _DT_START    = 0.05
    _DT_MAX      = 0.5
    _N_MIN       = 5
    _F_INC       = 1.1
    _F_DEC       = 0.5
    _ALPHA_START = 0.1
    _F_ALPHA     = 0.99
    _MAX_DISP    = 0.1   # A per step (tighter than single-image FIRE)

    def __init__(self,
                 chain       : NEBChain,
                 fmax        : float = 0.05,
                 max_steps   : int   = 500,
                 dt_start    : Optional[float] = None,
                 dt_max      : Optional[float] = None,
                 output_dir  : str   = ".",
                 verbose     : bool  = True,
                 print_every : int   = 10,
                 timing      : bool  = False):

        self.chain       = chain
        self.fmax        = float(fmax)
        self.max_steps   = int(max_steps)
        self.dt_start    = float(dt_start or self._DT_START)
        self.dt_max      = float(dt_max   or self._DT_MAX)
        self.output_dir  = output_dir
        self.verbose     = verbose
        self.print_every = print_every
        self._tc         = TimingCollector(enabled=timing)

    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """
        Execute the CI-NEB optimisation loop.

        Returns
        -------
        dict with keys:

        ``converged``           bool
        ``n_steps``             int
        ``activation_energy``   float [eV]
        ``reverse_barrier``     float [eV]
        ``climbing_image_pos``  (N, 3) ndarray
        ``climbing_image_index`` int (among intermediates, 0-indexed)
        ``reaction_path``       (arc_lengths, energies) tuple of 1-D arrays
        ``chain``               NEBChain (updated in-place)
        """
        self._tc.start("total")  # timing
        import time as _time
        os.makedirs(self.output_dir, exist_ok=True)
        # Only rank 0 writes the real log file; every other MPI rank
        # redirects to os.devnull (same convention as the MD classes in
        # molecular_dynamics.py). Every rank runs an identical FIRE loop
        # on the now MPI-replicated NEB forces, so without this all ranks
        # would try to write the same path concurrently.
        log_path = (os.path.join(self.output_dir, "neb.log")
                   if _mpi_rank == 0 else os.devnull)

        chain   = self.chain
        n_img   = chain.n_images
        N       = chain.N
        ndof    = n_img * N * 3

        dt      = self.dt_start
        alpha   = self._ALPHA_START
        n_pos   = 0
        v       = np.zeros(ndof)

        if self.verbose:
            print(f"\n  CI-NEB  ({n_img} intermediate images, "
                  f"k_spring={chain.k_spring:.2f} eV/A^2, "
                  f"climbing={chain.climbing})")

        # Initial evaluation
        neb_f  = chain.evaluate()          # (n_img, N, 3)

        header = (f"{'Step':>6}  {'Ea(eV)':>10}  "
                  f"{'Fmax(eV/A)':>12}  {'CI':>4}  "
                  f"{'dt(fs)':>8}")
        if self.verbose:
            print(f"\n  {header}\n  {'='*len(header)}")

        t0        = _time.perf_counter()
        converged = False

        with open(log_path, 'w') as log:
            log.write(header + "\n")

            for step in range(self.max_steps):
                self._tc.start("step")  # timing
                f_max = float(np.max(np.abs(neb_f)))
                Ea    = chain.activation_energy
                ci    = chain._ci_index

                row = (f"{step:6d}  {Ea:10.4f}  "
                       f"{f_max:12.6f}  {ci:4d}  {dt:8.4f}")
                log.write(row + "\n")
                self._tc.lap("step")  # timing
                if self.verbose and step % self.print_every == 0:
                    print(f"  {row}")

                if f_max < self.fmax:
                    if self.verbose:
                        elapsed = _time.perf_counter() - t0
                        print(f"\n  Converged at step {step}  "
                              f"({elapsed:.1f} s)")
                        print(f"  Fmax = {f_max:.4e} eV/A  "
                              f"Ea = {Ea:.4f} eV  "
                              f"Ea' = {chain.reverse_barrier:.4f} eV")
                    converged = True
                    break

                # ---- FIRE (velocity Verlet + direction mixing) ----------
                F_flat = neb_f.reshape(ndof)
                P      = float(np.dot(v, F_flat))

                f_norm = np.linalg.norm(F_flat) + 1e-16
                v_norm = np.linalg.norm(v)
                v = (1.0 - alpha) * v + alpha * (v_norm / f_norm) * F_flat

                if P > 0.0:
                    n_pos += 1
                    # Bug-fix note (S11): the original used `n_pos >=
                    # self._N_MIN` here, one off from GeometryOptimiser's
                    # `_run_fire` (`n_pos > self._N_MIN`, matching the
                    # Bitzek et al. 2006 FIRE reference) -- meaning
                    # NEBOptimiser started growing dt one step earlier
                    # than a single-image FIRE relaxation with the same
                    # _N_MIN. Changed to `>` for consistency between the
                    # two FIRE implementations in this codebase.
                    if n_pos > self._N_MIN:
                        dt    = min(dt * self._F_INC, self.dt_max)
                        alpha = alpha * self._F_ALPHA
                else:
                    n_pos = 0
                    v[:]  = 0.0
                    dt    = dt * self._F_DEC
                    alpha = self._ALPHA_START

                v   += 0.5 * dt * F_flat
                dx   = dt * v   # (ndof,)

                # Clamp displacement
                max_d = np.max(np.abs(dx))
                if max_d > self._MAX_DISP:
                    dx *= self._MAX_DISP / max_d

                # Update image positions
                dx3 = dx.reshape(n_img, N, 3)
                inv_lat = np.linalg.inv(chain.lattice)
                for k in range(n_img):
                    pos = chain.images[k + 1] + dx3[k]
                    # Bug-fix note (S20, newly found during the port --
                    # present identically in the original monolithic
                    # file): positions were never wrapped back into the
                    # primary cell after a FIRE step, unlike
                    # GeometryOptimiser's own position updates (which do
                    # `frac -= np.floor(frac)`). An atom that drifts
                    # outside [0, box) during optimisation -- entirely
                    # normal for a diffusion/hop pathway, the very thing
                    # NEB is used for -- then made NeighbourList's
                    # cKDTree(..., boxsize=...) raise ValueError
                    # ("Some input data are greater than the size of the
                    # periodic box"), since it requires all coordinates
                    # strictly within the box. Wrapping here matches the
                    # convention already used throughout GeometryOptimiser.
                    frac = pos @ inv_lat
                    # `frac -= np.floor(frac)` alone maps a tiny negative frac to
                    # exactly 1.0 in float64, which scipy's cKDTree rejects; see
                    # wrap_positions_into_cell.
                    frac -= np.floor(frac)
                    frac[frac >= 1.0] = 0.0
                    chain.images[k + 1] = frac @ chain.lattice

                neb_f  = chain.evaluate()
                F_flat = neb_f.reshape(ndof)
                v     += 0.5 * dt * F_flat

            log.write(f"\nConverged: {converged}\n")

        # ---- Output -------------------------------------------------------
        if not converged:
            print(f"\n  CI-NEB: NOT CONVERGED after {self.max_steps} steps. "
                  f"Fmax = {f_max:.4e} eV/A")

        s_arr, e_arr = chain.reaction_path()
        if _mpi_rank == 0:
            self._save_path(s_arr, e_arr)
            self._save_ts_poscar()
            self._save_all_images()
            if HAS_MATPLOTLIB:
                self._plot_path(s_arr, e_arr)

        self._tc.stop("total")  # timing
        _timing = self._tc.summary()
        if _timing.get("enabled") and _mpi_rank == 0:
            self._tc.print_summary(label="NEBOptimiser")

        return {
            "converged"            : converged,
            "n_steps"              : step + 1,
            "activation_energy"    : chain.activation_energy,
            "reverse_barrier"      : chain.reverse_barrier,
            "climbing_image_pos"   : chain.climbing_image_pos,
            "climbing_image_index" : chain._ci_index,
            "reaction_path"        : (s_arr, e_arr),
            "chain"                : chain,
            "timing"   : _timing,
        }

    # ------------------------------------------------------------------ #
    #  Output helpers                                                      #
    # ------------------------------------------------------------------ #

    def _save_path(self, s: np.ndarray, e: np.ndarray) -> None:
        path = os.path.join(self.output_dir, "neb_path.csv")
        with open(path, 'w') as fh:
            fh.write("image,arc_length_Ang,energy_eV,dE_eV\n")
            e0 = e[0]
            for i, (si, ei) in enumerate(zip(s, e)):
                fh.write(f"{i},{si:.6f},{ei:.6f},{ei - e0:.6f}\n")
        print(f"  NEB path   -> {path}")

    def _save_ts_poscar(self) -> None:
        chain = self.chain
        path  = os.path.join(self.output_dir, "POSCAR_TS")
        write_poscar(path, chain.lattice,
                     chain.climbing_image_pos, chain.atom_types,
                     title=(f"Transition state  CI={chain._ci_index}  "
                               f"Ea={chain.activation_energy:.4f} eV"))
        print(f"  TS POSCAR  -> {path}")

    def _save_all_images(self) -> None:
        """Save every intermediate image as POSCAR_image_k."""
        chain = self.chain
        for k in range(chain.n_images):
            p = os.path.join(self.output_dir, f"POSCAR_image_{k:03d}")
            write_poscar(p, chain.lattice, chain.images[k + 1],
                         chain.atom_types,
                         title=(f"NEB image {k}  "
                                   f"E={chain.energies[k+1]:.6f} eV"))

    def _plot_path(self, s: np.ndarray, e: np.ndarray) -> None:
        chain = self.chain
        ci    = chain._ci_index + 1   # index in full image list (incl. endpoints)
        fig, ax = plt.subplots(figsize=(7, 4))
        de = e - e[0]
        ax.plot(s, de, 'o-', color='steelblue', lw=2, ms=6, label='NEB path')
        ax.plot(s[ci], de[ci], '*', color='crimson', ms=14, zorder=5,
                label=f'TS  Ea={chain.activation_energy:.3f} eV')
        ax.axhline(0.0, color='grey', lw=0.8, ls='--')
        ax.set_xlabel("Arc length [A]")
        ax.set_ylabel("delta_E [eV]")
        ax.set_title("CI-NEB energy profile")
        ax.legend(); fig.tight_layout()
        path = os.path.join(self.output_dir, "neb_profile.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  NEB plot   -> {path}")
