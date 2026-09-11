"""Transition-state analysis: Hessian-based normal-mode analysis, IRC
(Intrinsic Reaction Coordinate) confirmation, and Eyring TST rate
constants, starting from a CI-NEB climbing-image structure.

References
----------
Eyring TST : Eyring, J. Chem. Phys. 3, 107 (1935)
"""
import os
import warnings
from typing import Callable, Optional

import numpy as np

from ..constants import (
    ATOMIC_MASS,
    EV_ANG_TO_AMU_ANG_FS2,
    FREQ_RAD_FS_TO_WAVENUMBER_CM,
    K_B,
)
from ..io.poscar import write_poscar
from ..utils.mpi import (
    _HAS_MPI,
    _mpi_allreduce_sum,
    _mpi_domain_split,
    _mpi_rank,
    _mpi_size,
    local_only,
)
from .geometry_optimiser import GeometryOptimiser

__all__ = ["TransitionStateAnalyser"]

# -- Physical constant (Planck) ------------------------
H_PLANCK_EV_S = 4.135667696e-15   # Planck constant [eV.s]


class TransitionStateAnalyser:
    """
    Confirm a transition state and compute reaction properties.

    Starting from the climbing-image structure produced by NEBOptimiser,
    this class:

    1. Computes the Hessian matrix by central-difference finite differences
       (6N force evaluations).
    2. Analyses normal modes: eigenvalues, imaginary frequency, and the
       transition vector (negative-eigenvalue eigenvector).
    3. Confirms that the structure is a first-order saddle point (exactly
       one negative eigenvalue).
    4. Follows the transition vector in +/-delta and relaxes with FIRE to confirm
       connectivity to the intended initial and final states (IRC).
    5. Computes the Eyring TST reaction rate constant k(T) [s^-1].

    Parameters
    ----------
    lattice    : (3,3) lattice vectors [A].
    positions  : (N_atoms, 3) atomic positions of the candidate TS [A].
    atom_types : list of element symbols.
    force_fn   : unified force callable from make_force_fn().
    masses     : (N_atoms,) array of atomic masses [amu].
                 If None, ATOMIC_MASS is used to look up by element symbol.

    Examples
    --------
    >>> analyser = TransitionStateAnalyser(
    ...     lattice    = lattice,
    ...     positions  = neb_result['climbing_image_pos'],
    ...     atom_types = atom_types,
    ...     force_fn   = make_force_fn(engine))
    >>> analyser.compute_hessian(delta=0.01)
    >>> modes = analyser.analyse_modes()
    >>> print(analyser.is_first_order_saddle())   # True -> valid TS
    >>> irc = analyser.run_irc()
    >>> pos_A, pos_B = irc['pos_forward'], irc['pos_backward']
    >>> print(f"k(1000 K) = {analyser.reaction_rate(1000, E_init):.3e} s^-1")
    """

    # verbose flag for compute_hessian (set in analyse_modes)
    verbose_hessian: bool = True

    def __init__(self,
                 lattice    : np.ndarray,
                 positions  : np.ndarray,
                 atom_types : list,
                 force_fn   : Callable,
                 masses     : Optional[np.ndarray] = None):

        self.lattice    = lattice.copy()
        self.positions  = positions.copy()
        self.atom_types = list(atom_types)
        self.force_fn   = force_fn
        self.N          = len(atom_types)

        if masses is not None:
            self.masses = np.asarray(masses, dtype=np.float64)
        else:
            self.masses = np.array(
                [ATOMIC_MASS.get(a, 1.0) for a in atom_types])

        self._hessian      : Optional[np.ndarray] = None   # (3N, 3N) eV/A^2
        self._eigenvalues  : Optional[np.ndarray] = None   # (3N,) eV/A^2
        self._eigenvectors : Optional[np.ndarray] = None   # (3N, 3N)
        self._modes        : Optional[dict]       = None

    # ------------------------------------------------------------------ #
    #  1. Hessian matrix by central differences                           #
    # ------------------------------------------------------------------ #

    def compute_hessian(self, delta: float = 0.01) -> np.ndarray:
        """
        Compute the 3Nx3N Cartesian Hessian matrix by central-difference
        finite differences on atomic forces.

        .. math::

            H_{ij} = -\\frac{F_i(\\mathbf{r}+\\delta\\hat{e}_j)
                             - F_i(\\mathbf{r}-\\delta\\hat{e}_j)}{2\\delta}

        Cost: 6N force evaluations, split across MPI ranks (each
        displacement column is a fully independent force_fn call),
        matching the pattern already established and verified for
        :meth:`~akitsushima.solvers.phonon.PhononCalculator.compute_ifc`
        (also a finite-displacement, per-column force-constant build).

        Bug-fix note (Phase 5): this method previously ran the whole
        column loop identically on every rank with no MPI splitting at
        all (confirmed via grep -- the only class in this "finite
        displacement -> force-constant matrix" family without it).
        Reusing PhononCalculator's own ``local_only()`` discipline is
        required here for the same reason it was needed there: each
        rank's ``force_fn`` call for its own displaced structure would
        otherwise unconditionally Allreduce its internal pair/atom sums
        against every OTHER rank's call for a DIFFERENT displaced
        structure on the same global communicator, silently mixing
        unrelated columns together (no crash, no hang -- just a wrong
        Hessian). Verified directly with ``mpirun -np 1/3/5`` on a real
        LJ climbing-image structure: identical (bit-for-bit) Hessian,
        eigenvalues, and imaginary-mode count regardless of rank count.

        Parameters
        ----------
        delta : finite-difference displacement [A].  Default 0.01 A.

        Returns
        -------
        H : (3N, 3N) Hessian matrix [eV/A^2].
        """
        N3 = self.N * 3
        H_local = np.zeros((N3, N3))

        if self.verbose_hessian and _mpi_rank == 0:
            mpi_str = f", {_mpi_size} MPI ranks" if _HAS_MPI else ""
            print(f"  Computing Hessian ({6*self.N} force evaluations, "
                  f"delta={delta} A{mpi_str}) ...")

        i_start, i_end = _mpi_domain_split(N3)

        with local_only():
            for j in range(i_start, i_end):
                atom_j, cart_j = divmod(j, 3)

                pos_p = self.positions.copy(); pos_p[atom_j, cart_j] += delta
                pos_m = self.positions.copy(); pos_m[atom_j, cart_j] -= delta

                _, F_p, _ = self.force_fn(self.lattice, pos_p, self.atom_types)
                _, F_m, _ = self.force_fn(self.lattice, pos_m, self.atom_types)

                # H[:, j] = -dF/dr_j
                H_local[:, j] = -(F_p - F_m).ravel() / (2.0 * delta)

        H = _mpi_allreduce_sum(H_local)

        # Symmetrise (numerical asymmetry <= delta^2 / force_accuracy)
        H = 0.5 * (H + H.T)

        self._hessian = H
        self._eigenvalues  = None   # invalidate cache
        self._eigenvectors = None
        return H

    # ------------------------------------------------------------------ #
    #  2. Normal mode analysis                                             #
    # ------------------------------------------------------------------ #

    def analyse_modes(self,
                      delta          : float = 0.01,
                      mass_weighted  : bool  = True,
                      ) -> dict:
        """
        Compute eigenvalues, frequencies, and mode vectors.

        If the Hessian has not been computed yet, ``compute_hessian()``
        is called automatically with the provided ``delta``.

        Parameters
        ----------
        mass_weighted : if True, diagonalise the mass-weighted Hessian
                        (dynamical matrix) to get proper harmonic frequencies.
                        If False, diagonalise the raw Cartesian Hessian.

        Returns
        -------
        dict with keys:

        ``eigenvalues``      : (3N,) [eV/A^2] sorted ascending
        ``frequencies_cm``   : (3N,) [cm^-1] (imaginary -> negative sign)
        ``n_negative``       : int -- number of negative eigenvalues
        ``imag_freq_cm``     : float -- frequency of most negative mode [cm^-1]
        ``transition_vector``: (N, 3) normalised eigenvector of the most
                                negative eigenvalue (in Cartesian coordinates)
        ``is_ts``            : bool -- True when n_negative == 1
        ``eigenvalues_raw``  : (3N,) raw Cartesian eigenvalues (always present)
        """
        if self._hessian is None:
            self.compute_hessian(delta=delta)

        H = self._hessian   # (3N, 3N)
        N3 = self.N * 3

        if mass_weighted:
            # Mass-weighted Hessian D_ij = H_ij / sqrt(m_a m_b)
            m_vec = np.repeat(self.masses, 3)   # (3N,)
            M_inv = 1.0 / np.sqrt(np.outer(m_vec, m_vec) + 1e-300)
            D = H * M_inv
            D = 0.5 * (D + D.T)
        else:
            D = H

        eigenvalues, eigenvectors = np.linalg.eigh(D)
        # eigh returns ascending order

        # Convert eigenvalues -> frequencies
        # lambda [eV/A^2] (mass-weighted) or lambda/m [eV/(A^2.amu)]
        # omega^2 = lambda x EV_ANG_TO_AMU_ANG_FS2  [rad^2/fs^2]
        # nu [cm^-1] = omega/(2pi) / (2.99792458e-5 cm/fs)
        frequencies = np.zeros(N3)
        for i, lam in enumerate(eigenvalues):
            # For mass-weighted H, units are eV/A^2/amu -> omega^2 in rad^2/fs^2
            omega2 = lam * EV_ANG_TO_AMU_ANG_FS2
            if omega2 >= 0.0:
                frequencies[i] = np.sqrt(omega2) * FREQ_RAD_FS_TO_WAVENUMBER_CM
            else:
                frequencies[i] = -np.sqrt(-omega2) * FREQ_RAD_FS_TO_WAVENUMBER_CM

        # Raw Cartesian eigenvalues (for threshold-based counting)
        eig_raw, _ = np.linalg.eigh(H)

        n_negative = int(np.sum(eigenvalues < -1e-3))

        # Transition vector: mass-weighted -> convert back to Cartesian
        tv_mw = eigenvectors[:, 0]   # most negative
        if mass_weighted:
            m_vec = np.repeat(self.masses, 3)
            tv    = tv_mw / np.sqrt(m_vec + 1e-300)
        else:
            tv = tv_mw.copy()
        tv_norm = tv / (np.linalg.norm(tv) + 1e-16)
        tv_cart = tv_norm.reshape(self.N, 3)

        self._eigenvalues  = eigenvalues
        self._eigenvectors = eigenvectors

        result = {
            "eigenvalues"      : eigenvalues,
            "eigenvalues_raw"  : eig_raw,
            "frequencies_cm"   : frequencies,
            "n_negative"       : n_negative,
            "imag_freq_cm"     : float(frequencies[0]),   # most imaginary
            "transition_vector": tv_cart,
            "is_ts"            : n_negative == 1,
        }
        self._modes = result

        print(f"  Hessian eigenvalues: n_negative={n_negative}  "
              f"nu_imag={frequencies[0]:.1f} cm^-1  "
              f"{'[OK] 1st-order saddle' if n_negative == 1 else '[X] NOT a 1st-order saddle'}")
        return result

    # ------------------------------------------------------------------ #
    #  3. Is it a first-order saddle point?                               #
    # ------------------------------------------------------------------ #

    def is_first_order_saddle(self,
                               delta     : float = 0.01,
                               threshold : float = -1e-3) -> bool:
        """
        Return True when the structure has exactly one negative Hessian
        eigenvalue (first-order saddle point / true transition state).

        Parameters
        ----------
        threshold : eigenvalue threshold [eV/A^2] for classifying as negative.
                    Default -1e-3 eV/A^2 (ignores near-zero acoustic modes).

        Bug-fix note (S11): the original accepted a ``threshold`` argument
        but never used it -- it just returned ``self._modes["is_ts"]``,
        which is always computed with ``analyse_modes()``'s own hard-coded
        -1e-3 cutoff, silently ignoring whatever the caller passed here.
        The count is now redone with the actual ``threshold`` argument.
        """
        if self._modes is None:
            self.analyse_modes(delta=delta)
        n_negative = int(np.sum(self._modes["eigenvalues"] < threshold))
        return n_negative == 1

    # ------------------------------------------------------------------ #
    #  4. IRC: follow transition vector to the two minima                 #
    # ------------------------------------------------------------------ #

    def run_irc(self,
                irc_delta   : float = 0.1,
                fmax        : float = 0.05,
                max_steps   : int   = 500,
                skin_width  : float = 0.5,
                output_dir  : Optional[str] = None,
                verbose     : bool  = True,
                ) -> dict:
        """
        Follow the transition vector (most imaginary mode) in +/-irc_delta
        and relax each displaced structure to a local minimum with FIRE.

        This confirms that the TS connects the intended initial and final
        states (IRC = Intrinsic Reaction Coordinate).

        Parameters
        ----------
        irc_delta : displacement along the transition vector [A].
        fmax      : convergence criterion for FIRE relaxation [eV/A].
        max_steps : max FIRE steps for each side.
        output_dir: directory for POSCAR_IRC_forward, POSCAR_IRC_backward.
                    Defaults to current directory.
        verbose   : print FIRE progress.

        Returns
        -------
        dict with keys ``pos_forward``, ``pos_backward`` ((N, 3) relaxed
        structures along +/-TV), ``e_forward``, ``e_backward`` (their
        energies [eV]), and ``converged_forward``, ``converged_backward``
        (bool).

        Bug-fix note (S6): the original returned a bare
        ``(pos_forward, pos_backward)`` tuple, discarding the energy and
        convergence information that ``GeometryOptimiser.run()`` already
        computed for each side (the CLI layer needed exactly this
        information -- e.g. to report activation/reverse barriers -- and
        assumed a dict, raising ``AttributeError`` on the bare tuple).
        """
        if self._modes is None:
            self.analyse_modes()

        odir = output_dir or "."
        os.makedirs(odir, exist_ok=True)

        tv = self._modes["transition_vector"]   # (N, 3) unit vector

        results = {}
        for sign, label in [(+1.0, "forward"), (-1.0, "backward")]:
            pos_disp = self.positions + sign * irc_delta * tv

            if verbose:
                print(f"\n  IRC {label}: FIRE relaxation from +/-{irc_delta:.3f} A "
                      f"displacement ...")

            go = GeometryOptimiser(
                lattice    = self.lattice,
                positions  = pos_disp,
                atom_types = self.atom_types,
                force_fn   = self.force_fn,
                fmax       = fmax,
                max_steps  = max_steps,
                skin_width = skin_width,
                output_dir = os.path.join(odir, f"irc_{label}"),
                verbose    = verbose,
            )
            res = go.run()

            out_path = os.path.join(odir, f"POSCAR_IRC_{label}")
            write_poscar(out_path, self.lattice, res["positions"],
                         self.atom_types,
                         title=(f"IRC {label}  "
                                   f"E={res['energy']:.6f} eV  "
                                   f"converged={res['converged']}"))
            print(f"  IRC {label:8s} -> {out_path}  "
                  f"E={res['energy']:.4f} eV  "
                  f"converged={res['converged']}")
            results[label] = res

        return {
            "pos_forward"        : results["forward"]["positions"],
            "pos_backward"       : results["backward"]["positions"],
            "e_forward"          : results["forward"]["energy"],
            "e_backward"         : results["backward"]["energy"],
            "converged_forward"  : results["forward"]["converged"],
            "converged_backward" : results["backward"]["converged"],
        }

    # ------------------------------------------------------------------ #
    #  5. Eyring TST reaction rate constant                               #
    # ------------------------------------------------------------------ #

    def reaction_rate(self,
                      T         : float,
                      E_init    : Optional[float] = None,
                      Ea        : Optional[float] = None,
                      delta_hess: float = 0.01,
                      wigner    : bool  = False,
                      ) -> float:
        """
        Compute the Eyring transition-state theory (TST) rate constant.

        .. math::

            k(T) = \\kappa \\frac{k_B T}{h}
                   \\frac{Q_{\\rm TS}}{Q_{\\rm IS}}
                   \\exp\\!\\left(-\\frac{E_a}{k_B T}\\right)

        where the partition function ratio is approximated in the harmonic
        oscillator limit.  The imaginary frequency mode is excluded from
        the product.

        Parameters
        ----------
        T       : temperature [K].
        E_init  : energy of the initial (stable) state [eV].
                  If None, Ea must be supplied directly.
        Ea      : activation energy [eV].  Overrides E_init-based calculation.
        delta_hess : finite-difference delta for Hessian if not yet computed [A].
        wigner  : apply first-order Wigner quantum tunnelling correction
                  (default False).

        Returns
        -------
        k_TST : float [s^-1]
        """
        if self._modes is None:
            self.analyse_modes(delta=delta_hess)

        # Activation energy
        if Ea is None:
            if E_init is None:
                raise ValueError(
                    "reaction_rate: supply either Ea= or E_init=.")
            E_ts, _, _ = self.force_fn(self.lattice, self.positions,
                                       self.atom_types)
            Ea = float(E_ts) - float(E_init)

        if Ea < 0.0:
            warnings.warn(f"reaction_rate: Ea={Ea:.4f} eV < 0  "
                    f"(initial state is higher in energy than TS?)",
                    UserWarning)

        kT = K_B * T          # eV
        h  = H_PLANCK_EV_S    # eV.s

        # --- Partition function ratio (harmonic, T > 0) -------------------
        # Bug-fix note (S11): the original had a for-loop over
        # freqs_cm[1:] here that never assigned to any variable (an
        # `if ... continue` / `if ... pass` with no else and no
        # accumulation) -- i.e. dead code that always left q_ts_q_is at
        # its initial 1.0 while looking like it was computing something.
        # A full TST partition-function ratio would need the *initial*
        # state's Hessian too (not just the TS's, which is all this class
        # computes), so this remains the simplified Q_TS/Q_IS = 1
        # (plain Arrhenius-with-TST-prefactor) approximation -- now
        # stated directly instead of via an inert loop.
        q_ts_q_is = 1.0

        # Wigner tunnelling correction: kappa ~= 1 + (1/24)(h nu_imag/kT)^2
        freqs_cm = self._modes["frequencies_cm"]   # [cm^-1], signed
        c_cm_s = 2.99792458e10   # cm/s
        hc_eV  = H_PLANCK_EV_S * c_cm_s  # eV.cm

        k_base = (kT / h) * q_ts_q_is * np.exp(-Ea / kT)

        kappa = 1.0
        if wigner and abs(freqs_cm[0]) > 10.0:
            nu_imag_eV = abs(freqs_cm[0]) * hc_eV
            kappa = 1.0 + (1.0 / 24.0) * (nu_imag_eV / kT) ** 2

        k_tst = kappa * k_base

        print(f"  TST rate  T={T:.0f} K  Ea={Ea:.4f} eV  "
              f"k(T) = {k_tst:.4e} s^-1"
              + (f"  (kappa_Wigner={kappa:.4f})" if wigner else ""))
        return float(k_tst)

    # ------------------------------------------------------------------ #
    #  Summary report                                                      #
    # ------------------------------------------------------------------ #

    def report(self,
               output_dir : str  = ".",
               delta      : float = 0.01) -> dict:
        """
        Run the full TS analysis pipeline and write a summary CSV.

        Steps: compute_hessian -> analyse_modes -> is_first_order_saddle.

        Returns
        -------
        dict with Hessian analysis results, suitable for printing or saving.
        """
        if self._hessian is None:
            self.compute_hessian(delta=delta)
        modes = self.analyse_modes()

        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "ts_modes.csv")
        freqs    = modes["frequencies_cm"]
        eigs_raw = modes["eigenvalues_raw"]

        with open(csv_path, 'w') as fh:
            fh.write("mode,eigenvalue_eV_Ang2,frequency_cm-1\n")
            for i, (ev, nu) in enumerate(zip(eigs_raw, freqs)):
                fh.write(f"{i},{ev:.6e},{nu:.4f}\n")
        print(f"  Mode CSV   -> {csv_path}")

        # Save transition vector as POSCAR-style displacement
        tv_path = os.path.join(output_dir, "transition_vector.csv")
        tv      = modes["transition_vector"]
        with open(tv_path, 'w') as fh:
            fh.write("atom,species,tv_x,tv_y,tv_z\n")
            for i, (sp, row) in enumerate(zip(self.atom_types, tv)):
                fh.write(f"{i},{sp},{row[0]:.6f},{row[1]:.6f},{row[2]:.6f}\n")
        print(f"  TV CSV     -> {tv_path}")

        return modes
