"""Unified force-function interface for FIRE / MD / NEB / phonon solvers.

A "force function" is any callable::

    fn(lattice, positions, atom_types, neigh_list=None)
        -> (energy: float, forces: np.ndarray, stress: np.ndarray)

``make_force_fn()`` builds this callable from the most common backends:
  'buckingham'     : Buckingham + Coulomb (built internally as a
                     one-potential PotentialEngine + compute_all_general).
  PotentialEngine  : arbitrary PotentialEngine (LJ, Morse, BornMayer, SW, ...).
  ELMModel         : ML potential (numerical or analytical differentiation).
  callable         : pass your own function directly.

``.kind`` / ``.engine`` attributes (bug-fix note, S12/F13)
-----------------------------------------------------------
The original code let ``solvers/`` introspect a force_fn's *identity* by
either an ``isinstance(force_fn, ESMCalculator)`` check or, in
``OSMMCSimulation._get_lj_params``, by reaching directly into the
closure's ``__closure__`` cells to find a captured ``buck_params``-like
variable.  Both are brittle: the isinstance check only ever worked for
the one hard-coded class, and the closure introspection breaks the
moment the closure's captured-variable order or count changes (exactly
what happened here, since this refactor stopped inlining a bare
Buckingham closure and instead always builds a real
``PotentialEngine``).  Every force_fn returned by ``make_force_fn`` now
carries two public, introspectable attributes instead:

  fn.kind    : one of ``"buckingham"``, ``"engine"``, ``"elm"``, or
               ``None`` for a passed-through user callable that doesn't
               declare its own ``.kind`` (e.g. ``ESMCalculator`` sets
               ``self.kind = "esm"`` itself, and that is preserved
               here rather than overwritten).
  fn.engine  : the underlying ``PotentialEngine`` when one exists
               (``"buckingham"`` and ``"engine"`` kinds); absent for
               ``"elm"`` and arbitrary callables.

Solver code should use ``getattr(force_fn, "kind", None)`` /
``getattr(force_fn, "engine", None)`` rather than ``isinstance`` or
closure inspection.
"""
import warnings
from typing import Callable, Optional

from ..engine import BuckinghamPotential, PotentialEngine, compute_all_general
from .models.elm import ELMModel
from .models.quantized_elm import QuantizedELMModel

__all__ = ["compute_all", "make_force_fn", "_resolve_force_fn"]


def compute_all(lattice         ,
                positions       ,
                atom_types      : list,
                buck_params     : dict,
                charges         : dict,
                r_cut           : float,
                ewald_alpha     : Optional[float] = None,
                coulomb_method  : str   = "ewald",
                pppm_accuracy   : float = 1e-5,
                pppm_mesh       : Optional[int] = None,
                pppm_order      : int   = 6,
                neigh_list      = None,
                use_numba       : bool  = False,
                ):
    """
    Combined Buckingham + Coulomb energy, forces, and stress.

    V_total = V_Buckingham(r) + V_Coulomb(r)

    Standard model for ionic crystals (MgO, NaCl, SiO2, etc).
    For covalent / metallic systems use compute_all_general() with
    a PotentialEngine instead.

    Consolidation note: this used to be a hand-written, standalone
    implementation with its own module-level PPPM solver cache (a
    plain ``dict`` that called ``.clear()`` on every miss -- see the
    engine-side bug-fix note on ``_pppm_cache`` in
    ``engine/potential_engine.py``, bug E6).  It is now a thin wrapper
    around :class:`~akitsushima.engine.BuckinghamPotential` +
    :func:`~akitsushima.engine.compute_all_general`, so it automatically
    inherits every engine-level bug fix (Ewald/PPPM sign and
    normalisation fixes, the shared LRU PPPM cache, MPI domain
    splitting) with no duplicated logic.

    Parameters
    ----------
    buck_params    : dict  frozenset({sp1,sp2}) -> {'A','rho','C'}.
    charges        : dict  symbol -> formal charge [e].
    r_cut          : real-space cutoff [A].
    ewald_alpha    : Ewald alpha [A^-1]; auto-set to 5/r_cut if None.
    coulomb_method : 'ewald' (default) | 'pppm'.
    pppm_accuracy  : target relative accuracy for PPPM (default 1e-5).
    pppm_mesh      : PPPM mesh size (None = auto).
    pppm_order     : B-spline order for PPPM (4, 6, or 8).
    neigh_list     : NeighbourList (optional).
    use_numba      : Use Numba JIT for the short-range pair loop.

    Returns
    -------
    energy : float        [eV]
    forces : (N,3) array  [eV/A]
    stress : (3,3) array  [eV/A^3]
    """
    engine = PotentialEngine().add(BuckinghamPotential(buck_params, r_cut=r_cut))
    return compute_all_general(
        lattice, positions, atom_types, engine,
        charges        = charges,
        r_cut          = r_cut,
        ewald_alpha    = ewald_alpha,
        coulomb_method = coulomb_method,
        pppm_accuracy  = pppm_accuracy,
        pppm_mesh      = pppm_mesh,
        pppm_order     = pppm_order,
        neigh_list     = neigh_list,
        use_numba      = use_numba,
    )


def make_force_fn(backend,
                  *,
                  # Buckingham kwargs
                  buck_params    : Optional[dict]  = None,
                  charges        : Optional[dict]  = None,
                  r_cut          : Optional[float] = None,
                  ewald_alpha    : Optional[float] = None,
                  coulomb_method : str             = "ewald",
                  pppm_accuracy  : float           = 1e-5,
                  pppm_mesh      : Optional[int]   = None,
                  pppm_order     : int             = 6,
                  # ELM kwargs
                  descriptor_fn  : Optional[Callable] = None,
                  ) -> Callable:
    """
    Build a unified force function for use with GeometryOptimiser, NVTSimulation,
    NVESimulation, and NPTSimulation.

    Signature of the returned callable::

        fn(lattice    : np.ndarray,   # (3,3)
           positions  : np.ndarray,   # (N,3)
           atom_types : list,
           neigh_list = None,
           ) -> (energy: float, forces: (N,3) array, stress: (3,3) array)

    Parameters
    ----------
    backend : one of:
        ``'buckingham'``
            Buckingham + Coulomb.  Requires buck_params=, charges=, r_cut=.
        PotentialEngine instance
            Calls engine.compute().  r_cut is auto-detected from the engine's
            maximum internal cutoff (engine.r_cut property).  Pass r_cut=
            explicitly to override (e.g. when the NeighbourList should use
            a different value than the potential cutoff).
        ELMModel / QuantizedELMModel instance
            Numerical differentiation of (1/2) sum_i.
            Requires descriptor_fn=.  r_cut is not auto-detected; pass
            r_cut= to the simulator explicitly.
        callable
            Passed through as-is.  If it already carries its own
            ``.r_cut`` (e.g. ``ExternalFileEngine``) and no ``r_cut=``
            override is given here, that value is preserved rather than
            being reset to ``None``.

    r_cut : float or None
        Short-range cutoff [A] used for the NeighbourList in the simulator.
        - 'buckingham': required (no default).
        - PotentialEngine: auto-detected from engine.r_cut if None; a
          UserWarning is raised when the provided r_cut is smaller than
          the engine's maximum cutoff (would miss pairs).
        - ELMModel / callable: optional; the simulator uses this value
          directly for NeighbourList construction.

    Notes on r_cut and NeighbourList
    ---------------------------------
    The simulator's NeighbourList is built with the r_cut passed to the
    simulator constructor (force_fn=..., r_cut=...).  For a PotentialEngine
    this should equal the potential's cutoff; passing a smaller value
    silently misses atom pairs.  make_force_fn() checks for this and warns.

    Examples
    --------
    >>> # Buckingham + Coulomb (ionic crystals)
    >>> fn = make_force_fn('buckingham',
    ...         buck_params=bp, charges=ch, r_cut=8.0)

    >>> # LJ / Morse / BornMayer via PotentialEngine
    >>> engine = PotentialEngine().add(LennardJonesPotential(
    ...     {frozenset({'Ar','Ar'}): {'epsilon': 0.0103, 'sigma': 3.405}},
    ...     r_cut=10.0))
    >>> fn = make_force_fn(engine)
    >>> # r_cut is auto-detected as 10.0 from the engine
    >>> # Pass it to the simulator:  NVTSimulation(..., force_fn=fn, r_cut=fn.r_cut)

    >>> # ELMModel
    >>> def desc(lat, pos, types): return compute_soap(lat, pos, types)
    >>> fn = make_force_fn(elm_model, descriptor_fn=desc)
    """
    # -- callable passthrough (excludes ELMModel which is also callable) ----
    if callable(backend) and not isinstance(
            backend, (PotentialEngine, ELMModel, QuantizedELMModel)):
        fn = backend
        # Bug-fix note: the original unconditionally did `fn.r_cut = r_cut`,
        # which silently overwrote an already-configured attribute (e.g.
        # ExternalFileEngine's own self.r_cut) with None whenever the
        # caller forgot to re-pass r_cut= here.  Only override when an
        # explicit value was given; otherwise keep whatever the callable
        # already carries (defaulting to None only if it truly has none).
        if r_cut is not None:
            fn.r_cut = float(r_cut)
        elif not hasattr(fn, 'r_cut'):
            fn.r_cut = None
        if not hasattr(fn, 'kind'):
            fn.kind = None   # e.g. ESMCalculator already sets its own "esm"
        return fn

    # -- Buckingham + Coulomb ------------------------------
    if backend == 'buckingham':
        if buck_params is None or charges is None:
            raise ValueError(
                "make_force_fn('buckingham'): buck_params= and charges= required.")
        if r_cut is None:
            raise ValueError(
                "make_force_fn('buckingham'): r_cut= is required.")
        _rc     = float(r_cut)
        _engine = PotentialEngine().add(BuckinghamPotential(buck_params, r_cut=_rc))

        def _buck(lattice, positions, atom_types, neigh_list=None):
            return compute_all_general(
                lattice, positions, atom_types, _engine,
                charges        = charges,
                r_cut          = _rc,
                ewald_alpha    = ewald_alpha,
                coulomb_method = coulomb_method,
                pppm_accuracy  = pppm_accuracy,
                pppm_mesh      = pppm_mesh,
                pppm_order     = pppm_order,
                neigh_list     = neigh_list)
        _buck.r_cut  = _rc
        _buck.kind   = "buckingham"
        _buck.engine = _engine
        return _buck

    # -- PotentialEngine -----------------------------------
    if isinstance(backend, PotentialEngine):
        _eng         = backend
        _eng_r_cut   = float(_eng.r_cut)   # maximum cutoff from all potentials

        if r_cut is None:
            _rc = _eng_r_cut          # auto-detect from engine
        else:
            _rc = float(r_cut)
            if _rc < _eng_r_cut - 1e-6:
                warnings.warn(
                    f"make_force_fn: simulator r_cut={_rc:.3f} A is smaller than "
                    f"the PotentialEngine's maximum cutoff {_eng_r_cut:.3f} A.  "
                    f"Atom pairs beyond r_cut will be missed by the NeighbourList. "
                    f"Set r_cut>={_eng_r_cut:.3f} A or omit r_cut= to auto-detect.",
                    UserWarning, stacklevel=2)

        from ..utils.jit import _NUMBA_ENV

        def _engine_fn(lattice, positions, atom_types, neigh_list=None):
            return _eng.compute(lattice, positions, atom_types,
                                neigh_list=neigh_list,
                                use_numba=_NUMBA_ENV)   # honour NUMBA_ENABLE env var
        _engine_fn.r_cut  = _rc
        _engine_fn.kind   = "engine"
        _engine_fn.engine = _eng
        return _engine_fn

    # -- ELMModel / QuantizedELMModel ----------------------
    if isinstance(backend, (ELMModel, QuantizedELMModel)):
        if not backend._fitted:
            raise ValueError("make_force_fn: ELMModel has not been fitted.")
        if descriptor_fn is None:
            raise ValueError(
                "make_force_fn: descriptor_fn= is required for ELMModel.\n"
                "  descriptor_fn(lattice, positions, atom_types) -> (N, D) array")
        _elm  = backend
        _desc = descriptor_fn
        _rc   = float(r_cut) if r_cut is not None else 10.0

        def _elm_fn(lattice, positions, atom_types, neigh_list=None):
            return _elm.compute(lattice, positions, atom_types,
                                descriptor_fn=_desc)
        _elm_fn.r_cut = _rc
        _elm_fn.kind  = "elm"
        return _elm_fn

    raise ValueError(
        f"make_force_fn: unsupported backend {type(backend)}. "
        "Pass 'buckingham', a PotentialEngine, an ELMModel, or a callable.")


def _legacy_force_fn(buck_params, charges, r_cut, ewald_alpha,
                      coulomb_method, pppm_accuracy, pppm_mesh, pppm_order
                      ) -> Callable:
    """Build a Buckingham force_fn from legacy keyword arguments."""
    return make_force_fn('buckingham',
                         buck_params=buck_params, charges=charges,
                         r_cut=r_cut, ewald_alpha=ewald_alpha,
                         coulomb_method=coulomb_method,
                         pppm_accuracy=pppm_accuracy,
                         pppm_mesh=pppm_mesh, pppm_order=pppm_order)


def _resolve_force_fn(force_fn, r_cut,
                      buck_params, charges, ewald_alpha,
                      coulomb_method, pppm_accuracy, pppm_mesh, pppm_order,
                      caller_name):
    """
    Resolve force_fn vs Buckingham kwargs, and derive r_cut from force_fn.

    Rules
    -----
    - If force_fn is given and r_cut is None, use force_fn.r_cut.
    - If buck_params/charges are given, build a legacy force_fn (r_cut required).
    - Otherwise raise ValueError.
    """
    if force_fn is not None:
        if r_cut is None:
            if hasattr(force_fn, 'r_cut') and force_fn.r_cut is not None:
                r_cut = float(force_fn.r_cut)
            else:
                raise ValueError(
                    f"{caller_name}: r_cut= is required when force_fn does not "
                    "carry a .r_cut attribute.  Set r_cut= explicitly.")
        return force_fn, float(r_cut)
    elif buck_params is not None and charges is not None:
        if r_cut is None:
            raise ValueError(f"{caller_name}: r_cut= is required with buck_params=.")
        fn = _legacy_force_fn(buck_params, charges, float(r_cut), ewald_alpha,
                              coulomb_method, pppm_accuracy, pppm_mesh, pppm_order)
        return fn, float(r_cut)
    else:
        raise ValueError(
            f"{caller_name}: provide force_fn= or (buck_params=, charges=, r_cut=).")
