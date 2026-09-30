"""Fittable parameter sets for classical pair/many-body potentials.

Each :class:`FittableParameterSet` subclass maps a flat optimisation
vector ``x`` (as used by ``scipy.optimize``) to a physical parameter
dict via :meth:`decode`, and builds a ready-to-use
:class:`~akitsushima.engine.PotentialEngine` from those parameters via
:meth:`build_engine`.  Positive parameters are optimised in log-space so
that unconstrained/box-constrained optimisers never visit unphysical
negative values.
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..engine import (
    BornMayerPotential,
    BuckinghamPotential,
    LennardJonesPotential,
    MiePotential,
    MorsePotential,
    PotentialEngine,
    StillingerWeberPotential,
    TersoffPotential,
    YukawaPotential,
)

__all__ = [
    "FittableParameterSet",
    "BuckinghamParameterSet",
    "LJParameterSet",
    "MorseParameterSet",
    "BornMayerParameterSet",
    "MieParameterSet",
    "YukawaParameterSet",
    "StillingerWeberParameterSet",
    "MultiSpeciesStillingerWeberParameterSet",
    "TersoffParameterSet",
    "MultiSpeciesTersoffParameterSet",
    "Tersoff1989ParameterSet",
    "ZhouEAMParameterSet",
]


class FittableParameterSet(ABC):
    """
    Abstract base class for parameter sets that can be optimised against
    DFT reference data.

    Subclasses must implement:
      x0         : np.ndarray  -- initial optimisation vector
      bounds     : list of (lo, hi) tuples
      decode(x)  : np.ndarray -> dict of force-field parameters
      build_engine(params) : dict -> PotentialEngine
    """

    @property
    @abstractmethod
    def x0(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def bounds(self) -> list: ...

    @abstractmethod
    def decode(self, x: np.ndarray) -> dict:
        """Return force-field parameter dict from optimisation vector x."""

    @abstractmethod
    def build_engine(self, params: dict) -> PotentialEngine:
        """Build a PotentialEngine from decoded parameters."""


# ---------------------------------------------------------------------------
#  BuckinghamParameterSet
# ---------------------------------------------------------------------------

class BuckinghamParameterSet(FittableParameterSet):
    """
    Parameter set for Buckingham + optional Coulomb fitting.

    V(r) = A . exp(-r/rho) - C/r^6

    Optimisation variables (log-space for positivity)
    -------------------------------------------------
    Per pair: log(A), log(rho), C  (C >= 0 by floor)
    Per species (when fix_charges=False): q_i (unbounded)
    """

    def __init__(self,
                 pairs        : list,
                 species      : list,
                 initial_buck : dict,
                 charges      : dict,
                 r_cut        : float = 10.0,
                 shift        : bool  = True,
                 fix_charges  : bool  = True):
        self.pairs        = pairs
        self.pair_list    = list(pairs)
        self.species      = species
        self.charges      = charges.copy()
        self.fix_charges  = fix_charges
        self.r_cut        = r_cut
        self.shift        = shift

        _x0, _lo, _hi = [], [], []
        for pair in self.pair_list:
            p = initial_buck[pair]
            _x0 += [np.log(max(p["A"], 1e-6)), np.log(p["rho"]), p["C"]]
            _lo  += [np.log(1.0),  np.log(0.01), 0.0]
            _hi  += [np.log(1e7),  np.log(5.0),  1e4]

        if not fix_charges:
            for sp in self.species:
                _x0.append(charges[sp])
                _lo.append(-6.0); _hi.append(6.0)

        self._x0     = np.array(_x0)
        self._bounds = list(zip(_lo, _hi))

    @property
    def x0(self) -> np.ndarray:
        return self._x0.copy()

    @property
    def bounds(self) -> list:
        return self._bounds

    def decode(self, x: np.ndarray) -> dict:
        """Return {'buck': {pair: {A,rho,C}}, 'charges': {sp: q}}."""
        buck = {}
        idx  = 0
        for pair in self.pair_list:
            A   = np.exp(x[idx]);   idx += 1
            rho = np.exp(x[idx]);   idx += 1
            C   = max(x[idx], 0.0); idx += 1
            buck[pair] = {"A": A, "rho": rho, "C": C}
        charges = self.charges.copy()
        if not self.fix_charges:
            for sp in self.species:
                charges[sp] = x[idx]; idx += 1
        return {"buck": buck, "charges": charges}

    def build_engine(self, params: dict) -> PotentialEngine:
        """Build PotentialEngine from decoded Buckingham params.

        Bug-fix note (F4): the original implementation split this into a
        BornMayerPotential(A, rho) and silently dropped the dispersion
        term C, so every fit_potential()/fit_potential-based diagnostic
        for Buckingham ran with C implicitly forced to zero.  Now that
        BuckinghamPotential(A, rho, C) exists as a first-class
        BasePairPotential subclass, build it directly from the full
        decoded dict.

        Second bug-fix note: this used to hard-code ``r_cut = 10.0``,
        ignoring the cutoff the caller and the reference data actually
        use -- every other parameter set in this module passes
        ``self.r_cut``.  While the potential had no force shift the
        mismatch was invisible, because the neighbour list only ever
        supplied pairs out to the caller's (smaller) cutoff, so the
        engine's larger one never bit.  With the shift it does: the
        subtracted constant V(r_cut) is evaluated at the engine's r_cut,
        so a mismatch biases every pair.  Surfaced by
        test_evaluate_diagnostics_generic_includes_coulomb, whose energy
        MAE jumped from below 1e-6 to 6.1e-04 the moment Buckingham
        gained a shift.
        """
        return PotentialEngine().add(
            BuckinghamPotential(params["buck"], r_cut=self.r_cut,
                                shift=self.shift))


# ---------------------------------------------------------------------------
#  LJParameterSet
# ---------------------------------------------------------------------------

class LJParameterSet(FittableParameterSet):
    """
    Parameter set for Lennard-Jones 12-6 fitting.

    V(r) = 4epsilon [(sigma/r)^12 - (sigma/r)^6]

    Optimisation variables
    ----------------------
    Per pair: log(epsilon), log(sigma)
    Bounds:  epsilon in [1e-5, 10] eV,  sigma in [0.5, 10] A

    Parameters
    ----------
    pairs         : list of frozenset({sp1, sp2}) pairs to fit.
    species       : list of all element symbols.
    initial_lj    : dict  frozenset({sp1,sp2}) -> {'epsilon': epsilon, 'sigma': sigma}
    r_cut         : interaction cut-off [A].
    shift         : apply force-shifted truncation (default True).
    """

    def __init__(self,
                 pairs      : list,
                 species    : list,
                 initial_lj : dict,
                 r_cut      : float = 10.0,
                 shift      : bool  = True):
        self.pair_list = list(pairs)
        self.species   = species
        self.r_cut     = r_cut
        self.shift     = shift

        _x0, _lo, _hi = [], [], []
        for pair in self.pair_list:
            p = initial_lj[pair]
            _x0 += [np.log(max(p["epsilon"], 1e-8)), np.log(p["sigma"])]
            _lo  += [np.log(1e-5), np.log(0.5)]
            _hi  += [np.log(10.0), np.log(10.0)]

        self._x0     = np.array(_x0)
        self._bounds = list(zip(_lo, _hi))

    @property
    def x0(self) -> np.ndarray:
        return self._x0.copy()

    @property
    def bounds(self) -> list:
        return self._bounds

    def decode(self, x: np.ndarray) -> dict:
        """Return {'lj': {pair: {'epsilon': epsilon, 'sigma': sigma}}}."""
        lj  = {}
        idx = 0
        for pair in self.pair_list:
            eps  = np.exp(x[idx]); idx += 1
            sig  = np.exp(x[idx]); idx += 1
            lj[pair] = {"epsilon": eps, "sigma": sig}
        return {"lj": lj}

    def build_engine(self, params: dict) -> PotentialEngine:
        return (PotentialEngine()
                .add(LennardJonesPotential(
                    params["lj"], r_cut=self.r_cut, shift=self.shift)))


# ---------------------------------------------------------------------------
#  MorseParameterSet
# ---------------------------------------------------------------------------

class MorseParameterSet(FittableParameterSet):
    """
    Parameter set for Morse potential fitting.

    V(r) = D_e [1 - exp(-alpha(r - r0))]^2 - D_e

    Optimisation variables
    ----------------------
    Per pair: log(D_e), log(alpha), r0
    Bounds:  D_e in [0.01, 100] eV,  alpha in [0.1, 10] A^-1,  r0 in [0.5, 6] A

    Parameters
    ----------
    pairs         : list of frozenset({sp1, sp2}) pairs to fit.
    species       : list of all element symbols.
    initial_morse : dict  frozenset({sp1,sp2}) -> {'D_e':val, 'alpha':val, 'r0':val}
    r_cut         : interaction cut-off [A].
    shift         : apply force-shifted truncation (default True).
    """

    def __init__(self,
                 pairs         : list,
                 species       : list,
                 initial_morse : dict,
                 r_cut         : float = 10.0,
                 shift         : bool  = True):
        self.pair_list = list(pairs)
        self.species   = species
        self.r_cut     = r_cut
        self.shift     = shift

        _x0, _lo, _hi = [], [], []
        for pair in self.pair_list:
            p = initial_morse[pair]
            _x0 += [np.log(max(p["D_e"],   1e-4)),
                    np.log(max(p["alpha"],  1e-4)),
                    p["r0"]]
            _lo  += [np.log(0.01), np.log(0.1), 0.5]
            _hi  += [np.log(100.),  np.log(10.), 6.0]

        self._x0     = np.array(_x0)
        self._bounds = list(zip(_lo, _hi))

    @property
    def x0(self) -> np.ndarray:
        return self._x0.copy()

    @property
    def bounds(self) -> list:
        return self._bounds

    def decode(self, x: np.ndarray) -> dict:
        """Return {'morse': {pair: {'D_e': val, 'alpha': val, 'r0': val}}}."""
        morse = {}
        idx   = 0
        for pair in self.pair_list:
            De    = np.exp(x[idx]); idx += 1
            alpha = np.exp(x[idx]); idx += 1
            r0    = x[idx];          idx += 1
            morse[pair] = {"D_e": De, "alpha": alpha, "r0": r0}
        return {"morse": morse}

    def build_engine(self, params: dict) -> PotentialEngine:
        return (PotentialEngine()
                .add(MorsePotential(
                    params["morse"], r_cut=self.r_cut, shift=self.shift)))


# ---------------------------------------------------------------------------
#  BornMayerParameterSet
# ---------------------------------------------------------------------------

class BornMayerParameterSet(FittableParameterSet):
    """
    Parameter set for Born--Mayer (short-range) + optional Coulomb fitting.

    V_SR(r) = A . exp(-r/rho)
    V_total(r) = V_SR(r) + q_i q_j / r  (Ewald/PPPM for the Coulomb part)

    Optimisation variables
    ----------------------
    Per pair: log(A), log(rho)
    Per species (fix_charges=False): q_i
    Bounds:  A in [1, 1e7] eV,  rho in [0.01, 5] A

    Parameters
    ----------
    pairs         : list of frozenset({sp1, sp2}) pairs to fit.
    species       : list of all element symbols.
    initial_bm    : dict  frozenset({sp1,sp2}) -> {'A': val, 'rho': val}
    charges       : dict  symbol -> charge [e]  (fixed when fix_charges=True).
    r_cut         : interaction cut-off [A].
    fix_charges   : if True, charges are not optimised.
    """

    def __init__(self,
                 pairs       : list,
                 species     : list,
                 initial_bm  : dict,
                 charges     : dict,
                 r_cut       : float = 10.0,
                 shift       : bool  = True,
                 fix_charges : bool  = True):
        self.pair_list  = list(pairs)
        self.species    = species
        self.charges    = charges.copy()
        self.fix_charges = fix_charges
        self.r_cut      = r_cut
        self.shift      = shift

        _x0, _lo, _hi = [], [], []
        for pair in self.pair_list:
            p = initial_bm[pair]
            _x0 += [np.log(max(p["A"], 1.0)), np.log(p["rho"])]
            _lo  += [np.log(1.0),  np.log(0.01)]
            _hi  += [np.log(1e7),  np.log(5.0)]

        if not fix_charges:
            for sp in self.species:
                _x0.append(charges[sp])
                _lo.append(-6.0); _hi.append(6.0)

        self._x0     = np.array(_x0)
        self._bounds = list(zip(_lo, _hi))

    @property
    def x0(self) -> np.ndarray:
        return self._x0.copy()

    @property
    def bounds(self) -> list:
        return self._bounds

    def decode(self, x: np.ndarray) -> dict:
        """Return {'bm': {pair: {A, rho}}, 'charges': {sp: q}}."""
        bm  = {}
        idx = 0
        for pair in self.pair_list:
            A   = np.exp(x[idx]); idx += 1
            rho = np.exp(x[idx]); idx += 1
            bm[pair] = {"A": A, "rho": rho}
        charges = self.charges.copy()
        if not self.fix_charges:
            for sp in self.species:
                charges[sp] = x[idx]; idx += 1
        return {"bm": bm, "charges": charges}

    def build_engine(self, params: dict) -> PotentialEngine:
        return (PotentialEngine()
                .add(BornMayerPotential(params["bm"], r_cut=self.r_cut,
                                       shift=self.shift)))


# ---------------------------------------------------------------------------
#  MieParameterSet
# ---------------------------------------------------------------------------

class MieParameterSet(FittableParameterSet):
    """
    Parameter set for Mie potential fitting.

    V(r) = C(n,m) . epsilon . [(sigma/r)^n - (sigma/r)^m]

    Optimisation variables
    ----------------------
    Per pair: log(epsilon), log(sigma)  -- same bounds as LJParameterSet.

    n, m (the repulsive/attractive exponents) are fixed hyperparameters,
    not fit -- see MiePotential's docstring for why.

    Parameters
    ----------
    pairs       : list of frozenset({sp1, sp2}) pairs to fit.
    species     : list of all element symbols.
    initial_mie : dict  frozenset({sp1,sp2}) -> {'epsilon': epsilon, 'sigma': sigma}
    n, m        : repulsive / attractive exponents (fixed, not fit).
                  Default (12, 6) reproduces standard Lennard-Jones.
    r_cut       : interaction cut-off [A].
    shift       : apply force-shifted truncation (default True).
    """

    def __init__(self,
                 pairs       : list,
                 species     : list,
                 initial_mie : dict,
                 n           : float = 12.0,
                 m           : float = 6.0,
                 r_cut       : float = 10.0,
                 shift       : bool  = True):
        self.pair_list = list(pairs)
        self.species   = species
        self.n         = float(n)
        self.m         = float(m)
        self.r_cut     = r_cut
        self.shift     = shift

        _x0, _lo, _hi = [], [], []
        for pair in self.pair_list:
            p = initial_mie[pair]
            _x0 += [np.log(max(p["epsilon"], 1e-8)), np.log(p["sigma"])]
            _lo  += [np.log(1e-5), np.log(0.5)]
            _hi  += [np.log(10.0), np.log(10.0)]

        self._x0     = np.array(_x0)
        self._bounds = list(zip(_lo, _hi))

    @property
    def x0(self) -> np.ndarray:
        return self._x0.copy()

    @property
    def bounds(self) -> list:
        return self._bounds

    def decode(self, x: np.ndarray) -> dict:
        """Return {'mie': {pair: {'epsilon': epsilon, 'sigma': sigma}}}."""
        mie = {}
        idx = 0
        for pair in self.pair_list:
            eps = np.exp(x[idx]); idx += 1
            sig = np.exp(x[idx]); idx += 1
            mie[pair] = {"epsilon": eps, "sigma": sig}
        return {"mie": mie}

    def build_engine(self, params: dict) -> PotentialEngine:
        return (PotentialEngine()
                .add(MiePotential(
                    params["mie"], n=self.n, m=self.m,
                    r_cut=self.r_cut, shift=self.shift)))


# ---------------------------------------------------------------------------
#  YukawaParameterSet
# ---------------------------------------------------------------------------

class YukawaParameterSet(FittableParameterSet):
    """
    Parameter set for Yukawa (screened Coulomb) potential fitting.

    V(r) = A . exp(-kappa.r) / r

    Optimisation variables
    ----------------------
    Per pair: A (signed -- folds in the effective q_i.q_j charge
    product, so unlike LJ's epsilon or Buckingham's A/rho, A is NOT
    forced positive via log-space here: opposite-charge pairs need
    A < 0, like-charge pairs need A > 0).
    Bounds: A in [-1e4, 1e4] eV.A

    kappa (the global inverse screening length) is a fixed
    hyperparameter, not fit -- see YukawaPotential's docstring for why.

    Parameters
    ----------
    pairs          : list of frozenset({sp1, sp2}) pairs to fit.
    species        : list of all element symbols.
    initial_yukawa : dict  frozenset({sp1,sp2}) -> {'A': A[eV.A]}
    kappa          : inverse screening length [A^-1] (fixed, not fit).
    r_cut          : interaction cut-off [A].
    shift          : apply force-shifted truncation (default True).
    """

    def __init__(self,
                 pairs          : list,
                 species        : list,
                 initial_yukawa : dict,
                 kappa          : float,
                 r_cut          : float = 10.0,
                 shift          : bool  = True):
        self.pair_list = list(pairs)
        self.species   = species
        self.kappa     = float(kappa)
        self.r_cut     = r_cut
        self.shift     = shift

        _x0, _lo, _hi = [], [], []
        for pair in self.pair_list:
            p = initial_yukawa[pair]
            _x0.append(float(p["A"]))
            _lo.append(-1e4); _hi.append(1e4)

        self._x0     = np.array(_x0)
        self._bounds = list(zip(_lo, _hi))

    @property
    def x0(self) -> np.ndarray:
        return self._x0.copy()

    @property
    def bounds(self) -> list:
        return self._bounds

    def decode(self, x: np.ndarray) -> dict:
        """Return {'yukawa': {pair: {'A': A}}}."""
        yukawa = {}
        for idx, pair in enumerate(self.pair_list):
            yukawa[pair] = {"A": float(x[idx])}
        return {"yukawa": yukawa}

    def build_engine(self, params: dict) -> PotentialEngine:
        return (PotentialEngine()
                .add(YukawaPotential(
                    params["yukawa"], kappa=self.kappa,
                    r_cut=self.r_cut, shift=self.shift)))


# ---------------------------------------------------------------------------
#  StillingerWeberParameterSet
# ---------------------------------------------------------------------------

class StillingerWeberParameterSet(FittableParameterSet):
    """
    Parameter set for Stillinger-Weber potential fitting.

    V2(r) = epsilon . A (B(sigma/r)^p - (sigma/r)^q) exp(sigma/(r - a.sigma))
    V3     = epsilon . lambda (costheta - costheta0)^2 exp(gammasigma/(r_ij-asigma)) exp(gammasigma/(r_ik-asigma))

    Fitted parameters (log-space for positivity)
    ---------------------------------------------
    epsilon : energy scale [eV]
    sigma   : length scale [A]
    A       : 2-body prefactor
    B       : 2-body stiffness
    lambda  : 3-body strength
    gamma   : 3-body angular decay

    Fixed parameters (set via fix_params, not optimised)
    -----------------------------------------------------
    p         : repulsive exponent   (default 4)
    q         : attractive exponent  (default 0)
    a         : cutoff / sigma       (default 1.80)
    costheta0 : ideal cos(angle)     (default -1/3 = cos 109.47deg)

    Note on identifiability
    -----------------------
    epsilon and A only appear as the product epsilon.A in V2, and epsilon and lambda appear as
    epsilon.lambda in V3.  Without additional constraints, the individual values of
    epsilon, A, and lambda are not uniquely identifiable from E/F/sigma data alone.
    The fitting will converge to a correct energy surface but the recovered
    parameters may differ from the "true" reference values.
    To improve identifiability, use structures with diverse coordination
    (bulk, surface, vacancy) and include both energy and force data.

    Parameters
    ----------
    initial_sw  : dict  -- initial guess for all SW parameters (including fixed).
    species     : list  -- element symbols.
    fix_params  : dict  -- parameters to hold fixed (default: p, q, a, costheta0).
    fit_keys    : list  -- parameters to optimise (default: epsilon, sigma, A, B, lambda, gamma).
    """

    _DEFAULT_FIX = {'p': 4, 'q': 0, 'a': 1.80, 'costheta0': -1.0/3.0}
    _DEFAULT_FIT = ['epsilon', 'sigma', 'A', 'B', 'lambda', 'gamma']

    # Which keys are optimised as log(value).  This used to be inferred
    # per key as `lo < -10` on the bound, which is true for no bound in
    # this table whose lower limit is above exp(-10) = 4.5e-5 -- i.e. for
    # every one of them here.  The log-valued bounds were therefore
    # applied to RAW parameter values: canonical Stillinger-Weber Si's
    # A = 7.05 and lambda = 21.0 sat outside their own search boxes
    # (-2.30..4.61 and -4.61..6.22), and every parameter was allowed to
    # go negative.  Stating the set explicitly removes the inference.
    _LOG_KEYS = frozenset({'epsilon', 'sigma', 'A', 'B', 'lambda', 'gamma'})

    _LOG_BOUNDS = {
        'epsilon': (np.log(1e-3), np.log(50.)),
        'sigma':   (np.log(0.3),  np.log(10.)),
        'A':       (np.log(0.1),  np.log(100.)),
        'B':       (np.log(0.01), np.log(20.)),
        'lambda':  (np.log(0.01), np.log(500.)),
        'gamma':   (np.log(0.05), np.log(20.)),
        # straight bounds for signed / small-range params
        'a':       (1.5, 2.5),
        'costheta0': (-1.0, 0.0),
    }

    def __init__(self,
                 initial_sw : dict,
                 species    : list,
                 fix_params : Optional[dict] = None,
                 fit_keys   : Optional[list] = None):
        self.species    = list(species)
        self.fixed      = dict(self._DEFAULT_FIX)
        if fix_params:
            self.fixed.update(fix_params)

        self.fit_keys = list(fit_keys) if fit_keys else list(self._DEFAULT_FIT)
        # Remove keys that are fixed
        self.fit_keys = [k for k in self.fit_keys if k not in self.fixed]

        _x0, _lo, _hi = [], [], []
        for k in self.fit_keys:
            v = initial_sw[k]
            lo, hi = self._LOG_BOUNDS[k]
            if k in self._LOG_KEYS:
                _x0.append(np.log(max(v, 1e-12)))
            else:                 # direct key (a, costheta0)
                _x0.append(float(v))
            _lo.append(lo);  _hi.append(hi)

        self._x0     = np.array(_x0)
        self._bounds = list(zip(_lo, _hi))
        self._initial = initial_sw.copy()

    @property
    def x0(self) -> np.ndarray:
        return self._x0.copy()

    @property
    def bounds(self) -> list:
        return self._bounds

    def decode(self, x: np.ndarray) -> dict:
        """Return {'sw': full_params_dict}."""
        p = self.fixed.copy()
        for i, k in enumerate(self.fit_keys):
            p[k] = np.exp(x[i]) if k in self._LOG_KEYS else float(x[i])
        return {'sw': p}

    def build_engine(self, params: dict) -> PotentialEngine:
        return (PotentialEngine()
                .add(StillingerWeberPotential(params['sw'],
                                              species=self.species)))


class MultiSpeciesStillingerWeberParameterSet(FittableParameterSet):
    """
    Stillinger-Weber fitting for two or more species.

    Works on the multi-species record format StillingerWeberPotential
    already evaluates -- one parameter record per ORDERED triple
    (i, j, k) -- so a fitted result drops straight into a relax/md YAML.
    :class:`StillingerWeberParameterSet` holds a single flat parameter
    set and cannot tell species apart at all: every pair and triple gets
    the same numbers, whatever the atom labels (measured on a 128-atom
    GeSn cell: identical energy to 10 digits with the real Ge/Sn labels,
    all atoms relabelled Ge, or all relabelled X).

    How the engine uses a record, and therefore which parameters are
    tied together here
    ------------------------------------------------------------------
    * Two-body.  Each unordered pair is evaluated ONCE, from the
      diagonal record ``(si, sj, sj)`` of whichever atom of the pair has
      the lower index.  A Ge-Sn pair therefore reads ``(Ge,Sn,Sn)`` or
      ``(Sn,Ge,Ge)`` depending on atom ORDER in the cell.  If those two
      records disagree on any two-body input (sigma, a, epsilon, A, B,
      p, q), the energy depends on how the atoms happen to be listed.
      These keys are one shared variable per unordered pair here, and an
      initial parameter set whose two records disagree is rejected.
      (StillingerWeberPotential itself refuses the physically harmful
      case -- differing EVALUATED two-body terms -- for every caller.
      This class is stricter: it ties sigma/a/epsilon of the two
      records even when both have A = 0, because it fits them as one
      variable.)
    * Three-body (LAMMPS pair_sw; Tomita et al., Jpn. J. Appl. Phys. 57,
      04FB04 (2018), eq. 4).  Each leg i-j of an angle centred on i takes
      sigma, a and gamma from the pair record ``(i, j, j)``; the record
      ``(i, j, k)`` supplies lambda, epsilon and costheta0 only.  A
      diagonal record's gamma is therefore its own variable (it is the
      leg gamma of every angle centred on i with a j neighbour), and a
      mixed record (j != k) contributes only lambda here: its sigma, a
      and gamma are never read and are not fitted.

    Identifiability (read before judging a fit by its parameters)
    -------------------------------------------------------------
    * epsilon enters only as epsilon*A and epsilon*lambda, so it is held
      fixed by default; fitting it as well gives a one-parameter family
      of exact solutions.
    * Each unordered neighbour pair counts once with the MEAN of the
      ``(i, j, k)`` and ``(i, k, j)`` lambdas, so only their mean is
      determined; the two are one variable here (started at their mean,
      which leaves the energy unchanged).
    * A parameter is only as determined as the environments that
      exercise it.  (Sn;Ge,Sn) needs an Sn with both a Ge and an Sn
      neighbour; a dilute-Sn training set has few (30 in three 8-Sn
      cells) and leaves it, and Sn-Sn two-body terms, loosely fixed.

    Parameters
    ----------
    initial_records : list of dicts with keys i, j, k plus the SW
                      parameters (the YAML record-list format), or a dict
                      {(i, j, k): {...}}.  Supplies the starting point AND
                      the values of every fixed parameter.
    species         : element symbols.
    fix_params      : names of the parameters held at their initial
                      values, in every record.  Default: epsilon, p, q,
                      a, costheta0.  A dict is accepted for symmetry with
                      the single-species class, but only its KEYS are
                      used -- fixed values always come from the records.
    fit_keys        : names to optimise.  Default: sigma, A, B, lambda,
                      gamma (minus anything fixed).
    """

    _DEFAULT_FIX = ("epsilon", "p", "q", "a", "costheta0")
    _DEFAULT_FIT = ("sigma", "A", "B", "lambda", "gamma")
    _PAIR_KEYS = ("sigma", "a", "epsilon", "A", "B", "p", "q")
    _TWO_BODY_ONLY = ("A", "B", "p", "q")
    _LOG_KEYS = frozenset({"sigma", "A", "B", "lambda", "gamma", "epsilon"})
    _BOUNDS = {                      # physical (not log) limits
        "sigma":   (0.3, 10.0),
        "A":       (0.1, 100.0),
        "B":       (0.01, 20.0),
        "lambda":  (0.01, 500.0),
        "gamma":   (0.05, 20.0),
        "epsilon": (1e-3, 50.0),
        "a":       (1.2, 2.5),
        "p":       (0.0, 12.0),
        "q":       (0.0, 6.0),
        "costheta0": (-1.0, 0.0),
    }

    def __init__(self,
                 initial_records,
                 species    : list,
                 fix_params = None,
                 fit_keys   : Optional[list] = None):
        self.species = list(species)
        self.records = self._to_nested(initial_records)

        fixed = set(self._DEFAULT_FIX)
        if fix_params:
            fixed |= set(fix_params)
        self.fixed_keys = frozenset(fixed)
        keys = list(fit_keys) if fit_keys else list(self._DEFAULT_FIT)
        self.fit_keys = [k for k in keys if k not in self.fixed_keys]
        unknown = [k for k in self.fit_keys if k not in self._BOUNDS]
        if unknown:
            raise ValueError(f"MultiSpeciesStillingerWeberParameterSet: "
                             f"unknown fit key(s) {unknown}")

        self._check_pair_consistency()

        # Each variable: (name, [(triple, key), ...]).  Every target of a
        # variable receives the same value on decode.
        self.variables = []
        done_pairs = set()
        for (i, j, k) in self.records:
            if j == k:                                   # diagonal record
                pair = frozenset((i, j))
                if pair not in done_pairs:
                    done_pairs.add(pair)
                    targets = [(i, j, j)] + ([(j, i, i)] if i != j else [])
                    for key in self.fit_keys:
                        if key in self._PAIR_KEYS:
                            self.variables.append(
                                (f"{i}-{j}:{key}", [(t, key) for t in targets]))
            if j == k:
                for key in self.fit_keys:
                    if key not in self._PAIR_KEYS:          # lambda, gamma, ...
                        self.variables.append(
                            (f"{i};{j},{j}:{key}", [((i, j, j), key)]))
                continue
            # Mixed record: only lambda (and epsilon, costheta0 if fitted)
            # is read by the engine.  lambda of (i, j, k) and (i, k, j)
            # enter as their mean: one variable for the unordered pair.
            for key in self.fit_keys:
                if key == "lambda":
                    if (j, k) > (k, j) and (i, k, j) in self.records:
                        continue                         # owned by its twin
                    targets = [((i, j, k), key)]
                    if (i, k, j) in self.records:
                        targets.append(((i, k, j), key))
                    self.variables.append((f"{i};{{{j},{k}}}:{key}", targets))
                elif key in ("epsilon", "costheta0"):
                    self.variables.append((f"{i};{j},{k}:{key}", [((i, j, k), key)]))

        x0, bounds = [], []
        for name, targets in self.variables:
            key = targets[0][1]
            v = float(np.mean([self.records[t][key] for t, _ in targets]))
            lo, hi = self._BOUNDS[key]
            if not lo <= v <= hi:
                raise ValueError(
                    f"MultiSpeciesStillingerWeberParameterSet: initial {name} = "
                    f"{v} lies outside its search range [{lo}, {hi}].")
            if key in self._LOG_KEYS:
                x0.append(np.log(v)); bounds.append((np.log(lo), np.log(hi)))
            else:
                x0.append(v);         bounds.append((lo, hi))
        self._x0 = np.array(x0)
        self._bounds = bounds

    # ------------------------------------------------------------------
    @staticmethod
    def _to_nested(recs) -> dict:
        if isinstance(recs, dict):
            return {tuple(t): dict(v) for t, v in recs.items()}
        nested = {}
        for rec in recs:
            rec = dict(rec)
            nested[(rec.pop("i"), rec.pop("j"), rec.pop("k"))] = rec
        return nested

    @staticmethod
    def to_records(nested: dict) -> list:
        """{(i, j, k): {...}} -> the YAML record-list format."""
        return [dict(i=t[0], j=t[1], k=t[2], **{k: float(v) for k, v in p.items()})
                for t, p in nested.items()]

    def _check_pair_consistency(self) -> None:
        for (i, j, k), rec in self.records.items():
            if j != k or i == j:
                continue
            twin = self.records.get((j, i, i))
            if twin is None:
                raise ValueError(
                    f"SW record ({i},{j},{j}) has no partner ({j},{i},{i}).  "
                    f"The {i}-{j} pair is evaluated from whichever of the two "
                    f"belongs to the lower-indexed atom, so both are required.")
            for key in self._PAIR_KEYS:
                a, b = rec.get(key), twin.get(key)
                if (a is None) != (b is None) or (
                        a is not None and not np.isclose(float(a), float(b),
                                                         rtol=1e-12, atol=0.0)):
                    raise ValueError(
                        f"SW records ({i},{j},{j}) and ({j},{i},{i}) disagree on "
                        f"'{key}' ({a} vs {b}).  Both describe the same {i}-{j} "
                        f"pair, and which one the engine reads depends on atom "
                        f"order in the cell, so the energy would too.")

    # ------------------------------------------------------------------
    @property
    def x0(self) -> np.ndarray:
        return self._x0.copy()

    @property
    def bounds(self) -> list:
        return self._bounds

    @property
    def variable_names(self) -> list:
        return [name for name, _ in self.variables]

    def decode(self, x: np.ndarray) -> dict:
        """Return {'sw': {(i, j, k): full parameter dict}}."""
        out = {t: dict(p) for t, p in self.records.items()}
        for xi, (name, targets) in zip(x, self.variables):
            key = targets[0][1]
            v = float(np.exp(xi)) if key in self._LOG_KEYS else float(xi)
            for triple, k in targets:
                out[triple][k] = v
        return {"sw": out}

    def build_engine(self, params: dict) -> PotentialEngine:
        return (PotentialEngine()
                .add(StillingerWeberPotential(params["sw"],
                                              species=self.species)))


# ---------------------------------------------------------------------------
#  TersoffParameterSet
# ---------------------------------------------------------------------------

class TersoffParameterSet(FittableParameterSet):
    """
    Parameter set for Tersoff bond-order potential fitting.

    E = sum_{i<j} fc(r_ij) [ A exp(-lambda1 r) - b_ij B exp(-lambda2 r) ]
    b_ij = (1 + beta^n zeta^n)^{-1/(2n)}

    Fitted parameters (log-space for positive quantities)
    ------------------------------------------------------
    A       : repulsive prefactor [eV]
    B       : attractive prefactor [eV]
    lambda1 : repulsive decay [A^-1]
    lambda2 : attractive decay [A^-1]
    beta    : bond-order coefficient
    n       : bond-order exponent
    c       : angular function numerator
    d       : angular function denominator
    h       : angular function shift  (direct optimisation, range [-1, 1])

    Fixed parameters (set via fix_params, not optimised)
    -----------------------------------------------------
    lambda3 : 3-body distance decay (default = lambda2)
    R       : cutoff midpoint [A]    (default 2.85)
    D       : cutoff half-width [A]  (default 0.15)

    Parameters
    ----------
    initial_t   : dict  -- initial guess for all Tersoff parameters.
    species     : list  -- element symbols.
    fix_params  : dict  -- parameters to hold fixed (default: lambda3, R, D).
    fit_keys    : list  -- parameters to optimise (default: all except lambda3,R,D).
    """

    _DEFAULT_FIX = {'lambda3': None, 'R': 2.85, 'D': 0.15}
    # lambda3=None means "copy lambda2 on decode"
    _DEFAULT_FIT = ['A', 'B', 'lambda1', 'lambda2', 'beta', 'n', 'c', 'd', 'h']

    # Which keys are optimised as log(value).  This used to be inferred
    # per key as `lo < -10` on the bound, which is true for no bound in
    # this table whose lower limit is above exp(-10) = 4.5e-5 -- i.e. for
    # every one of them here.  The log-valued bounds were therefore
    # applied to RAW parameter values: Tersoff T3 Si's A = 1830.8 and
    # c = 100390 sat outside their own search boxes (0..11.5 and
    # 0..16.1); only beta, bounded below by log(1e-10) = -23, escaped, and every parameter was allowed to
    # go negative.  Stating the set explicitly removes the inference.
    _LOG_KEYS = frozenset({'A', 'B', 'lambda1', 'lambda2', 'beta',
                           'n', 'c', 'd'})

    _LOG_BOUNDS = {
        'A':       (np.log(1.),    np.log(1e5)),
        'B':       (np.log(0.1),   np.log(1e4)),
        'lambda1': (np.log(0.5),   np.log(10.)),
        'lambda2': (np.log(0.1),   np.log(8.)),
        'beta':    (np.log(1e-10), np.log(1.)),
        'n':       (np.log(0.1),   np.log(10.)),
        'c':       (np.log(1.),    np.log(1e7)),
        'd':       (np.log(0.5),   np.log(1e4)),
        # direct bounds (h is signed, range [-1, 0] typical)
        'h':       (-1.0, 0.5),
        'R':       (1.5,  5.0),
        'D':       (0.05, 0.5),
    }

    def __init__(self,
                 initial_t  : dict,
                 species    : list,
                 fix_params : Optional[dict] = None,
                 fit_keys   : Optional[list] = None):
        self.species  = list(species)
        self.fixed    = dict(self._DEFAULT_FIX)
        if fix_params:
            self.fixed.update(fix_params)

        self.fit_keys = list(fit_keys) if fit_keys else list(self._DEFAULT_FIT)
        self.fit_keys = [k for k in self.fit_keys if k not in self.fixed]

        _x0, _lo, _hi = [], [], []
        for k in self.fit_keys:
            v = initial_t[k]
            lo, hi = self._LOG_BOUNDS[k]
            _x0.append(np.log(max(v, 1e-15)) if k in self._LOG_KEYS
                       else float(v))
            _lo.append(lo);  _hi.append(hi)

        self._x0     = np.array(_x0)
        self._bounds = list(zip(_lo, _hi))
        self._initial = initial_t.copy()

    @property
    def x0(self) -> np.ndarray:
        return self._x0.copy()

    @property
    def bounds(self) -> list:
        return self._bounds

    def decode(self, x: np.ndarray) -> dict:
        """Return {'tersoff': full_params_dict}."""
        p = {}
        # Fixed params
        for k, v in self.fixed.items():
            if k != 'lambda3':
                p[k] = float(v)
        # Fitted params
        for i, k in enumerate(self.fit_keys):
            p[k] = np.exp(x[i]) if k in self._LOG_KEYS else float(x[i])
        # lambda3: default to lambda2 if not fixed to a specific value
        if 'lambda3' in self.fixed and self.fixed['lambda3'] is not None:
            p['lambda3'] = float(self.fixed['lambda3'])
        else:
            p['lambda3'] = p.get('lambda2', self._initial.get('lambda2', 1.0))
        return {'tersoff': p}

    def build_engine(self, params: dict) -> PotentialEngine:
        return (PotentialEngine()
                .add(TersoffPotential(params['tersoff'],
                                      species=self.species)))


# ---------------------------------------------------------------------------
#  Multi-species Tersoff (ADR-0015)
# ---------------------------------------------------------------------------

def _tersoff_records_to_nested(recs) -> dict:
    """YAML record list [{i, j, k, ...}] or {(i, j, k): {...}} -> nested dict."""
    if isinstance(recs, dict):
        return {tuple(t): {k: float(v) for k, v in p.items()}
                for t, p in recs.items()}
    nested = {}
    for rec in recs:
        rec = dict(rec)
        key = (rec.pop("i"), rec.pop("j"), rec.pop("k"))
        nested[key] = {k: float(v) for k, v in rec.items()}
    return nested


class _VariableTableMixin:
    """x <-> named variables, each writing one value to several targets."""

    @property
    def x0(self) -> np.ndarray:
        return self._x0.copy()

    @property
    def bounds(self) -> list:
        return self._bounds

    @property
    def variable_names(self) -> list:
        return [name for name, _, _ in self.variables]

    def _setup_vector(self, initial_values, owner: str) -> None:
        x0, bounds = [], []
        for (name, key, _), v in zip(self.variables, initial_values):
            lo, hi = self._BOUNDS[key]
            if not lo <= v <= hi:
                raise ValueError(f"{owner}: initial {name} = {v} lies outside "
                                 f"its search range [{lo}, {hi}].")
            if key in self._LOG_KEYS:
                x0.append(np.log(v)); bounds.append((np.log(lo), np.log(hi)))
            else:
                x0.append(v);         bounds.append((lo, hi))
        self._x0 = np.array(x0)
        self._bounds = bounds

    def _value(self, key: str, xi: float) -> float:
        return float(np.exp(xi)) if key in self._LOG_KEYS else float(xi)


class MultiSpeciesTersoffParameterSet(_VariableTableMixin, FittableParameterSet):
    """
    Tersoff fitting on the (i, j, k) record table (LAMMPS .tersoff
    convention) that TersoffPotential evaluates (ADR-0015, form (a)).

    How the engine reads a record, and the default variables
    -------------------------------------------------------
    * Pair terms A, B, lambda1, lambda2 (and the cutoff R, D) come from
      the diagonal record (i, j, j) of the LOWER-INDEXED atom of a pair,
      so (i, j, j) and (j, i, i) must agree or the energy depends on atom
      order.  They are one variable per unordered pair, and initial
      records that disagree are rejected.
    * beta, n: bond order b_ij of the centre i, from (i, j, j): one
      variable per diagonal record.
    * c, d, h (and lambda3, gamma if fitted): from the full record
      (i, j, k), one variable per record; ``tie="centre"`` makes them one
      variable per centre element i (every record with that i), the
      usual physical choice.
    * Fixed by default: R, D (cutoffs are set, not fitted), m (1 or 3),
      gamma, lambda3, and beta, n (ADR-0015 Q17): with near-tetrahedral
      training angles, beta and the scale of zeta (c, d) trade off almost
      exactly (the degeneracy of ADR-0003 FT1).  Fitted, they drifted by
      orders of magnitude even from a start 1 % off the reference, and
      Nelder-Mead stalled at loss ~1e-5 (AlN, wurtzite and with zinc blende
      and rocksalt added).  Name them in ``fit_keys`` to fit them anyway.

    Parameters
    ----------
    initial_records : record list (YAML format) or {(i, j, k): {...}}; the
                      starting point and the values of every fixed key.
    species         : element symbols.
    fix_params      : keys held at their record values (added to the
                      default fixed set; only the names are used).
    fit_keys        : keys to optimise (default A, B, lambda1, lambda2,
                      beta, n, c, d, h).
    tie             : "record" (default) or "centre" for the angular keys.
    """

    _DEFAULT_FIX = ("R", "D", "m", "gamma", "lambda3", "beta", "n")
    _DEFAULT_FIT = ("A", "B", "lambda1", "lambda2", "c", "d", "h")
    _PAIR_KEYS = ("A", "B", "lambda1", "lambda2", "R", "D")
    _BOND_KEYS = ("beta", "n")
    _ANGLE_KEYS = ("c", "d", "h", "lambda3", "gamma")
    _LOG_KEYS = frozenset({"A", "B", "lambda1", "lambda2", "beta", "n",
                           "c", "d", "gamma"})
    _BOUNDS = {                      # physical (not log) limits
        "A": (1.0, 1e5), "B": (0.1, 1e4),
        "lambda1": (0.5, 10.0), "lambda2": (0.1, 8.0),
        "beta": (1e-10, 10.0), "n": (0.05, 10.0),
        "c": (1e-3, 1e7), "d": (1e-2, 1e4), "h": (-1.0, 1.0),
        "lambda3": (0.0, 5.0), "gamma": (1e-3, 100.0),
        "R": (1.0, 5.0), "D": (0.02, 0.5),
    }

    def __init__(self, initial_records, species: list, fix_params=None,
                 fit_keys: Optional[list] = None, tie: str = "record"):
        if tie not in ("record", "centre"):
            raise ValueError("MultiSpeciesTersoffParameterSet: tie must be "
                             "'record' or 'centre'")
        self.species = list(species)
        self.records = _tersoff_records_to_nested(initial_records)
        for sp in {e for t in self.records for e in t}:
            if sp not in self.species:
                raise ValueError(f"Tersoff record element '{sp}' is not among "
                                 f"the species {self.species}")
        fixed = set(self._DEFAULT_FIX) | set(fix_params or ())
        keys = list(fit_keys) if fit_keys else list(self._DEFAULT_FIT)
        self.fit_keys = [k for k in keys if k not in fixed]
        unknown = [k for k in self.fit_keys if k not in self._BOUNDS]
        if unknown:
            raise ValueError(f"MultiSpeciesTersoffParameterSet: unknown fit "
                             f"key(s) {unknown}")
        self._check_pairs()

        # variables: (name, key, [(triple, key), ...])
        self.variables = []
        seen_pairs, seen_centres = set(), set()
        for (i, j, k) in self.records:
            if j == k:
                pair = frozenset((i, j))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    tw = [(i, j, j)] + ([(j, i, i)] if i != j else [])
                    for key in self.fit_keys:
                        if key in self._PAIR_KEYS:
                            self.variables.append(
                                (f"{i}-{j}:{key}", key, [(t, key) for t in tw]))
                for key in self.fit_keys:
                    if key in self._BOND_KEYS:
                        self.variables.append(
                            (f"{i};{j}:{key}", key, [((i, j, j), key)]))
            for key in self.fit_keys:
                if key not in self._ANGLE_KEYS:
                    continue
                if tie == "centre":
                    if (i, key) in seen_centres:
                        continue
                    seen_centres.add((i, key))
                    tg = [(t, key) for t in self.records if t[0] == i]
                    self.variables.append((f"{i};*:{key}", key, tg))
                else:
                    self.variables.append(
                        (f"{i};{j},{k}:{key}", key, [((i, j, k), key)]))
        init = []
        for name, key, targets in self.variables:
            try:
                init.append(float(np.mean([self.records[t][kk] for t, kk in targets])))
            except KeyError:
                raise ValueError(f"MultiSpeciesTersoffParameterSet: a record "
                                 f"of {name} has no '{key}'") from None
        self._setup_vector(init, "MultiSpeciesTersoffParameterSet")

    def _check_pairs(self) -> None:
        for (i, j, k), rec in self.records.items():
            if j != k or i == j:
                continue
            twin = self.records.get((j, i, i))
            if twin is None:
                raise ValueError(
                    f"Tersoff record ({i},{j},{j}) has no partner ({j},{i},{i}); "
                    f"the {i}-{j} pair is read from whichever belongs to the "
                    f"lower-indexed atom, so both are required.")
            for key in self._PAIR_KEYS:
                a, b = rec.get(key), twin.get(key)
                if (a is None) != (b is None) or (
                        a is not None and not np.isclose(a, b, rtol=1e-12, atol=0.0)):
                    raise ValueError(
                        f"Tersoff records ({i},{j},{j}) and ({j},{i},{i}) disagree "
                        f"on '{key}' ({a} vs {b}); the engine would read one or "
                        f"the other depending on atom order.")

    def decode(self, x: np.ndarray) -> dict:
        """Return {'tersoff': {(i, j, k): full parameter dict}}."""
        out = {t: dict(p) for t, p in self.records.items()}
        for xi, (_, key, targets) in zip(x, self.variables):
            v = self._value(key, xi)
            for triple, kk in targets:
                out[triple][kk] = v
        return {"tersoff": out}

    def build_engine(self, params: dict) -> PotentialEngine:
        return PotentialEngine().add(
            TersoffPotential(params["tersoff"], species=self.species))


class Tersoff1989ParameterSet(_VariableTableMixin, FittableParameterSet):
    """
    Tersoff fitting in the multicomponent mixing form of Tersoff, Phys.
    Rev. B 39, 5566 (1989) (ADR-0015, form (b)): per-element A, B,
    lambda, mu, beta, n, c, d, h, R, S plus one chi per unordered pair of
    distinct elements; the (i, j, k) table follows from
    akitsushima.engine.tersoff.tersoff1989_to_triplets.

    Default variables: A, B, lambda, mu, c, d, h of every element and every
    chi; R and S fixed (the paper did not optimise them either), beta and
    n fixed (the beta-c/d degeneracy, see MultiSpeciesTersoffParameterSet;
    ADR-0015 Q17; list them in ``fit_keys`` to fit them).
    ``fit_keys=["chi"]`` reproduces the paper's own procedure (elements
    fixed, chi from compound data).

    Parameters
    ----------
    elements : {symbol: {A, B, lambda, mu, beta, n, c, d, h, R, S}}.
    chi      : {"a-b": value} for every pair of distinct species.
    species  : element symbols.
    fix_params, fit_keys : names, as for the other parameter sets.
    omega    : optional {"i-k": omega_ik} (fixed; default 1).
    """

    _DEFAULT_FIX = ("R", "S", "beta", "n")
    _DEFAULT_FIT = ("A", "B", "lambda", "mu", "c", "d", "h", "chi")
    _LOG_KEYS = frozenset({"A", "B", "lambda", "mu", "beta", "n", "c", "d",
                           "chi", "R", "S"})
    _BOUNDS = {
        "A": (1.0, 1e5), "B": (0.1, 1e4), "lambda": (0.5, 10.0),
        "mu": (0.1, 8.0), "beta": (1e-10, 10.0), "n": (0.05, 10.0),
        "c": (1e-3, 1e7), "d": (1e-2, 1e4), "h": (-1.0, 1.0),
        "chi": (0.3, 3.0), "R": (1.0, 5.0), "S": (1.0, 5.5),
    }

    def __init__(self, elements: dict, chi: dict, species: list,
                 fix_params=None, fit_keys: Optional[list] = None,
                 omega: Optional[dict] = None):
        from ..engine.tersoff import TERSOFF1989_ELEMENT_KEYS, tersoff1989_to_triplets
        self.species = list(species)
        self.elements = {s: {k: float(v) for k, v in elements[s].items()}
                         for s in self.species if s in elements}
        self.omega = dict(omega or {})
        self._expand = tersoff1989_to_triplets
        # canonical chi keys, species order
        self.chi = {}
        for a_i, a in enumerate(self.species):
            for b in self.species[a_i + 1:]:
                v = chi.get(f"{a}-{b}", chi.get(f"{b}-{a}"))
                if v is None:
                    raise ValueError(f"Tersoff1989ParameterSet: chi for {a}-{b} "
                                     f"is not given")
                self.chi[f"{a}-{b}"] = float(v)
        tersoff1989_to_triplets(self.elements, self.chi, self.species, self.omega)

        fixed = set(self._DEFAULT_FIX) | set(fix_params or ())
        keys = list(fit_keys) if fit_keys else list(self._DEFAULT_FIT)
        self.fit_keys = [k for k in keys if k not in fixed]
        unknown = [k for k in self.fit_keys
                   if k not in TERSOFF1989_ELEMENT_KEYS and k != "chi"]
        if unknown:
            raise ValueError(f"Tersoff1989ParameterSet: unknown fit key(s) {unknown}")
        self.variables, init = [], []
        for s in self.species:
            for key in self.fit_keys:
                if key != "chi":
                    self.variables.append((f"{s}:{key}", key, (s, key)))
                    init.append(self.elements[s][key])
        if "chi" in self.fit_keys:
            for pair, v in self.chi.items():
                self.variables.append((f"chi:{pair}", "chi", pair))
                init.append(v)
        self._setup_vector(init, "Tersoff1989ParameterSet")

    def decode(self, x: np.ndarray) -> dict:
        """Return {'tersoff': (i, j, k) table, 'tersoff1989': {elements, chi}}."""
        el = {s: dict(p) for s, p in self.elements.items()}
        chi = dict(self.chi)
        for xi, (_, key, target) in zip(x, self.variables):
            v = self._value(key, xi)
            if key == "chi":
                chi[target] = v
            else:
                el[target[0]][key] = v
        table = self._expand(el, chi, self.species, self.omega)
        return {"tersoff": table,
                "tersoff1989": {"elements": el, "chi": chi, "omega": dict(self.omega)}}

    def build_engine(self, params: dict) -> PotentialEngine:
        return PotentialEngine().add(
            TersoffPotential(params["tersoff"], species=self.species))


class ZhouEAMParameterSet(_VariableTableMixin, FittableParameterSet):
    """
    Alloy EAM fitting in the analytic form of Zhou, Johnson and Wadley,
    Phys. Rev. B 69, 144113 (2004) (ADR-0015; engine/eam_zhou.py).

    Variables (default, per element): A, B, alpha, beta of the pair term
    and Fn2, Fn3, F2, F3, eta of the embedding function.  Fn0, Fn1, F0
    and F_e are never variables: they follow from the continuity of F and
    dF/drho at rho_n and rho_0 (eam_zhou.continuous).  Fixed: r_e, f_e,
    rho_e, rho_s (f_e and rho set the density scale, which a single
    element cannot determine), kappa, lambda (cutoff shape), F1 (0 in
    Table III).  The cross pair follows eq. (A5); ``"chi"`` in fit_keys
    adds one factor phi_ab -> chi_ab phi_ab per pair of distinct elements
    (not in the paper, ADR-0015 Q8(b)).

    The cutoff is fixed for the whole fit (``r_cut``, default: from the
    initial parameters, eam_zhou.zhou_cutoff), so the loss stays smooth.

    Parameters
    ----------
    elements : {symbol: Table III parameter dict} (the start point).
    species  : element symbols.
    chi      : optional {"a-b": value}, default 1.
    fix_params, fit_keys : names, as for the other parameter sets.
    r_cut    : cutoff [A]; None -> computed once from ``elements``.
    """

    _DEFAULT_FIX = ("r_e", "f_e", "rho_e", "rho_s", "kappa", "lambda", "F1")
    _DEFAULT_FIT = ("A", "B", "alpha", "beta", "Fn2", "Fn3", "F2", "F3", "eta")
    _LOG_KEYS = frozenset({"A", "B", "alpha", "beta", "chi", "r_e", "f_e",
                           "rho_e", "rho_s"})
    _BOUNDS = {
        "A": (1e-3, 10.0), "B": (1e-3, 10.0), "alpha": (1.0, 30.0),
        "beta": (0.5, 20.0), "Fn2": (-20.0, 20.0), "Fn3": (-20.0, 20.0),
        "F2": (-20.0, 20.0), "F3": (-20.0, 20.0), "eta": (-3.0, 3.0),
        "F1": (-20.0, 20.0), "chi": (0.2, 5.0), "r_e": (1.5, 5.0),
        "f_e": (0.05, 20.0), "rho_e": (0.5, 100.0), "rho_s": (0.5, 100.0),
        "kappa": (0.05, 2.0), "lambda": (0.05, 2.0),
    }

    def __init__(self, elements: dict, species: list, chi: Optional[dict] = None,
                 fix_params=None, fit_keys: Optional[list] = None,
                 r_cut: Optional[float] = None):
        from ..engine.eam_zhou import DERIVED, KEYS, continuous, zhou_cutoff
        self.species = list(species)
        self.elements = {}
        for s in self.species:
            if s not in elements:
                raise ValueError(f"ZhouEAMParameterSet: no parameters for '{s}'")
            self.elements[s] = continuous(elements[s])
        self.chi = {}
        for a_i, a in enumerate(self.species):
            for b in self.species[a_i + 1:]:
                v = (chi or {}).get(f"{a}-{b}", (chi or {}).get(f"{b}-{a}", 1.0))
                self.chi[f"{a}-{b}"] = float(v)
        self.r_cut = float(r_cut) if r_cut else max(
            zhou_cutoff(self.elements[s]) for s in self.species)

        fixed = set(self._DEFAULT_FIX) | set(fix_params or ())
        keys = list(fit_keys) if fit_keys else list(self._DEFAULT_FIT)
        bad = [k for k in keys if k in DERIVED]
        if bad:
            raise ValueError(f"ZhouEAMParameterSet: {bad} follow from the "
                             f"continuity of F and cannot be fitted")
        self.fit_keys = [k for k in keys if k not in fixed]
        unknown = [k for k in self.fit_keys if k not in KEYS and k != "chi"]
        if unknown:
            raise ValueError(f"ZhouEAMParameterSet: unknown fit key(s) {unknown}")
        self.variables, init = [], []
        for s in self.species:
            for key in self.fit_keys:
                if key != "chi":
                    self.variables.append((f"{s}:{key}", key, (s, key)))
                    init.append(self.elements[s][key])
        if "chi" in self.fit_keys:
            for pair, v in self.chi.items():
                self.variables.append((f"chi:{pair}", "chi", pair))
                init.append(v)
        self._setup_vector(init, "ZhouEAMParameterSet")

    def decode(self, x: np.ndarray) -> dict:
        """Return {'eam': {'form': 'zhou2004', 'elements', 'chi', 'r_cut'}}."""
        from ..engine.eam_zhou import continuous
        el = {s: dict(p) for s, p in self.elements.items()}
        chi = dict(self.chi)
        for xi, (_, key, target) in zip(x, self.variables):
            v = self._value(key, xi)
            if key == "chi":
                chi[target] = v
            else:
                el[target[0]][key] = v
        el = {s: continuous(p) for s, p in el.items()}
        return {"eam": {"form": "zhou2004", "elements": el, "chi": chi,
                        "r_cut": self.r_cut}}

    def build_engine(self, params: dict) -> PotentialEngine:
        from ..engine.eam_zhou import zhou_eam
        p = params["eam"]
        return PotentialEngine().add(
            zhou_eam(p["elements"], self.species, r_cut=p["r_cut"], chi=p["chi"]))
