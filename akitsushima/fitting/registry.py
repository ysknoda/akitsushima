"""Central registry for classical potential types.

Consolidates the "add a new classical potential" recipe, which
previously required manually keeping 4 separate locations in sync with
no single source of truth:

1. ``cli/parser.py``'s hardcoded ``choices=[...]`` list for ``--potential``.
2. ``cli/cmd_fit.py``'s ``_build_param_set`` if/elif chain.
3. ``cli/templates.py``'s ``_FIT_CFG_TEMPLATES`` dict (one default config
   per potential, printed by ``--write-config``).
4. ``solvers/config_bridge.py``'s ``_build_force_fn_from_yaml`` if/elif
   chain (the separate YAML bridge used by ``relax``/``md``/etc.).

Every potential type register()'s exactly one ``PotentialSpec`` here;
the four consumer sites above now derive their behaviour from this one
registry instead of each keeping an independent list/chain/dict. Adding
a brand-new potential (see Mie/Yukawa below) means adding one
``PotentialSpec`` and touching no other file's dispatch logic.

Two lookups are kept deliberately separate, since they answer different
questions:

- :func:`get` (keyed by the *canonical* name only) answers "what does
  ``fit --potential <name>`` build?" -- used by ``cmd_fit.py`` and to
  populate ``parser.py``'s ``--potential`` choices via :func:`names`.
- :func:`get_by_yaml_type` (keyed by every accepted spelling, e.g. both
  ``"lj"`` and ``"lennard-jones"``) answers "what does a YAML
  ``potential: {type: ...}`` block build?" -- used by
  ``config_bridge.py``. Not every registered potential supports this
  (EAM has no per-type entry at all in ``fit --potential``'s choices to
  begin with, since EAM parameters are tabulated functions, not scalars
  a generic optimiser can fit) or the reverse (Buckingham's YAML path is
  a structurally different legacy JSON-loading branch, kept
  special-cased in ``config_bridge.py`` rather than unified here).
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from ..engine import (
    BornMayerPotential,
    LennardJonesPotential,
    MiePotential,
    MorsePotential,
    PotentialEngine,
    StillingerWeberPotential,
    TersoffPotential,
    YukawaPotential,
)
from .force_fn import make_force_fn
from .parameter_sets import (
    BornMayerParameterSet,
    BuckinghamParameterSet,
    LJParameterSet,
    MieParameterSet,
    MorseParameterSet,
    StillingerWeberParameterSet,
    TersoffParameterSet,
    YukawaParameterSet,
)

__all__ = ["PotentialSpec", "register", "get", "get_by_yaml_type", "names"]


@dataclass(frozen=True)
class PotentialSpec:
    name            : str
    build_param_set : Optional[Callable] = None
    default_cfg     : Optional[dict]     = None
    build_from_yaml : Optional[Callable] = None
    yaml_aliases    : tuple              = field(default_factory=tuple)


_REGISTRY: dict = {}        # canonical name -> PotentialSpec ("fit --potential")
_YAML_ALIASES: dict = {}    # every accepted YAML "type:" spelling -> PotentialSpec


def register(spec: PotentialSpec) -> None:
    _REGISTRY[spec.name] = spec
    _YAML_ALIASES[spec.name] = spec
    for alias in spec.yaml_aliases:
        _YAML_ALIASES[alias] = spec


def get(name: str) -> PotentialSpec:
    """Look up by canonical ``fit --potential`` name."""
    try:
        return _REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown potential '{name}'. Choose from: {names()}.")


def get_by_yaml_type(pot_type: str) -> PotentialSpec:
    """Look up by any accepted YAML ``potential.type`` spelling."""
    try:
        return _YAML_ALIASES[pot_type.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown potential type '{pot_type}'. "
            f"Supported: SW, Tersoff, LJ, Morse, BornMayer, EAM, Buckingham")


def names() -> list:
    """Canonical names usable with ``fit --potential`` (i.e. that have a
    build_param_set -- EAM is registered for the YAML bridge only, since
    its parameters are tabulated functions, not scalars a generic
    optimiser can fit)."""
    return sorted(n for n, spec in _REGISTRY.items()
                 if spec.build_param_set is not None)


def _parse_pairs(raw: dict) -> dict:
    """Convert "Sp1-Sp2" string keys -> frozenset keys."""
    out = {}
    for k, v in raw.items():
        parts = k.split("-")
        out[frozenset(parts)] = v
    return out


def _pairs_to_frozenset_dict(params, species: list, pot_cfg: dict,
                             pot_label: str) -> Tuple[dict, float]:
    """
    Convert a YAML/JSON-native ``potential.params`` value into the
    ``frozenset({sp_i, sp_j}) -> {param: value, ...}`` dict that
    LennardJonesPotential/MorsePotential/BornMayerPotential/
    MiePotential/YukawaPotential actually expect.

    Supported ``params`` forms
    ---------------------------
    (a) List of records (required for more than one species)::

            params:
              - {i: Al, j: Al, epsilon: 0.4,  sigma: 2.5}
              - {i: Al, j: O,  epsilon: 0.1,  sigma: 3.0}

    (b) Flat dict (only valid when ``species:`` names exactly one
        species -- the single species interacting with itself)::

            species: [Al]
            params: {epsilon: 0.4, sigma: 2.5}

    An optional top-level ``r_cut`` in the flat form (or ``potential.
    r_cut``) sets the cutoff; ``potential.r_cut`` is otherwise the only
    way to set it for the list form.

    Returns
    -------
    (pair_params, r_cut)
    """
    if isinstance(params, list):
        pair_params = {}
        for rec in params:
            rec = dict(rec)
            key = frozenset({rec.pop("i"), rec.pop("j")})
            pair_params[key] = rec
        r_cut = float(pot_cfg.get("r_cut", 10.0))
        return pair_params, r_cut

    if len(species) == 1:
        flat = dict(params)
        r_cut = float(flat.pop("r_cut", pot_cfg.get("r_cut", 10.0)))
        return {frozenset({species[0]}): flat}, r_cut

    raise ValueError(
        f"{pot_label} 'params:' must be a list of {{i, j, ...}} records "
        f"for multi-species systems (i.e. when 'species:' names more "
        f"than one element), or a flat dict when 'species:' names "
        f"exactly one species.")


# ============================================================================
#  Buckingham (fit-only here -- its YAML bridge is a structurally
#  different legacy JSON-loading branch, special-cased directly in
#  config_bridge.py rather than unified into build_from_yaml)
# ============================================================================

_DEFAULT_FIT_CFG_BUCKINGHAM = {
    "potential"     : "buckingham",
    "training_csv"  : "./training_set/structures.csv",
    "output_dir"    : "./fitting_output",
    "r_cut"         : 10.0,
    "ewald_alpha"   : None,
    "weights"       : {"energy": 1.0, "forces": 1.0, "stress": 0.1,
                       "r0": 1.0, "eps0": 1.0},
    "fix_charges"   : True,
    "charges"       : {"Mg": 2.0, "O": -2.0},
    "initial_buckingham": {
        "Mg-O" : {"A": 1279.69, "rho": 0.2997, "C": 0.0},
        "O-O"  : {"A": 22764.3, "rho": 0.1490, "C": 27.879},
        "Mg-Mg": {"A": 0.0,    "rho": 0.3000, "C": 0.0},
    },
}


def _build_param_set_buckingham(cfg, all_species, pairs, r_cut):
    charges      = cfg.get("charges", {})
    initial_buck = _parse_pairs(cfg["initial_buckingham"])
    # Auto-fill missing pairs with zeros
    for pair in pairs:
        if pair not in initial_buck:
            s1, s2 = sorted(pair)
            import warnings
            warnings.warn(
                f"No initial Buckingham params for {s1}-{s2}; using zeros.")
            initial_buck[pair] = {"A": 1.0, "rho": 0.3, "C": 0.0}
    for sp in all_species:
        if sp not in charges:
            charges[sp] = 0.0
    return BuckinghamParameterSet(
        pairs=list(initial_buck.keys()), species=all_species,
        initial_buck=initial_buck, charges=charges,
        fix_charges=cfg.get("fix_charges", True))


register(PotentialSpec("buckingham", _build_param_set_buckingham,
                       _DEFAULT_FIT_CFG_BUCKINGHAM,
                       build_from_yaml=None, yaml_aliases=("buck",)))


# ============================================================================
#  Lennard-Jones
# ============================================================================

_DEFAULT_FIT_CFG_LJ = {
    "potential"    : "lj",
    "training_csv" : "./training_set/structures.csv",
    "output_dir"   : "./fitting_output",
    "r_cut"        : 10.0,
    "shift"        : True,
    "weights"      : {"energy": 1.0, "forces": 1.0, "stress": 0.1,
                      "r0": 1.0, "eps0": 1.0},
    "use_de"       : False,
    "de_maxiter"   : 300,
    "de_popsize"   : 10,
    "local_method" : "L-BFGS-B",
    "val_fraction" : 0.2,
    "val_seed"     : 0,
    "initial_lj"   : {
        "Ar-Ar": {"epsilon": 0.0103, "sigma": 3.405},
    },
}


def _build_param_set_lj(cfg, all_species, pairs, r_cut):
    initial_lj = _parse_pairs(cfg["initial_lj"])
    for pair in pairs:
        if pair not in initial_lj:
            initial_lj[pair] = {"epsilon": 0.01, "sigma": 3.0}
    return LJParameterSet(
        pairs=list(initial_lj.keys()), species=all_species,
        initial_lj=initial_lj, r_cut=r_cut,
        shift=cfg.get("shift", True))


def _build_from_yaml_lj(pot_cfg):
    species = pot_cfg.get("species", [])
    params  = pot_cfg.get("params", {})
    pair_params, r_cut = _pairs_to_frozenset_dict(params, species, pot_cfg, "LJ")
    engine = PotentialEngine().add(
        LennardJonesPotential(params=pair_params, r_cut=r_cut,
                             shift=bool(pot_cfg.get("shift", True))))
    return make_force_fn(engine), r_cut


register(PotentialSpec("lj", _build_param_set_lj, _DEFAULT_FIT_CFG_LJ,
                       _build_from_yaml_lj,
                       yaml_aliases=("lennardjones", "lennard-jones")))


# ============================================================================
#  Mie (Phase 2: first new potential added via this registry -- a
#  generalized-exponent LJ, exercising the "add a pair potential" recipe
#  end-to-end with minimal risk; see MiePotential's docstring)
# ============================================================================

_DEFAULT_FIT_CFG_MIE = {
    "potential"    : "mie",
    "training_csv" : "./training_set/structures.csv",
    "output_dir"   : "./fitting_output",
    "r_cut"        : 10.0,
    "shift"        : True,
    # n, m: fixed hyperparameters (not fit -- see MiePotential's
    # docstring). Default reproduces standard Lennard-Jones (12, 6).
    "n"            : 12.0,
    "m"            : 6.0,
    "weights"      : {"energy": 1.0, "forces": 1.0, "stress": 0.1,
                      "r0": 1.0, "eps0": 1.0},
    "use_de"       : False,
    "de_maxiter"   : 300,
    "de_popsize"   : 10,
    "local_method" : "L-BFGS-B",
    "val_fraction" : 0.2,
    "val_seed"     : 0,
    "initial_mie"  : {
        "Ar-Ar": {"epsilon": 0.0103, "sigma": 3.405},
    },
}


def _build_param_set_mie(cfg, all_species, pairs, r_cut):
    initial_mie = _parse_pairs(cfg["initial_mie"])
    for pair in pairs:
        if pair not in initial_mie:
            initial_mie[pair] = {"epsilon": 0.01, "sigma": 3.0}
    return MieParameterSet(
        pairs=list(initial_mie.keys()), species=all_species,
        initial_mie=initial_mie, n=cfg.get("n", 12.0), m=cfg.get("m", 6.0),
        r_cut=r_cut, shift=cfg.get("shift", True))


def _build_from_yaml_mie(pot_cfg):
    species = pot_cfg.get("species", [])
    params  = pot_cfg.get("params", {})
    pair_params, r_cut = _pairs_to_frozenset_dict(params, species, pot_cfg, "Mie")
    engine = PotentialEngine().add(
        MiePotential(params=pair_params,
                    n=float(pot_cfg.get("n", 12.0)),
                    m=float(pot_cfg.get("m", 6.0)),
                    r_cut=r_cut, shift=bool(pot_cfg.get("shift", True))))
    return make_force_fn(engine), r_cut


register(PotentialSpec("mie", _build_param_set_mie, _DEFAULT_FIT_CFG_MIE,
                       _build_from_yaml_mie))


# ============================================================================
#  Morse
# ============================================================================

_DEFAULT_FIT_CFG_MORSE = {
    "potential"    : "morse",
    "training_csv" : "./training_set/structures.csv",
    "output_dir"   : "./fitting_output",
    "r_cut"        : 10.0,
    "shift"        : True,
    "weights"      : {"energy": 1.0, "forces": 1.0, "stress": 0.1,
                      "r0": 1.0, "eps0": 1.0},
    "use_de"       : False,
    "de_maxiter"   : 300,
    "de_popsize"   : 10,
    "local_method" : "L-BFGS-B",
    "val_fraction" : 0.2,
    "val_seed"     : 0,
    "initial_morse": {
        "Fe-Fe": {"D_e": 0.42, "alpha": 1.40, "r0": 2.55},
    },
}


def _build_param_set_morse(cfg, all_species, pairs, r_cut):
    initial_morse = _parse_pairs(cfg["initial_morse"])
    for pair in pairs:
        if pair not in initial_morse:
            initial_morse[pair] = {"D_e": 0.5, "alpha": 1.5, "r0": 2.5}
    return MorseParameterSet(
        pairs=list(initial_morse.keys()), species=all_species,
        initial_morse=initial_morse, r_cut=r_cut,
        shift=cfg.get("shift", True))


def _build_from_yaml_morse(pot_cfg):
    species = pot_cfg.get("species", [])
    params  = pot_cfg.get("params", {})
    pair_params, r_cut = _pairs_to_frozenset_dict(params, species, pot_cfg, "Morse")
    engine = PotentialEngine().add(
        MorsePotential(params=pair_params, r_cut=r_cut,
                      shift=bool(pot_cfg.get("shift", True))))
    return make_force_fn(engine), r_cut


register(PotentialSpec("morse", _build_param_set_morse, _DEFAULT_FIT_CFG_MORSE,
                       _build_from_yaml_morse))


# ============================================================================
#  Born-Mayer
# ============================================================================

_DEFAULT_FIT_CFG_BORNMAYER = {
    "potential"    : "bornmayer",
    "training_csv" : "./training_set/structures.csv",
    "output_dir"   : "./fitting_output",
    "ewald_alpha"  : None,
    "fix_charges"  : True,
    "charges"      : {"Na": 1.0, "Cl": -1.0},
    "r_cut"        : 10.0,
    "weights"      : {"energy": 1.0, "forces": 1.0, "stress": 0.1,
                      "r0": 1.0, "eps0": 1.0},
    "use_de"       : False,
    "de_maxiter"   : 300,
    "de_popsize"   : 10,
    "local_method" : "L-BFGS-B",
    "val_fraction" : 0.2,
    "val_seed"     : 0,
    "initial_bornmayer": {
        "Na-Cl": {"A": 2314.7, "rho": 0.290},
        "Na-Na": {"A":  106.0, "rho": 0.317},
        "Cl-Cl": {"A": 1227.2, "rho": 0.321},
    },
}


def _build_param_set_bornmayer(cfg, all_species, pairs, r_cut):
    charges    = cfg.get("charges", {})
    initial_bm = _parse_pairs(cfg["initial_bornmayer"])
    for pair in pairs:
        if pair not in initial_bm:
            initial_bm[pair] = {"A": 1000.0, "rho": 0.3}
    for sp in all_species:
        if sp not in charges:
            charges[sp] = 0.0
    return BornMayerParameterSet(
        pairs=list(initial_bm.keys()), species=all_species,
        initial_bm=initial_bm, charges=charges,
        r_cut=r_cut, fix_charges=cfg.get("fix_charges", True))


def _build_from_yaml_bornmayer(pot_cfg):
    species = pot_cfg.get("species", [])
    params  = pot_cfg.get("params", {})
    pair_params, r_cut = _pairs_to_frozenset_dict(params, species, pot_cfg, "BornMayer")
    engine = PotentialEngine().add(
        BornMayerPotential(params=pair_params, r_cut=r_cut))
    return make_force_fn(engine), r_cut


register(PotentialSpec("bornmayer", _build_param_set_bornmayer,
                       _DEFAULT_FIT_CFG_BORNMAYER, _build_from_yaml_bornmayer,
                       yaml_aliases=("born-mayer",)))


# ============================================================================
#  Yukawa (Phase 2: second new potential, demonstrating a case where the
#  fitted per-pair parameter is signed rather than positive; see
#  YukawaPotential's docstring)
# ============================================================================

_DEFAULT_FIT_CFG_YUKAWA = {
    "potential"    : "yukawa",
    "training_csv" : "./training_set/structures.csv",
    "output_dir"   : "./fitting_output",
    "r_cut"        : 10.0,
    "shift"        : True,
    # kappa: fixed global inverse screening length [A^-1] (not fit --
    # see YukawaPotential's docstring).
    "kappa"        : 1.0,
    "weights"      : {"energy": 1.0, "forces": 1.0, "stress": 0.1,
                      "r0": 1.0, "eps0": 1.0},
    "use_de"       : False,
    "de_maxiter"   : 300,
    "de_popsize"   : 10,
    "local_method" : "L-BFGS-B",
    "val_fraction" : 0.2,
    "val_seed"     : 0,
    # dict of "Sp1-Sp2": {"A": <eV.A>}. A ~ k_e.q_i.q_j (k_e = 14.3996
    # eV.A, see constants.K_E) -- signed: opposite charges (Na-Cl)
    # attractive (A<0), like charges (Na-Na/Cl-Cl) repulsive (A>0).
    "initial_yukawa": {
        "Na-Cl": {"A": -14.3996},
        "Na-Na": {"A":  14.3996},
        "Cl-Cl": {"A":  14.3996},
    },
}


def _build_param_set_yukawa(cfg, all_species, pairs, r_cut):
    initial_yukawa = _parse_pairs(cfg["initial_yukawa"])
    for pair in pairs:
        if pair not in initial_yukawa:
            initial_yukawa[pair] = {"A": 0.0}
    return YukawaParameterSet(
        pairs=list(initial_yukawa.keys()), species=all_species,
        initial_yukawa=initial_yukawa, kappa=cfg.get("kappa", 1.0),
        r_cut=r_cut, shift=cfg.get("shift", True))


def _build_from_yaml_yukawa(pot_cfg):
    species = pot_cfg.get("species", [])
    params  = pot_cfg.get("params", {})
    pair_params, r_cut = _pairs_to_frozenset_dict(params, species, pot_cfg, "Yukawa")
    engine = PotentialEngine().add(
        YukawaPotential(params=pair_params,
                       kappa=float(pot_cfg.get("kappa", 1.0)),
                       r_cut=r_cut, shift=bool(pot_cfg.get("shift", True))))
    return make_force_fn(engine), r_cut


register(PotentialSpec("yukawa", _build_param_set_yukawa,
                       _DEFAULT_FIT_CFG_YUKAWA, _build_from_yaml_yukawa))


# ============================================================================
#  Stillinger-Weber
# ============================================================================

_DEFAULT_FIT_CFG_SW = {
    "potential"    : "sw",
    "training_csv" : "./training_set/structures.csv",
    "output_dir"   : "./fitting_output",
    # Bug-fix note (S25): this template must define "r_cut" itself --
    # cmd_fit's r_cut = float(cfg["r_cut"]) is unconditional for every
    # potential type. a*sigma (matching _build_from_yaml_sw's own SW
    # cutoff formula) with the example initial_sw values below:
    # 1.80 * 2.0951 = 3.77118 A.
    "r_cut"        : 3.77118,
    "weights"      : {"energy": 1.0, "forces": 1.0, "stress": 0.1,
                      "r0": 1.0, "eps0": 1.0},
    "use_de"       : True,
    "de_maxiter"   : 500,
    "de_popsize"   : 15,
    "local_method" : "L-BFGS-B",
    "val_fraction" : 0.2,
    "val_seed"     : 0,
    # Keys: species list + a FLAT parameter dict matching
    # StillingerWeberParameterSet (F2 bug-fix note -- NOT keyed by
    # species pairs/triples).
    "species"      : ["Si"],
    "initial_sw"   : {
        "epsilon": 2.1683, "sigma": 2.0951, "A": 7.0495820,
        "B": 0.6022245, "p": 4.0, "q": 0.0,
        "a": 1.80, "gamma": 1.20, "lambda": 21.0, "costheta0": -0.333333,
    },
    "fix_params"   : None,
    "fit_keys"     : None,
}


def _build_param_set_sw(cfg, all_species, pairs, r_cut):
    # Bug-fix note (F2): StillingerWeberParameterSet.initial_sw is a FLAT
    # dict of scalar parameter names ("epsilon", "sigma", ...), not
    # keyed by species pairs/triples like LJ/Morse/BornMayer/Buckingham
    # -- passed through unmodified, NOT through _parse_pairs.
    return StillingerWeberParameterSet(
        initial_sw=dict(cfg["initial_sw"]),
        species=cfg.get("species", all_species),
        fix_params=cfg.get("fix_params"),
        fit_keys=cfg.get("fit_keys"))


def _build_from_yaml_sw(pot_cfg):
    species = pot_cfg.get("species", [])
    params  = pot_cfg.get("params", {})
    # Two supported YAML formats for potential.params:
    # (a) Legacy flat dict (single species, e.g. Si).
    # (b) Multi-species list-of-triples (LAMMPS .sw-file-like
    #     convention), required whenever more than one species is
    #     present (e.g. GeSn).
    if isinstance(params, list):
        nested = {}
        for rec in params:
            rec = dict(rec)
            key = (rec.pop("i"), rec.pop("j"), rec.pop("k"))
            nested[key] = rec
        pot = StillingerWeberPotential(params=nested, species=species)
        r_cut = max(float(v["a"]) * float(v["sigma"])
                   for (i, j, k), v in nested.items() if j == k)
    else:
        pot   = StillingerWeberPotential(params=params, species=species)
        r_cut = params.get("a", 1.80) * params.get("sigma", 2.0951)
    engine = PotentialEngine().add(pot)
    return make_force_fn(engine), r_cut


register(PotentialSpec("sw", _build_param_set_sw, _DEFAULT_FIT_CFG_SW,
                       _build_from_yaml_sw,
                       yaml_aliases=("stillingerweber", "stillinger-weber")))


# ============================================================================
#  Tersoff
# ============================================================================

_DEFAULT_FIT_CFG_TERSOFF = {
    "potential"    : "tersoff",
    "training_csv" : "./training_set/structures.csv",
    "output_dir"   : "./fitting_output",
    # Bug-fix note (S25): see the identical note in _DEFAULT_FIT_CFG_SW.
    # R + D (matching _build_from_yaml_tersoff's own cutoff formula)
    # with the example initial_tersoff values below: 1.95+0.15 = 2.10 A.
    "r_cut"        : 2.10,
    "weights"      : {"energy": 1.0, "forces": 1.0, "stress": 0.1,
                      "r0": 1.0, "eps0": 1.0},
    "use_de"       : True,
    "de_maxiter"   : 500,
    "de_popsize"   : 15,
    "local_method" : "L-BFGS-B",
    "val_fraction" : 0.2,
    "val_seed"     : 0,
    # Standard Tersoff parameters for C (Tersoff 1988); a FLAT dict, not
    # keyed by species pairs/triples (F2 bug-fix note).
    "species"      : ["C"],
    "initial_tersoff": {
        "A": 1393.6, "B": 346.74, "lambda1": 3.4879,
        "lambda2": 2.2119, "lambda3": 0.0, "alpha": 0.0,
        "beta": 1.5724e-7, "n": 0.72751, "c": 38049.0,
        "d": 4.3484, "h": -0.57058, "R": 1.95, "D": 0.15,
    },
    "fix_params"   : None,
    "fit_keys"     : None,
}


def _build_param_set_tersoff(cfg, all_species, pairs, r_cut):
    return TersoffParameterSet(
        initial_t=dict(cfg["initial_tersoff"]),
        species=cfg.get("species", all_species),
        fix_params=cfg.get("fix_params"),
        fit_keys=cfg.get("fit_keys"))


def _build_from_yaml_tersoff(pot_cfg):
    species = pot_cfg.get("species", [])
    params  = pot_cfg.get("params", {})
    # Two supported YAML formats for potential.params (see SW above for
    # the identical pattern): legacy flat dict, or multi-species
    # list-of-triples (LAMMPS .tersoff-file convention).
    if isinstance(params, list):
        nested = {}
        for rec in params:
            rec = dict(rec)
            key = (rec.pop("i"), rec.pop("j"), rec.pop("k"))
            nested[key] = rec
        pot = TersoffPotential(params=nested, species=species)
        r_cut = max(float(v["R"]) + float(v["D"])
                   for (i, j, k), v in nested.items() if j == k)
    else:
        pot   = TersoffPotential(params=params, species=species)
        r_cut = float(params.get("R", 2.85)) + float(params.get("D", 0.15))
    engine = PotentialEngine().add(pot)
    return make_force_fn(engine), r_cut


register(PotentialSpec("tersoff", _build_param_set_tersoff,
                       _DEFAULT_FIT_CFG_TERSOFF, _build_from_yaml_tersoff,
                       yaml_aliases=("t3",)))


# ============================================================================
#  EAM (YAML-bridge lookup only -- no `fit --potential` support: EAM's
#  parameters are tabulated per-species functions, not scalars a generic
#  optimiser can fit; see _build_from_yaml_eam for why the bridge itself
#  is also unimplemented)
# ============================================================================

def _build_from_yaml_eam(pot_cfg):
    # Bug-fix note (S19): EAMPotential.__init__ takes F_table/rho_table/
    # phi_table (tabulated per-species embedding/density/pair-potential
    # functions, normally loaded from a DYNAMO/setfl file), not a flat
    # `params: {key: value}` scalar dict. This project has no setfl/
    # DYNAMO table-file parser to source those from, so there is no
    # working design here to "restore" -- constructing an EAMPotential
    # directly in Python (bypassing YAML) remains fully supported and
    # tested (see test_engine_forces.py); only the YAML bridge for it is
    # unimplemented.
    raise ValueError(
        "_build_force_fn_from_yaml: 'EAM' cannot be constructed from "
        "a flat YAML 'params:' dict -- EAMPotential requires tabulated "
        "F_table/rho_table/phi_table functions per species (typically "
        "from a DYNAMO/setfl file), which this project has no YAML-"
        "level loader for. Construct EAMPotential directly in Python "
        "and pass it to PotentialEngine/make_force_fn instead.")


register(PotentialSpec("eam", build_from_yaml=_build_from_yaml_eam))
