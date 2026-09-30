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
    MultiSpeciesStillingerWeberParameterSet,
    StillingerWeberParameterSet,
    TersoffParameterSet,
    MultiSpeciesTersoffParameterSet,
    Tersoff1989ParameterSet,
    ZhouEAMParameterSet,
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
    # Further `fit --write-config --template NAME` configs (ADR-0015),
    # e.g. the multi-species forms of Tersoff.
    templates       : dict               = field(default_factory=dict)


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
            f"Supported: SW, Tersoff, LJ, Morse, BornMayer, EAM, Buckingham, "
            f"MLIP")


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
        initial_buck=initial_buck, charges=charges, r_cut=r_cut,
        shift=bool(cfg.get("shift", True)),
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
    "local_method" : "Nelder-Mead",
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
    "local_method" : "Nelder-Mead",
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
    "local_method" : "Nelder-Mead",
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
    "local_method" : "Nelder-Mead",
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
        BornMayerPotential(params=pair_params, r_cut=r_cut,
                          shift=bool(pot_cfg.get("shift", True))))
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
    "local_method" : "Nelder-Mead",
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
    "local_method" : "Nelder-Mead",
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
    #
    # Two formats for initial_sw, the same two the relax/md YAML accepts:
    # a flat dict (one parameter set for everything -- single species) or
    # a list of i/j/k records (one per ordered triple -- any number of
    # species).  The flat form cannot distinguish species at all, so a
    # multi-species training set needs the record list.
    init = cfg["initial_sw"]
    if isinstance(init, list):
        return MultiSpeciesStillingerWeberParameterSet(
            init,
            species=cfg.get("species", all_species),
            fix_params=cfg.get("fix_params"),
            fit_keys=cfg.get("fit_keys"))
    if len(set(all_species)) > 1:
        raise ValueError(
            f"the training data contain {sorted(set(all_species))}, but "
            f"initial_sw is a single flat parameter set, which applies the "
            f"same numbers to every pair and triple regardless of species.  "
            f"Give initial_sw as a list of i/j/k records instead (the "
            f"format used by potential.params in a relax/md YAML).")
    return StillingerWeberParameterSet(
        initial_sw=dict(init),
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
    # Off since 2026-09-26 (ADR-0007, decision 8): on the T8 Si test DE
    # settled in a wrong basin along the beta / c / d directions E, F and S
    # cannot resolve (loss 210, beta at its bound) and the fit failed,
    # while Nelder-Mead alone from +10% recovered E0 / V0 / B0 to 2e-4.
    "use_de"       : False,
    "de_maxiter"   : 500,
    "de_popsize"   : 15,
    "local_method" : "Nelder-Mead",
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


def _is_tersoff1989(init) -> bool:
    return (isinstance(init, dict)
            and str(init.get("mixing", "")).lower() == "tersoff1989")


def _build_param_set_tersoff(cfg, all_species, pairs, r_cut):
    # Three formats for initial_tersoff (ADR-0015), the same three the
    # relax/md YAML accepts for potential.params:
    #   flat dict                      -- one element (legacy, unchanged)
    #   list of i/j/k records          -- LAMMPS table, any number of elements
    #   {mixing: tersoff1989, elements: {...}, chi: {...}}
    #                                  -- Tersoff (1989) mixing rules
    init = cfg["initial_tersoff"]
    species = cfg.get("species", all_species)
    if isinstance(init, list) or _is_tersoff1989(init):
        # The elements come from the training data: a `species` key
        # back-filled from the single-element default template ([C]) must
        # not narrow a multi-species fit.
        species = list(all_species)
    if isinstance(init, list):
        return MultiSpeciesTersoffParameterSet(
            init, species=species, fix_params=cfg.get("fix_params"),
            fit_keys=cfg.get("fit_keys"), tie=cfg.get("tie", "record"))
    if _is_tersoff1989(init):
        return Tersoff1989ParameterSet(
            init["elements"], init.get("chi", {}), species=species,
            fix_params=cfg.get("fix_params"), fit_keys=cfg.get("fit_keys"),
            omega=init.get("omega"))
    if len(set(all_species)) > 1:
        raise ValueError(
            f"the training data contain {sorted(set(all_species))}, but "
            f"initial_tersoff is a single flat parameter set, which applies "
            f"the same numbers to every pair and triple regardless of "
            f"species.  Give initial_tersoff as a list of i/j/k records, or "
            f"as {{mixing: tersoff1989, elements: ..., chi: ...}} (ADR-0015).")
    return TersoffParameterSet(
        initial_t=dict(init),
        species=species,
        fix_params=cfg.get("fix_params"),
        fit_keys=cfg.get("fit_keys"))


def _build_from_yaml_tersoff(pot_cfg):
    species = pot_cfg.get("species", [])
    params  = pot_cfg.get("params", {})
    # Two supported YAML formats for potential.params (see SW above for
    # the identical pattern): legacy flat dict, or multi-species
    # list-of-triples (LAMMPS .tersoff-file convention).
    if _is_tersoff1989(params):
        # Tersoff (1989) mixing form (ADR-0015): expanded to the record
        # table below.
        from ..engine.tersoff import tersoff1989_to_triplets
        species = species or list(params["elements"])
        params = tersoff1989_to_triplets(params["elements"],
                                         params.get("chi", {}), species,
                                         params.get("omega"))
    if isinstance(params, (list, dict)) and (
            isinstance(params, list) or any(isinstance(k, tuple) for k in params)):
        nested = {}
        for rec in (params if isinstance(params, list) else
                    [dict(i=t[0], j=t[1], k=t[2], **v) for t, v in params.items()]):
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


def _tersoff_multi_template(initial, r_cut, **extra):
    cfg = {k: v for k, v in _DEFAULT_FIT_CFG_TERSOFF.items()
           if k not in ("species", "initial_tersoff", "r_cut")}
    # least_squares (dogbox on the residual vector): Nelder-Mead crept in
    # the ill-conditioned c-d valley; least_squares reached loss ~1e-15 on
    # Si-C and AlN (ADR-0015).  restarts: a cheap safeguard.
    cfg.update(r_cut=r_cut, initial_tersoff=initial, local_method="least_squares",
               restarts=2, **extra)
    return cfg


def _tersoff1989_sic_initial():
    from ..engine.published_parameters import TERSOFF_1989
    el = TERSOFF_1989["elements"]
    return {"mixing": "tersoff1989",
            "elements": {s: dict(el[s]) for s in ("C", "Si")},
            "chi": {"C-Si": TERSOFF_1989["chi"]["C-Si"]}}


# Tersoff records for AlN: Karaaslan, Yapicioglu and Sevik, Phys. Rev.
# Applied 13, 034027 (2020), Table I (Al-N and N-Al columns), as used by
# example/AlN.  The pair keys of the two records are equal, as required.
_TERSOFF_ALN_RECORDS = [
    {"i": "Al", "j": "N", "k": "N", "A": 1258.567263, "B": 453.228512,
     "lambda1": 2.434869, "lambda2": 1.71768, "beta": 2.133047e-06,
     "n": 0.598233, "R": 2.55, "D": 0.15, "lambda3": 1.186759,
     "c": 19110.741778, "d": 10.88209, "h": -0.972662},
    {"i": "N", "j": "Al", "k": "Al", "A": 1258.567263, "B": 453.228512,
     "lambda1": 2.434869, "lambda2": 1.71768, "beta": 2.496023e-06,
     "n": 1.220882, "R": 2.55, "D": 0.15, "lambda3": 1.100709,
     "c": 27568.039128, "d": 8.292436, "h": -0.816226},
]

register(PotentialSpec("tersoff", _build_param_set_tersoff,
                       _DEFAULT_FIT_CFG_TERSOFF, _build_from_yaml_tersoff,
                       yaml_aliases=("t3",),
                       templates={
                           # R + D = 2.70 A
                           "records": _tersoff_multi_template(
                               _TERSOFF_ALN_RECORDS, 2.70, tie="record"),
                           # S_Si = 3.0 A is the largest cutoff
                           "tersoff1989": _tersoff_multi_template(
                               _tersoff1989_sic_initial(), 3.0),
                       }))


# ============================================================================
#  EAM (ADR-0015): the analytic Zhou et al. (2004) form is fitted and
#  built from YAML; a tabulated setfl file is read from YAML
# ============================================================================

def _zhou_elements(spec, species):
    """`elements:` of a Zhou EAM block: a {symbol: {...}} dict, or a list of
    symbols meaning the published Table III rows (with ZHOU_2004_ERRATA)."""
    from ..engine.published_parameters import ZHOU_2004
    if isinstance(spec, (list, tuple)):
        missing = [s for s in spec if s not in ZHOU_2004]
        if missing:
            raise ValueError(f"EAM zhou2004: {missing} not in Zhou et al. "
                             f"(2004) Table III")
        return {s: dict(ZHOU_2004[s]) for s in spec}
    return {s: dict(v) for s, v in spec.items()}


def _build_from_yaml_eam(pot_cfg):
    # ADR-0015: two YAML forms.
    #   {type: EAM, setfl: file.eam.alloy}          tabulated (LAMMPS eam/alloy)
    #   {type: EAM, form: zhou2004, elements: {...} | [Ni, Al], chi, r_cut}
    # A flat `params:` dict still cannot describe an EAM (S19).
    from ..engine.eam_zhou import read_setfl, zhou_eam
    species = pot_cfg.get("species")
    if "setfl" in pot_cfg:
        pot = read_setfl(pot_cfg["setfl"], species=species)
    elif str(pot_cfg.get("form", "")).lower() == "zhou2004":
        el = _zhou_elements(pot_cfg["elements"], species)
        species = species or list(el)
        pot = zhou_eam(el, species, r_cut=pot_cfg.get("r_cut"),
                       chi=pot_cfg.get("chi"))
    else:
        raise ValueError(
            "_build_force_fn_from_yaml: an EAM potential needs either "
            "'setfl: <file>' (a LAMMPS eam/alloy table) or 'form: zhou2004' "
            "with 'elements:' (Zhou, Johnson & Wadley 2004; ADR-0015).  A "
            "flat YAML 'params:' dict cannot describe the tabulated "
            "F / rho / phi functions of an EAM.")
    engine = PotentialEngine().add(pot)
    return make_force_fn(engine), float(pot.r_cut)


def _build_param_set_eam(cfg, all_species, pairs, r_cut):
    init = cfg["initial_eam"]
    if str(init.get("form", "")).lower() != "zhou2004":
        raise ValueError("fit --potential eam: initial_eam.form must be "
                         "'zhou2004' (the analytic form that can be fitted)")
    return ZhouEAMParameterSet(
        _zhou_elements(init["elements"], all_species), list(all_species),
        chi=init.get("chi"), fix_params=cfg.get("fix_params"),
        fit_keys=cfg.get("fit_keys"), r_cut=r_cut)


def _default_fit_cfg_eam():
    from ..engine.published_parameters import ZHOU_2004
    return {
        "potential"    : "eam",
        "training_csv" : "./training_set/structures.csv",
        "output_dir"   : "./fitting_output",
        # max over Ni, Al of eam_zhou.zhou_cutoff (Al: 6.690 A): every
        # |phi_aa| / max(A, B) and f / f_e below 1e-6 beyond it
        "r_cut"        : 6.6904128564,
        "weights"      : {"energy": 1.0, "forces": 1.0, "stress": 0.1,
                          "r0": 1.0, "eps0": 1.0},
        "use_de"       : False,
        # least_squares: 3 min and loss 7e-14 on the Ni-Al test, where
        # Nelder-Mead with restarts stopped at 2.7e-4 after 8.8 h (ADR-0015)
        "local_method" : "least_squares",
        "val_fraction" : 0.2,
        "val_seed"     : 0,
        "restarts"     : 2,
        # Zhou, Johnson & Wadley (2004) Table III rows for Ni and Al (Al's
        # A corrected, published_parameters.ZHOU_2004_ERRATA); Fn0, Fn1,
        # F0, F_e are re-derived from continuity whatever is given here.
        "initial_eam"  : {"form": "zhou2004",
                          "elements": {s: dict(ZHOU_2004[s]) for s in ("Al", "Ni")},
                          "chi": {"Al-Ni": 1.0}},
        "fix_params"   : None,
        "fit_keys"     : None,
    }


register(PotentialSpec("eam", _build_param_set_eam, _default_fit_cfg_eam(),
                       build_from_yaml=_build_from_yaml_eam))


# ============================================================================
#  MLIP (YAML-bridge lookup only; trained by `akitsushima mlip`)
# ============================================================================

def _build_from_yaml_mlip(pot_cfg):
    """``potential: {type: MLIP, model: path}`` (ADR-0004, decision 6).
    The descriptor and element list come from the model file."""
    from .mlip_engine import MLIPForceField
    if pot_cfg.get("coulomb_method") or pot_cfg.get("charges") or pot_cfg.get("esm"):
        # ADR-0008: a model trained on total energies already contains the
        # electrostatics; adding a Coulomb term would count it twice.
        raise ValueError("potential.type MLIP takes no coulomb_method / charges / "
                         "esm: the model's energy already includes electrostatics "
                         "(ADR-0008).")
    path = pot_cfg.get("model")
    if not path:
        raise ValueError("potential.type MLIP needs potential.model: the "
                         "model file written by `akitsushima mlip`.")
    fn = MLIPForceField.from_file(path)
    return fn, fn.r_cut


register(PotentialSpec("mlip", build_from_yaml=_build_from_yaml_mlip))


# ============================================================================
#  External code through files (ExternalFileEngine, ADR-0011)
# ============================================================================

_EXTERNAL_KEYS = {"type", "cmd", "work_dir", "input_file", "output_file",
                  "stress_to_eV_Ang3", "timeout", "mpi_prefix", "cache_tol",
                  "r_cut", "verbose"}


def _build_from_yaml_external(pot_cfg):
    """``potential: {type: external, cmd: ...}``: every call writes POSCAR,
    runs ``cmd`` in ``work_dir`` and reads the EFS file ``output_file``
    (akitsushima.io.efs; E per atom, F, S in the akitsushima conventions --
    a wrapper script converts the external code's output, see
    ``python -m akitsushima.tools.ase_efs``)."""
    from .external import ExternalFileEngine
    if pot_cfg.get("coulomb_method") or pot_cfg.get("charges") or pot_cfg.get("esm"):
        raise ValueError(
            "potential.type external takes no coulomb_method / charges / esm: "
            "the external total energy already contains the electrostatics "
            "(ADR-0011).")
    unknown = sorted(set(pot_cfg) - _EXTERNAL_KEYS)
    if unknown:
        raise ValueError(f"potential.type external: unknown key(s) {unknown}; "
                         f"allowed: {sorted(_EXTERNAL_KEYS - {'type'})}.")
    cmd = str(pot_cfg.get("cmd", "")).strip()
    if not cmd:
        raise ValueError("potential.type external needs potential.cmd: the "
                         "shell command that writes the EFS file.")
    timeout = pot_cfg.get("timeout")
    r_cut = float(pot_cfg.get("r_cut", 1.0))
    fn = ExternalFileEngine(
        cmd=cmd,
        work_dir=str(pot_cfg.get("work_dir", "./external_run")),
        input_file=str(pot_cfg.get("input_file", "POSCAR")),
        output_file=str(pot_cfg.get("output_file", "result.efs")),
        mpi_prefix=str(pot_cfg.get("mpi_prefix", "")),
        timeout=None if timeout is None else float(timeout),
        stress_to_eV_Ang3=float(pot_cfg.get("stress_to_eV_Ang3", 1.0)),
        cache_tol=float(pot_cfg.get("cache_tol", 1e-6)),
        r_cut=r_cut,
        verbose=bool(pot_cfg.get("verbose", False)),
    )
    return fn, r_cut


register(PotentialSpec("external", build_from_yaml=_build_from_yaml_external))
