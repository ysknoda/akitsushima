"""Potential persistence: save / load fitted parameters to / from JSON,
plus the CSV-based per-structure loss-weight utilities.
"""
import json
import os
from typing import Optional, Tuple

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
    "save_potential",
    "load_potential",
    "load_fit_results",
    "_load_potential_json",
    "make_scale_csv",
    "load_scale_csv",
    "apply_scales_to_structures",
]

# Serialisation helpers for frozenset pair keys
_POT_TYPE_KEY    = "potential_type"
_POT_PARAMS_KEY  = "params"
_POT_RCUT_KEY    = "r_cut"
_POT_SPECIES_KEY = "species"
_POT_LOSS_KEY    = "final_loss"


def _pair_to_str(pair: frozenset) -> str:
    """frozenset -> 'Sp1-Sp2' string (sorted for determinism)."""
    parts = sorted(pair)
    if len(parts) == 1:
        # Same-species pair: repeat the element
        return f"{parts[0]}-{parts[0]}"
    return "-".join(parts)


def _str_to_pair(s: str) -> frozenset:
    """'Sp1-Sp2' string -> frozenset({Sp1, Sp2})."""
    parts = s.split("-")
    return frozenset(parts)   # frozenset handles duplicates: {'Ar','Ar'}->{'Ar'}


def save_potential(result        : dict,
                   path          : str,
                   potential_type: str,
                   r_cut         : Optional[float] = None,
                   species       : Optional[list]  = None,
                   extra         : Optional[dict]  = None) -> None:
    """
    Save fitted potential parameters to a JSON file.

    Compatible with the ``result`` dict returned by
    :func:`~akitsushima.fitting.driver.fit_potential` (LJ, Morse,
    BornMayer, Buckingham, SW, Tersoff).  The saved JSON can be reloaded
    with ``load_potential()`` to reconstruct a ``PotentialEngine`` ready
    for ``make_force_fn()``.

    Parameters
    ----------
    result         : dict returned by fit_potential().
    path           : output file path (.json extension added if absent).
    potential_type : ``'lj'`` | ``'morse'`` | ``'bornmayer'`` |
                     ``'buckingham'`` | ``'sw'`` | ``'tersoff'``
    r_cut          : cutoff [A] stored in the JSON for reload convenience.
    species        : species list stored in the JSON.
    extra          : additional key-value pairs to include in the JSON.

    JSON layout (example -- LJ)
    --------------------------
    .. code-block:: json

        {
          "potential_type": "lj",
          "r_cut": 10.0,
          "species": ["Ar"],
          "final_loss": 1.23e-8,
          "params": { "Ar-Ar": {"epsilon": 0.0103, "sigma": 3.405} }
        }

    Ionic potentials (Buckingham, BornMayer) additionally store ``"charges"``.
    SW and Tersoff store all model parameters as a flat dict under ``"params"``.

    Examples
    --------
    >>> result = fit_potential(LJParameterSet(prs, ['Ar'], init, r_cut=10.),
    ...                        structs, r_cut=10.)
    >>> save_potential(result, "lj_Ar.json", potential_type='lj',
    ...                r_cut=10., species=['Ar'])
    """
    if not path.endswith('.json'):
        path = path + '.json'

    pot_type = potential_type.lower().strip()

    # extract params and charges from result dict
    params_raw = None
    charges    = None

    if 'params' in result:
        inner = result['params']
        if pot_type in inner:
            # direct match: e.g. pot_type='lj' and inner has 'lj' key
            params_raw = inner[pot_type]
        elif pot_type == 'buckingham' and 'buck' in inner:
            # BuckinghamParameterSet uses 'buck' as key
            params_raw = inner['buck']
        else:
            # SW / Tersoff: result['params'] is the full param dict itself
            params_raw = inner
        if 'charges' in inner:
            charges = inner['charges']

    if params_raw is None:
        raise ValueError(
            f"save_potential: cannot extract '{pot_type}' params from result. "
            f"Available keys: {list(result.keys())}")

    def _serialise(p):
        """Recursively convert frozenset keys -> 'Sp1-Sp2' strings."""
        if isinstance(p, dict):
            return {(_pair_to_str(k) if isinstance(k, frozenset) else str(k)):
                    _serialise(v) for k, v in p.items()}
        return p

    # Multi-species SW decodes to {(i, j, k): {...}}.  Tuple keys are not
    # JSON; stringifying them (as _serialise would) produced
    # "('Ge', 'Ge', 'Sn')" keys that nothing could read back.  Write the
    # i/j/k record list instead -- the same format as potential.params in
    # a relax/md YAML, so a fitted result can be pasted straight in.
    if (pot_type in ('sw', 'stillinger_weber', 'stillingerweber')
            and isinstance(params_raw, dict) and params_raw
            and all(isinstance(t, tuple) for t in params_raw)):
        serialised_params = [
            dict(i=t[0], j=t[1], k=t[2], **{k: float(v) for k, v in p.items()})
            for t, p in params_raw.items()]
    else:
        serialised_params = _serialise(params_raw)
    out: dict = {
        _POT_TYPE_KEY  : pot_type,
        _POT_PARAMS_KEY: serialised_params,
        _POT_LOSS_KEY  : float(result.get('loss', float('nan'))),
    }
    if r_cut is not None:
        out[_POT_RCUT_KEY] = float(r_cut)
    if species is not None:
        out[_POT_SPECIES_KEY] = list(species)
    if charges is not None:
        out['charges'] = charges
    if extra:
        out.update(extra)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"  Potential saved -> {path}")


def load_potential(path    : str,
                   r_cut   : Optional[float] = None,
                   species : Optional[list]  = None,
                   shift   : bool            = True,
                   ) -> Tuple['PotentialEngine', dict]:
    """
    Load a fitted potential from a JSON file and reconstruct a
    ready-to-use ``PotentialEngine``.

    Parameters
    ----------
    path    : path to the JSON file (.json extension auto-appended).
    r_cut   : override the cutoff stored in the file [A].
    species : override the species list stored in the file.
    shift   : apply force-shifted truncation (default True).  Honoured
              by LJ, Morse, BornMayer and Buckingham.  The last two
              used to ignore it -- they had no shift support at all --
              which this docstring already claimed for BornMayer.

    Returns
    -------
    engine  : PotentialEngine --- pass directly to make_force_fn(engine).
    meta    : dict with keys ``potential_type``, ``r_cut``, ``species``,
              ``params``, ``charges`` (ionic only), ``loss``.

    Notes
    -----
    For **Buckingham + Coulomb**, ``engine`` contains the full
    ``BuckinghamPotential`` (A, rho, *and* the C dispersion term); pass
    ``meta['charges']`` to ``make_force_fn`` / ``compute_all_general``
    separately to add the Ewald/PPPM electrostatics on top.

    Examples
    --------
    >>> # LJ / Morse / BornMayer / SW / Tersoff
    >>> engine, meta = load_potential("lj_Ar.json")
    >>> fn = make_force_fn(engine)
    >>> sim = NVTSimulation(lat, pos, types, force_fn=fn, temperature=300)

    >>> # Buckingham + Coulomb (ionic)
    >>> engine, meta = load_potential("mgo_buck.json")
    >>> fn = make_force_fn('buckingham',
    ...         buck_params=meta['params'], charges=meta['charges'],
    ...         r_cut=meta['r_cut'])
    >>> sim = NVTSimulation(lat, pos, types, force_fn=fn, temperature=1000)
    """
    if not path.endswith('.json'):
        path = path + '.json'
    if not os.path.isfile(path):
        raise FileNotFoundError(f"load_potential: '{path}' not found.")

    with open(path) as fh:
        data = json.load(fh)

    pot_type = data[_POT_TYPE_KEY].lower()
    rc       = float(r_cut if r_cut is not None
                     else data.get(_POT_RCUT_KEY, 10.0))
    sp       = list(species if species is not None
                    else data.get(_POT_SPECIES_KEY, []))
    loss     = data.get(_POT_LOSS_KEY, float('nan'))
    charges  = data.get('charges')

    # deserialise pair-keyed params ("Sp1-Sp2" -> frozenset)
    raw = data[_POT_PARAMS_KEY]
    def _deser(d):
        out = {}
        for k, v in d.items():
            # heuristic: if the key contains '-' and is not purely numeric,
            # treat it as a species-pair string
            parts = k.split('-')
            if len(parts) == 2 and not k.lstrip('-').replace('.','').replace('e','').isdigit():
                out[frozenset(parts)] = v
            else:
                out[k] = v
        return out
    params = _deser(raw) if isinstance(raw, dict) else raw

    # build PotentialEngine
    engine = PotentialEngine()

    if pot_type == 'lj':
        engine.add(LennardJonesPotential(params, r_cut=rc, shift=shift))

    elif pot_type == 'morse':
        engine.add(MorsePotential(params, r_cut=rc, shift=shift))

    elif pot_type in ('bornmayer', 'born_mayer'):
        engine.add(BornMayerPotential(params, r_cut=rc, shift=shift))

    elif pot_type == 'buckingham':
        # Bug-fix note (F4, load side): the original reconstructed only a
        # BornMayerPotential(A, rho) here, silently discarding the C
        # dispersion term and warning the caller to reload it another
        # way.  Now that BuckinghamPotential(A, rho, C) exists, build it
        # directly from the full deserialised params -- no information
        # is lost and no extra reload step is required.
        engine.add(BuckinghamPotential(params, r_cut=rc, shift=shift))

    elif pot_type in ('sw', 'stillinger_weber', 'stillingerweber'):
        sw_params = params.get('sw', params) if isinstance(params, dict) else params
        if isinstance(sw_params, list):          # multi-species records
            nested = {}
            for rec in sw_params:
                rec = dict(rec)
                nested[(rec.pop('i'), rec.pop('j'), rec.pop('k'))] = rec
            sw_params = nested
            # The multi-species format requires `species`, and
            # save_fit_result_generic does not persist it.  The records
            # name every element themselves, so read them from there.
            if not sp:
                sp = list(dict.fromkeys(e for t in nested for e in t))
        engine.add(StillingerWeberPotential(sw_params, r_cut=None,
                                             species=sp or None))

    elif pot_type == 'tersoff':
        engine.add(TersoffPotential(params, species=sp or None))

    elif pot_type == 'mie':
        # n/m are fixed (not-fit) hyperparameters -- see
        # save_fit_result_generic's extra_meta note. Persisted verbatim
        # by cmd_fit.py; fall back to the LJ-reducing (12, 6) default
        # only for a file saved/hand-written without them.
        engine.add(MiePotential(params, n=data.get('n', 12.0),
                                m=data.get('m', 6.0), r_cut=rc, shift=shift))

    elif pot_type == 'yukawa':
        # kappa is a fixed (not-fit) hyperparameter -- see
        # save_fit_result_generic's extra_meta note.
        engine.add(YukawaPotential(params, kappa=data.get('kappa', 1.0),
                                   r_cut=rc, shift=shift))

    else:
        raise ValueError(
            f"load_potential: unknown type '{pot_type}'. "
            "Supported: lj, morse, bornmayer, buckingham, sw, tersoff, "
            "mie, yukawa.")

    meta = {'potential_type': pot_type, 'r_cut': rc, 'species': sp,
            'params': params, 'charges': charges, 'loss': loss}
    print(f"  Potential loaded <- {path}  "
          f"(type={pot_type}  r_cut={rc}  loss={loss:.3e})")
    return engine, meta


def load_fit_results(path: str) -> dict:
    """
    Load Buckingham + Coulomb fitting results from a JSON file.

    Supports files saved by ``save_potential(..., potential_type='buckingham')``.

    Returns a dict with keys ``buck_params``, ``charges``, ``final_loss``,
    and optionally ``r_cut``, ready to pass directly to
    ``make_force_fn('buckingham', ...)``.

    Examples
    --------
    >>> res = load_fit_results("fitted_potential.json")
    >>> fn  = make_force_fn('buckingham',
    ...           buck_params = res['buck_params'],
    ...           charges     = res['charges'],
    ...           r_cut       = res.get('r_cut', 10.0))
    >>> sim = NVTSimulation(lat, pos, types, force_fn=fn, temperature=1000)
    """
    if not path.endswith('.json'):
        path = path + '.json'
    with open(path) as fh:
        data = json.load(fh)

    if _POT_PARAMS_KEY in data:
        inner_ser = data[_POT_PARAMS_KEY]             # save_potential format
        # params may be nested under 'buck' key (from fit_potential result)
        params_ser = inner_ser.get('buck', inner_ser)
    else:
        raise ValueError(
            f"load_fit_results: '{path}' has no recognised params key.")

    buck_params = {_str_to_pair(k): v for k, v in params_ser.items()}
    charges     = data.get('charges', {})
    loss        = data.get('final_loss', data.get(_POT_LOSS_KEY, float('nan')))
    r_cut       = data.get(_POT_RCUT_KEY)

    print(f"  Buckingham params loaded <- {path}  (loss={loss:.3e})")
    result: dict = {'buck_params': buck_params, 'charges': charges,
                    'final_loss': loss}
    if r_cut is not None:
        result['r_cut'] = float(r_cut)
    return result


def _load_potential_json(path: str) -> Tuple[dict, dict]:
    """
    Load a saved potential JSON and return ``(buck_params, charges)``.

    Bug-fix note (S13): the original script called this exact function
    name from both ``config_bridge.py`` and ``cmd_esm`` (YAML configs of
    the form ``potential: {type: buckingham, potential_json: ...}``),
    but it was never defined anywhere -- every one of those code paths
    raised ``NameError`` unconditionally.  This is a thin, explicit
    wrapper around :func:`load_potential` that extracts just the two
    values those callers need.
    """
    _, meta = load_potential(path)
    return meta['params'], (meta.get('charges') or {})


# ---------------------------------------------------------------------------
#  Unit-conversion scale utilities for the loss function
# ---------------------------------------------------------------------------

def make_scale_csv(structures : list,
                   path       : str,
                   e_scale    : str   = "natoms",
                   f_scale    : float = 1.0,
                   s_scale    : str   = "volume_x_eps0",
                   eps0       : float = 1.0,
                   verbose    : bool  = True):
    """
    Generate per-row unit-conversion scale weights for the ELM loss function.

    .. deprecated::
        This function is superseded by the new 4-column CSV format of
        :func:`~akitsushima.fitting.models.elm.load_mlip_csv`, which
        embeds the weight directly in col 2 of the training CSV.  The
        weight values produced by this function correspond exactly to
        col 2 of the new format:

        * E rows : number of atoms  (``e_scale='natoms'``)
        * F rows : r0 [A]  typical atomic displacement (``f_scale``, default 1.0)
        * S rows : cell volume x eps0  [A^3] (``s_scale='volume_x_eps0'``)

        ``make_scale_csv`` is retained for backward compatibility but
        may be removed in a future version.

    The scale coefficient ``c_i`` for each training row converts the raw
    residual to eV so that ``(c_i x residual)^2`` has unit eV^2:

    ============  ====================  ==========================  ===========
    Category      Residual unit         Scale -> eV                  CSV value
    ============  ====================  ==========================  ===========
    E (energy)    eV/atom               N_atom                       N_atom
    F (force)     eV/A                  r_0 [A]                      r_0
    S (stress)    eV/A^3                 V_s [A^3] x epsilon_0               V_s x epsilon_0
    ============  ====================  ==========================  ===========

    Parameters
    ----------
    structures : list of structure dicts.  Must contain ``natoms``;
                 ``volume`` is read from the dict or computed from
                 ``structure["lattice"]``.
    path       : output path (.csv appended if absent).
    e_scale    : ``'natoms'`` (default) or float constant for E rows.
    f_scale    : scale for every F-component row.
                 Pass the typical atomic displacement r_0 [A] (default 1.0).
    s_scale    : ``'volume_x_eps0'`` (default, uses V_s x epsilon_0) or float
                 constant for S rows.
    eps0       : dimensionless cell strain epsilon_0 used when
                 ``s_scale='volume_x_eps0'`` (default 1.0).
    verbose    : print summary.

    Returns
    -------
    scales : (N_rows,) ndarray written to the CSV.
    """
    import numpy as np

    if not path.endswith('.csv'):
        path = path + '.csv'

    rows: list = []
    for s in structures:
        N   = int(s["natoms"])
        vol = float(s.get("volume",
                          abs(np.linalg.det(s["lattice"]))))

        # E scale: N_atom (so N_s x deltaE_per_atom -> eV)
        ce = float(N)   if e_scale == "natoms"  else float(e_scale)

        # F scale: r_0 [A]
        cf = float(f_scale)

        # S scale: V_s x epsilon_0 [A^3 x --] -> eV/A^3 x V x epsilon_0 = eV
        if s_scale == "volume_x_eps0":
            cs = float(vol * eps0)
        elif s_scale == "volume":
            cs = float(vol)
        else:
            cs = float(s_scale)

        rows.append(ce)                              # 1 E row
        if s.get("ref_forces") is not None:
            rows.extend([cf] * (N * 3))              # N_atom x 3 F rows
        if s.get("ref_stress") is not None:
            rows.extend([cs] * 9)                    # 9 S rows

    scales = np.array(rows, dtype=np.float64)

    with open(path, 'w') as fh:
        fh.write("scale\n")
        for v in scales:
            fh.write(f"{v:.8g}\n")

    if verbose:
        print(f"  Scale CSV saved -> {path}  "
              f"(N_rows={len(scales)}  "
              f"min={scales.min():.4g}  max={scales.max():.4g})")
    return scales


def load_scale_csv(path    : str,
                   n_rows  : Optional[int] = None,
                   verbose : bool = True):
    """
    Load per-row unit-conversion scales from a CSV file.

    The file must have exactly one data column with an optional header row.

    .. deprecated::
        For ELM fitting, embed the scale values directly in col 2 of the
        training CSV and pass via::

            targets, labels, weights, X = load_mlip_csv("structures.csv")
            model.fit(targets, labels, X, weights=weights)

        This function is retained for classical force-field workflows
        (``apply_scales_to_structures(structs, scales)``).

    Parameters
    ----------
    path   : path to the CSV (extension auto-appended if absent).
    n_rows : expected row count (raises ValueError on mismatch).
    verbose: print summary.

    Returns
    -------
    scales : (N_rows,) float64 array, all values >= 0.
    """
    import numpy as np

    if not path.endswith('.csv'):
        path = path + '.csv'

    values: list = []
    with open(path) as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                values.append(float(line))
            except ValueError:
                if lineno == 1:
                    continue    # header row
                raise ValueError(
                    f"load_scale_csv: cannot parse value at line {lineno}: "
                    f"{repr(raw_line.rstrip())!r}")

    scales = np.array(values, dtype=np.float64)
    if np.any(scales < 0.0):
        raise ValueError("load_scale_csv: negative scale values are not allowed.")

    if n_rows is not None and len(scales) != n_rows:
        raise ValueError(
            f"load_scale_csv: expected {n_rows} rows, got {len(scales)}.")

    if verbose:
        print(f"  Scale CSV loaded <- {path}  "
              f"(N_rows={len(scales)}  "
              f"min={scales.min():.4g}  max={scales.max():.4g})")
    return scales


def apply_scales_to_structures(structures, scales) -> list:
    """
    Embed per-row scale coefficients into structure dicts as
    ``structure["scale"]``.

    The ``scales`` array is consumed in the same order as ``make_scale_csv()``:
    E row (1) -> F rows (Nx3, if present) -> S rows (9, if present).

    The per-structure ``"scale"`` is taken from the **E-row coefficient**
    (first value consumed for each structure).  This value is then squared
    inside ``generic_loss_function()``:

        loss contribution ~ c_s^2 x (residual)^2

    Parameters
    ----------
    structures : list of structure dicts (modified in-place).
    scales     : 1-D array from ``load_scale_csv()``.

    Returns
    -------
    structures : same list with ``structure["scale"]`` set.
    """
    import warnings

    idx = 0
    for s in structures:
        N     = int(s["natoms"])
        n_f   = N * 3 if s.get("ref_forces") is not None else 0
        n_s   = 9     if s.get("ref_stress") is not None else 0
        n_tot = 1 + n_f + n_s

        if idx >= len(scales):
            raise ValueError(
                "apply_scales_to_structures: ran out of scale values. "
                f"Expected >= {idx + n_tot} rows.")

        s["scale"] = float(scales[idx])
        idx += n_tot

    if idx != len(scales):
        warnings.warn(
            f"apply_scales_to_structures: {len(scales)} scale rows but "
            f"consumed only {idx}.  Trailing rows ignored.",
            UserWarning, stacklevel=2)

    return structures
