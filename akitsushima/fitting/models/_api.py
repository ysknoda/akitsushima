"""Physics options and structure-level fitting for every MLIP model
(ADR-0024, Q23).

Before this, a model was a force field with a pair baseline or an
isolated-atom energy only after hand-steps that nothing documented: a
``physics_config`` JSON string built by hand, the baseline subtracted from
the training labels by hand, and -- for a tabular model -- a
``descriptor_config`` set by hand before and after ``fit`` (ADR-0023 G3,
G4, O10).  The decorators here add, without touching each signature:

* ``baseline=`` (a :class:`~akitsushima.fitting.physics.PairBaseline`) and
  ``isolated_atom_energy=`` ({species: eV}) to every model constructor,
  and ``model.baseline`` / ``model.set_physics(...)``;
* tabular models: ``fit(..., descriptor=)`` sets ``descriptor_config``,
  and ``fit_structures(structures, descriptor)`` builds the ACSF table
  from structure dicts, minus the baseline, and fits;
* atom-centred models: ``fit`` and ``score_structures`` subtract the
  model's baseline from structure dicts themselves.
"""

import functools
import inspect
import json


def _physics_json(baseline, isolated_atom_energy) -> str:
    cfg = {}
    if baseline is not None:
        cfg["baseline"] = baseline.config()
    if isolated_atom_energy:
        cfg["isolated_atom_energy"] = {str(k): float(v) for k, v in isolated_atom_energy.items()}
    return json.dumps(cfg) if cfg else ""


def _baseline(self):
    """The model's pair baseline (PairBaseline) or None."""
    from ..physics import physics_from_config
    return physics_from_config(getattr(self, "physics_config", ""))[0]


def _set_physics(self, baseline=None, isolated_atom_energy=None):
    """Set the pair baseline and / or isolated-atom energies (eV) the model
    is trained and evaluated with.  Call before ``fit``."""
    self.physics_config = _physics_json(baseline, isolated_atom_energy)
    return self


def _wrap_init(cls):
    init = cls.__init__

    @functools.wraps(init)
    def __init__(self, *args, baseline=None, isolated_atom_energy=None, **kw):
        init(self, *args, **kw)
        if baseline is not None or isolated_atom_energy:
            _set_physics(self, baseline, isolated_atom_energy)

    sig = inspect.signature(init)
    params = [p for p in sig.parameters.values() if p.kind is not p.VAR_KEYWORD]
    extra = [inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=None)
             for n in ("baseline", "isolated_atom_energy")]
    tail = [p for p in sig.parameters.values() if p.kind is p.VAR_KEYWORD]
    pos = [p for p in params if p.kind is not p.KEYWORD_ONLY]
    kwo = [p for p in params if p.kind is p.KEYWORD_ONLY]
    __init__.__signature__ = sig.replace(parameters=pos + kwo + extra + tail)
    cls.__init__ = __init__


def _common(cls):
    if "__init__" in vars(cls):
        _wrap_init(cls)
    cls.baseline = property(_baseline)
    cls.set_physics = _set_physics


def tabular_mlip(cls):
    """Class decorator for a tabular model (fit on a descriptor table)."""
    _common(cls)
    if "fit" in vars(cls):
        fit = cls.fit

        @functools.wraps(fit)
        def fit_with_descriptor(self, *args, descriptor=None, **kw):
            if descriptor is not None:
                # read by fit (w_correlation's channel layout) and by the
                # force field afterwards
                self.descriptor_config = descriptor.to_json()
            out = fit(self, *args, **kw)
            if descriptor is not None:
                self.descriptor_config = descriptor.to_json()
            return out

        cls.fit = fit_with_descriptor

    def fit_structures(self, structures, descriptor, val_fraction=0.2, val_seed=0,
                       weight_csv=None):
        """Fit on structure dicts (``akitsushima.io.load_training_set``) with
        an ACSF ``descriptor``: the table is built with the model's ``r0``
        and ``eps0`` after subtracting the model's baseline, and the model
        becomes a force field (``make_force_fn(model)``).  ``seed`` (given to
        the constructor) fixes the model's random initialisation;
        ``val_seed`` fixes which structures go to the validation split."""
        from ..descriptors import build_table
        base = self.baseline
        if base is not None:
            structures = base.subtract(structures)
        y, lab, w, X = build_table(structures, descriptor,
                                   r0=getattr(self, "r0", 1.0), eps0=getattr(self, "eps0", 0.01))
        kw = {}
        if "columns" in inspect.signature(self.fit).parameters:   # LinearModel
            kw["columns"] = descriptor.feature_names()
        if weight_csv is not None:
            kw["weight_csv"] = weight_csv
        return self.fit(y, lab, X, weights=w, val_fraction=val_fraction, val_seed=val_seed,
                        descriptor=descriptor, **kw)

    if "fit_structures" not in vars(cls):
        cls.fit_structures = fit_structures
    return cls


def atomic_mlip(cls):
    """Class decorator for an atom-centred model (fit on structure dicts)."""
    _common(cls)

    def subtracting(method):
        @functools.wraps(method)
        def wrapped(self, structures, *args, **kw):
            base = self.baseline
            if base is not None and isinstance(structures, (list, tuple)):
                structures = base.subtract(structures)
            return method(self, structures, *args, **kw)
        return wrapped

    for name in ("fit", "score_structures"):
        if name in vars(cls) and not (name == "score_structures" and cls.__name__ == "AtomicNNModel"):
            setattr(cls, name, subtracting(vars(cls)[name]))
    return cls
