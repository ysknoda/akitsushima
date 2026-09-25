"""ML-potential model classes.

``NNModel`` and ``LightGBMModel`` are resolved lazily (PEP 562 module
``__getattr__``) rather than imported eagerly here.

Rationale (found during Phase 3 testing, not a correctness bug): merely
*importing* ``lightgbm``/``optuna``/``torch`` into the same process as an
already-active numba ``parallel=True`` kernel is enough to destabilise
this codebase's OpenMP runtime on at least one development machine (a
reproducible segfault inside LightGBM's C extension, and independently
inside a numba-compiled kernel, depending on import order -- root-caused
to numba's and LightGBM's respective OpenMP thread pools coexisting in
one process, confirmed by the crash disappearing under
``OMP_NUM_THREADS=1``). ``ELMModel``/``QuantizedELMModel`` only need
scipy (a hard dependency already used throughout the engine), so they
stay eager; ``NNModel``/``LightGBMModel`` need optional, OpenMP-based
native libraries that most users of the classical/ELM force-field path
never touch, so importing ``akitsushima.fitting`` (or ``.models``) no
longer drags them in unless a caller actually asks for
``NNModel``/``LightGBMModel`` by name.
"""
from .atomic import AtomicELMModel, AtomicQELMModel
from .csv_io import load_mlip_csv, load_weights_csv
from .elm import ELMModel
from .linear import LinearModel
from .quantized_elm import QuantizedELMModel

__all__ = [
    "ELMModel",
    "QuantizedELMModel",
    "LinearModel",
    "NNModel",
    "LightGBMModel",
    "AtomicELMModel",
    "AtomicQELMModel",
    "AtomicNNModel",
    "load_mlip_csv",
    "load_weights_csv",
]

_LAZY = {
    "NNModel": ".nn",
    "LightGBMModel": ".lightgbm_model",
    "AtomicNNModel": ".atomic_nn",
}


def __getattr__(name):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value   # cache on the module so re-access is direct
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))
