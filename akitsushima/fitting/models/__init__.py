"""ML-potential model classes.

``NNModel`` and ``AtomicNNModel`` are resolved lazily (PEP 562 module
``__getattr__``) rather than imported eagerly here.

Rationale (found during Phase 3 testing, not a correctness bug): merely
*importing* an OpenMP-based native library (``torch``; formerly also
``lightgbm``, removed in ADR-0012) into the same process as an
already-active numba ``parallel=True`` kernel was enough to destabilise
this codebase's OpenMP runtime on at least one development machine
(root-caused to two OpenMP thread pools coexisting in one process,
confirmed by the crash disappearing under ``OMP_NUM_THREADS=1``).
``ELMModel``/``QuantizedELMModel``/``LinearModel`` only need scipy, so
they stay eager; the NN models need torch, which most users of the
classical/ELM path never touch, so importing ``akitsushima.fitting`` (or
``.models``) does not drag it in unless a caller asks for them by name.
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
    "AtomicELMModel",
    "AtomicQELMModel",
    "AtomicNNModel",
    "load_mlip_csv",
    "load_weights_csv",
]

_LAZY = {
    "NNModel": ".nn",
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
