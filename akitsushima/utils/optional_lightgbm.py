"""Optional LightGBM + Optuna detection, isolated in its own module.

Bug/stability note: importing lightgbm (its C extension links its own
OpenMP runtime) into the same process as an already-active numba
``parallel=True`` kernel reproducibly segfaulted during Phase 3 testing
on at least one development machine (confirmed via ``OMP_NUM_THREADS=1``
making the crash disappear -- a native OpenMP thread-pool conflict
between numba and LightGBM, not a Python-level bug). Keeping this import
in its own module -- imported only by ``fitting/models/lightgbm_model.py``,
itself lazily resolved by ``fitting/models/__init__.py`` -- means
``import akitsushima`` / ``akitsushima.engine`` / ``akitsushima.fitting``
never pulls lightgbm in unless a caller actually asks for
``LightGBMModel`` by name.
"""
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    HAS_OPTUNA = False

__all__ = ["HAS_LIGHTGBM", "lgb", "HAS_OPTUNA", "optuna"]
