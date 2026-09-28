from .driver import (
    ScoreCard,
    compute_scorecard,
    evaluate_diagnostics_generic,
    export_lammps_buckingham,
    fit_potential,
    parity_plot_all,
    print_fit_result,
    print_scorecard_comparison,
    save_fit_result_generic,
    split_train_val,
)
from .external import ExternalFileEngine, ScaledForceEngine
from .force_fn import compute_all, make_force_fn
from .io_utils import (
    _load_potential_json,
    apply_scales_to_structures,
    load_fit_results,
    load_potential,
    load_scale_csv,
    make_scale_csv,
    save_potential,
)
from .loss import generic_loss_function
from .models import (
    AtomicELMModel,
    AtomicQELMModel,
    ELMModel,
    LinearModel,
    QuantizedELMModel,
    load_mlip_csv,
    load_weights_csv,
)
from .parameter_sets import (
    BornMayerParameterSet,
    BuckinghamParameterSet,
    FittableParameterSet,
    LJParameterSet,
    MieParameterSet,
    MorseParameterSet,
    StillingerWeberParameterSet,
    MultiSpeciesStillingerWeberParameterSet,
    TersoffParameterSet,
    YukawaParameterSet,
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
    "generic_loss_function",
    "split_train_val",
    "ScoreCard",
    "compute_scorecard",
    "print_scorecard_comparison",
    "parity_plot_all",
    "fit_potential",
    "print_fit_result",
    "evaluate_diagnostics_generic",
    "save_fit_result_generic",
    "export_lammps_buckingham",
    "save_potential",
    "load_potential",
    "load_fit_results",
    "_load_potential_json",
    "make_scale_csv",
    "load_scale_csv",
    "apply_scales_to_structures",
    "make_force_fn",
    "compute_all",
    "ExternalFileEngine",
    "ScaledForceEngine",
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

# NNModel / AtomicNNModel are resolved lazily -- see fitting/models/__init__.py
# for why (avoids an eager torch import, and the OpenMP runtime conflict
# with numba that motivated it, for every caller who never touches them).
_LAZY = {"NNModel": "models", "AtomicNNModel": "models"}


def __getattr__(name):
    from . import models as _models

    attr_module = _LAZY.get(name)
    if attr_module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_models, name)
    globals()[name] = value
    return value
