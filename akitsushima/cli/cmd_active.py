"""``akitsushima active`` -- active learning driven by the ensemble
uncertainty (ADR-0037): explore -> select -> label -> train, round after
round, resumable from ``active_ledger.jsonl``.  See
:mod:`akitsushima.active` for the stages and ``--dump-default active`` for
the input.
"""

import time as _time

from ..active import ActiveConfigError, active
from .common import _check_yaml, _fmt_time, _print_header, _print_results_header

__all__ = ["cmd_active"]


def cmd_active(args) -> int:
    """Active learning from YAML input.  Exit code 0 when the loop ended
    (converged or all rounds done), 2 while VASP labels are awaited, 1 on
    an error."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1
    _print_header("Active learning (active)", cfg)
    t0 = _time.perf_counter()
    try:
        res = active(args.yaml)
    except (ActiveConfigError, FileNotFoundError) as err:
        print(f"[ERROR] {err}")
        return 1
    _print_results_header("active")
    print(f"\n  Status      : {res['status']}")
    print(f"  Rounds done : {len(res['rounds'])}")
    for row in res["rounds"]:
        s = row["summary"]
        print(f"    round {row['round']}: {s['n_added']} added, "
              f"test RMSE {s.get('rmse_test', '-')}")
    print(f"  Model       : {res['model']}")
    print(f"  Ledger      : {res['output_dir']}/active_ledger.jsonl")
    print(f"  Wall time   : {_fmt_time(_time.perf_counter() - t0)}")
    return 2 if res["status"] == "waiting_for_labels" else 0
