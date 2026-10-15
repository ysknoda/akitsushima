"""Wall-clock timing utility used by all solvers."""
import time


class TimingCollector:
    """
    Wall-clock timing utility for akitsushima solvers.

    Accumulates per-event timing data and produces a summary dict and
    formatted report.  All times are in **seconds** unless noted.

    Parameters
    ----------
    enabled : bool
        When False all methods are no-ops; overhead is negligible.

    Usage
    -----
    >>> from akitsushima.utils.timing import TimingCollector
    >>> tc = TimingCollector(enabled=True)
    >>> tc.start("total")
    >>> for step in range(3):
    ...     tc.start("step")
    ...     ...                         # one MD / optimiser step
    ...     tc.stop("step")
    >>> tc.stop("total")
    >>> summary = tc.summary()
    >>> tc.print_summary()

    Summary dict keys
    -----------------
    total_s          : float   -- total elapsed time [s]
    n_steps          : int     -- number of individual steps recorded
    step_mean_s      : float   -- mean step time [s]  (0 if no steps)
    step_min_s       : float   -- min  step time [s]
    step_max_s       : float   -- max  step time [s]
    step_times_s     : list[float] -- per-step times [s]
    per_step_ms      : float   -- mean step time [ms]
    throughput_steps_per_s : float
    custom           : dict[str, float] -- any extra named timers
    """

    def __init__(self, enabled: bool = True):
        self.enabled: bool = enabled
        self._starts: dict = {}
        self._total: float = 0.0
        self._steps: list = []
        self._custom: dict = {}

    # ------------------------------------------------------------------ #

    def start(self, label: str = "step") -> None:
        """Record start time for *label*."""
        if not self.enabled:
            return
        self._starts[label] = time.perf_counter()

    def stop(self, label: str = "step") -> float:
        """
        Stop *label* timer and accumulate elapsed time.

        Returns elapsed seconds for this interval, or 0.0 when disabled.
        """
        if not self.enabled:
            return 0.0
        t0 = self._starts.pop(label, None)
        if t0 is None:
            return 0.0
        elapsed = time.perf_counter() - t0
        if label == "total":
            self._total = elapsed
        elif label == "step":
            self._steps.append(elapsed)
        else:
            self._custom[label] = self._custom.get(label, 0.0) + elapsed
        return elapsed

    def lap(self, label: str = "step") -> float:
        """
        Record a lap split for *label*: stop and immediately re-start.

        Equivalent to ``stop(label); start(label)``; returns elapsed time
        for the completed lap.
        """
        elapsed = self.stop(label)
        self.start(label)
        return elapsed

    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        """Return a dict with all timing statistics."""
        if not self.enabled:
            return {"enabled": False}
        n = len(self._steps)
        total = self._total or sum(self._steps) + sum(self._custom.values())
        mean = (sum(self._steps) / n) if n > 0 else 0.0
        mn = (min(self._steps)) if n > 0 else 0.0
        mx = (max(self._steps)) if n > 0 else 0.0
        thr = (n / total) if total > 1e-12 else 0.0
        return {
            "enabled": True,
            "total_s": round(total, 6),
            "n_steps": n,
            "step_mean_s": round(mean, 6),
            "step_min_s": round(mn, 6),
            "step_max_s": round(mx, 6),
            "step_times_s": [round(t, 6) for t in self._steps],
            "per_step_ms": round(mean * 1e3, 3),
            "throughput_steps_per_s": round(thr, 3),
            "custom_s": {k: round(v, 6) for k, v in self._custom.items()},
        }

    def print_summary(self, label: str = "") -> None:
        """Print a human-readable timing table."""
        if not self.enabled:
            return
        d = self.summary()
        title = f"  Timing{(' -- ' + label) if label else ''}"
        print(f"\n{title}")
        print("  " + "=" * 46)
        tot = d["total_s"]
        h, rem = divmod(int(tot), 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            tot_str = f"{h}h {m:02d}m {s:02d}s"
        elif m > 0:
            tot_str = f"{m}m {s:02d}.{int((tot % 1) * 10):01d}s"
        else:
            tot_str = f"{tot:.3f} s"
        print(f"  {'Total time':<30s}: {tot_str}")
        n = d["n_steps"]
        if n > 0:
            print(f"  {'Steps':<30s}: {n}")
            print(f"  {'Mean / step':<30s}: {d['per_step_ms']:.2f} ms")
            print(f"  {'Min  / step':<30s}: {d['step_min_s'] * 1e3:.2f} ms")
            print(f"  {'Max  / step':<30s}: {d['step_max_s'] * 1e3:.2f} ms")
            print(f"  {'Throughput':<30s}: {d['throughput_steps_per_s']:.1f} steps/s")
        for k, v in d.get("custom_s", {}).items():
            print(f"  {k:<30s}: {v:.4f} s")
        print("  " + "=" * 46)
