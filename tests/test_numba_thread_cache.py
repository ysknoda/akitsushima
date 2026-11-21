"""Cached parallel kernels must not freeze the thread count.

The pair, SW (2- and 3-body) and Tersoff (3-body) kernels accumulate in
per-thread buffers indexed by ``numba.get_thread_id()``.  They used to
size those buffers with ``numba.config.NUMBA_NUM_THREADS`` read inside the
jitted function; with ``cache=True`` that value was frozen when the cache
was built.  A cache built by a one-thread process and loaded by a process
with more threads then wrote past the buffer: the extra threads'
contributions were lost (wrong forces) or the heap was corrupted (abort,
segfault).  Found with ADR-0039.

Here one subprocess builds the cache with NUMBA_NUM_THREADS=1, a second
loads that cache with 4 threads; both must agree.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

numba = pytest.importorskip("numba")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

_SCRIPT = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    sys.path.insert(0, {tests!r})
    sys.path.insert(0, {bench!r})
    from bench_force import TERSOFF_SI
    from test_md_restart import SW_SI, _structure
    import akitsushima as ak

    lat, pos, types = _structure()
    lj = {{"type": "LJ", "species": ["Si"],
           "params": {{"epsilon": 0.1, "sigma": 2.1}}, "r_cut": 5.0}}
    out = {{}}
    for name, cfg in (("pair", lj), ("sw", SW_SI), ("tersoff", TERSOFF_SI)):
        fn = ak.force_fn_from_config(cfg)
        for _ in range(3):  # several calls: corruption need not show at once
            e, f, s = fn(lat, pos, types)
        out[name] = [float(e), np.asarray(f).tolist(), np.asarray(s).tolist()]
    json.dump(out, open(sys.argv[1], "w"))
    """
)

# kernels whose per-thread buffers are sized from the thread count
_KERNELS = (
    "_pair_body_numpy_par",
    "_sw_2body_numpy_par",
    "_sw_3body_numpy_par",
    "_te_3body_force_numpy_par",
)


def _run(tmp_path, threads, cache, tag):
    script = tmp_path / "run.py"
    script.write_text(
        _SCRIPT.format(tests=HERE, bench=os.path.join(REPO, "benchmarks"))
    )
    env = {
        **os.environ,
        "NUMBA_NUM_THREADS": str(threads),
        "NUMBA_CACHE_DIR": str(cache),
        "PYTHONPATH": REPO,
    }
    out = tmp_path / f"{tag}.json"
    p = subprocess.run(
        [sys.executable, str(script), str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert p.returncode == 0, (threads, p.returncode, p.stderr[-2000:])
    return json.loads(out.read_text())


@pytest.mark.slow
@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs two cores")
def test_cache_built_with_one_thread_is_safe_with_four(tmp_path):
    cache = tmp_path / "numba_cache"
    ref = _run(tmp_path, 1, cache, "one")
    built = " ".join(p.name for p in cache.rglob("*.nbi"))
    for k in _KERNELS:  # the one-thread run really built these caches
        assert k in built, (k, built)
    got = _run(tmp_path, 4, cache, "four")
    for name in ref:
        e0, f0, s0 = ref[name]
        e1, f1, s1 = got[name]
        f0 = np.array(f0)
        assert e1 == pytest.approx(e0, rel=1e-12, abs=1e-12), name
        np.testing.assert_allclose(
            f1, f0, atol=1e-10 * max(1.0, np.abs(f0).max()), err_msg=name
        )
        np.testing.assert_allclose(s1, s0, atol=1e-12, err_msg=name)
