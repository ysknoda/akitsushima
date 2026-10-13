"""numba parallel kernels next to torch, whichever is imported first.

torch ships its own libomp; numba's default ``omp`` threading layer aborted
the process ("Fatal Python error: Aborted") when torch's runtime had been
initialised first.  The suite only passed because alphabetical order ran
the CLI template tests before the torch-based MLIP tests.
utils/jit.py now starts numba's threading layer on import (or selects
``workqueue`` when torch is already loaded).  Each order is checked in a
fresh interpreter, since the failure is process state.
"""

import os
import subprocess
import sys

import pytest

pytest.importorskip("numba")
pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A real engine kernel (parallel pair loop, use_numba=True) and a torch
# forward / backward pass, after the imports in the given order.
_BODY = """
import numpy as np
from akitsushima import LennardJonesPotential
a = 5.26
base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
cells = np.array([[i, j, k] for i in range(3) for j in range(3) for k in range(3)])
pos = ((base[None] + cells[:, None]).reshape(-1, 3)) * a
lat = np.eye(3) * 3 * a
pot = LennardJonesPotential({frozenset({"Ar"}): {"epsilon": 0.0104, "sigma": 3.4}}, r_cut=7.0)
e, f, s = pot.compute(lat, pos, ["Ar"] * len(pos), use_numba=True)[:3]
x = torch.randn(64, 8, requires_grad=True)
torch.nn.Linear(8, 4)(x).sum().backward()
import numba
print("OK", numba.threading_layer(), float(e))
"""


@pytest.mark.parametrize("first", ["torch", "akitsushima"])
def test_numba_kernel_runs_whichever_is_imported_first(first):
    imports = (
        "import torch\nimport akitsushima\n"
        if first == "torch"
        else "import akitsushima\nimport torch\n"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("NUMBA_THREADING_LAYER", "KMP_DUPLICATE_LIB_OK")
    }
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, "-c", imports + _BODY],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, f"order {first} first:\n{r.stdout}\n{r.stderr[-2000:]}"
    assert r.stdout.strip().splitlines()[-1].startswith("OK")
