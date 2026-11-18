"""Checkpoint files for ``md`` and ``relax`` (ADR-0031).

A checkpoint is one ``.npz``: the arrays of the state, plus a JSON
``meta`` entry with the scalars, the settings that a restart must
match, and the byte length of every output file at the time of writing.
It is written by MPI rank 0 through a temporary file and
``os.replace``, so an interrupted write leaves the previous checkpoint
intact.  ``nemd`` and ``anneal`` use the same pattern.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable

import numpy as np

from ..utils.mpi import _mpi_rank

__all__ = [
    "check_settings",
    "load_checkpoint",
    "output_sizes",
    "potential_hash",
    "save_checkpoint",
    "truncate_outputs",
]


def save_checkpoint(path: str, meta: dict, arrays: dict[str, np.ndarray]) -> None:
    """Write ``meta`` (JSON-serialisable) and ``arrays`` to ``path``
    atomically; a no-op on MPI ranks other than 0."""
    if _mpi_rank != 0:
        return
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> tuple[dict, dict[str, np.ndarray]]:
    """(meta, arrays) of a checkpoint written by :func:`save_checkpoint`."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k].copy() for k in z.files if k != "meta"}
    return meta, arrays


def _same(a, b) -> bool:
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return list(a) == list(b)
    if (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    ):
        return bool(np.isclose(float(a), float(b), rtol=1e-12, atol=0.0))
    return a == b


def check_settings(saved: dict, now: dict, who: str, path: str) -> None:
    """Refuse a restart whose settings differ from the checkpoint's.

    The two dicts must have the same keys and equal values (floats to
    relative 1e-12); an item present on one side only is a difference."""
    diff = []
    for k in sorted(set(saved) | set(now)):
        a = saved.get(k, "<absent>")
        b = now.get(k, "<absent>")
        if k not in saved or k not in now or not _same(a, b):
            diff.append(f"{k}: checkpoint {a!r}, now {b!r}")
    if diff:
        raise ValueError(
            f"{who}: cannot continue from {path}; these settings differ from "
            "the run that wrote it:\n  "
            + "\n  ".join(diff)
            + "\nRestore them, or start afresh (restart: false in the CLI, or "
            "delete the checkpoint file)."
        )


def output_sizes(handles: dict[str, object]) -> dict[str, int]:
    """Byte length of each open output file, after flushing it."""
    out = {}
    for name, fh in handles.items():
        fh.flush()
        out[name] = int(os.fstat(fh.fileno()).st_size)
    return out


def truncate_outputs(paths: dict[str, str], sizes: dict[str, int]) -> None:
    """Cut each output file back to its length at the checkpoint, so the
    lines written after it (and lost with the state) are not repeated.
    Rank 0 only; a missing file is an error, since the run would then
    continue a file that no longer holds its beginning."""
    if _mpi_rank != 0:
        return
    for name, path in paths.items():
        if name not in sizes:
            continue
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"restart: {path} is missing but the checkpoint expects "
                f"{sizes[name]} bytes of it."
            )
        if os.path.getsize(path) < sizes[name]:
            raise ValueError(
                f"restart: {path} is shorter ({os.path.getsize(path)} bytes) "
                f"than at the checkpoint ({sizes[name]} bytes)."
            )
        with open(path, "r+b") as fh:
            fh.truncate(sizes[name])


def potential_hash(section: dict | None, files: Iterable[str] = ()) -> str:
    """SHA-256 of a potential's config section (canonical JSON) and of the
    bytes of the files it reads; what the CLI compares on a restart."""
    h = hashlib.sha256()
    h.update(json.dumps(section or {}, sort_keys=True, default=str).encode())
    for p in sorted(set(files)):
        if p and os.path.isfile(p):
            h.update(os.path.basename(p).encode())
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()
