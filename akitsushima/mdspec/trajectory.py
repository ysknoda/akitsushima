"""Streaming XDATCAR reader (ADR-0004).

Frames are read block by block (a few tens of MB); each block yields the
fractional coordinates of its complete frames and their lattices, so memory
stays O(N_atoms x frames per block).  Single-header (NVT / NVE) and
per-frame-header (NPT, GCMC) files are both handled.  Coordinates are
parsed by a numba kernel when numba is importable (and NUMBA_ENABLE is not
0, akitsushima's switch), else by numpy.loadtxt; both give the same floats for the fixed-format
decimals VASP and akitsushima write.
"""

from __future__ import annotations

import io
import os
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DIRECT = re.compile(rb"(?m)^[ \t]*[Dd]irect[^\n]*\n")

try:  # optional accelerator (ADR-0004 item 6)
    if os.environ.get("NUMBA_ENABLE", "1") == "0":
        raise ImportError
    import numba as _nb

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised when numba is absent
    _nb = None
    HAVE_NUMBA = False


@dataclass
class Header:
    """First header of an XDATCAR file."""

    comment: str
    lattice: np.ndarray  # (3, 3) rows = lattice vectors [Angstrom]
    species: list[str]
    counts: list[int]
    n_header_lines: int  # lines before the first 'Direct' line
    variable: bool = False  # a full header precedes every frame (NPT, GCMC)

    @property
    def n_atoms(self) -> int:
        return int(sum(self.counts))

    @property
    def symbols(self) -> list[str]:
        return [s for s, c in zip(self.species, self.counts) for _ in range(c)]


def _lattice(scale: float, vecs: np.ndarray) -> np.ndarray:
    if scale < 0:  # VASP convention: negative scale = target volume
        return vecs * (abs(scale) / abs(np.linalg.det(vecs))) ** (1.0 / 3.0)
    return vecs * scale


def _parse_header_lines(lines: list[str], species: list[str] | None):
    scale = float(lines[1].split()[0])
    vecs = np.array([[float(x) for x in ln.split()[:3]] for ln in lines[2:5]])
    tokens = lines[5].split()
    try:
        counts = [int(x) for x in tokens]
        if species is None:
            raise ValueError(
                "XDATCAR header has no element names (VASP 4 format); give trajectory.species"
            )
        n_lines = 6
    except ValueError as exc:
        if "VASP 4" in str(exc):
            raise
        species = tokens
        counts = [int(x) for x in lines[6].split()]
        n_lines = 7
    if len(species) != len(counts):
        raise ValueError(f"species {species} and counts {counts} differ in length")
    return _lattice(scale, vecs), list(species), counts, n_lines


def read_header(path: str | Path, species: list[str] | None = None) -> Header:
    """Read the first header of an XDATCAR file and detect per-frame headers."""
    with open(path) as fh:
        lines = [fh.readline() for _ in range(8)]
        lattice, sp, counts, n = _parse_header_lines(lines, species)
        tag = lines[n].lstrip().lower()
        if tag.startswith("cartesian"):
            raise ValueError("Cartesian XDATCAR frames are not supported (VASP writes Direct)")
        if not tag.startswith("direct"):
            raise ValueError(f"{path}: expected a 'Direct configuration' line after the header")
        for _ in range(sum(counts)):
            fh.readline()
        nxt = fh.readline()
    variable = bool(nxt.strip()) and not nxt.lstrip().lower().startswith("direct")
    return Header(lines[0].strip(), lattice, sp, counts, n, variable)


# --------------------------------------------------------------------------
# Coordinate parsers
# --------------------------------------------------------------------------

if HAVE_NUMBA:

    @_nb.njit(cache=True)
    def _pow10(k):
        r = 1.0
        for _ in range(k):
            r *= 10.0
        return r

    @_nb.njit(cache=True)
    def _parse_range(buf, s, e, out):
        """Parse decimal numbers in buf[s:e] into out; return how many were read.

        The mantissa is accumulated as an integer and divided once by an exact
        power of ten, so values with <= 15 significant digits are correctly
        rounded (identical to strtod / numpy.loadtxt).
        """
        i = s
        k = 0
        n_out = out.size
        while i < e:
            c = buf[i]
            if c == 32 or c == 9 or c == 10 or c == 13:
                i += 1
                continue
            neg = False
            if c == 45:
                neg = True
                i += 1
            elif c == 43:
                i += 1
            mant = 0
            nd = 0
            frac = 0
            while i < e and 48 <= buf[i] <= 57:
                mant = mant * 10 + (buf[i] - 48)
                nd += 1
                i += 1
            if i < e and buf[i] == 46:
                i += 1
                while i < e and 48 <= buf[i] <= 57:
                    mant = mant * 10 + (buf[i] - 48)
                    nd += 1
                    frac += 1
                    i += 1
            ex = 0
            if i < e and (buf[i] == 69 or buf[i] == 101):
                i += 1
                es = 1
                if buf[i] == 45:
                    es = -1
                    i += 1
                elif buf[i] == 43:
                    i += 1
                while i < e and 48 <= buf[i] <= 57:
                    ex = ex * 10 + (buf[i] - 48)
                    i += 1
                ex *= es
            if nd == 0 or nd > 18:
                return -1  # not a plain number: let the caller fall back
            p = ex - frac
            v = float(mant)
            if p < 0:
                v = v / _pow10(-p)
            elif p > 0:
                v = v * _pow10(p)
            if k >= n_out:
                return -1
            out[k] = -v if neg else v
            k += 1
        return k

    @_nb.njit(parallel=True, cache=True)
    def _parse_frames_numba(buf, starts, ends, out):
        counts = np.empty(starts.size, dtype=np.int64)
        for f in _nb.prange(starts.size):
            counts[f] = _parse_range(buf, starts[f], ends[f], out[f])
        return counts


def _parse_numpy(raw: bytes, starts, ends, n_values: int) -> np.ndarray:
    text = b"".join(raw[s:e] for s, e in zip(starts, ends)).decode()
    vals = np.loadtxt(io.StringIO(text), dtype=np.float64, usecols=(0, 1, 2), ndmin=2)
    if vals.size != len(starts) * n_values:
        raise ValueError("malformed coordinate lines in XDATCAR")
    return vals.reshape(len(starts), n_values)


def parse_frames(
    raw: bytes, starts, ends, n_atoms: int, use_numba: bool | None = None
) -> np.ndarray:
    """Parse coordinate byte ranges into an array (n_frames, n_atoms, 3)."""
    use_numba = HAVE_NUMBA if use_numba is None else (use_numba and HAVE_NUMBA)
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    if use_numba and len(starts):
        buf = np.frombuffer(raw, dtype=np.uint8)
        out = np.empty((len(starts), 3 * n_atoms))
        counts = _parse_frames_numba(buf, starts, ends, out)
        if np.all(counts == 3 * n_atoms):
            return out.reshape(len(starts), n_atoms, 3)
    if not len(starts):
        return np.empty((0, n_atoms, 3))
    return _parse_numpy(raw, starts, ends, 3 * n_atoms).reshape(len(starts), n_atoms, 3)


# --------------------------------------------------------------------------
# Streaming iterator
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    frac: np.ndarray  # (k, n_atoms, 3) wrapped fractional coordinates
    lattice: np.ndarray  # (k, 3, 3)


def iter_chunks(
    path: str | Path,
    header: Header,
    chunk_bytes: int = 64 * 2**20,
    max_frames: int | None = None,
    use_numba: bool | None = None,
) -> Iterator[Chunk]:
    """Yield blocks of complete frames from one XDATCAR file."""
    n_atoms, hl = header.n_atoms, header.n_header_lines
    n_read = 0
    carry = b""
    with open(path, "rb") as fh:
        while True:
            data = fh.read(chunk_bytes)
            eof = not data
            raw = carry + data
            if not eof:  # keep the trailing partial line for the next block
                cut = raw.rfind(b"\n") + 1
                raw, rest = raw[:cut], raw[cut:]
            else:
                rest = b""
                if raw and not raw.endswith(b"\n"):  # file without a final newline
                    raw += b"\n"
            if not raw:
                if eof:
                    break
                carry = rest
                continue
            nl = np.flatnonzero(np.frombuffer(raw, dtype=np.uint8) == 10)
            starts, ends, lattices = [], [], []
            tail = None  # byte offset of the first incomplete frame
            for m in _DIRECT.finditer(raw):
                p, e = m.start(), m.end()
                li = int(np.searchsorted(nl, p)) - 1  # newline before the 'Direct' line
                fbeg = p if not header.variable else (0 if li - hl < 0 else int(nl[li - hl]) + 1)
                idx = int(np.searchsorted(nl, e))
                if idx + n_atoms - 1 >= nl.size:
                    tail = fbeg
                    break
                if header.variable:
                    hlines = raw[fbeg:p].decode().splitlines()
                    if len(hlines) < hl:  # header began in an earlier, already used block
                        tail = fbeg
                        break
                    lat, _, cnt, _ = _parse_header_lines(hlines[-hl:] + [""], header.species)
                    if list(cnt) != list(header.counts):
                        raise ValueError(
                            f"{path}: atom counts change between frames; not supported"
                        )
                else:
                    lat = header.lattice
                starts.append(e)
                ends.append(int(nl[idx + n_atoms - 1]) + 1)
                lattices.append(lat)
            if max_frames is not None:
                k = max(0, max_frames - n_read)
                starts, ends, lattices = starts[:k], ends[:k], lattices[:k]
            if starts:
                n_read += len(starts)
                yield Chunk(parse_frames(raw, starts, ends, n_atoms, use_numba), np.array(lattices))
            if max_frames is not None and n_read >= max_frames:
                return
            if tail is None:  # no incomplete frame: resume after the last complete one
                tail = ends[-1] if ends else 0
            if eof:
                if _DIRECT.search(raw[tail:]):
                    warnings.warn(f"{path}: last frame is incomplete and was dropped", stacklevel=2)
                break
            carry = raw[tail:] + rest


def count_frames(path: str | Path, header: Header, chunk_bytes: int = 64 * 2**20) -> int:
    """Number of complete frames in an XDATCAR file without parsing coordinates.

    Counts the 'Direct configuration' lines block by block; the last frame
    counts only if all its n_atoms coordinate lines are present, as in
    iter_chunks.
    """
    n, after, carry = 0, 0, b""
    with open(path, "rb") as fh:
        while True:
            data = fh.read(chunk_bytes)
            raw = carry + data
            if data:
                cut = raw.rfind(b"\n") + 1
                raw, carry = raw[:cut], raw[cut:]
            elif raw and not raw.endswith(b"\n"):
                raw += b"\n"
            last = None
            for m in _DIRECT.finditer(raw):
                n += 1
                last = m
            after = raw.count(b"\n", last.end()) if last else after + raw.count(b"\n")
            if not data:
                break
    if n and after < header.n_atoms:
        n -= 1  # incomplete last frame
    return n


def read_frames(path: str | Path, header: Header, max_frames: int, **kw) -> Chunk:
    """Read the first max_frames frames into memory (used for site mapping)."""
    parts = list(iter_chunks(path, header, max_frames=max_frames, **kw))
    if not parts:
        raise ValueError(f"{path}: no complete frames")
    return Chunk(
        np.concatenate([c.frac for c in parts]), np.concatenate([c.lattice for c in parts])
    )
