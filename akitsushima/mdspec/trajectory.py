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
    fmt: str = "xdatcar"  # xdatcar | extxyz | lammps-dump (ADR-0052)
    species_hint: list | None = None  # trajectory.species (LAMMPS types)

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
    """Read the first header of an XDATCAR file and detect per-frame headers.
    extxyz and LAMMPS dumps (ADR-0052) are recognised and read from their
    first frames."""
    fmt = detect_format(path)
    if fmt != "xdatcar":
        h = _read_text_header(path, fmt, species)
        h.species_hint = list(species) if species else None
        return h
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
    """Yield blocks of complete frames from one XDATCAR file (or an extxyz /
    LAMMPS dump, ADR-0052: Cartesian frames converted to wrapped fractional
    coordinates)."""
    if header.fmt != "xdatcar":
        yield from _text_chunks(path, header, chunk_bytes, max_frames)
        return
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
    if header.fmt != "xdatcar":
        return _count_text_frames(path, header)
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


# --------------------------------------------------------------------------
# extxyz and LAMMPS dump (ADR-0052): akitsushima's other trajectory formats
# --------------------------------------------------------------------------

_LATTICE_RX = re.compile(r'Lattice="([^"]*)"')
_PROPS_RX = re.compile(r"Properties=(\S+)")
_DUMP_POS = (("x", "y", "z"), ("xu", "yu", "zu"), ("xs", "ys", "zs"))


def detect_format(path: str | Path) -> str:
    """'xdatcar', 'extxyz' or 'lammps-dump', from the name, else the content."""
    name = Path(path).name.lower()
    if name.endswith((".extxyz", ".xyz")):
        return "extxyz"
    if name.endswith((".lammpstrj", ".dump")):
        return "lammps-dump"
    with open(path, errors="replace") as fh:
        first, second = fh.readline(), fh.readline()
    if first.startswith("ITEM:"):
        return "lammps-dump"
    if first.strip().isdigit() and "Lattice=" in second:
        return "extxyz"
    return "xdatcar"


def _blocks(symbols: list[str], path) -> tuple[list[str], list[int]]:
    """Species and counts of contiguous blocks; refuse an interleaved order."""
    species, counts = [], []
    for s in symbols:
        if species and species[-1] == s:
            counts[-1] += 1
        else:
            if s in species:
                raise ValueError(
                    f"{path}: atoms of one species must be contiguous (as akitsushima "
                    "writes them, POSCAR order); this file interleaves them"
                )
            species.append(s)
            counts.append(1)
    return species, counts


def _extxyz_columns(props: str) -> dict:
    """Column slices of species, pos, forces and velocities in a Properties string."""
    tok = props.split(":")
    cols, k = {}, 0
    for i in range(0, len(tok) - 2, 3):
        name, n = tok[i], int(tok[i + 2])
        cols[name] = slice(k, k + n)
        k += n
    if "species" not in cols or "pos" not in cols:
        raise ValueError(f"extxyz Properties without species / pos: {props}")
    return cols


def _extxyz_frame_head(line: str, path) -> tuple[np.ndarray, dict]:
    m = _LATTICE_RX.search(line)
    p = _PROPS_RX.search(line)
    if not m or not p:
        raise ValueError(f"{path}: extxyz comment line without Lattice= / Properties=")
    lat = np.array(m.group(1).split(), dtype=float).reshape(3, 3)
    return lat, _extxyz_columns(p.group(1))


def _dump_lattice(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Lattice (rows) and origin from the three BOX BOUNDS lines (LAMMPS convention)."""
    b = np.array([[float(x) for x in ln.split()[:3]] if len(ln.split()) >= 3
                  else [float(x) for x in ln.split()[:2]] + [0.0] for ln in lines])
    xy, xz, yz = b[0, 2], b[1, 2], b[2, 2]
    xlo = b[0, 0] - min(0.0, xy, xz, xy + xz)
    xhi = b[0, 1] - max(0.0, xy, xz, xy + xz)
    ylo = b[1, 0] - min(0.0, yz)
    yhi = b[1, 1] - max(0.0, yz)
    zlo, zhi = b[2, 0], b[2, 1]
    lat = np.array([[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi - zlo]])
    return lat, np.array([xlo, ylo, zlo])


def _dump_columns(head: str, species_hint) -> dict:
    names = head.split()[2:]
    idx = {n: i for i, n in enumerate(names)}
    pos = next((p for p in _DUMP_POS if all(c in idx for c in p)), None)
    if pos is None or "id" not in idx:
        raise ValueError(f"LAMMPS dump needs id and x y z (or xu.. / xs..) columns: {head}")
    if "element" not in idx and ("type" not in idx or not species_hint):
        raise ValueError("LAMMPS dump without an element column: give trajectory.species "
                         "(the element of each atom type, in type order)")
    out = {"names": names, "id": idx["id"], "pos": [idx[c] for c in pos],
           "scaled": pos[0] == "xs", "element": idx.get("element"), "type": idx.get("type")}
    for key, cs in (("forces", ("fx", "fy", "fz")), ("velocities", ("vx", "vy", "vz"))):
        out[key] = [idx[c] for c in cs] if all(c in idx for c in cs) else None
    return out


def _iter_text_frames(path, fmt: str, n_atoms: int, frames_per_chunk: int, species_hint):
    """Blocks of frames of an extxyz / LAMMPS dump: dicts with 'cart' (or 'frac'
    for scaled dumps), 'lattice', 'symbols' of the first frame, and 'forces' /
    'velocities' when present (velocities in A/fs)."""
    from itertools import islice

    per = n_atoms + (2 if fmt == "extxyz" else 9)
    with open(path, errors="replace") as fh:
        while True:
            lines = list(islice(fh, per * frames_per_chunk))
            n_full = len(lines) // per
            if len(lines) % per and lines[n_full * per:] and any(
                ln.strip() for ln in lines[n_full * per:]
            ):
                warnings.warn(f"{path}: last frame is incomplete and was dropped", stacklevel=3)
            if n_full == 0:
                return
            lats, xyz, frc, vel, syms = [], [], [], [], None
            for f in range(n_full):
                blk = lines[f * per:(f + 1) * per]
                if fmt == "extxyz":
                    if int(blk[0]) != n_atoms:
                        raise ValueError(f"{path}: the atom count changes between frames")
                    lat, cols = _extxyz_frame_head(blk[1], path)
                    rows = [ln.split() for ln in blk[2:]]
                    if syms is None:
                        syms = [r[cols["species"].start] for r in rows]
                    arr = np.array([r[cols["species"].stop:] for r in rows], dtype=float)
                    base = cols["species"].stop
                    xyz.append(arr[:, cols["pos"].start - base:cols["pos"].stop - base])
                    if "forces" in cols:
                        frc.append(arr[:, cols["forces"].start - base:cols["forces"].stop - base])
                    if "velocities" in cols:
                        vel.append(arr[:, cols["velocities"].start - base:
                                       cols["velocities"].stop - base])
                    lats.append(lat)
                else:
                    if not blk[0].startswith("ITEM: TIMESTEP") or int(blk[3]) != n_atoms:
                        raise ValueError(f"{path}: unexpected LAMMPS dump frame layout")
                    lat, origin = _dump_lattice(blk[5:8])
                    c = _dump_columns(blk[8], species_hint)
                    rows = [ln.split() for ln in blk[9:]]
                    order = np.argsort([int(r[c["id"]]) for r in rows], kind="stable")
                    rows = [rows[i] for i in order]
                    if syms is None:
                        syms = ([r[c["element"]] for r in rows] if c["element"] is not None
                                else [species_hint[int(r[c["type"]]) - 1] for r in rows])
                    p = np.array([[r[i] for i in c["pos"]] for r in rows], dtype=float)
                    xyz.append(p if c["scaled"] else p - origin)
                    if c["forces"] is not None:
                        frc.append(np.array([[r[i] for i in c["forces"]] for r in rows], float))
                    if c["velocities"] is not None:  # metal units A/ps -> A/fs
                        vel.append(1e-3 * np.array([[r[i] for i in c["velocities"]]
                                                    for r in rows], float))
                    lats.append(lat)
                    if c["scaled"]:
                        xyz[-1] = xyz[-1] @ lat  # to Cartesian, as the other branches
            out = {"cart": np.array(xyz), "lattice": np.array(lats), "symbols": syms}
            if frc:
                out["forces"] = np.array(frc)
            if vel:
                out["velocities"] = np.array(vel)
            yield out
            if n_full < frames_per_chunk:
                return


def _read_text_header(path, fmt: str, species) -> Header:
    with open(path, errors="replace") as fh:
        first = fh.readline()
        if fmt == "extxyz":
            n = int(first)
        else:
            fh.readline()
            fh.readline()
            n = int(fh.readline())
    blocks = list(islice_frames(path, fmt, n, 2, species))
    fr = blocks[0]
    sp, counts = _blocks(fr["symbols"], path)
    lat = fr["lattice"]
    variable = len(lat) > 1 and not np.allclose(lat[0], lat[1], atol=1e-10)
    return Header(f"{fmt} trajectory", lat[0], sp, counts, 0, variable, fmt)


def islice_frames(path, fmt, n_atoms, n_frames, species):
    """The first n_frames frames as one block (header detection)."""
    for blk in _iter_text_frames(path, fmt, n_atoms, n_frames, species):
        yield blk
        return


def _text_chunks(path, header: Header, chunk_bytes: int, max_frames, want_extra: bool = False):
    per_frame_bytes = 60 * header.n_atoms + 200
    k = max(1, int(chunk_bytes // per_frame_bytes))
    if max_frames is not None:
        k = min(k, max_frames)
    n_read = 0
    for blk in _iter_text_frames(path, header.fmt, header.n_atoms, k, header.species_hint):
        if blk["symbols"] != header.symbols:
            raise ValueError(f"{path}: the atom order differs from the first frame")
        cart, lat = blk["cart"], blk["lattice"]
        if max_frames is not None:
            take = max(0, max_frames - n_read)
            cart, lat = cart[:take], lat[:take]
            for key in ("forces", "velocities"):
                if key in blk:
                    blk[key] = blk[key][:take]
        if not len(cart):
            return
        frac = np.einsum("kni,kij->knj", cart, np.linalg.inv(lat))
        frac -= np.floor(frac)
        n_read += len(cart)
        yield (Chunk(frac, lat), blk) if want_extra else Chunk(frac, lat)
        if max_frames is not None and n_read >= max_frames:
            return


def iter_forces(path: str | Path, header: Header, chunk_bytes: int = 64 * 2**20):
    """Forces [eV/A] (k, n_atoms, 3) of every frame of an extxyz / LAMMPS dump."""
    if header.fmt == "xdatcar":
        raise ValueError(f"{path}: an XDATCAR holds no forces")
    for _, blk in _text_chunks(path, header, chunk_bytes, None, want_extra=True):
        if "forces" not in blk:
            raise ValueError(f"{path}: no forces in this trajectory")
        yield blk["forces"]


def _count_text_frames(path, header: Header) -> int:
    per = header.n_atoms + (2 if header.fmt == "extxyz" else 9)
    n = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(64 * 2**20)
            if not b:
                break
            n += b.count(b"\n")
    return n // per
