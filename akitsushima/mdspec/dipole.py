"""Total cell dipole M(t) from a trajectory (ADR-0002, ADR-0004).

Positions are unwrapped frame to frame by the minimum-image rule in
fractional coordinates, continuing across block (and, for concatenated
runs, file) boundaries, and reduced block by block, so memory stays
O(N_atoms), to the displacement form (ADR-0011 A1)

    M(t) = sum_k Z_k . u_k(t),   u_k(t) = (f_k(t) - fbar_k) h(t),

with fbar_k the time-averaged unwrapped fractional position of atom k over
the run and h(t) the lattice (rows = lattice vectors).  The absolute form
sum_k Z_k . f_k h(t) differs by sum_k Z_k . fbar_k h(t), which is constant
for a fixed cell but follows the cell fluctuations of an NPT run and
depends on the periodic image chosen for every atom.  The same holds for
the Gamma normal coordinates Q_m.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import charges as ch
from . import forces as fr
from . import modes as md
from .constants import DEBYE_TO_E_ANG
from .trajectory import Header, count_frames, iter_chunks, read_frames, read_header

CACHE_FORMAT = 2  # 2: displacement-form M and Q, per-file frame counts (0.2.0a2)
VARIABLE_CELL_TOL = 1e-6  # Angstrom: largest change of a lattice component that counts as fixed


@dataclass
class Run:
    """One analysed trajectory: dipoles and volumes of the kept frames."""

    dipole: np.ndarray  # (n, 3) e*Angstrom
    volume: np.ndarray  # (n,) Angstrom^3
    dt_fs: float  # time between kept frames
    files: list[str]
    info: dict = field(default_factory=dict)
    modes: np.ndarray | None = None  # (n, n_ir) Gamma normal coordinates, sqrt(amu) A
    forces: np.ndarray | None = None  # (n_f, n_ir) generalised forces, eV / (A sqrt(amu))
    force_idx: np.ndarray | None = None  # (n_f,) frame index of each force row


class Unwrapper:
    """Carry-over state for frame-to-frame unwrapping in fractional coordinates."""

    def __init__(self, jump_warn: float = 0.25):
        self.prev = None  # last wrapped frame
        self.acc = None  # last unwrapped frame
        self.jump_warn = jump_warn
        self.max_jump = 0.0

    def __call__(self, frac: np.ndarray) -> np.ndarray:
        if self.prev is None:
            first = frac[:1]
            d = np.diff(frac, axis=0)
            base = first
        else:
            d = np.diff(np.concatenate([self.prev[None], frac]), axis=0)
            base = self.acc[None]
        d -= np.round(d)
        if d.size:
            self.max_jump = max(self.max_jump, float(np.abs(d).max()))
        uw = np.concatenate([base, base + np.cumsum(d, axis=0)])
        if self.prev is not None:
            uw = uw[1:]
        self.prev, self.acc = frac[-1].copy(), uw[-1].copy()
        return uw


class RunStream:
    """Displacement-form accumulation over the frames of one run (ADR-0011 A1).

    Frames are taken relative to the first unwrapped frame f_ref, which needs
    no second pass; fbar - f_ref is accumulated on the way, and the exactly
    linear remainder (terms in (fbar - f_ref) h(t)) is removed by
    remove_reference() once the run is complete.
    """

    def __init__(self, unwrapper: Unwrapper | None = None):
        self.unwrapper = unwrapper or Unwrapper()
        self.ref = None
        self.sum = None
        self.n = 0
        self.lattices = []

    def __call__(self, frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
        """Unwrapped fractional coordinates relative to the first frame of the run."""
        uw = self.unwrapper(frac)
        if self.ref is None:
            self.ref = uw[0].copy()
            self.sum = np.zeros_like(self.ref)
        rel = uw - self.ref[None]
        self.sum += rel.sum(axis=0)
        self.n += len(rel)
        self.lattices.append(np.asarray(lattice))
        return rel

    @property
    def offset(self) -> np.ndarray:
        """fbar - f_ref per atom, (n_atoms, 3) fractional."""
        return self.sum / self.n

    @property
    def lattice(self) -> np.ndarray:
        """Lattice of every frame of the run so far, (n, 3, 3)."""
        return np.concatenate(self.lattices)

    @property
    def variable_cell(self) -> bool:
        lat = self.lattice
        return bool(np.abs(lat - lat[:1]).max() > VARIABLE_CELL_TOL) if len(lat) else False


def remove_reference(x: np.ndarray, h: np.ndarray, d: np.ndarray, weights: np.ndarray, kind: str):
    """Subtract the term of a linear quantity computed from f - f_ref that belongs to fbar - f_ref.

    kind 'dipole': x (t, 3) = sum_k Z_k . [(f_k - f_ref,k) h(t)], weights = Z (n, 3, 3).
    kind 'modes':  x (t, n_ir) = sum_k [(f_k - f_ref,k) h(t)] . W_k, weights = W (n, 3, n_ir).
    kind 'mass':   x (t, 3) = sum_k m_k (f_k - f_ref,k) h(t), weights = m (n,).
    d = fbar - f_ref (n, 3), h (t, 3, 3).
    """
    if kind == "dipole":
        a = np.einsum("kac,kb->abc", weights, d)
        return x - np.einsum("abc,tbc->ta", a, h)
    if kind == "modes":
        b = np.einsum("kb,kcm->bcm", d, weights)
        return x - np.einsum("bcm,tbc->tm", b, h)
    if kind == "mass":
        return x - np.einsum("b,tbc->tc", weights @ d, h)
    raise ValueError(kind)


def dipole_from_chunk(uw: np.ndarray, lattice: np.ndarray, z: np.ndarray, scalar_q) -> np.ndarray:
    """M (k, 3) for unwrapped fractional uw (k, n, 3) and lattices (k, 3, 3)."""
    if scalar_q is not None:  # point charges: sum q f, then to Cartesian
        mf = np.einsum("n,knb->kb", scalar_q, uw)
        return np.einsum("kb,kbc->kc", mf, lattice)
    r = np.einsum("knb,kbc->knc", uw, lattice)
    return np.einsum("nac,knc->ka", z, r)


def _scalar(z: np.ndarray):
    iso = np.allclose(z, z[:, :1, :1] * np.eye(3)[None], atol=0)
    return z[:, 0, 0].copy() if iso else None


def _cache_key(cfg: dict, z: np.ndarray, w: np.ndarray | None = None) -> str:
    h = hashlib.sha256()
    forces = cfg.get("forces") or {"files": []}
    for f in cfg["trajectory"]["files"] + forces["files"]:
        st = Path(f).stat()
        h.update(f"{f}:{st.st_size}:{st.st_mtime_ns}".encode())
    # skip_frames and stride are applied after the cache (ADR-0011 A7)
    tr = {k: v for k, v in cfg["trajectory"].items() if k not in ("skip_frames", "stride")}
    h.update(f"format {CACHE_FORMAT}".encode())
    h.update(json.dumps(tr, sort_keys=True, default=str).encode())
    h.update(json.dumps(cfg.get("forces"), sort_keys=True, default=str).encode())
    h.update(np.ascontiguousarray(z).tobytes())
    if w is not None:
        h.update(np.ascontiguousarray(w).tobytes())
    return h.hexdigest()[:16]


def build_charges(cfg: dict, header: Header) -> np.ndarray:
    """Per-atom Z tensors (n, 3, 3) for the 'point_charge' or 'born_tensor' model."""
    dp = cfg["dipole"]
    if dp["model"] == "point_charge":
        return ch.point_charges(header.symbols, dp["charges"], dp["neutrality_tol"])
    b = dp["born"]
    frac_mean, lattice = mean_positions(cfg, header, int(b["map_frames"]))
    return ch.born_charges(b, header.symbols, frac_mean, lattice)


def mean_positions(cfg: dict, header: Header, n_frames: int = 200):
    """Unwrapped mean fractional positions of the first frames and the first lattice."""
    chunk = read_frames(cfg["trajectory"]["files"][0], header, max_frames=n_frames)
    return Unwrapper()(chunk.frac).mean(axis=0), chunk.lattice[0]


def build_modes(cfg: dict, header: Header, z: np.ndarray, log=print) -> dict | None:
    """Gamma-mode projection for the 'mode' estimator, or None (ADR-0003 item 10)."""
    mcfg = cfg.get("modes")
    if not mcfg:
        return None
    if mcfg.get("implicit") and not md.has_outcar_modes(mcfg["outcar"]):
        log(f"[modes] {mcfg['outcar']} has no eigenvectors: mode estimator unavailable")
        return None
    masses, _, _ = ch.run_masses(cfg, header)
    frac_mean, lattice = mean_positions(cfg, header)
    proj = md.build_projection(mcfg, header.symbols, frac_mean, lattice, z, masses)
    proj["WG"] = fr.weights(proj["W"], masses)  # force projection (ADR-0009)
    src = mcfg["outcar"] or mcfg.get("phonopy") or mcfg["file"]
    log(
        f"[modes] {len(proj['freq_THz'])} IR-active Gamma modes from {src} "
        f"(per component {proj['n_ir_per_component']})"
    )
    return proj


def _stream_file(path, header, z, scalar_q, stream: RunStream, chunk_bytes, w=None):
    """M, volume and Q of one file, relative to the first frame of the run (see RunStream)."""
    ms, vs, qs = [], [], []
    for chunk in iter_chunks(path, header, chunk_bytes=chunk_bytes):
        uw = stream(chunk.frac, chunk.lattice)
        ms.append(dipole_from_chunk(uw, chunk.lattice, z, scalar_q))
        vs.append(np.abs(np.linalg.det(chunk.lattice)))
        if w is not None:  # Gamma normal coordinates from Cartesian positions
            qs.append(np.einsum("knc,ncm->km", np.einsum("knb,kbc->knc", uw, chunk.lattice), w))
    if not ms:
        raise ValueError(f"{path}: no complete frames")
    return np.concatenate(ms), np.concatenate(vs), (np.concatenate(qs) if w is not None else None)


def _file_forces(cfg: dict, i: int, path, header: Header, proj: dict, n_frames: int, log=print):
    """Generalised forces of trajectory file i and the frame index of each row (ADR-0009)."""
    fcfg = cfg["forces"]
    src = fcfg["files"][i]
    if src.endswith(".npy"):
        k = fcfg["stride_frames"]
        g = fr.project_npy(src, header.n_atoms, proj["WG"])
        expect = (n_frames + k - 1) // k
    else:
        every = int(cfg["trajectory"]["frame_every"])
        g, first = fr.project_outcar(src, header.n_atoms, proj["WG"], every)
        chunk = read_frames(path, header, max_frames=1)
        fr.check_alignment(src, first, chunk.frac[0] @ chunk.lattice[0], chunk.lattice[0])
        k, expect = 1, n_frames
    if len(g) != expect:
        raise ValueError(
            f"{src}: {len(g)} force frames for {n_frames} trajectory frames of {path} "
            f"(expected {expect}; same run? frame_every / stride_frames?)"
        )
    log(f"[forces] {src}: {len(g)} frames")
    return g, np.arange(len(g)) * k


def _external_runs(cfg: dict, header: Header, files: list[str], dt: float):
    """The 'external' dipole model: one run, checked against the trajectory frame count."""
    dpc = cfg["dipole"]
    data = np.loadtxt(dpc["external_file"], comments=("#", "!"), ndmin=2)
    if data.shape[1] < 3:
        raise ValueError(f"{dpc['external_file']}: needs three dipole columns (Mx My Mz)")
    m = data[:, -3:]
    if dpc["external_unit"] == "Debye":
        m = m * DEBYE_TO_E_ANG
    counts = [count_frames(f, read_header(f, cfg["trajectory"]["species"])) for f in files]
    if len(m) != sum(counts):
        raise ValueError(
            f"{dpc['external_file']}: {len(m)} dipole rows for {sum(counts)} trajectory frames "
            f"({' + '.join(map(str, counts))}); one row per XDATCAR frame is needed (ADR-0011 B9)"
        )
    vol = np.full(len(m), abs(np.linalg.det(header.lattice)))
    info = {"source": "external", "file_frames": counts, "variable_cell": False}
    return [Run(m, vol, dt, list(files), info)]


def _stream_run(cfg, idx, header, z, scalar_q, proj, chunk_bytes, log):
    """Displacement-form M, V, Q (+ forces) of one run made of the files `idx` (ADR-0011 A1)."""
    files = [cfg["trajectory"]["files"][i] for i in idx]
    w = proj["W"] if proj else None
    stream = RunStream()
    ms, vs, qs, counts = [], [], [], []
    for f in files:
        m, v, q = _stream_file(f, header, z, scalar_q, stream, chunk_bytes, w)
        log(f"[dipole] {f}: {len(m)} frames")
        ms.append(m)
        vs.append(v)
        qs.append(q)
        counts.append(len(m))
        uw = stream.unwrapper
        if uw.max_jump > uw.jump_warn:
            warnings.warn(
                f"largest frame-to-frame displacement {uw.max_jump:.2f} of a lattice "
                "vector: frames may be too sparse for unwrapping",
                stacklevel=3,
            )
    h, d = stream.lattice, stream.offset
    m = remove_reference(np.concatenate(ms), h, d, z, "dipole")
    q = remove_reference(np.concatenate(qs), h, d, w, "modes") if w is not None else None
    variable = stream.variable_cell
    if variable:
        if cfg.get("forces"):
            raise ValueError(
                "forces with a variable cell (NPT) are not supported: the effective harmonic "
                "fit under a fluctuating strain is not validated (ADR-0011)"
            )
        warnings.warn(
            "variable cell (NPT): M(t) and Q(t) use the displacement form; NPT support is "
            "experimental and the constant-stress permittivity is not validated (ADR-0011)",
            stacklevel=3,
        )
    g = gi = None
    if cfg.get("forces"):
        gs, gis, off = [], [], 0
        for i, f, n in zip(idx, files, counts):
            gf, gif = _file_forces(cfg, i, f, header, proj, n, log)
            gs.append(gf)
            gis.append(gif + off)
            off += n
        g, gi = np.concatenate(gs), np.concatenate(gis)
    return (m, np.concatenate(vs), q, g, gi), counts, variable


def compute_runs(cfg: dict, log=print) -> tuple[list[Run], np.ndarray, Header]:
    """Read the trajectory files and return the analysed runs, Z tensors and header.

    Run.info holds 'file_frames' (frames read from each file of the run, before
    skip_frames / stride) and 'variable_cell'.
    """
    tr, perf = cfg["trajectory"], cfg["performance"]
    files = tr["files"]
    header = read_header(files[0], tr["species"])
    for f in files[1:]:
        h = read_header(f, tr["species"])
        if h.species != header.species or h.counts != header.counts:
            raise ValueError(f"{f}: species/counts differ from {files[0]}")
    dt = float(tr["timestep_fs"]) * int(tr["frame_every"])
    chunk_bytes = int(float(perf["chunk_mb"]) * 2**20)
    groups = (
        [list(range(len(files)))]
        if tr["mode"] == "concatenate"
        else [[i] for i in range(len(files))]
    )

    if cfg["dipole"]["model"] == "external":
        runs = _external_runs(cfg, header, files, dt)
        z = np.zeros((header.n_atoms, 3, 3))
    else:
        z = build_charges(cfg, header)
        scalar_q = _scalar(z)
        proj = build_modes(cfg, header, z, log)
        if cfg.get("forces") and proj is None:
            raise ValueError("forces need Gamma eigenvectors, but no modes are available")
        key = _cache_key(cfg, z, proj["W"] if proj else None)
        out_dir = Path(cfg["output"]["dir"])
        cache = out_dir / f"{cfg['output']['prefix']}_dipole_cache.npz"
        raw_runs = None
        if cfg["output"]["dipole_cache"] and cache.exists():
            d = np.load(cache, allow_pickle=False)
            if str(d["key"]) == key:
                log(f"[dipole] cache hit: {cache}")
                raw_runs = [
                    (
                        tuple(d.get(f"{x}{i}") for x in ("m", "v", "q", "g", "gi")),
                        d[f"nf{i}"].tolist(),
                        bool(d[f"var{i}"]),
                    )
                    for i in range(int(d["n"]))
                ]
        if raw_runs is None:
            raw_runs = [
                _stream_run(cfg, grp, header, z, scalar_q, proj, chunk_bytes, log) for grp in groups
            ]
            if cfg["output"]["dipole_cache"]:
                out_dir.mkdir(parents=True, exist_ok=True)
                arrs = {}
                for i, (arr, counts, var) in enumerate(raw_runs):
                    for x, a in zip(("m", "v", "q", "g", "gi"), arr):
                        if a is not None:
                            arrs[f"{x}{i}"] = a
                    arrs[f"nf{i}"] = np.asarray(counts)
                    arrs[f"var{i}"] = np.asarray(var)
                np.savez(cache, key=key, n=len(raw_runs), **arrs)
        base = {"modes": proj} if proj else {}
        runs = [
            Run(
                m,
                v,
                dt,
                [files[i] for i in grp],
                {**base, "file_frames": counts, "variable_cell": var},
                q,
                g,
                gi,
            )
            for ((m, v, q, g, gi), counts, var), grp in zip(raw_runs, groups)
        ]

    skip, stride = int(tr["skip_frames"]), int(tr["stride"])
    out = []
    for r in runs:
        if skip >= len(r.dipole):
            raise ValueError(f"skip_frames = {skip} leaves no frames ({len(r.dipole)} read)")
        if stride > 1:
            warnings.warn("stride > 1 subsamples M(t) without filtering (aliasing)", stacklevel=2)
        g, gi = r.forces, r.force_idx
        if g is not None:  # stride is 1 with forces (config)
            keep = gi >= skip
            g, gi = g[keep], gi[keep] - skip
        out.append(
            Run(
                r.dipole[skip::stride],
                r.volume[skip::stride],
                dt * stride,
                r.files,
                r.info,
                None if r.modes is None else r.modes[skip::stride],
                g,
                gi,
            )
        )
    return out, z, header
