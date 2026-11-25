"""XDATCAR streaming reader, parsers and unwrapping (ADR-0004)."""

import numpy as np
import pytest

from akitsushima.mdspec import trajectory as tj
from akitsushima.mdspec.dipole import Unwrapper

from .conftest import write_xdatcar


def _walk(n_frames=40, n_atoms=7, seed=1, step=0.06):
    rng = np.random.default_rng(seed)
    uw = np.cumsum(rng.normal(scale=step, size=(n_frames, n_atoms, 3)), axis=0) + rng.random(
        (1, n_atoms, 3)
    )
    uw = np.round(uw, 8)
    return uw, np.round(uw % 1.0, 8)


def _read_all(path, chunk_bytes, use_numba):
    h = tj.read_header(path)
    parts = list(tj.iter_chunks(path, h, chunk_bytes=chunk_bytes, use_numba=use_numba))
    return h, np.concatenate([c.frac for c in parts]), np.concatenate([c.lattice for c in parts])


@pytest.mark.parametrize("use_numba", [False, True])
@pytest.mark.parametrize("chunk_bytes", [97, 1000, 1 << 20])
def test_fixed_header_roundtrip(tmp_path, use_numba, chunk_bytes):
    if use_numba and not tj.HAVE_NUMBA:
        pytest.skip("numba not installed")
    _, wrapped = _walk()
    lat = np.diag([5.0, 6.0, 7.0])
    p = tmp_path / "XDATCAR"
    write_xdatcar(p, ["Si", "C"], [3, 4], wrapped, lat)
    h, frac, lats = _read_all(p, chunk_bytes, use_numba)
    assert not h.variable and h.symbols == ["Si"] * 3 + ["C"] * 4
    # the parser must give exactly the floats of the 8-decimal text
    expect = np.array([[float(f"{x:.8f}") for x in row] for row in wrapped.reshape(-1, 3)]).reshape(
        wrapped.shape
    )
    assert np.array_equal(frac, expect)
    assert np.allclose(lats, lat)


@pytest.mark.parametrize("chunk_bytes", [150, 5000])
def test_variable_header_lattices(tmp_path, chunk_bytes):
    _, wrapped = _walk(n_frames=12)
    lats = np.array([np.diag([5 + 0.01 * k, 6.0, 7.0 - 0.02 * k]) for k in range(12)])
    p = tmp_path / "XDATCAR"
    write_xdatcar(p, ["Zn", "O"], [3, 4], wrapped, lats, variable=True)
    h, frac, got = _read_all(p, chunk_bytes, None)
    assert h.variable
    assert np.allclose(got, lats, atol=1e-6)
    assert np.allclose(frac, wrapped, atol=1e-12)


def test_unwrap_across_chunks_and_files(tmp_path):
    uw, wrapped = _walk(n_frames=60, step=0.08)
    lat = np.eye(3) * 5.0
    p1, p2 = tmp_path / "a", tmp_path / "b"
    write_xdatcar(p1, ["Mg", "O"], [3, 4], wrapped[:25], lat)
    write_xdatcar(p2, ["Mg", "O"], [3, 4], wrapped[25:], lat, start=26)
    h = tj.read_header(p1)
    u = Unwrapper()
    got = [u(c.frac) for f in (p1, p2) for c in tj.iter_chunks(f, h, chunk_bytes=200)]
    got = np.concatenate(got)
    assert np.allclose(got - got[0], uw - uw[0], atol=1e-7)


def test_truncated_last_frame_dropped(tmp_path):
    _, wrapped = _walk(n_frames=5)
    p = tmp_path / "XDATCAR"
    write_xdatcar(p, ["Mg", "O"], [3, 4], wrapped, np.eye(3) * 5)
    text = p.read_text().splitlines(keepends=True)
    p.write_text("".join(text[:-3]))  # cut the last frame
    with pytest.warns(UserWarning, match="incomplete"):
        _, frac, _ = _read_all(p, 1 << 20, None)
    assert len(frac) == 4


def test_numba_parser_matches_strtod():
    if not tj.HAVE_NUMBA:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(3)
    vals = np.concatenate([rng.random(3000), -rng.random(3000) * 7, [0.0, 1.0, 0.5, 1e-8]])
    vals = vals[: len(vals) // 3 * 3]
    text = "".join(f"  {a:.8f}  {b:.8f}  {c:.8f}\n" for a, b, c in vals.reshape(-1, 3)).encode()
    out = tj.parse_frames(text, [0], [len(text)], len(vals) // 3, use_numba=True)
    ref = np.array([float(t) for t in text.split()])
    assert np.array_equal(out.ravel(), ref)


def test_vasp4_header_needs_species(tmp_path):
    p = tmp_path / "XDATCAR"
    p.write_text(
        "x\n 1\n 5 0 0\n 0 5 0\n 0 0 5\n 1 1\nDirect configuration= 1\n 0 0 0\n 0.5 0.5 0.5\n"
    )
    with pytest.raises(ValueError, match="species"):
        tj.read_header(p)
    h = tj.read_header(p, species=["Na", "Cl"])
    assert h.symbols == ["Na", "Cl"]
