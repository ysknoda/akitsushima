"""ADR-0025: the ACSF neighbour search is a cell list (O(N)) that returns
exactly what the former dense (N, N, images, 3) search returned -- the
same entries, in the same order, with bitwise-equal vectors -- for any
cell, including cells thinner than the cutoff and atoms outside the cell.
"""

import math
import time

import numpy as np
import pytest

from akitsushima.fitting.descriptors import acsf


def _dense_neighbours(lattice, positions, r_cut):
    """The pre-ADR-0025 implementation, kept as the reference."""
    lattice = np.asarray(lattice, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    N = len(pos)
    vol = abs(np.linalg.det(lattice))
    n_img = []
    for k in range(3):
        cross = np.cross(lattice[(k + 1) % 3], lattice[(k + 2) % 3])
        n_img.append(math.ceil(r_cut / (vol / np.linalg.norm(cross))))
    rng = [np.arange(-n, n + 1) for n in n_img]
    shifts = np.array(np.meshgrid(*rng, indexing="ij")).reshape(3, -1).T
    shift_cart = shifts @ lattice
    frac = pos @ np.linalg.inv(lattice)
    pos_w = (frac - np.floor(frac)) @ lattice
    dv = pos_w[None, :, None, :] + shift_cart[None, None, :, :] - pos_w[:, None, None, :]
    r2 = np.einsum("ijsk,ijsk->ijs", dv, dv)
    ii, jj, ss = np.nonzero((r2 < r_cut * r_cut) & (r2 > 1e-20))
    ptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(ptr, ii + 1, 1)
    return np.cumsum(ptr), jj.astype(np.int64), dv[ii, jj, ss]


def _cells():
    rng = np.random.default_rng(3)
    a = 5.65
    prim = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    yield prim, np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) @ prim
    conv = np.eye(3) * a
    base = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    frac = np.vstack([base, base + 0.25])
    sc = np.array([[i, j, k] for i in range(3) for j in range(3) for k in range(3)])
    yield 3 * conv, ((frac[None] + sc[:, None]).reshape(-1, 3) / 3) @ (3 * conv)
    for _ in range(12):  # thin and sheared triclinic cells, atoms outside the cell
        L = np.diag(rng.uniform(2.5, 9.0, 3)) + np.triu(rng.uniform(-3, 3, (3, 3)), 1)
        n = int(rng.integers(1, 10))
        yield L, rng.uniform(-1.5, 2.5, (n, 3)) @ L


@pytest.mark.parametrize("r_cut", [3.1, 5.0, 7.3])
def test_cell_list_equals_dense_search_bitwise(r_cut):
    for L, P in _cells():
        ref = _dense_neighbours(L, P, r_cut)
        got = acsf._neighbours(L, P, r_cut)
        for a, b in zip(ref, got):
            assert np.array_equal(a, b)


def test_replicated_cell_gives_the_same_descriptors():
    L, P = list(_cells())[1]
    d = acsf.ACSF.from_grid(["Ge", "Sn"], 5.0, g2={"eta": [4.0], "rs": [2.4, 3.8]},
                            g5={"eta": [0.05], "zeta": [1, 2], "lambda": [-1, 1]})
    T = ["Ge"] * len(P)
    T[::7] = ["Sn"] * len(T[::7])
    G1 = d.atomic(L, P, T)
    sh = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)]) @ L
    G2 = d.atomic(2 * L, (P[None] + sh[:, None]).reshape(-1, 3), T * 8)
    for r in range(8):
        np.testing.assert_allclose(G2[r * len(P):(r + 1) * len(P)], G1, rtol=0, atol=1e-12)


def test_large_cell_is_linear():
    """3456 atoms: the dense search needed ~8 GB here; the cell list ~MB."""
    L, P = list(_cells())[1]
    sc = np.array([[i, j, k] for i in range(4) for j in range(4) for k in range(4)]) @ L
    big = (P[None] + sc[:, None]).reshape(-1, 3)
    acsf._neighbours(L, P, 5.0)  # compile
    t = time.perf_counter()
    ptr, _nbj, vec = acsf._neighbours(4 * L, big, 5.0)
    assert time.perf_counter() - t < 20.0
    per_atom = np.diff(ptr)
    assert per_atom.min() == per_atom.max()  # every diamond site is alike
    assert np.all(np.linalg.norm(vec, axis=1) < 5.0)
