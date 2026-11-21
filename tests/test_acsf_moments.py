"""G5 from Cartesian moment tensors (ADR-0038).

With ``angular_kernel="auto"`` (the default) the G5 terms with integer
zeta are computed from moment tensors, with a cost linear in the number
of neighbours; ``"pairs"`` keeps the pair loop of ADR-0030 for every
angular term.  The two must give the same G and dGdv to 1e-10 of the
largest value of each column (columns that are exactly zero, such as a
species pair with no neighbour of one species, rule out an element-wise
relative test), and the forces and stress must be the derivatives of
the energy.
"""

import numpy as np
import pytest
from test_acsf_kernel_reference import DIAMOND, G2, G4, G5_ISO_R2, _cell, _species
from test_mlip_atom_centred import _fd_check

from akitsushima.fitting.descriptors import ACSF

TOL = 1e-10
G5_ZETA16 = {"eta": [0.0, 0.05], "zeta": [1, 16], "lambda": [-1, 1]}
G5_MIXED_ZETA = {"eta": [0.02], "zeta": [2, 2.5], "lambda": [-1, 1]}

# grid -> (descriptor keywords, angular_kernel, number of angular terms
# on the tensors); auto leaves zeta > MOMENT_ZETA_MAX to the pair loop
GRIDS = {
    "g5_iso_r2": ({"g2": G2, "g5": G5_ISO_R2}, "auto", 12),
    "g5_zeta16_moments": ({"g2": G2, "g5": G5_ZETA16}, "moments", 8),
    "g5_zeta16_auto": ({"g2": G2, "g5": G5_ZETA16}, "auto", 4),
    "g4_and_g5": ({"g2": G2, "g4": G4, "g5": G5_ISO_R2}, "auto", 12),
    "g5_mixed_zeta": ({"g2": G2, "g5": G5_MIXED_ZETA}, "auto", 2),
    "g5_inner_cutoff": (
        {"g2": G2, "g5": G5_ISO_R2, "inner_cutoff": (1.9, 2.5)}, "auto", 12),
}


def _structure(name, kind):
    """64-atom rattled diamond cell, cubic or sheared, compressed or
    stretched by 15 % in volume-scale, or an 8-atom cell whose edge is
    shorter than 2 r_c (several images of one atom in the list)."""
    if kind == "small":
        rng = np.random.default_rng(5)
        a = 5.431 if name == "Si" else 5.9
        lat = np.array([[a, 0.0, 0.0], [0.4, a, 0.0], [0.3, -0.2, a]])
        pos = DIAMOND @ lat + rng.normal(0.0, 0.1, (8, 3))
        types = ["Si"] * 8 if name == "Si" else ["Ge", "Sn", "Ge", "Ge", "Sn", "Ge", "Ge", "Sn"]
        return lat, pos, types
    lat, pos, types = _cell(name, kind != "cubic")
    scale = {"cubic": 1.0, "triclinic": 1.0, "compressed": 0.85, "stretched": 1.15}[kind]
    return lat * scale, pos * scale, types


def _colrel(a, b):
    """Largest |a - b| of each last-axis column over the largest |b| of
    that column (1 where the column is zero)."""
    a = np.asarray(a, float).reshape(-1, a.shape[-1])
    b = np.asarray(b, float).reshape(-1, b.shape[-1])
    scale = np.abs(b).max(axis=0)
    scale[scale == 0.0] = 1.0
    return float((np.abs(a - b).max(axis=0) / scale).max())


@pytest.mark.parametrize("kind", ["cubic", "triclinic", "compressed", "stretched", "small"])
@pytest.mark.parametrize("grid", sorted(GRIDS))
@pytest.mark.parametrize("name", ["Si", "GeSn"])
@pytest.mark.slow
def test_moments_match_the_pair_loop(name, grid, kind):
    kw, mode, n_mom = GRIDS[grid]
    auto = ACSF.from_grid(_species(name), 5.0, **kw, angular_kernel=mode)
    pairs = ACSF.from_grid(_species(name), 5.0, **kw, angular_kernel="pairs")
    assert int(auto._moment_terms().sum()) == n_mom
    lat, pos, types = _structure(name, kind)
    a = auto._run(lat, pos, types, True)
    p = pairs._run(lat, pos, types, True)
    assert np.abs(p[4]).max() > 0
    assert _colrel(a[4], p[4]) < TOL, f"G: {_colrel(a[4], p[4]):.2e}"
    assert _colrel(a[5], p[5]) < TOL, f"dGdv: {_colrel(a[5], p[5]):.2e}"
    # the descriptor-only path gives the same G
    assert _colrel(auto._run(lat, pos, types, False)[4], p[4]) < TOL


@pytest.mark.parametrize(
    "grid", ["g5_zeta16_moments", "g4_and_g5", "g5_mixed_zeta", "g5_inner_cutoff"]
)
@pytest.mark.slow
def test_moment_forces_and_stress_match_finite_differences(grid):
    """E = sum_i c_i . G_i with random c on the small two-species cell:
    forces and stress from dGdv against central differences, and the
    forces sum to zero."""
    kw, mode, _ = GRIDS[grid]
    ac = ACSF.from_grid(["Ge", "Sn"], 5.0, **kw, angular_kernel=mode)
    lat, pos, types = _structure("GeSn", "small")
    c = np.random.default_rng(1).normal(size=(len(types), ac.n_per_atom))
    # scale c per column so that every column contributes to E
    G0 = ac.atomic(lat, pos, types)
    c /= np.maximum(np.abs(G0).max(axis=0), 1e-12)

    def ff(lat_, pos_, types_):
        J = ac.atomic_jacobian(lat_, pos_, types_)
        F, S = J.forces_and_stress(c)
        return float(np.sum(c * J.G)), F, S

    _, F, _ = ff(lat, pos, types)
    np.testing.assert_allclose(F.sum(axis=0), 0.0, atol=1e-10)
    _fd_check(ff, lat, pos, types, 1e-5, 1e-6, 1e-7)


def test_angular_kernel_choice_and_validation():
    g5 = {"g2": G2, "g5": G5_ISO_R2}
    auto = ACSF.from_grid(["Ge", "Sn"], 5.0, **g5)
    assert auto.angular_kernel == "auto" and auto._moment_terms().all()
    assert not ACSF.from_grid(["Ge", "Sn"], 5.0, **g5, angular_kernel="pairs")._moment_terms().any()
    assert ACSF.from_grid(["Ge", "Sn"], 5.0, **g5, angular_kernel="moments")._moment_terms().all()
    # the choice is not part of the descriptor
    assert "angular_kernel" not in auto.config()
    assert ACSF.from_config(auto.config()).angular_kernel == "auto"
    assert ACSF.from_config(auto.to_json(), angular_kernel="pairs").angular_kernel == "pairs"
    with pytest.raises(ValueError, match="angular_kernel must be"):
        ACSF.from_grid(["Ge"], 5.0, **g5, angular_kernel="fast")
    for bad in ({"g2": G2, "g4": G4}, {"g2": G2, "g5": G5_MIXED_ZETA}):
        with pytest.raises(ValueError, match="needs every angular term"):
            ACSF.from_grid(["Ge"], 5.0, **bad, angular_kernel="moments")
    # G4 and non-integer zeta stay on the pair loop under auto
    mixed = ACSF.from_grid(["Ge"], 5.0, g2=G2, g4=G4, g5=G5_MIXED_ZETA)
    assert mixed._moment_terms().tolist() == [False] * 12 + [True, True, False, False]
    # auto caps zeta at MOMENT_ZETA_MAX; moments does not
    z16 = {"g2": G2, "g5": G5_ZETA16}
    assert ACSF.from_grid(["Ge"], 5.0, **z16)._moment_terms().tolist() == [
        True, True, False, False, True, True, False, False]
    assert ACSF.from_grid(["Ge"], 5.0, **z16, angular_kernel="moments")._moment_terms().all()
