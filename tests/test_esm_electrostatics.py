"""ESM electrostatics against independent references (ADR-0003 T6, E1-E3).

The references are Gaussian-charge energies from an independent 3D Ewald
sum of point charges minus the erfc short-range part, adapted to each
boundary condition (example/esm/check_esm.py: bc1 padded z-period with the
Yeh-Berkowitz slab correction; bc2 / bc3 with the image charges of one /
two grounded plates), converged in their padding.  The +-1 pair sits at
x = y = 8 A in a 16 x 16 x 32 A cell, sigma = 0.5 A.

Before the fixes every value below was off by 0.26-1.8 eV per ion (the
self-energy divided by the cell area, E1; the G|| = 0 field with the wrong
sign, E3; spurious images in bc1 and a wrong bc3 kernel, E3).  What is left
is the z-grid sampling of the Gaussians, O(dz^2): 5 meV per ion at
n_grid_z = 512, 1.3 at 1024.
"""

import numpy as np
import pytest

from akitsushima.engine.coulomb.esm import ESMCalculator

LAT = np.diag([16.0, 16.0, 32.0])
CHARGES = {"Na": 1.0, "Cl": -1.0}

# (bc, d) -> reference energy [eV], check_esm.py 2026-09-26
REF = {
    ("bc1", 3.0): -4.6589353609869075,
    ("bc1", 5.0): -2.4982647789252743,
    ("bc2", 3.0): -4.658945213307015,
    ("bc2", 5.0): -2.498298169653953,
    ("bc3", 3.0): -4.758354573667387,
    ("bc3", 5.0): -2.77444130498018,
}


def _pair(d):
    return np.array([[8.0, 8.0, -d / 2], [8.0, 8.0, d / 2]])


def _calc(bc, pos, field=0.0, nz=512):
    return ESMCalculator(
        LAT,
        pos,
        ["Na", "Cl"],
        CHARGES,
        bc=bc,
        sigma=0.5,
        n_grid_xy=32,
        n_grid_z=nz,
        external_field=field,
    )


@pytest.mark.parametrize("bc,d", sorted(REF))
def test_pair_energy_matches_the_image_ewald_reference(bc, d):
    pos = _pair(d)
    E = _calc(bc, pos)(LAT, pos, ["Na", "Cl"])[0]
    assert abs(E - REF[(bc, d)]) / 2 < 8e-3


def test_z_and_x_pairs_differ_only_by_the_lateral_images():
    """E3 made a pair along z 1.5 eV more bound than the same pair along x;
    the lateral-image difference between them is 0.2 eV (reference
    -4.659 vs -4.874 eV)."""
    z = _pair(3.0)
    x = np.array([[6.5, 8.0, 0.0], [9.5, 8.0, 0.0]])
    Ez = _calc("bc1", z)(LAT, z, ["Na", "Cl"])[0]
    Ex = _calc("bc1", x)(LAT, x, ["Na", "Cl"])[0]
    assert Ez - Ex == pytest.approx(-4.6589 + 4.8741, abs=0.02)


def test_external_field_slope_is_the_field_in_v_per_angstrom():
    """E2: the field was applied K = 14.40 times too strongly.  The change
    of the planar-averaged potential between E = 0 and E = 0.1 V/A must be
    linear in z with slope magnitude 0.1 eV/A per unit charge."""
    pos = _pair(3.0)
    profiles = []
    for field in (0.0, 0.1):
        c = _calc("bc3", pos, field=field, nz=256)
        c(LAT, pos, ["Na", "Cl"])
        profiles.append(c.get_potential_profile())
    z, v0 = profiles[0]
    dv = profiles[1][1] - v0
    slope = np.polyfit(z, dv, 1)[0]
    assert abs(slope) == pytest.approx(0.1, rel=1e-3)
