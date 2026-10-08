"""Site-energy decomposition and the exact many-body heat flux (ADR-0007).

1. For every potential with ``site_terms``, the forces and stress rebuilt
   from the site terms equal the engine's own (the decomposition describes
   the same energy), and the site energies sum to the total.
2. The heat flux of kappa's calculator equals J V = d/dt sum_i r_i E_i,
   E_i = K_i + U_i, by central differences over one velocity-Verlet step
   each way, on isolated clusters (no pair crosses a periodic boundary, so
   sum_i r_i E_i is well defined) -- for Stillinger-Weber, Tersoff, EAM
   and an atom-centred MLIP, where the former code handed the whole
   many-body virial to a centre-of-mass approximation that is ~0.
"""

import numpy as np
import pytest
from conftest import diamond_si_dict, fcc_al_dict, randomly_perturbed

from akitsushima.constants import EV_ANG_TO_AMU_ANG_FS2 as C
from akitsushima.engine import (
    EAMPotential,
    LennardJonesPotential,
    PotentialEngine,
    StillingerWeberPotential,
    TersoffPotential,
)
from akitsushima.fitting import make_force_fn
from akitsushima.solvers.thermal_conductivity import ThermalConductivityCalculator

# Stillinger & Weber, PRB 31, 5262 (1985), Si
SW_SI = {
    "epsilon": 2.1683,
    "sigma": 2.0951,
    "a": 1.80,
    "lambda": 21.0,
    "gamma": 1.20,
    "A": 7.049556277,
    "B": 0.6022245584,
    "p": 4.0,
    "q": 0.0,
    "costheta0": -1.0 / 3.0,
}
# Tersoff T3 Si, PRB 38, 9902 (1988), except lambda3: the paper has 0;
# 1.7322 is kept so the exp(lambda3^3 dr^3) term is exercised
T3_SI = {
    "A": 1830.8,
    "B": 471.18,
    "lambda1": 2.4799,
    "lambda2": 1.7322,
    "lambda3": 1.7322,
    "beta": 1.1e-6,
    "n": 0.78734,
    "c": 100390.0,
    "d": 16.217,
    "h": -0.59825,
    "R": 2.85,
    "D": 0.15,
}
T_XY = {
    ("X", "Y", "Y"): dict(T3_SI),
    ("Y", "X", "X"): {
        "A": 1500.0,
        "B": 400.0,
        "lambda1": 2.2,
        "lambda2": 1.5,
        "R": 2.85,
        "D": 0.15,
        "beta": 0.9e-6,
        "n": 0.7,
        "c": 90000.0,
        "d": 14.0,
        "h": -0.55,
        "lambda3": 1.5,
    },
}


def _eam():
    return EAMPotential(
        lambda rho: -0.5 * np.sqrt(np.maximum(rho, 1e-12)),
        lambda r: np.exp(-r / 1.5),
        lambda r: 500.0 * np.exp(-r / 0.4),
        r_cut=6.0,
        species=["Al"],
    )


def _gesn_sw():
    from test_sw_three_body_counting import _tomita

    return StillingerWeberPotential(_tomita(), species=["Ge", "Sn"])


def _si(seed):
    s = randomly_perturbed(diamond_si_dict(2), magnitude=0.1, seed=seed)
    return s["lattice"], s["positions"], s["atom_types"]


def _gesn():
    lat, pos, _ = _si(2)
    f = 5.9 / 5.43
    types = list(np.random.default_rng(3).choice(["Ge", "Sn"], len(pos)))
    return lat * f, pos * f, types


def _xy():
    lat, pos, _ = _si(4)
    return lat, pos, (["X"] * 4 + ["Y"] * 4) * 8


def _al():
    s = randomly_perturbed(fcc_al_dict(3), magnitude=0.1, seed=3)
    return s["lattice"], s["positions"], s["atom_types"]


CASES = {
    "sw": (
        lambda: StillingerWeberPotential(dict(SW_SI), species=["Si"]),
        lambda: _si(1),
        1e-9,
    ),
    "sw_multi_gesn": (_gesn_sw, _gesn, 1e-9),
    "tersoff": (
        lambda: TersoffPotential(dict(T3_SI), species=["Si"]),
        lambda: _si(1),
        1e-6,
    ),
    "tersoff_nested": (
        lambda: TersoffPotential({("Si", "Si", "Si"): dict(T3_SI)}, species=["Si"]),
        lambda: _si(1),
        1e-6,
    ),
    "tersoff_xy": (lambda: TersoffPotential(T_XY, species=["X", "Y"]), _xy, 1e-6),
    "eam": (_eam, _al, 1e-12),
    "lj": (
        lambda: LennardJonesPotential(
            {frozenset({"Al"}): {"epsilon": 0.4, "sigma": 2.6}}, r_cut=6.0
        ),
        _al,
        1e-12,
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_site_terms_reproduce_the_engine(name):
    """Energy, forces and stress from the site terms equal compute()'s.
    Tersoff's engine differentiates zeta by finite differences (errors
    ~1e-7 eV/A against central differences of its energy, where the
    analytic site terms agree to ~1e-8), hence its looser force bound."""
    make, struct, ftol = CASES[name]
    pot = make()
    lat, pos, types = struct()
    E, F, S = pot.compute(lat, pos, types)
    st = pot.site_terms(lat, pos, types)
    assert st.energy == pytest.approx(E, rel=1e-12, abs=1e-10)
    np.testing.assert_allclose(st.forces(), F, rtol=0, atol=ftol)
    np.testing.assert_allclose(
        st.stress(abs(np.linalg.det(lat))),
        S,
        rtol=0,
        atol=max(ftol, 1e-12) * 1e-2 + 1e-12,
    )


def test_tersoff_site_forces_are_the_energy_derivative():
    """The analytic site-term forces against central differences of the
    engine's own energy (the arbiter between the two)."""
    pot = TersoffPotential(dict(T3_SI), species=["Si"])
    lat, pos, types = _si(1)
    Fs = pot.site_terms(lat, pos, types).forces()
    h = 1e-5
    for a, c in [(0, 0), (5, 1), (17, 2), (33, 0), (48, 1), (63, 2)]:
        p = pos.copy()
        p[a, c] += h
        Ep = pot.compute(lat, p, types)[0]
        p[a, c] -= 2 * h
        Em = pot.compute(lat, p, types)[0]
        assert Fs[a, c] == pytest.approx(-(Ep - Em) / (2 * h), abs=5e-8)


# ---------------------------------------------------------------- heat flux


def _cluster(n_cells, a, types_fn, seed, L=40.0, jitter=0.08):
    """A small crystallite in the middle of an L box, rattled, with
    thermal-scale random velocities (zero total momentum)."""
    lat = np.eye(3) * L
    s = diamond_si_dict(1) if types_fn != "fcc" else fcc_al_dict(1)
    frac = s["positions"] / 5.43 if types_fn != "fcc" else s["positions"] / 4.05
    pts = np.concatenate(
        [
            (frac + [i, j, k]) * a
            for i in range(n_cells)
            for j in range(n_cells)
            for k in range(n_cells)
        ]
    )
    rng = np.random.default_rng(seed)
    pos = pts - pts.mean(0) + L / 2 + rng.normal(scale=jitter, size=pts.shape)
    return lat, pos, rng


def _flux_check(fn, lat, pos, types, masses, rng):
    tc = ThermalConductivityCalculator(lat, pos, types, fn, masses=masses)
    vel = rng.normal(
        scale=np.sqrt(8.617e-5 * 300 * C / masses[:, None]), size=pos.shape
    )
    vel -= (masses[:, None] * vel).sum(0) / masses.sum()
    V = abs(np.linalg.det(lat))

    def moment(p, v):
        e, f, _ = fn(lat, p, types)
        pd = tc._gather_pair_data(p)
        E = 0.5 * masses * np.sum(v**2, axis=1) / C + tc._per_atom_energy(
            p, f, float(e), pd
        )
        return (E[:, None] * p).sum(axis=0)

    def verlet(p, v, dt):
        v = v + 0.5 * dt * fn(lat, p, types)[1] * C / masses[:, None]
        p = p + dt * v
        v = v + 0.5 * dt * fn(lat, p, types)[1] * C / masses[:, None]
        return p, v

    e, f, s = fn(lat, pos, types)
    pd = tc._gather_pair_data(pos)
    assert pd is not None
    J = tc._compute_heat_flux(
        pos, vel, f, s, tc._per_atom_energy(pos, f, float(e), pd), pd
    )
    dt = 1e-3
    J_fd = (
        (moment(*verlet(pos, vel, dt)) - moment(*verlet(pos, vel, -dt))) / (2 * dt) / V
    )
    # the stress-based centre-of-mass fallback alone, for contrast
    v_mean = vel.mean(0)
    J_old = (
        np.einsum(
            "i,ix->x",
            0.5 * masses * np.sum(vel**2, 1) / C + np.full(len(pos), e / len(pos)),
            vel,
        )
        - V * s @ v_mean * 0.5
    ) / V
    return J, J_fd, J_old


@pytest.mark.parametrize("kind", ["sw", "sw_gesn", "tersoff_xy", "eam"])
def test_many_body_heat_flux_is_the_time_derivative_of_the_energy_moment(kind):
    if kind in ("sw", "sw_gesn"):
        lat, pos, rng = _cluster(2, 5.43 if kind == "sw" else 5.9, "d", 0)
        if kind == "sw":
            pot, types = (
                StillingerWeberPotential(dict(SW_SI), species=["Si"]),
                ["Si"] * len(pos),
            )
            m = np.full(len(pos), 28.0855)
        else:
            pot = _gesn_sw()
            types = list(rng.choice(["Ge", "Sn"], len(pos)))
            m = np.array([72.63 if t == "Ge" else 118.71 for t in types])
    elif kind == "tersoff_xy":
        lat, pos, rng = _cluster(2, 5.43, "d", 1)
        pot, types = (
            TersoffPotential(T_XY, species=["X", "Y"]),
            (["X"] * 4 + ["Y"] * 4) * 8,
        )
        m = np.full(len(pos), 28.0855)
    else:
        lat, pos, rng = _cluster(2, 4.05, "fcc", 2)
        pot, types = _eam(), ["Al"] * len(pos)
        m = np.full(len(pos), 26.98)
    fn = make_force_fn(PotentialEngine().add(pot))
    J, J_fd, J_old = _flux_check(fn, lat, pos, types, m, rng)
    scale = np.abs(J_fd).max()
    np.testing.assert_allclose(J, J_fd, rtol=0, atol=1e-5 * scale)
    # the approximation it replaces misses by far more than the tolerance
    assert np.abs(J_old - J_fd).max() > 1e-2 * scale


def test_atom_centred_mlip_heat_flux_is_the_time_derivative():
    from test_mlip_atom_centred import _si_acsf
    from test_mlip_chain_rule import _sw_si_structures

    from akitsushima.fitting.mlip_engine import MLIPForceField
    from akitsushima.fitting.models.atomic import AtomicELMModel

    model = AtomicELMModel(n_hidden=20, r0=0.1, eps0=0.01, verbose=False)
    model.fit(_sw_si_structures(8, 0), _si_acsf())
    fn = MLIPForceField(model)
    assert fn.per_atom_energy
    lat, pos, rng = _cluster(2, 5.43, "d", 5)
    types = ["Si"] * len(pos)
    J, J_fd, _ = _flux_check(fn, lat, pos, types, np.full(len(pos), 28.0855), rng)
    np.testing.assert_allclose(J, J_fd, rtol=0, atol=1e-5 * np.abs(J_fd).max())


def test_tabular_mlip_has_no_site_terms():
    from test_mlip_atom_centred import _si_acsf
    from test_mlip_chain_rule import _sw_si_structures

    from akitsushima.fitting.descriptors import build_table
    from akitsushima.fitting.mlip_engine import MLIPForceField
    from akitsushima.fitting.models import ELMModel

    structs = _sw_si_structures(8, 0)
    ac = _si_acsf()
    y, lab, w, X = build_table(structs, ac, r0=0.1, eps0=0.01)
    m = ELMModel(n_hidden=10, verbose=False)
    m.fit(y, lab, X, weights=w)
    m.descriptor_config = ac.to_json()
    fn = MLIPForceField(m)
    assert not fn.per_atom_energy
    with pytest.raises(ValueError, match="atom-centred"):
        fn.site_terms(
            structs[0]["lattice"], structs[0]["positions"], structs[0]["atom_types"]
        )
