"""ADR-0023, decision 5: the code examples in the public docstrings are
specification, so each one is run.

None of them is self-contained: they use names (``lattice``, ``bp``,
``pos_init`` ...) and files (``POSCAR``, ``lj_Ar.json`` ...) that the
docstring never defines.  Each test below supplies that context, then
executes the examples verbatim, so a renamed keyword, a removed method or
a changed result key in the API breaks the test.  The only edits are the
listed ``subs``, which shrink step counts so that the tests stay fast.
"""

import doctest
import importlib
import shutil

import numpy as np
import pytest
from test_elm_options import _si_acsf
from test_mlip_chain_rule import _sw_si_structures

import akitsushima as ak
from akitsushima.fitting.descriptors import build_table, write_table_csv
from akitsushima.fitting.io_utils import load_fit_results, load_potential
from akitsushima.fitting.mlip_engine import load_mlip_model
from akitsushima.fitting.models.atomic import AtomicELMModel
from akitsushima.utils.timing import TimingCollector

# Lennard-Jones argon, Verlet, Phys. Rev. 159, 98 (1967).
LJ = {frozenset({"Ar"}): {"epsilon": 0.0103, "sigma": 3.405}}
A_AR = 5.256  # fcc Ar lattice constant [A], close to the LJ minimum 1.5496 sigma


def _examples(module, name):
    """Sources of the docstring examples of akitsushima.<module>.<name>."""
    mod = importlib.import_module("akitsushima." + module)
    for test in doctest.DocTestFinder().find(mod):
        if test.name == f"akitsushima.{module}.{name}":
            assert test.examples, f"{test.name} has no examples"
            return [e.source for e in test.examples]
    raise AssertionError(f"no docstring examples found for {module}.{name}")


def _run(sources, namespace, subs=()):
    for i, src in enumerate(sources):
        for old, new in subs:
            if old in src:
                src = src.replace(old, new)
        exec(compile(src, f"<docstring example {i}>", "exec"), namespace)
    return namespace


def _ns(**extra):
    ns = {k: getattr(ak, k) for k in ak.__all__ if k != "__version__"}
    ns.update(np=np, load_mlip_model=load_mlip_model,
              load_fit_results=load_fit_results, load_potential=load_potential,
              TimingCollector=TimingCollector)
    ns.update(extra)
    return ns


def _fcc(a, n):
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    pos = (cells[:, None, :] + base[None, :, :]).reshape(-1, 3) * a
    return np.eye(3) * a * n, pos


# Buckingham MgO, Lewis & Catlow, J. Phys. C 18, 1149 (1985).
BUCK_MGO = {frozenset({"Mg", "O"}): {"A": 1428.5, "rho": 0.2945, "C": 0.0},
            frozenset({"O"}): {"A": 22764.0, "rho": 0.1490, "C": 27.88}}
Q_MGO = {"Mg": 2.0, "O": -2.0}


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    # Every example writes its output files (neb.log, POSCAR_TS, ...) to
    # the working directory.
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _rocksalt_mgo(n):
    lat, pos = _fcc(4.212, n)
    pos = np.vstack([pos, pos + [2.106, 0.0, 0.0]])
    return lat, pos, ["Mg"] * (len(pos) // 2) + ["O"] * (len(pos) // 2)


def test_rdf_example():
    lat, pos = _fcc(A_AR, 5)
    ns = _run(_examples("solvers.rdf", "RadialDistributionFunction"),
              _ns(lattice=lat, positions=pos, atom_types=["Ar"] * len(pos)))
    # First fcc shell at a / sqrt(2), 12 neighbours.
    r, g = ns["result"]["r"], ns["result"]["g_total"]
    assert r[np.argmax(g)] == pytest.approx(A_AR / np.sqrt(2), abs=0.1)


def test_timing_example(capsys):
    ns = _run(_examples("utils.timing", "TimingCollector"), _ns(n_steps=3))
    assert ns["summary"]


def test_poscar_example(in_tmp):
    lat, pos = _fcc(A_AR, 1)
    ak.write_poscar("POSCAR", lat, pos, ["Ar"] * 4)
    _run(_examples("io.poscar", "write_poscar"), _ns())
    back = ak.parse_poscar("POSCAR_out")
    np.testing.assert_allclose(back["positions"], pos, atol=1e-8)


def test_save_and_load_potential_lj_examples(in_tmp):
    lat, pos = _fcc(A_AR, 5)  # 26.3 A > 2 * (10 + 0.5)
    types = ["Ar"] * len(pos)
    fn = ak.make_force_fn(ak.PotentialEngine().add(
        ak.LennardJonesPotential(LJ, r_cut=10.0)))
    rng = np.random.default_rng(0)
    structs = []
    for s in (0.98, 1.0, 1.02):
        L, P = lat * s, pos * s + rng.normal(0, 0.02, pos.shape)
        e, f, st = fn(L, P, types)
        structs.append({"lattice": L, "positions": P, "atom_types": types,
                        "natoms": len(P), "ref_energy": e / len(P),
                        "ref_forces": f, "ref_stress": st})
    ns = _ns(prs=[frozenset({"Ar"})],
             init={frozenset({"Ar"}): {"epsilon": 0.012, "sigma": 3.3}},
             structs=structs, lat=lat, pos=pos, types=types)
    _run(_examples("fitting.io_utils", "save_potential"), ns)
    _run(_examples("fitting.io_utils", "load_potential")[:3], ns)
    assert ns["meta"]["potential_type"] == "lj"


def test_buckingham_examples(in_tmp):
    lat, pos, types = _rocksalt_mgo(5)  # 21.1 A > 2 * (8 + 0.5)
    result = {"params": {"buck": BUCK_MGO, "charges": Q_MGO}, "final_loss": 0.0}
    for path in ("mgo_buck.json", "fitted_potential.json"):
        ak.save_potential(result, path, potential_type="buckingham",
                          r_cut=8.0, species=["Mg", "O"])
    ns = _ns(bp=BUCK_MGO, ch=Q_MGO, lat=lat, pos=pos, types=types)
    _run(_examples("fitting.force_fn", "make_force_fn")[:3], ns)
    _run(_examples("fitting.io_utils", "load_potential")[3:], ns)
    _run(_examples("fitting.io_utils", "load_fit_results"), ns)
    assert ns["res"]["charges"] == Q_MGO


@pytest.fixture
def si_mlip(in_tmp):
    """A small Si model and descriptor table (SW teacher, ACSF)."""
    structs = _sw_si_structures(6, 0)
    ac = _si_acsf()
    AtomicELMModel(n_hidden=20, verbose=False).fit(structs, ac).save(
        "atomic_elm_model.npz")
    y, lab, w, X = build_table(structs, ac)
    write_table_csv("training.csv", y, lab, w, X, ac.feature_names())
    shutil.copy("training.csv", "structures.csv")
    return in_tmp


def test_make_force_fn_mlip_example(si_mlip):
    ns = _run(_examples("fitting.force_fn", "make_force_fn")[3:], _ns())
    assert callable(ns["fn"])


def test_qelm_and_elm_fit_csv_examples(si_mlip):
    _run(_examples("fitting.models.quantized_elm", "QuantizedELMModel"), _ns(),
         subs=[("n_hidden=512", "n_hidden=16")])
    _run(_examples("fitting.models.elm", "ELMModel.fit_csv")[:1],
         _ns(model=ak.ELMModel(n_hidden=16)))


def test_phonon_example(in_tmp):
    # r_cut 5 A so that the 3x3x3 supercell of the 4-atom cell (15.8 A)
    # clears the minimum-image bound 2 * (5 + 0.5) A.
    ak.save_potential({"params": {"lj": LJ}, "final_loss": 0.0}, "lj_Ar.json",
                      potential_type="lj", r_cut=5.0, species=["Ar"])
    lat, pos = _fcc(A_AR, 1)
    ns = _run(_examples("solvers.phonon", "PhononCalculator"),
              _ns(lattice=lat, positions=pos, atom_types=["Ar"] * 4))
    assert ns["bands"] is not None


def test_neb_and_tsa_examples():
    """Vacancy hop in fcc Ar: NEB, then the TSA example on its saddle."""
    lat, pos = _fcc(A_AR, 3)
    types = ["Ar"] * (len(pos) - 1)
    engine = ak.PotentialEngine().add(ak.LennardJonesPotential(LJ, r_cut=5.0))
    ini = pos[1:].copy()
    fin = ini.copy()
    fin[0] = pos[0]  # a nearest neighbour of site 0 hops into the vacancy
    ns = _run(_examples("solvers.neb", "NEBOptimiser"),
              _ns(pos_init=ini, pos_final=fin, lattice=lat, atom_types=types,
                  engine=engine),
              subs=[("max_steps=500", "max_steps=40")])
    assert ns["result"]["activation_energy"] > 0.0
    _run(_examples("solvers.tsa", "TransitionStateAnalyser"),
         _ns(lattice=lat, atom_types=types, engine=engine,
             neb_result=ns["result"],
             E_init=ak.make_force_fn(engine)(lat, ini, types)[0]),
         subs=[("run_irc()", "run_irc(max_steps=20)")])


def test_gcmc_example(in_tmp):
    n = 20
    pos = np.random.default_rng(1).uniform(0.0, 24.0, (n, 3))
    ns = _run(_examples("solvers.gcmc", "GCMCMDSimulation"),
              _ns(lj=LJ, lattice=np.eye(3) * 24.0, positions=pos, N=n),
              subs=[("n_cycles=2000", "n_cycles=3")])
    assert np.isfinite(ns["result"]["N_mean"])


def test_diffusion_example():
    lat, pos = _fcc(A_AR, 5)
    fn = ak.make_force_fn(ak.PotentialEngine().add(
        ak.LennardJonesPotential(LJ, r_cut=10.0)))
    _run(_examples("solvers.dynamics_analysis", "DiffusionAnalyzer"),
         _ns(lattice=lat, positions=pos, atom_types=["Ar"] * len(pos),
             force_fn=fn),
         subs=[("n_steps=20000", "n_steps=40"),
               ("equil_steps=2000", "equil_steps=10")])


def test_esm_example():
    # NaCl slab centred on z = 0: ESM's boundaries sit at z = +/- Lz/2.
    a = 5.64
    slab = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    slab -= [0.0, 0.0, a / 4]
    pos = np.vstack([slab, slab + [a / 2, 0.0, 0.0]])
    _run(_examples("engine.coulomb.esm", "ESMCalculator"),
         _ns(lattice=np.diag([a, a, 3 * a]), positions=pos,
             atom_types=["Na"] * 4 + ["Cl"] * 4),
         subs=[("relax_cell=False)", "relax_cell=False, max_steps=3)")])


def test_scaled_force_engine_example_constructs():
    # The last two examples run VASP ('vasp_std'); only construction is run.
    lat, pos = _fcc(A_AR, 1)
    ns = _run(_examples("fitting.external", "ScaledForceEngine")[:3],
              _ns(lat=lat, pos=pos, types=["Ar"] * 4))
    assert callable(ns["fn"])
