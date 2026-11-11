"""Force-call benchmark of akitsushima (ADR-0028, tier A step 1).

Times one warm force call for each potential family at about 500 and
4000 atoms, in one process, and appends one JSON line per case to a
ledger.  Everything is built inside this script from parameters that are
part of the package or cited below, so it runs from a clean checkout.

Measured per case
    nl_build_s   one ``NeighbourList.build`` (best of ``--repeat``)
    call_nl_s    force call with that neighbour list (best of ``--repeat``)
    call_s       force call without a neighbour list (best of ``--repeat``)
    profile      with ``--profile``: the functions with the largest self time

Checks printed with every case: energy per atom, and the largest component
of the summed force (translation invariance, expected ~1e-12 eV/A).

Usage
    python benchmarks/bench_force.py --threads 1 --out benchmarks/baseline.jsonl
    python benchmarks/bench_force.py --threads 4 --out new.jsonl \
        --compare benchmarks/baseline.jsonl

``--compare`` prints old / new for every timing that both ledgers hold
(same case, size, thread count and tag ``--tag``).  ADR-0028 adopts a
speed-up at a ratio of 1.5 or more.

Parameter sources
    SW Si       Stillinger and Weber, Phys. Rev. B 31, 5262 (1985)
    Tersoff Si  Tersoff, Phys. Rev. B 38, 9902 (1988), Si(C)
    EAM Ni      Zhou, Johnson and Wadley, Phys. Rev. B 69, 144113 (2004),
                Table III (akitsushima.engine.published_parameters)
    MgO         Lewis and Catlow, J. Phys. C 18, 1149 (1985), formal charges
Lattice constants (Si 5.431, Ni 3.52, MgO 4.212 A) and the 0.05 A rattle
only define the benchmark geometry; nothing is fitted to them.  The MgO
cutoff of 6.0 A is a benchmark setting, not a converged value.
"""

import argparse
import cProfile
import json
import os
import platform
import pstats
import subprocess
import sys
import time

import numpy as np

DIAMOND = [
    [0, 0, 0],
    [0, 0.5, 0.5],
    [0.5, 0, 0.5],
    [0.5, 0.5, 0],
    [0.25, 0.25, 0.25],
    [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75],
    [0.75, 0.75, 0.25],
]
FCC = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
ROCKSALT = FCC + [[0.5, 0, 0], [0.5, 0.5, 0.5], [0, 0, 0.5], [0, 0.5, 0]]

SW_SI = {
    "type": "SW",
    "species": ["Si"],
    "params": {
        "epsilon": 2.1683,
        "sigma": 2.0951,
        "a": 1.80,
        "gamma": 1.20,
        "lambda": 21.0,
        "A": 7.049556277,
        "B": 0.6022245584,
        "p": 4.0,
        "q": 0.0,
        "costheta0": -1 / 3,
    },
}
TERSOFF_SI = {
    "type": "Tersoff",
    "species": ["Si"],
    "params": {
        "A": 1830.8,
        "B": 471.18,
        "lambda1": 2.4799,
        "lambda2": 1.7322,
        "lambda3": 0.0,
        "beta": 1.1e-6,
        "n": 0.78734,
        "c": 100390.0,
        "d": 16.217,
        "h": -0.59825,
        "R": 2.85,
        "D": 0.15,
    },
}
EAM_NI = {"type": "EAM", "form": "zhou2004", "elements": ["Ni"], "species": ["Ni"]}
BUCK_MGO = {
    frozenset({"Mg", "O"}): {"A": 1428.5, "rho": 0.2945, "C": 0.0},
    frozenset({"O"}): {"A": 22764.0, "rho": 0.1490, "C": 27.88},
}
Q_MGO = {"Mg": 2.0, "O": -2.0}
R_CUT_MGO = 6.0

# ACSF grid of the GeSn atom-centred ELM of ADR-0010 A1 (ac_elm_iso_r2).
ACSF_GRID = {
    "r_cut": 5.0,
    "g2": {"eta": [12.5], "rs": [2.0 + 0.2 * k for k in range(15)]},
    "g5": {"eta": [0.0, 0.05], "zeta": [1, 2, 4], "lambda": [-1, 1]},
}


def cubic(basis, a, n, types, rattle=0.05, seed=0):
    """n^3 cubic cells (edge a) of a fractional basis, rattled by `rattle` A."""
    rng = np.random.default_rng(seed)
    cells = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    frac = (np.asarray(basis, float)[None] + cells[:, None]).reshape(-1, 3)
    pos = frac * a + rng.normal(0.0, rattle, frac.shape)
    return np.eye(3) * a * n, pos, list(types) * len(cells)


def synthetic_mlip(n_hidden):
    """Atom-centred ELM on the ACSF_GRID descriptor, trained on SW silicon.

    Only the cost of the call matters here, not the accuracy: eight rattled
    64-atom cells labelled by Stillinger-Weber stand in for DFT.
    """
    import akitsushima as ak
    from akitsushima.fitting import AtomicELMModel
    from akitsushima.fitting.descriptors import ACSF

    sw = ak.force_fn_from_config(SW_SI)
    rng = np.random.default_rng(1)
    structures = []
    for k in range(8):
        lat, pos, types = cubic(
            DIAMOND, 5.431 * rng.uniform(0.98, 1.02), 2, ["Si"] * 8, seed=10 + k
        )
        e, f, s = sw(lat, pos, types)
        structures.append(
            {
                "lattice": lat,
                "positions": pos,
                "atom_types": types,
                "ref_energy": e / len(types),
                "ref_forces": f,
                "ref_stress": s,
            }
        )
    acsf = ACSF.from_grid(
        ["Si"], ACSF_GRID["r_cut"], g2=ACSF_GRID["g2"], g5=ACSF_GRID["g5"]
    )
    model = AtomicELMModel(n_hidden=n_hidden, seed=42, verbose=False).fit(
        structures, acsf
    )
    return ak.make_force_fn(model), acsf.n_per_atom


def build_cases(args):
    """(name, force_fn, r_cut or None, [(lattice, positions, types), ...])."""
    import akitsushima as ak

    si = [cubic(DIAMOND, 5.431, n, ["Si"] * 8) for n in args.cells_diamond]
    ni = [cubic(FCC, 3.52, n, ["Ni"] * 4) for n in args.cells_fcc]
    mgo = [
        cubic(ROCKSALT, 4.212, n, ["Mg"] * 4 + ["O"] * 4) for n in args.cells_rocksalt
    ]
    cases = []
    for name, cfg, structs in (
        ("sw_si", SW_SI, si),
        ("tersoff_si", TERSOFF_SI, si),
        ("eam_ni", EAM_NI, ni),
    ):
        fn = ak.force_fn_from_config(cfg)
        cases.append((name, fn, float(fn.r_cut), structs))
    for method, structs in (("pppm", mgo), ("ewald", mgo[:1])):
        fn = ak.make_force_fn(
            "buckingham",
            buck_params=BUCK_MGO,
            charges=Q_MGO,
            r_cut=R_CUT_MGO,
            coulomb_method=method,
        )
        cases.append((f"buck_{method}_mgo", fn, R_CUT_MGO, structs))
    fn, n_feat = synthetic_mlip(args.n_hidden)
    cases.append((f"acsf{n_feat}_elm{args.n_hidden}_si", fn, None, si))
    if args.mlip_model:
        fn = ak.force_fn_from_config({"type": "MLIP", "model": args.mlip_model})
        species = list(fn.species) if hasattr(fn, "species") else args.mlip_species
        if args.mlip_species:
            species = args.mlip_species
        rng = np.random.default_rng(2)
        structs = []
        for lat, pos, types in si:
            # the Si benchmark geometry with the model's species assigned at random
            structs.append(
                (lat, pos, [str(s) for s in rng.choice(species, len(types))])
            )
        cases.append(
            (
                f"mlip_file_{os.path.basename(os.path.dirname(os.path.dirname(args.mlip_model)))}",
                fn,
                None,
                structs,
            )
        )
    return cases


def best_time(call, repeat):
    out, times = None, []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = call()
        times.append(time.perf_counter() - t0)
    return min(times), out


def top_self(call, n_top):
    pr = cProfile.Profile()
    pr.enable()
    call()
    pr.disable()
    st = pstats.Stats(pr)
    total = st.total_tt
    rows = sorted(
        (
            (tt / total, f"{os.path.basename(k[0])}:{k[2]}")
            for k, (_, _, tt, _, _) in st.stats.items()
        ),
        reverse=True,
    )
    return [{"self_frac": round(x, 4), "where": w} for x, w in rows[:n_top]]


def environment(args):
    import akitsushima as ak

    try:
        import numba

        numba_version, threads = numba.__version__, numba.get_num_threads()
    except ImportError:
        numba_version, threads = None, 1
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        commit = ""
    return {
        "tag": args.tag,
        "threads": threads,
        "akitsushima": ak.__version__,
        "commit": commit,
        "numba": numba_version,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()} {os.cpu_count()} cpus",
        "load_avg_1min": round(os.getloadavg()[0], 2),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def key(rec):
    return rec["case"], rec["n_atoms"], rec["threads"], rec.get("tag", "")


def compare(old_path, new_recs):
    old = {}
    with open(old_path) as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                old[key(rec)] = rec  # a later line replaces an earlier one
    print(f"\nold / new against {old_path} (1.5 or more = adopted, ADR-0028):")
    for rec in new_recs:
        ref = old.get(key(rec))
        if ref is None:
            continue
        parts = []
        for q in ("nl_build_s", "call_nl_s", "call_s"):
            if rec.get(q) and ref.get(q):
                parts.append(f"{q} {ref[q] / rec[q]:6.2f}")
        print(f"  {rec['case']:28s} N={rec['n_atoms']:6d}  " + "  ".join(parts))


def run_case(name, fn, r_cut, structure, env, args):
    """Time one potential on one structure; append the record to the ledger."""
    lat, pos, types = structure
    from akitsushima import NeighbourList

    rec = {"case": name, "n_atoms": len(types), "r_cut": r_cut, **env}
    repeat = args.repeat

    def call(nl=None):
        return fn(lat, pos, types) if nl is None else fn(lat, pos, types, nl)

    if r_cut is not None:
        nl = NeighbourList(r_cut=r_cut)
        nl.build(lat, pos)  # warm-up (JIT, imports)
        rec["nl_build_s"], _ = best_time(lambda: nl.build(lat, pos), repeat)
        call(nl)
        rec["call_nl_s"], (e, f, _) = best_time(lambda: call(nl), repeat)
        if args.profile:
            rec["profile_nl"] = top_self(lambda: call(nl), args.top)
    t0 = time.perf_counter()
    e0, f0, _ = call()  # warm-up; also decides how often to repeat
    first = time.perf_counter() - t0
    rec["call_s"], (e0, f0, _) = best_time(
        call, 1 if first > args.max_seconds else repeat
    )
    if args.profile and first <= args.max_seconds:
        rec["profile"] = top_self(call, args.top)
    if r_cut is None:
        e, f = e0, f0
    else:
        rec["nl_vs_no_nl_de_per_atom"] = abs(float(e) - float(e0)) / len(types)
        rec["nl_vs_no_nl_df_max"] = float(np.abs(f - f0).max())
    rec["e_per_atom"] = float(e) / len(types)
    rec["sum_force_max"] = float(np.abs(f.sum(axis=0)).max())
    with open(args.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    cols = [
        f"{q}={rec[q]:.4f}" for q in ("nl_build_s", "call_nl_s", "call_s") if q in rec
    ]
    print(
        f"{name:28s} N={len(types):6d} "
        + " ".join(cols)
        + f" E/N={rec['e_per_atom']:.5f}"
        f" |sumF|={rec['sum_force_max']:.1e}"
        + (
            f" dF(nl)={rec['nl_vs_no_nl_df_max']:.1e}"
            if "nl_vs_no_nl_df_max" in rec
            else ""
        ),
        flush=True,
    )
    return rec


def main():
    ap = argparse.ArgumentParser(
        description="Force-call benchmark of akitsushima (ADR-0028).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--out", required=True, help="JSONL ledger to append to")
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="numba threads (sets NUMBA_NUM_THREADS before the import)",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="free label stored with every line, part of the comparison key",
    )
    ap.add_argument(
        "--only", default="", help="comma-separated case-name prefixes to run"
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="timed calls per quantity; the best is kept",
    )
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=60.0,
        help="stop repeating a call slower than this",
    )
    ap.add_argument(
        "--cells-diamond",
        type=int,
        nargs="+",
        default=[4, 8],
        help="n of the n^3 8-atom Si cells",
    )
    ap.add_argument(
        "--cells-fcc",
        type=int,
        nargs="+",
        default=[5, 10],
        help="n of the n^3 4-atom Ni cells",
    )
    ap.add_argument(
        "--cells-rocksalt",
        type=int,
        nargs="+",
        default=[4, 8],
        help="n of the n^3 8-atom MgO cells",
    )
    ap.add_argument(
        "--n-hidden",
        type=int,
        default=400,
        help="hidden units of the synthetic atom-centred ELM",
    )
    ap.add_argument(
        "--mlip-model",
        default="",
        help="also time this atomic model file (.npz), on the Si geometry",
    )
    ap.add_argument(
        "--mlip-species",
        nargs="+",
        default=None,
        help="species assigned at random for --mlip-model",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="store the functions with the largest self time",
    )
    ap.add_argument("--top", type=int, default=6, help="profile rows kept per call")
    ap.add_argument(
        "--compare", default="", help="ledger to compare the new timings with"
    )
    args = ap.parse_args()

    if args.threads is not None:
        os.environ["NUMBA_NUM_THREADS"] = str(args.threads)
    env = environment(args)
    print(json.dumps(env))
    only = [s for s in args.only.split(",") if s]
    done = []
    for name, fn, r_cut, structs in build_cases(args):
        if only and not any(name.startswith(o) for o in only):
            continue
        for structure in structs:
            done.append(run_case(name, fn, r_cut, structure, env, args))
    if args.compare:
        compare(args.compare, done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
