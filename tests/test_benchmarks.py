"""Smoke test of benchmarks/bench_force.py (ADR-0028, tier A step 1)."""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "benchmarks", "bench_force.py")


@pytest.mark.slow
def test_bench_force_small_cells(tmp_path):
    out = tmp_path / "ledger.jsonl"
    cmd = [
        sys.executable,
        SCRIPT,
        "--threads",
        "1",
        "--out",
        str(out),
        "--repeat",
        "1",
        "--cells-diamond",
        "2",
        "--cells-fcc",
        "4",
        "--cells-rocksalt",
        "4",
        "--n-hidden",
        "32",
        "--only",
        "sw_si,eam_ni,buck_pppm_mgo,acsf",
    ]
    env = dict(
        os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    subprocess.run(cmd, check=True, env=env, cwd=tmp_path, capture_output=True)
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["case"] for r in recs] == [
        "sw_si",
        "eam_ni",
        "buck_pppm_mgo",
        "acsf27_elm32_si",
    ]
    for r in recs:
        assert r["call_s"] > 0.0 and r["threads"] == 1
        # translation invariance of the forces
        assert r["sum_force_max"] < 1e-9
    for r in recs[:3]:
        # the call with a neighbour list and the all-pairs call agree
        assert r["call_nl_s"] > 0.0 and r["nl_build_s"] > 0.0
        assert r["nl_vs_no_nl_de_per_atom"] < 1e-9
        assert r["nl_vs_no_nl_df_max"] < 1e-8

    # comparing a ledger with itself gives ratios of 1
    cmp = subprocess.run(
        cmd[:2]
        + [
            "--threads",
            "1",
            "--out",
            str(tmp_path / "b.jsonl"),
            "--repeat",
            "1",
            "--cells-diamond",
            "2",
            "--only",
            "sw_si",
            "--compare",
            str(out),
        ],
        check=True,
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert "old / new" in cmp.stdout and "sw_si" in cmp.stdout.split("old / new")[1]
