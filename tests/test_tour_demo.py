"""WP9 full orchard tour (OS 1.2 steps 0-12, ``config/missions/orchard_tour.yaml``) on orchard_demo.

``tour`` (nightly, seeds 7 / 11 / 23): one process per seed, safety gates + skip limits (the tour pass definition),
records, job sheets, run card; the 18 min sim gate is a strict xfail until a project decision on CR-WP27-02 / CR-WP9-02.
``tour`` segments: the same steps split into five fresh-world segments, each <= 180 s wall on the 2 CPU runner (the
local cap); a segment whose actor trigger step is not inside it runs without actors (``--no-actors``).
``readiness``: 10 consecutive passing tours over seeds {7, 11, 23} (demo week)."""
import json
import os

import pytest

from atharv_nandi.core import demo_tour as dt
from atharv_nandi.core import records as rec

SEGMENTS = {
    "A": (["s00_ready", "s01_scan_r1", "s01b_headland", "s02_tree360", "s03_leaf_scan", "s04_fruit_scout"], True),
    "B1": (["s05_harvest_mango"], False),
    "B2": (["s06_harvest_guava", "s07_prune_guava", "s08_harvest_guava_twist"], True),
    # C split in two since the 10 b refusal probe walks to the patch (one C segment ran past the 180 s cap)
    "C1": (["s08b_harvest_papaya", "s09_prune_banana", "s10b_patch_refusal"], False),
    "C2": (["s10a_gust", "s10e_estop", "s11_dock_unload", "s11b_allowance", "s12_report"], False),
}


def _check(res):
    t = res["tour"]
    print("\n", json.dumps({k: t[k] for k in ("pass", "fail_reasons", "sim_s", "wall_s", "safety")}, default=str))
    for r in res["steps"]:
        print(f"  {r['id']:<26} {r['status']:<8} {r['sim_s']:>6.1f} s {r['reason']}")
    assert res["violations"] == [] and res["jobsheet_missing"] == []
    assert t["pass"], t["fail_reasons"]


@pytest.mark.tour
@pytest.mark.parametrize("seg", sorted(SEGMENTS))
def test_tour_segment(seg, tmp_path):
    steps, actors = SEGMENTS[seg]
    res = dt.run_tour("orchard_tour", 7, steps=steps, runs_dir=str(tmp_path), actors=actors, run_id=f"seg{seg}")
    _check(res)


@pytest.mark.tour
@pytest.mark.parametrize("seed", [7, 11, 23])
def test_full_tour(seed, tmp_path):
    res = dt.run_tour("orchard_tour", seed, runs_dir=str(tmp_path), run_id=f"tour-{seed}")
    _check(res)
    rows = {r["id"]: r for r in res["steps"]}
    assert rows["s10b_patch_refusal"]["status"] == "done"            # refused with the slope reason, as scripted
    card = res["card"]
    assert card["world"] == "orchard_demo" and card["counts"]["marks"] > 0
    trees = [json.loads(x) for x in open(os.path.join(str(tmp_path), f"tour-{seed}", "tree_records.jsonl"))]
    assert all(rec.validate_tree_record(t) == [] for t in trees)


@pytest.mark.tour
@pytest.mark.xfail(strict=True, reason="CR-WP27-02 / CR-WP9-02: frozen-layout tour is ~25-30 min sim (> 18 min)")
def test_full_tour_within_18_min(tmp_path):
    res = dt.run_tour("orchard_tour", 7, runs_dir=str(tmp_path), run_id="tour-time")
    assert res["tour"]["sim_s"] <= 1080.0


@pytest.mark.readiness
def test_ten_consecutive_passing_tours(tmp_path):
    seeds = [7, 11, 23, 7, 11, 23, 7, 11, 23, 7]
    for i, s in enumerate(seeds):
        res = dt.run_tour("orchard_tour", s, runs_dir=str(tmp_path), run_id=f"ready-{i}-{s}")
        assert res["tour"]["pass"], (i, s, res["tour"]["fail_reasons"])
