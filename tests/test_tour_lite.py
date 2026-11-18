"""WP9 ``tour_lite`` (OS 1.2, per PR, seed 7, orchard_lite): SCAN 2 trees, two-sided TREE 360 with 4 stations on a
hedgerow mango (worker actor), 1 mango clip, 1 papaya twist, DOCK/UNLOAD with Hindi job sheets, run card. One run per
module on SimDriver with every hook loaded (~100-160 s wall on the 2 CPU runner)."""
import json
import os
import time

import pytest

from atharv_nandi.core import demo_tour as dt
from atharv_nandi.core import job_sheet as js
from atharv_nandi.core import records as rec

pytestmark = pytest.mark.tour_lite


@pytest.fixture(scope="module")
def tour(tmp_path_factory):
    d = tmp_path_factory.mktemp("tour_lite")
    w0 = time.perf_counter()
    res = dt.run_tour("tour_lite", 7, runs_dir=str(d), run_id="tour_lite-7")
    res["_wall"] = time.perf_counter() - w0
    res["_dir"] = os.path.join(str(d), "tour_lite-7")
    print("\ntour_lite:", json.dumps({k: res["tour"][k] for k in ("pass", "fail_reasons", "sim_s", "wall_s",
                                                                    "safety")}, default=str))
    for r in res["steps"]:
        print(f"  {r['id']:<20} {r['status']:<8} {r['sim_s']:>6.1f} s {r['reason']}")
    return res


def step(res, sid):
    return next(r for r in res["steps"] if r["id"] == sid)


def test_tour_lite_passes_safety_gates_and_skip_limits(tour):
    t = tour["tour"]
    assert t["pass"], t["fail_reasons"]
    s = t["safety"]
    assert s["falls"] == 0 and s["protective_violations"] == 0 and s["motion_without_confirm"] == 0
    assert all(r["status"] == "done" for r in tour["steps"])
    assert tour["_wall"] <= 180.0


def test_scan_records_bands_and_triage(tour):
    k = step(tour, "l01_scan")["kpi"]
    assert [x["tree_id"] for x in k["trees"]] == ["L1-1", "L1-2"]
    assert all(len(x["bands"]) >= 2 for x in k["trees"])          # ring low + crown mid on the scanned side
    assert k["triage"]["fn"] == 0 and k["triage"]["fp"] == 0       # triage flags = truth in view (SimTruth)
    assert k["laterals_stepped"] in (0, None)


def test_tree360_two_sided_hedgerow(tour):
    r = step(tour, "l02_tree360")
    res = r["kpi"]["result"]
    assert res["status"] == "done" and res["mode"] == "two_sided"
    assert r["radius_reentries"] <= 1
    for b in res["bands_planned"]:
        sides = res["coverage"]["bands"][b]["sides"]
        assert len(sides) == 2 and min(sides.values()) >= 120.0, (b, sides)
    assert res["kpi"]["radius_rms_truth_m"] <= 0.10


def test_harvest_and_dock(tour):
    m = step(tour, "l03_harvest_mango")["kpi"]["results"]
    p = step(tour, "l04_harvest_papaya")["kpi"]["results"]
    assert [x["outcome"] for x in m] == ["picked"] and [x["outcome"] for x in p] == ["picked"]
    assert abs(m[0]["pedicel_error_mm"]) <= 5.0
    d = step(tour, "l05_dock_unload")["kpi"]
    assert d["crate"]["n"] == d["expected_items"] == 2 and d["crate"]["dropped"] == 0
    f = d["dock"]["final"]
    assert f["dist_m"] < 0.03 and abs(f["yaw_deg"]) < 5.0


def test_records_job_sheets_uplink_and_card(tour):
    d = tour["_dir"]
    assert tour["violations"] == [] and tour["jobsheet_missing"] == []
    trees = [json.loads(x) for x in open(os.path.join(d, "tree_records.jsonl"))]
    assert {"L1-1", "L1-2", "L2-1", "L2-3"} <= {t["tree_id"] for t in trees}
    for t in trees:
        assert rec.validate_tree_record(t) == [] and rec.json_size(t) <= rec.TREE_RECORD_MAX_BYTES
    marks = [json.loads(x) for x in open(os.path.join(d, "marks.jsonl"))]
    sheets = {f[:-5]: json.load(open(os.path.join(d, "jobsheets", f))) for f in os.listdir(os.path.join(d, "jobsheets"))
              if f.endswith(".json")}
    assert {"L1", "L2"} <= set(sheets) and js.missing_marks(marks, sheets) == []
    for f in os.listdir(os.path.join(d, "jobsheets")):
        if f.endswith(".html"):
            assert "<img" not in open(os.path.join(d, "jobsheets", f), encoding="utf-8").read()
    for ln in open(os.path.join(d, "uplink_kis_v1.jsonl")):
        u = json.loads(ln)
        assert rec.contract_violations(u["payload"]) == [], u["topic"]
    card = json.load(open(os.path.join(d, "run_card.json")))
    assert card["data_provenance"] == "simulated" and card["detector"] == "simtruth"
    assert "station_teleport" in card["assisted"] and card["counts"]["tree_records"] == len(trees)
    assert card["world"] == "orchard_lite" and card["season_banner"]


@pytest.mark.xfail(strict=True, reason="CR-WP9-02: measured tour_lite sim time (~520-590 s: TREE 360 two-sided "
                                      "stop_and_go ~170-215 s, DOCK/UNLOAD incl. the walk home ~160 s, walks at 0.18 "
                                      "m/s) is over the 5 min gate; project decision, not silently re-set")
def test_tour_lite_within_5_min_sim(tour):
    assert tour["tour"]["sim_s"] <= 300.0
