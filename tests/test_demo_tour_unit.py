"""WP9 tour state machine, pass definition, records plumbing and run card with a scripted backend (ROS-free,
MuJoCo-free). OS 1.2 tour pass definition: safety gates + per-step skip limits + no step halted."""
import json
import os

import pytest

from atharv_nandi.core import demo_tour as dt
from atharv_nandi.core import job_sheet as js
from atharv_nandi.core import records as rec
from atharv_nandi.core.tasks import common as tc


class Fake(dt.Backend):
    """Each step advances the clock by its budget and returns what ``script[step_id]`` says."""

    def __init__(self, script=None, safety=None):
        self.t = 0.0
        self.script = script or {}
        self._safety = {"falls": 0, "protective_violations": 0, "unmasked_person_px": 0,
                        "motion_without_confirm": 0}
        self._safety.update(safety or {})
        self.ran = []

    def context(self, step):
        return {"robot_xy": (0.0, 0.0)}

    def wait(self, s):
        self.t += s

    def safety(self):
        return dict(self._safety)

    def assisted(self):
        return ["fake_backend"]

    def run_step(self, step, ready, runner):
        self.ran.append(step.id)
        self.t += step.budget_s or step.t_s or 1.0
        fn = self.script.get(step.id)
        return fn(step, ready, runner) if fn else dt.StepResult()


def entry(field, tid):
    return next(t for t in field["trees"] if t["tree_id"] == tid)


def mark_for(field, tid, kind="fruit", z=1.5, tgt="x"):
    e = entry(field, tid)
    tree = {"tree_id": tid, "physical_tag": e["physical_tag"], "pos": (e["geo"]["x"], e["geo"]["y"])}
    return tc.make_mark(tree, kind, (e["geo"]["x"] + 0.3, e["geo"]["y"] + 0.2, z), "out_of_reach_marked",
                        stage="mature", tool_hint="pole_picker", target_id=f"{tid}:{tgt}")


def lite_script(field):
    def scan(step, ready, runner):
        e = entry(field, "L1-1")
        frag = rec.scan_fragment(e, "S", ["low", "mid"], [], {"height_m": 2.0, "width_m": 1.7}, ["tree360"], 2)
        return dt.StepResult(tree_fragments=[("scan", "L1-1", frag), ("scan", "L1-2", frag)], rows_visited=["L1"],
                             kpi={"trees": [{"tree_id": "L1-1", "seen": True, "bands": ["low", "mid"]},
                                            {"tree_id": "L1-2", "seen": True, "bands": ["low", "mid"]}],
                                  "triage": {"tp": 2, "fp": 0, "fn": 0}, "laterals_stepped": 0})

    def t360(step, ready, runner):
        frag = {"coverage": {"mode": "two_sided", "sides": ["-y", "+y"]}, "media": {"manifest_root": "ab" * 32,
                                                                                 "keyframe_paths": ["a.jpg"]},
                "next_actions": ["deep_scan"], "detector": ["simtruth"], "assisted": ["truth_radius"]}
        return dt.StepResult(tree_fragments=[("tree360", "L1-2", frag)], radius_reentries=1,
                             alerts=[{"id": "orbit_pause", "t": 70.0}, {"id": "stuck", "t": 75.0, "x": 1.0}])

    def mango(step, ready, runner):
        v1 = tc.harvest_record("L2-1", "L2-1:f07", "mango", "clip", "picked", pedicel_mm=19.8, pedicel_error_mm=-0.2)
        return dt.StepResult(harvest=[v1], marks=[mark_for(field, "L2-1", tgt="f09"), mark_for(field, "L2-1",
                                                                                               tgt="f10")],
                             kg_by_tree={"L2-1": 0.22}, skipped_targets=1,
                             kpi={"results": [dict(v1, crop="mango")], "crop": "mango", "marks_planned": 2})

    def dock(step, ready, runner):
        sheets = runner.job_sheets()
        return dt.StepResult(harvest_kind="unload", kg_by_tree={"L2-1": 0.22},
                             scripted_delays=[{"what": "charge compressed x240", "s": 0}],
                             kpi={"dock": {"final": {"dist_m": 0.016, "yaw_deg": -0.3}},
                                  "crate": {"n": 1, "per_tree": {"L2-1": {"crate_n": 1}}, "impact_mps_max": 0.37},
                                  "expected_items": 1, "jobsheets": {"rows": sorted(sheets)}})

    return {"l01_scan": scan, "l02_tree360": t360, "l03_harvest_mango": mango, "l05_dock_unload": dock}


@pytest.fixture
def lite(tmp_path):
    prog = dt.load_program("tour_lite")
    field = dt.load_field(prog.world)
    return prog, field, tmp_path


def run(prog, be, tmp_path, **kw):
    r = dt.TourRunner(prog, be, runs_dir=str(tmp_path), run_id="unit-run-7", **kw)
    return r, r.run()


def test_programs_load_and_select():
    lite = dt.load_program("tour_lite")
    assert lite.world == "orchard_lite" and lite.gate_s == 300
    assert [s.id for s in lite.steps][0] == "s00_ready" and lite.steps[-1].id == "s12_report"
    tour = dt.load_program("orchard_tour")
    assert tour.world == "orchard_demo" and tour.gate_s == 1080
    assert {"s10b_patch_refusal", "s11_dock_unload"} <= {s.id for s in tour.steps}
    assert tour.select(["s05_harvest_mango"]).steps[0].preset == "harvest_mango"
    with pytest.raises(ValueError):
        tour.select(["nope"])
    s10b = next(s for s in tour.steps if s.id == "s10b_patch_refusal")
    assert s10b.expect_refusal == "slope"


def test_lite_tour_passes_with_records_job_sheets_uplink_and_card(lite):
    prog, field, tmp = lite
    be = Fake(lite_script(field))
    r, res = run(prog, be, tmp)
    t = res["tour"]
    assert t["pass"], t["fail_reasons"]
    assert be.ran == [s.id for s in prog.steps]
    assert res["violations"] == []
    assert res["counts"]["tree_records"] == 3 and res["counts"]["marks"] == 2
    d = r.run_dir
    trees = [json.loads(x) for x in open(os.path.join(d, "tree_records.jsonl"))]
    assert all(rec.validate_tree_record(x) == [] for x in trees)
    l21 = next(x for x in trees if x["tree_id"] == "L2-1")
    assert l21["harvest"]["kg"] == 0.22 and "mark_jobs:2" in l21["next_actions"]
    assert res["jobsheet_missing"] == []
    assert {"L1", "L2"} <= set(os.listdir(os.path.join(d, "jobsheets")) and
                               {f.split(".")[0] for f in os.listdir(os.path.join(d, "jobsheets"))})
    up = [json.loads(x) for x in open(os.path.join(d, "uplink_kis_v1.jsonl"))]
    streams = {u["topic"].rsplit("/", 1)[-1] for u in up}
    assert {"tree", "mark", "harvest", "mission", "alert"} <= streams
    for u in up:
        assert rec.contract_violations(u["payload"]) == [], u
        assert u["payload"]["data_provenance"] == "simulated" and "/sim-nandi/" in u["topic"]
    persons = [u["payload"] for u in up if u["payload"].get("id") == "person_hold_count"]
    assert persons and persons[0]["count"] == 1 and "t_sim" not in persons[0]
    card = json.load(open(os.path.join(d, "run_card.json")))
    for k in ("run_id", "device_id", "date", "data_provenance", "detector", "assisted", "scripted_delays", "counts"):
        assert card[k] not in (None, ""), k
    assert card["data_provenance"] == "simulated" and "fake_backend" in card["assisted"]
    assert card["season_banner"] and card["tour"]["pass"]
    md = open(os.path.join(d, "run_card.md")).read()
    assert "Tour pass: PASS" in md and "SIMULATED" in md


@pytest.mark.parametrize("safety", [{"falls": 1}, {"protective_violations": 1}, {"unmasked_person_px": 4},
                                    {"motion_without_confirm": 1}])
def test_any_safety_gate_fails_the_tour(lite, safety):
    prog, field, tmp = lite
    _, res = run(prog, Fake(lite_script(field), safety=safety), tmp)
    assert not res["tour"]["pass"] and any(f.startswith("safety:") for f in res["tour"]["fail_reasons"])


def test_skip_and_reentry_limits(lite):
    prog, field, tmp = lite
    sc = lite_script(field)
    sc["l03_harvest_mango"] = lambda s, rd, rn: dt.StepResult(skipped_targets=2)
    sc["l02_tree360"] = lambda s, rd, rn: dt.StepResult(radius_reentries=2)
    _, res = run(prog, Fake(sc), tmp)
    f = res["tour"]["fail_reasons"]
    assert "skips:l03_harvest_mango=2" in f and "reentries:l02_tree360=2" in f


def test_halted_step_fails_and_a_crash_is_a_halt(lite):
    prog, field, tmp = lite
    sc = lite_script(field)

    def boom(step, ready, runner):
        raise RuntimeError("planner exploded")
    sc["l04_harvest_papaya"] = boom
    r, res = run(prog, Fake(sc), tmp)
    row = next(x for x in r.step_rows if x["id"] == "l04_harvest_papaya")
    assert row["status"] == "halted" and "planner exploded" in row["reason"]
    assert not res["tour"]["pass"]
    assert r.step_rows[-1]["status"] == "done"       # the tour continued to the report


def test_ready_refusal_stops_the_step_and_expected_refusal_passes(tmp_path):
    prog = dt.load_program("orchard_tour").select(["s01_scan_r1", "s10b_patch_refusal"])
    be = Fake()
    be.context = lambda step: {"robot_xy": (0.0, 0.0), "estop": step.id == "s01_scan_r1"}
    r, res = run(prog, be, tmp_path)
    rows = {x["id"]: x for x in r.step_rows}
    assert rows["s01_scan_r1"]["status"] == "refused" and "estop" in rows["s01_scan_r1"]["reason"]
    assert rows["s10b_patch_refusal"]["status"] == "done" and "slope" in rows["s10b_patch_refusal"]["reason"]
    assert be.ran == []                               # nothing ran: one refused, one refused as expected
    assert res["tour"]["fail_reasons"] == ["refused:s01_scan_r1:estop"]


def test_expected_refusal_that_is_not_raised_is_a_contradiction(tmp_path):
    prog = dt.load_program("orchard_tour").select(["s10b_patch_refusal"])
    prog.steps[0].ctx = {"cross_slope_deg": 0.0}        # the gate table now allows it: the script contradicts it
    r, res = run(prog, Fake(), tmp_path)
    assert r.step_rows[0]["status"] == "halted" and "expected refusal" in r.step_rows[0]["reason"]


def test_abort_marks_the_rest_aborted(lite):
    prog, field, tmp = lite
    sc = lite_script(field)

    def abort_now(step, ready, runner):
        runner.abort("operator")
        return dt.StepResult()
    sc["l02_tree360"] = abort_now
    r, res = run(prog, Fake(sc), tmp)
    st = [x["status"] for x in r.step_rows]
    assert st[:3] == ["done", "done", "done"] and set(st[3:]) == {"aborted"}
    assert not res["tour"]["pass"]


def test_status_payload_and_publish(lite):
    prog, field, tmp = lite
    pub = []
    r, _ = run(prog, Fake(lite_script(field)), tmp, publish=lambda t, p: pub.append((t, p)))
    topics = {t for t, _ in pub}
    assert {"/nandi/demo/state", "/nandi/demo/events", "/nandi/mark", "/nandi/jobsheet", "/nandi/tree/record",
            "/nandi/kpi", "/nandi/kpi/scopes"} <= topics
    st = r.status()
    assert st["schema"] == "nandi.demo_state/1" and st["pass"] is True and st["pending"] == []
    kpi = next(p for t, p in pub if t == "/nandi/kpi")
    assert {"run_id", "device_id", "date", "data_provenance", "detector", "assisted"} <= set(kpi)


def test_route_avoids_canopies_and_alley_lines():
    f = dt.load_field("orchard_lite")
    path = dt.plan_route(f, (0.0, 0.0), (6.8, 4.5))
    pts = [(0.0, 0.0)] + path
    for e in f["trees"]:
        c = (e["geo"]["x"], e["geo"]["y"])
        for a, b in zip(pts, pts[1:]):
            assert dt._seg_point_dist(a, b, c) >= e["canopy_r_m"] + 0.45 - 1e-6, (e["tree_id"], a, b)
    assert dt.plan_route(f, (0.5, 0.5), (3.0, 0.5)) == [(3.0, 0.5)]
    assert dt.alley_lines(f) == [0.5, 4.5, 8.5]
    demo = dt.load_field("orchard_demo")
    assert dt.row_alley_y(demo, "R1", "north") == 5.0 and dt.row_alley_y(demo, "OT", "south") == -5.5


def test_scan_flags_and_triage_score():
    leads = [{"unit": "trunk", "score": 0.9}, {"unit": "per_panicle", "score": 0.3}, {"unit": "per_leaf",
                                                                                     "score": 0.7}]
    assert dt.scan_flags(leads, 2, 0) == ["harvest_window", "leaf_scan", "trunk_scan"]
    assert dt.scan_flags(leads, 0, 1, threshold=0.2) == ["leaf_scan", "prune", "tree360", "trunk_scan"]
    assert dt.triage_score(["a", "b"], ["b", "c"]) == {"tp": 1, "fp": 1, "fn": 1}


def test_job_sheet_rows_follow_the_visited_rows(lite):
    prog, field, tmp = lite
    r, res = run(prog, Fake(lite_script(field)), tmp)
    sheets = r.job_sheets()
    assert set(r.rows_visited) <= set(sheets) and js.missing_marks(r.marks, sheets) == []


def test_nandi_kpi_card_reads_the_tour_card_and_manifest_verifies(lite):
    import sys
    from tools import nandi_kpi
    from atharv_nandi.core import manifest as mf
    prog, field, tmp = lite
    r, res = run(prog, Fake(lite_script(field)), tmp)
    card = nandi_kpi.run_card(r.run_dir)
    assert card["run_id"] == "unit-run-7" and all(k["run_id"] and k["device_id"] and k["date"] for k in card["kpis"])
    assert mf.verify(r.run_dir)[0]
    assert nandi_kpi.main([r.run_dir, "--card", "--no-write"]) == 0
    os.remove(os.path.join(r.run_dir, "run_card.json"))
    with pytest.raises(ValueError):
        nandi_kpi.run_card(r.run_dir)


def test_demo_run_payloads():
    import atharv_nandi.nodes.demo_tour_node as node
    assert node.parse_run('{"steps": "full"}') == ("run", {"program": "tour_lite", "steps": None, "seed": 7,
                                                           "transit": "walk"})
    assert node.parse_run('{"program": "orchard_tour", "steps": ["s05_harvest_mango"], "seed": 11}')[1]["steps"] == \
        ["s05_harvest_mango"]
    assert node.parse_run('{"cmd": "abort"}') == ("abort", None)
    for bad in ("nope", "[]", '{"program": "x"}', '{"steps": "half"}', '{"steps": ["zz"]}', '{"seed": -1}',
                '{"seed": true}', '{"transit": "fly"}'):
        with pytest.raises(ValueError):
            node.parse_run(bad)
    job = node.TourJob(lambda t, p: None)
    assert job.state()["state"] == "idle" and not job.busy
