"""WP9 review fixes (ROS-free, MuJoCo-free): timed safety gates decide the tour, abort / pause inside a step, the
demo node's load window, record errors halt a step instead of crashing the report, skipped preset steps fail,
the refusal probe, the O-18 gust clock, harvest/2 stage and mass, the personal-key scan, report-only rows that were
not measured, Hindi job-sheet details, tele/kpi and per-event uplink timestamps."""
import json
import os
import threading
import time
import types

import pytest

from atharv_nandi.core import demo_tour as dt
from atharv_nandi.core import job_sheet as js
from atharv_nandi.core import records as rec
from atharv_nandi.core import scope_kpi as sk
from atharv_nandi.core.tasks import common as tc

from test_demo_tour_unit import Fake, lite_script, run  # noqa: F401  (fixtures and helpers of the unit tier)


@pytest.fixture
def lite(tmp_path):
    prog = dt.load_program("tour_lite")
    return prog, dt.load_field(prog.world), tmp_path


def recovery(tmp_path, safety, script=None, steps=("s10a_gust", "s10e_estop")):
    prog = dt.load_program("orchard_tour").select(list(steps))
    be = Fake(script or {}, safety=dict({"gust_stance_s": None, "crown_stow_s": None, "estop_stationary_s": None},
                                        **safety))
    return run(prog, be, tmp_path)


# ------------------------------------------------------------------ timed safety gates
def test_estop_that_never_measured_stationary_fails_the_tour(tmp_path):
    _, res = recovery(tmp_path, {"gust_stance_s": 0.3, "crown_stow_s": 3.0, "estop_stationary_s": None})
    assert not res["tour"]["pass"]
    assert "safety:estop_stationary_s=unmeasured" in res["tour"]["fail_reasons"]


@pytest.mark.parametrize("safety,reason", [
    ({"gust_stance_s": 0.9, "crown_stow_s": 3.0, "estop_stationary_s": 0.1}, "safety:gust_stance_s=0.9>0.5"),
    ({"gust_stance_s": 0.3, "crown_stow_s": 20.0, "estop_stationary_s": 0.1}, "safety:crown_stow_s=20.0>8.5"),
    ({"gust_stance_s": 0.3, "crown_stow_s": 3.0, "estop_stationary_s": 3.0}, "safety:estop_stationary_s=3.0>0.5"),
])
def test_timed_safety_gate_over_its_limit_fails_the_tour(tmp_path, safety, reason):
    _, res = recovery(tmp_path, safety)
    assert not res["tour"]["pass"] and reason in res["tour"]["fail_reasons"]
    assert any(g.startswith("safety.") for g in res["card"]["failed_gates"])


def test_timed_gates_inside_their_limits_pass(tmp_path):
    _, res = recovery(tmp_path, {"gust_stance_s": 0.3, "crown_stow_s": 3.0, "estop_stationary_s": 0.1})
    assert res["tour"]["pass"], res["tour"]["fail_reasons"]


def test_timed_gate_not_required_when_its_step_did_not_run(lite):
    prog, field, tmp = lite
    be = Fake(lite_script(field), safety={"gust_stance_s": None, "crown_stow_s": None, "estop_stationary_s": None})
    _, res = run(prog, be, tmp)
    assert res["tour"]["pass"], res["tour"]["fail_reasons"]


# ------------------------------------------------------------------ abort / pause inside a step
def test_abort_inside_a_step_aborts_that_step(lite):
    prog, field, tmp = lite
    sc = lite_script(field)

    def long_step(step, ready, runner):
        runner.abort("operator")
        runner.check_control()            # what SimBackend's 10 Hz driver task does
        raise AssertionError("not reached")
    sc["l02_tree360"] = long_step
    r, res = run(prog, Fake(sc), tmp)
    st = {x["id"]: x for x in r.step_rows}
    assert st["l02_tree360"]["status"] == "aborted" and "inside the step" in st["l02_tree360"]["reason"]
    assert all(st[s.id]["status"] == "aborted" for s in prog.steps[2:])
    assert not res["tour"]["pass"] and res["card"]          # the report still comes out


def test_pause_inside_a_step_blocks_until_resume(lite):
    prog, field, tmp = lite
    r = dt.TourRunner(prog, Fake(lite_script(field)), runs_dir=str(tmp), run_id="p", write=False)
    r.pause()
    threading.Timer(0.25, r.resume).start()
    t0 = time.perf_counter()
    r.check_control()
    assert time.perf_counter() - t0 >= 0.2


def test_sim_backend_raises_the_abort_from_its_driver_task():
    calls = []
    fake_runner = types.SimpleNamespace(check_control=lambda: calls.append(1))
    be = types.SimpleNamespace(_ctl_runner=fake_runner)
    dt.SimBackend._control_task(be, None)
    be._ctl_runner = None
    dt.SimBackend._control_task(be, None)
    assert calls == [1]
    assert issubclass(dt.StepAborted, BaseException) and not issubclass(dt.StepAborted, Exception)


# ------------------------------------------------------------------ demo node load window
def test_tour_job_queues_commands_while_the_world_loads_and_reports_load_errors(lite):
    import atharv_nandi.nodes.demo_tour_node as node
    prog, field, tmp = lite
    events = []
    gate = threading.Event()

    def factory(world, seed, transit):
        gate.wait(5.0)
        return Fake(lite_script(field))

    os.environ["NANDI_RUNS_DIR"] = str(tmp)
    try:
        job = node.TourJob(lambda t, p: events.append((t, p)), backend_factory=factory)
        assert job.control("abort") == (False, "no tour running")
        assert job.start({"program": "tour_lite", "steps": None, "seed": 7, "transit": "walk"})
        assert job.state()["state"] == "loading" and job.runner is None
        assert job.control("abort") == (True, "queued")
        gate.set()
        job.thread.join(10.0)
        assert job.last is not None
        assert {r["status"] for r in job.last["steps"]} == {"aborted"}
        assert job.control("pause") == (False, "no tour running")       # never reaches the finished tour

        def broken(world, seed, transit):
            raise RuntimeError("model load failed")
        job2 = node.TourJob(lambda t, p: events.append((t, p)), backend_factory=broken)
        job2.start({"program": "tour_lite", "steps": None, "seed": 7, "transit": "walk"})
        job2.thread.join(5.0)
        err = [p for t, p in events if p.get("event") == "tour_error"]
        assert err and "model load failed" in err[-1]["error"]
        assert job2.state()["state"] == "error"
    finally:
        os.environ.pop("NANDI_RUNS_DIR", None)


# ------------------------------------------------------------------ record errors, skipped presets
def test_record_merge_error_halts_the_step_and_the_report_still_comes_out(tmp_path):
    prog = dt.load_program("orchard_tour").select(["s03_leaf_scan", "s12_report"])
    bad = {"symptoms": [{"class": "anthracnose", "band": "mid", "clock": 3, "unit": "per_leaf"}]}   # no stage
    tid = prog.steps[0].targets["tree_ids"][0]
    sc = {"s03_leaf_scan": lambda s, rd, rn: dt.StepResult(tree_fragments=[("leaf_scan", tid, bad)])}
    r, res = run(prog, Fake(sc), tmp_path)
    row = r.step_rows[0]
    assert row["status"] == "halted" and row["reason"].startswith("merge_error:ValueError")
    assert r.step_rows[-1]["status"] == "done" and res["card"] and not res["tour"]["pass"]
    assert os.path.isfile(os.path.join(r.run_dir, "run_card.md"))


def test_tree_record_too_big_after_trimming_is_a_violation_not_a_crash(lite):
    prog, field, tmp = lite
    sc = lite_script(field)

    def bloat(step, ready, runner):
        frag = {"units": {f"u{i}": "x" * 200 for i in range(120)}}
        return dt.StepResult(tree_fragments=[("leaf_scan", "L2-1", frag)])
    sc["l04_harvest_papaya"] = bloat
    r, res = run(prog, Fake(sc), tmp)
    assert any(v.startswith("L2-1:build:ValueError") for v in res["violations"])
    assert res["card"] and os.path.isfile(os.path.join(r.run_dir, "run_card.json"))


def test_a_preset_step_the_backend_skips_fails_the_tour(lite):
    prog, field, tmp = lite
    sc = lite_script(field)
    sc["l03_harvest_mango"] = lambda s, rd, rn: dt.StepResult(status="skipped", reason="no tour runner")
    _, res = run(prog, Fake(sc), tmp)
    assert not res["tour"]["pass"]
    assert "skipped:l03_harvest_mango:no tour runner" in res["tour"]["fail_reasons"]


# ------------------------------------------------------------------ refusal probe (OS 1.2 step 10 b)
@pytest.mark.parametrize("slope,status", [(10.0, "done"), (0.5, "halted")])
def test_expected_refusal_is_checked_again_on_the_measured_context(tmp_path, slope, status):
    prog = dt.load_program("orchard_tour").select(["s10b_patch_refusal"])
    be = Fake()
    probed = []

    def probe(step, runner):
        probed.append(step.id)
        return {"cross_slope_deg": slope}, [f"probe {slope}"]
    be.refusal_probe = probe
    r, _ = run(prog, be, tmp_path)
    row = r.step_rows[0]
    assert probed == ["s10b_patch_refusal"] and row["status"] == status, row
    if status == "done":
        assert "measured on the spot" in row["reason"]


def test_gust_clock_starts_at_the_truth_onset():
    field = dt.load_field("orchard_demo")
    ev = next(e for e in field["events"] if e["id"] == "gust_11")
    g = ev["script"]["gusts"][0]
    onset, how = dt.SimBackend.gust_onset(types.SimpleNamespace(field=field), "gust_11")
    assert how == "WP7 truth gust criterion"
    assert g["t0"] <= onset <= g["t0"] + g["rise_s"]


# ------------------------------------------------------------------ harvest/2 stage and mass
def test_harvest_v2_carries_stage_and_mass(lite):
    prog, field, tmp = lite
    sc = lite_script(field)

    def mango(step, ready, runner):
        v1 = tc.harvest_record("L2-1", "L2-1:f07", "mango", "clip", "picked", pedicel_mm=19.8, pedicel_error_mm=-0.2)
        v1.update(stage="mature", mass_kg=0.24)
        return dt.StepResult(harvest=[v1], kpi={"results": [dict(v1, crop="mango")], "crop": "mango"})
    sc["l03_harvest_mango"] = mango
    r, res = run(prog, Fake(sc), tmp)
    h = r.harvest[0]
    assert h["stage"] == "mature" and h["mass_g"] == 240.0 and rec.validate_harvest(h) == []


# ------------------------------------------------------------------ personal keys, person-caused alerts
@pytest.mark.parametrize("key", ["people_n", "humans", "operator_id", "nearby_worker_xy", "persons", "aadhar",
                                 "aadhaar_no", "first_name", "userName", "face_id", "track_id", "worker_name",
                                 "name", "phone"])
def test_personal_keys_are_caught(key):
    assert any(v.startswith("personal_key") for v in rec.contract_violations({key: 1})), key


@pytest.mark.parametrize("key", ["cultivar_name", "tool_holder", "filename", "tracking_ok", "face_count",
                                 "person_hold_count", "tree_id", "physical_tag", "stage"])
def test_harmless_keys_are_not_flagged(key):
    assert not any(v.startswith("personal_key") for v in rec.contract_violations({key: 1})), key


def test_an_alert_caused_by_a_person_is_only_counted():
    u = rec.Uplink()
    assert u.alert(1, {"id": "protective_stop", "reason": "person", "t": 123.45}) is None
    assert u.alert(1, {"id": "protective_stop", "class": "worker", "t": 10.0}) is None
    _, p = u.alert(1, {"id": "protective_stop", "reason": "animal", "t": 5.0})
    assert p["id"] == "obstacle_stop"
    agg = u.person_aggregates(1)
    assert len(agg) == 1 and agg[0][1]["count"] == 2 and "t_sim" not in agg[0][1]


# ------------------------------------------------------------------ report-only rows that were not measured
def test_unmeasured_person_pixels_and_unaudited_cuts_are_report_only():
    t = sk.KpiTable("r", "nandi-sim", "2026-11-18")
    sk.safety_kpis(t, {"falls": 0, "protective_violations": 0, "unmasked_person_px": 0, "motion_without_confirm": 0,
                       "person_pixels_measured": False})
    row = next(r for r in t.rows if r["metric"] == "safety.unmasked_person_px")
    assert row["pass"] is None and "not measured" in row["note"]
    sk.prune_kpis(t, [{"results": [{"outcome": "cut"}], "keep_cuts": None, "green_cuts": None}])
    for m in ("prune.keep_cuts", "prune.green_leaf_cuts"):
        r = next(x for x in t.rows if x["metric"] == m)
        assert r["pass"] is None and r["value"] is None and "not audited" in r["note"]
    t2 = sk.KpiTable("r", "nandi-sim", "2026-11-18")
    sk.safety_kpis(t2, {"unmasked_person_px": 0, "person_pixels_measured": True})
    assert next(r for r in t2.rows if r["metric"] == "safety.unmasked_person_px")["pass"] is True
    md = sk.format_card(sk.run_card(ident=rec.identity("r"), date="2026-11-18", world="w", season="", seed=7,
                                    program="p", detector="simtruth", mode="assisted", assisted=[],
                                    scripted_delays=[], table=t, counts={}, tour={"pass": True}))
    line = next(x for x in md.splitlines() if "safety.unmasked_person_px" in x)
    assert "| ok |" not in line


# ------------------------------------------------------------------ Hindi job sheet details
def test_banana_marks_show_leaf_id_sigatoka_hands_and_sucker_type_in_hindi():
    field = dt.load_field("orchard_demo")
    e = next(t for t in field["trees"] if t["crop"] == "banana")
    tree = {"tree_id": e["tree_id"], "physical_tag": e["physical_tag"], "pos": (e["geo"]["x"], e["geo"]["y"])}
    p = (e["geo"]["x"] + 0.3, e["geo"]["y"], 1.4)
    leaf = tc.make_mark(tree, "sanitation_deleaf", p, "human_cut", tool_hint="sickle", target_id="B-2:leaf3",
                        extra={"sigatoka_pct": 35})
    bunch = tc.make_mark(tree, "bunch", p, "human_cut", stage="mature", tool_hint="human_cut", target_id="b1",
                         extra={"hands_n_lead": 8, "shooting_date": "2026-08-20"})
    suck = tc.make_mark(tree, "sucker", (p[0], p[1], 0.2), "human_cut", tool_hint="human_cut", target_id="s1",
                        extra={"sucker_type": "sword"})
    sheets = js.build_job_sheets([leaf, bunch, suck], field["trees"], rec.identity("r"), rows=[e["row_id"]])
    lines = sheets[e["row_id"]]["lines"]
    hi = " | ".join(ln["notes_hi"] for ln in lines)
    en = " | ".join(ln["notes_en"] for ln in lines)
    assert "B-2:leaf3" in hi and "सिगाटोका 35%" in hi and "8 हत्थे" in hi and "तलवार पुत्ती" in hi
    assert "अवस्था: पका" in hi and "mature" not in hi
    assert "Sigatoka 35 %" in en and "shooting date 2026-08-20" in en
    for reason in ("not_in_mvp", "keep_wood", "operator", "sanitation", "desucker", "bunch"):
        assert reason in js.NOTE


# ------------------------------------------------------------------ uplink: tele/kpi and event timestamps
def test_uplink_has_tele_kpi_and_event_timestamps(lite):
    prog, field, tmp = lite
    r, res = run(prog, Fake(lite_script(field)), tmp)
    up = [json.loads(x) for x in open(os.path.join(r.run_dir, "uplink_kis_v1.jsonl"))]
    kpi = [u["payload"] for u in up if u["topic"].endswith("/kpi")]
    assert len(kpi) == 1 and kpi[0]["rows"] and "pass" in kpi[0] and rec.contract_violations(kpi[0]) == []
    assert res["counts"]["uplink_messages"] == len(up) == res["card"]["counts"]["uplink_messages"]
    start = int(r.now.timestamp() * 1000)
    alerts = [u["payload"] for u in up if u["payload"].get("id") == "stuck"]
    assert alerts and alerts[0]["ts"] == start + 75000
    assert len({u["payload"]["ts"] for u in up}) > 1
