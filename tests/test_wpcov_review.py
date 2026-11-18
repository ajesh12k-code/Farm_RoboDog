"""WPCOV code review fixes, each with the failure it closes (ROS-free, unit tier).

* safety_logic: NaN / inf operator commands, NaN tilt (IMU fault) and a NaN geofence margin failed open;
* safety_fields: a NaN speed dropped the d_stop term from the protective field;
* tools: the blade / jaw close gate moved out of the MuJoCo hook into the pure ``tool_changer.close_gate`` (one
  ``stability_ok`` rule shared with the WP8 executor; a NaN margin used to pass); ``swap_gate`` NaN inputs;
* ready_check: the noVNC + actors interlock is a named pure function (mutation target);
* web_auth: ``AuditLog.record`` dedupe check-and-set is one critical section (concurrent bridge threads);
* mission_logic: ``PresetQueue.add`` mission-id suffix could collide with a queued id.
"""
import json
import math
import os
import threading

import pytest

from atharv_nandi.core import mission_logic as ml
from atharv_nandi.core import ready_check as rc
from atharv_nandi.core import safety_fields as sf
from atharv_nandi.core import safety_logic as sl
from atharv_nandi.core import stability as st
from atharv_nandi.core import web_auth as wa
from atharv_nandi.core.tools import tool_changer as tc

NAN, INF = float("nan"), float("inf")


# ------------------------------------------------------------------ safety_logic
def test_speed_cap_non_finite_commands_fail_closed():
    cap = sl.SpeedCap(max_speed_scale=0.5, max_cmd=0.4)
    assert cap.clamp_cmd(0.1, 0.0, NAN) == (0.1, 0.0, 0.0)          # was (0.1, 0.0, +0.4): full yaw rate
    assert cap.clamp_cmd(INF, 0.0, 0.0) == (0.0, 0.0, 0.0)          # was (nan, 0.0, 0.0)
    assert cap.clamp_cmd(NAN, -INF, 0.2) == (0.0, 0.0, 0.2)
    vx, vy, yaw = cap.clamp_cmd(3.0, 4.0, -9.0)
    assert math.hypot(vx, vy) == pytest.approx(0.4) and yaw == -0.4
    assert cap.clamp_speed_scale(NAN) == 0.0                        # was nan
    assert cap.clamp_speed_scale(0.9) == 0.5 and cap.clamp_speed_scale(0.2) == 0.2


def test_tilt_detector_nan_tilt_is_a_fall_cause():
    f = sl.TiltFallDetector(tilt_deg=35.0, sustain_s=0.3)
    assert not f.update(NAN, False, 0.0)                            # sustain still applies
    assert f.update(NAN, False, 0.35)
    assert f.cause_present(NAN, False)                              # re-arm refused on a broken IMU
    assert not f.cause_present(3.0, False)
    f.reset()
    assert not f.update(10.0, False, 1.0)


def _core():
    cfg = sl.load_safety_config("/nonexistent.yaml")
    cfg["geofence"]["polygon"] = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    return sl.SafetyCore(cfg, wall_clock=lambda: 1000.0)


def test_nan_pose_latches_and_refuses_the_rearm():
    core = _core()
    core.engage_operator(0.0)
    core.update_pose(0.1, (NAN, 0.0), 0.0, False)
    assert math.isnan(core.margin) and not core.geofence_ok
    ok, msg = core.reset()
    assert not ok and "geofence" in msg                             # NaN < need was False: the re-arm passed
    core.update_pose(0.2, (0.0, 0.0), 0.0, False)
    assert core.reset()[0]


def test_nan_tilt_refuses_the_rearm():
    core = _core()
    core.engage_operator(0.0)
    core.update_pose(0.1, (0.0, 0.0), NAN, False)
    assert "fall" in core.causes_present(0.1)
    core.update_pose(0.2, (0.0, 0.0), 2.0, False)
    assert core.reset()[0]


def test_field_latch_engages_the_supervisor_latch():
    core = _core()
    assert core.update_fields(0.0, {"latched": True, "latch_reason": "protective", "protective_clear": False}) \
        == "protective"
    assert core.latch.estopped and not core.reset(0.1)[0]


# ------------------------------------------------------------------ safety_fields
def test_nan_speed_keeps_the_stopping_distance_in_the_protective_field():
    capped = sf.field_offsets(sf.MODE_WALK, None)[sf.FIELD_PROTECTIVE]["front"]
    nan = sf.field_offsets(sf.MODE_WALK, NAN)[sf.FIELD_PROTECTIVE]["front"]
    still = sf.field_offsets(sf.MODE_WALK, 0.0)[sf.FIELD_PROTECTIVE]["front"]
    assert nan == pytest.approx(capped) and nan > still               # was == still (d_stop(nan) = 0)
    back = sf.field_offsets(sf.MODE_WALK, -INF)[sf.FIELD_PROTECTIVE]
    assert back["rear"] == pytest.approx(sf.WALK_PROTECTIVE["rear"] + sf.d_stop(sf.WALK_SPEED_CAP_MPS))


# ------------------------------------------------------------------ tools: close gate, stability, swap gate
CLOSE_OK = {"work_field": {"person": False}, "stability": {"ok": True, "margin_m": 0.08}}


@pytest.mark.parametrize("kind,value,closing", [("blade", 1.0, True), ("blade", 0.5, True), ("blade", 0.49, False),
                                                ("twist", 0.0, True), ("twist", 0.2, False)])
def test_close_gate_only_judges_closing_commands(kind, value, closing):
    far = 1.0                                                        # no target in reach: refuses a close
    assert tc.close_gate(kind, value, CLOSE_OK, far) == ("no_target" if closing else None)


def test_close_gate_reasons_in_order():
    near = 0.05
    assert tc.close_gate("blade", 1.0, CLOSE_OK, near) is None
    assert tc.close_gate("blade", 1.0, {"work_field": {"person": True}, "stability": {"ok": False}}, near) \
        == "person_in_work_field"
    assert tc.close_gate("blade", 1.0, {"stability": {"ok": False}}, near) == "stability"
    assert tc.close_gate("blade", 1.0, {"stability": {"margin_m": 0.01}}, near) == "stability"
    assert tc.close_gate("blade", 1.0, {"stability": {"margin_m": NAN}}, near) == "stability"     # NaN passed
    assert tc.close_gate("blade", 1.0, {"stability": {"margin_m": "x"}}, near) == "stability"     # raised
    assert tc.close_gate("blade", 1.0, None, near) is None                                        # nothing published
    assert tc.close_gate("blade", 1.0, CLOSE_OK, NAN) == "no_target"
    for bad in (True, None, "1", NAN, INF):
        assert tc.close_gate("blade", bad, {"stability": {"ok": False}}, near) is None            # the value check


def test_close_gate_measures_the_target_only_when_needed():
    calls = []

    def dist():
        calls.append(1)
        return 0.05
    assert tc.close_gate("blade", 0.0, CLOSE_OK, dist) is None and not calls                     # opening
    assert tc.close_gate("blade", 1.0, {"stability": {"ok": False}}, dist) == "stability" and not calls
    assert tc.close_gate("blade", 1.0, CLOSE_OK, dist) is None and calls == [1]


def test_stability_ok_is_one_rule():
    assert tc.stability_ok(None) and tc.stability_ok({}) and tc.stability_ok({"stability": "bad"})
    assert not tc.stability_ok({"stability": {"ok": True, "margin_m": None}})   # published None = non-finite (fixer)
    assert tc.stability_ok({"stability": {"ok": True}})                            # no margin key: not gated
    assert tc.stability_ok({"stability": {"margin_m": tc.MIN_MARGIN_M}})
    assert not tc.stability_ok({"stability": {"margin_m": tc.MIN_MARGIN_M - 1e-6}})
    assert not tc.stability_ok({"stability": {"margin_m": True}})
    assert not tc.stability_ok({"stability": {"ok": False, "margin_m": 1.0}})


def test_executor_reads_the_same_stability_rule():
    from atharv_nandi.core.tasks import sim_executor as sx
    assert sx.STABILITY_MIN_MARGIN_M == tc.MIN_MARGIN_M
    ex = object.__new__(sx.SimTaskExecutor)              # stability_ok reads only .values
    for vals, want in (({}, True), ({"stability": {"margin_m": NAN}}, False),
                       ({"stability": {"margin_m": "x"}}, False), ({"stability": {"margin_m": 0.05}}, True)):
        ex.values = vals
        assert ex.stability_ok() is want, vals


def test_swap_gate_nan_inputs_refuse():
    assert tc.swap_gate(tc.Interlocks(crown_ext_m=NAN)).reason == "crown_up"
    assert tc.swap_gate(tc.Interlocks(margin_m=NAN)).reason == "low_margin"
    assert tc.swap_gate(tc.Interlocks(margin_m=None))                  # unknown = not gated (documented)


def test_static_margin_gate():
    assert st.gate_static_margin(0.05)
    for bad in (0.0, NAN, -INF):
        assert not st.gate_static_margin(bad)


# ------------------------------------------------------------------ ready_check: noVNC + actors
def test_novnc_actor_conflict():
    assert rc.novnc_actor_conflict(True, True)
    assert not rc.novnc_actor_conflict(True, False) and not rc.novnc_actor_conflict(False, True)


def test_ready_check_refuses_novnc_with_actors():
    reg = __import__("atharv_nandi.core.presets", fromlist=["Registry"]).Registry.load()
    import os
    field = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config", "fields", "orchard_demo.json"),
                           encoding="utf-8"))
    ctx = dict(season_profile="demo_all_fruiting", month=11, frames_age_h=0.1)
    bad = rc.check(reg.get("scan"), rc.ReadyContext.nominal(novnc=True, actors=True, **ctx), {"row_ids": ["R1"]},
                   None, field)
    good = rc.check(reg.get("scan"), rc.ReadyContext.nominal(novnc=False, actors=True, **ctx), {"row_ids": ["R1"]},
                    None, field)
    assert "novnc_actors" in bad.reasons and bad.level == rc.NO_GO
    assert "novnc_actors" not in good.reasons


# ------------------------------------------------------------------ web_auth: audit dedupe under threads
def test_audit_dedupe_is_atomic_under_threads(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = wa.AuditLog(str(path))
    typ = sorted(wa.AUDITED_TYPES - {"service"})[0] if "service" in wa.AUDITED_TYPES else sorted(wa.AUDITED_TYPES)[0]
    n, barrier, results = 16, threading.Barrier(16), []

    def go():
        barrier.wait()
        results.append(log.record(1.0, "c1", "operator", typ, "same", [1, 2]))
    th = [threading.Thread(target=go) for _ in range(n)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert results.count(True) == 1                                  # one row for one command inside the window
    assert len(path.read_text().splitlines()) == 1
    log.forget("c1")
    assert log.record(1.1, "c1", "operator", typ, "same", [1, 2])


def test_audit_write_errors_are_counted_not_raised(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    log = wa.AuditLog(str(blocker / "sub" / "audit.jsonl"))          # parent is a file: makedirs fails
    typ = sorted(wa.AUDITED_TYPES)[0]
    assert log.record(0.0, "c", "operator", typ, "n", 1) is True
    assert log.errors == 1


# ------------------------------------------------------------------ mission_logic: unique queued mission ids
def test_queue_mission_ids_stay_unique():
    q = ml.PresetQueue()
    est = {"t_s": 10.0}

    def add(mid):
        return q.add({"mission_id": mid, "plan": {"preset": "scan"}}, est, (0, 0), (0, 0), level="go")
    add("m")
    add("m-3")
    add("m")                    # the old rule gave "m-<n+1>" = "m-3": already queued
    add("m")
    ids = [i.cmd["mission_id"] for i in q.items]
    assert ids == ["m", "m-3", "m-4", "m-5"], ids
    q.remove(q.items[0].qid)
    add("m")                    # "m" is free again
    assert q.items[-1].cmd["mission_id"] == "m"


def test_loaded_safety_config_never_aliases_the_defaults():
    before = json.dumps(sl.DEFAULT_CONFIG, sort_keys=True)
    cfg = sl.load_safety_config("/nonexistent.yaml")
    cfg["geofence"]["polygon"] = [[0, 0], [1, 0], [1, 1]]             # used to edit DEFAULT_CONFIG in place
    cfg["geofence"]["polygon"].append([0, 1])
    merged = sl._merge(sl.DEFAULT_CONFIG, {"speed": {"max_cmd": 0.2}, "extra": {"a": [1]}})
    merged["extra"]["a"].append(2)
    assert json.dumps(sl.DEFAULT_CONFIG, sort_keys=True) == before
    assert merged["speed"]["max_cmd"] == 0.2 and "max_speed_scale" in merged["speed"]


# ------------------------------------------------------------------ recovery: e-stop confirmation backstop
def test_estop_recovery_backstop_without_both_confirmations():
    """The ``confirmed`` backstop on its own: a recovery forced into the checklist (or re-arm) step without the two
    ordered confirmations goes back to await_local and never asks for a stand-up."""
    from atharv_nandi.core import recovery_behaviours as rb
    for step in ("checklist", "rearm"):
        b = rb.EstopRecovery()
        o = rb.Obs(t=1.0, x=0.0, y=0.0, yaw=0.0, holding=True)
        b.start(o)
        b.confirms["operator"] = 1.2                                # an operator confirm without the local one
        b._to(step, 1.2)
        a = b.update(rb.Obs(t=1.3, x=0.0, y=0.0, yaw=0.0, holding=True))
        assert not a.stand_up and a.lock, step
        assert not b.confirmed
        if step == "checklist":
            assert b.step == "await_local" and b.log[-1][2] == "confirm_lost"
        else:
            assert b.step == "rearm"                                  # waits; no stand-up without both
    b = rb.EstopRecovery()
    b.start(rb.Obs(t=1.0, x=0.0, y=0.0, yaw=0.0, holding=True))
    assert b.confirm(1.1, "local") == (True, "local confirmed")
    assert b.confirm(1.2, "operator") == (True, "operator confirmed") and b.confirmed
    a = b.update(rb.Obs(t=1.3, x=0.0, y=0.0, yaw=0.0, holding=True))
    a = b.update(rb.Obs(t=1.4, x=0.0, y=0.0, yaw=0.0, holding=True))
    assert b.step == "rearm" and a.stand_up


# ------------------------------------------------------------------ run_log: JSON object reader
def test_read_json_dict(tmp_path):
    from atharv_nandi.core import run_log as rl
    p = tmp_path / "meta.json"
    assert rl.read_json_dict(str(p)) is None                         # missing
    p.write_text("[1, 2]")
    assert rl.read_json_dict(str(p)) is None                         # not an object: old code raised on .update
    p.write_text("{half")
    assert rl.read_json_dict(str(p)) is None
    rl.write_json(str(p), {"final": True, "closed": 5})
    assert rl.read_json_dict(str(p)) == {"final": True, "closed": 5}
    assert rl.read_json_dict(str(tmp_path)) is None                  # a directory (OSError)


# ------------------------------------------------------------------ WPCOV fixer: live stability publisher encoding
def _published(margin):
    """The stability hook's encoding of a predicted margin (``sim/hooks/stability.py`` tick): None when non-finite."""
    return round(margin, 4) if math.isfinite(margin) else None


def test_stability_ok_refuses_the_hooks_non_finite_margin():
    feet = {"lf": (0.3, 0.2, -0.4), "rh": (-0.3, -0.2, -0.4)}         # 2 feet: degenerate tripod -> -inf
    m = st.predicted_tripod_margin((0.0, 0.0, 0.0), feet)["margin_m"]
    assert not math.isfinite(m)
    vals = {"stability": {"margin_m": _published(m), "latched": True, "degenerate": True}}
    assert vals["stability"]["margin_m"] is None
    assert not tc.stability_ok(vals)                                  # was True: None read as "no published value"
    assert tc.close_gate("blade", 1.0, vals, 0.05) == "stability"     # was None (allowed)
    assert tc.close_gate("twist", 0.0, vals, 0.05) == "stability"
    for flag in ("latched", "degenerate"):                            # either monitor flag refuses at a good margin
        assert not tc.stability_ok({"stability": {"margin_m": 0.2, flag: True}}), flag
    assert tc.stability_ok({"stability": {"margin_m": 0.2, "latched": False, "degenerate": False}})


def test_stability_ok_accepts_numpy_scalars():
    np = pytest.importorskip("numpy")
    for good in (np.float32(0.2), np.float64(0.2), np.int64(1)):
        assert tc.stability_ok({"stability": {"margin_m": good}}), type(good)
    assert not tc.stability_ok({"stability": {"margin_m": np.float32(0.01)}})
    assert not tc.stability_ok({"stability": {"margin_m": np.float32("nan")}})
    assert not tc.stability_ok({"stability": {"margin_m": np.bool_(True)}})
    assert tc.swap_gate(tc.Interlocks(margin_m=np.float32(0.2)))


def test_static_margin_rule_is_the_live_one(monkeypatch):
    """swap_gate and stability_ok call stability.gate_static_margin at call time (not a copy of its threshold)."""
    assert tc.MIN_MARGIN_M == st.PLAN_MIN_STATIC_M
    seen = []
    real = st.gate_static_margin

    def spy(m, what="plan"):
        seen.append(what)
        return real(m, what)
    monkeypatch.setattr(st, "gate_static_margin", spy)
    assert tc.swap_gate(tc.Interlocks(margin_m=0.02)).reason == "low_margin"
    assert not tc.stability_ok({"stability": {"margin_m": 0.02}})
    assert seen == ["tool_change", "close"]


def test_ready_check_rtk_is_the_gnss_rule(monkeypatch):
    from atharv_nandi.core import gnss_model as gm
    assert gm.rtk_start_gate(gm.FLOAT, "scan", float_ok=True) == (True, "")
    assert gm.rtk_start_gate(gm.FLOAT, "return_home", float_ok=False) == (False, "rtk")
    assert gm.rtk_start_gate(gm.FIXED, "x", float_ok=False) == (True, "")
    calls = []
    real = gm.rtk_start_gate
    monkeypatch.setattr(gm, "rtk_start_gate", lambda *a, **k: calls.append((a, k)) or real(*a, **k))
    p = rc.ps.Registry.load().get("scan")
    ctx = rc.ReadyContext.nominal(rtk=gm.FLOAT, season_profile="demo_all_fruiting", month=11, frames_age_h=0.1)
    with open(os.path.join(os.path.dirname(__file__), "..", "config", "fields", "orchard_demo.json"), encoding="utf-8") as f:
        field = json.load(f)
    res = rc.check(p, ctx, {"row_ids": ["R1"]}, None, field)
    assert calls and calls[0][1] == {"float_ok": p.float_ok}
    assert "rtk" in res.reasons and res.level == rc.NO_GO
