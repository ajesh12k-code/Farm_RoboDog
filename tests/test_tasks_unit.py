"""WP8 task planners, unit tier (ROS-free, headless, each <= 10 s): every state machine transition and the retry /
skip / MARK paths, the MP 0.4 target loop, the per-tree job planner, MARK records, the tool-aware arm IK additions,
the station geometry helpers and the WP2 swap arm-target forwarding. MuJoCo is used only for the IK checks and the
bagged-season check (a compiled world, no stepping)."""
import json
import math

import numpy as np
import pytest

from atharv_nandi.core import task_retry as tr
from atharv_nandi.core.tasks import banana_tidy, common, tree_task
from atharv_nandi.core.tasks.clip_pedicel import ClipPedicelSM
from atharv_nandi.core.tasks.common import Obs, Target
from atharv_nandi.core.tasks.prune_cut import PruneCutSM, tool_hint_for
from atharv_nandi.core.tasks.twist_pick import TwistPickSM

TREE = {"tree_id": "T-1", "physical_tag": "17", "pos": [0.0, 0.0, 0.0]}


# ---------------------------------------------------------------------- helpers
def tgt(task="clip_pedicel", tool="T2", crop="mango", point=(0.4, -0.3, 0.5), **kw):
    meta = {"tree": TREE, "fruit_id": "T-1:f01", "harvest_stage": True}
    meta.update(kw.pop("meta", {}))
    return Target(target_id=kw.pop("target_id", "T-1:f01"), tree_id="T-1", crop=crop, task=task, tool=tool,
                  point=tuple(point), meta=meta, **kw)


class Drive:
    """Feeds an SM with Obs; ``t`` advances 20 ms per tick. ``obs(**kw)`` sets the sticky fields."""

    def __init__(self, sm):
        self.sm, self.t = sm, 0.0
        self.base = {"arm_done": False, "tcp_err_m": 1.0, "precheck": {}}

    def tick(self, n=1, **kw):
        """Up to ``n`` ticks; stops early at the first state change (returns that tick's command) or at the end."""
        cmd = None
        for _ in range(n):
            before = self.sm.state
            o = dict(self.base)
            o.update(kw)
            cmd = self.sm.step(Obs(t=self.t, **o))
            self.t += 0.02
            kw.pop("events", None)            # events are one-shot
            if self.sm.done or self.sm.state != before:
                break
        return cmd

    def until(self, state, limit=500, **kw):
        for _ in range(limit):
            if self.sm.state == state or self.sm.done:
                return
            self.tick(**kw)
        raise AssertionError(f"never reached {state}, at {self.sm.state}")

    def set(self, **kw):
        self.base.update(kw)


def states(sm):
    return [h[2] for h in sm.history]


def run_clip_to_cut(sm, drv):
    drv.tick()                                       # start -> plan
    drv.tick(arm_done=True)                          # plan -> roll_align
    drv.until("approach")
    drv.tick(arm_done=True)                          # approach -> align
    drv.until("close", tcp_err_m=0.0005)


# ---------------------------------------------------------------------- common
def test_h_class_side_clock_band_and_paste():
    assert [common.h_class(h) for h in (0.3, 0.8, 1.8, 1.81)] == ["neeche", "beech", "beech", "upar"]
    assert common.side_of((0, 0), (0, 1)) == "N" and common.side_of((0, 0), (1, 0)) == "E"
    assert common.side_of((0, 0), (0, -1)) == "S" and common.side_of((0, 0), (-1, 0)) == "W"
    # seen from the south: far side 12, near side 6, east (right) 3, west 9
    assert common.clock_of((0, 0), (0, 1)) == 12 and common.clock_of((0, 0), (0, -1)) == 6
    assert common.clock_of((0, 0), (1, 0)) == 3 and common.clock_of((0, 0), (-1, 0)) == 9
    assert common.in_pick_band(0.8) and not common.in_pick_band(0.81) and not common.in_pick_band(0.5, 0.85)
    assert common.jaw_clearance_ok(16.9) and not common.jaw_clearance_ok(-4.4) and common.jaw_clearance_ok(None)
    assert common.paste_required(12.5) and not common.paste_required(12.0) and not common.paste_required(None)


def test_mark_record_shape_and_validation():
    mk = common.make_mark(TREE, "prune", (0.0, 0.5, 1.2), "paste_required", diameter_mm=19.4, tool_hint="lopper",
                          target_id="T-1:prune:5")
    assert mk["schema"] == "nandi.mark/1" and mk["paste_required"] and mk["h_class"] == "beech"
    assert mk["side"] == "N" and mk["clock"] == 12 and mk["physical_tag"] == "17"
    assert set(mk) >= {"tree_id", "physical_tag", "kind", "clock", "side", "h_class", "h_m", "stage", "reason",
                       "diameter_mm", "paste_required", "tool_hint"}
    assert not common.make_mark(TREE, "fruit", (0, 0, 2.0), "out_of_reach_marked")["paste_required"]
    with pytest.raises(ValueError):
        common.make_mark(TREE, "branch", (0, 0, 0), "x")
    with pytest.raises(ValueError):
        common.make_mark(TREE, "fruit", (0, 0, 0), "x", tool_hint="chainsaw")
    rec = common.harvest_record("T-1", "T-1:f01", "mango", "clip", "picked", pedicel_mm=19.712, cell=3)
    assert rec["pedicel_mm"] == 19.71 and rec["cell"] == 3
    with pytest.raises(ValueError):
        common.harvest_record("T-1", "x", "mango", "clip", "great")


# ---------------------------------------------------------------------- ClipPedicelSM
def test_clip_happy_path_transitions_and_record():
    sm = ClipPedicelSM(tgt(pedicel_leave_m=0.02))
    drv = Drive(sm)
    c = drv.tick()
    assert sm.state == "plan" and c.arm == "plan_pre" and c.tool == "open"
    run_clip_to_cut(sm, drv)
    cut = {"event": "cut", "target": "T-1:f01", "pedicel_mm": 19.7, "pedicel_error_mm": 0.3,
           "assisted": ["logical_cut", "weld_grasp"]}
    drv.tick(events=[cut])
    assert sm.state == "hold_verify" and sm.committed
    c = drv.tick(caught=True)
    assert sm.state == "retreat" and c.arm == "retreat"
    drv.tick(arm_done=True, caught=True)
    assert sm.state == "park"
    c = drv.tick(arm_done=True, caught=True)
    assert sm.state == "place" and c.place
    c = drv.tick(placed="seated", precheck={"cell": 2})
    assert sm.done and sm.outcome == "picked" and c.tool == "open"
    assert states(sm) == ["plan", "roll_align", "approach", "align", "close", "hold_verify", "retreat", "park",
                          "place", "done"]
    r = sm.record
    assert r["method"] == "clip" and r["pedicel_mm"] == 19.7 and r["pedicel_error_mm"] == 0.3 and r["cell"] == 2
    assert r["outcome"] == "picked" and r["reason"] == ""


def test_clip_preconditions_bagged_immature_band_torque_margin():
    sm = ClipPedicelSM(tgt(bagged=True))
    Drive(sm).tick()
    assert (sm.outcome, sm.reason) == ("skipped", "bagged")
    sm = ClipPedicelSM(tgt(stage="immature", meta={"harvest_stage": False}))
    Drive(sm).tick()
    assert (sm.outcome, sm.reason) == ("skipped", "immature")
    sm = ClipPedicelSM(tgt(point=(0.4, -0.3, 1.6), in_band=False, stage="mature"))
    Drive(sm).tick()
    assert (sm.outcome, sm.reason) == ("marked", "out_of_reach_marked")
    assert sm.marks[0]["tool_hint"] == "pole_picker" and sm.marks[0]["kind"] == "fruit"
    for pre, why in (({"torque_ok": False}, "torque_limit"), ({"margin_ok": False}, "stability")):
        sm = ClipPedicelSM(tgt())
        Drive(sm).tick(precheck=pre)
        assert (sm.outcome, sm.reason) == ("skipped", why)


def test_clip_miss_realigns_10deg_then_cuts():
    sm = ClipPedicelSM(tgt(), roll_deg=5.0)
    drv = Drive(sm)
    run_clip_to_cut(sm, drv)
    c = drv.tick(events=[{"event": "cut_miss", "conditions": {"angle": False}}])
    assert sm.state == "realign" and c.tool == "open" and c.roll_deg == pytest.approx(15.0) and sm.attempts == 2
    drv.until("close", tcp_err_m=0.0005)
    drv.tick(events=[{"event": "cut", "pedicel_mm": 20.0, "pedicel_error_mm": 0.0}])
    assert sm.state == "hold_verify"


def test_clip_two_misses_skip_cut_failed_after_retreat_and_park():
    sm = ClipPedicelSM(tgt())
    drv = Drive(sm)
    run_clip_to_cut(sm, drv)
    drv.tick(n=60)                                   # 1 s with no cut event -> realign (timeout counts as a miss)
    assert sm.attempts == 2
    drv.until("close", tcp_err_m=0.0005)
    c = drv.tick(events=[{"event": "cut_miss"}])
    assert sm.state == "retreat_skip" and c.arm == "retreat" and c.tool == "open"
    c = drv.tick(arm_done=True)
    assert sm.state == "park_skip" and c.arm == "park"
    drv.tick(arm_done=True)
    assert (sm.outcome, sm.reason) == ("skipped", "cut_failed")


def test_clip_refused_crop_dropped_and_place_refused():
    sm = ClipPedicelSM(tgt(crop="papaya"))
    drv = Drive(sm)
    run_clip_to_cut(sm, drv)
    drv.tick(events=[{"event": "cut_refused", "reason": "no_cut_force"}])
    drv.tick(arm_done=True)
    drv.tick(arm_done=True)
    assert (sm.outcome, sm.reason) == ("skipped", "not_in_mvp")
    # cut fired but the catcher has nothing: dropped
    sm = ClipPedicelSM(tgt())
    drv = Drive(sm)
    run_clip_to_cut(sm, drv)
    drv.tick(events=[{"event": "cut", "pedicel_mm": 20, "pedicel_error_mm": 0}])
    drv.tick(n=20, caught=False)
    assert (sm.outcome, sm.reason) == ("dropped", "dropped") and sm.record["outcome"] == "dropped"
    # the transfer refuses (tray full): delivered as dropped with the reason
    sm = ClipPedicelSM(tgt())
    drv = Drive(sm)
    run_clip_to_cut(sm, drv)
    drv.tick(events=[{"event": "cut", "pedicel_mm": 20, "pedicel_error_mm": 0}])
    drv.tick(caught=True)
    drv.tick(arm_done=True, caught=True)
    drv.tick(arm_done=True, caught=True)
    drv.tick(placed="refused:payload_full")
    assert sm.outcome == "dropped" and sm.reason == "place_refused:payload_full"


def test_clip_pedicel_out_of_tolerance_is_delivered_and_reported():
    sm = ClipPedicelSM(tgt())
    drv = Drive(sm)
    run_clip_to_cut(sm, drv)
    drv.tick(events=[{"event": "cut", "pedicel_mm": 26.2, "pedicel_error_mm": 6.2}])
    drv.tick(caught=True)
    drv.tick(arm_done=True, caught=True)
    drv.tick(arm_done=True, caught=True)
    drv.tick(placed="seated")
    assert sm.outcome == "picked" and sm.reason == "pedicel_out_of_tolerance"


def test_base_drift_pauses_and_reapproaches_then_gives_up():
    sm = ClipPedicelSM(tgt(), max_reapproach=1)
    drv = Drive(sm)
    drv.tick()
    drv.tick(arm_done=True)                          # roll_align
    c = drv.tick(base_drift_m=0.025)
    assert sm.state == "base_pause" and c.arm == "hold" and c.tool == "open"
    c = drv.tick(base_settled=True)
    assert sm.state == "plan" and c.rebase and sm.reapproaches == 1
    drv.tick(arm_done=True)
    drv.tick(base_drift_m=0.03)
    assert sm.state == "base_pause"
    drv.tick(n=200, base_settled=False)              # never settles: after 3 s give up
    drv.tick(arm_done=True)
    drv.tick(arm_done=True)
    assert (sm.outcome, sm.reason) == ("skipped", "base_moved")


def test_drift_after_the_commit_is_ignored_and_estop_latches():
    sm = ClipPedicelSM(tgt())
    drv = Drive(sm)
    run_clip_to_cut(sm, drv)
    drv.tick(events=[{"event": "cut", "pedicel_mm": 20, "pedicel_error_mm": 0}])
    drv.tick(caught=True, base_drift_m=0.05)
    assert sm.state == "retreat"                     # atomic hold_verify -> place block
    c = drv.tick(estop=True)
    assert sm.state == "halted" and sm.outcome == "halted" and c.tool == "stop" and c.done
    c = drv.tick(estop=False)
    assert sm.done and c.done                        # latched: never resumes by itself


def test_person_in_work_field_holds_then_reapproaches():
    sm = ClipPedicelSM(tgt())
    drv = Drive(sm)
    drv.tick()
    drv.tick(arm_done=True)
    drv.until("approach")
    c = drv.tick(person=True)
    assert sm.state == "person_hold" and c.arm == "hold" and c.tool == "open"
    drv.tick(n=10, person=True)
    assert sm.state == "person_hold"
    c = drv.tick(person=False)
    assert sm.state == "plan" and c.rebase


# ---------------------------------------------------------------------- TwistPickSM
def twist_to_grip(sm, drv):
    drv.tick()
    drv.tick(arm_done=True)                          # plan -> roll_align
    drv.until("approach")
    drv.tick(arm_done=True)
    drv.until("grip", tcp_err_m=0.001)


def test_twist_no_clearance_marks_and_payload_skip():
    sm = TwistPickSM(tgt("twist_pick", "T4", "papaya", clearance_mm=-4.4, target_id="R3-1:f02"))
    Drive(sm).tick()
    assert (sm.outcome, sm.reason) == ("marked", "no_clearance")
    assert sm.marks[0]["reason"] == "no_clearance" and sm.marks[0]["clearance_mm"] == -4.4
    sm = TwistPickSM(tgt("twist_pick", "T4", "papaya", clearance_mm=16.9))
    Drive(sm).tick(precheck={"payload_ok": False})
    assert (sm.outcome, sm.reason) == ("skipped", "payload_full")


def test_twist_happy_path_pre_roll_release_carry_place():
    sm = TwistPickSM(tgt("twist_pick", "T4", "papaya", clearance_mm=16.9, mass_kg=1.8))
    drv = Drive(sm)
    c = drv.tick()
    assert c.arm == "plan_pre" and c.roll_deg == 0.0
    c = drv.tick(arm_done=True)
    assert sm.state == "roll_align" and c.roll_deg == -20.0           # pre-roll against the twist
    drv.until("approach")
    drv.tick(arm_done=True)
    c = drv.until("grip", tcp_err_m=0.001)
    drv.tick(n=20, touch_n=6.0, touch_max_n=7.0)
    assert sm.state == "twist"
    c = drv.tick(touch_n=6.0, touch_max_n=7.0, roll_deg=-10.0)
    assert c.roll_deg == 270.0 and c.arm_args["roll_rate_dps"] > 0
    rel = {"event": "twist_release", "tau_peak_nm": 2.1, "roll_deg": 30.2, "assisted": ["logical_twist_release"]}
    drv.tick(events=[rel], roll_deg=11.0, touch_n=6.0, touch_max_n=7.0)
    assert sm.state == "hold_verify" and sm.committed
    c = drv.tick(n=20, caught=True, touch_n=1.5, touch_max_n=6.0)
    assert sm.state == "carry" and c.arm_args["max_tcp_mps"] == 0.5            # papaya carry <= 0.5 m/s
    c = drv.tick(arm_done=True, caught=True, touch_max_n=6.0)
    assert sm.state == "park" and c.arm_args["max_tcp_mps"] == 0.5
    drv.tick(arm_done=True, caught=True, touch_max_n=6.0)
    drv.tick(placed="seated", precheck={"cell": "cradle"})
    assert sm.outcome == "picked" and sm.record["method"] == "twist" and sm.record["cell"] == "cradle"
    assert sm.record["tau_peak_nm"] == 2.1 and sm.record["mass_kg"] == 1.8


def test_twist_grip_fail_regrips_once_then_skips():
    sm = TwistPickSM(tgt("twist_pick", "T4", "papaya", clearance_mm=20.0))
    drv = Drive(sm)
    twist_to_grip(sm, drv)
    drv.tick(n=80, touch_n=0.0)
    assert "regrip" in states(sm) and sm.attempts == 2
    drv.until("grip", tcp_err_m=0.001)
    drv.tick(n=80, touch_n=0.0)
    drv.tick(arm_done=True)
    drv.tick(arm_done=True)
    assert (sm.outcome, sm.reason) == ("skipped", "grip_failed")


def test_twist_no_release_at_540_regrips_other_way_then_marks_twist_failed():
    sm = TwistPickSM(tgt("twist_pick", "T4", "guava", clearance_mm=None))
    drv = Drive(sm)
    twist_to_grip(sm, drv)
    drv.tick(n=20, touch_n=5.0, touch_max_n=5.0)
    assert sm.state == "twist"
    c = drv.tick(touch_n=5.0, roll_travel_deg=541.0)
    assert sm.state == "regrip" and c.tool == "open" and sm.pre_roll == 20.0      # grip the other way round
    drv.until("grip", tcp_err_m=0.001)
    drv.tick(n=20, touch_n=5.0, touch_max_n=5.0)
    c = drv.tick(touch_n=5.0, roll_deg=10.0)
    assert c.roll_deg == -270.0                                                   # and twist the other way
    drv.tick(touch_n=5.0, roll_travel_deg=545.0)
    drv.tick(arm_done=True)
    drv.tick(arm_done=True)
    assert (sm.outcome, sm.reason) == ("marked", "twist_failed")
    assert sm.marks[0]["reason"] == "twist_failed" and sm.marks[0]["roll_deg"] == 545.0


def test_twist_stroke_reverses_at_the_end_stop():
    sm = TwistPickSM(tgt("twist_pick", "T4", "guava"))
    drv = Drive(sm)
    twist_to_grip(sm, drv)
    drv.tick(n=20, touch_n=5.0, touch_max_n=5.0)
    c = drv.tick(touch_n=5.0, roll_deg=269.0, roll_travel_deg=289.0)
    assert c.roll_deg == -270.0


def test_twist_slip_in_cradle_is_dropped_and_weld_only_carry_watches_the_catch():
    sm = TwistPickSM(tgt("twist_pick", "T4", "papaya", clearance_mm=20.0))
    drv = Drive(sm)
    twist_to_grip(sm, drv)
    drv.tick(n=20, touch_n=5.0, touch_max_n=5.0)
    drv.tick(events=[{"event": "twist_release"}], touch_n=5.0, touch_max_n=5.0)
    drv.tick(n=12, caught=True, touch_max_n=5.0)
    assert sm.state == "carry" and sm.watch_pads
    c = drv.tick(n=15, caught=True, touch_max_n=0.3)                 # pads below 1 N for 0.2 s: slip
    assert sm.state == "slip" and c.arm == "lower"
    drv.tick(arm_done=True)
    assert (sm.outcome, sm.reason) == ("dropped", "dropped")
    # pads never pressed after the release (cam-over): the assisted catch weld is what the carry watches
    sm = TwistPickSM(tgt("twist_pick", "T4", "papaya", clearance_mm=20.0))
    drv = Drive(sm)
    twist_to_grip(sm, drv)
    drv.tick(n=20, touch_n=5.0, touch_max_n=5.0)
    drv.tick(events=[{"event": "twist_release"}])
    drv.tick(n=12, caught=True, touch_max_n=0.0)
    assert sm.state == "carry" and not sm.watch_pads
    drv.tick(n=20, caught=True, touch_max_n=0.0)
    assert sm.state == "carry"
    drv.tick(caught=False)
    assert sm.state == "slip"


# ---------------------------------------------------------------------- PruneCutSM / BananaTidySM
def prune_tgt(**kw):
    kw.setdefault("diameter_mm", 6.0)
    return tgt("prune_cut", "T5", "mango", point=kw.pop("point", (0.04, -0.1, 0.22)), job=kw.pop("job", "rootstock_sprout"),
               target_id="T-1:prune:1", **kw)


def test_prune_preconditions_keep_capacity_band():
    sm = PruneCutSM(prune_tgt(keep=True))
    Drive(sm).tick()
    assert (sm.outcome, sm.reason) == ("skipped", "keep_wood") and not sm.marks
    sm = PruneCutSM(prune_tgt(diameter_mm=19.4, job="skirt_lift", point=(0.7, 0.0, 0.58)))
    Drive(sm).tick()
    assert (sm.outcome, sm.reason) == ("marked", "over_capacity")
    mk = sm.marks[0]
    assert mk["kind"] == "skirt_lift" and mk["reason"] == "over_capacity" and mk["paste_required"]
    assert mk["diameter_mm"] == 19.4 and mk["tool_hint"] == "lopper"
    sm = PruneCutSM(prune_tgt(job="water_shoot", diameter_mm=9.0, point=(0.1, 0.1, 1.6), in_band=False))
    Drive(sm).tick()
    assert (sm.outcome, sm.reason) == ("marked", "out_of_reach_marked") and sm.marks[0]["kind"] == "prune"
    assert tool_hint_for(30.0, 0.5) == "saw" and tool_hint_for(8.0, 2.0) == "pole_picker"


def test_prune_happy_path_and_refusal_and_no_cut():
    sm = PruneCutSM(prune_tgt())
    drv = Drive(sm)
    run_clip_to_cut(sm, drv)
    c = drv.tick(events=[{"event": "cut", "target": "T-1:prune:1", "job": "rootstock_sprout", "diameter_mm": 6.0,
                          "debris": 0}])
    assert sm.state == "retreat" and c.tool == "open" and sm.committed
    drv.tick(arm_done=True)
    drv.tick(arm_done=True)
    assert sm.outcome == "cut" and sm.record["job"] == "rootstock_sprout"
    assert states(sm) == ["plan", "roll_align", "approach", "align", "close", "retreat", "park", "done"]
    # the PRUNER refuses (live diameter over capacity): open, MARK over_capacity with paste_required
    sm = PruneCutSM(prune_tgt())
    drv = Drive(sm)
    run_clip_to_cut(sm, drv)
    drv.tick(events=[{"event": "prune_refused", "reason": "over_capacity", "paste_required": True}])
    drv.tick(arm_done=True)
    drv.tick(arm_done=True)
    assert (sm.outcome, sm.reason) == ("marked", "over_capacity") and sm.marks[0]["reason"] == "over_capacity"
    assert sm.marks[0]["paste_required"]                  # the live refusal flag wins over the 6 mm truth
    # no cut 1 s after the close: re-align once, then skip
    sm = PruneCutSM(prune_tgt())
    drv = Drive(sm)
    run_clip_to_cut(sm, drv)
    drv.tick(n=55)
    assert sm.state in ("realign", "align") and sm.attempts == 2
    drv.until("close", tcp_err_m=0.0005)
    drv.tick(n=55)
    drv.tick(arm_done=True)
    drv.tick(arm_done=True)
    assert (sm.outcome, sm.reason) == ("skipped", "cut_failed")


def test_banana_tidy_refuses_green_and_non_tidy_jobs():
    t = tgt("banana_tidy", "T5", "banana", point=(0.0, -0.5, 0.72), job="banana_tidy", diameter_mm=8.0)
    sm = banana_tidy.BananaTidySM(t)
    Drive(sm).tick()
    assert sm.state == "plan"
    for kw in ({"green": True}, {"keep": True}, {"job": "sucker"}):
        base = {"job": "banana_tidy", "diameter_mm": 8.0}
        base.update(kw)
        sm = banana_tidy.BananaTidySM(tgt("banana_tidy", "T5", "banana", point=(0.0, -0.5, 0.72), **base))
        Drive(sm).tick()
        assert (sm.outcome, sm.reason) == ("skipped", "keep_wood")


def banana_plan():
    return {"tree_id": "B-1", "crop": "banana", "pos": [2.0, 14.5, 0.0], "truth": {
        "physical_tag": "B1",
        "bunch": {"geom": "tree_B-1_bunch_0", "ribbon_colour": "red", "shooting_date": "2026-08-10", "stage": "immature",
                  "harvest_window": {"from": "2026-11-08", "to": "2026-12-08"}, "h_m": 1.5, "world": [2.0, 14.2, 1.5],
                  "hands_n": 10, "exclusion_m": 3.0},
        "blades": [{"id": "B-1:blade:1", "functional": False, "dead": True, "sigatoka_pct": 0.0, "prune": "B-1:prune:1"},
                   {"id": "B-1:blade:3", "functional": True, "sigatoka_pct": 34.2}],
        "sanitation_deleaf": ["B-1:blade:1"],
        "suckers": ["B-1:prune:3", "B-1:prune:4"],
        "prune": [
            {"id": "B-1:prune:1", "job": "banana_tidy", "keep": False, "diameter_mm": 8.0, "cut_world": [1.9, 13.96, 0.72],
             "a": [0, 0, 1.75], "b": [0, -0.5, 0.4], "in_band": True, "green": False, "geom": "g1"},
            {"id": "B-1:prune:2", "job": "banana_tidy", "keep": False, "diameter_mm": 8.0, "cut_world": [1.9, 13.9, 0.95],
             "a": [0, 0, 1.75], "b": [0, -0.5, 0.4], "in_band": False, "green": False, "geom": "g2"},
            {"id": "B-1:prune:5", "job": "banana_tidy", "keep": False, "diameter_mm": 8.0, "cut_world": [1.9, 13.9, 0.7],
             "a": [0, 0, 1.75], "b": [0, -0.5, 0.4], "in_band": True, "green": True, "geom": "g5"},
            {"id": "B-1:prune:3", "job": "sucker", "keep": True, "follower": True, "diameter_mm": 82.0,
             "cut_world": [2.3, 14.6, 0.1], "h_m": 0.65},
            {"id": "B-1:prune:4", "job": "sucker", "keep": False, "follower": False, "sucker_type": "peeper",
             "diameter_mm": 66.0, "cut_world": [1.7, 14.4, 0.1], "h_m": 0.48}]}}


def test_banana_marks_bunch_deleaf_desucker_and_tidy_targets():
    plan = banana_plan()
    marks = banana_tidy.banana_marks(plan)
    kinds = [m["kind"] for m in marks]
    assert kinds == ["bunch", "sanitation_deleaf", "sucker"]
    b = marks[0]
    assert b["tool_hint"] == "human_cut" and b["ribbon_colour"] == "red" and b["hands_n_lead"] == 10
    assert b["harvest_window"]["from"] == "2026-11-08" and b["h_class"] == "beech" and b["exclusion_m"] == 3.0
    assert marks[2]["target_id"] == "B-1:prune:4"                    # the follower sword is kept, never MARKed
    job = tree_task.prune_job(plan, {"prune": {"jobs": [], "max_stem_mm": 12}})
    assert [t.target_id for t in job.targets] == ["B-1:prune:1"]
    assert {"target_id": "B-1:prune:5", "reason": "keep_wood"} in job.skipped           # green: never a target
    assert {"target_id": "B-1:prune:2", "reason": "out_of_reach_marked"} in job.skipped
    assert all(m["kind"] != "bunch" for m in job.marks)
    hjob = tree_task.harvest_job(plan, {"harvest_method": "mark", "methods": ["mark"]})
    assert not hjob.targets and [m["kind"] for m in hjob.marks][0] == "bunch"


def test_bunch_exclusion_and_blade_clean():
    marks = banana_tidy.banana_marks(banana_plan())
    bunches = {"B-1": (2.0, 14.2)}
    assert banana_tidy.bunch_exclusion_ok((2.0, 10.0), marks, bunches, ["B-1"])
    assert not banana_tidy.bunch_exclusion_ok((2.0, 12.0), marks, bunches, ["B-1"])
    assert banana_tidy.bunch_exclusion_ok((2.0, 12.0), marks, bunches, [])            # no cut in progress
    assert not banana_tidy.needs_blade_clean(None, "banana", None, "B-1")
    assert banana_tidy.needs_blade_clean("banana", "banana", "B-1", "B-2")
    assert not banana_tidy.needs_blade_clean("mango", "mango", "R1-1", "R1-2")
    assert banana_tidy.needs_blade_clean("mango", "guava", "R1-1", "R2-1")


# ---------------------------------------------------------------------- tree_task
def fruit(fid, z, stage="mature", hs=True, bagged=False, in_band=None, clearance=None, kind="connect", mass=0.3):
    c = [0.3, -0.3, z]
    return {"fruit_id": fid, "stage": stage, "harvest_stage": hs, "bagged": bagged,
            "in_band": (z <= 0.8) if in_band is None else in_band, "clearance_mm": clearance, "pool_kind": kind,
            "mass_kg": mass, "centre": c, "centre_world": c, "target_z_m": z, "cut_k_world": [0.3, -0.3, z + 0.05],
            "stem_end_world": [0.3, -0.3, z + 0.03], "anchor_top_world": [0.3, -0.25, z + 0.06], "clock": 7,
            "wp0c": {"id": f"wp:{fid}"}}


def profiles():
    from atharv_nandi.sim.world import crops
    return {c: crops.load_profile(c) for c in ("mango", "guava", "papaya", "banana")}


def test_mango_one_round_marks_up_band_and_refuses_bagged():
    P = profiles()
    plan = {"tree_id": "OT-1", "crop": "mango", "pos": [6, -3.5, 0], "truth": {"fruit": [
        fruit("f1", 0.5), fruit("f2", 0.66), fruit("f3", 0.74), fruit("f4", 0.6, stage="immature", hs=False),
        fruit("f5", 1.6), fruit("f6", 0.55, bagged=True)]}}
    st = {"wp:f1": {"station": {"xy": [5, -4], "heading_deg": -40}, "axis_r": [0, 1, -0.2]}}
    job = tree_task.harvest_job(plan, P["mango"], stations=st)
    assert [t.target_id for t in job.targets] == ["f1", "f2", "f3"]           # every mature in-band fruit, low first
    assert all(t.task == "clip_pedicel" and t.tool == "T2" for t in job.targets)
    assert job.targets[0].station["heading_deg"] == -40 and job.targets[0].pitch == pytest.approx(0.2014, abs=1e-3)
    assert job.targets[0].point == (0.3, -0.3, 0.55)                          # the cut point (cut_k)
    assert [m["target_id"] for m in job.marks] == ["f5"] and job.marks[0]["reason"] == "out_of_reach_marked"
    assert {"target_id": "f6", "reason": "bagged"} in job.skipped
    assert tree_task.harvest_job(plan, P["mango"], exclude=["f1"]).targets[0].target_id == "f2"
    j2 = tree_task.harvest_job(plan, P["mango"], payload_free_cells=2)
    assert len(j2.targets) == 2 and j2.deferred == [{"target_id": "f3", "reason": "payload_full"}]
    assert tree_task.harvest_job(plan, P["mango"], method="twist").skipped[0]["reason"] == "not_in_mvp"


def test_guava_twist_option_and_papaya_one_per_trip_with_clearance():
    P = profiles()
    g = {"tree_id": "R2-1", "crop": "guava", "pos": [2, 7.5, 0], "truth": {"fruit": [fruit("g1", 0.48, kind="weld")]}}
    assert tree_task.harvest_job(g, P["guava"]).targets[0].tool == "T2"
    t = tree_task.harvest_job(g, P["guava"], method="twist").targets[0]
    assert (t.task, t.tool) == ("twist_pick", "T4") and t.point == (0.3, -0.3, 0.48)
    pap = {"tree_id": "R3-1", "crop": "papaya", "pos": [2, 11, 0], "truth": {"fruit": [
        fruit("p1", 0.64, stage="colour_break", clearance=16.9, kind="weld", mass=1.54),
        fruit("p2", 0.75, stage="colour_break", clearance=-4.4, kind="weld"),
        fruit("p3", 0.78, stage="colour_break", clearance=25.0, kind="weld")]}}
    job = tree_task.harvest_job(pap, P["papaya"])
    assert [x.target_id for x in job.targets] == ["p1"] and job.targets[0].mass_kg == 1.54
    assert job.deferred == [{"target_id": "p3", "reason": "payload_full"}]
    assert [m["reason"] for m in job.marks] == ["no_clearance"] and job.marks[0]["target_id"] == "p2"
    assert job.targets[0].meta["pedicel"] == pytest.approx([0.0, 0.05, 0.03])
    full = tree_task.harvest_job(pap, P["papaya"], cradle_free=False)
    assert not full.targets and len(full.deferred) == 2


def test_prune_job_filters_marks_and_keeps():
    P = profiles()
    cand = [
        {"id": "s1", "job": "rootstock_sprout", "keep": False, "diameter_mm": 6.0, "cut_world": [6.0, -3.6, 0.22],
         "z_m": 0.22, "a": [0, 0, 0.18], "b": [0.07, -0.2, 0.3], "wp0c": {"id": "wp:s1"}},
        {"id": "k1", "job": "skirt_lift", "keep": True, "diameter_mm": 10.0, "cut_world": [6.0, -3.6, 0.4], "z_m": 0.4},
        {"id": "b1", "job": "skirt_lift", "keep": False, "diameter_mm": 19.4, "cut_world": [6.7, -3.5, 0.58],
         "z_m": 0.58},
        {"id": "w1", "job": "water_shoot", "keep": False, "diameter_mm": 9.0, "cut_world": [6.1, -3.5, 1.6], "z_m": 1.6},
        {"id": "d1", "job": "dead_twig", "keep": False, "diameter_mm": 9.0, "cut_world": [6.3, -3.0, 0.6], "z_m": 0.6}]
    plan = {"tree_id": "OT-1", "crop": "mango", "pos": [6, -3.5, 0], "truth": {"prune": cand}}
    job = tree_task.prune_job(plan, P["mango"])
    assert [t.target_id for t in job.targets] == ["s1", "d1"]
    assert all(t.tool == "T5" and t.task == "prune_cut" for t in job.targets)
    assert {"target_id": "k1", "reason": "keep_wood"} in job.skipped
    mk = {m["target_id"]: m for m in job.marks}
    assert mk["b1"]["paste_required"] and mk["b1"]["kind"] == "skirt_lift" and mk["b1"]["reason"] == "over_capacity"
    assert mk["w1"]["reason"] == "out_of_reach_marked" and not mk["w1"]["paste_required"]
    only = tree_task.prune_job(plan, P["mango"], jobs=["rootstock_sprout"])
    assert [t.target_id for t in only.targets] == ["s1"] and not only.marks


def test_bagged_season_fruit_are_refused_by_planner_and_pool(monkeypatch):
    from atharv_nandi.core import farm_scene
    from atharv_nandi.core.fruit_pool import FruitPool, PoolError
    from atharv_nandi.sim.world import crops
    from atharv_nandi.sim.world.modules import trees as trees_mod
    import mujoco
    monkeypatch.setenv("NANDI_WORLD", "orchard_lite")
    monkeypatch.setenv("NANDI_SEASON", "bagged")
    m = farm_scene.load_farm_model()
    plans = trees_mod.plans_for_model(m)
    mango = [p for p in plans if p["crop"] == "mango" and any(f.get("bagged") for f in p["truth"]["fruit"])]
    assert mango
    p = mango[0]
    job = tree_task.harvest_job(p, crops.load_profile("mango"))
    bagged = [f["fruit_id"] for f in p["truth"]["fruit"] if f.get("bagged")]
    assert not [t for t in job.targets if t.target_id in bagged]
    assert {s["target_id"] for s in job.skipped if s["reason"] == "bagged"} == set(bagged)
    d = mujoco.MjData(m)
    with pytest.raises(PoolError, match="bagged"):
        FruitPool(m).assign(d, bagged[0])


# ---------------------------------------------------------------------- task_retry
def test_target_loop_retry_skip_and_final():
    loop = tr.TargetLoop(["a", "b", "c", "d"], tr.RetryPolicy(per_target=1))
    assert loop.next_target() == "a"
    assert loop.report("a", "skipped", "cut_failed")["decision"] == "retry"
    assert loop.next_target() == "a"
    d = loop.report("a", "skipped", "cut_failed")
    assert d["decision"] == "next" and loop.state["a"].outcome == "skipped" and loop.next_target() == "b"
    assert loop.report("b", "picked")["decision"] == "next"
    assert loop.report("c", "marked", "no_clearance")["decision"] == "next"     # MARKs are final, never retried
    assert loop.report("d", "dropped", "dropped", committed=True)["decision"] == "tree_done"
    assert loop.finished and loop.summary()["counts"] == {"skipped": 1, "picked": 1, "marked": 1, "dropped": 1}


def test_target_loop_halts_tree_over_half_skipped():
    loop = tr.TargetLoop(["a", "b", "c"], tr.RetryPolicy(per_target=0), tree_id="T")
    loop.report("a", "skipped", "bagged")
    d = loop.report("b", "skipped", "plan_failed")
    assert d["decision"] == "halt_tree" and d["next"] == "return_home" and loop.halted == "tree_halt"
    assert loop.next_target() is None and loop.finished


def test_target_loop_assist_auto_and_manual_replies():
    loop = tr.TargetLoop(["a", "b", "c"], tr.RetryPolicy(per_target=0, assist_auto_reply=None))
    d = loop.report("a", "skipped", "grip_failed")
    assert d["decision"] == "assist" and d["options"] == ["take_over", "skip", "halt"] and loop.next_target() is None
    d = loop.assist_reply("take_over")
    assert loop.state["a"].reason == "operator_take_over" and d["decision"] == "next"
    loop.report("b", "skipped", "grip_failed")
    d = loop.assist_reply("halt")
    assert d["decision"] == "halt_tree" and loop.halted == "operator_halt"
    assert loop.assist_reply("skip")["decision"] == "ignored"
    pend = tr.TargetLoop(["x"], tr.RetryPolicy(per_target=0, assist_auto_reply=None))
    pend.report("x", "skipped", "grip_failed")
    with pytest.raises(ValueError):
        pend.assist_reply("maybe")
    auto = tr.TargetLoop(["a", "b"], tr.RetryPolicy(per_target=0))
    d = auto.report("a", "skipped", "base_moved")
    assert d["decision"] == "next" and d["auto"] and d["assist"] == "skip"


def test_target_loop_safety_latches():
    loop = tr.TargetLoop(["a", "b"])
    d = loop.report("a", "halted", "estop")
    assert d["decision"] == "halt_safety" and loop.latched == "estop" and loop.next_target() is None
    loop.clear_latch()
    assert loop.next_target() == "a"                      # the interrupted target resumes once the latch is cleared
    assert tr.TargetLoop(["a"]).latch("protective_stop")["decision"] == "halt_safety"


# ---------------------------------------------------------------------- arm IK additions (compiled legacy model)
@pytest.fixture(scope="module")
def ik():
    from atharv_nandi.core.arm_ik_solver import DorsalArmIKSolver
    from atharv_nandi.core.farm_scene import load_farm_model
    return DorsalArmIKSolver(load_farm_model())


def test_tool_tcp_table_matches_the_wp2_registry(ik):
    from atharv_nandi.core import arm_ik_solver as aik
    from atharv_nandi.core.tools import registry as reg
    for k, v in aik.TOOL_TCP_M.items():
        assert reg.get(k).tcp_from_wrist_m == pytest.approx(v)
    assert ik.tool_tcp_m(None) == pytest.approx(ik.l3) and ik.tool_tcp_m("T4") == 0.25


@pytest.mark.parametrize("tool, pitch", [("T2", 0.2), ("T4", 0.0), ("T5", math.pi / 2)])
def test_tcp_solve_puts_the_tool_tcp_on_target(ik, tool, pitch):
    qpos = ik.data.qpos.copy()
    tp, tR = ik.trunk_pose(qpos)
    target = tp + tR @ np.array([0.10, 0.62, 0.2])
    q, ax = ik.tcp_solve(qpos, target, tool, pitch, gantry=None)
    assert q is not None
    ik._fk(qpos, q)
    wp, wr = ik._wrist_frame()
    tcp = wp + wr[:, 0] * ik.tool_tcp_m(tool)
    assert np.linalg.norm(tcp - target) < 1e-6
    assert np.allclose(wr[:, 0], ax, atol=1e-6) and ax[2] == pytest.approx(-math.sin(pitch), abs=1e-6)
    assert ik.keepout_clearance(qpos, q, tool) > 0.0


def test_mast_keepout_refuses_poses_through_the_mast(ik):
    qpos = ik.data.qpos.copy()
    tp, tR = ik.trunk_pose(qpos)
    behind = tp + tR @ np.array([-0.45, 0.05, 0.40])          # the tool segment crosses the mast axis
    q_free, _ = ik.tcp_solve(qpos, behind, "T2", 0.0, gantry=0.0, keepout=False)
    assert q_free is not None and ik.keepout_clearance(qpos, q_free, "T2") < 0.0
    assert ik.tcp_solve(qpos, behind, "T2", 0.0, gantry=0.0)[0] is None


def test_gravity_loads_torque_and_predicted_com(ik):
    qpos = ik.data.qpos.copy()
    tp, tR = ik.trunk_pose(qpos)
    q, _ = ik.tcp_solve(qpos, tp + tR @ np.array([0.1, 0.62, 0.2]), "T4", 0.0, gantry=None)
    tool = [(0.35, np.array([0.17, 0.0, 0.0]))]
    fruit = tool + [(1.8, np.array([0.25, 0.0, 0.0]))]
    off0, off1, off2 = (ik.gravity_offsets(qpos, q, 0.0, loads=L) for L in ([], tool, fruit))
    assert abs(off2[2]) > abs(off1[2]) > abs(off0[2])
    t0, t2 = ik.shoulder_torque_nm(qpos, q), ik.shoulder_torque_nm(qpos, q, fruit)
    # (0.35 x 0.17 + 1.8 x 0.25 m beyond the wrist) on a ~0.4 m lever: the papaya case adds well over 5 N.m
    assert t2 - t0 > 5.0 and t2 < 59.5
    c0, c2 = ik.predicted_com(qpos, q), ik.predicted_com(qpos, q, fruit)
    left = tR[:, 1]
    assert float((c2 - c0) @ left) > 0.01                                       # the load pulls the CoM to the left
    assert np.allclose(ik.gravity_offsets_loads(qpos, q, fruit), off2)


# ---------------------------------------------------------------------- executor geometry helpers (pure)
def test_cut_and_twist_station_geometry():
    from atharv_nandi.core.tasks import sim_executor as sx
    t = tgt("prune_cut", "T5", point=(6.04, -3.6, 0.22), job="rootstock_sprout",
            meta={"a": [0.02, -0.056, 0.18], "b": [0.076, -0.205, 0.308]})
    st, cc = sx.cut_station(t, [(6.0, -3.5)])
    assert st is not None and cc <= math.sin(math.radians(20.0)) + 1e-9
    assert math.hypot(st["xy"][0] - 6.0, st["xy"][1] + 3.5) >= sx.ROBOT_CLEAR_TRUNK_M
    assert sx.best_cut_cos([0, 0, 1], [1, 0]) == pytest.approx(0.0)          # vertical pedicel, level approach
    assert sx.best_cut_cos([1, 0, 0], [1, 0]) == pytest.approx(0.0, abs=1e-9)  # radial shoot: from above
    # T4: the station bearing runs along a level pedicel (papaya radial into the trunk)
    pap = tgt("twist_pick", "T4", "papaya", point=(2.0, 10.85, 0.64), station={"xy": [2.4, 10.4], "heading_deg": 0.0},
              meta={"pedicel": [0.0, 0.08, 0.01]})
    st, err = sx.twist_station(pap, [(2.0, 11.0)])
    assert err <= sx.TWIST_BEARING_TOL_DEG and st["source"] == "wp8_twist_station"
    hang = tgt("twist_pick", "T4", "guava", station={"xy": [1.0, 1.0], "heading_deg": 90.0},
               meta={"pedicel": [0.0, 0.0, 0.03]})
    st, err = sx.twist_station(hang, [])
    assert err == 0.0 and st["xy"] != [1.0, 1.0]                               # backed off 0.10 m, WP0c otherwise
    b = sx.backed_off_station(hang)
    assert math.hypot(b["xy"][0] - 1.0, b["xy"][1] - 1.0) == pytest.approx(0.10)
    s2 = sx.station_for(tgt(point=(1.0, 0.5, 0.5)), (0.0, 0.0))
    h = math.radians(s2["heading_deg"])
    left = np.array([-math.sin(h), math.cos(h)])
    assert float(left @ (np.array([1.0, 0.5]) - np.array(s2["xy"]))) == pytest.approx(0.42, abs=1e-6)


# ---------------------------------------------------------------------- WP2 swap arm targets -> /nandi/arm_targets
def test_tool_arm_command_adds_feed_forward(ik):
    from atharv_nandi.core import harvest_logic as hl
    qpos = ik.data.qpos.copy()
    msg = {"q5": [0.0, -2.385, 0.3, -1.2, 1.0], "fingers": [0.01, 0.01], "payload_kg": 0.28, "phase": "rack_pre"}
    out = hl.tool_arm_command(json.dumps(msg), ik, qpos)
    off = ik.gravity_offsets(qpos, np.array(msg["q5"]), 0.28)
    assert out == pytest.approx([*(np.array(msg["q5"]) + off).tolist(), 0.01, 0.01])
    assert hl.tool_arm_command({**msg, "phase": "idle"}, ik, qpos) is None
    assert hl.tool_arm_command({**msg, "q5": None}, ik, qpos) is None
    assert hl.tool_arm_command("not json", ik, qpos) is None
    assert hl.tool_arm_command({**msg, "q5": [0, 0, "x", 0, 0]}, ik, qpos) is None
    no_f = hl.tool_arm_command({k: v for k, v in msg.items() if k != "fingers"}, ik, qpos, (0.0, 0.0))
    assert no_f[5:] == [0.0, 0.0]


def test_legacy_harvest_sequence_api_unchanged():
    from atharv_nandi.core import harvest_logic as hl
    assert hl.REST_ARM == (0.0, 0.0, 0.1, -1.8, 1.4, 0.0, 0.0) and hl.REACH_M == 1.08
    for name in ("HarvestSequence", "select_target", "grasp_check", "world_reset_detected", "status_json", "slew"):
        assert callable(getattr(hl, name))


@pytest.mark.parametrize("cls, task, tool", [(ClipPedicelSM, "clip_pedicel", "T2"), (TwistPickSM, "twist_pick", "T4"),
                                             (PruneCutSM, "prune_cut", "T5")])
def test_plan_and_approach_failures_skip_cleanly(cls, task, tool):
    """Every SM: an unplannable pre-pose skips ``plan_failed`` with no arm move; a failed approach backs out."""
    kw = {"diameter_mm": 6.0, "job": "rootstock_sprout"} if task == "prune_cut" else {}
    sm = cls(tgt(task, tool, **kw))
    drv = Drive(sm)
    drv.tick()
    drv.tick(arm_failed="plan_failed")
    assert (sm.outcome, sm.reason) == ("skipped", "plan_failed")
    sm = cls(tgt(task, tool, **kw))
    drv = Drive(sm)
    drv.tick()
    drv.tick(arm_done=True)
    drv.until("approach")
    c = drv.tick(arm_failed="blocked")
    assert sm.state == "retreat_skip" and c.arm == "retreat"
    drv.tick(arm_done=True)
    drv.tick(arm_done=True)
    assert (sm.outcome, sm.reason) == ("skipped", "plan_failed")


def test_twist_preconditions_bagged_band_torque_margin():
    for kw, pre, want in (({"bagged": True}, {}, ("skipped", "bagged")),
                          ({"point": (0.3, 0.0, 1.2), "in_band": False}, {}, ("marked", "out_of_reach_marked")),
                          ({}, {"torque_ok": False}, ("skipped", "torque_limit")),
                          ({}, {"margin_ok": False}, ("skipped", "stability"))):
        sm = TwistPickSM(tgt("twist_pick", "T4", "papaya", clearance_mm=20.0, **kw))
        Drive(sm).tick(precheck=pre)
        assert (sm.outcome, sm.reason) == want


def test_twist_lost_before_hold_and_place_refused():
    sm = TwistPickSM(tgt("twist_pick", "T4", "papaya", clearance_mm=20.0))
    drv = Drive(sm)
    twist_to_grip(sm, drv)
    drv.tick(n=20, touch_n=5.0, touch_max_n=5.0)
    drv.tick(events=[{"event": "twist_release"}])
    drv.tick(n=40, caught=False)                         # released but nothing caught
    assert sm.state == "slip"
    drv.tick(n=300)
    assert not sm.done                                   # the jaws stay shut while the executor lowers the fruit
    drv.tick(n=110)                                      # a lower that never reports done times out (8 s)
    assert (sm.outcome, sm.reason) == ("dropped", "dropped")
    sm = TwistPickSM(tgt("twist_pick", "T4", "papaya", clearance_mm=20.0))
    drv = Drive(sm)
    twist_to_grip(sm, drv)
    drv.tick(n=20, touch_n=5.0, touch_max_n=5.0)
    drv.tick(events=[{"event": "twist_release"}])
    drv.tick(n=12, caught=True, touch_max_n=5.0)
    drv.tick(arm_done=True, caught=True, touch_max_n=5.0)
    drv.tick(n=12, caught=True, touch_max_n=0.2)        # slip while parking
    assert sm.state == "slip"
    sm = TwistPickSM(tgt("twist_pick", "T4", "papaya", clearance_mm=20.0))
    drv = Drive(sm)
    twist_to_grip(sm, drv)
    drv.tick(n=20, touch_n=5.0, touch_max_n=5.0)
    drv.tick(events=[{"event": "twist_release"}])
    drv.tick(n=12, caught=True, touch_max_n=5.0)
    drv.tick(arm_done=True, caught=True, touch_max_n=5.0)
    drv.tick(arm_done=True, caught=True, touch_max_n=5.0)
    drv.tick(placed="refused:payload_full")
    assert (sm.outcome, sm.reason) == ("dropped", "place_refused:payload_full")


# ---------------------------------------------------------------------- WP8 review round 1 (interlocks, assist, MARKs)
@pytest.mark.parametrize("cls,task,tool,pre,roll", [(ClipPedicelSM, "clip_pedicel", "T2", 0.10, 0.0),
                                                    (TwistPickSM, "twist_pick", "T4", 0.15, 0.0),
                                                    (PruneCutSM, "prune_cut", "T5", 0.10, 0.0)])
def test_person_hold_and_base_pause_replan_from_the_pre_pose(cls, task, tool, pre, roll):
    """After a person hold or a base pause the SM asks for a fresh plan_pre (new IK + OMPL transit to the stand-off),
    not ``hold``: the executor treats hold as done, which used to skip the transit and re-close in place."""
    kw = {"clearance_mm": 20.0} if tool == "T4" else ({"diameter_mm": 6.0, "job": "rootstock_sprout"}
                                                      if tool == "T5" else {})
    sm = cls(tgt(task, tool, "papaya" if tool == "T4" else "mango", **kw))
    drv = Drive(sm)
    drv.tick()
    drv.tick(arm_done=True)
    drv.until("approach")
    drv.tick(arm_done=True)                          # approach -> align
    c = drv.tick(person=True)
    assert sm.state == "person_hold" and c.arm == "hold" and c.tool == "open"
    c = drv.tick(person=False)
    assert sm.state == "plan" and c.arm == "plan_pre" and c.rebase and c.tool == "open"
    assert c.arm_args["standoff_m"] == pytest.approx(pre) and c.roll_deg == pytest.approx(roll)
    drv.tick()                                       # the transit is running: plan waits for it
    assert sm.state == "plan"
    drv.tick(arm_done=True)
    drv.until("approach")
    c = drv.tick(base_drift_m=0.03)
    assert sm.state == "base_pause"
    c = drv.tick(base_settled=True)
    assert sm.state == "plan" and c.arm == "plan_pre" and c.rebase and c.arm_args["standoff_m"] == pytest.approx(pre)


def test_person_during_the_transit_pauses_the_plan_state():
    sm = ClipPedicelSM(tgt())
    drv = Drive(sm)
    drv.tick()
    assert sm.state == "plan"
    c = drv.tick(person=True)                        # the executor aborted the transit for the person
    assert sm.state == "person_hold" and c.arm == "hold"
    c = drv.tick(person=False)
    assert sm.state == "plan" and c.arm == "plan_pre"


def test_stability_loss_and_close_refused_for_stability_skip_before_the_commit():
    sm = ClipPedicelSM(tgt())
    drv = Drive(sm)
    run_clip_to_cut(sm, drv)
    c = drv.tick(events=[{"event": "actuate_refused", "reason": "stability", "tool": "T2"}])
    assert sm.state == "retreat_skip" and c.arm == "retreat" and c.tool == "open"
    assert c.arm_args["s"] == pytest.approx(-0.10)   # backs out along the axis (was s=0: towards the target)
    drv.tick(arm_done=True)
    drv.tick(arm_done=True)
    assert (sm.outcome, sm.reason) == ("skipped", "stability")
    # a refusal for another reason does not skip (no_target: the 1 s cut timeout / re-align handles it)
    sm = ClipPedicelSM(tgt())
    drv = Drive(sm)
    run_clip_to_cut(sm, drv)
    drv.tick(events=[{"event": "actuate_refused", "reason": "no_target"}])
    assert sm.state == "close"
    # the live interlock (Obs.stability_ok) while approaching
    sm = TwistPickSM(tgt("twist_pick", "T4", "papaya", clearance_mm=20.0))
    drv = Drive(sm)
    drv.tick()
    drv.tick(arm_done=True)
    drv.until("approach")
    c = drv.tick(stability_ok=False)
    assert sm.state == "retreat_skip" and c.arm_args["s"] == pytest.approx(-0.15)
    # after the commit the hold-verify -> place block is atomic: stability does not abort it
    sm = ClipPedicelSM(tgt())
    drv = Drive(sm)
    run_clip_to_cut(sm, drv)
    drv.tick(events=[{"event": "cut", "pedicel_mm": 20, "pedicel_error_mm": 0}])
    drv.tick(caught=True, stability_ok=False)
    assert sm.state == "retreat"


def test_executor_stability_interlock_reads_the_wp2_source():
    from types import SimpleNamespace
    from atharv_nandi.core.tasks.sim_executor import SimTaskExecutor
    ok = lambda v: SimTaskExecutor.stability_ok(SimpleNamespace(values=v))  # noqa: E731
    assert ok({}) and ok({"stability": {"margin_m": 0.05}}) and ok({"stability": "junk"})
    assert not ok({"stability": {"ok": False}}) and not ok({"stability": {"margin_m": 0.01}})


def test_twist_failed_raises_the_operator_assist_and_stays_marked():
    loop = tr.TargetLoop(["a", "b", "c"], tr.RetryPolicy(assist_auto_reply=None))
    d = loop.report("a", "marked", "twist_failed")
    assert d["decision"] == "assist" and d["options"] == ["take_over", "skip", "halt"] and loop.next_target() is None
    d = loop.assist_reply("skip")
    assert d["decision"] == "next" and loop.state["a"].outcome == "marked" and loop.state["a"].reason == "twist_failed"
    assert loop.state["a"].attempts == 1                  # never retried (the SM already re-gripped the other way)
    d = loop.report("b", "skipped", "twist_failed")
    assert d["decision"] == "assist"
    loop.assist_reply("take_over")
    assert (loop.state["b"].outcome, loop.state["b"].reason) == ("skipped", "operator_take_over")
    auto = tr.TargetLoop(["a", "b", "c"], tr.RetryPolicy())          # headless: the assist times out to Skip
    d = auto.report("a", "marked", "twist_failed")
    assert d["decision"] == "next" and d["assist"] == "skip" and d["auto"]
    assert auto.summary()["counts"] == {"marked": 1, "pending": 2}
    # grip_failed still retries first, then assists; a committed / dropped result never assists
    g = tr.TargetLoop(["a", "b"], tr.RetryPolicy(per_target=1, assist_auto_reply=None))
    assert g.report("a", "skipped", "grip_failed")["decision"] == "retry"
    assert g.report("a", "skipped", "grip_failed")["decision"] == "assist"
    c = tr.TargetLoop(["a", "b"], tr.RetryPolicy(assist_auto_reply=None))
    assert c.report("a", "dropped", "twist_failed", committed=True)["decision"] == "next"


def test_twist_sm_marks_twist_failed_then_the_loop_asks_for_assist():
    sm = TwistPickSM(tgt("twist_pick", "T4", "papaya", clearance_mm=20.0), retries=0)
    drv = Drive(sm)
    twist_to_grip(sm, drv)
    drv.tick(n=20, touch_n=5.0)
    drv.tick(touch_n=5.0, roll_travel_deg=541.0)
    drv.tick(arm_done=True)
    drv.tick(arm_done=True)
    assert (sm.outcome, sm.reason) == ("marked", "twist_failed")
    loop = tr.TargetLoop([sm.target.target_id], tr.RetryPolicy(assist_auto_reply=None))
    assert loop.report(sm.target.target_id, sm.outcome, sm.reason)["decision"] == "assist"


def test_slip_waits_for_the_lower_before_opening():
    sm = TwistPickSM(tgt("twist_pick", "T4", "papaya", clearance_mm=20.0))
    drv = Drive(sm)
    twist_to_grip(sm, drv)
    drv.tick(n=20, touch_n=5.0, touch_max_n=5.0)
    drv.tick(events=[{"event": "twist_release"}], touch_n=5.0, touch_max_n=5.0)
    drv.tick(n=12, caught=True, touch_max_n=5.0)
    c = drv.tick(n=15, caught=True, touch_max_n=0.3)
    assert sm.state == "slip" and c.arm == "lower" and c.tool is None
    c = drv.tick(n=100, caught=True)                 # 2 s of lowering: the jaws stay shut
    assert not sm.done and c.tool is None
    c = drv.tick(arm_done=True)
    assert (sm.outcome, sm.reason) == ("dropped", "dropped") and c.tool == "open"
