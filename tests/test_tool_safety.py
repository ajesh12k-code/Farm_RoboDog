"""WP2 review round 1: tools runtime safety (MVP-SPEC 2.3 interlocks and blade-close interlock, 3.4 cut rule;
ORCHARD-MVP-SPEC 3.4). ROS-free; MuJoCo headless through ``SimDriver``.

    a papaya (no cut force) assigned while T2 is mounted never faults the hook; it is refused, not cut
    no cut and no twist release while e-stopped; a closed blade needs a fresh closing after the hold
    person / crown up / low margin DURING a swap fault it; the mast refuses raises while a swap runs
    malformed /nandi/tool/change messages are refused bad_request (never a dock, never a hook fault)
    {"ack_fault"} and {"release"} reach clear_fault / release_catch over the topic
    blade / jaw close refused without a target in 0.15 m, with a person in the WORK field or stability not OK
    the T1 picker holding a crop interlocks the swap; verify checks tool presence and the empty cradle
"""
import mujoco
import numpy as np
import pytest

from atharv_nandi.sim import hooks
import atharv_nandi.sim.hooks.mast as mast_hook
import atharv_nandi.sim.hooks.tools as ht
from atharv_nandi.core.sim_driver import SimDriver
from atharv_nandi.core.tools.tool_changer import SwapObs, ToolChanger
from test_cut_logic import clip, mount, move_slot, stage_mango, world  # noqa: F401 - fixture
from test_tool_changer import _good, _plan, _run, orchard  # noqa: F401 - fixture

PAPAYA = "L2-3:f01"


def hook_world(orchard_build):
    m = orchard_build("orchard_lite")
    reg_ = hooks.HookRegistry()
    ht.register(reg_)
    drv = SimDriver(m, hooks=reg_)
    drv.wait(0.1)
    return m, drv, reg_, ht._RUNTIMES[m]


def force_mount(rt, tool):
    """Lock ``tool`` onto the wrist without the arm motion (the hook path has no arm follower)."""
    rt.plant.apply_ops([("lock_grip", tool)])
    rt.changer.mounted = tool


# ---------------------------------------------------------------------------------------------- cut targets
def test_papaya_assigned_with_t2_mounted_is_refused_not_a_hook_fault(world):
    m, drv, rt = world()
    mount(drv, rt, "T2")
    d, pool = drv.d, rt.pool
    kp = pool.assign(d, PAPAYA)                        # far away: skipped by the 3 cm range filter
    drv.wait(0.2)
    ids = rt.h["T2"]
    move_slot(m, d, pool, kp, d.site_xpos[ids.blade_a])   # papaya stalk between the blades
    n0 = len(rt.events)
    assert rt.actuate({"value": 1.0})
    drv.wait(0.4)
    ev = rt.events[n0:]
    assert [e["event"] for e in ev] == ["cut_refused"] and ev[0]["reason"] == "no_cut_force"
    assert pool.is_attached(d, kp)
    assert rt.actuate({"value": 0.0})
    drv.wait(0.2)
    move_slot(m, d, pool, kp, d.site_xpos[ids.tcp] + np.array([0.0, 0.0, 0.5]))   # out of the way
    k, s = stage_mango(m, drv, rt, 0.02)               # a mango with the papaya still assigned: cuts
    _err, ev, _arm, _ax = clip(m, drv, rt, k, s)
    assert [e["event"] for e in ev if e["event"].startswith("cut")] == ["cut"]


def test_papaya_on_the_live_hook_path_never_disables_the_hook(orchard):
    m, drv, reg_, rt = hook_world(orchard)
    force_mount(rt, "T2")
    drv.wait(0.2)
    rt.pool.assign(drv.d, PAPAYA)
    rt.pool.assign(drv.d, "L2-2:f01")                  # a guava after it
    drv.wait(0.3)
    assert reg_.faulted() == {} and not drv.holding


# ---------------------------------------------------------------------------------------------- e-stop
def test_no_cut_while_estopped_and_a_fresh_closing_after_it(world):
    m, drv, rt = world()
    mount(drv, rt, "T2")
    k, s = stage_mango(m, drv, rt, 0.02)
    _err, ev, _arm, _ax = clip(m, drv, rt, k, s, side_mm=12.0)
    assert [e["event"] for e in ev] == ["cut_miss"]     # closed, 12 mm beside the stalk
    drv.enter_hold("estop")
    drv.wait(0.1)
    ids = rt.h["T2"]
    assert rt.act.stopped and drv.d.ctrl[ids.act_act] > 0.95   # the blade is held closed
    n0 = len(rt.events)
    move_slot(m, drv.d, rt.pool, k, drv.d.site_xpos[ids.blade_a])   # the stalk drifts into the trap
    drv.wait(0.5)
    assert rt.pool.is_attached(drv.d, k) and not [e for e in rt.events[n0:] if e["event"] == "cut"]
    drv.leave_hold()
    drv.wait(0.5)
    assert rt.pool.is_attached(drv.d, k) and not [e for e in rt.events[n0:] if e["event"] == "cut"]
    assert rt.actuate({"value": 0.0})
    drv.wait(0.2)
    move_slot(m, drv.d, rt.pool, k, drv.d.site_xpos[ids.blade_a])   # the re-arm blend moved the arm: re-aim
    assert rt.actuate({"value": 1.0})
    drv.wait(0.3)
    assert [e["event"] for e in rt.events[n0:]] == ["cut"], rt.events[n0:] and not rt.pool.is_attached(drv.d, k)


# ---------------------------------------------------------------------------------------------- swap interlocks
@pytest.mark.parametrize("key, val, reason", [("work_field", {"person": True}, "person_in_work_field"),
                                              ("mast", {"ext_m": 1.2}, "crown_up"),
                                              ("stability", {"margin_m": 0.0}, "low_margin")])
def test_interlock_during_a_swap_faults_it(world, key, val, reason):
    m, drv, rt = world()
    assert rt.request_change("T2")
    drv.wait(0.5)
    assert rt.changer.phase == "transit"
    drv.bus.values[key] = val
    drv.wait(0.1)
    ev = [e for e in rt.events if e["event"].startswith("tool_change")]
    assert ev[-1]["event"] == "tool_change_fault" and ev[-1]["reason"] == reason
    q_frozen = tuple(rt.changer.q_cmd)
    drv.wait(1.5)
    if reason == "person_in_work_field":                 # the arm does not move while the person is there
        assert rt.changer.phase == "safe_park" and tuple(rt.changer.q_cmd) == q_frozen
        drv.bus.values.pop(key)
        drv.wait(1.5)
    assert rt.changer.phase == "fault" and rt.changer.mounted is None
    assert np.allclose(rt.changer.q_cmd, rt.plant.q_home)
    for ids in rt.h.tools.values():
        assert drv.d.eq_active[ids.rack_eq] + drv.d.eq_active[ids.grip_eq] == 1


def test_queued_job_meets_an_interlock_before_it_starts():
    ch = ToolChanger()
    from atharv_nandi.core.tools.tool_changer import Interlocks
    assert ch.request([_plan()], Interlocks(), (0.0,) * 5)
    cmd = ch.step(SwapObs(t=0.0, interlock="person_in_work_field"))
    assert cmd.events[0]["event"] == "tool_change_fault" and cmd.events[0]["reason"] == "person_in_work_field"
    assert ch.phase == "safe_park" and ch.job is None and not ch.queue


def test_mast_refuses_raises_while_a_swap_runs(world):
    m, drv, rt = world()
    assert ht._mast_gate in mast_hook._GATES
    assert ht._mast_gate(0.0, 0.6, {"m": m})
    rt.request_change("T5")
    drv.wait(0.3)
    g = ht._mast_gate(0.0, 0.6, {"m": m})
    assert not g and g.reason == "tool_change_busy"
    assert ht._mast_gate(0.0, 0.6, {"m": None})          # another world: not gated


# ---------------------------------------------------------------------------------------------- messages
def test_malformed_change_messages_are_refused(world):
    m, drv, rt = world()
    mount(drv, rt, "T5")
    for msg in ({"tol": "T2"}, ["T2"], None, {"tool": 5}, "null", "T9", 3.0, {"tool": ["T2"]}):
        n0 = len(rt.events)
        rt.change_message(msg)
        ev = rt.events[n0:]
        assert len(ev) == 1 and ev[0]["event"] == "tool_change_refused", (msg, ev)
        assert ev[0]["reason"].startswith("bad_request"), (msg, ev)
        assert not rt.changer.busy and rt.changer.mounted == "T5"
    rt.change_message("T2")                              # a bare tool id is a change request
    assert rt.changer.busy and rt.changer.state()["queued"] == [("T5", "dock"), ("T2", "mount")]


def test_malformed_change_on_the_hook_never_faults_it(orchard):
    m, drv, reg_, rt = hook_world(orchard)
    for raw in ('["T2"]', '{"tol": "T2"}', "{not json", "5"):
        n0 = len(rt.events)
        drv.deliver(ht.CHANGE_TOPIC, raw)
        drv.wait(0.05)
        assert [e["event"] for e in rt.events[n0:]] == ["tool_change_refused"], raw
    assert reg_.faulted() == {} and not drv.holding and not rt.changer.busy
    drv.deliver(ht.CHANGE_TOPIC, "T2")                   # plain String, not JSON-quoted
    drv.wait(0.05)
    assert rt.changer.state()["job"] == {"tool": "T2", "direction": "mount"}


def test_ack_fault_and_release_over_the_topic(orchard):
    m, drv, reg_, rt = hook_world(orchard)
    drv.deliver(ht.CHANGE_TOPIC, {"tool": "T2"})         # no arm follows on this path: grip_align
    assert drv.run_until(lambda dr: rt.changer.phase == "fault", 8.0)
    assert rt.changer.reason == "grip_align"
    drv.deliver(ht.CHANGE_TOPIC, {"tool": "T4"})
    drv.wait(0.05)
    assert rt.events[-1]["reason"] == "fault_needs_assist"
    drv.deliver(ht.CHANGE_TOPIC, {"ack_fault": True})
    drv.wait(0.05)
    assert rt.events[-1]["event"] == "tool_fault_cleared" and rt.changer.phase == "idle"
    drv.deliver(ht.CHANGE_TOPIC, {"ack_fault": True})
    drv.wait(0.05)
    assert rt.events[-1] == {**rt.events[-1], "event": "ack_fault_refused", "reason": "no_fault"}
    assert rt._catch("clipper", 0)                       # pool_0 held by the clipper catch weld
    e = rt.caught["eq"]
    drv.deliver(ht.CHANGE_TOPIC, '{"release": true}')
    drv.wait(0.05)
    assert rt.events[-1]["event"] == "catch_released" and rt.events[-1]["slot"] == 0
    assert drv.d.eq_active[e] == 0 and not rt.caught
    drv.deliver(ht.CHANGE_TOPIC, {"release": True})
    drv.wait(0.05)
    assert rt.events[-1]["event"] == "release_refused" and rt.events[-1]["reason"] == "nothing_caught"
    assert reg_.faulted() == {}


# ---------------------------------------------------------------------------------------------- blade-close
def test_blade_close_interlock(world):
    m, drv, rt = world()
    mount(drv, rt, "T2")
    ids = rt.h["T2"]
    for vals, reason in (({"work_field": {"person": True}}, "person_in_work_field"),
                         ({"stability": {"ok": False}}, "stability"),
                         ({"stability": {"margin_m": 0.01}}, "stability"),
                         ({}, "no_target")):
        drv.bus.values.pop("work_field", None)
        drv.bus.values.pop("stability", None)
        drv.bus.values.update(vals)
        drv.wait(0.05)
        assert not rt.actuate({"value": 1.0}) and rt.events[-1]["reason"] == reason
        assert drv.d.ctrl[ids.act_act] == 0.0
    drv.bus.values["work_field"] = {"person": True}
    assert rt.actuate({"value": 0.0}) and rt.actuate({"roll": 0.3}) and rt.actuate({"open": True})   # opening is free
    drv.bus.values.pop("work_field")
    k = rt.pool.assign(drv.d, "L2-1:f07")
    move_slot(m, drv.d, rt.pool, k, drv.d.site_xpos[ids.tcp] + np.array([0.0, 0.0, 0.10]))   # tracked, 10 cm
    assert rt.actuate({"value": 1.0})


def test_jaw_close_interlock_t4(world):
    m, drv, rt = world()
    mount(drv, rt, "T4")
    ids = rt.h["T4"]
    assert not rt.actuate({"value": 0.0}) and rt.events[-1]["reason"] == "no_target"
    assert rt.actuate({"value": 0.16})                  # open
    k = rt.pool.assign(drv.d, PAPAYA)
    move_slot(m, drv.d, rt.pool, k, drv.d.site_xpos[ids.tcp] + np.array([0.0, 0.0, 0.25]), site=None)
    assert not rt.actuate({"value": 0.0}) and rt.events[-1]["reason"] == "no_target"   # 25 cm: not tracked
    move_slot(m, drv.d, rt.pool, k, drv.d.site_xpos[ids.tcp] + np.array([0.0, 0.0, 0.12]), site=None)
    assert rt.actuate({"value": 0.0})


# ---------------------------------------------------------------------------------------------- picker + verify
def test_picker_holding_a_crop_interlocks_the_swap(world):
    m, drv, rt = world()
    d = drv.d
    assert not rt.picker_holding()
    k = rt.pool.assign(d, "L2-1:f07")
    tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
    move_slot(m, d, rt.pool, k, d.site_xpos[tip], site=None)    # the mango between the open fingers
    drv.wait(0.2)
    assert not rt.picker_holding()
    drv.arm = list(drv.arm[:5]) + [0.04, 0.04]                  # close the fingers on it
    drv.wait(0.6)
    assert rt.picker_holding() and rt.holding()
    g = rt.request_change("T2")
    assert not g and g.reason == "holding_payload"
    drv.arm = list(drv.arm[:5]) + [0.0, 0.0]
    drv.wait(0.6)
    assert not rt.picker_holding()


@pytest.mark.parametrize("direction, obs_kw, reason", [
    ("mount", {"tool_present": False}, "verify"),
    ("dock", {"cradle_occupied": False}, "verify"),
    ("mount", {"cradle_occupied": True}, "verify_presence"),     # tool still in the cradle after the lift
    ("dock", {"tool_present": True}, "verify_presence"),         # tool still on the hand after the lift
])
def test_verify_checks_presence(direction, obs_kw, reason):
    ch = ToolChanger(mounted="T2" if direction == "dock" else None)
    from atharv_nandi.core.tools.tool_changer import Interlocks
    assert ch.request([_plan("T2", direction)], Interlocks(), (0.0,) * 5)
    good_present = {"tool_present": direction == "mount", "cradle_occupied": direction == "dock"}

    def obs(t, c):
        o = _good(t, c)
        o.tool_present, o.cradle_occupied = good_present["tool_present"], good_present["cradle_occupied"]
        if reason == "verify":
            for k_, v in obs_kw.items():
                setattr(o, k_, v)
        elif c.phase == "lift":
            for k_, v in obs_kw.items():
                setattr(o, k_, v)
        return o

    _phases, events, _t = _run(ch, obs)
    assert events[-1]["event"] == "tool_change_fault" and events[-1]["reason"] == reason, events
