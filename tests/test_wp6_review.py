"""WP6 review round 1 fixes: the stability latch reaches the e-stop latch and clears, a restarted supervisor does not
re-arm a field latch blind, the CoM carries the WP2 tools and basket fruit, motion_inhibit is honoured by the gait
owners, the wind retract survives an e-stop hold, the raise gate fails safe, reverse walking gets d_stop at the rear,
the crown field scaling uses the lens height above the local ground, the bumper is sampled at 50 Hz.
"""
import json
import os

import mujoco
import numpy as np
import pytest

import atharv_nandi.sim.hooks as hook_registry
from atharv_nandi.core import farm_scene
from atharv_nandi.core import safety_fields as sf
from atharv_nandi.core import sim_driver as sd
from atharv_nandi.core import stability as st
from atharv_nandi.core.locomotion_logic import MotionCommand, MotionInhibitGate
from atharv_nandi.core.safety_logic import DEFAULT_CONFIG, REASON_PROTECTIVE, SafetyCore

FIELD_CORE_REASON = "stability"


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _core():
    core = SafetyCore(json.loads(json.dumps(DEFAULT_CONFIG)), wall_clock=_Clock())
    core.update_pose(0.0, (0.0, 0.0), 0.0, False)
    return core


# ====================================================================== pure
def test_restored_field_latch_is_not_reset_before_a_fresh_clear_payload():
    a = _core()
    a.update_fields(0.1, {"latched": True, "latch_reason": REASON_PROTECTIVE, "protective_clear": False})
    saved = a.state()
    b = _core()
    assert b.restore(saved, 5.0)
    ok, msg = b.reset(5.1)                                      # no payload seen since the restart: stale
    assert not ok and "protective" in msg
    b.update_fields(5.2, {"latched": False, "protective_clear": False})
    assert not b.reset(5.3)[0]                                  # payload says occupied
    b.update_fields(5.4, {"latched": False, "protective_clear": True})
    assert b.reset(5.5)[0]


def test_restored_operator_latch_still_resets_without_field_payloads():
    a = _core()
    a.engage_operator(0.1)
    b = _core()
    assert b.restore(a.state(), 1.0)
    assert b.reset(1.1)[0]


def test_stability_latch_payload_is_adopted_and_gates_the_reset():
    core = _core()
    assert core.update_fields(0.1, {"latched": True, "latch_reason": FIELD_CORE_REASON, "protective_clear": False,
                                    "detail": "stability margin < 0.01 m for 0.3 s"}) == FIELD_CORE_REASON
    assert core.latch.reason == FIELD_CORE_REASON
    assert not core.reset(0.2)[0]
    core.update_fields(0.3, {"latched": False, "protective_clear": True})
    assert core.reset(0.4)[0]


def test_crown_supervisor_wind_retract_re_fires_after_a_blocked_tick():
    sup = st.CrownSupervisor()
    sent = []
    t = 0.0
    while t < 1.2:
        holding = t < 1.0                                       # e-stop hold across the stance -> retract tick
        out = sup.update(t, 1.6, False, True, 0.1, 0.0, 0.0, wind_mps=9.0 if t < 0.1 else 10.5,
                         retracting=holding)
        if out["retract"]:
            sent.append((round(t, 2), out["retract"], holding))
        t += 0.02
    assert sent and not any(h for _, _, h in sent)
    assert sent[0][0] >= 1.0 and sent[0][1] == "wind"           # first tick after the hold
    # once a safety retract runs nothing more is sent; stowed -> idle
    out = sup.update(1.3, 1.0, False, True, 0.1, 0.0, 0.0, wind_mps=10.5, retracting=True)
    assert out["retract"] is None
    out = sup.update(1.4, 0.0, False, True, 0.1, 0.0, 0.0, wind_mps=10.5, retracting=False)
    assert out["wind_state"] == st.W_IDLE


def test_reverse_walking_gets_d_stop_at_the_rear():
    f = sf.field_offsets(sf.MODE_WALK, 0.3)["protective"]
    r = sf.field_offsets(sf.MODE_WALK, -0.3)["protective"]
    assert f["front"] == pytest.approx(0.25 + sf.d_stop(0.3)) and f["rear"] == pytest.approx(0.30)
    assert r["rear"] == pytest.approx(0.30 + sf.d_stop(0.3)) and r["front"] == pytest.approx(0.25)


def test_motion_inhibit_gate():
    g = MotionInhibitGate(max_age_s=0.5)
    walk = MotionCommand(0.6, 0.0, 0.2, True)
    assert g.filter_cmd(walk) == walk                           # nothing seen: nothing inhibited
    assert g.update({"stability": {"stance": True, "walk": True, "reasons": ["wind_stance"]}}, 1.0)
    assert g.halt and g.filter_cmd(walk) is None and g.filter_cmd(MotionCommand(0, 0, 0, False)) is not None
    assert not g.update({"stability": {"stance": True}}, 1.1)   # no new rising edge
    g.update({"stability": {}, "safety_fields": {"stop": False, "speed_scale": 0.3}}, 1.2)
    c = g.filter_cmd(walk)
    assert not g.halt and c.vx == pytest.approx(0.18) and c.yaw == pytest.approx(0.06)
    assert g.update({"safety_fields": {"stop": True, "reasons": ["protective"]}}, 1.3) and g.reasons == ("protective",)
    assert g.expire(2.0) and not g.halt and g.scale == 1.0


# ====================================================================== MuJoCo
def _world(**env):
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return farm_scene.build_world().spec.compile()
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def lite():
    return _world(NANDI_WORLD="orchard_lite", NANDI_TERRAIN="flat", NANDI_ACTORS="0")


def _rt():
    from atharv_nandi.sim.hooks import stability as hs
    return hs._RT[0]


def test_com_carries_the_welded_tools_and_basket_fruit(lite):
    m = lite
    drv = sd.SimDriver(m, hooks=hook_registry.load_all(names=["mast", "stability"]))
    drv.wait(0.3)
    rt = _rt()
    O = mujoco.mjtObj
    names = {mujoco.mj_id2name(m, O.mjOBJ_BODY, b) for b in rt.carried}
    assert {"clipper", "twist", "pruner"} <= names                       # rack welds (eq rack_*)
    tools_kg = sum(float(m.body_subtreemass[mujoco.mj_name2id(m, O.mjOBJ_BODY, n)]) for n in ("clipper", "twist", "pruner"))
    s = drv.bus.values["stability"]
    assert s["mass_kg"] == pytest.approx(float(m.body_subtreemass[rt.trunk]) + tools_kg, abs=1e-3)
    assert not any(n.startswith("pool_") for n in names)                 # on the tree (welded to the world)
    # drop a free crop into the basket: it joins the CoM
    crop = mujoco.mj_name2id(m, O.mjOBJ_BODY, "crop_item_1")
    j = int(m.body_jntadr[crop])
    qa = int(m.jnt_qposadr[j])
    R, p0 = rt.frame()
    drv.d.qpos[qa:qa + 3] = p0 + R @ np.array([0.0, 0.0, 0.09])
    drv.d.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
    drv.d.qvel[int(m.jnt_dofadr[j]):int(m.jnt_dofadr[j]) + 6] = 0.0
    mujoco.mj_forward(m, drv.d)
    drv.wait(0.1)
    assert crop in rt.carried
    assert drv.bus.values["stability"]["mass_kg"] == pytest.approx(s["mass_kg"] + float(m.body_subtreemass[crop]), abs=1e-3)
    drv.close()


def test_stability_latch_rides_the_field_payload_and_clears_in_the_hold(lite, monkeypatch):
    drv = sd.SimDriver(lite, hooks=hook_registry.load_all(names=["mast", "stability"]))
    drv.wait(0.5)
    rt = _rt()
    core = _core()
    seen = []

    def feed(dr):
        f = dr.bus.values.get("safety_fields")
        if f and f.get("ok", True):
            core.update_fields(dr.t, f)
            seen.append((dr.t, f["latched"], f["latch_reason"], f["protective_clear"]))

    drv.add_task("supervisor", feed, rate_hz=10.0)
    drv.wait(0.3)
    assert core.cause_clear() and not core.latch.estopped
    monkeypatch.setattr(st, "LATCH_MARGIN_M", 5.0)              # no stance margin is >= 5 m: the latch trips for real
    drv.wait(0.6)
    assert rt.latch.latched
    assert core.latch.estopped and core.latch.reason == FIELD_CORE_REASON
    assert drv.bus.values["motion_inhibit"]["stability"]["stop"]
    drv.enter_hold(FIELD_CORE_REASON)                            # sim_runner adopts the e-stop latch
    drv.wait(0.05)
    assert rt.latch.latched and not core.reset(drv.t)[0]         # refused while the stability cause is not clear
    drv.wait(0.8)
    assert rt.latch.latched and not core.reset(drv.t)[0]         # the cause persists: still refused
    monkeypatch.setattr(st, "LATCH_MARGIN_M", 0.01)             # the cause goes away
    drv.wait(0.8)                                                # margin ok for >= 0.5 s: handed over, cleared
    assert not rt.latch.latched and drv.bus.values["stability"]["stability_clear"]
    assert seen[-1][1] is False and seen[-1][3] is True
    ok, msg = core.reset(drv.t)
    assert ok, msg
    drv.leave_hold()
    drv.wait(0.2)
    assert not drv.bus.values["motion_inhibit"]["stability"]["stop"]
    drv.close()


def test_driver_stops_the_gait_on_a_stance_request_and_scales_on_warning(lite):
    drv = sd.SimDriver(lite, hooks=hook_registry.HookRegistry())   # no hooks: the test writes motion_inhibit
    drv.wait(0.2)
    drv.walk(0.6)
    drv.wait(0.6)
    assert drv.gait.is_walking
    drv.bus.values["motion_inhibit"] = {"stability": {"stance": True, "walk": True, "reasons": ["wind_stance"]}}
    drv.wait(0.05)
    assert not drv.gait.is_active
    drv.walk(0.6)                                                # refused while inhibited
    drv.wait(0.1)
    assert not drv.gait.is_active
    drv.bus.values["motion_inhibit"] = {"stability": {}, "safety_fields": {"stop": False, "speed_scale": 0.3}}
    drv.walk(0.6)
    drv.wait(0.1)
    assert drv.gait.cmd[0] == pytest.approx(0.18)
    drv.bus.values["motion_inhibit"] = {"stability": {}, "safety_fields": {"stop": False, "speed_scale": 1.0}}
    drv.wait(0.1)
    assert drv.gait.cmd[0] == pytest.approx(0.6)                 # the scale lifts, no compounding
    drv.close()


def test_raise_gate_fails_safe_without_a_bound_model(lite):
    drv = sd.SimDriver(lite, hooks=hook_registry.load_all(names=["mast", "stability"]))
    drv.wait(0.2)
    rt = _rt()
    assert not rt.raise_gate(0.0, 0.6, {"bus": drv.bus})
    rt.ok = False
    try:
        dec = rt.raise_gate(0.0, 0.6, {"m": drv.m, "d": drv.d, "bus": drv.bus})
        assert not dec and dec.values.get("detail") == "stability_unbound"
    finally:
        rt.ok = True
    drv.close()


def test_lens_height_is_above_the_local_ground_not_world_z(lite):
    drv = sd.SimDriver(lite, hooks=hook_registry.load_all(names=["mast", "stability"]))
    drv.wait(0.3)
    rt = _rt()
    h0 = rt.lens_height()
    z0 = float(drv.d.cam_xpos[rt.cam][2])
    # the whole scene 0.08 m lower (a dip): world z drops, the height above the local ground does not
    drv.d.qpos[2] -= 0.08
    mujoco.mj_forward(drv.m, drv.d)
    assert float(drv.d.cam_xpos[rt.cam][2]) == pytest.approx(z0 - 0.08, abs=1e-3)
    assert rt.lens_height() == pytest.approx(h0, abs=1e-3)
    assert h0 == pytest.approx(z0, abs=0.02)                     # flat ground at z = 0: the same as world z
    drv.close()


def test_bumper_touch_between_field_ticks_is_not_missed(lite):
    drv = sd.SimDriver(lite, hooks=hook_registry.load_all(names=["mast", "stability"]))
    drv.wait(0.3)
    rt = _rt()
    if rt.bumper_adr < 0:
        pytest.skip("no bumper sensor in this model")
    rt.bumper_seen = True                                         # a 20 ms touch seen by the 50 Hz tick only
    t0 = drv.t
    while drv.bus.values["safety_fields"].get("t", -1) <= t0 + 1e-9:
        drv.run_steps(1)
    f = drv.bus.values["safety_fields"]
    assert f["latched"] and f["latch_reason"] == sf.R_BUMPER
    drv.close()
