"""WP6 e-stop: tools safe + crown brake, field latches adopted by the supervisor core, e-stop timings, sit-down.

Acceptance (WP6; ORCHARD-MVP-SPEC 2.3 rule 6, 5.3; MVP-SPEC 5.6): e-stop timings + crown brake
(the sections hold in place while e-stopped); WP6 codes the e-stop against ``ToolActuation`` (blade opens, tool
stops); a protective-field latch becomes an e-stop latch that cannot be reset while the field is occupied; the
e-stop sit-down at ext 1.6 on 5 deg cross-slope does not fall and records its support-polygon margin.
"""
import json
import os
import sys

import numpy as np
import pytest

import atharv_nandi.sim.hooks as hook_registry
from atharv_nandi.core import farm_scene
from atharv_nandi.core import sim_driver as sd
from atharv_nandi.core.interfaces import MastRequest, NullMastControl, NullToolActuation
from atharv_nandi.core.safety_logic import (DEFAULT_CONFIG, FIELD_REASONS, REASON_ANIMAL, REASON_HANDHELD,
                                      REASON_PROTECTIVE, EstopActions, SafetyCore)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GATES_JSON = os.path.join(REPO, "config", "crown_gates.json")


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ====================================================================== pure
def test_estop_actions_open_blades_stop_tools_and_brake_the_crown():
    tools = [NullToolActuation(), NullToolActuation()]
    mast = NullMastControl()
    ea = EstopActions(tools, mast)
    assert ea.engage(1.0, "operator")
    assert all(t.log[:2] == ["open_blade", "stop"] and t.stopped for t in tools)
    assert mast.braked
    assert ea.log[-1]["actions"] == ["tool_safe", "tool_safe", "crown_brake"]
    assert not ea.engage(1.1)                                   # once per latch
    tools[0].set(0.8)
    assert tools[0].value == 0.0 and tools[0].log[-1] == "set refused (stopped)"
    ea.release()
    assert ea.engage(2.0)


def test_estop_actions_survive_a_failing_tool():
    class Bad:
        def open_blade(self):
            raise RuntimeError("jammed")

        def stop(self):
            pass

    good = NullToolActuation()
    ea = EstopActions([Bad(), good], NullMastControl())
    ea.engage(0.0)
    assert ea.log[-1]["actions"][0] == "tool_error:RuntimeError" and good.stopped
    assert ea.log[-1]["actions"][-1] == "crown_brake"


def test_field_latch_is_adopted_and_reset_waits_for_a_clear_field():
    core = SafetyCore(json.loads(json.dumps(DEFAULT_CONFIG)), wall_clock=_Clock())
    core.update_pose(0.0, (0.0, 0.0), 0.0, False)
    assert core.update_fields(0.1, {"latched": False, "protective_clear": True}) is None
    assert core.update_fields(0.2, {"latched": True, "latch_reason": "animal", "protective_clear": False}) == REASON_ANIMAL
    assert core.latch.estopped and core.latch.reason == REASON_ANIMAL
    ok, msg = core.reset(0.3)
    assert not ok and "animal" in msg
    core.update_fields(0.9, {"latched": False, "protective_clear": True})     # hook handed over, field clear
    assert core.reset(1.0)[0]
    core.update_fields(1.1, {"latched": True, "latch_reason": "weird", "protective_clear": False})
    assert core.latch.reason == REASON_PROTECTIVE                              # unknown reason -> protective
    core.update_fields(1.2, {"latched": False, "protective_clear": True})
    assert not core.reset(5.0)[0]                                              # stale fields (> 1 s): refused
    assert core.reset(1.5)[0]
    assert set(FIELD_REASONS) >= {"protective", "animal", "bumper", "stability"}


def test_handheld_press_latches_and_reset_waits_for_the_release():
    """WP1 change request: the supervisor latches on /nandi/handheld_estop/state estop_request (the WP1 decoder's
    own state dict); a reset is refused while the mushroom is pressed, also when the handheld then goes silent,
    and after a restart until a fresh state says released."""
    from atharv_nandi.sim.hooks.beacon import HandheldEstop
    hh = HandheldEstop()
    core = SafetyCore(json.loads(json.dumps(DEFAULT_CONFIG)), wall_clock=_Clock())
    core.update_pose(0.0, (0.0, 0.0), 0.0, False)
    assert core.update_handheld(0.0, hh.state(0.0)) is None and not core.latch.estopped      # never heard
    hh.update({"device": "hh-1", "seq": 1, "pressed": False}, 0.1)
    assert core.update_handheld(0.1, hh.state(0.1)) is None and not core.latch.estopped
    hh.update({"device": "hh-1", "seq": 2, "pressed": True}, 0.2)
    assert core.update_handheld(0.2, hh.state(0.2)) == REASON_HANDHELD
    assert core.latch.estopped and core.latch.reason == REASON_HANDHELD
    assert core.events[-1]["reason"] == REASON_HANDHELD and "hh-1" in core.events[-1]["detail"]
    ok, msg = core.reset(0.3)
    assert not ok and "handheld" in msg
    core.update_handheld(1.5, hh.state(1.5))                  # silent handheld: link lost, the press stands
    assert hh.state(1.5)["local_link_lost"] and not core.reset(1.5)[0]
    hh.update({"device": "hh-1", "seq": 3, "pressed": False}, 2.0)
    core.update_handheld(2.0, hh.state(2.0))
    assert not core.reset(4.0)[0]                             # released, but the state is stale (> 1 s)
    core.update_handheld(4.0, hh.state(4.0))
    assert core.reset(4.1)[0] and not core.latch.estopped
    # an operator latch with the handheld pressed meanwhile: the press is a cause too
    core.engage_operator(5.0)
    hh.update({"device": "hh-1", "seq": 4, "pressed": True}, 5.1)
    core.update_handheld(5.1, hh.state(5.1))
    assert core.latch.reason == "operator" and not core.reset(5.2)[0]
    # restart with a saved handheld latch: pressed until a fresh payload says released
    saved = core.state()
    core2 = SafetyCore(json.loads(json.dumps(DEFAULT_CONFIG)), wall_clock=_Clock())
    saved["reason"] = REASON_HANDHELD
    assert core2.restore(saved, 0.0) and core2.latch.reason == REASON_HANDHELD
    core2.update_pose(0.0, (0.0, 0.0), 0.0, False)
    assert not core2.reset(0.1)[0]
    core2.update_handheld(0.2, {"seen": True, "pressed": False, "estop_request": False})
    assert core2.reset(0.3)[0]


# ====================================================================== MuJoCo
@pytest.fixture(scope="module")
def flat_spec():
    keep = {k: os.environ.get(k) for k in ("NANDI_WORLD", "NANDI_TERRAIN", "NANDI_ACTORS")}
    os.environ.update(NANDI_WORLD="orchard_lite", NANDI_TERRAIN="flat", NANDI_ACTORS="0")
    try:
        return farm_scene.build_world().spec
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def flat_model(flat_spec):
    return flat_spec.compile()


def test_estop_while_walking_crown_up_brakes_the_crown_and_stops(flat_model):
    m = flat_model
    drv = sd.SimDriver(m, hooks=hook_registry.load_all(names=["mast", "stability"]))
    tool = NullToolActuation()
    drv.bus.values["tool_actuation"] = tool
    drv.wait(0.3)
    mrt = drv.bus.values["mast_control"]
    mrt.arb.reset(1.6)
    drv.wait(0.5)
    drv.walk(0.62)                                  # ~0.15 m/s, the crown-up walking cap
    drv.wait(2.0)
    v0 = float(np.linalg.norm(drv.d.qvel[0:2]))
    drv.enter_hold("estop")
    t_e = drv.t
    exts = []
    for _ in range(100):
        drv.wait(0.02)
        exts.append(drv.bus.values["mast"]["ext_m"])
    st = drv.bus.values["stability"]["estop"]
    assert st["actions"] == ["tool_safe", "crown_brake"] and tool.stopped
    assert drv.bus.values["mast"]["braked"]
    assert max(exts) - min(exts) < 1e-9 and exts[0] == pytest.approx(1.6)       # sections hold in place
    stop = st["stopped"]
    assert stop is not None, "trunk never stopped"
    print(f"\n[estop] v0 {v0:.3f} m/s -> stop {stop}")
    # driver hold = joints frozen at PLAIN gains: a regression bound only, NOT the P-08 bar (stationary <= 0.5 s).
    # P-08 is checked with the sim_runner hold physics in test_estop_crown_up_stop_time_against_p08 below.
    assert stop["t_s"] <= 2.0 and stop["dist_m"] <= 0.05
    # a retract during the hold is refused (e-stop beats safety); it needs a new request after the hold
    dec = mrt.submit(MastRequest("safety", 0.0, "test", t=drv.t))
    drv.wait(0.2)
    assert drv.bus.values["mast"]["ext_m"] == pytest.approx(1.6)
    drv.leave_hold()
    drv.wait(0.1)
    assert not drv.bus.values["mast"]["braked"]
    assert drv.bus.values["stability"]["estop"]["t_engage"] == pytest.approx(t_e, abs=0.03)
    del dec
    drv.close()


def test_estop_sitdown_at_ext_1p6_on_5_deg_is_recorded_without_a_fall():
    with open(GATES_JSON, encoding="utf-8") as fh:
        sd_block = json.load(fh)["sitdown"]
    assert sd_block["meta"]["run_id"].startswith("wp6-sitdown-")
    assert sd_block["no_fall"] is True
    cases = sd_block["cases"]
    assert {c["rack"] for c in cases} == {"downhill", "uphill"} and all(c["ext"] == 1.6 for c in cases)
    assert all(abs(c["roll_deg"]) == 5.0 and c["margin_m"] is not None and c["margin_m"] > 0 for c in cases)
    from tools import crown_gate_matrix as cgm
    live = cgm.sitdown_case(5.0, kp_scale=0.0, seconds=3.0)            # power-off sit-down, rack downhill
    assert not live["fell"] and live["margin_m"] > 0.1
    assert live["trunk_z_after_m"] < live["trunk_z_before_m"] - 0.15


def _runner_hold_physics(drv, on: bool, base=None):
    """sim_runner._set_hold_physics on the driver's model (config/safety.yaml estop.hold_*): kp / kv scaled and the
    joint brake on every robot actuator but the fingers (arm joints half). Returns the saved model fields."""
    from atharv_nandi.core import model_layout as mlay
    from atharv_nandi.core.safety_logic import load_safety_config
    m = drv.m
    if not on:
        m.actuator_gainprm[:], m.actuator_biasprm[:], m.dof_frictionloss[:] = base
        return None
    base = (m.actuator_gainprm.copy(), m.actuator_biasprm.copy(), m.dof_frictionloss.copy())
    est = load_safety_config()["estop"]
    i = np.array([k for k in range(mlay.ROBOT_NU) if k not in set(drv._finger_idx)], dtype=np.int64)
    kp, kv, br = float(est["hold_stiffness_scale"]), float(est["hold_damping_scale"]), float(est["hold_brake_nm"])
    m.actuator_gainprm[i, 0] *= kp
    m.actuator_biasprm[i, 1] *= kp
    m.actuator_biasprm[i, 2] *= kv
    dofs = np.array([int(m.jnt_dofadr[m.actuator_trnid[a, 0]]) for a in i], dtype=np.int64)
    m.dof_frictionloss[dofs] += np.where(i < 7, 0.5, 1.0) * br
    return base


P08_STOP_S = 0.5      # MVP-SPEC P-08 / pilot plan: stationary within 500 ms of the e-stop


def test_estop_crown_up_stop_time_against_p08(flat_spec):
    worst = 0.0
    rows = []
    # gait phases at engage (2.13 s was the worst phase of the sweep), crown up at the crown-up walking cap and
    # stowed at full command; each run on a fresh model (the mast hook writes body_pos)
    for ext, vx, walk_s in ((1.6, 0.62, 2.0), (1.6, 0.62, 2.13), (1.6, 0.62, 2.27), (1.6, 1.0, 2.13),
                            (0.0, 1.0, 2.13), (0.0, 1.0, 2.0)):
        m = flat_spec.compile()
        drv = sd.SimDriver(m, hooks=hook_registry.load_all(names=["mast", "stability"]))
        drv.wait(0.3)
        if ext:
            drv.bus.values["mast_control"].arb.reset(ext)
        drv.wait(0.5)
        drv.walk(vx)
        drv.wait(walk_s)
        base = _runner_hold_physics(drv, True)
        try:
            drv.enter_hold("estop")
            drv.wait(1.5)
            stop = drv.bus.values["stability"]["estop"]["stopped"]
        finally:
            _runner_hold_physics(drv, False, base)
            drv.close()
        assert stop is not None
        rows.append((ext, vx, walk_s, stop))
        worst = max(worst, stop["t_s"])
    print(f"\n[estop P-08] crown-up, sim_runner hold: {rows}")
    assert worst <= P08_STOP_S, rows
