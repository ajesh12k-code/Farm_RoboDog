"""WP2 tools: registry, compose module, rack swap state machine and interlocks (ORCHARD-MVP-SPEC 2.4, 2.6;
MVP-SPEC 2.3). ROS-free; MuJoCo headless through ``SimDriver`` (WP2 acceptance):

    swap <= 2 mm / 2 deg, residual < 1 mm, <= 6 s per tool; 4 mm offset refused; no rack weld left active;
    0 tool-finger contacts / 1,000 steps; swap refused while holding, e-stopped, crown up or low margin.
"""
import math

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import farm_scene, sim_state
from atharv_nandi.core.interfaces import GateDecision
from atharv_nandi.core.sim_driver import SimDriver
from atharv_nandi.core.tools import grasp_sense, registry as reg
from atharv_nandi.core.tools.tool_changer import (DOCK_PHASES, MOUNT_PHASES, Interlocks, SwapObs, SwapPlan, ToolChanger,
                                            swap_gate)
from atharv_nandi.sim.world.modules import tools as tools_mod
from atharv_nandi.sim import hooks
import atharv_nandi.sim.hooks.tools as ht


@pytest.fixture
def orchard(monkeypatch):
    def build(world="orchard_lite", tools=None):
        monkeypatch.setenv("NANDI_WORLD", world)
        if tools is not None:
            monkeypatch.setenv("NANDI_TOOLS", tools)
        return farm_scene.load_farm_model()
    return build


def driver_with_tools(m):
    drv = SimDriver(m, load_hooks=False)
    rt = ht.install(drv)
    drv.wait(0.3)
    return drv, rt


def finger_bodies(m):
    return [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in reg.EXCLUDE_ROBOT_BODIES]


# ---------------------------------------------------------------------------------------------- registry + model
def test_registry_names_and_capacity():
    assert reg.TOOL_IDS == ("T2", "T4", "T5")
    assert reg.actuator_names() == ["clipper_roll", "clipper_blade", "twist_roll", "twist_jaw", "pruner_roll",
                                    "pruner_blade"]
    c = reg.get("T2")
    assert c.capacity_n == pytest.approx(80.0)                 # spec: 4 N.m / 0.05 m
    assert reg.get("T5").max_stem_mm == 12.0
    assert reg.f_cut_n("mango") == 40.0 and reg.f_cut_n("x", {"cut_force_n": 7}) == 7.0
    assert reg.prune_force_n(12.0) == pytest.approx(565.5, abs=0.1)
    assert reg.get("T5").capacity_n >= reg.prune_force_n(12.0)
    assert reg.tool_for_task("papaya_twist").tool_id == "T4" and reg.tool_for_task("nope") is None
    assert reg.by_name("pruner").tool_id == "T5"
    with pytest.raises(KeyError):
        reg.get("T9")


def test_orchard_world_has_three_tools_appended_by_name(orchard):
    m = orchard("orchard_lite")
    assert (m.nv, m.nu) == (151, 31)                           # OS 2.6: nv 151, actuators 25 -> 31
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(25, 31)]
    assert names == reg.actuator_names()
    h = tools_mod.handles(m)
    for k, ids in h.tools.items():
        t = ids.spec
        assert m.body_dofnum[ids.body] == 6 and m.body_dofnum[ids.bodies[1]] == 1 and m.body_dofnum[ids.bodies[2]] == 1
        mass = sum(float(m.body_mass[b]) for b in ids.bodies)
        assert mass == pytest.approx(t.mass_kg, abs=1e-6), k
        # the tool file's own joint dynamics survive the attach (not the robot's default damping 1.5)
        assert m.dof_damping[ids.roll_dadr] == pytest.approx(0.05)
        tcp_local = np.array(m.site_pos[ids.tcp]) + np.array(m.body_pos[ids.bodies[1]])
        assert tcp_local == pytest.approx(np.array(t.tcp_local), abs=1e-9)
        assert m.eq_active0[ids.rack_eq] == 1 and m.eq_active0[ids.grip_eq] == 0
        for g in ids.geoms:                                   # parked: collides with nothing
            assert (m.geom_contype[g], m.geom_conaffinity[g]) in ((16, 0), (0, 0))
    # 63 equality rows at rest (OS 2.6): 3 rack welds x 6 + pool 45
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    assert int(np.sum(d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)) == 63


def test_legacy_world_has_no_tools_and_can_force_them(orchard):
    m = orchard("legacy")
    assert m.nu == 25 and tools_mod.handles(m) is None and not tools_mod.debris_ids(m)
    m2 = orchard("legacy", tools="on")
    assert m2.nu == 31 and tools_mod.handles(m2) is not None
    assert not tools_mod.catch_ids(m2)                        # no pool in the legacy world: no catch welds


def test_racked_tools_hold_still_and_touch_nothing(orchard):
    m = orchard("orchard_lite")
    drv = SimDriver(m, load_hooks=False)
    h = tools_mod.handles(m)
    rows = drv.d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    for ids in h.tools.values():
        r = np.where(rows & (drv.d.efc_id == ids.rack_eq))[0]
        assert np.abs(drv.d.efc_pos[r]).max() < 1e-9        # home keyframe at the rack pose: no weld error at t = 0
    geoms = [g for ids in h.tools.values() for g in ids.geoms]
    seen = 0
    for _ in range(1000):
        drv.step()
        seen += grasp_sense.contacts_between(m, drv.d, geoms, range(m.nbody))
    assert seen == 0
    for ids in h.tools.values():
        assert grasp_sense.cradle_occupied(drv.d.site_xpos[ids.handle], drv.d.site_xpos[ids.cradle], 0.5)


def test_rack_pose_matches_the_wp0c_docking_yaw():
    shoulder0 = np.array([0.05, 0.0, 0.195])
    assert tools_mod.rack_yaw(shoulder0, (-0.15, -0.17, 0.26), -0.02) == pytest.approx(-2.3848, abs=1e-3)
    assert tools_mod.rack_yaw(shoulder0, (0.0, -0.17, 0.26), -0.06) == pytest.approx(-1.512, abs=1e-3)
    assert tools_mod.rack_yaw(shoulder0, (0.15, -0.17, 0.26), -0.06) == pytest.approx(-0.8157, abs=1e-3)


# ---------------------------------------------------------------------------------------------- pure state machine
def _plan(tool="T2", direction="mount"):
    z = (0.0,) * 5
    return SwapPlan(tool, direction, (0.0, -1.0, 0.1, -1.8, 1.4), (0.0, -1.0, 0.0, 0.5, 1.0),
                    (0.0, -1.0, -0.1, 0.6, 1.0), z, 0.28)


def _run(ch, obs_fn, t_end=10.0, dt=0.02):
    t, phases, events = 0.0, [], []
    while t < t_end:
        cmd = ch.step(obs_fn(t, ch))
        events += cmd.events
        if not phases or phases[-1] != cmd.phase:
            phases.append(cmd.phase)
        if not ch.busy and cmd.phase in ("idle", "fault"):
            break
        t += dt
    return phases, events, t


def _good(t, ch):
    return SwapObs(t=t, flange_handle_mm=0.3, pose_err_mm=0.4, pose_err_deg=0.3, residual_mm=0.1,
                   other_weld_active=False)


def test_state_machine_mount_then_dock_phase_order_and_time():
    ch = ToolChanger()
    assert ch.request([_plan()], Interlocks(), (0.0,) * 5)
    phases, ev, t = _run(ch, _good)
    assert phases == list(MOUNT_PHASES) + ["idle"]
    assert ch.mounted == "T2" and ev[-1]["event"] == "tool_change_done" and ev[-1]["ok"]
    assert ev[-1]["duration_s"] <= 6.0
    assert ch.request([_plan(direction="dock")], Interlocks(), (0.0,) * 5)
    phases, ev, _ = _run(ch, _good)
    assert phases == list(DOCK_PHASES) + ["idle"] and ch.mounted is None


def test_state_machine_change_queues_dock_then_mount():
    ch = ToolChanger(mounted="T2")
    assert ch.request([_plan("T2", "dock"), _plan("T5", "mount")], Interlocks(), (0.0,) * 5)
    _phases, ev, _ = _run(ch, _good, 20.0)
    done = [e for e in ev if e["event"] == "tool_change_done"]
    assert [(e["tool"], e["direction"]) for e in done] == [("T2", "dock"), ("T5", "mount")]
    assert ch.mounted == "T5"


@pytest.mark.parametrize("bad, reason", [({"pose_err_mm": 4.0}, "pose_check"), ({"pose_err_deg": 2.5}, "pose_check"),
                                         ({"residual_mm": 1.2}, "verify"), ({"other_weld_active": True}, "verify"),
                                         ({"flange_handle_mm": 8.0}, "grip_align")])
def test_state_machine_faults(bad, reason):
    ch = ToolChanger()
    ch.request([_plan()], Interlocks(), (0.0,) * 5)

    def obs(t, c):
        o = _good(t, c)
        for k, v in bad.items():
            setattr(o, k, v)
        return o
    phases, ev, _ = _run(ch, obs)
    assert phases[-2:] == ["safe_park", "fault"]
    assert ev[-1]["event"] == "tool_change_fault" and ev[-1]["reason"] == reason
    assert ch.mounted == (None if reason in ("pose_check", "grip_align") else "T2")
    assert not ch.request([_plan()], Interlocks(), (0.0,) * 5)          # assist first
    ch.clear_fault()
    assert ch.phase == "idle"


def test_state_machine_estop_mid_swap_faults_then_safe_parks():
    ch = ToolChanger()
    ch.request([_plan()], Interlocks(), (0.0,) * 5)

    def obs(t, c):
        o = _good(t, c)
        o.estop = 1.0 <= t < 2.0
        return o
    phases, ev, t = _run(ch, obs)
    assert "safe_park" in phases and phases[-1] == "fault" and ev[-1]["reason"] == "estop"
    assert ch.mounted is None and t > 2.0


def test_swap_gate_interlocks():
    assert swap_gate(Interlocks())
    for il, reason in ((Interlocks(estop=True), "estop"), (Interlocks(holding=True), "holding_payload"),
                       (Interlocks(crown_ext_m=0.3), "crown_up"), (Interlocks(margin_m=0.02), "low_margin"),
                       (Interlocks(person_in_field=True), "person_in_work_field")):
        g = swap_gate(il)
        assert isinstance(g, GateDecision) and not g and g.reason == reason
    assert swap_gate(Interlocks(margin_m=None))                           # unknown margin is not gated
    ch = ToolChanger()
    assert not ch.request([_plan(direction="dock")], Interlocks(), (0.0,) * 5)   # nothing mounted
    assert ch.request([_plan()], Interlocks(), (0.0,) * 5)
    assert ch.request([_plan("T4")], Interlocks(), (0.0,) * 5).reason == "busy"


# ---------------------------------------------------------------------------------------------- MuJoCo swap
@pytest.mark.parametrize("tool", reg.TOOL_IDS)
def test_rack_swap_acceptance(orchard, tool):
    m = orchard("orchard_lite")
    drv, rt = driver_with_tools(m)
    ids = rt.h[tool]
    tool_geoms = [g for i in rt.h.tools.values() for g in i.geoms]
    hand = finger_bodies(m)
    contacts = []
    drv.add_observer(lambda dr: contacts.append(grasp_sense.contacts_between(m, dr.d, tool_geoms, hand)))
    assert rt.request_change(tool)
    assert drv.run_until(lambda dr: not rt.changer.busy, 8.0)
    done = rt.events[-1]
    assert done["event"] == "tool_change_done" and done["ok"], rt.events
    assert done["duration_s"] <= 6.0
    assert done["pose_err_mm"] <= 2.0 and done["pose_err_deg"] <= 2.0
    locked = [e for e in rt.events if e["event"] == "tool_locked"][-1]
    assert locked["residual_mm"] < 1.0
    d = drv.d
    assert d.eq_active[ids.grip_eq] == 1 and d.eq_active[ids.rack_eq] == 0          # no rack weld left on it
    assert rt.plant.weld_counts() == {"rack_active": 2, "grip_active": 1}
    assert grasp_sense.tool_present(d.site_xpos[rt.h.flange], d.site_xpos[ids.handle])
    assert not grasp_sense.cradle_occupied(d.site_xpos[ids.handle], d.site_xpos[ids.cradle])
    assert tools_mod.is_active(m, tool)
    drv.run_steps(1000)                                                              # mounted, arm parked
    assert grasp_sense.tool_present(d.site_xpos[rt.h.flange], d.site_xpos[ids.handle])
    assert sum(contacts) == 0                                                        # 0 tool-finger contacts
    # and back into the rack
    assert rt.request_change(None)
    assert drv.run_until(lambda dr: not rt.changer.busy, 8.0)
    back = rt.events[-1]
    assert back["event"] == "tool_change_done" and back["direction"] == "dock" and back["duration_s"] <= 6.0
    assert rt.plant.weld_counts() == {"rack_active": 3, "grip_active": 0} and not tools_mod.is_active(m, tool)
    drv.run_steps(200)
    assert grasp_sense.cradle_occupied(d.site_xpos[ids.handle], d.site_xpos[ids.cradle], 1.0)
    assert sum(contacts) == 0


@pytest.mark.parametrize("offset_mm, ok", [(4.0, False), (1.5, True)])
def test_offset_tool_is_refused_at_check_pose(orchard, offset_mm, ok):
    m = orchard("orchard_lite")
    drv, rt = driver_with_tools(m)
    ids = rt.h["T2"]
    # a tool seated 4 mm off its design pose in the cradle (bumped rack): shift the rack relpose sideways
    m.eq_data[ids.rack_eq][4] += offset_mm / 1000.0
    drv.wait(0.5)
    rt.request_change("T2")
    drv.run_until(lambda dr: not rt.changer.busy, 10.0)
    ev = rt.events[-1]
    if ok:
        assert ev["event"] == "tool_change_done" and ev["pose_err_mm"] == pytest.approx(offset_mm, abs=0.4)
    else:
        assert ev["event"] == "tool_change_fault" and ev["reason"] == "pose_check"
        assert ev["pose_err_mm"] == pytest.approx(offset_mm, abs=0.4)
        assert rt.changer.phase == "fault" and rt.changer.mounted is None
        assert drv.d.eq_active[ids.rack_eq] == 1 and drv.d.eq_active[ids.grip_eq] == 0
        assert np.allclose(rt.changer.q_cmd, rt.plant.q_home)                       # safe-parked


def test_swap_refused_while_holding_estopped_crown_up_or_low_margin(orchard):
    m = orchard("orchard_lite")
    drv, rt = driver_with_tools(m)
    cases = [({"hold": True}, "estop"), ({"values": {"mast": {"ext_m": 0.4}}}, "crown_up"),
             ({"values": {"stability": {"margin_m": 0.02}}}, "low_margin"),
             ({"values": {"work_field": {"person": True}}}, "person_in_work_field")]
    for kw, reason in cases:
        g = rt.request_change("T4", hold=kw.get("hold", False), values=kw.get("values"))
        assert not g and g.reason == reason and rt.events[-1]["event"] == "tool_change_refused"
    rt.caught = {"slot": 0, "eq": -1, "mass": 0.2}                                   # a fruit in the catcher
    g = rt.request_change("T4")
    assert not g and g.reason == "holding_payload"
    rt.caught = {}
    assert rt.plant.margin_m() > 0.03 and rt.plant.crown_ext_m() == 0.0              # the model's own readings
    assert rt.request_change("T4")
    assert not rt.changer.request([], Interlocks(), (0.0,) * 5)                      # busy now


def test_estop_during_swap_never_leaves_a_half_lock(orchard):
    m = orchard("orchard_lite")
    drv, rt = driver_with_tools(m)
    rt.request_change("T5")
    drv.wait(2.0)                                                                    # mid rack_pre
    drv.enter_hold("estop")
    drv.wait(0.5)
    assert rt.changer.phase == "safe_park" and rt.events[-1]["reason"] == "estop"
    assert not rt.actuate({"value": 1.0}, hold=True)
    drv.leave_hold()
    drv.wait(2.5)
    assert rt.changer.phase == "fault"
    for ids in rt.h.tools.values():
        assert drv.d.eq_active[ids.rack_eq] + drv.d.eq_active[ids.grip_eq] == 1      # exactly one weld per tool


# ---------------------------------------------------------------------------------------------- state + reset
def test_tools_and_debris_providers_round_trip(orchard):
    m = orchard("orchard_lite")
    drv, rt = driver_with_tools(m)
    rt.request_change("T2")
    drv.run_until(lambda dr: not rt.changer.busy, 8.0)
    wid = farm_scene.world_identity()[1]
    st = sim_state.pack(m, drv.d, wid)
    assert {"tools", "debris"} <= set(st["mutations"]["providers"])
    ids = rt.h["T2"]
    assert st["mutations"]["providers"]["tools"]["data"]["tools"]["T2"]["geoms"][0] == [8, 1]
    m2 = orchard("orchard_lite")
    d2 = mujoco.MjData(m2)
    ok, why = sim_state.apply(m2, d2, sim_state.decode(sim_state.encode(st)), wid)
    assert ok, why
    mujoco.mj_kinematics(m2, d2)
    mujoco.mj_kinematics(m, drv.d)                 # the stepped data's site_xpos lag its qpos by one step
    assert tools_mod.is_active(m2, "T2") and d2.eq_active[ids.grip_eq] == 1
    assert np.allclose(d2.site_xpos[ids.handle], drv.d.site_xpos[ids.handle])
    bad = sim_state.decode(sim_state.encode(st))
    bad["mutations"]["providers"]["tools"]["data"]["tools"]["T2"]["geoms"] = [[8, 1]]
    ok, why = sim_state.apply(m2, d2, bad, wid)
    assert not ok and "tools" in why
    # a legacy world packs no WP2 block
    ml = orchard("legacy")
    assert not {"tools", "debris"} & set(sim_state.pack(ml, mujoco.MjData(ml), "x")["mutations"]["providers"])


def test_hook_registers_and_reset_restores_the_rack(orchard):
    m = orchard("orchard_lite")
    reg_ = hooks.HookRegistry()
    ht.register(reg_)
    tab = {h["name"]: h for h in reg_.table()}
    assert tab["tools"]["mutates_model"] and tab["tools"]["rate_hz"] == 50.0
    assert set(tab["tools"]["topics_in"]) == {ht.CHANGE_TOPIC, ht.ACTUATE_TOPIC}
    drv = SimDriver(m, hooks=reg_)
    drv.wait(0.1)
    rt = ht._RUNTIMES[m]
    drv.deliver(ht.CHANGE_TOPIC, {"tool": "T4"})
    drv.wait(1.0)
    assert rt.changer.busy
    arm = [p for _t, topic, p in drv.published if topic == ht.ARM_TOPIC]
    assert arm and "q5" in arm[-1]                    # the hook publishes arm targets for the arm owner
    states = [p for _t, topic, p in drv.published if topic == ht.STATE_TOPIC]
    assert states and '"phase"' in states[-1]
    tools_mod.set_contact(m, "T4", True)
    drv.d.eq_active[rt.h["T4"].grip_eq] = 1
    drv.reset()
    assert rt.plant.weld_counts() == {"rack_active": 3, "grip_active": 0}
    assert not any(tools_mod.is_active(m, k) for k in reg.TOOL_IDS)
    assert not rt.changer.busy and rt.changer.mounted is None


# ---------------------------------------------------------------------------------------------- OS 2.4 tool sensors
def test_tool_sensors_blade_roll_jointpos_and_tcp_range(orchard):
    """OS 2.4 sensors: blade jointpos (T2, T5), roll jointpos (T4), T2 TCP rangefinder, T4 pad touches."""
    m = orchard("orchard_lite")
    h = tools_mod.handles(m)
    assert [h[k].spec.pos_sensor for k in reg.TOOL_IDS] == ["clipper_blade_pos", "twist_roll_pos", "pruner_blade_pos"]
    assert h["T2"].range_site >= 0 and h["T4"].range_site < 0 and h["T5"].range_site < 0
    drv, rt = driver_with_tools(m)
    assert rt.state()["sensors"] == {}
    rt.request_change("T2")
    assert drv.run_until(lambda dr: not rt.changer.busy, 8.0) and rt.changer.mounted == "T2"
    ids = rt.h["T2"]
    drv.d.ctrl[ids.act_act] = 1.0                    # direct ctrl (no target in reach: actuate would refuse)
    drv.wait(0.5)
    s = rt.state()["sensors"]
    assert s["clipper_blade_pos"] == pytest.approx(float(drv.d.qpos[ids.act_qadr]), abs=1e-3)
    assert s["clipper_blade_pos"] > 0.9 * reg.get("T2").close_rad
    # the rangefinder looks along the tool axis from the TCP: put a target 0.10 m ahead, read 0.10 m minus radius
    mujoco.mj_kinematics(m, drv.d)
    R = drv.d.site_xmat[ids.tcp].reshape(3, 3)
    p = drv.d.site_xpos[ids.tcp] + 0.10 * R[:, 0]
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, tools_mod.debris_name(0) + "_g")
    mid = tools_mod.debris_ids(m)[0]["mocap"]
    m.geom_rgba[gid][3] = 1.0                        # alpha 0 geoms are invisible to rays
    drv.d.mocap_pos[mid] = p
    q = np.zeros(4)
    mujoco.mju_quatZ2Vec(q, R[:, 1].copy())          # capsule axis across the ray: the side is hit
    drv.d.mocap_quat[mid] = q
    drv.wait(0.02)
    r = rt.state()["sensors"]["clipper_tcp_range"]
    assert r == pytest.approx(0.10 - tools_mod.DEBRIS_RADIUS, abs=0.004)
    rt.request_change("T4")
    assert drv.run_until(lambda dr: not rt.changer.busy, 12.0) and rt.changer.mounted == "T4"
    s = rt.state()["sensors"]
    assert set(s) == {"twist_roll_pos", "twist_jaw_a_touch", "twist_jaw_b_touch"}
    assert s["twist_roll_pos"] == pytest.approx(float(drv.d.qpos[rt.h["T4"].roll_qadr]), abs=1e-3)


# ---------------------------------------------------------------------------------------------- rigid builds
@pytest.fixture
def rigid(monkeypatch, orchard):
    def build(tool, world="orchard_lite"):
        monkeypatch.setenv("NANDI_TOOL_MOUNT", f"rigid:{tool}")
        return orchard(world)
    return build


@pytest.mark.parametrize("tool", reg.TOOL_IDS)
def test_rigid_build_mounts_one_tool_on_the_flange(rigid, tool):
    """MVP-SPEC 2.3 rigid builds: NANDI_TOOL_MOUNT=rigid:<tool> = only that tool, on the flange at X_nom from
    compile (grip weld active, no rack weld: design rule 1 keeps the tool a free body after the pool), the other
    tools not built; mounted for the whole run, a change is refused ``rigid_mount``."""
    m = rigid(tool)
    t = reg.get(tool)
    h = tools_mod.handles(m)
    assert h.rigid == tool and set(h.tools) == {tool}
    ids = h[tool]
    assert ids.rigid and ids.rack_eq < 0 and ids.grip_eq >= 0 and m.eq_active0[ids.grip_eq] == 1
    assert (m.nv, m.nu) == (151 - 16, 25 + 2)                  # one tool: 8 dof, 2 actuators
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(25, 27)]
    assert names == [t.roll_act, t.act_act] and h.act_ids() == [25, 26]
    assert sum(float(m.body_mass[b]) for b in ids.bodies) == pytest.approx(t.mass_kg, abs=1e-6)
    assert all(int(m.geom_contype[g]) == reg.ACTIVE_CONTYPE for g in ids.collide)
    assert not any(n.startswith("rack_") for n in
                   (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, e) for e in range(m.neq)))
    assert {n for n, _k in tools_mod.catch_ids(m)} == ({t.name} if t.name in tools_mod.CATCH_SLOTS else set())
    drv, rt = driver_with_tools(m)
    d = drv.d
    # the home keyframe puts the tool on the flange: zero weld error at reset
    d0 = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d0, 0)
    mujoco.mj_forward(m, d0)
    assert np.linalg.norm(d0.site_xpos[ids.handle] - d0.site_xpos[h.flange]) < 1e-9
    wrist = h.wrist
    assert np.linalg.norm(d0.site_xpos[ids.tcp] - d0.xpos[wrist]) == pytest.approx(t.tcp_from_wrist_m, abs=1e-9)
    assert rt.changer.mounted == tool and rt.payload_kg() == pytest.approx(t.mass_kg)
    st = rt.state()
    assert st["mount"] == "rigid" and st["parked"] == [] and st["tcp_frame"] == t.tcp_site
    other = next(k for k in reg.TOOL_IDS if k != tool)
    for target in (other, None):
        assert not rt.request_change(target)
        assert rt.events[-1]["event"] == "tool_change_refused" and rt.events[-1]["reason"] == "rigid_mount"
    rt.request_change(tool)
    assert rt.events[-1]["reason"] == "already_mounted"
    assert rt.plant.weld_counts() == {"rack_active": 0, "grip_active": 1}
    # 1,000 steps: the tool stays on the flange (< 1 mm) and never touches the fingers / wrist
    fingers = finger_bodies(m)
    seen, worst = 0, 0.0
    for _ in range(1000):
        drv.step()
        seen += grasp_sense.contacts_between(m, d, ids.geoms, fingers)
        worst = max(worst, float(np.linalg.norm(d.site_xpos[ids.handle] - d.site_xpos[h.flange])))
    assert seen == 0 and worst < 0.001 and not rt.changer.busy
    print(f"rigid {tool}: flange-handle worst {1000 * worst:.3f} mm over 1,000 steps")
    tools_mod.set_contact(m, tool, False)
    d.eq_active[ids.grip_eq] = 0
    rt.reset_model()
    assert all(int(m.geom_contype[g]) == reg.ACTIVE_CONTYPE for g in ids.collide) and rt.changer.mounted == tool
    assert d.eq_active[ids.grip_eq] == 1


def test_rigid_build_state_round_trip_and_bad_values(rigid, orchard, monkeypatch):
    m = rigid("T5")
    drv, rt = driver_with_tools(m)
    wid = farm_scene.world_identity()[1]
    st = sim_state.pack(m, drv.d, wid)
    blk = st["mutations"]["providers"]["tools"]["data"]["tools"]
    assert set(blk) == {"T5"} and set(blk["T5"]) == {"grip", "geoms", "bodies"}
    monkeypatch.setenv("NANDI_TOOL_MOUNT", "rack")
    m2 = orchard("orchard_lite")
    wid2 = farm_scene.world_identity()[1]
    assert wid2 != wid                                 # the mount is part of the world id (module fingerprint)
    ok, why = sim_state.apply(m2, mujoco.MjData(m2), sim_state.decode(sim_state.encode(st)), wid2)
    assert not ok                                      # a rigid state never applies to a rack build
    m3 = rigid("T5")
    ok, why = sim_state.apply(m3, mujoco.MjData(m3), sim_state.decode(sim_state.encode(st)), wid)
    assert ok, why
    for bad in ("rigid:T9", "glued"):
        monkeypatch.setenv("NANDI_TOOL_MOUNT", bad)
        with pytest.raises(ValueError):
            farm_scene.load_farm_model()
    monkeypatch.setenv("NANDI_TOOL_MOUNT", "rigid:T2")
    monkeypatch.setenv("NANDI_TOOLS", "off")
    with pytest.raises(ValueError):
        farm_scene.load_farm_model()
    monkeypatch.setenv("NANDI_TOOLS", "auto")
    monkeypatch.setenv("NANDI_WORLD", "legacy")
    ml = farm_scene.load_farm_model()                  # rigid implies tools in any world
    assert tools_mod.handles(ml).rigid == "T2" and ml.nu == 27 and not tools_mod.catch_ids(ml)
    monkeypatch.setenv("NANDI_TOOL_MOUNT", "rack")
    monkeypatch.setenv("NANDI_WORLD", "orchard_lite")
    assert tools_mod.handles(farm_scene.load_farm_model()).rigid is None
