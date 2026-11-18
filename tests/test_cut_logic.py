"""WP2 cut rule (CLIPPER pedicel clip, PRUNER prune) and debris (MVP-SPEC 3.4, ORCHARD-MVP-SPEC 3.4).

Pure-logic tests for every condition of the rule, then headless MuJoCo with the real tools, the WP23 fruit pool
and ``SimDriver`` (WP2 acceptance):

    cut fires at nominal closure in the trap window, not on near-misses; pedicel cut point +-2 mm at 1, 2, 5 cm;
    PRUNER refuses > 12 mm.

A pool fruit is assigned, then moved (anchor and body together, so the equality still holds it at rest) in front
of the standing robot: the test is about the tool and the rule, not about reaching a tree (WP8).
"""
import math

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import farm_scene
from atharv_nandi.core.sim_driver import SimDriver
from atharv_nandi.core.tools import cut_logic, registry as reg
from atharv_nandi.core.tools.cut_logic import CutInputs, CutRule
from atharv_nandi.core.tools.tool_changer import tcp_closed_form
from atharv_nandi.sim.world.modules import tools as tools_mod
from atharv_nandi.sim.world.modules import trees as trees_mod
import atharv_nandi.sim.hooks.tools as ht


# ---------------------------------------------------------------------------------------------- geometry
def test_geometry_helpers():
    d, c, s = cut_logic.point_segment([0, 1, 0], [-1, 0, 0], [1, 0, 0])
    assert d == pytest.approx(1.0) and s == pytest.approx(0.5) and c == pytest.approx([0, 0, 0])
    assert cut_logic.point_segment([3, 0, 0], [0, 0, 0], [1, 0, 0])[0] == pytest.approx(2.0)
    assert cut_logic.point_segment([0, 0, 1], [0, 0, 0], [0, 0, 0])[0] == pytest.approx(1.0)   # degenerate
    c1, c2, dd = cut_logic.segment_segment([0, 0, -1], [0, 0, 1], [-1, 0.5, 0], [1, 0.5, 0])
    assert dd == pytest.approx(0.5) and c1 == pytest.approx([0, 0, 0]) and c2 == pytest.approx([0, 0.5, 0])
    assert cut_logic.segment_segment([0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0])[2] == pytest.approx(1.0)
    assert cut_logic.segment_segment([0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0])[2] == pytest.approx(1.0)
    assert cut_logic.line_angle_deg([0, 0, 1], [0, 0, -1]) == pytest.approx(0.0)
    assert cut_logic.line_angle_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert cut_logic.line_angle_deg([0, 0, 0], [0, 1, 0]) == 90.0
    p = cut_logic.pedicel_cut_point([0, 0, 0], [0, 0, 0.06], [-0.01, 0, 0.02], [0.01, 0, 0.02])
    assert p == pytest.approx([0, 0, 0.02])
    assert cut_logic.pedicel_error_mm(p, [0, 0, 0], 0.02) == pytest.approx(0.0, abs=1e-9)
    assert cut_logic.capacity_n(4.0, 0.05) == pytest.approx(80.0) and cut_logic.capacity_n(1, 0) == 0.0


# ---------------------------------------------------------------------------------------------- rule
def _feed(rule, t_end, **kw):
    base = dict(blade_ctrl=1.0, closed_frac=1.0, capacity_n=80.0, f_cut_n=40.0, dist_mm=1.0, angle_deg=5.0)
    base.update(kw)
    t, fired = 0.0, None
    while t <= t_end + 1e-9:
        dec = rule.update(CutInputs(t, **base))
        if dec.fire and fired is None:
            fired = t
        t += 0.02
    return fired, dec


def test_cut_fires_at_nominal_closure_after_30ms():
    fired, dec = _feed(CutRule(), 0.2)
    assert fired == pytest.approx(0.04)          # 50 Hz: ctrl >= 0.95 from t 0, 30 ms sustained at the 3rd tick
    assert all(dec.conditions.values())
    r = CutRule(sustain_s=0.03)
    assert not r.update(CutInputs(0.0, 1.0, 1.0, 80, 40, 1, 5)).fire
    assert not r.update(CutInputs(0.02, 1.0, 1.0, 80, 40, 1, 5)).fire
    assert r.update(CutInputs(0.03, 1.0, 1.0, 80, 40, 1, 5)).fire
    assert not r.update(CutInputs(0.05, 1.0, 1.0, 80, 40, 1, 5)).fire       # once per closing
    r.reset()
    assert not r.fired


@pytest.mark.parametrize("kw, failing", [
    ({"blade_ctrl": 0.94}, "ctrl_sustained"), ({"closed_frac": 0.89}, "closed"), ({"f_cut_n": 81.0}, "capacity"),
    ({"dist_mm": 6.1}, "trap"), ({"angle_deg": 30.5}, "angle")])
def test_cut_needs_every_condition(kw, failing):
    fired, dec = _feed(CutRule(), 0.3, **kw)
    assert fired is None and not dec.conditions[failing]
    assert all(v for k, v in dec.conditions.items() if k != failing)


def test_trap_window_demo_vs_physics():
    assert _feed(CutRule(mode="demo"), 0.1, dist_mm=5.9)[0] is not None
    assert _feed(CutRule(mode="physics"), 0.1, dist_mm=5.9)[0] is None
    assert _feed(CutRule(mode="physics"), 0.1, dist_mm=3.9)[0] is not None
    with pytest.raises(ValueError):
        CutRule(mode="nope")


def test_ctrl_drop_restarts_the_sustain_timer():
    r = CutRule()
    assert not r.update(CutInputs(0.00, 1.0, 1.0, 80, 40, 1, 5)).fire
    assert not r.update(CutInputs(0.02, 0.5, 1.0, 80, 40, 1, 5)).fire
    assert not r.update(CutInputs(0.04, 1.0, 1.0, 80, 40, 1, 5)).fire
    assert r.update(CutInputs(0.08, 1.0, 1.0, 80, 40, 1, 5)).fire


def test_prune_gate_capacity_keep_and_band():
    assert cut_logic.prune_gate(12.0)
    g = cut_logic.prune_gate(12.5)
    assert not g and g.reason == "over_capacity" and g.values["paste_required"] == 1.0
    assert cut_logic.prune_gate(20.0).reason == "over_capacity"
    assert cut_logic.prune_gate(8.0, keep=True).reason == "keep_wood"
    assert cut_logic.prune_gate(8.0, in_band=False).reason == "out_of_reach_marked"
    assert cut_logic.prune_gate(float("nan")).reason == "bad_diameter"


def test_debris_free_fall_lands_and_rests():
    deb = cut_logic.Debris((1.0, 2.0, 0.5), (1, 0, 0, 0), 10.0, 0.004)
    assert deb.pos(10.0) == pytest.approx((1.0, 2.0, 0.5))
    assert deb.pos(10.1)[2] == pytest.approx(0.5 - 0.5 * 9.81 * 0.01)
    assert not deb.landed
    assert deb.pos(11.0)[2] == pytest.approx(0.004) and deb.landed
    assert deb.impact_speed() == pytest.approx(math.sqrt(2 * 9.81 * 0.496))


# ---------------------------------------------------------------------------------------------- MuJoCo helpers
@pytest.fixture
def world(monkeypatch):
    def build(world="orchard_lite", cut=None):
        monkeypatch.setenv("NANDI_WORLD", world)
        if cut:
            monkeypatch.setenv("NANDI_CUT", cut)
        m = farm_scene.load_farm_model()
        drv = SimDriver(m, load_hooks=False)
        rt = ht.install(drv)
        drv.wait(0.3)
        return m, drv, rt
    return build


def move_slot(m, d, pool, k, point, site="cut_k"):
    """Move an assigned pool fruit so that ``site`` (or the body) is at ``point``; the equality anchor moves along."""
    s = pool.slots[k]
    mujoco.mj_forward(m, d)
    cur = d.site_xpos[getattr(s, site)] if site else d.xpos[s.body]
    delta = np.asarray(point, float) - cur
    if s.kind == "connect":
        m.eq_data[s.eq][3:6] += delta
    else:
        m.eq_data[s.eq][0:3] += delta
    d.qpos[s.qadr:s.qadr + 3] += delta
    d.qvel[s.dadr:s.dadr + 6] = 0.0
    mujoco.mj_forward(m, d)


def mount(drv, rt, tool):
    rt.request_change(tool)
    assert drv.run_until(lambda dr: not rt.changer.busy, 8.0)
    assert rt.changer.mounted == tool, rt.events[-1]


def stage_mango(m, drv, rt, leave_m, fid="L2-1:f07"):
    pool, d = rt.pool, drv.d
    k = pool.assign(d, fid)
    s = pool.slots[k]
    stem, anc = np.array(m.site_pos[s.stem_end]), np.array(m.site_pos[s.anchor])
    m.site_pos[s.cut_k] = stem + leave_m * (anc - stem) / np.linalg.norm(anc - stem)   # pedicel_leave setpoint
    pool.fruit[fid] = dict(pool.fruit[fid], pedicel_leave_m=leave_m)
    tp, tR = drv.ik.trunk_pose(d.qpos)
    move_slot(m, d, pool, k, tp + tR @ np.array([0.62, -0.02, 0.30]))
    drv.wait(0.5)
    return k, s


def clip(m, drv, rt, k, s, side_mm=0.0, value=1.0):
    d = drv.d
    arm = ht.DriverArm(drv, rt)
    cut = d.site_xpos[s.cut_k].copy()
    _q, ax = tcp_closed_form(drv.ik, d.qpos, cut, reg.get("T2").tcp_from_wrist_m)
    arm.tcp_to(cut - 0.07 * ax, 1.5)                                  # behind the stem, blades open
    side = np.cross(ax, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)
    arm.tcp_to(cut, 1.0)                                              # the pedicel enters the V-trap
    err = arm.align(lambda: d.site_xpos[s.cut_k] + side_mm / 1000.0 * side)
    n0 = len(rt.events)
    rt.actuate({"value": value})
    drv.wait(0.5)
    return err, rt.events[n0:], arm, ax


@pytest.mark.parametrize("leave_cm", [1, 2, 5])
def test_mango_clip_cuts_at_the_pedicel_setpoint(world, leave_cm):
    m, drv, rt = world()
    mount(drv, rt, "T2")
    k, s = stage_mango(m, drv, rt, leave_cm / 100.0)
    err, ev, arm, ax = clip(m, drv, rt, k, s)
    assert err < 0.001
    cuts = [e for e in ev if e["event"] == "cut"]
    assert len(cuts) == 1, ev
    c = cuts[0]
    assert c["dist_mm"] < cut_logic.WINDOW_MM["demo"] and c["angle_deg"] < 30.0
    assert abs(c["pedicel_error_mm"]) <= 2.0 and c["pedicel_mm"] == pytest.approx(10.0 * leave_cm, abs=2.0)
    assert not rt.pool.is_attached(drv.d, k) and c["caught"] and rt.caught["slot"] == k
    # the caught fruit leaves with the clipper (assisted catch) and swap is now interlocked (holding)
    tcp = rt.h["T2"].tcp
    gap0 = np.linalg.norm(drv.d.xpos[s.body] - drv.d.site_xpos[tcp])
    arm.tcp_to(drv.d.site_xpos[tcp] - 0.08 * ax, 1.0)
    assert np.linalg.norm(drv.d.xpos[s.body] - drv.d.site_xpos[tcp]) == pytest.approx(gap0, abs=0.003)
    assert not rt.request_change(None) and rt.events[-1]["reason"] == "holding_payload"
    assert rt.release_catch() == k and not rt.caught


@pytest.mark.parametrize("side_mm", [9.0, 15.0])
def test_near_miss_never_cuts(world, side_mm):
    m, drv, rt = world()
    mount(drv, rt, "T2")
    k, s = stage_mango(m, drv, rt, 0.02)
    _err, ev, _arm, _ax = clip(m, drv, rt, k, s, side_mm=side_mm)
    assert not [e for e in ev if e["event"] == "cut"]
    miss = [e for e in ev if e["event"] == "cut_miss"]
    assert miss and miss[0]["conditions"]["trap"] is False and miss[0]["conditions"]["closed"] is True
    assert miss[0]["dist_mm"] == pytest.approx(side_mm, abs=1.0)
    assert rt.pool.is_attached(drv.d, k)


def test_partial_closure_never_cuts(world):
    m, drv, rt = world()
    mount(drv, rt, "T2")
    k, s = stage_mango(m, drv, rt, 0.02)
    _err, ev, _arm, _ax = clip(m, drv, rt, k, s, value=0.85)          # ctrl below 0.95: no nominal closure
    assert not [e for e in ev if e["event"] in ("cut", "cut_miss")]
    assert rt.pool.is_attached(drv.d, k)
    n0 = len(rt.events)
    rt.actuate({"value": 1.0})
    drv.wait(0.3)
    assert [e["event"] for e in rt.events[n0:]] == ["cut"]


def test_resist_mode_tests_the_blade_margin(world):
    m, drv, rt = world(cut="resist")
    assert rt.cut_rule.window_mm == cut_logic.WINDOW_MM["physics"]
    mount(drv, rt, "T2")
    k, s = stage_mango(m, drv, rt, 0.02)
    _err, ev, _arm, _ax = clip(m, drv, rt, k, s)
    assert [e["event"] for e in ev] == ["cut"]                         # 2 N.m against 4 N.m: still closes > 90 %
    ids = rt.h["T2"]
    assert drv.d.qfrc_applied[ids.act_dadr] == 0.0                    # opposing torque removed after the cut


# ---------------------------------------------------------------------------------------------- pruner
def stage_prune(m, drv, rt, diameter_mm, offset=(0.60, 0.0, 0.28), keep=False):
    """A shoot of the given diameter in front of the robot: a real hideable prune geom of the orchard_lite trees,
    moved there (geom pos) and registered as the runtime's only prune target."""
    d = drv.d
    tp, tR = drv.ik.trunk_pose(d.qpos)
    cut_w = tp + tR @ np.array(offset)
    gid = next(g for g in trees_mod.hideable_geoms(m)
               if "_prune_" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""))
    body = int(m.geom_bodyid[gid])
    a_w, b_w = cut_w - np.array([0, 0, 0.06]), cut_w + np.array([0, 0, 0.10])
    R = np.array(d.xmat[body]).reshape(3, 3)
    m.geom_pos[gid] = R.T @ ((a_w + b_w) / 2.0 - d.xpos[body])
    rt.prune_targets = [{"id": "test:prune:1", "geom": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid), "gid": gid,
                         "job": "rootstock_sprout", "keep": keep, "in_band": True, "diameter_mm": diameter_mm,
                         "a_w": a_w, "b_w": b_w, "cut_w": cut_w}]
    return gid, cut_w


@pytest.mark.parametrize("diameter_mm, cuts", [(10.0, True), (20.0, False)])
def test_pruner_cuts_within_capacity_and_refuses_over_12mm(world, diameter_mm, cuts):
    m, drv, rt = world()
    mount(drv, rt, "T5")
    gid, cut_w = stage_prune(m, drv, rt, diameter_mm)
    arm = ht.DriverArm(drv, rt)
    arm.q = np.array(rt.plant.q_home)
    ax = arm.tcp_to(cut_w - np.array([0.07, 0.0, 0.0]), 1.5)
    arm.tcp_to(cut_w, 1.0)
    assert arm.align(lambda: cut_w) < 0.001
    n0 = len(rt.events)
    rt.actuate({"value": 1.0})
    drv.wait(0.4)
    ev = rt.events[n0:]
    hidden = gid in trees_mod.hidden(m)
    if cuts:
        cut = [e for e in ev if e["event"] == "cut"]
        assert len(cut) == 1 and cut[0]["job"] == "rootstock_sprout" and hidden
        k = cut[0]["debris"]
        deb = tools_mod.debris_ids(m)[k]
        assert m.geom_rgba[deb["geom"]][3] == 1.0
        assert m.geom_size[deb["geom"]][0] == pytest.approx(diameter_mm / 2000.0)
        arm.tcp_to(cut_w - 0.1 * ax, 0.5)
        drv.wait(1.0)
        z = drv.d.mocap_pos[deb["mocap"]][2]
        assert z == pytest.approx(rt._ground_z(cut_w) + diameter_mm / 2000.0, abs=1e-3)
        assert tools_mod.debris_sources(m)[k] == rt.prune_targets[0]["geom"]
        # the cut is in the sim state (hidden geom + debris shape); reset parks the debris again
        rt.reset_model()
        assert m.geom_rgba[deb["geom"]][3] == 0.0 and tools_mod.debris_sources(m)[k] is None
    else:
        ref = [e for e in ev if e["event"] == "prune_refused"]
        assert ref and ref[0]["reason"] == "over_capacity" and ref[0]["paste_required"] is True
        assert not hidden and not [e for e in ev if e["event"] == "cut"]


def test_actuation_is_refused_without_a_tool_or_while_held(world):
    m, drv, rt = world()
    assert not rt.actuate({"value": 1.0}) and rt.events[-1]["reason"] == "no_tool_mounted"
    mount(drv, rt, "T2")
    assert not rt.actuate({"value": 1.0}, hold=True) and rt.events[-1]["reason"] == "estop"
    assert not rt.actuate({"value": "x"}) and rt.events[-1]["reason"] == "bad_value"
    assert not rt.actuate({"tool": "T4", "value": 1.0}) and rt.events[-1]["reason"] == "tool_not_mounted"
    assert not rt.actuate("close") and rt.events[-1]["reason"] == "bad_request"
    assert rt.actuate({"roll": 0.5, "value": 0.2})
    ids = rt.h["T2"]
    assert drv.d.ctrl[ids.roll_act] == pytest.approx(0.5) and drv.d.ctrl[ids.act_act] == pytest.approx(0.2)
    rt.act.open_blade()
    assert drv.d.ctrl[ids.act_act] == 0.0
    drv.enter_hold("estop")
    drv.wait(0.1)
    assert rt.act.stopped and not rt.act.set(1.0)                    # ToolActuation.stop holds, refuses
    drv.leave_hold()
    drv.wait(0.1)
    assert not rt.act.stopped and rt.act.state()["mounted"] == "T2"


# ---------------------------------------------------------------------------------------------- failure mode, rigid
def test_mango_12n_pull_never_releases_physics_mode(world):
    """OS 3.4: no pull release for mango. A 12 N pull in physics mode is a failure-mode test only (snap-off is a
    defect): the connect holds, the anchor stays put and the tool runtime emits no cut."""
    m, drv, rt = world(cut="physics")
    k, s = stage_mango(m, drv, rt, 0.02)
    d = drv.d
    mujoco.mj_forward(m, d)
    anchor0 = np.array(d.site_xpos[s.anchor])
    pull = np.array(d.xpos[s.body]) - anchor0                       # away from the anchor (along the pedicel)
    pull = 12.0 * pull / np.linalg.norm(pull)
    n0 = len(rt.events)
    d.xfrc_applied[s.body, :3] = pull
    worst = 0.0
    for _ in range(int(2.0 / m.opt.timestep)):
        drv.step()
        worst = max(worst, float(np.linalg.norm(d.site_xpos[s.anchor] - anchor0)))
    d.xfrc_applied[s.body, :] = 0.0
    assert rt.pool.is_attached(d, k)
    assert worst < 0.002                                            # the connect holds the anchor (< 2 mm)
    assert not [e for e in rt.events[n0:] if e["event"] in ("cut", "twist_release")]
    print(f"mango 12 N pull, 2 s: worst anchor drift {1000 * worst:.3f} mm, still attached")


def test_rigid_t2_clips_a_mango(world, monkeypatch):
    """MVP-SPEC 2.3 rigid build (NANDI_TOOL_MOUNT=rigid:T2): the same cut rule and pedicel accuracy, no swap."""
    monkeypatch.setenv("NANDI_TOOL_MOUNT", "rigid:T2")
    m, drv, rt = world()
    assert rt.h.rigid == "T2" and rt.changer.mounted == "T2"
    k, s = stage_mango(m, drv, rt, 0.02)
    err, ev, _arm, _ax = clip(m, drv, rt, k, s)
    assert err < 0.001
    cut = [e for e in ev if e["event"] == "cut"]
    assert len(cut) == 1 and abs(cut[0]["pedicel_error_mm"]) <= 2.0 and cut[0]["caught"], ev
    assert not rt.pool.is_attached(drv.d, k) and rt.release_catch() == k
