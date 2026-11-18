"""WP24 crown mast runtime: arbiter, per-step interpolation, swept clearance, geometric strike, gimbal, e-stop brake.

Acceptance (WP24; ORCHARD-MVP-SPEC 2.3, 3.5): layout parity (names, qpos / dof addresses, actuator
ids) with the mast in / out; dynamic parity (ext held 0, hook on / off -> byte-identical qpos over 2,000 steps);
lens z +-2 cm; stroke <= 8.5 s; per-step setpoint delta <= 0.4 mm; head driven into a wood capsule and into a
foliage envelope: strike event, retract within 1 tick, robot tilt change <= 3 deg; raise refused when the swept
cylinder hits a rope line; arbiter priority table; e-stop holds; each test <= 10 s. Headless MuJoCo, ROS-free.
"""
import contextlib
import json
import math
import os

import mujoco
import numpy as np
import pytest

import atharv_nandi.sim.hooks as hook_registry
from atharv_nandi.core import camera_rig, farm_scene, interfaces
from atharv_nandi.core import mast as mc
from atharv_nandi.core import model_layout as mlay
from atharv_nandi.core import sim_driver as sd
from atharv_nandi.core import sim_state
from atharv_nandi.core.interfaces import GateDecision, MastRequest
from atharv_nandi.core.locomotion_logic import quat_to_roll_pitch
from atharv_nandi.sim.hooks import mast as hm

O = mujoco.mjtObj
WORLD_ENV = ("NANDI_WORLD", "NANDI_TERRAIN", "NANDI_SEED", "NANDI_CROPS", "NANDI_MAST_H", "NANDI_MAST_HEAD_KG",
             "NANDI_DT", "NANDI_PAYLOAD_KG", "NANDI_BUND_H")
V1_XML = os.path.join(os.path.dirname(farm_scene.ROBOT_XML), "atharv_nandi_v1.xml")
DT = 0.002
TICK_STEPS = 10                       # one 50 Hz hook tick at dt 0.002


@contextlib.contextmanager
def world_env(**env):
    saved = {k: os.environ.get(k) for k in WORLD_ENV}
    try:
        for k in WORLD_ENV:
            os.environ.pop(k, None)
        for k, v in env.items():
            os.environ[k] = str(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def build(extra=None, **env):
    """Compile a world (``extra(spec)`` adds test bodies) without touching ``model_layout.current()``."""
    env.setdefault("NANDI_TERRAIN", "flat")
    with world_env(**env):
        w = farm_scene.build_world()
        if extra is not None:
            extra(w.spec)
        return w.spec.compile()


def _id(m, kind, name):
    i = mujoco.mj_name2id(m, kind, name)
    assert i >= 0, name
    return i


def mast_driver(m, **kw):
    """SimDriver with only the mast hooks; returns (driver, runtime)."""
    reg = hook_registry.HookRegistry()
    rt = hm.register(reg)
    drv = sd.SimDriver(m, hooks=reg, **kw)
    return drv, rt


def tilt_deg(d) -> float:
    r, p = quat_to_roll_pitch(*d.qpos[3:7])
    return math.degrees(max(abs(r), abs(p)))


@pytest.fixture(scope="module")
def flat():
    return build()


# ====================================================================== pure logic (controllers.mast)
def test_split_rate_and_lens():
    assert mc.split(1.2) == pytest.approx((0.4, 0.4, 0.4))
    assert mc.split(2.0) == pytest.approx((1.6 / 3,) * 3) and mc.split(-1) == (0.0, 0.0, 0.0)
    assert mc.lens_z(0) == pytest.approx(1.10) and mc.lens_z(0.6) == pytest.approx(1.70)
    assert mc.lens_z(1.6) == pytest.approx(2.70)
    ext, n, deltas = 0.0, 0, []
    while ext < mc.EXT_MAX_M:
        new = mc.rate_step(ext, 1.6, DT)
        deltas.append(new - ext)
        ext, n = new, n + 1
    assert max(deltas) <= 0.4e-3 + 1e-12            # <= 0.4 mm per 2 ms step
    assert n * DT <= 8.5 and n * DT == pytest.approx(8.0, abs=DT)


def test_arbiter_priority_table():
    """e-stop beats safety beats recovery beats preset; lower sources are refused and dropped."""
    assert [s for s, _ in mc.priority_table()] == ["estop", "safety", "recovery", "preset"]
    a = mc.MastArbiter()
    assert a.submit(MastRequest("preset", 1.6, "tree360_high", t=0.0))
    assert a.winner().source == "preset"
    assert a.submit(MastRequest("recovery", 0.3, "safe_park", t=0.1))
    assert a.winner().source == "recovery" and "preset" not in a.requests      # preset dropped
    assert ("preset", "recovery") in a.preempted
    dec = a.submit(MastRequest("preset", 1.6, "again", t=0.2))
    assert not dec and dec.reason == "crown_gate" and a.last.detail.startswith("preempted:recovery")
    assert a.submit(MastRequest("safety", 0.0, "wind", t=0.3))
    assert a.winner().source == "safety" and "recovery" not in a.requests
    assert not a.submit(MastRequest("recovery", 0.5, "x", t=0.4))
    assert a.submit(MastRequest("estop", None, "estop", t=0.5))
    assert a.braked and a.requests == {} and a.state()["arbiter_source"] == "estop"
    for src in ("safety", "recovery", "preset"):
        d = a.submit(MastRequest(src, 0.0, "x", t=0.6))
        assert not d and a.last.detail == "estop"
    a.release("estop")
    assert not a.braked and a.winner() is None
    # the reference rule in interfaces agrees
    reqs = [MastRequest("preset", 1.0, "p", t=2), MastRequest("recovery", 0.0, "r", t=1),
            MastRequest("preset", 0.5, "p2", t=3)]
    assert interfaces.arbitrate(reqs).source == "recovery"


def test_newest_request_of_a_source_wins_and_moves_at_the_rate_limit():
    a = mc.MastArbiter()
    a.submit(MastRequest("preset", 1.6, "high", t=0))
    for _ in range(500):
        a.step(DT)
    assert a.ext == pytest.approx(0.2, abs=1e-9) and a.moving
    a.submit(MastRequest("preset", 0.1, "lower", t=1))
    for _ in range(1000):
        a.step(DT)
    assert a.ext == pytest.approx(0.1) and not a.moving and a.requests["preset"].reason == "lower"


def test_safety_request_completes_at_its_target_and_does_not_latch():
    a = mc.MastArbiter(ext=0.04)
    a.submit(MastRequest("safety", 0.0, "strike"))
    for _ in range(200):
        a.step(DT)
    assert a.ext == 0.0 and a.winner() is None and a.completed == ["strike"]
    assert a.submit(MastRequest("preset", 0.6, "mid"))         # a fresh request is possible again


def test_raises_are_gated_and_retracts_are_not():
    calls = []

    def check(e0, e1, req):
        calls.append((e0, e1))
        return GateDecision.refuse("crown_gate", "crown_clearance", detail="rope:r1")

    a = mc.MastArbiter(check=check, ext=0.8)
    dec = a.submit(MastRequest("preset", 1.6, "high"))
    assert not dec and dec.reason == "crown_clearance" and a.last.detail == "rope:r1" and calls == [(0.8, 1.6)]
    assert a.submit(MastRequest("preset", 0.2, "lower")) and len(calls) == 1      # retract never gated
    a.check = lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))
    dec = a.submit(MastRequest("preset", 1.0, "up"))
    assert not dec and dec.reason == "stability" and "gate_error" in a.last.detail


def test_hold_raise_stops_only_upward_motion():
    a = mc.MastArbiter(ext=0.5)
    a.submit(MastRequest("preset", 1.0, "up"))
    a.step(DT, hold_raise=True)
    assert a.ext == 0.5 and a.held_by == "near_obstacle"
    a.submit(MastRequest("preset", 0.0, "down"))
    a.step(DT, hold_raise=True)
    assert a.ext < 0.5


def test_brake_holds_in_place():
    a = mc.MastArbiter(ext=0.7)
    a.submit(MastRequest("preset", 1.6, "up"))
    a.brake()
    for _ in range(100):
        a.step(DT)
    assert a.ext == 0.7 and a.held_by == "estop" and not a.moving


def test_gimbal_limits():
    assert mc.clamp_gimbal(math.radians(190), 0)[0] == pytest.approx(math.radians(-170))
    assert mc.clamp_gimbal(0, math.radians(-80))[1] == pytest.approx(math.radians(-60))
    assert mc.clamp_gimbal(0, math.radians(80))[1] == pytest.approx(math.radians(45))
    a = mc.MastArbiter()
    a.submit(MastRequest("preset", 0.6, "mid", gimbal=(0.5, math.radians(-10))))
    assert a.gimbal == pytest.approx((0.5, math.radians(-10)))


def test_segment_and_point_distances():
    assert mc.seg_seg([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]) == pytest.approx(1.0)
    assert mc.seg_seg([0, 0, 0], [1, 0, 0], [0.5, -1, 1], [0.5, 1, 1]) == pytest.approx(1.0)
    assert mc.seg_seg([0, 0, 0], [0, 0, 0], [2, 0, 0], [3, 0, 0]) == pytest.approx(2.0)
    P = np.array([[2.0, 0, 0], [0, 0, 0], [0, 0, 3.0]])
    sph = mc.points_ellipsoid(P, [0, 0, 0], np.eye(3), [1, 1, 1])
    assert sph == pytest.approx([1.0, -1.0, 2.0])          # exact on a sphere
    # first-order ellipsoid distance vs a fine surface sample, near the surface (the strike / clearance regime)
    radii = np.array([1.0, 1.0, 0.6])
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 400), np.linspace(0, np.pi, 200))
    S = np.stack([radii[0] * np.cos(u) * np.sin(v), radii[1] * np.sin(u) * np.sin(v), radii[2] * np.cos(v)], -1)
    S = S.reshape(-1, 3)
    for p in ([1.1, 0.0, 0.3], [0.5, 0.5, 0.75], [0.0, 0.0, 0.8]):
        true = float(np.min(np.linalg.norm(S - np.asarray(p), axis=1)))
        est = float(mc.points_ellipsoid(np.array([p]), [0, 0, 0], np.eye(3), radii)[0])
        assert est == pytest.approx(true, abs=0.01)
    box = mc.points_box(np.array([[2.0, 0, 0], [0, 0, 0], [2, 2, 0]]), [0, 0, 0], np.eye(3), [1, 1, 1])
    assert box == pytest.approx([1.0, -1.0, math.sqrt(2)])


def _rope(z, y=0.0):
    return mc.Obstacle("rope_T", "rope", mc.CAPSULE, np.array([0.0, y, z]),
                       np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float), np.array([0.006, 1.0, 0]), 1.006)


def test_swept_clearance_hits_a_rope_above_and_not_beside():
    c0, c1 = np.array([0.0, 0, 1.1]), np.array([0.0, 0, 2.7])
    dist, ob = mc.swept_clearance([_rope(1.6)], c0, c1, 0.104)
    assert ob.name == "rope_T" and dist < 0
    dist, ob = mc.swept_clearance([_rope(1.6, y=0.25)], c0, c1, 0.104)
    assert dist == pytest.approx(0.25 - 0.006 - 0.104) and dist >= mc.CLEARANCE_M - 0.06
    dist, ob = mc.swept_clearance([_rope(1.6, y=0.5)], c0, c1, 0.104)
    assert dist > mc.CLEARANCE_M                           # bounding-box reject keeps far obstacles out
    assert mc.swept_clearance([], c0, c1, 0.1) == (math.inf, None)


def test_alley_clear_height_applies_in_the_overhang_zone_of_its_side():
    rules = mc.alley_rules({"tree_id": "T", "canopy_r_m": 1.0, "alley_sides": ["-y", "+y"],
                            "alley_clear_height_m": [1.6, 2.5]}, [0.0, 0.0, 0.0])
    assert [r.side for r in rules] == ["-y", "+y"]
    margin, rule = mc.alley_clear([0.0, -0.8], 1.76, rules)
    assert rule is not None and rule.side == "-y" and margin == pytest.approx(1.6 - 1.86)
    assert mc.alley_clear([0.0, 0.8], 1.76, rules)[1] is None          # +y side is clear to 2.5
    assert mc.alley_clear([0.0, -1.5], 2.76, rules)[1] is None         # outside canopy_r + 0.3
    allr = mc.alley_rules({"tree_id": "A", "canopy_r_m": 1.0, "alley_sides": ["all"],
                           "alley_clear_height_m": [2.5]}, [5.0, 5.0, 0.2])
    assert mc.alley_clear([5.5, 5.5], 2.76, allr)[1].tree_id == "A"
    assert mc.alley_rules({"canopy_r_m": None}, [0, 0, 0]) == []


def test_runtime_is_a_mast_control():
    assert isinstance(hm.MastRuntime(), interfaces.MastControl)
    assert isinstance(mc.MastArbiter(), interfaces.MastControl)


# ====================================================================== model: layout and dynamic parity
def test_layout_parity_mast_in_and_out():
    """Names, qpos / dof addresses and actuator ids identical with the mast composed in (atharv_nandi.xml) or
    out (atharv_nandi_v1.xml), and unchanged by mast body_pos writes."""
    robot, v1 = mujoco.MjModel.from_xml_path(farm_scene.ROBOT_XML), mujoco.MjModel.from_xml_path(V1_XML)
    assert hm.has_mast(robot) and not hm.has_mast(v1)
    a, b = mlay.Layout.from_model(robot), mlay.Layout.from_model(v1)
    for f in ("nq", "nv", "nu", "nmocap", "neq", "joints", "actuators", "actuator_joint", "equalities", "mocap"):
        assert getattr(a, f) == getattr(b, f), f
    for s, z in zip(hm.SECTIONS, mc.split(1.6)):
        robot.body_pos[_id(robot, O.mjOBJ_BODY, s), 2] = z
    assert mlay.Layout.from_model(robot) == a and mlay.frozen_violations(a) == []


def test_dynamic_parity_ext_held_zero_hook_on_vs_off(flat):
    """ext held at 0: the mast hooks on vs off give byte-identical qpos over 2,000 steps of a walk."""
    def run(reg):
        drv = sd.SimDriver(flat, hooks=reg)
        drv.gait.set_command(0.6, 0.0, 0.2, hold=True)
        drv.run_steps(2000)
        out = drv.qpos()
        drv.close()
        return out, drv

    off, _ = run(hook_registry.HookRegistry())
    reg = hook_registry.HookRegistry()
    rt = hm.register(reg)
    on, drv = run(reg)
    assert off.tobytes() == on.tobytes()
    assert rt.substeps == 2000 and rt.arb.ext == 0.0
    assert np.linalg.norm(on[0:2] - flat.key_qpos[0][0:2]) > 0.1          # the robot walked


# ====================================================================== model: ramp, lens, provider
def test_ramp_0_to_1p6_to_0_per_step_lens_and_stroke(flat):
    sim_state.discover_providers()
    drv, rt = mast_driver(flat)
    s1 = _id(flat, O.mjOBJ_BODY, "mast_s1")
    cam = _id(flat, O.mjOBJ_CAMERA, "crown_cam")
    drv.stand(0.3)
    lens0 = float(drv.d.cam_xpos[cam][2])
    assert lens0 == pytest.approx(mc.LENS_STOWED_M, abs=0.02)
    assert rt.submit(MastRequest("preset", 1.6, "tree360_high", t=drv.t))
    t0, prev, deltas, lens = drv.t, 0.0, [], {}
    marks = {0.6: None, 1.6: None}
    while rt.arb.ext < 1.6 and drv.t - t0 < 9.0:
        drv.step()
        z = 3.0 * float(flat.body_pos[s1, 2])
        deltas.append(abs(z - prev))
        prev = z
        for k in marks:
            if marks[k] is None and rt.arb.ext >= k - 1e-9:
                marks[k] = drv.t
    stroke = drv.t - t0
    drv.step()                                                     # kinematics of the last write
    lens[1.6] = float(drv.d.cam_xpos[cam][2])
    assert stroke <= 8.5 and stroke == pytest.approx(8.0, abs=0.01)
    assert max(deltas) <= 0.4e-3 + 1e-9 and len(deltas) >= 3999            # per physics step, not 50 Hz
    assert lens[1.6] == pytest.approx(lens0 + 1.6, abs=0.02)
    assert lens[1.6] == pytest.approx(2.70, abs=0.02)
    st = rt.state()
    assert st["ext_m"] == pytest.approx(1.6) and st["arbiter_source"] == "preset" and not st["moving"]
    # all three sections share the stroke (ext / 3 each)
    for s in hm.SECTIONS:
        assert flat.body_pos[_id(flat, O.mjOBJ_BODY, s), 2] == pytest.approx(1.6 / 3)
    # sim_state provider round trip into a fresh model: renderer shows the raised head
    state = drv.sim_state("w")
    blk = state["mutations"]["providers"]["mast"]
    assert blk["v"] == 1 and blk["data"]["ext_m"] == pytest.approx(1.6)
    fresh = build()
    fd = mujoco.MjData(fresh)
    ok, why = camera_rig.set_state(fresh, fd, json.loads(sim_state.encode(state)), world_id="w")
    assert ok, why
    assert float(fd.cam_xpos[cam][2]) == pytest.approx(lens[1.6], abs=1e-6)
    # back down to 0
    assert rt.submit(MastRequest("preset", 0.0, "stow", t=drv.t))
    t1 = drv.t
    drv.run_until(lambda dr: rt.arb.ext <= 0.0, timeout_s=9.0)
    assert drv.t - t1 <= 8.5 and flat.body_pos[s1, 2] == 0.0
    drv.stand(0.1)
    assert float(drv.d.cam_xpos[cam][2]) == pytest.approx(lens0, abs=0.01)
    assert tilt_deg(drv.d) < 3.0
    drv.close()


def test_ext_0p6_lens_band_mid(flat):
    drv, rt = mast_driver(flat)
    drv.stand(0.2)
    rt.submit(MastRequest("preset", 0.6, "mid", t=drv.t))
    drv.run_until(lambda dr: not rt.arb.moving, timeout_s=4.0)
    drv.stand(0.1)
    assert rt.state()["lens_z"] == pytest.approx(1.70, abs=0.02)
    drv.reset()                                                     # world reset: fields back to rest
    assert all(flat.body_pos[_id(flat, O.mjOBJ_BODY, s), 2] == 0.0 for s in hm.SECTIONS)
    assert rt.arb.ext == 0.0 and rt.arb.winner() is None
    drv.close()


def test_provider_packs_gimbal_and_refuses_bad_blocks(flat):
    assert hm.PROVIDER_KEY in sim_state.discover_providers()
    m = build()
    cam = _id(m, O.mjOBJ_CAMERA, "crown_cam")
    assert hm.gimbal_quat(0, 0) == pytest.approx(m.cam_quat[cam])          # WP1 rest orientation
    assert hm._pack(m) == {"ext_m": 0.0, "gimbal": [0.0, 0.0]}
    hm._unpack(m, {"ext_m": 0.9, "gimbal": [math.radians(90), math.radians(-20)]})
    got = hm._pack(m)
    assert got["ext_m"] == pytest.approx(0.9)
    assert got["gimbal"] == pytest.approx([math.radians(90), math.radians(-20)], abs=1e-9)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    fwd = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
    assert fwd == pytest.approx([0.0, math.cos(math.radians(20)), -math.sin(math.radians(20))], abs=2e-3)
    up = d.cam_xmat[cam].reshape(3, 3)[:, 1]
    assert up[2] > 0.9                                                   # level horizon, no roll
    for bad in ({"ext_m": 2.0}, {"ext_m": float("nan")}, {"ext_m": 0.1, "gimbal": [0, 2.0]}, {"ext_m": "x"},
                {"ext_m": 0.1, "gimbal": [1]}):
        with pytest.raises(ValueError):
            hm._unpack(m, bad)
    v1 = mujoco.MjModel.from_xml_path(V1_XML)
    assert hm._pack(v1) == {} and not hm.has_mast(v1)
    with pytest.raises(ValueError):
        hm._unpack(v1, {"ext_m": 0.1})


def test_gimbal_aim_writes_cam_quat_every_step(flat):
    drv, rt = mast_driver(flat)
    cam = _id(flat, O.mjOBJ_CAMERA, "crown_cam")
    rest = flat.cam_quat[cam].copy()
    rt.aim(math.radians(45), math.radians(-30))
    drv.run_steps(2)
    fwd = -drv.d.cam_xmat[cam].reshape(3, 3)[:, 2]
    assert math.degrees(math.asin(fwd[2])) == pytest.approx(-30, abs=1.0)
    rt.aim(math.radians(400), math.radians(-90))                        # clamped: pan wraps, tilt -60
    assert rt.arb.gimbal == pytest.approx((math.radians(40), math.radians(-60)))
    drv.reset()
    assert flat.cam_quat[cam] == pytest.approx(rest)
    drv.close()


# ====================================================================== model: strike
def _strike_world(kind):
    """Legacy flat world + a mocap obstacle that sweeps horizontally into the crown head at ext 0.6."""
    def extra(spec):
        b = spec.worldbody.add_body(name="strike_obstacle", mocap=True, pos=[-0.30, 1.0, 1.70])
        if kind == "wood":   # horizontal scaffold branch along x, wood class (group 0, contype 1 / conaffinity 0)
            b.add_geom(name="tree_T_wood_0", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[-0.6, 0, 0, 0.6, 0, 0],
                       size=[0.03, 0, 0], group=0, contype=1, conaffinity=0)
        else:                # foliage envelope ellipsoid, query only (group 5, contype 0)
            b.add_geom(name="tree_T_envelope_0", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.5, 0.35, 0.4],
                       group=5, contype=0, conaffinity=0)
    return build(extra)


@pytest.mark.parametrize("kind,geom_distance", [("wood", True), ("envelope", True), ("wood", False)])
def test_head_driven_into_wood_or_envelope_strikes_and_retracts_within_one_tick(kind, geom_distance):
    m = _strike_world(kind)
    reg = hook_registry.HookRegistry()
    rt = hm.register(reg)
    rt.use_geom_distance = geom_distance
    drv = sd.SimDriver(m, hooks=reg)
    mid = int(m.body_mocapid[_id(m, O.mjOBJ_BODY, "strike_obstacle")])
    drv.stand(0.3)
    assert rt.submit(MastRequest("preset", 0.6, "mid", t=drv.t)), rt.arb.last
    drv.run_until(lambda dr: not rt.arb.moving, timeout_s=4.0)
    drv.stand(0.2)
    tilt0 = tilt_deg(drv.d)
    head_y = float(drv.d.xpos[rt.head_b][1])
    # the obstacle approaches the head along -y at 0.5 m/s
    exts, strike_step, near_seen = [], None, False
    for k in range(int(4.0 / DT)):
        drv.d.mocap_pos[mid][1] = max(head_y - 0.05, 1.0 - 0.5 * DT * k)
        drv.step()
        exts.append(rt.arb.ext)
        near_seen |= rt.near
        if strike_step is None and rt.strikes:
            strike_step = k
        if strike_step is not None and k > strike_step + TICK_STEPS + 1:
            break
    assert strike_step is not None, f"no strike, last distance {rt.dist}"
    ev = rt.strikes[0]
    assert ev["kind"] == kind and ev["geom"].startswith("tree_T_") and ev["ext_m"] == pytest.approx(0.6, abs=1e-6)
    assert near_seen
    # retract within 1 hook tick: the setpoint is already moving down within TICK_STEPS of the event
    assert exts[strike_step + TICK_STEPS] < 0.6 - 1e-6
    assert exts[strike_step] < 0.6                                       # in fact on the same step
    assert rt.arb.winner().source == "safety" and rt.arb.winner().reason == "strike"
    # the next tick publishes the event
    drv.run_steps(TICK_STEPS)
    pubs = [json.loads(p) for _, topic, p in drv.published if topic == hm.STATE_TOPIC]
    assert pubs[-1]["strike_count"] == 1 and pubs[-1]["strike"]["kind"] == kind
    # the strike never pushes the robot: tilt change <= 3 deg over the event and the retract
    tilts = []
    drv.add_observer(lambda dr: tilts.append(tilt_deg(dr.d)))
    drv.run_until(lambda dr: rt.arb.ext <= 0.0, timeout_s=4.0)
    drv.stand(0.5)
    assert max(abs(t - tilt0) for t in tilts) <= 3.0
    assert rt.arb.winner() is None and rt.arb.ext == 0.0                # safety retract completed
    drv.close()


def test_strike_while_braked_is_recorded_but_the_mast_holds():
    m = _strike_world("wood")
    drv, rt = mast_driver(m)
    mid = int(m.body_mocapid[_id(m, O.mjOBJ_BODY, "strike_obstacle")])
    drv.stand(0.2)
    rt.submit(MastRequest("preset", 0.6, "mid", t=drv.t))
    drv.run_until(lambda dr: not rt.arb.moving, timeout_s=4.0)
    drv.enter_hold("estop")
    drv.d.mocap_pos[mid][1] = float(drv.d.xpos[rt.head_b][1])
    drv.run_steps(150)            # a teleport outruns the 2 m/s coherence bound: detected within 0.25 s
    assert rt.strikes and rt.strikes[0]["braked"] and rt.arb.ext == pytest.approx(0.6)
    drv.close()


# ====================================================================== model: raise gates
def _rope_world():
    def extra(spec):
        spec.worldbody.add_geom(name="rope_T", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                fromto=[-1.5, 0.0, 1.6, 1.5, 0.0, 1.6], size=[0.006, 0, 0], group=5, contype=0,
                                conaffinity=0)
        spec.worldbody.add_geom(name="foliage_T", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[3.0, 0.0, 2.0],
                                size=[0.4, 0.4, 0.1], group=3, contype=0, conaffinity=0)
    return build(extra)


def _bound(m):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    rt = hm.MastRuntime()
    rt.bus = hook_registry.Bus()
    rt.bind(m, d)
    return rt, d


def test_raise_refused_when_the_swept_cylinder_hits_a_rope_line():
    m = _rope_world()
    rt, d = _bound(m)
    dec = rt.submit(MastRequest("preset", 1.6, "high"))
    assert not dec and dec.reason == "crown_clearance" and dec.values["detail"] == "rope:rope_T"
    assert rt.state()["gated"] and rt.state()["detail"] == "rope:rope_T" and rt.arb.ext == 0.0
    assert rt.submit(MastRequest("preset", 0.25, "below the rope"))      # swept sphere top 1.474, 0.12 clear
    assert not rt.submit(MastRequest("preset", 0.35, "too close"))       # 0.02 m < 0.10 m clearance


def test_raise_refused_under_the_orchard_rope_B1_B2():
    m = build(NANDI_WORLD="orchard_demo")
    rt, d = _bound(m)
    g = _id(m, O.mjOBJ_GEOM, "rope_B1_B2")
    d.qpos[0:2] = d.geom_xpos[g][:2]
    mujoco.mj_forward(m, d)
    dec = rt.submit(MastRequest("preset", 1.6, "high"))
    assert not dec and dec.reason == "crown_clearance"
    ropes = [o for o in rt.obstacles() if o.kind == "rope"]
    c0 = rt.head_pos()
    dist, ob = mc.swept_clearance(ropes, c0, c0 + rt.axis() * 1.6, rt.r_head)
    assert ob.name == "rope_B1_B2" and dist < mc.CLEARANCE_M
    assert len(rt.alleys) > 0                                            # registry alley rules bound


def test_lidar_return_in_the_swept_cylinder_refuses():
    m = _rope_world()
    rt, d = _bound(m)
    d.qpos[0] = 3.3                                    # head (trunk x - 0.30) under the foliage box, rope away
    mujoco.mj_forward(m, d)
    dec = rt.submit(MastRequest("preset", 1.6, "high"))
    assert not dec and dec.values["detail"] == "lidar:foliage_T"
    rt.lidar = False
    assert rt.submit(MastRequest("preset", 1.6, "high"))   # foliage is not an obstacle class: only LiDAR sees it


def test_raise_gates_estop_standing_slope_wind_and_wp6(flat):
    rt, d = _bound(build())
    bus = rt.bus
    bus.values["hold"] = {"reason": "estop", "since": 0.0}
    dec = rt.check_raise(0.0, 0.6, MastRequest("preset", 0.6, "x"))
    assert not dec and dec.values["detail"] == "estop"
    bus.values["hold"] = None
    d.qvel[0] = 0.2
    assert rt.check_raise(0.0, 0.6, MastRequest("preset", 0.6, "x")).reason == "stability"
    d.qvel[0] = 0.0
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([1.0, 0, 0]), math.radians(5))
    keep = d.qpos[3:7].copy()
    d.qpos[3:7] = q
    assert rt.check_raise(0.0, 0.6, MastRequest("preset", 0.6, "x")).reason == "slope"
    d.qpos[3:7] = keep
    bus.values["wind"] = {"speed_mps": 10.0, "gust": False}
    assert rt.check_raise(0.0, 0.6, MastRequest("preset", 0.6, "x")).reason == "wind"
    bus.values["wind"] = {"speed_mps": 4.0, "gust": True}
    assert rt.check_raise(0.0, 0.6, MastRequest("preset", 0.6, "x")).reason == "wind"
    bus.values["wind"] = {"speed_mps": 4.0, "gust": False}

    def wp6(e0, e1, ctx):
        return GateDecision.refuse("crown_margin", "stability") if e1 > 1.0 else GateDecision.ok("crown_margin")

    hm.register_gate(wp6)
    try:
        assert rt.check_raise(0.0, 0.6, MastRequest("preset", 0.6, "x"))
        dec = rt.check_raise(0.0, 1.6, MastRequest("preset", 1.6, "x"))
        assert not dec and dec.reason == "stability" and dec.values["detail"] == "crown_margin"
    finally:
        hm.unregister_gate(wp6)


# ====================================================================== model: e-stop brake
def test_estop_brake_holds_and_needs_a_new_request_after_the_hold(flat):
    drv, rt = mast_driver(flat)
    drv.stand(0.2)
    rt.submit(MastRequest("preset", 1.6, "high", t=drv.t))
    drv.wait(2.0)
    drv.enter_hold("estop")
    drv.step()
    held = rt.arb.ext
    assert 0.3 < held < 0.5 and rt.arb.braked
    for src in ("safety", "recovery", "preset"):
        assert not rt.submit(MastRequest(src, 0.0, "retract", t=drv.t))
    rt._backstops(drv.t)
    drv.wait(1.5)
    assert rt.arb.ext == held and rt.state()["arbiter_source"] == "estop"
    pubs = [json.loads(p) for _, topic, p in drv.published if topic == hm.STATE_TOPIC]
    assert pubs[-1]["braked"] and pubs[-1]["arbiter_reason"] == "estop_brake"
    drv.leave_hold()
    drv.wait(1.0)
    assert rt.arb.ext == held and not rt.arb.braked and rt.arb.winner() is None   # no auto-resume
    assert rt.submit(MastRequest("recovery", 0.0, "estop_recovery_retract", t=drv.t))
    drv.run_until(lambda dr: rt.arb.ext <= 0.0, timeout_s=3.0)
    assert rt.arb.ext == 0.0
    drv.close()


def test_explicit_brake_is_not_released_by_the_runner(flat):
    rt, d = _bound(build())
    rt.brake()
    m = rt.m
    rt.substep(m, d, 0.0, rt.bus)
    assert rt.arb.braked
    rt.release("estop")
    assert not rt.arb.braked


# ====================================================================== model: commands, backstops, hooks
def test_cmd_topic_requests_release_aim_and_rejects():
    m = build()
    reg = hook_registry.HookRegistry()
    rt = hm.register(reg)
    drv = sd.SimDriver(m, hooks=reg)
    drv.stand(0.2)
    drv.deliver(hm.CMD_TOPIC, {"source": "preset", "band": "mid", "reason": "tree360", "gimbal_deg": [90, -10]})
    drv.run_steps(TICK_STEPS + 1)
    assert rt.arb.winner().target_ext == 0.6 and rt.arb.gimbal == pytest.approx((math.pi / 2, math.radians(-10)))
    for bad in ({"source": "estop", "target_ext": 0.0, "reason": "x"}, {"source": "preset", "target_ext": 9},
                "not json", {"source": "preset", "band": "top"}, [1, 2], {"release": "estop"},
                {"aim_deg": [1]}):
        drv.deliver(hm.CMD_TOPIC, bad)
        drv.run_steps(TICK_STEPS)
    assert rt.rejected == 7 and rt.arb.winner().source == "preset"
    drv.deliver(hm.CMD_TOPIC, {"aim_deg": [0, 20]})
    drv.run_steps(TICK_STEPS)
    assert rt.arb.gimbal == pytest.approx((0.0, math.radians(20)))
    drv.deliver(hm.CMD_TOPIC, json.dumps({"release": "preset"}))
    drv.run_steps(TICK_STEPS)
    assert rt.arb.winner() is None
    st = drv.bus.values["mast"]
    assert st["rejected"] == 7 and "ext_m" in st and drv.bus.values["mast_control"] is rt
    pubs = [json.loads(p) for _, topic, p in drv.published if topic == hm.STATE_TOPIC]
    assert pubs and all(json.dumps(p, allow_nan=False) for p in pubs)
    assert drv.hook_faults == {} and not drv.holding
    drv.close()


def test_wind_gust_and_tilt_backstops_retract():
    rt, d = _bound(build())
    rt.arb.ext = 1.0
    rt.bus.values["wind"] = {"speed_mps": 11.0, "gust": False}
    rt._backstops(1.0)                                  # standing: retract at once
    assert rt.arb.winner().source == "safety" and rt.arb.winner().reason == "wind"
    rt.arb.reset(1.0)
    rt.bus.values["wind"] = {"speed_mps": 5.0, "gust": True}
    d.qvel[0] = 0.15                                    # walking: WP6 stance first, retract after 0.5 s
    rt._backstops(2.0)
    assert rt.arb.winner() is None
    rt._backstops(2.5)
    assert rt.arb.winner().reason == "gust"
    rt.arb.reset(1.0)
    rt.bus.values["wind"] = None
    d.qvel[0] = 0.0
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([0, 1.0, 0]), math.radians(13))
    d.qpos[3:7] = q
    rt._backstops(3.0)
    assert rt.arb.winner().reason == "tilt"
    rt.arb.reset(0.0)
    rt._backstops(4.0)                                  # stowed: nothing to retract
    assert rt.arb.winner() is None


def test_hook_file_registers_two_hooks():
    reg = hook_registry.load_all(names=["mast"])
    rows = {r["name"]: r for r in reg.table()}
    assert set(rows) == {"mast", "mast_step"}
    assert rows["mast_step"]["every_step"] and rows["mast_step"]["mutates_model"]
    assert rows["mast"]["rate_hz"] == hm.RATE_HZ and rows["mast"]["topics_in"] == [hm.CMD_TOPIC]
    assert reg.has_substep


def test_hooks_are_harmless_without_a_mast():
    v1 = mujoco.MjModel.from_xml_path(V1_XML)
    d = mujoco.MjData(v1)
    mujoco.mj_forward(v1, d)
    rt = hm.MastRuntime()
    bus = hook_registry.Bus()
    rt.substep(v1, d, 0.0, bus)
    rt.tick(v1, d, 0.0, bus)
    assert bus.values["mast"] == {"ext_m": 0.0, "has_mast": False}
    assert not rt.submit(MastRequest("preset", 0.6, "x"))


# ====================================================================== review fixes (WP24 fix round 1)
def test_safety_source_only_retracts():
    a = mc.MastArbiter(ext=0.5)
    dec = a.submit(MastRequest("safety", 1.6, "console"))
    assert not dec and a.last.detail == "safety_raise" and a.winner() is None
    assert a.submit(MastRequest("safety", 0.5, "hold_here")) and a.submit(MastRequest("safety", 0.2, "down"))


def test_safety_raise_on_cmd_cannot_preempt_recovery_or_mute_the_backstops():
    m = build()
    drv, rt = mast_driver(m)
    drv.stand(0.3)
    assert rt.submit(MastRequest("preset", 1.0, "x", t=drv.t))
    drv.run_until(lambda dr: not rt.arb.moving, timeout_s=8.0)
    assert rt.submit(MastRequest("recovery", 0.0, "link_loss", t=drv.t))
    drv.run_steps(200)
    drv.deliver(hm.CMD_TOPIC, json.dumps({"source": "safety", "target_ext": 1.6, "reason": "console"}))
    drv.run_steps(TICK_STEPS + 1)
    assert list(rt.arb.requests) == ["recovery"] and rt.arb.last.detail == "safety_raise"
    assert rt.arb.target() == 0.0


def test_backstops_run_under_a_partial_safety_retract():
    rt, d = _bound(build())
    rt.arb.ext = 1.0
    assert rt.submit(MastRequest("safety", 0.6, "wp6_margin"))         # a safety retract to a lower band
    rt.bus.values["wind"] = {"speed_mps": 4.0, "gust": True}
    rt._backstops(1.0)
    assert rt.arb.winner().reason == "gust" and rt.arb.winner().target_ext == 0.0
    rt.bus.values["wind"] = {"speed_mps": 11.0, "gust": False}
    rt._backstops(1.1)                                                  # a retract to 0 runs: not re-submitted
    assert rt.arb.winner().reason == "gust"


def test_running_raise_is_held_and_dropped_when_the_robot_walks(flat):
    drv, rt = mast_driver(flat)
    drv.stand(0.3)
    assert rt.submit(MastRequest("preset", 1.6, "high", t=drv.t))
    drv.walk(0.6, seconds=2.0)
    held = rt.arb.ext
    assert held < 0.15, held                       # stopped within a tick of losing the stance (was 1.0 after 5 s)
    assert rt.arb.winner() is None and rt.arb.aborted == [("preset", "not_standing")]
    last = rt.state()["last_decision"]
    assert not last["allowed"] and last["reason"] == "stability" and last["detail"] == "regate:not_standing"
    drv.walk(0.6, seconds=1.0)
    assert rt.arb.ext == held                       # nothing re-raises by itself
    drv.close()


def test_running_raise_is_regated_against_wp6_gates_every_tick(flat):
    drv, rt = mast_driver(flat)
    drv.stand(0.3)
    assert rt.submit(MastRequest("preset", 1.6, "high", t=drv.t))
    drv.wait(1.0)

    def wp6(e0, e1, ctx):
        return GateDecision.refuse("crown_margin", "stability")

    hm.register_gate(wp6)
    try:
        drv.run_steps(TICK_STEPS + 1)
        e = rt.arb.ext
        assert rt.arb.held_by == "gate:crown_margin" and rt.arb.winner() is not None
        drv.run_steps(TICK_STEPS * 10)
        assert rt.arb.ext == e                       # held, not moving
        drv.wait(0.5)
        assert rt.arb.winner() is None and rt.state()["last_decision"]["detail"] == "regate:crown_margin"
    finally:
        hm.unregister_gate(wp6)
    drv.close()


def test_transient_gate_failure_holds_then_resumes(flat):
    drv, rt = mast_driver(flat)
    drv.stand(0.3)
    assert rt.submit(MastRequest("preset", 1.0, "x", t=drv.t))
    drv.wait(0.5)
    until = drv.t + 0.2

    def wp6(e0, e1, ctx):
        return GateDecision.refuse("crown_margin", "stability") if ctx["d"].time < until else GateDecision.ok("m")

    hm.register_gate(wp6)
    try:
        drv.run_steps(TICK_STEPS + 1)
        e = rt.arb.ext
        assert rt.arb.held_by == "gate:crown_margin"
        drv.wait(0.3)                                # failing < REGATE_DROP_S: the raise resumes
        assert rt.arb.ext > e and rt.arb.winner().source == "preset" and rt.arb.aborted == []
    finally:
        hm.unregister_gate(wp6)
    drv.close()


def test_wind_during_a_raise_retracts(flat):
    drv, rt = mast_driver(flat)
    drv.stand(0.3)
    assert rt.submit(MastRequest("preset", 1.0, "x", t=drv.t))
    drv.wait(0.5)
    drv.bus.values["wind"] = {"speed_mps": 10.5, "gust": False}       # wind gate fails, backstop retracts
    drv.run_steps(TICK_STEPS + 1)
    assert rt.arb.winner().source == "safety" and rt.arb.winner().reason == "wind"
    drv.close()


def test_strike_during_the_estop_brake_retracts_after_release():
    m = _strike_world("wood")
    drv, rt = mast_driver(m)
    mid = int(m.body_mocapid[_id(m, O.mjOBJ_BODY, "strike_obstacle")])
    drv.stand(0.2)
    rt.submit(MastRequest("preset", 0.6, "mid", t=drv.t))
    drv.run_until(lambda dr: not rt.arb.moving, timeout_s=4.0)
    drv.enter_hold("estop")
    drv.d.mocap_pos[mid][1] = float(drv.d.xpos[rt.head_b][1])
    drv.run_steps(150)
    assert rt.strikes and rt.strikes[0]["braked"] and rt.arb.ext == pytest.approx(0.6)
    drv.leave_hold()
    drv.step()
    w = rt.arb.winner()
    assert w is not None and w.source == "safety" and w.reason == "strike" and rt.arb.ext < 0.6
    drv.run_until(lambda dr: rt.arb.ext <= 0.0, timeout_s=4.0)
    assert rt.arb.ext == 0.0 and len(rt.strikes) == 1
    drv.close()


def test_head_reference_is_the_head_box_not_the_body_origin():
    rt, d = _bound(build())
    off = d.geom_xpos[rt.head_g] - d.xpos[rt.head_b]
    assert off[2] == pytest.approx(0.02, abs=1e-6) and rt.head_half == pytest.approx(0.06)
    assert rt.head_pos() == pytest.approx(d.geom_xpos[rt.head_g])
    top = float(d.geom_xpos[rt.head_g][2]) + 0.06                    # real box top
    hx, hy = d.geom_xpos[rt.head_g][:2]
    # the physical head top is the anemometer, 0.05 m above the box top: a clear height 0.09 m above the box top
    # is refused with margin -0.06 (the box-top reference gave -0.01), 0.15 m above the box top passes
    assert rt.head_top == pytest.approx(0.11, abs=1e-6)
    rt.alleys = [mc.AlleyRule("T", float(hx), float(hy) + 0.5, 0.0, 1.0, "all", top + 0.09)]
    dec = rt.clearance(0.0, 0.0)
    assert not dec and dec.values["detail"] == "alley_clear_height:T"
    assert dec.values["margin_m"] == pytest.approx(-0.06, abs=1e-6)
    rt.alleys = [mc.AlleyRule("T", float(hx), float(hy) + 0.5, 0.0, 1.0, "all", top + 0.10)]
    assert not rt.clearance(0.0, 0.0)                                 # passed with the box-top reference
    rt.alleys = [mc.AlleyRule("T", float(hx), float(hy) + 0.5, 0.0, 1.0, "all", top + 0.15)]
    assert rt.clearance(0.0, 0.0)


def test_two_cmds_in_one_tick_both_apply_and_overflow_is_counted():
    m = build()
    drv, rt = mast_driver(m)
    drv.stand(0.3)
    drv.deliver(hm.CMD_TOPIC, json.dumps({"source": "preset", "target_ext": 1.0, "reason": "a"}))
    drv.deliver(hm.CMD_TOPIC, json.dumps({"aim_deg": [30, -10]}))
    drv.run_steps(TICK_STEPS + 1)
    assert list(rt.arb.requests) == ["preset"] and rt.state()["gimbal_deg"] == [30.0, -10.0]
    n = hook_registry.HISTORY_N + 5
    for k in range(n):
        drv.deliver(hm.CMD_TOPIC, {"aim_deg": [k, 0]})
    drv.run_steps(TICK_STEPS + 1)
    st = rt.state()
    assert st["cmd_lost"] == 5 and st["gimbal_deg"][0] == pytest.approx(n - 1)
    drv.close()


def test_bus_since_window():
    bus = hook_registry.Bus()
    assert bus.since("/nandi/x", 0) == ([], 0)
    for k in range(3):
        bus.deliver("/nandi/x", k)
    assert bus.since("/nandi/x", 1) == ([1, 2], 0) and bus.since("/nandi/x", 3) == ([], 0)
    for k in range(hook_registry.HISTORY_N + 2):
        bus.deliver("/nandi/x", k)
    got, lost = bus.since("/nandi/x", 0)
    assert len(got) == hook_registry.HISTORY_N and lost == 5 and got[-1] == hook_registry.HISTORY_N + 1


def test_aim_of_a_refused_request_is_not_applied():
    m = build()
    drv, rt = mast_driver(m)
    drv.stand(0.3)
    drv.enter_hold("estop")
    drv.step()
    drv.deliver(hm.CMD_TOPIC, {"source": "preset", "target_ext": 0.6, "reason": "x", "aim_deg": [45, 0]})
    drv.run_steps(TICK_STEPS + 1)
    assert rt.arb.gimbal == (0.0, 0.0) and rt.arb.last.detail == "estop"
    drv.deliver(hm.CMD_TOPIC, {"aim_deg": [45, 0]})                  # aim alone is render-only: allowed
    drv.run_steps(TICK_STEPS + 1)
    assert rt.arb.gimbal == pytest.approx((math.pi / 4, 0.0))
    drv.close()


def test_hold_lower_stops_preset_and_recovery_retracts_but_never_safety():
    for src, moves in (("preset", False), ("recovery", False), ("safety", True)):
        a = mc.MastArbiter(ext=0.5)
        a.submit(MastRequest(src, 0.0, "down"))
        a.step(DT, hold_lower=True)
        assert (a.ext < 0.5) == moves and (a.held_by == "near_obstacle") != moves, src


def test_near_obstacle_stops_lowering_only_when_it_closes():
    m = _strike_world("wood")
    drv, rt = mast_driver(m)
    mid = int(m.body_mocapid[_id(m, O.mjOBJ_BODY, "strike_obstacle")])
    drv.stand(0.2)
    assert rt.submit(MastRequest("preset", 0.6, "mid", t=drv.t))
    drv.run_until(lambda dr: not rt.arb.moving, timeout_s=4.0)
    hc = drv.d.geom_xpos[rt.head_g].copy()
    # branch beside and below the head box corner (0.015 m away): lowering closes on it -> held
    drv.d.mocap_pos[mid][:] = [-0.30, hc[1] + 0.10, hc[2] - 0.08]
    drv.run_steps(60)
    assert rt.near and not rt.strikes
    assert rt.submit(MastRequest("preset", 0.0, "stow", t=drv.t))
    drv.run_steps(50)
    assert rt.arb.ext == pytest.approx(0.6) and rt.arb.held_by == "near_obstacle"
    # a safety retract always runs
    assert rt.submit(MastRequest("safety", 0.0, "operator", t=drv.t))
    drv.run_steps(50)
    assert rt.arb.ext < 0.6 - 0.008
    drv.close()
    # branch just above the head: lowering moves away -> a preset stow runs
    drv, rt = mast_driver(m)
    drv.d.mocap_pos[mid][:] = [-0.30, 1.0, 1.70]
    drv.stand(0.2)
    assert rt.submit(MastRequest("preset", 0.6, "mid", t=drv.t))
    drv.run_until(lambda dr: not rt.arb.moving, timeout_s=4.0)
    hc = drv.d.geom_xpos[rt.head_g].copy()
    drv.d.mocap_pos[mid][:] = [-0.30, hc[1], hc[2] + rt.head_top + 0.03 + 0.02]   # above the anemometer top
    drv.run_steps(60)
    assert rt.near and not rt.strikes
    assert rt.submit(MastRequest("preset", 0.0, "stow", t=drv.t))
    drv.run_steps(50)
    assert rt.arb.ext < 0.6 - 0.008
    drv.close()


def test_provider_carries_lens_z_and_arbiter_source(flat):
    drv, rt = mast_driver(flat)
    drv.stand(0.2)
    assert rt.submit(MastRequest("preset", 0.6, "mid", t=drv.t))
    drv.run_until(lambda dr: not rt.arb.moving, timeout_s=4.0)
    blk = drv.sim_state("w")["mutations"]["providers"]["mast"]["data"]
    assert blk["arbiter_source"] == "preset"
    assert blk["crown_lens_z"] == pytest.approx(float(drv.d.cam_xpos[rt.cam][2]), abs=1e-6)
    hm._unpack(flat, blk)                                              # informational fields pass the checks
    for bad in (dict(blk, arbiter_source="console"), dict(blk, crown_lens_z=float("inf"))):
        with pytest.raises(ValueError):
            hm._unpack(flat, bad)
    drv.close()


# ====================================================================== review fixes (round 2)
def test_head_geoms_cover_the_anemometer_and_the_gnss_puck():
    rt, d = _bound(build())
    m = rt.m
    names = [mujoco.mj_id2name(m, O.mjOBJ_GEOM, g) for g in rt.strike_geoms]
    assert names[0] == "crown_head_g" and names[-1] == "mast_s3_g"
    assert {"anemometer_g", "gnss_puck"} <= set(names)
    c = d.geom_xpos[rt.head_g]
    an = _id(m, O.mjOBJ_GEOM, "anemometer_g")
    an_top = float(d.geom_xpos[an][2] + m.geom_size[an][1])
    assert float(c[2]) + rt.head_top == pytest.approx(an_top, abs=1e-6)          # anemometer top, box top + 0.05
    # the swept bounding sphere encloses every head geom (the box rbound 0.104 missed the anemometer by 0.016)
    for g in rt.head_geoms:
        assert np.linalg.norm(d.geom_xpos[g] - c) + m.geom_rbound[g] <= rt.r_head + 1e-9
    assert rt.r_head > float(m.geom_rbound[rt.head_g]) + 0.015


@pytest.mark.parametrize("geom_distance", [True, False])
def test_thin_branch_at_anemometer_height_strikes_and_retracts(geom_distance):
    """A 1 cm branch 3 cm above the box top (below the anemometer top) swept into the crown: before the fix the box
    stayed 0.02 m clear, so there was no strike while the branch went 0.022 m into the anemometer."""
    def extra(spec):
        b = spec.worldbody.add_body(name="strike_obstacle", mocap=True, pos=[-0.30, 1.0, 1.70])
        b.add_geom(name="tree_T_wood_0", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[-0.6, 0, 0, 0.6, 0, 0],
                   size=[0.01, 0, 0], group=0, contype=0, conaffinity=0)
    m = build(extra)
    drv, rt = mast_driver(m)
    rt.use_geom_distance = geom_distance
    mid = int(m.body_mocapid[_id(m, O.mjOBJ_BODY, "strike_obstacle")])
    drv.stand(0.3)
    assert rt.submit(MastRequest("preset", 0.6, "mid", t=drv.t)), rt.arb.last
    drv.run_until(lambda dr: not rt.arb.moving, timeout_s=4.0)
    drv.stand(0.2)
    hb = rt.head_g
    z = float(drv.d.geom_xpos[hb][2] + m.geom_size[hb][2]) + 0.03
    for k in range(int(3.0 / DT)):
        drv.d.mocap_pos[mid][:] = [float(drv.d.geom_xpos[hb][0]), 1.0 - 0.5 * DT * k, z]
        drv.step()
        if rt.strikes:
            break
    assert rt.strikes and rt.strikes[0]["geom"] == "tree_T_wood_0", f"no strike, last distance {rt.dist}"
    w = rt.arb.winner()
    assert w is not None and w.source == "safety" and w.reason == "strike" and rt.arb.ext < 0.6
    drv.close()


def test_brake_outside_a_hold_latches_and_one_inside_ends_with_the_hold():
    for how in ("brake", "estop_request"):
        # outside a hold: latches, survives a later runner hold, only release("estop") ends it
        rt, d = _bound(build())
        m = rt.m
        assert rt.submit(MastRequest("preset", 0.3, "up"))
        if how == "brake":
            rt.brake()
        else:
            assert rt.submit(MastRequest("estop", None, "operator"))
        assert rt.arb.braked and rt.arb.brake_latched and rt.state()["brake_latched"]
        rt.bus.values["hold"] = {"reason": "estop", "since": 0.0}
        rt.substep(m, d, 0.0, rt.bus)
        rt.bus.values["hold"] = None
        rt.substep(m, d, 0.002, rt.bus)
        assert rt.arb.braked, how
        rt.release("estop")
        rt.substep(m, d, 0.004, rt.bus)
        assert not rt.arb.braked and not rt.arb.brake_latched
        # inside a hold (WP6 EstopActions brakes on the latch edge): the same e-stop, ends with the hold
        rt.bus.values["hold"] = {"reason": "estop", "since": 0.0}
        if how == "brake":
            rt.brake()                                              # before the substep saw the hold
        else:
            rt.submit(MastRequest("estop", None, "operator"))
        rt.substep(m, d, 0.006, rt.bus)
        assert rt.arb.braked and not rt.arb.brake_latched
        rt.bus.values["hold"] = None
        rt.substep(m, d, 0.008, rt.bus)
        assert not rt.arb.braked, how
    # the pure arbiter: an explicit brake latches, a hold brake does not
    a = mc.MastArbiter()
    a.brake(latch=False)
    assert a.braked and not a.brake_latched
    a.brake()
    assert a.brake_latched
    a.release_brake()
    assert not a.braked and not a.brake_latched


def test_held_lowering_is_flagged_lower_blocked():
    m = _strike_world("wood")
    drv, rt = mast_driver(m)
    mid = int(m.body_mocapid[_id(m, O.mjOBJ_BODY, "strike_obstacle")])
    drv.stand(0.2)
    assert rt.submit(MastRequest("preset", 0.6, "mid", t=drv.t))
    drv.run_until(lambda dr: not rt.arb.moving, timeout_s=4.0)
    hc = drv.d.geom_xpos[rt.head_g].copy()
    drv.d.mocap_pos[mid][:] = [-0.30, hc[1] + 0.10, hc[2] - 0.08]     # beside and below the box corner
    drv.run_steps(60)
    assert rt.submit(MastRequest("recovery", 0.0, "link_loss", t=drv.t))
    drv.wait(hm.LOWER_STALL_S * 0.5)
    assert rt.arb.held_by == "near_obstacle" and rt.state()["lower_blocked"] is None
    drv.wait(hm.LOWER_STALL_S * 0.6)
    lb = rt.state()["lower_blocked"]
    assert lb is not None and lb["source"] == "recovery" and lb["geom"] == "tree_T_wood_0"
    assert lb["target_ext"] == 0.0 and lb["ext_m"] == pytest.approx(0.6)
    pubs = [json.loads(p) for _, topic, p in drv.published if topic == hm.STATE_TOPIC]
    assert pubs[-1]["lower_blocked"] is not None
    # the obstacle goes away: the retract resumes and the flag clears
    drv.d.mocap_pos[mid][:] = [-0.30, 1.0, 1.70]
    drv.run_until(lambda dr: rt.arb.ext <= 0.0, timeout_s=4.0)
    assert rt.arb.ext == 0.0 and rt.state()["lower_blocked"] is None
    drv.close()


def test_recovery_retract_completes_and_a_later_preset_raise_is_accepted():
    a = mc.MastArbiter(ext=0.5)
    assert a.submit(MastRequest("recovery", 0.0, "estop_recovery_retract", t=1.0))
    for _ in range(int(0.5 / 0.2 / DT) + 5):
        a.step(DT)
    assert a.ext == 0.0 and "recovery" not in a.requests and "estop_recovery_retract" in a.completed
    assert a.submit(MastRequest("preset", 0.6, "mid", t=2.0))        # was refused preempted:recovery
    # a recovery raise is a setpoint: it stays and still preempts presets
    a = mc.MastArbiter(ext=0.0)
    assert a.submit(MastRequest("recovery", 0.3, "safe_park", t=1.0))
    for _ in range(int(0.3 / 0.2 / DT) + 5):
        a.step(DT)
    assert a.ext == pytest.approx(0.3) and a.winner().source == "recovery"
    assert not a.submit(MastRequest("preset", 0.6, "mid", t=2.0))


def test_gimbal_clamp_is_flagged_and_low_band_is_stow():
    a = mc.MastArbiter()
    a.aim(0.0, math.radians(-90.0))                                   # WP35 registry allows -90; spec -60
    assert a.gimbal[1] == pytest.approx(mc.TILT_MIN_RAD) and a.state()["gimbal_clamped"]
    a.aim(0.0, math.radians(-30.0))
    assert not a.state()["gimbal_clamped"]
    a.aim(math.radians(270.0), 0.0)                                   # pan wraps, not a clamp
    assert not a.state()["gimbal_clamped"]
    assert hm.BANDS["low"] == hm.BANDS["stow"] == 0.0
