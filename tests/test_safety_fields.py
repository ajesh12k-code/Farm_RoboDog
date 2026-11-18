"""WP6 safety fields: hull-relative fields, crown scaling, orbit person pause, animal stop, WORK hand stop, map mask.

Acceptance (WP6; MVP-SPEC 5.3 tests; ORCHARD-MVP-SPEC 5.2, 6): person pauses the orbit; protective
latches; adult / child / crouching / lying / ladder / pole / animal actors stop the robot before a hull distance of
0.30 m from the front, the side and behind; a hand at 0.6 m in the WORK envelope stops the arm; a person in the
alley during WORK stops the arm; 0 false stops passing trunks at 0.45 m; interlock mutation test (orbit pause,
animal stop, hand-in-envelope stop, crown-scaled fields). Pure logic first, then the MuJoCo hook (headless, ROS-free).
"""
import contextlib
import math
import os

import mujoco
import numpy as np
import pytest

import atharv_nandi.sim.hooks as hook_registry
from atharv_nandi.core import farm_scene
from atharv_nandi.core import safety_fields as sf
from atharv_nandi.core import sim_driver as sd
from atharv_nandi.sim.world import actors as wa

WORLD_ENV = ("NANDI_WORLD", "NANDI_TERRAIN", "NANDI_SEED", "NANDI_CROPS", "NANDI_MAST_H", "NANDI_MAST_HEAD_KG",
             "NANDI_DT", "NANDI_PAYLOAD_KG", "NANDI_BUND_H", "NANDI_ACTORS")
HULL = sf.robot_hull()                         # trunk box only: (+-0.33, +-0.15)


# ====================================================================== pure geometry
def test_d_stop_and_walk_offsets():
    assert sf.d_stop(0.0) == 0.0
    assert sf.d_stop(0.3) == pytest.approx(0.03 + 0.09 / 1.2, abs=1e-12)
    o = sf.field_offsets(sf.MODE_WALK, 0.3)
    assert o["protective"] == pytest.approx({"front": 0.25 + sf.d_stop(0.3), "side": 0.25, "rear": 0.30})
    assert o["warning"] == {"front": 2.5, "side": 1.0, "rear": 1.5}
    w = sf.field_offsets(sf.MODE_WORK)
    assert w["protective"] == {"front": 0.3, "side": 0.3, "rear": 0.3} and w["warning"]["rear"] == 1.0
    with pytest.raises(ValueError):
        sf.field_offsets("RUN")


def test_fields_scale_with_the_crown():
    o = sf.field_offsets(sf.MODE_WORK, lens_z=2.7, ext=1.6)
    assert o["protective"]["side"] == pytest.approx(1.35) and o["warning"]["front"] == pytest.approx(2.7)
    assert sf.field_offsets(sf.MODE_WORK, lens_z=1.10, ext=0.0)["warning"]["front"] == 1.0   # stowed: unscaled
    walk = sf.field_offsets(sf.MODE_WALK, 0.15, lens_z=1.7, ext=0.6)
    assert walk["warning"]["front"] == 2.5 and walk["warning"]["side"] == pytest.approx(1.7)


def test_hull_includes_feet_arm_envelope_and_head():
    h = sf.robot_hull(feet_xy=[(0.3, 0.25), (-0.3, -0.25)], arm_xy=[(0.6, 0.0)], head_xy=(-0.3, 0.0))
    assert sf.hull_distance((0.69, 0.0), h) == 0.0                      # arm point + 0.1 m pad
    assert sf.hull_distance((0.75, 0.0), h) == pytest.approx(0.05, abs=0.01)
    assert sf.hull_distance((0.0, 0.0), h) == 0.0
    assert sf.hull_distance((0.0, 1.0), HULL) == pytest.approx(0.85)
    assert sf.sector((1, 0.2)) == "front" and sf.sector((-1, 0.2)) == "rear" and sf.sector((0, 1)) == "side"


def test_map_mask_classification():
    for n in ("floor", "orchard_ground", "tree_L1-1_wood_0", "tree_L1-1_foliage_2", "pool_3_fruit", "crate_wall_px",
              "dock_backboard", "prop_bamboo_1", "rope_B1_B2", "lateral_L1_q0_1", "patch_mud_1", "stake_7"):
        assert sf.classify_geom(n) is None, n
    assert sf.classify_geom("", "crop_item_1") is None
    assert sf.classify_geom("actor_person_w1__body_0") == sf.CLS_OBSTACLE
    assert sf.classify_geom("actor_person_w1__ladder_3") == sf.CLS_OBSTACLE
    assert sf.classify_geom("actor_animal_dog1__limb_2") == sf.CLS_ANIMAL
    assert sf.classify_geom("mystery_box") == sf.CLS_OBSTACLE                # unknown: an obstacle
    assert sf.classify_geom("", "") == sf.CLS_OBSTACLE


# ====================================================================== pure monitor
def R(x, y, cls=sf.CLS_OBSTACLE, z=0.5):
    return sf.Return(x, y, z, cls, "g", "test")


def test_protective_latches_and_clears_only_when_clear():
    fm = sf.FieldMonitor()
    st = fm.update(0.0, sf.MODE_WORK, [R(0.6, 0.0)], HULL)               # 0.27 m from the hull
    assert st.latched and st.latch_reason == "protective" and st.arm_stop and st.speed_scale == 0.0
    st = fm.update(0.1, sf.MODE_WORK, [], HULL)
    assert st.latched                                                    # latched: leaving does not clear
    assert fm.clear(0.3)[0] is False                                     # clear for < 0.5 s
    assert fm.clear(0.7)[0] is True and not fm.latched
    fm.update(0.8, sf.MODE_WORK, [R(0.6, 0.0)], HULL)
    fm.update(1.0, sf.MODE_WORK, [R(0.6, 0.0)], HULL)
    assert fm.clear(2.0)[0] is False                                     # still in the field


def test_warning_responses_by_mode():
    fm = sf.FieldMonitor()
    assert fm.update(0.0, sf.MODE_WALK, [R(2.0, 0.0)], HULL, v=0.2).speed_scale == sf.WARNING_SPEED_SCALE
    st = fm.update(0.1, sf.MODE_WORK, [R(0.0, 1.0)], HULL)               # person in the alley during WORK
    assert st.arm_stop and st.blade_block and not st.latched and st.beacon == "warning_flash"
    st = fm.update(0.2, sf.MODE_ORBIT, [R(0.0, 1.0)], HULL, arc=sf.OrbitArc([]))
    assert st.orbit_pause and not st.latched
    assert fm.update(0.3, sf.MODE_WALK, [R(4.0, 0.0)], HULL).speed_scale == 1.0


def test_animals_stop_from_the_warning_field_and_bumper():
    fm = sf.FieldMonitor()
    st = fm.update(0.0, sf.MODE_WALK, [R(1.5, 0.0, sf.CLS_ANIMAL)], HULL)
    assert st.latched and st.latch_reason == "animal"
    fm = sf.FieldMonitor()
    assert fm.update(0.0, sf.MODE_STOPPED, [], HULL, bumper=True).latch_reason == "bumper"


def test_orbit_arc_sweeps_the_hull():
    arc = sf.OrbitArc([(0.5 * math.sin(a), 1.0 - math.cos(a) * 1.0, a) for a in np.linspace(0, 1.0, 6)])
    p = (0.42, 0.25)
    assert arc.distance(p, HULL) < sf.hull_distance(p, HULL)
    fm = sf.FieldMonitor()
    st = fm.update(0.0, sf.MODE_ORBIT, [R(1.0, 1.2)], HULL, v=0.1, arc=arc)
    assert st.latched or st.orbit_pause


def test_field_polygon_contains_the_offset_hull():
    off = sf.field_offsets(sf.MODE_WORK)["warning"]
    poly = sf.field_polygon(HULL, off, n=36)
    assert len(poly) == 36
    for p in poly:
        assert 0.7 <= sf.hull_distance(p, HULL) <= 1.0 + 1e-3       # 1.0 on the faces, less at the corners


def test_bunch_exclusion():
    b = [(5.0, 0.0)]
    assert sf.bunch_exclusion((0.0, 0.0), b, True)
    dec = sf.bunch_exclusion((2.5, 0.0), b, True)
    assert not dec and dec.reason == "bunch_exclusion" and dec.values["dist_m"] == pytest.approx(2.5)
    assert sf.bunch_exclusion((4.9, 0.0), b, False)                      # no human cut flagged
    assert sf.bunch_exclusion((4.9, 0.0), [], True)


# ====================================================================== interlock mutation test
INTERLOCKS = {
    # name: (scenario -> True when the interlock acts, mutation that disables it)
    "orbit_person_pause": (lambda: sf.FieldMonitor().update(0, sf.MODE_ORBIT, [R(0.0, 1.0)], HULL,
                                                            arc=sf.OrbitArc([])).orbit_pause,
                           ("WALK_WARNING", {"front": 0.0, "side": 0.0, "rear": 0.0})),
    "animal_stop": (lambda: sf.FieldMonitor().update(0, sf.MODE_WALK, [R(1.5, 0, sf.classify_geom("actor_animal_dog_0"))],
                                                     HULL).latched,
                    ("animal_hits", lambda hits: [])),
    "hand_in_work_envelope": (lambda: sf.FieldMonitor().update(0, sf.MODE_WORK, [R(0.55, 0.0, z=0.6)],
                                                               HULL).latched,
                              ("WORK_PROTECTIVE", {"front": 0.0, "side": 0.0, "rear": 0.0})),
    "crown_scaled_fields": (lambda: sf.FieldMonitor().update(0, sf.MODE_WORK, [R(0.0, 1.2)], HULL, lens_z=2.7,
                                                             ext=1.6).latched,
                            ("CROWN_PROTECTIVE_K", 0.0)),
}


@pytest.mark.parametrize("name", sorted(INTERLOCKS))
def test_interlock_acts_and_its_mutation_is_caught(name, monkeypatch):
    scenario, (attr, value) = INTERLOCKS[name]
    assert scenario() is True                    # the interlock acts
    with monkeypatch.context() as mp:
        mp.setattr(sf, attr, value)
        assert scenario() is False               # mutated: the same scenario no longer acts -> this test catches it


# ====================================================================== MuJoCo: hook, actors, map mask
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


ACTOR_TYPES = ("adult", "child", "crouching", "lying", "ladder_picker", "pole_picker", "dog", "monkey", "cattle")
_TYPES = {"capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
          "box": mujoco.mjtGeom.mjGEOM_BOX, "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}


def _add_actor(spec, atype: str, aid: str):
    cls = wa.actor_class(atype)
    body = spec.worldbody.add_body(name=f"actor_{cls}_{aid}", pos=[-30.0, -30.0, 0.0], mocap=True)
    counts = {}
    for g in wa.geoms_for(atype):
        k = counts.get(g["role"], 0)
        counts[g["role"]] = k + 1
        size = list(g["size"]) + [0.0] * (3 - len(g["size"]))
        kw = dict(name=f"actor_{cls}_{aid}__{g['role']}_{k}", type=_TYPES[g["type"]], size=size, rgba=g["rgba"],
                  contype=0, conaffinity=0, group=1, mass=0.001)
        if "fromto" in g:
            kw["fromto"] = g["fromto"]
        else:
            kw["pos"] = g["pos"]
        body.add_geom(**kw)


def _hand(spec):
    """A forearm reaching in at arm height 0.6 m (MVP / OS 5.2 positive control)."""
    b = spec.worldbody.add_body(name="actor_person_hand", pos=[-30.0, -30.0, 0.0], mocap=True)
    b.add_geom(name="actor_person_hand__limb_0", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0.6, 0.6, 0, 0.6],
               size=[0.045, 0, 0], contype=0, conaffinity=0, group=1, mass=0.001)


SAFETY_BAND_TOP_M = 0.98                    # mast LiDAR ring (lens) height: the top of the safety-ray band
STAKE_XY = (2.0, -(0.25 + 0.45 + 0.03))      # 0.45 m between the hull side (feet at y +-0.25) and the stake
TRUNK_XY = (3.0, 0.25 + 0.45 + 0.08)


def _stake_and_trunk(spec):
    """A stake (MVP-SPEC 5.3) and a scaffold-free trunk (OS: passing trunks at 0.45 m), both static map geoms."""
    spec.worldbody.add_geom(name="stake_wp6", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.03, 0, 0],
                            fromto=[STAKE_XY[0], STAKE_XY[1], 0.0, STAKE_XY[0], STAKE_XY[1], 1.2], group=0)
    spec.worldbody.add_geom(name="tree_wp6_wood_0", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.08, 0, 0],
                            fromto=[TRUNK_XY[0], TRUNK_XY[1], 0.0, TRUNK_XY[0], TRUNK_XY[1], 1.5], group=0)


@pytest.fixture(scope="module")
def actor_world():
    with world_env(NANDI_WORLD="orchard_lite", NANDI_TERRAIN="flat", NANDI_ACTORS="0"):
        w = farm_scene.build_world()
        for a in ACTOR_TYPES:
            _add_actor(w.spec, a, f"t{a.replace('_', '')}")
        _hand(w.spec)
        _stake_and_trunk(w.spec)
        return w.spec.compile()


def _driver(m):
    reg = hook_registry.load_all(names=["mast", "stability"])
    drv = sd.SimDriver(m, hooks=reg)
    return drv


def _mocap(m, name):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    assert b >= 0, name
    return int(m.body_mocapid[b])


def _true_dist(m, d, body_name, hull, robot_xy, yaw):
    """Smallest planar hull distance of the actor's surfaces (person, animal and ladder geoms; an overhead pole is
    not part of it) below the mast LiDAR ring (parts above it, e.g. a ladder top leaning over the robot,
    are the WP24 swept-clearance / strike domain): capsule axis sampled every 2 cm minus its radius, ellipsoid / box centre minus its largest
    horizontal half size (conservative)."""
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    best = math.inf
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if m.geom_bodyid[g] != b or "__pole_" in name:
            continue
        c, R = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        s = m.geom_size[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            pts = [c + R[:, 2] * a for a in np.linspace(-s[1], s[1], max(2, int(2 * s[1] / 0.02) + 1))]
            r = float(s[0])
        else:
            pts, r = [c], float(max(s[0], s[1]) if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX else math.hypot(s[0], s[1]))
        for p in pts:
            if p[2] - r > SAFETY_BAND_TOP_M:
                continue
            q = sf.to_base_frame(np.asarray(p)[None, :], robot_xy, yaw)[0]
            best = min(best, sf.hull_distance(q[:2], hull) - r)
    return best


def _approach(drv, body, direction_deg, start=2.0, speed=0.5, timeout=6.0):
    """Move an actor toward the robot along the bearing from the hull centre; returns (latched, field, state)."""
    m, d = drv.m, drv.d
    mid = _mocap(m, body)
    xy = d.qpos[0:2].copy()
    u = np.array([math.cos(math.radians(direction_deg)), math.sin(math.radians(direction_deg))])
    r0 = 0.33 + start if abs(u[0]) > 0.7 else 0.25 + start
    face = math.atan2(-u[1], -u[0])
    d.mocap_quat[mid] = [math.cos(face / 2), 0, 0, math.sin(face / 2)]
    t0 = drv.t
    while drv.t - t0 < timeout:
        r = r0 - speed * (drv.t - t0)
        d.mocap_pos[mid] = [xy[0] + r * u[0], xy[1] + r * u[1], 0.0]
        drv.run_steps(10)
        f = drv.bus.values.get("safety_fields") or {}
        if f.get("latched"):
            return True, f, r
    return False, drv.bus.values.get("safety_fields"), None


@pytest.mark.parametrize("atype", ACTOR_TYPES)
def test_actor_stops_the_robot_before_0p30_from_front_side_and_rear(actor_world, atype):
    m = actor_world
    body = f"actor_{wa.actor_class(atype)}_t{atype.replace('_', '')}"
    for direction in (0.0, 90.0, 180.0, -90.0):
        drv = _driver(m)
        drv.wait(0.3)
        hull = sf.robot_hull()
        latched, f, _r = _approach(drv, body, direction)
        assert latched, (atype, direction, f)
        true_d = _true_dist(m, drv.d, body, hull, drv.d.qpos[0:2], 0.0)
        if wa.actor_class(atype) == "animal":
            assert f["latch_reason"] == "animal" and true_d > 0.30, (atype, direction, true_d)
        else:
            assert f["latch_reason"] == "protective", (atype, direction, f["latch_reason"])
            assert true_d >= 0.30 - 0.5 * 0.1 - 0.06, (atype, direction, true_d)   # one 10 Hz tick at 0.5 m/s
        drv.close()


def test_hand_at_arm_height_in_the_work_envelope_stops_the_arm(actor_world):
    m = actor_world
    drv = _driver(m)
    drv.bus.values["work"] = {"active": True}
    drv.wait(0.3)
    mid = _mocap(m, "actor_person_hand")
    xy = drv.d.qpos[0:2]
    drv.d.mocap_quat[mid] = [0, 0, 0, 1]                     # forearm points toward -x: the hand tip leads
    drv.d.mocap_pos[mid] = [xy[0] + 3.0, xy[1] + 0.05, 0.0]  # hand tip 2.4 m ahead: outside the 1.0 m WORK warning
    drv.wait(0.3)
    assert not drv.bus.values["safety_fields"]["arm_stop"]
    drv.d.mocap_pos[mid] = [xy[0] + 1.6, xy[1] + 0.05, 0.0]  # tip 1.0 m ahead: 0.67 m from the hull (warning)
    drv.wait(0.3)
    f = drv.bus.values["safety_fields"]
    assert f["mode"] == "WORK" and f["arm_stop"] and f["blade_block"], f
    drv.d.mocap_pos[mid] = [xy[0] + 1.15, xy[1] + 0.05, 0.0]
    drv.wait(0.3)
    assert drv.bus.values["safety_fields"]["latched"]
    drv.close()


def test_person_in_the_alley_during_work_stops_the_arm(actor_world):
    m = actor_world
    drv = _driver(m)
    drv.bus.values["work"] = {"active": True}
    drv.wait(0.3)
    mid = _mocap(m, "actor_person_tadult")
    xy = drv.d.qpos[0:2]
    drv.d.mocap_pos[mid] = [xy[0], xy[1] + 0.25 + 0.9, 0.0]  # 0.9 m beside the hull, in the alley
    drv.wait(0.3)
    f = drv.bus.values["safety_fields"]
    assert f["arm_stop"] and not f["latched"], f
    assert drv.bus.values["work_field"]["person"] and not drv.bus.values["work_field"]["animal"]   # WP2 interlock input
    drv.close()


def test_orbit_person_pause_with_crown_scaled_fields(actor_world):
    m = actor_world
    drv = _driver(m)
    drv.wait(0.2)
    mrt = drv.bus.values["mast_control"]
    mrt.arb.reset(1.6)                                       # crown forced up by the harness (lens 2.7 m)
    drv.bus.values["orbit"] = {"active": True, "arc": []}
    drv.wait(0.3)
    mid = _mocap(m, "actor_person_tadult")
    xy = drv.d.qpos[0:2]
    drv.d.mocap_pos[mid] = [xy[0], xy[1] + 0.25 + 2.3, 0.0]  # 2.1-2.3 m off the hull: inside the 2.7 m warning
    drv.wait(0.3)
    f = drv.bus.values["safety_fields"]
    assert f["mode"] == "ORBIT" and f["orbit_pause"] and not f["latched"], f
    assert f["offsets"]["warning"]["side"] == pytest.approx(f["lens_z"], abs=1e-3)
    drv.d.mocap_pos[mid] = [xy[0], xy[1] + 0.25 + 1.3, 0.0]  # inside 0.5 x 2.7 = 1.35 m: protective
    drv.wait(0.3)
    f = drv.bus.values["safety_fields"]
    assert f["latched"] and f["latch_reason"] == "protective"
    drv.d.mocap_pos[mid] = [-30.0, -30.0, 0.0]
    drv.wait(1.0)
    assert drv.bus.values["safety_fields"]["latched"]          # protective latches: leaving does not clear it
    drv.enter_hold("protective")
    drv.wait(0.2)
    drv.leave_hold()                                           # operator re-arm with the field clear -> cleared
    drv.wait(0.2)
    assert not drv.bus.values["safety_fields"]["latched"]
    drv.close()


@pytest.mark.parametrize("world", ["legacy", "orchard_lite", "orchard_demo", "orchard_demo_monsoon"])
def test_every_static_world_geom_is_map_masked(world):
    with world_env(NANDI_WORLD=world, NANDI_TERRAIN="rough" if world != "legacy" else "ridges", NANDI_ACTORS="0"):
        m = farm_scene.build_world().spec.compile()
    from atharv_nandi.sim.hooks import stability as hs
    rt = hs.StabilityRuntime()
    rt.bind(m, mujoco.MjData(m))
    assert rt.ok and rt.unmasked_static == []


def test_zero_false_stops_passing_a_stake_and_a_trunk_at_0p45(actor_world):
    """Walk 3 m past a stake and a trunk, each 0.45 m from the hull side, no person present: 0 latches and 0 field
    returns. Both are static map geoms (masked; world-body geoms are left out of the ray test, RAY_BODY_EXCLUDE)."""
    m = actor_world
    drv = _driver(m)
    drv.d.qpos[0:2] = [1.0, 0.0]
    mujoco.mj_forward(m, drv.d)
    from atharv_nandi.sim.world.modules import tools as tools_mod
    tools_mod.seat_tools(m, drv.d)                 # WP2: racked tools follow a teleported robot
    mujoco.mj_forward(m, drv.d)
    drv.wait(0.5)
    from atharv_nandi.sim.hooks import stability as hs
    rt = hs._RT[0]
    seen = set()
    orig = rt._rays

    def spy(P, V, sizes):
        gid, dist = orig(P, V, sizes)
        for g in set(int(x) for x in gid if x >= 0):
            seen.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
        return gid, dist

    rt._rays = spy
    drv.walk(0.8)
    latched, warn = False, 0
    t0 = drv.t
    while drv.t - t0 < 25.0 and drv.d.qpos[0] < TRUNK_XY[0] + 0.8:
        drv.wait(0.1)
        f = drv.bus.values["safety_fields"]
        latched = latched or f["latched"]
        warn += f["n_warning"] + f["n_protective"]
    assert drv.d.qpos[0] > TRUNK_XY[0] + 0.5, f"the robot did not pass the trunk: x {drv.d.qpos[0]:.2f}"
    assert abs(drv.d.qpos[1]) < 0.15, "the robot drifted toward the stake or the trunk"
    assert not ({"stake_wp6", "tree_wp6_wood_0"} & seen)                      # never an obstacle
    assert rt.geom_cls[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "stake_wp6")] is None
    assert rt.geom_cls[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "tree_wp6_wood_0")] is None
    assert not latched and warn == 0
    drv.close()


def test_vectorised_hull_distances_match_the_scalar_form():
    rng = np.random.default_rng(3)
    hull = sf.robot_hull(feet_xy=[(0.3, 0.25), (-0.3, -0.25), (0.0, 0.31)], arm_xy=[(0.55, -0.1)])
    P = rng.uniform(-3.0, 3.0, size=(400, 2))
    v = sf.hull_distances(P, hull)
    s = np.array([sf.hull_distance(p, hull) for p in P])
    assert np.allclose(v, s, atol=1e-12)
    arc = sf.OrbitArc([(0.2, 0.1, 0.3), (0.4, 0.3, 0.6)])
    assert np.allclose(arc.distances(P, hull), [arc.distance(p, hull) for p in P], atol=1e-12)
