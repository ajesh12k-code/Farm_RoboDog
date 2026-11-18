"""WP1 robot joint-less additions (WP1; ORCHARD-MVP-SPEC 2.3-2.6; MVP-SPEC 2.4-2.5).

Acceptance (WP1): compiles; nv unchanged by the mast; standing 2 s, no fall; mast <= 2.4 kg,
head <= 0.6 kg, moving <= 0.8 kg; no mast section below 0.40 m or above the outer tube top when stowed;
subtree_mass[trunk] = baseline + additions +-2 %; ring lens z 0.98 +- 0.01; no robot geom in groups 0/1;
each test <= 10 s. Plus the beacon / handheld e-stop logic of atharv_nandi/sim/hooks/beacon.py and the WP0c change request
(rail end, rear pillar). Headless MuJoCo, ROS-free.
"""
import json
import math
import os
import sys

import mujoco
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PHYS = os.path.join(REPO, "tools", "physics")

from tools.physics import measure_baseline as mb  # noqa: E402
import atharv_nandi.sim.hooks as hook_registry  # noqa: E402
from atharv_nandi.core import farm_scene  # noqa: E402
from atharv_nandi.core import model_layout as mlay  # noqa: E402
from atharv_nandi.core import sim_state  # noqa: E402
from atharv_nandi.sim.hooks import beacon as bc  # noqa: E402

XML = farm_scene.ROBOT_XML
V1_XML = os.path.join(os.path.dirname(XML), "atharv_nandi_v1.xml")
CANDIDATES = os.path.join(REPO, "config", "fields", "orchard_candidates.json")

O = mujoco.mjtObj
SECTIONS = ("mast_s1", "mast_s2", "mast_s3")
STROKE_M = 1.60
LENS_STOWED_M = 1.10
RING_LENS_M = 0.98
SPEC_OVERLAP_M = 0.067          # OS 2.3: "fully up, overlap 0.067 m per joint"

# The WP1 additions with their (ASSUMED) masses, kg. the WP1 contract notes carries the same table.
ADDED_GEOMS_KG = {
    # crown mast system: fixed 1.495 + beacon 0.03 + moving 0.80 = 2.325
    "mast_drive": 0.40, "mast_outer": 0.35, "mast_ring": 0.745, "beacon": 0.03,
    "mast_s1_g": 0.15, "mast_s2_g": 0.13, "mast_s3_g": 0.12,
    "crown_head_g": 0.36, "anemometer_g": 0.02, "gnss_puck": 0.02,
    # rack 0.12
    "rack_foot_f": 0.005, "rack_foot_r": 0.005, "rack_post_f": 0.015, "rack_post_r": 0.015, "rack_rail": 0.05,
    "rack_cradle_0": 0.01, "rack_cradle_1": 0.01, "rack_cradle_2": 0.01,
    # bumper, belly rests, lift marks, e-stop heads, handheld receiver antenna
    "bumper": 0.03, "belly_rest_fl": 0.01, "belly_rest_fr": 0.01, "belly_rest_rl": 0.01, "belly_rest_rr": 0.01,
    "lift_mark_fl": 0.002, "lift_mark_fr": 0.002, "lift_mark_rl": 0.002, "lift_mark_rr": 0.002,
    "estop_l": 0.008, "estop_r": 0.008, "handheld_rx_antenna": 0.005,
    # desap tray, divider, papaya cradle 0.040
    "basket_divider": 0.008, "tray_lane_wall_l": 0.006, "tray_lane_wall_r": 0.006,
    "tray_div_l0": 0.002, "tray_div_l1": 0.002, "tray_div_l2": 0.002,
    "tray_div_r0": 0.002, "tray_div_r1": 0.002, "tray_div_r2": 0.002,
    "papaya_cradle_l": 0.004, "papaya_cradle_r": 0.004,
}
BALLAST_KG = 0.001
ADDED_KG = sum(ADDED_GEOMS_KG.values()) + BALLAST_KG
NONCOLLIDING = [n for n in ADDED_GEOMS_KG if not n.startswith(("bumper", "belly_rest_"))]
COLLIDING = ["bumper", "belly_rest_fl", "belly_rest_fr", "belly_rest_rl", "belly_rest_rr"]


def _id(m, kind, name):
    i = mujoco.mj_name2id(m, kind, name)
    assert i >= 0, name
    return i


def _home(m):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return d


def _robot_bodies(m):
    trunk = _id(m, O.mjOBJ_BODY, "trunk")
    out = set()
    for b in range(m.nbody):
        p = b
        while p > 0 and p != trunk:
            p = int(m.body_parentid[p])
        if p == trunk:
            out.add(b)
    return out


def _set_ext(m, ext):
    for s in SECTIONS:
        m.body_pos[_id(m, O.mjOBJ_BODY, s)] = [0.0, 0.0, ext / 3.0]


def _zspan(m, d, geom):
    """World z extent of a vertical capsule / cylinder geom."""
    g = _id(m, O.mjOBJ_GEOM, geom)
    zc = float(d.geom_xpos[g][2])
    r, half = float(m.geom_size[g][0]), float(m.geom_size[g][1])
    cap = r if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
    return zc - half - cap, zc + half + cap


@pytest.fixture(scope="module")
def robot():
    return mujoco.MjModel.from_xml_path(XML)


@pytest.fixture(scope="module")
def v1():
    return mujoco.MjModel.from_xml_path(V1_XML)


@pytest.fixture(scope="module")
def flat():
    return mb.build(NANDI_TERRAIN="flat", NANDI_CROPS=1)


# ------------------------------------------------------------------------------------- compile, dof, layout
def test_compiles_with_the_frozen_layout_and_no_new_dof(robot, v1):
    assert (robot.nq, robot.nv, robot.nu, robot.njnt) == (v1.nq, v1.nv, v1.nu, v1.njnt)
    lay = mlay.Layout.from_model(robot)
    assert mlay.frozen_violations(lay) == []
    # every WP1 body is joint-free (MVP-SPEC rule 1: joint-less additions inside trunk)
    for b in ("mast_base", *SECTIONS, "crown_head", "ballast"):
        bid = _id(robot, O.mjOBJ_BODY, b)
        assert robot.body_jntnum[bid] == 0 and robot.body_dofnum[bid] == 0, b
    assert robot.body_parentid[_id(robot, O.mjOBJ_BODY, "mast_base")] == _id(robot, O.mjOBJ_BODY, "trunk")


@pytest.mark.parametrize("terrain", ["flat", "orchard"])
def test_composed_world_layout_unchanged(monkeypatch, terrain):
    monkeypatch.setenv("NANDI_TERRAIN", terrain)
    m = farm_scene.load_farm_model()
    assert mlay.frozen_violations(mlay.Layout.from_model(m)) == []
    assert _id(m, O.mjOBJ_BODY, "crown_head") > 0 and _id(m, O.mjOBJ_CAMERA, "crown_cam") >= 0


def test_mast_extension_changes_no_dof_and_moves_the_lens(robot):
    m = mujoco.MjModel.from_xml_path(XML)
    nv0 = m.nv
    lens = {}
    for ext in (0.0, 0.6, STROKE_M):
        _set_ext(m, ext)
        d = _home(m)
        assert m.nv == nv0 and m.nq == robot.nq
        lens[ext] = float(d.cam_xpos[_id(m, O.mjOBJ_CAMERA, "crown_cam")][2])
        assert float(d.site_xpos[_id(m, O.mjOBJ_SITE, "crown_ref")][2]) == pytest.approx(lens[ext], abs=1e-9)
    assert lens[0.0] == pytest.approx(LENS_STOWED_M, abs=1e-6)       # OS 2.3: 1.10 m stowed
    assert lens[0.6] == pytest.approx(1.70, abs=1e-6)                  # mid band lens
    assert lens[STROKE_M] == pytest.approx(2.70, abs=1e-6)             # high band lens


def test_sections_overlap_when_fully_up(robot):
    m = mujoco.MjModel.from_xml_path(XML)
    _set_ext(m, STROKE_M)
    d = _home(m)
    tube_top = _zspan(m, d, "mast_outer")[1]

    def cyl(geom):   # cylinder part of a capsule (the sliding length)
        g = _id(m, O.mjOBJ_GEOM, geom)
        zc, half = float(d.geom_xpos[g][2]), float(m.geom_size[g][1])
        return zc - half, zc + half

    s = [cyl(f"{n}_g") for n in SECTIONS]
    overlaps = [tube_top - s[0][0], s[0][1] - s[1][0], s[1][1] - s[2][0]]
    print(f"WP1 overlaps fully up (m): tube/s1 {overlaps[0]:.4f}, s1/s2 {overlaps[1]:.4f}, s2/s3 {overlaps[2]:.4f}")
    assert all(o >= SPEC_OVERLAP_M - 5e-4 for o in overlaps)


# ------------------------------------------------------------------------------------- mass
def test_mast_head_and_moving_mass_caps(robot):
    sub = lambda b: float(robot.body_subtreemass[_id(robot, O.mjOBJ_BODY, b)])  # noqa: E731
    mast, moving, head = sub("mast_base"), sub("mast_s1"), sub("crown_head")
    print(f"WP1 mast {mast:.3f} kg, moving {moving:.3f} kg, head {head:.3f} kg")
    assert mast <= 2.4 + 1e-9
    assert head <= 0.6 + 1e-9
    assert moving <= 0.8 + 1e-9


def test_trunk_subtree_mass_is_baseline_plus_additions(robot, v1):
    base = float(v1.body_subtreemass[_id(v1, O.mjOBJ_BODY, "trunk")])
    total = float(robot.body_subtreemass[_id(robot, O.mjOBJ_BODY, "trunk")])
    spec = mujoco.MjSpec.from_file(XML)   # MjModel keeps no per-geom mass; the spec does
    declared = {n: float(spec.geom(n).mass) for n in ADDED_GEOMS_KG}
    assert declared == pytest.approx(ADDED_GEOMS_KG, abs=1e-9)
    assert float(robot.body_mass[_id(robot, O.mjOBJ_BODY, "ballast")]) == pytest.approx(BALLAST_KG)
    print(f"WP1 trunk subtree {total:.3f} kg = baseline {base:.3f} + additions {ADDED_KG:.3f} "
          f"(diff {total - base - ADDED_KG:+.2e})")
    assert total == pytest.approx(base + ADDED_KG, rel=0.02)
    assert total == pytest.approx(base + ADDED_KG, abs=1e-6)   # nothing else changed mass
    # ORCHARD-MVP-SPEC 2.6: mast <= 2.4 + rack/beacon/bumper/tray 0.2 = 2.6 kg added to the trunk subtree
    assert total - base <= 2.6 + 1e-9


def test_home_com_stays_over_the_foot_centroid(robot, v1):
    """Physics alignment v1 rule |CoM x - foot centroid x| <= 0.010 m, kept by moving the battery to x 0.11."""
    d = _home(robot)
    feet = np.array([d.geom_xpos[_id(robot, O.mjOBJ_GEOM, f"{n}_foot")] for n in ("FR", "FL", "MR", "ML", "RR", "RL")])
    com = d.subtree_com[_id(robot, O.mjOBJ_BODY, "trunk")]
    dv = _home(v1)
    com_v1 = dv.subtree_com[_id(v1, O.mjOBJ_BODY, "trunk")]
    print(f"WP1 home CoM x {com[0]:+.4f} (v1 {com_v1[0]:+.4f}), y {com[1]:+.4f}, z {com[2]:.4f} (v1 {com_v1[2]:.4f}); "
          f"foot centroid x {feet[:, 0].mean():+.4f}")
    assert abs(com[0] - feet[:, 0].mean()) <= 0.010
    assert abs(com[1]) <= 0.005


# ------------------------------------------------------------------------------------- stowed geometry
def test_stowed_sections_inside_the_outer_tube(robot):
    d = _home(robot)   # keyframe: trunk at 0.34 on flat ground (z = height above ground)
    tube_lo, tube_top = _zspan(robot, d, "mast_outer")
    print(f"WP1 outer tube world z {tube_lo:.3f}-{tube_top:.3f}")
    for n in SECTIONS:
        lo, hi = _zspan(robot, d, f"{n}_g")
        print(f"WP1 stowed {n}: {lo:.3f}-{hi:.3f} m")
        assert lo >= 0.40, n
        assert hi <= tube_top + 1e-9, n
        g = _id(robot, O.mjOBJ_GEOM, f"{n}_g")
        assert np.allclose(d.geom_xpos[g][:2], d.geom_xpos[_id(robot, O.mjOBJ_GEOM, "mast_outer")][:2])
    radii = [float(robot.geom_size[_id(robot, O.mjOBJ_GEOM, g)][0]) for g in ("mast_outer", "mast_s1_g", "mast_s2_g",
                                                                               "mast_s3_g")]
    assert radii == sorted(radii, reverse=True) and len(set(radii)) == 4   # nested: outer > s1 > s2 > s3
    head = _id(robot, O.mjOBJ_GEOM, "crown_head_g")
    assert float(d.geom_xpos[head][2]) - float(robot.geom_size[head][2]) >= tube_top - 1e-9   # head on the tube


def test_mast_mount_position(robot):
    b = _id(robot, O.mjOBJ_BODY, "mast_base")
    assert np.allclose(robot.body_pos[b], [-0.30, 0.0, 0.05])
    d = _home(robot)
    assert float(d.xpos[b][2]) == pytest.approx(0.39)


def test_ring_cameras_at_lens_098(robot, flat):
    for m, settle in ((robot, 0.0), (flat, 2.0)):
        d = _home(m)
        d.ctrl[:] = m.key_ctrl[0]
        for _ in range(int(settle / m.opt.timestep)):
            mujoco.mj_step(m, d)
        for i, yaw in enumerate((0, 90, 180, 270)):
            c = _id(m, O.mjOBJ_CAMERA, f"ring_{i}")
            z = float(d.cam_xpos[c][2])
            look = -d.cam_xmat[c].reshape(3, 3)[:, 2]          # MuJoCo cameras look along -z
            tilt = math.degrees(math.asin(-look[2]))
            az = math.degrees(math.atan2(look[1], look[0])) % 360.0
            print(f"WP1 ring_{i} (settle {settle:.0f} s): lens z {z:.4f}, down {tilt:.1f} deg, az {az:.1f}")
            assert z == pytest.approx(RING_LENS_M, abs=0.01)
            assert tilt == pytest.approx(25.0, abs=1.0)
            assert min(abs(az - yaw), 360 - abs(az - yaw)) <= 1.0
            assert float(m.cam_fovy[c]) == pytest.approx(83.6)
            assert m.cam_bodyid[c] == _id(m, O.mjOBJ_BODY, "mast_base")
    lidar = _id(robot, O.mjOBJ_SITE, "mast_lidar")
    assert float(_home(robot).site_xpos[lidar][2]) == pytest.approx(RING_LENS_M, abs=0.01)


def test_crown_head_carries_camera_and_sites(robot):
    head = _id(robot, O.mjOBJ_BODY, "crown_head")
    assert robot.body_parentid[head] == _id(robot, O.mjOBJ_BODY, "mast_s3")
    assert robot.cam_bodyid[_id(robot, O.mjOBJ_CAMERA, "crown_cam")] == head
    assert float(robot.cam_fovy[_id(robot, O.mjOBJ_CAMERA, "crown_cam")]) == pytest.approx(60.0)
    for s in ("crown_ref", "anemometer", "gnss_antenna"):
        assert robot.site_bodyid[_id(robot, O.mjOBJ_SITE, s)] == head, s
    g = _id(robot, O.mjOBJ_GEOM, "crown_head_g")
    assert np.allclose(robot.geom_size[g], [0.06, 0.06, 0.06])
    trunk = _id(robot, O.mjOBJ_BODY, "trunk")
    for s in ("imu", "estop", "handheld_estop_rx", "bumper_touch", "cradle_T2", "cradle_T4",
              "cradle_T5", "lift_point_fl", "lift_point_fr", "lift_point_rl", "lift_point_rr"):
        assert robot.site_bodyid[_id(robot, O.mjOBJ_SITE, s)] == trunk, s
    assert robot.site_bodyid[_id(robot, O.mjOBJ_SITE, "mast_top")] == _id(robot, O.mjOBJ_BODY, "mast_base")
    assert robot.site_bodyid[_id(robot, O.mjOBJ_SITE, "tcp_range")] == _id(robot, O.mjOBJ_BODY, "arm_wrist")
    for s in ("bumper", "gnss_pos", "imu_quat", "imu_gyro", "imu_acc"):
        assert _id(robot, O.mjOBJ_SENSOR, s) >= 0


# ------------------------------------------------------------------------------------- groups, contacts, rays
@pytest.mark.parametrize("terrain", ["flat", "rough", "orchard"])
def test_no_robot_geom_in_groups_0_or_1(monkeypatch, terrain):
    monkeypatch.setenv("NANDI_TERRAIN", terrain)
    m = farm_scene.load_farm_model()
    rob = _robot_bodies(m)
    groups = {int(m.geom_group[g]) for g in range(m.ngeom) if m.geom_bodyid[g] in rob}
    assert groups and not groups & {0, 1}, groups
    # ORCHARD-MVP-SPEC 2.5: 3 foliage (coverage rays), 4 decals, 5 query-only world geoms (crown strike, alley
    # clearance): no robot geom there, the hidden mass-only ones included (group 2, alpha 0)
    assert groups == {2}, groups
    visible = [g for g in range(m.ngeom) if m.geom_bodyid[g] in rob and m.geom_contype[g]]
    assert all(m.geom_group[g] == 2 for g in visible)   # every colliding robot geom is drawn, in group 2
    for n in ("battery", "electronics", "FR_hipabd_motor", "RL_knee_thigh_motors"):
        g = _id(m, O.mjOBJ_GEOM, n)
        assert m.geom_rgba[g][3] == 0 and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0, n
    hidden = {g for g in range(m.ngeom) if m.geom_bodyid[g] in rob and m.geom_rgba[g][3] == 0}
    assert len(hidden) == 14
    # a group-5 query (foliage envelope, rope lines) never meets the robot, and the renderer skips alpha 0
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    scn = mujoco.MjvScene(m, maxgeom=20000)
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scn)
    drawn = {scn.geoms[i].objid for i in range(scn.ngeom) if scn.geoms[i].objtype == O.mjOBJ_GEOM}
    assert not hidden & drawn
    assert m.geom_group[_id(m, O.mjOBJ_GEOM, "floor")] == 0


@pytest.mark.parametrize("world", ["orchard_lite", "orchard_demo"])
def test_orchard_worlds_keep_the_wp1_acceptance(monkeypatch, world):
    """The composed orchard MVP worlds (trees, fruit pool, tools, actors, dock): layout frozen, the mast adds
    no DOF, robot geoms only in group 2, and the robot stands 2 s with the crown fully up while no WP1
    non-colliding part ever makes a contact."""
    # monkeypatch, not mb.build(NANDI_WORLD=...): scene_env does not restore NANDI_WORLD, so it would leak
    monkeypatch.setenv("NANDI_WORLD", world)
    m = farm_scene.load_farm_model()
    assert mlay.frozen_violations(mlay.Layout.from_model(m)) == []
    rob = _robot_bodies(m)
    assert {int(m.geom_group[g]) for g in range(m.ngeom) if m.geom_bodyid[g] in rob} == {2}
    nv0, nq0 = m.nv, m.nq
    _set_ext(m, STROKE_M)
    assert (m.nv, m.nq) == (nv0, nq0)
    d = _home(m)
    assert float(d.cam_xpos[_id(m, O.mjOBJ_CAMERA, "crown_cam")][2]) == pytest.approx(2.70, abs=1e-6)
    s = mb.stand(seconds=2.0, settle_s=2.0, model=m)
    assert not s["fell"] and s["n_feet"] == 6
    nc = {_id(m, O.mjOBJ_GEOM, n) for n in NONCOLLIDING}
    d = _home(m)
    hits = 0
    for _ in range(500):
        mujoco.mj_step(m, d)
        hits += sum(1 for c in d.contact[:d.ncon] if c.geom1 in nc or c.geom2 in nc)
    print(f"WP1 {world}: nq {m.nq} nv {m.nv} nu {m.nu}; stand at ext 1.6 fell {s['fell']} feet {s['n_feet']} "
          f"trunk speed rms {s['trunk_speed_rms']:.5f}; WP1 non-colliding contacts over 1 s: {hits}")
    assert hits == 0


def test_mast_head_and_rack_never_collide(robot):
    for n in NONCOLLIDING:
        g = _id(robot, O.mjOBJ_GEOM, n)
        if n.startswith(("tray_", "basket_divider", "papaya_cradle")):
            assert robot.geom_contype[g] == 0 and robot.geom_conaffinity[g] == 4, n   # tray bit only
        else:
            assert robot.geom_contype[g] == 0 and robot.geom_conaffinity[g] == 0, n
    for n in COLLIDING:
        g = _id(robot, O.mjOBJ_GEOM, n)
        assert robot.geom_contype[g] == 1 and robot.geom_conaffinity[g] == 1, n


def test_tray_cells_ignore_legacy_crops_and_arm_but_hold_tray_bit_fruit(robot):
    def collide(a, b):
        return bool((robot.geom_contype[a] & robot.geom_conaffinity[b]) or (robot.geom_contype[b] & robot.geom_conaffinity[a]))
    crop = [g for g in range(robot.ngeom) if robot.geom_bodyid[g] == _id(robot, O.mjOBJ_BODY, "crop_item_1")][0]
    palm = _id(robot, O.mjOBJ_GEOM, "gripper_palm")
    for n in ("tray_div_l0", "basket_divider", "papaya_cradle_l"):
        g = _id(robot, O.mjOBJ_GEOM, n)
        assert not collide(g, crop) and not collide(g, palm), n
    # a fruit with contype bit 4 (set by WP3 when it places it) is held by the cell walls
    assert robot.geom_conaffinity[_id(robot, O.mjOBJ_GEOM, "tray_div_l0")] & 4


def test_safety_and_coverage_rays_pass_through_the_robot(robot):
    d = _home(robot)
    rob = _robot_bodies(robot)
    gid = np.array([-1], dtype=np.int32)
    lidar = d.site_xpos[_id(robot, O.mjOBJ_SITE, "mast_lidar")].copy()
    for mask in ([1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0]):     # safety {0, 1}, coverage {0, 1, 3}
        groups = np.array(mask, dtype=np.uint8)
        hits = []
        for az in np.radians(np.arange(0.0, 360.0, 15.0)):
            for el in (-15.0, -30.0, -60.0, -85.0):
                v = np.array([math.cos(az) * math.cos(math.radians(el)), math.sin(az) * math.cos(math.radians(el)),
                              math.sin(math.radians(el))])
                dist = mujoco.mj_ray(robot, d, lidar, v, groups, 1, -1, gid)
                hits.append((dist, int(gid[0])))
        robot_hits = [g for _, g in hits if g >= 0 and robot.geom_bodyid[g] in rob]
        assert robot_hits == [], [mujoco.mj_id2name(robot, O.mjOBJ_GEOM, g) for g in robot_hits]
        # the -30 deg beam meets the ground at about 1.7 m (MVP-SPEC 2.4) unless the legacy crop is in the way
        floor = _id(robot, O.mjOBJ_GEOM, "floor")
        assert any(g == floor for _, g in hits)
    # horizontal ray straight through the trunk at belly height: only the far floor-less void, no robot hit
    p = np.array([1.0, 0.0, 0.34])
    dist = mujoco.mj_ray(robot, d, p, np.array([-1.0, 0.0, 0.0]), np.array([1, 1, 0, 1, 0, 0], dtype=np.uint8), 1, -1,
                         gid)
    assert gid[0] < 0 or robot.geom_bodyid[gid[0]] not in rob


def test_bumper_touch_reads_a_contact():
    spec = mujoco.MjSpec.from_file(XML)
    spec.worldbody.add_geom(name="wp1_test_wall", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.362 + 0.01 - 0.002, 0, 0.32],
                            size=[0.01, 0.3, 0.02])
    m = spec.compile()
    d = _home(m)
    s = _id(m, O.mjOBJ_SENSOR, "bumper")
    print(f"WP1 bumper touch {float(d.sensordata[m.sensor_adr[s]]):.2f} N")
    assert float(d.sensordata[m.sensor_adr[s]]) > 0.0
    d0 = _home(mujoco.MjModel.from_xml_path(XML))
    assert float(d0.sensordata[m.sensor_adr[s]]) == 0.0


def test_belly_rests_are_the_lowest_trunk_geoms_and_clear_at_stand(robot):
    d = _home(robot)
    trunk = _id(robot, O.mjOBJ_BODY, "trunk")
    bottoms = {}
    for g in range(robot.ngeom):
        if robot.geom_bodyid[g] == trunk and robot.geom_contype[g]:
            bottoms[mujoco.mj_id2name(robot, O.mjOBJ_GEOM, g) or str(g)] = float(d.geom_xpos[g][2] - robot.geom_size[g][2]) \
                if robot.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else None
    pads = [bottoms[f"belly_rest_{k}"] for k in ("fl", "fr", "rl", "rr")]
    assert max(pads) < bottoms["trunk_geom"]
    assert min(pads) == pytest.approx(0.278, abs=1e-6)        # 12 mm below the trunk box at the home pose
    xs = [float(robot.geom_pos[_id(robot, O.mjOBJ_GEOM, f"belly_rest_{k}")][0]) for k in ("fl", "fr", "rl", "rr")]
    assert min(xs) < 0 < max(xs)


def test_rack_cradles_match_the_wp0c_frozen_rack(robot):
    with open(CANDIDATES) as f:
        rack = json.load(f)["frozen"]["rack"]
    assert rack["ok"]
    by_tool = {c["tool"]: c["pos"] for c in rack["cradles"]}
    assert sorted(by_tool) == ["T2", "T4", "T5"]
    for tool, pos in by_tool.items():
        assert np.allclose(robot.site_pos[_id(robot, O.mjOBJ_SITE, f"cradle_{tool}")], pos), tool
    # cradle geoms: the same boxes the WP0c sweep cleared (pos - [0, 0.01, 0.02], half 0.035 x 0.012 x 0.02)
    for i, tool in enumerate(("T2", "T4", "T5")):
        g = _id(robot, O.mjOBJ_GEOM, f"rack_cradle_{i}")
        assert np.allclose(robot.geom_pos[g], np.array(by_tool[tool]) - [0, 0.01, 0.02])
        assert np.allclose(robot.geom_size[g], [0.035, 0.012, 0.02])


def test_wp0c_change_request_rail_and_pillar_clear_the_mast(robot):
    """feasibility-report 5.1: the mast at x -0.30 overlapped the rail end and the rear pillar."""
    gx = lambda n: (float(robot.geom_pos[_id(robot, O.mjOBJ_GEOM, n)][0]),   # noqa: E731
                    float(robot.geom_size[_id(robot, O.mjOBJ_GEOM, n)][0]))
    tube_front = -0.30 + float(robot.geom_size[_id(robot, O.mjOBJ_GEOM, "mast_outer")][0])
    rail_c, rail_h = gx("gantry_rail")
    pil_c, pil_h = gx("gantry_pillar_rear")
    print(f"WP1 rail rear end {rail_c - rail_h:+.3f}, rear pillar {pil_c - pil_h:+.3f}..{pil_c + pil_h:+.3f}, "
          f"mast tube front edge {tube_front:+.3f}")
    assert rail_c - rail_h - tube_front >= 0.010
    assert pil_c - pil_h - tube_front >= 0.010
    assert rail_c + rail_h == pytest.approx(0.326)                       # front end unchanged
    # the carriage (centre 0.05 + gantry, half 0.042) is still on the rail at the gantry limit -0.26
    assert 0.05 - 0.26 - 0.042 >= rail_c - rail_h
    # masses unchanged
    spec = mujoco.MjSpec.from_file(XML)
    assert spec.geom("gantry_rail").mass == pytest.approx(0.45)
    assert spec.geom("gantry_pillar_rear").mass == pytest.approx(0.1)


def test_tray_has_8_cells_clear_of_the_rail_and_a_rear_papaya_cradle(robot):
    basket = _id(robot, O.mjOBJ_BODY, "dorsal_basket")
    cells = np.array([robot.site_pos[_id(robot, O.mjOBJ_SITE, f"tray_cell_{i}")] for i in range(8)])
    assert all(robot.site_bodyid[_id(robot, O.mjOBJ_SITE, f"tray_cell_{i}")] == basket for i in range(8))
    dist = [np.linalg.norm(a[:2] - b[:2]) for i, a in enumerate(cells) for b in cells[i + 1:]]
    assert min(dist) >= 0.085                                           # one fruit per cell, no shared cell
    rail_half_y = float(robot.geom_size[_id(robot, O.mjOBJ_GEOM, "gantry_rail")][1])
    assert np.all(np.abs(cells[:, 1]) - 0.035 >= rail_half_y)          # 7 cm fruit clear of the rail in y
    assert np.all(np.abs(cells[:, 1]) + 0.035 <= 0.108)                # and inside the side walls
    div_x = float(robot.geom_pos[_id(robot, O.mjOBJ_GEOM, "basket_divider")][0])
    assert div_x == pytest.approx(-0.05)
    assert np.all(cells[:, 0] - 0.04 >= div_x) and np.all(cells[:, 0] + 0.04 <= 0.308)   # front compartment
    pap = robot.site_pos[_id(robot, O.mjOBJ_SITE, "papaya_cradle")]
    assert -0.30 + 0.022 < pap[0] - 0.095 and pap[0] + 0.095 < div_x   # behind the divider, ahead of the mast


# ------------------------------------------------------------------------------------- standing
def test_standing_2s_no_fall(flat):
    s = mb.stand(seconds=2.0, settle_s=2.0, model=flat)   # AT2 settle
    print(f"WP1 stand 2 s: {json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in s.items()})}")
    assert not s["fell"] and s["n_feet"] == 6
    assert s["trunk_speed_rms"] <= 0.002


def test_standing_2s_with_the_crown_fully_up_no_fall(flat):
    m = mb.build(NANDI_TERRAIN="flat", NANDI_CROPS=1)
    _set_ext(m, STROKE_M)
    s = mb.stand(seconds=2.0, settle_s=1.0, model=m)
    d = _home(m)
    print(f"WP1 stand 2 s at ext 1.6: fell {s['fell']}, feet {s['n_feet']}, trunk speed rms "
          f"{s['trunk_speed_rms']:.5f}, lens {float(d.cam_xpos[_id(m, O.mjOBJ_CAMERA, 'crown_cam')][2]):.3f} m")
    assert not s["fell"] and s["n_feet"] == 6


# ------------------------------------------------------------------------------------- beacon logic
def test_default_patterns_follow_rth_docking_e11():
    p = bc.load_patterns(path=os.devnull + ".missing")
    assert p["working"] == {"rgb": bc.GREEN, "mode": "steady", "hz": 0.0}
    assert p["going_home"]["rgb"] == bc.GREEN and p["going_home"]["mode"] == "pulse"
    assert p["driving"]["rgb"] == bc.BLUE and p["charging"]["rgb"] == bc.CYAN
    assert p["charging"]["mode"] == "breathe"
    assert p["stopped"]["rgb"] == bc.AMBER and p["estop"]["rgb"] == bc.RED and p["fault"]["rgb"] == bc.RED
    assert p["premotion"]["mode"] == "flash" and p["premotion"]["hz"] >= 3.0
    assert set(p) == set(bc.PRIORITY)


def test_intensity_modes():
    steady, flash = {"mode": "steady", "hz": 0.0}, {"mode": "flash", "hz": 2.0}
    assert bc.intensity(steady, 12.3) == 1.0
    assert bc.intensity(flash, 0.1) == 1.0 and bc.intensity(flash, 0.3) == bc.DIM
    pulse = {"mode": "pulse", "hz": 1.0}
    assert bc.intensity(pulse, 0.2) == 1.0 and bc.intensity(pulse, 0.5) == bc.DIM
    br = {"mode": "breathe", "hz": 0.5}
    assert bc.intensity(br, 0.0) == pytest.approx(bc.DIM) and bc.intensity(br, 1.0) == pytest.approx(1.0)
    assert bc.rgba_at({"rgb": (1.0, 0.5, 0.0), "mode": "flash", "hz": 1.0}, 0.75) == pytest.approx(
        (bc.DIM, 0.5 * bc.DIM, 0.0, 1.0))


@pytest.mark.parametrize("bad", [{"rgb": (1, 0)}, {"rgb": (2, 0, 0)}, {"rgb": (1, 0, 0), "mode": "disco"},
                                 {"rgb": (1, 0, 0), "mode": "flash", "hz": 0}, {"rgb": (1, 0, 0), "hz": float("nan")},
                                 "red", {"rgb": (True, 0, 0)}])
def test_bad_patterns_are_refused(bad):
    with pytest.raises(ValueError):
        bc.validate_pattern(bad)


def test_patterns_from_config_override_and_refuse_unknown(tmp_path, monkeypatch):
    cfg = tmp_path / "beacon.yaml"
    cfg.write_text("patterns:\n  working:\n    rgb: [0.0, 0.0, 1.0]\n  stopped:\n    mode: flash\n    hz: 1.0\n")
    p = bc.load_patterns(str(cfg))
    assert p["working"]["rgb"] == (0.0, 0.0, 1.0) and p["working"]["mode"] == "steady"
    assert p["stopped"]["rgb"] == bc.AMBER and p["stopped"]["mode"] == "flash"
    monkeypatch.setenv(bc.CONFIG_ENV, str(cfg))
    assert bc.load_patterns()["working"]["rgb"] == (0.0, 0.0, 1.0)
    cfg.write_text("patterns:\n  disco:\n    rgb: [1, 0, 1]\n")
    with pytest.raises(ValueError):
        bc.load_patterns(str(cfg))


def test_choose_state_priority():
    assert bc.choose_state(None, None) == ("idle", "default", "")
    assert bc.choose_state(None, {"state": "working", "source": "mission"})[:2] == ("working", "mission")
    assert bc.choose_state({"reason": "estop", "since": 1.0}, {"state": "working"})[0] == "estop"
    assert bc.choose_state({"reason": "hook_fault", "since": 1.0}, None)[0] == "fault"
    assert bc.choose_state({"reason": "supervisor_lost", "since": 1.0}, None)[0] == "fault"
    assert bc.choose_state(None, {"state": "premotion"})[0] == "premotion"
    # nobody may request a safety state, and unknown states are ignored
    assert bc.choose_state(None, {"state": "estop"})[0] == "idle"
    assert bc.choose_state(None, {"state": "fault"})[0] == "idle"
    assert bc.choose_state(None, {"state": "rainbow"})[0] == "idle"
    assert bc.choose_state(None, "working")[0] == "idle"
    # malformed requests (any JSON shape) are ignored, never raised (review: unhashable state killed the hook)
    for bad in ([1], {"x": 1}, 3, None, True, 1.5):
        assert bc.choose_state(None, {"state": bad, "source": ["a"], "reason": {"b": 1}}) == ("idle", "default", "")


def test_handheld_decoder_seq_timeout_boot_and_reset():
    h = bc.HandheldEstop()
    st = h.state(0.0)
    assert not st["seen"] and not st["link_ok"] and not st["estop_request"] and not st["local_link_lost"]
    assert h.update({"device": "hh-1", "boot": "a", "seq": 1, "pressed": False}, 0.0)
    assert h.state(0.3)["link_ok"] and not h.state(0.3)["estop_request"]
    assert not h.update({"device": "hh-1", "boot": "a", "seq": 1, "pressed": True}, 0.4)   # same seq: not new
    assert h.state(0.6)["local_link_lost"] and not h.state(0.6)["link_ok"]              # > 0.5 s silent
    assert h.update({"device": "hh-1", "boot": "a", "seq": 2, "pressed": True}, 0.7)
    st = h.state(0.8)
    assert st["estop_request"] and st["pressed"] and st["link_ok"] and st["seq"] == 2
    assert not h.update({"device": "hh-1", "boot": "a", "seq": 1, "pressed": False}, 0.9)  # replay refused
    # a second device cannot clear the press (the first device heard is paired)
    assert not h.update({"device": "hh-2", "seq": 0, "pressed": False}, 0.95)
    assert h.state(0.96)["estop_request"] and h.foreign == 1
    # the paired handheld reboots: accepted (link alive), but its "released" does not clear the earlier press
    assert h.update({"device": "hh-1", "boot": "b", "seq": 0, "pressed": False}, 1.0)
    st = h.state(1.1)
    assert st["estop_request"] and st["link_ok"] and st["device"] == "hh-1/b"
    assert h.update({"device": "hh-1", "boot": "b", "seq": 1, "pressed": True}, 1.15)     # pressed on boot b ...
    assert h.update({"device": "hh-1", "boot": "b", "seq": 2, "pressed": False}, 1.18)    # ... then released
    assert not h.state(1.19)["estop_request"]
    for bad in ({"seq": 3}, {"seq": "3", "pressed": False}, {"seq": True, "pressed": False}, [1], "pressed"):
        assert not h.update(bad, 1.2)
        assert not h.update(bad, 1.25)            # the same bad message every tick counts once
    assert h.rejected == 5
    h.reset(keep_seq=True)
    assert not h.update({"device": "hh-1", "boot": "b", "seq": 2, "pressed": False}, 0.0)  # stale after reset
    assert not h.state(0.1)["seen"]
    h.reset()
    assert h.update({"device": "hh-1", "boot": "b", "seq": 0, "pressed": False}, 0.0)


def test_handheld_link_timeout_counts_wall_time():
    """MISSION-PRESETS 8: 0.5 s of silence is link loss in wall time too (a sim at RTF 0.3 must not stretch it)."""
    wall = [100.0]
    h = bc.HandheldEstop(wall_clock=lambda: wall[0])
    assert h.update({"device": "hh-1", "seq": 1, "pressed": False}, 0.0)
    wall[0] += 0.3
    assert h.state(0.09)["link_ok"]
    wall[0] += 0.3                                # 0.6 s wall, 0.18 s sim (RTF 0.3)
    st = h.state(0.18)
    assert not st["link_ok"] and st["local_link_lost"] and st["age_s"] == pytest.approx(0.6)
    # and a sim faster than real time still times out on sim time
    h2 = bc.HandheldEstop(wall_clock=lambda: 0.0)
    h2.update({"device": "hh-1", "seq": 1, "pressed": False}, 0.0)
    assert h2.state(0.6)["local_link_lost"]


# ------------------------------------------------------------------------------------- beacon hook + provider
class _Bus(hook_registry.Bus):
    pass


def _published(bus, topic):
    return [json.loads(p) for t, p in bus.drain() if t == topic]


def test_beacon_file_registers_two_hooks_with_their_inputs():
    reg = hook_registry.load_all(names=["beacon"])
    rows = {r["name"]: r for r in reg.table()}
    assert set(rows) == {"beacon", "handheld_estop"}
    assert rows["beacon"]["mutates_model"] and not rows["beacon"]["every_step"]
    assert rows["beacon"]["rate_hz"] == bc.BEACON_RATE_HZ
    assert set(rows["beacon"]["topics_in"]) == {bc.REQUEST_TOPIC}
    assert not rows["handheld_estop"]["mutates_model"] and rows["handheld_estop"]["rate_hz"] == bc.HANDHELD_RATE_HZ
    assert set(rows["handheld_estop"]["topics_in"]) == {bc.HANDHELD_TOPIC}


def _reg_tick(reg, m, d, bus, t):
    d.time = t
    reg.tick(m, d, bus)


def test_malformed_request_never_disables_the_handheld_input(robot):
    """Review: {"state": [1]} raised TypeError in the shared hook, so the handheld input died with it."""
    m = mujoco.MjModel.from_xml_path(XML)
    d = _home(m)
    reg = hook_registry.load_all(names=["beacon"])
    bus = _Bus()
    for i, bad in enumerate(([1], {"x": 1}, 7, None)):
        bus.deliver(bc.REQUEST_TOPIC, {"state": bad, "source": "console"})
        _reg_tick(reg, m, d, bus, 0.05 * (i + 1))
    assert reg.drain_faults() == [] and reg.faulted() == {}
    bus.deliver(bc.HANDHELD_TOPIC, {"device": "hh-1", "seq": 2, "pressed": True})
    _reg_tick(reg, m, d, bus, 0.30)
    assert bus.values["handheld_estop"]["estop_request"]


def test_a_faulting_beacon_leaves_the_handheld_input_running(robot, monkeypatch):
    m = mujoco.MjModel.from_xml_path(XML)
    d = _home(m)
    reg = hook_registry.load_all(names=["beacon"])
    bus = _Bus()

    def boom(*a, **k):
        raise RuntimeError("lamp broke")
    monkeypatch.setattr(bc, "choose_state", boom)
    _reg_tick(reg, m, d, bus, 0.05)
    assert [n for n, _ in reg.drain_faults()] == ["beacon"]
    bus.deliver(bc.HANDHELD_TOPIC, {"device": "hh-1", "seq": 1, "pressed": True})
    _reg_tick(reg, m, d, bus, 0.10)
    assert bus.values["handheld_estop"]["estop_request"] and "handheld_estop" not in reg.faulted()


def test_malformed_newest_handheld_message_keeps_the_2hz_publish_rate():
    b = bc.BeaconHooks(wall_clock=None)
    bus = _Bus()
    bus.deliver(bc.HANDHELD_TOPIC, {"device": "hh-1", "seq": 1, "pressed": False})
    b.handheld_tick(None, None, 0.0, bus)
    bus.deliver(bc.HANDHELD_TOPIC, {"device": "hh-1", "seq": "2", "pressed": False})   # malformed, kept as newest
    bus.drain()
    for i in range(1, 21):                                                            # 1 s at 20 Hz
        b.handheld_tick(None, None, 0.05 * i, bus)
    pubs = _published(bus, bc.HANDHELD_STATE_TOPIC)
    assert len(pubs) <= 3, len(pubs)           # link-lost edge + the 2 Hz keep-alive, not 20
    assert bus.values["handheld_estop"]["rejected"] == 1


def test_beacon_hook_drives_the_lamp_and_publishes(robot):
    m = mujoco.MjModel.from_xml_path(XML)
    d = _home(m)
    g = bc.beacon_geom(m)
    rest = m.geom_rgba[g].copy()
    b = bc.BeaconHooks()
    bus = _Bus()
    b.tick(m, d, 0.0, bus)
    msgs = [json.loads(p) for t, p in bus.drain() if t == bc.BEACON_TOPIC]
    assert msgs and msgs[-1]["state"] == "idle" and msgs[-1]["source"] == "default"
    bus.deliver(bc.REQUEST_TOPIC, {"state": "working", "source": "mission"})
    b.tick(m, d, 0.05, bus)
    assert _published(bus, bc.BEACON_TOPIC)[-1]["state"] == "working"
    assert np.allclose(m.geom_rgba[g], (*bc.GREEN, 1.0))
    b.tick(m, d, 0.10, bus)
    assert _published(bus, bc.BEACON_TOPIC) == []                     # no change, < 1 s: quiet
    b.tick(m, d, 1.10, bus)
    assert len(_published(bus, bc.BEACON_TOPIC)) == 1                 # 1 Hz keep-alive
    # handheld pressed: its state is published and shared on the bus, but the lamp does NOT turn red on its own
    # (nothing latches on it until WP6; a red lamp would claim a stop that did not happen)
    bus.deliver(bc.HANDHELD_TOPIC, {"device": "hh-1", "seq": 5, "pressed": True})
    b.tick(m, d, 1.15, bus)
    out = bus.drain()
    hh = [json.loads(p) for t, p in out if t == bc.HANDHELD_STATE_TOPIC]
    bea = [json.loads(p) for t, p in out if t == bc.BEACON_TOPIC]
    assert hh and hh[-1]["estop_request"] and bus.values["handheld_estop"]["pressed"]
    assert bea == [] and np.allclose(m.geom_rgba[g], (*bc.GREEN, 1.0))
    # the handheld goes silent: link lost is reported on its topic, the lamp still shows the request
    b.tick(m, d, 1.80, bus)
    assert bus.values["handheld_estop"]["local_link_lost"]
    assert np.allclose(m.geom_rgba[g], (*bc.GREEN, 1.0))
    # the runner holds (the supervisor latched): only now red, source hold, reason reported
    bus.values["hold"] = {"reason": "estop", "since": 1.0}
    b.tick(m, d, 2.0, bus)
    last = _published(bus, bc.BEACON_TOPIC)[-1]
    assert (last["state"], last["source"], last["reason"]) == ("estop", "hold", "estop")
    assert np.allclose(m.geom_rgba[g], (*bc.RED, 1.0))
    # assist request flashes amber at 1 Hz once the hold and the handheld are clear
    bus.values["hold"] = None
    bus.deliver(bc.HANDHELD_TOPIC, {"device": "hh-1", "seq": 6, "pressed": False})
    bus.deliver(bc.REQUEST_TOPIC, {"state": "assist", "source": "harvest"})
    b.tick(m, d, 3.25, bus)
    assert _published(bus, bc.BEACON_TOPIC)[-1]["state"] == "assist"
    on = m.geom_rgba[g].copy()
    b.tick(m, d, 3.75, bus)
    assert np.allclose(on, (*bc.AMBER, 1.0)) and np.allclose(m.geom_rgba[g], (*(c * bc.DIM for c in bc.AMBER), 1.0))
    # world reset restores the rest colour and forgets the link
    b.reset(m, d)
    assert np.allclose(m.geom_rgba[g], rest)
    b.tick(m, d, 0.0, bus)
    assert not bus.values["handheld_estop"]["seen"]                   # the stale message is not fresh


def test_beacon_hook_is_harmless_without_a_beacon_geom(v1):
    d = _home(v1)
    b = bc.BeaconHooks()
    bus = _Bus()
    b.tick(v1, d, 0.0, bus)
    b.reset(v1, d)
    assert _published(bus, bc.BEACON_TOPIC)[-1]["state"] == "idle"
    assert bc._pack(v1) == {}
    with pytest.raises(ValueError):
        bc._unpack(v1, {"rgba": [1, 0, 0, 1]})


def test_beacon_provider_round_trips_through_sim_state():
    assert bc.PROVIDER_KEY in sim_state.discover_providers()
    m = mujoco.MjModel.from_xml_path(XML)
    d = _home(m)
    g = bc.beacon_geom(m)
    m.geom_rgba[g] = (*bc.RED, 1.0)
    state = sim_state.decode(sim_state.encode(sim_state.pack(m, d, "wp1test")))
    assert state["mutations"]["providers"]["beacon"]["data"]["rgba"] == pytest.approx([*bc.RED, 1.0])
    m2 = mujoco.MjModel.from_xml_path(XML)
    d2 = mujoco.MjData(m2)
    ok, why = sim_state.apply(m2, d2, state, "wp1test")
    assert ok, why
    assert np.allclose(m2.geom_rgba[g], (*bc.RED, 1.0))
    for bad in ({"rgba": [1, 0, 0]}, {"rgba": [1, 0, 0, 2]}, {"rgba": [1, 0, "x", 1]}):
        with pytest.raises(ValueError):
            bc._unpack(m2, bad)
    bc._unpack(m2, {})   # a world without the module's state: no-op
