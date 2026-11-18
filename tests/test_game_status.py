"""Unit tests for controllers.game_status (hand-computed expectations) + one MuJoCo integration check."""
import json
import math

import mujoco
import numpy as np
import pytest

from atharv_nandi.core.farm_scene import ROBOT_XML, crop_body_names, crop_qpos_slice, load_farm_model
from atharv_nandi.core.game_status import (
    BASKET_FLOOR_Z,
    BASKET_HALF_X,
    BASKET_HALF_Y,
    BASKET_TOP_Z,
    FallCounter,
    RtfMeter,
    build_game_status,
    in_basket,
    is_fallen,
    plan_catch_up,
    quat_to_mat,
    tilt_deg,
    world_to_trunk,
)

LEVEL = (1.0, 0.0, 0.0, 0.0)
YAW_90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))  # +90 deg about world z
TRUNK = (0.0, 0.0, 0.34)


def roll_quat(deg):
    a = math.radians(deg) / 2
    return (math.cos(a), math.sin(a), 0.0, 0.0)


def pitch_quat(deg):
    a = math.radians(deg) / 2
    return (math.cos(a), 0.0, math.sin(a), 0.0)


# ---------------- frames ----------------
def test_quat_to_mat_identity_and_yaw():
    assert np.allclose(quat_to_mat(LEVEL), np.eye(3))
    r = quat_to_mat(YAW_90)
    assert np.allclose(r @ [1, 0, 0], [0, 1, 0])   # body x points to world +y
    assert np.allclose(r @ [0, 1, 0], [-1, 0, 0])
    assert np.allclose(quat_to_mat((2.0, 0, 0, 0)), np.eye(3))  # normalised
    assert np.allclose(quat_to_mat((0, 0, 0, 0)), np.eye(3))    # degenerate -> identity


def test_world_to_trunk_translates_then_unrotates():
    # trunk at (1, 2, 0.34) yawed +90: world offset (0, +0.2, +0.1) is body (+0.2, 0, +0.1)
    p = world_to_trunk((1.0, 2.2, 0.44), (1.0, 2.0, 0.34), YAW_90)
    assert np.allclose(p, [0.2, 0.0, 0.1])


# ---------------- basket ----------------
def _crop_at(dx, dy, dz):
    return (TRUNK[0] + dx, TRUNK[1] + dy, TRUNK[2] + dz)


def test_crop_resting_on_basket_floor_is_in():
    # crop radius 0.035 on floor top 0.061 -> centre at trunk z + 0.096
    assert in_basket(_crop_at(0.0, 0.0, 0.096), TRUNK, LEVEL)


@pytest.mark.parametrize("dx, dy, dz, expected", [
    (0.30, 0.0, 0.096, True),
    (0.32, 0.0, 0.096, False),     # past |x| < 0.31
    (-0.32, 0.0, 0.096, False),
    (0.0, 0.10, 0.096, True),
    (0.0, -0.12, 0.096, False),    # past |y| < 0.11
    (0.0, 0.0, 0.05, False),       # below basket floor top (0.061)
    (0.0, 0.0, 0.12, True),        # contract top: trunk-frame z +0.12 (wall tops are at 0.125)
    (0.0, 0.0, 0.13, False),       # above the walls: still falling in, or held by the gripper
    (0.0, 0.0, 0.18, False),       # resting on the gantry rail (top 0.145 + radius 0.035)
    (0.0, 0.0, 0.19, False),       # held above the basket, not in it
    (0.62, 0.0, -0.305, False),    # crop_item_1 on the ground at home
])
def test_basket_footprint_edges(dx, dy, dz, expected):
    assert in_basket(_crop_at(dx, dy, dz), TRUNK, LEVEL) is expected


def test_basket_follows_trunk_yaw():
    crop = (0.0, 0.25, 0.436)          # 0.25 m to the world-left of the trunk
    assert not in_basket(crop, TRUNK, LEVEL)   # level: trunk y = 0.25 > 0.11
    assert in_basket(crop, TRUNK, YAW_90)      # yawed 90: trunk x = 0.25 < 0.31


def test_basket_constants_match_robot_mjcf():
    m = mujoco.MjModel.from_xml_path(ROBOT_XML)
    basket = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "dorsal_basket")
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "basket_floor")
    floor_top = m.body_pos[basket][2] + m.geom_pos[floor][2] + m.geom_size[floor][2]
    assert floor_top == pytest.approx(BASKET_FLOOR_Z)
    wall = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "basket_left_wall")
    wall_top = m.body_pos[basket][2] + m.geom_pos[wall][2] + m.geom_size[wall][2]
    rail = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "gantry_rail")
    rail_top = m.geom_pos[rail][2] + m.geom_size[rail][2]  # gantry_rail is a trunk geom
    assert BASKET_FLOOR_Z + 0.035 < BASKET_TOP_Z <= wall_top  # a crop on the floor counts; nothing above the walls
    assert rail_top + 0.035 > BASKET_TOP_Z                    # a crop lying on the rail does not
    assert BASKET_HALF_X < m.geom_size[floor][0]
    assert BASKET_HALF_Y < m.geom_size[floor][1]


# ---------------- tilt / fall ----------------
def test_tilt_deg_hand_values():
    assert tilt_deg(LEVEL) == pytest.approx(0.0)
    assert tilt_deg(YAW_90) == pytest.approx(0.0, abs=1e-6)   # yaw is not tilt
    assert tilt_deg(roll_quat(60)) == pytest.approx(60.0)     # R22 = 1 - 2 sin^2(30) = 0.5
    assert tilt_deg(pitch_quat(-30)) == pytest.approx(30.0)
    assert tilt_deg((0.0, 1.0, 0.0, 0.0)) == pytest.approx(180.0)  # upside down


@pytest.mark.parametrize("pos, quat, expected", [
    ((0, 0, 0.34), LEVEL, False),
    ((0, 0, 0.11), LEVEL, True),            # trunk z < 0.12
    ((0, 0, 0.12), LEVEL, False),           # boundary is strict
    ((0, 0, 0.34), pitch_quat(49), False),
    ((0, 0, 0.34), pitch_quat(51), True),   # tilt > 50 deg
    ((0, 0, 0.34), roll_quat(-51), True),
])
def test_is_fallen(pos, quat, expected):
    assert is_fallen(pos, quat) is expected


def test_fall_counter_counts_rising_edges_only():
    fc = FallCounter()
    edges = [fc.update(v) for v in (False, True, True, False, True, True)]
    assert edges == [False, True, False, False, True, False]
    assert fc.falls == 2 and fc.fallen is True
    fc.reset()
    assert fc.falls == 0 and fc.fallen is False
    assert fc.update(True) is True   # a robot lying down right after reset counts once


# ---------------- real-time pacing ----------------
def test_rtf_meter_windows_and_smoothing():
    m = RtfMeter(alpha=0.3, window_s=0.25)
    assert m.update(0.0, 0.0) == 0.0     # first call only anchors
    assert m.update(0.1, 0.1) == 0.0     # window not elapsed yet
    assert m.update(0.3, 0.3) == pytest.approx(1.0)    # first sample taken as-is
    assert m.update(0.4, 0.8) == pytest.approx(0.76)   # inst 0.1/0.5 = 0.2; 0.7*1 + 0.3*0.2
    m.rebase()                                          # sim time jumps back to 0 on reset
    assert m.update(0.0, 1.0) == pytest.approx(0.76)   # re-anchors, no negative sample
    assert m.update(0.5, 1.5) == pytest.approx(0.832)  # inst 1.0; 0.7*0.76 + 0.3


@pytest.mark.parametrize("behind, expected_n, expected_drop", [
    (-0.01, 0, 0.0),      # ahead of wall-clock: wait
    (0.0015, 0, 0.0),     # less than one step due
    (0.006, 3, 0.0),      # 0.006/0.002 is 2.999.. in float; epsilon keeps it 3
    (0.0105, 5, 0.0),
    (0.05, 25, 0.0),      # exactly the cap
    (0.30, 25, 0.25),     # far behind: run the cap, drop the rest
])
def test_plan_catch_up(behind, expected_n, expected_drop):
    n, dropped = plan_catch_up(behind, 0.002, 25)
    assert n == expected_n
    assert dropped == pytest.approx(expected_drop)


# ---------------- payload ----------------
def test_build_game_status_payload():
    crops = {
        "crop_item_1": _crop_at(0.1, 0.0, 0.096),   # in basket
        "crop_item_2": (1.0, 0.45, 0.035),         # on the ground
        "crop_item_3": _crop_at(-0.2, 0.05, 0.10),  # in basket
    }
    st = build_game_status(crops, TRUNK, LEVEL, falls=1, fallen=False, elapsed_s=12.3456, rtf=0.98765)
    assert st == {
        "crops_total": 3,
        "in_basket": 2,
        "in_basket_names": ["crop_item_1", "crop_item_3"],
        "falls": 1,
        "fallen": False,
        "elapsed_s": 12.35,
        "rtf": 0.988,
    }
    assert json.loads(json.dumps(st)) == st


def test_build_game_status_accepts_numpy_views():
    qpos = np.zeros(39)
    qpos[0:7] = [0, 0, 0.34, 1, 0, 0, 0]
    qpos[32:39] = [0, 0, 0.436, 1, 0, 0, 0]
    st = build_game_status({"crop_item_1": qpos[32:35]}, qpos[0:3], qpos[3:7], np.int64(0), np.bool_(False), 0.0, 0.0)
    assert st["in_basket_names"] == ["crop_item_1"]
    json.dumps(st)  # numpy scalars must not leak into the payload


# ---------------- integration with the live world ----------------
def test_crop_dropped_into_basket_scores_and_robot_stays_up(monkeypatch):
    monkeypatch.setenv("NANDI_TERRAIN", "flat")
    monkeypatch.setenv("NANDI_CROPS", "2")
    m = load_farm_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    s1 = crop_qpos_slice(1)
    # above the rear half of the basket, clear of the gantry rail (|y| < 0.022) and the arm (x > 0)
    d.qpos[s1] = [-0.2, 0.07, 0.34 + 0.25, 1, 0, 0, 0]  # trunk-frame z 0.25: not counted yet
    mujoco.mj_forward(m, d)

    def status():
        crops = {n: d.qpos[crop_qpos_slice(k)][:3] for k, n in enumerate(crop_body_names(m), start=1)}
        return build_game_status(crops, d.qpos[0:3], d.qpos[3:7], 0, is_fallen(d.qpos[0:3], d.qpos[3:7]), d.time, 1.0)

    assert status()["in_basket"] == 0
    for _ in range(int(1.5 / m.opt.timestep)):
        d.ctrl[:] = m.key_ctrl[0]
        mujoco.mj_step(m, d)
    st = status()
    assert st["in_basket_names"] == ["crop_item_1"]
    assert st["crops_total"] == 2
    assert st["fallen"] is False
