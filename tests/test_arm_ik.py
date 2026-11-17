"""IK of the dorsal arm on the shared farm world (ROS-free)."""
import json
import math

import mujoco
import numpy as np
import pytest

from controllers.arm_ik_solver import DorsalArmIKSolver, compute_work_envelope
from controllers.farm_scene import CROP_RADIUS, crop_body_names, crop_qpos_slice, load_farm_model
from controllers import harvest_logic as hl


@pytest.fixture(scope="module")
def ik():
    return DorsalArmIKSolver(load_farm_model())


def home_qpos(ik):
    d = mujoco.MjData(ik.model)
    mujoco.mj_resetDataKeyframe(ik.model, d, 0)
    return d.qpos.copy()


def robot_at(qpos, x, y, heading):
    """Copy of qpos with the trunk moved to (x, y) at its home height, rotated about z."""
    q = qpos.copy()
    q[0], q[1] = x, y
    q[3:7] = [math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)]
    return q


def assert_reaches(ik, qpos, target):
    res = ik.solve(qpos, target)
    tip = ik.tip_pos(qpos, res.q)
    axis = ik.tool_axis(qpos, res.q)
    assert res.ok, (target, res)
    assert np.linalg.norm(tip - target) < 0.01
    assert axis[2] < -0.99  # gripper points (within ~8 deg of) straight down
    assert np.all(res.q >= ik.limits[:, 0] - 1e-9) and np.all(res.q <= ik.limits[:, 1] + 1e-9)
    return res


def test_chain_constants_from_model(ik):
    assert np.allclose(ik.shoulder0, [0.05, 0.0, 0.195])
    assert (ik.l1, ik.l2, ik.l3) == pytest.approx((0.34, 0.34, 0.065))
    assert ik.grip_open < ik.grip_closed


def test_home_reaches_crop_item_1_hover_and_grasp(ik):
    q = home_qpos(ik)
    crop = q[crop_qpos_slice(1)][:3]
    for dz in (hl.GRASP_TIP_DZ, hl.GRASP_TIP_DZ + hl.APPROACH_DZ, hl.GRASP_TIP_DZ + hl.LIFT_DZ):
        assert_reaches(ik, q, crop + [0.0, 0.0, dz])


@pytest.mark.parametrize("offset", [(0.60, 0.0), (0.45, 0.25), (0.45, -0.25)])
@pytest.mark.parametrize("heading", [0.0, 0.7, -2.0])
def test_every_default_crop_reachable_with_robot_translated(ik, offset, heading):
    q0 = home_qpos(ik)
    names = crop_body_names(ik.model)
    assert len(names) == 6
    c, s = math.cos(heading), math.sin(heading)
    for k in range(1, len(names) + 1):
        crop = q0[crop_qpos_slice(k)][:3].copy()
        # put the trunk so the crop sits at `offset` in the trunk frame
        ox, oy = offset
        q = robot_at(q0, crop[0] - (c * ox - s * oy), crop[1] - (s * ox + c * oy), heading)
        assert np.allclose(hl.to_trunk(crop, q)[:2], offset)
        for dz in (hl.GRASP_TIP_DZ, hl.GRASP_TIP_DZ + hl.APPROACH_DZ):
            assert_reaches(ik, q, crop + [0.0, 0.0, dz])


def test_tilted_trunk_is_handled_in_world_frame(ik):
    q = home_qpos(ik)
    tilt = math.radians(6.0)  # pitch nose-down a little
    q[3:7] = [math.cos(tilt / 2), 0.0, math.sin(tilt / 2), 0.0]
    assert_reaches(ik, q, q[crop_qpos_slice(1)][:3] + [0.0, 0.0, hl.GRASP_TIP_DZ])


def test_carry_and_drop_poses_reachable_with_gantry_forward(ik):
    q = home_qpos(ik)
    g = ik.limits[0, 1]
    for p_trunk in ([0.62, 0.0, hl.CARRY_TIP_Z], [0.01, hl.DROP_Y, hl.DROP_TIP_Z], [0.01, hl.DROP_Y, hl.CARRY_TIP_Z]):
        target = hl.to_world(p_trunk, q)
        res = ik.solve(q, target, gantry=g)
        assert res.ok, (p_trunk, res)
        assert res.q[0] == pytest.approx(g, abs=0.02)


def test_out_of_reach_is_reported_not_faked(ik):
    q = home_qpos(ik)
    res = ik.solve(q, [1.3, 0.0, 0.06])
    assert not res.ok
    assert res.pos_err > 0.05


def test_gripper_opening_is_measured_from_model(ik):
    assert 0.0 < ik.gripper_opening() < 0.2


def test_gripper_can_pinch_a_crop(ik):
    """Open gap wider than a crop, fully-closed gap narrower: the pads can straddle and squeeze it."""
    crop_d = 2 * CROP_RADIUS
    opening = ik.gripper_opening()
    closed_gap = opening - 2 * (ik.grip_closed - ik.grip_open)  # both fingers slide inward
    assert ik.grip_open == pytest.approx(0.0)
    assert opening > crop_d + 0.01      # >= 5 mm clearance per side on the way down
    assert closed_gap < crop_d - 0.02   # position servo still pushes ~1 cm into the crop per finger
    assert closed_gap > 0.0             # fingers never drive into each other


def test_work_envelope_payload_is_compact():
    payload = compute_work_envelope(load_farm_model())
    assert {"arm_yaw", "arm_shoulder", "gantry_slide"} <= set(payload["joints_discovered"])
    assert payload["canopy_segments"] and payload["floor_segments"]
    seg = payload["canopy_segments"][0]
    assert len(seg) == 2 and len(seg[0]) == 3
    blob = json.dumps(payload, separators=(",", ":"))
    assert len(blob) < 60_000  # one Trigger response; the old 210 KB payload never arrived
