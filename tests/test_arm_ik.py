"""IK of the dorsal arm on the shared farm world (ROS-free)."""
import json
import math

import mujoco
import numpy as np
import pytest

from atharv_nandi.core.arm_ik_solver import DorsalArmIKSolver, compute_work_envelope
from atharv_nandi.core.farm_scene import CROP_RADIUS, crop_body_names, crop_qpos_slice, load_farm_model
from atharv_nandi.core import harvest_logic as hl


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
    assert (ik.l1, ik.l2, ik.l3) == pytest.approx((0.34, 0.34, 0.105))
    assert ik.grip_open < ik.grip_closed
    assert ik.crop_mass == pytest.approx(0.15)


def test_shoulder_range_stops_where_the_upper_arm_meets_its_carriage(ik):
    """The shoulder joint allows 2.4 rad but the upper arm lands on the gantry carriage at ~1.71 rad."""
    m, d = ik.model, mujoco.MjData(ik.model)
    lo, hi = ik.limits[2]
    assert lo == pytest.approx(-1.6) and 1.6 < hi < 1.71
    upper = [g for g in range(m.ngeom) if m.geom_bodyid[g] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_upper")]
    carriage = [g for g in range(m.ngeom)
                if m.geom_bodyid[g] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_carriage")]
    fromto = np.zeros(6)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for shoulder, touching in ((hi, False), (hi + 0.05, True)):
        d.qpos[ik.qadr] = [0.0, 0.0, shoulder, 0.0, 0.0]
        mujoco.mj_kinematics(m, d)
        dist = mujoco.mj_geomDistance(m, d, upper[0], carriage[0], 0.05, fromto)
        assert (dist < 0.0) == touching, (shoulder, dist)


def test_pinned_gantry_stays_pinned(ik):
    """With the tool down the chain is redundant along the rail; a pinned gantry must not drift
    (it used to wander 9 cm within two Cartesian waypoints of a side reach)."""
    q = home_qpos(ik)
    for p_trunk in ([0.12, 0.45, -0.18], [0.12, 0.45, -0.20], [0.5, -0.2, -0.25], [-0.3, 0.42, -0.2]):
        for g in (-0.2, 0.07, 0.26):
            res = ik.solve(q, hl.to_world(p_trunk, q), gantry=g)
            assert res.q[0] == pytest.approx(g), (p_trunk, g)


def test_gravity_offsets_hold_the_tip_on_target(ik):
    """Commanding IK joints alone leaves the tip sagging at long reach; adding gravity_offsets holds it.

    The MJCF has no body gravcomp (it lifted the chassis through the freejoint, review F1): the arm's gravity is fed
    forward only as these position-target offsets, so they must be non-zero even without a payload."""
    m = ik.model
    q0 = home_qpos(ik)
    target = q0[crop_qpos_slice(1)][:3] + [0.0, 0.0, hl.GRASP_TIP_DZ + hl.APPROACH_DZ]
    res = ik.solve(q0, target)
    assert res.ok
    errs = {}
    for ff in (False, True):
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[ik.qadr] = res.q
        cmd = res.q + (ik.gravity_offsets(d.qpos, res.q) if ff else 0.0)
        d.ctrl[:] = m.key_ctrl[0]
        d.ctrl[:5] = cmd
        for _ in range(1500):
            mujoco.mj_step(m, d)
        # trunk frame: the trunk settles ~1.5 cm lower on its legs than the keyframe
        errs[ff] = float(np.linalg.norm(hl.to_trunk(ik.tip_pos(d.qpos), d.qpos) - hl.to_trunk(target, q0)))
    print(f"tip error at hover: {errs[False] * 100:.1f} cm without, {errs[True] * 100:.1f} cm with gravity feed-forward")
    assert errs[True] < 0.005 and errs[True] < 0.5 * errs[False]
    off0, off1 = ik.gravity_offsets(q0, res.q), ik.gravity_offsets(q0, res.q, payload=0.5)
    assert np.any(np.abs(off0) > 1e-3) and np.linalg.norm(off1) > np.linalg.norm(off0)


def test_robot_collisions_flags_arm_on_the_robot(ik):
    q = home_qpos(ik)
    assert ik.robot_collisions(q, [0.0, 0.0, 0.1, -1.8, 1.4]) == []          # rest pose is clear
    beside_leg = hl.to_world([0.1, 0.2, 0.0], q)                               # tip beside the front leg
    beside_leg[2] = CROP_RADIUS + hl.GRASP_TIP_DZ
    res = ik.solve(q, beside_leg)
    assert ik.robot_collisions(q, res.q)


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
    assert ik.model.actuator_gainprm[mujoco.mj_name2id(ik.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_left"), 0] >= 200
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


@pytest.mark.parametrize("yaw", [2.5, -2.5, 0.75, -0.75])
def test_planner_flags_upper_arm_on_the_carriage_at_any_yaw(ik, yaw):
    """The shoulder limit is measured at yaw 0; the box carriage is closer at other yaws, so the
    planner's collision check covers upper-arm-vs-carriage clearance at the pose's actual yaw."""
    q = home_qpos(ik)
    assert ("arm_upper", "arm_carriage") not in ik.robot_collisions(q, [0.0, 0.0, ik.limits[2][1], 0.0, 0.0])
    assert ("arm_upper", "arm_carriage") in ik.robot_collisions(q, [0.0, yaw, ik.limits[2][1], 0.0, 0.0])


def test_work_envelope_stops_at_the_planner_shoulder_limit(ik):
    model = load_farm_model()
    assert compute_work_envelope(model) == compute_work_envelope(model, shoulder_max=float(ik.limits[2][1]))
    assert compute_work_envelope(model) != compute_work_envelope(model, shoulder_max=2.4)


def test_rl_policy_observes_the_original_tip_point(ik):
    """The harvest gripper moved gripper_tip 4 cm out; the RL balance policy keeps observing the old
    point (site policy_tip, 0.065 m from the wrist pivot) so trained policies see their old tip_rel."""
    from atharv_nandi.core.rl_balance_policy import RLBalanceController, policy_tip_site_id
    m = ik.model
    sid = policy_tip_site_id(m)
    assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, sid) == "policy_tip"
    assert m.site_pos[sid] == pytest.approx([0.065, 0.0, 0.0])
    assert m.site_bodyid[sid] == m.site_bodyid[ik.tip_site_id]
    assert RLBalanceController(model=m, load_policy=False).tip_site_id == sid
