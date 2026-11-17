import math

import mujoco
import numpy as np
import pytest

from controllers.farm_scene import load_farm_model
from controllers.locomotion_logic import GaitParams, quat_to_roll_pitch, quat_to_yaw
from controllers.tripod_gait import (
    FOOT_Y, HIP_X, HOME_CALF, LATERAL_ARM, LEG_NAMES, LEG_REACH, MAX_BALANCE_EXT, SIDE, TWO_PI, TripodGaitGenerator,
    balance_corrections, balance_extensions, home_leg_targets, leg_extension_to_joints, leg_strides,
)

LEFT = [i for i, s in enumerate(SIDE) if s > 0]
RIGHT = [i for i, s in enumerate(SIDE) if s < 0]
FRONT, MID, REAR = (0, 1), (2, 3), (4, 5)


def quat_axis_angle(axis, angle):
    s = math.sin(angle / 2.0)
    return (math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s)


# --- kinematic sign checks against the MJCF -------------------------------------------------
@pytest.fixture(scope="module")
def fk():
    m = load_farm_model()
    d = mujoco.MjData(m)
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{n}_foot") for n in LEG_NAMES]
    thighs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{n}_thigh") for n in LEG_NAMES]

    def run(legs, quat=(1.0, 0.0, 0.0, 0.0)):
        """Return (feet in trunk frame, thigh origins in trunk frame, feet in world) for 18 leg angles."""
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[3:7] = quat
        d.qpos[14:32] = legs
        mujoco.mj_kinematics(m, d)
        rot, pos = d.xmat[trunk].reshape(3, 3), d.xpos[trunk]
        feet_t = np.array([rot.T @ (d.geom_xpos[g] - pos) for g in feet])
        thighs_t = np.array([rot.T @ (d.xpos[b] - pos) for b in thighs])
        feet_w = np.array([d.geom_xpos[g].copy() for g in feet])
        return feet_t, thighs_t, feet_w

    mujoco.mj_resetDataKeyframe(m, d, 0)
    run.home_qpos_legs = d.qpos[14:32].copy()
    return run


def test_home_targets_match_keyframe(fk):
    assert np.allclose(home_leg_targets(), fk.home_qpos_legs)


def test_geometry_constants_match_mjcf(fk):
    feet, thighs, _ = fk(home_leg_targets())
    for i in range(6):
        assert np.linalg.norm(feet[i] - thighs[i]) == pytest.approx(LEG_REACH, abs=1e-6)
        assert feet[i][0] == pytest.approx(HIP_X[i], abs=1e-6)  # foot straight under the hip
        assert feet[i][1] == pytest.approx(FOOT_Y[i], abs=1e-6)


def test_positive_thigh_moves_foot_back_and_stride_mapping_is_exact(fk):
    home = home_leg_targets()
    feet0, _, _ = fk(home)
    for i in range(6):
        legs = list(home)
        legs[3 * i + 1] += 0.1
        assert fk(legs)[0][i][0] < feet0[i][0] - 0.02  # +thigh => foot -x (backward)
        legs = list(home)
        legs[3 * i + 1] -= math.asin(0.05 / LEG_REACH)  # the generator's stride mapping
        assert fk(legs)[0][i][0] - feet0[i][0] == pytest.approx(0.05, abs=1e-6)


def test_positive_hip_moves_foot_left_on_both_sides(fk):
    home = home_leg_targets()
    feet0, _, _ = fk(home)
    for i in range(6):
        legs = list(home)
        legs[3 * i] += 0.05
        dy = fk(legs)[0][i][1] - feet0[i][1]
        assert dy == pytest.approx(LATERAL_ARM * 0.05, rel=0.05)  # +hip => foot +y, left AND right legs


def test_leg_extension_lengthens_along_leg_axis(fk):
    home = home_leg_targets()
    feet0, thighs, _ = fk(home)
    d_thigh, d_calf = leg_extension_to_joints(0.02)
    assert d_calf > 0 and d_thigh < 0
    for i in range(6):
        legs = list(home)
        legs[3 * i + 1] += d_thigh
        legs[3 * i + 2] += d_calf
        feet, _, _ = fk(legs)
        grow = np.linalg.norm(feet[i] - thighs[i]) - LEG_REACH
        assert grow == pytest.approx(0.02, abs=0.002)
        assert abs(feet[i][0] - feet0[i][0]) < 1e-6  # stays under the hip


# --- balance ------------------------------------------------------------------------------
def test_balance_nose_down_extends_front_retracts_rear():
    roll, pitch = quat_to_roll_pitch(*quat_axis_angle((0, 1, 0), 0.1))  # nose down
    ext = balance_extensions(roll, pitch, gain=1.0)
    assert all(ext[i] > 0.005 for i in FRONT)
    assert all(ext[i] < -0.005 for i in REAR)
    assert all(abs(ext[i]) < 1e-9 for i in MID)
    corr = balance_corrections(roll, pitch, gain=1.0)
    assert corr[3 * 0 + 2] > 0 and corr[3 * 4 + 2] < 0  # FR calf straightens, RR calf flexes
    assert all(corr[3 * i] == 0.0 for i in range(6))     # hips untouched


def test_balance_left_up_retracts_left_extends_right():
    roll, pitch = quat_to_roll_pitch(*quat_axis_angle((1, 0, 0), 0.1))  # left side up
    ext = balance_extensions(roll, pitch, gain=1.0)
    assert all(ext[i] < -0.005 for i in LEFT)
    assert all(ext[i] > 0.005 for i in RIGHT)


def test_balance_gain_scales_and_clamps():
    roll, pitch = 0.0, 0.05
    e1 = balance_extensions(roll, pitch, 1.0)
    e2 = balance_extensions(roll, pitch, 2.0)
    assert e2[0] == pytest.approx(2.0 * e1[0])
    assert balance_extensions(roll, pitch, 0.0) == [0.0] * 6
    big = balance_extensions(0.6, 0.6, 2.0)
    assert max(abs(e) for e in big) == pytest.approx(MAX_BALANCE_EXT)


def test_balance_correction_pushes_low_side_feet_down_in_world(fk):
    """Body pitched nose-down: the corrected front feet reach lower (rear higher) than uncorrected, so with
    the feet on the ground the front of the trunk is pushed up -> towards level."""
    q = quat_axis_angle((0, 1, 0), 0.1)
    roll, pitch = quat_to_roll_pitch(*q)
    home = home_leg_targets()
    _, _, w0 = fk(home, q)
    corrected = [h + c for h, c in zip(home, balance_corrections(roll, pitch, 1.0))]
    _, _, w1 = fk(corrected, q)
    ext = balance_extensions(roll, pitch, 1.0)
    for i in FRONT:
        assert w1[i][2] < w0[i][2] - 0.5 * ext[i]
    for i in REAR:
        assert w1[i][2] > w0[i][2] + 0.5 * abs(ext[i])


# --- strides / turning --------------------------------------------------------------------
def test_turn_left_is_differential_and_tangential():
    s = leg_strides(0.0, 0.0, 1.0)
    assert all(s[i][0] < 0 for i in LEFT)    # left legs push the body backward
    assert all(s[i][0] > 0 for i in RIGHT)   # right legs push it forward -> CCW
    assert all(s[i][1] > 0 for i in FRONT)   # front pushed left
    assert all(s[i][1] < 0 for i in REAR)    # rear pushed right
    assert all(abs(s[i][1]) < 1e-12 for i in MID)
    # the pushes are tangent to a CCW rotation about the trunk centre: push . (-y, x) > 0, push . (x, y) ~ 0
    for i in range(6):
        px, py = HIP_X[i], FOOT_Y[i]
        assert s[i][0] * -py + s[i][1] * px > 0
        assert abs(s[i][0] * px + s[i][1] * py) < 1e-9


def test_arc_right_side_strides_longer_when_turning_left_while_walking():
    s = leg_strides(1.0, 0.0, 0.5)
    for left, right in ((1, 0), (3, 2), (5, 4)):
        assert s[right][0] > s[left][0] > 0


def test_speed_scale_scales_stride_and_clamps():
    a, b = leg_strides(0.5, 0.5, 0.0, 1.0), leg_strides(0.5, 0.5, 0.0, 1.5)
    assert b[0][0] == pytest.approx(1.5 * a[0][0]) and b[0][1] == pytest.approx(1.5 * a[0][1])
    big = leg_strides(1.0, 1.0, 1.0, 1.5)
    assert max(abs(x) for x, _ in big) <= 0.095 + 1e-12


# --- generator behaviour ------------------------------------------------------------------
def _gait(**kw):
    return TripodGaitGenerator(0.02, GaitParams(balance_gain=0.0, **kw))


def test_idle_outputs_home():
    assert _gait().step() == pytest.approx(home_leg_targets())


def test_single_step_is_one_cycle_then_home():
    g = _gait()
    g.set_command(1.0, 0.0, 0.0, hold=False)
    steps_per_cycle = TWO_PI / (TWO_PI * 1.8 * 0.02)
    n = 0
    while g.is_active:
        g.step()
        n += 1
    assert n == math.ceil(steps_per_cycle - 1e-9)
    for _ in range(10):  # start/stop envelope ramps down
        out = g.step()
    assert not g.is_walking and out == pytest.approx(home_leg_targets())


def test_hold_release_finishes_current_cycle_and_stop_is_immediate():
    g = _gait()
    g.set_command(0.0, 0.0, 1.0, hold=True)  # pure yaw is a valid command
    for _ in range(100):
        g.step()
    assert g.is_active and g.is_held
    phase = g.gait_phase
    g.set_command(0.0, 0.0, 0.0, hold=False)  # release
    assert g.is_active and not g.is_held
    assert g.remaining_phase == pytest.approx(TWO_PI - phase)
    g.stop()
    assert not g.is_walking and g.step() == pytest.approx(home_leg_targets())


def test_freq_sets_phase_rate():
    g = _gait(freq_hz=3.0)
    g.set_command(1.0, 0.0, 0.0, hold=True)
    g.step()
    assert g.gait_phase == pytest.approx(TWO_PI * 3.0 * 0.02)


def test_step_height_scales_lift():
    def calf_drop(scale):
        g = _gait(step_height_scale=scale)
        g.set_command(1.0, 0.0, 0.0, hold=True)
        for _ in range(20):
            g.step()
        outs = [g.step() for _ in range(30)]  # > one cycle at 1.8 Hz
        return min(out[3 * i + 2] for out in outs for i in range(6)) - HOME_CALF
    assert calf_drop(2.0) == pytest.approx(2.0 * calf_drop(1.0))
    assert calf_drop(1.0) < -0.3


# --- closed loop in MuJoCo (no ROS): gait at 50 Hz, physics at 500 Hz -----------------------
CTRL_DT = 0.02


def _closed_loop(monkeypatch, vx, vy, yaw, seconds=4.0, settle_s=0.6):
    # Flat ground and only crop_item_1 (0.62, 0): it passes between the front feet, under the trunk.
    monkeypatch.setenv("NANDI_TERRAIN", "flat")
    monkeypatch.setenv("NANDI_CROPS", "1")
    m = load_farm_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    substeps = int(round(CTRL_DT / m.opt.timestep))
    assert substeps == 10
    gait = TripodGaitGenerator(CTRL_DT)  # default params incl. balance_gain = 1
    st = {"min_z": math.inf, "max_tilt": 0.0}

    def tick():
        roll, pitch = quat_to_roll_pitch(*d.qpos[3:7])
        d.ctrl[7:25] = gait.step(roll, pitch)
        for _ in range(substeps):
            mujoco.mj_step(m, d)
        qx, qy = d.qpos[4], d.qpos[5]
        st["min_z"] = min(st["min_z"], float(d.qpos[2]))
        st["max_tilt"] = max(st["max_tilt"], math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy))))))

    for _ in range(int(round(settle_s / CTRL_DT))):
        tick()
    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    yaw_prev, yaw_acc = quat_to_yaw(*d.qpos[3:7]), 0.0
    gait.set_command(vx, vy, yaw, hold=True)
    n = int(round(seconds / CTRL_DT))
    for _ in range(n):
        tick()
        yaw_now = quat_to_yaw(*d.qpos[3:7])
        yaw_acc += (yaw_now - yaw_prev + math.pi) % TWO_PI - math.pi
        yaw_prev = yaw_now
    t = n * CTRL_DT
    res = {
        "vx_mps": (float(d.qpos[0]) - x0) / t,
        "vy_mps": (float(d.qpos[1]) - y0) / t,
        "yaw_dps": math.degrees(yaw_acc) / t,
        "min_z": st["min_z"],
        "max_tilt_deg": st["max_tilt"],
    }
    print(f"cmd vx={vx} vy={vy} yaw={yaw}: " + ", ".join(f"{k}={v:.3f}" for k, v in res.items()))
    return res


def _assert_upright(res):
    assert res["min_z"] > 0.12 and res["max_tilt_deg"] < 50.0, res


def test_closed_loop_forward(monkeypatch):
    res = _closed_loop(monkeypatch, 1.0, 0.0, 0.0)
    _assert_upright(res)
    assert res["vx_mps"] > 0.05, res


def test_closed_loop_turn_left_ccw(monkeypatch):
    res = _closed_loop(monkeypatch, 0.0, 0.0, 1.0)
    _assert_upright(res)
    assert res["yaw_dps"] > 5.0, res


def test_closed_loop_turn_right_cw(monkeypatch):
    res = _closed_loop(monkeypatch, 0.0, 0.0, -1.0)
    _assert_upright(res)
    assert res["yaw_dps"] < -5.0, res


def test_closed_loop_strafe_left(monkeypatch):
    res = _closed_loop(monkeypatch, 0.0, 1.0, 0.0)
    _assert_upright(res)
    assert res["vy_mps"] > 0.03, res


def test_closed_loop_stand_still_stays_put(monkeypatch):
    res = _closed_loop(monkeypatch, 0.0, 0.0, 0.0, seconds=2.0)
    _assert_upright(res)
    assert abs(res["vx_mps"]) < 0.02 and abs(res["vy_mps"]) < 0.02, res
