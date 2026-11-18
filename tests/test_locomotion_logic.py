import json
import math

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import rl_balance_policy
from atharv_nandi.core.farm_scene import crop_qpos_slice, load_farm_model
from atharv_nandi.core.locomotion_logic import (
    MSG_MODE_RL, MSG_MODE_TRIPOD, MSG_NO_POLICY, GaitParams, HoldDeadman, LocomotionModeSelector,
    locomotion_status_json, parse_cmd_vel, quat_to_roll_pitch, quat_to_yaw,
)
from atharv_nandi.core.tripod_gait import TripodGaitGenerator


def quat_axis_angle(axis, angle):
    s = math.sin(angle / 2.0)
    return (math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s)


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


# --- cmd_vel ------------------------------------------------------------------------------
def test_parse_cmd_vel_contract_fields():
    c = parse_cmd_vel(linear_x=0.5, linear_y=-0.25, linear_z=1.0, angular_z=0.75)
    assert (c.vx, c.vy, c.yaw, c.hold) == (0.5, -0.25, 0.75, True)
    c = parse_cmd_vel(linear_x=3.0, linear_y=-9.0, linear_z=0.0, angular_z=-2.0)
    assert (c.vx, c.vy, c.yaw, c.hold) == (1.0, -1.0, -1.0, False)
    c = parse_cmd_vel(float("nan"), 0.0, float("nan"), float("inf"))
    assert (c.vx, c.yaw, c.hold) == (0.0, 0.0, False)
    assert parse_cmd_vel(0.0, 0.0, 0.0, 0.0).is_zero
    assert not parse_cmd_vel(0.0, 0.0, 0.0, 0.3).is_zero  # pure yaw is a real command


# --- dead-man -----------------------------------------------------------------------------
def test_deadman_expires_once_after_timeout():
    dm = HoldDeadman(0.5)
    hold = parse_cmd_vel(1.0, 0.0, 1.0, 0.0)
    dm.feed(hold, now=10.0)
    assert dm.armed
    assert not dm.expired(10.4)
    assert not dm.expired(10.5)
    assert dm.expired(10.51)
    assert not dm.expired(10.6) and not dm.armed  # fires exactly once


def test_deadman_rearmed_by_repeated_holds_and_disarmed_by_others():
    dm = HoldDeadman(0.5)
    hold = parse_cmd_vel(0.0, 0.0, 1.0, 1.0)
    for t in (0.0, 0.2, 0.4, 0.6):  # 5 Hz sender
        dm.feed(hold, t)
        assert not dm.expired(t + 0.1)
    assert not dm.expired(1.05)
    assert dm.expired(1.11)
    dm.feed(hold, 2.0)
    dm.feed(parse_cmd_vel(1.0, 0.0, 0.0, 0.0), 2.1)  # single step disarms
    assert not dm.armed and not dm.expired(5.0)
    dm.feed(hold, 3.0)
    dm.feed(parse_cmd_vel(0.0, 0.0, 1.0, 0.0), 3.1)  # zero "hold" == release, disarms
    assert not dm.armed


def test_deadman_with_gait_releases_stale_hold_and_finishes_cycle():
    """Simulated node loop at 50 Hz with an injected clock: sender publishes hold at 5 Hz for 1 s then dies."""
    dt = 0.02
    gait, dm = TripodGaitGenerator(dt), HoldDeadman(0.5)
    hold = parse_cmd_vel(1.0, 0.0, 1.0, 0.0)
    released_at, stopped_at = None, None
    for k in range(200):
        now = k * dt
        if k % 10 == 0 and k <= 40:  # last command at t = 0.8 s
            dm.feed(hold, now)
            gait.set_command(hold.vx, hold.vy, hold.yaw, hold.hold)
        if dm.expired(now):
            gait.release()
            released_at = now
        gait.step()
        if released_at is not None and stopped_at is None and not gait.is_walking:
            stopped_at = now
    last_cmd = 0.8
    assert released_at is not None and last_cmd + 0.5 - 1e-9 <= released_at <= last_cmd + 0.5 + dt + 1e-9
    cycle = 1.0 / gait.params.freq_hz
    assert stopped_at is not None and released_at < stopped_at <= released_at + cycle + 0.2
    print(f"dead-man released at {released_at:.2f}s, legs still at {stopped_at:.2f}s")


# --- gait params --------------------------------------------------------------------------
def test_gait_params_clamped_and_partial_updates_keep_current():
    p = GaitParams.from_sequence([10.0, 0.0, 0.1, -1.0])
    assert p.as_list() == [1.5, 0.5, 0.8, 0.0]
    p = GaitParams.from_sequence([0.0, 9.0, 99.0, 5.0])
    assert p.as_list() == [0.2, 2.0, 3.0, 2.0]
    cur = GaitParams(1.2, 1.5, 2.0, 0.5)
    assert GaitParams.from_sequence([0.8], cur).as_list() == [0.8, 1.5, 2.0, 0.5]
    assert GaitParams.from_sequence([float("nan"), 1.0], cur).as_list() == [1.2, 1.0, 2.0, 0.5]
    assert GaitParams().as_list() == [1.0, 1.0, 1.8, 1.0]


# --- attitude -----------------------------------------------------------------------------
def test_roll_pitch_signs_and_yaw_invariance():
    roll, pitch = quat_to_roll_pitch(*quat_axis_angle((0, 1, 0), 0.2))
    assert abs(roll) < 1e-9 and pitch == pytest.approx(0.2)
    roll, pitch = quat_to_roll_pitch(*quat_axis_angle((1, 0, 0), -0.15))
    assert roll == pytest.approx(-0.15) and abs(pitch) < 1e-9
    # heading must not leak into roll/pitch: R = Rz(yaw) Ry(pitch) Rx(roll)
    q = quat_mul(quat_axis_angle((0, 0, 1), 2.5),
                 quat_mul(quat_axis_angle((0, 1, 0), 0.1), quat_axis_angle((1, 0, 0), 0.05)))
    roll, pitch = quat_to_roll_pitch(*q)
    assert roll == pytest.approx(0.05) and pitch == pytest.approx(0.1)
    assert quat_to_yaw(*q) == pytest.approx(2.5)


def test_pitch_positive_is_nose_down_in_mujoco():
    m = load_farm_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[3:7] = quat_axis_angle((0, 1, 0), 0.1)
    mujoco.mj_kinematics(m, d)
    z = {n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)][2] for n in ("FR_hip", "RR_hip")}
    assert quat_to_roll_pitch(*d.qpos[3:7])[1] > 0 and z["FR_hip"] < z["RR_hip"]
    d.qpos[3:7] = quat_axis_angle((1, 0, 0), 0.1)
    mujoco.mj_kinematics(m, d)
    zl = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ML_hip")][2]
    zr = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "MR_hip")][2]
    assert quat_to_roll_pitch(*d.qpos[3:7])[0] > 0 and zl > zr  # roll > 0: left side up


# --- truthful mode + status ---------------------------------------------------------------
def test_mode_without_policy_stays_tripod():
    sel = LocomotionModeSelector(policy_loaded=False, policy_path="/ignored.zip")
    assert sel.toggle() == (False, MSG_NO_POLICY)
    assert MSG_NO_POLICY == "No RL policy loaded - using Tripod gait"
    assert sel.mode == "tripod" and sel.policy_path is None


def test_mode_with_policy_toggles():
    sel = LocomotionModeSelector(policy_loaded=True, policy_path="/p.zip")
    assert sel.mode == "tripod"
    assert sel.toggle() == (True, MSG_MODE_RL) and sel.mode == "rl"
    assert sel.toggle() == (True, MSG_MODE_TRIPOD) and sel.mode == "tripod"
    assert (MSG_MODE_RL, MSG_MODE_TRIPOD) == ("Mode: RL Balance Policy", "Mode: Tripod Gait")


def test_locomotion_status_json_contract():
    sel = LocomotionModeSelector(False)
    data = json.loads(locomotion_status_json(sel, GaitParams(), walking=True))
    assert data == {"mode": "tripod", "policy_loaded": False, "policy_path": None,
                    "gait_params": [1.0, 1.0, 1.8, 1.0], "walking": True, "stop_seq": 0}
    assert json.loads(locomotion_status_json(sel, GaitParams(), False, stop_seq=3))["stop_seq"] == 3


# --- RL controller (no SB3 needed) --------------------------------------------------------
def test_rl_controller_without_sb3_is_truthfully_unavailable(monkeypatch):
    monkeypatch.setattr(rl_balance_policy, "HAS_SB3", False)
    rl = rl_balance_policy.RLBalanceController(0.02)
    assert rl.policy is None and rl.policy_path is None and rl.load_error
    assert not LocomotionModeSelector(rl.policy is not None, rl.policy_path).toggle()[0]
    with pytest.raises(RuntimeError):
        rl.predict(rl.data.qpos.copy(), rl.data.qpos.copy())


def test_rl_obs_uses_farm_model_and_robot_prefix_only():
    m = load_farm_model()
    rl = rl_balance_policy.RLBalanceController(0.02, model=m, load_policy=False)
    assert rl.model.nq == m.nq and m.nq > 32
    curr = rl.data.qpos.copy()
    prev = curr.copy()
    prev[0] -= 0.01   # trunk moved +0.01 m in x over dt = 0.02 s -> 0.5 m/s
    prev[15] -= 0.004  # FR_thigh moved +0.004 rad -> 0.2 rad/s
    obs = rl.build_obs(prev, curr)
    assert obs.shape == (rl_balance_policy.OBS_DIM,) and obs.dtype == np.float32
    assert obs[0] == pytest.approx(curr[2])        # trunk z
    assert obs[5] == pytest.approx(0.5, abs=1e-5)  # lin_vel x
    assert np.allclose(obs[8:11], 0.0, atol=1e-6)   # no rotation
    # leg velocity block starts after z,quat,lin,ang,tip,hold,arm_err,arm_vel,leg_err = 1+4+3+3+3+1+5+5+18
    assert obs[43 + 1] == pytest.approx(0.2, abs=1e-5)
    # moving crops must not change the observation
    curr2, prev2 = curr.copy(), prev.copy()
    for arr in (curr2, prev2):
        arr[crop_qpos_slice(1)][0:3] += 0.3
    assert np.allclose(rl.build_obs(prev2, curr2), obs)


# --- safety gate -----------------------------------------------------------------------------
from atharv_nandi.core.locomotion_logic import LocomotionSafetyGate  # noqa: E402
from atharv_nandi.core.safety_logic import SpeedCap  # noqa: E402


def test_safety_gate_refuses_cmd_while_estopped_but_passes_zero():
    gate = LocomotionSafetyGate(SpeedCap(1.0, 1.0), 0.5)
    assert gate.filter_cmd(parse_cmd_vel(0.5, 0, 1, 0)) is not None
    assert gate.set_estop(True, "operator") is True
    assert gate.set_estop(True, "operator") is False
    assert gate.filter_cmd(parse_cmd_vel(0.5, 0, 1, 0)) is None
    assert gate.filter_cmd(parse_cmd_vel(0, 0, 0, 0)) is not None
    assert gate.refused == 1
    gate.set_estop(False)
    assert gate.filter_cmd(parse_cmd_vel(0.5, 0, 1, 0)).vx == 0.5


def test_safety_gate_caps_cmd_and_speed_scale():
    gate = LocomotionSafetyGate(SpeedCap(max_speed_scale=1.0, max_cmd=0.6), 0.5)
    cmd = gate.filter_cmd(parse_cmd_vel(1.0, 1.0, 1.0, -1.0))
    assert math.hypot(cmd.vx, cmd.vy) == pytest.approx(0.6) and cmd.yaw == pytest.approx(-0.6) and cmd.hold
    p = gate.clamp_params(GaitParams.from_sequence([1.5, 1.0, 1.8, 1.0]))
    assert p.speed_scale == 1.0 and p.freq_hz == 1.8
    assert gate.clamp_params(GaitParams.from_sequence([0.4])).speed_scale == 0.4


def test_safety_gate_enforces_a_cap_below_the_gait_minimum():
    gate = LocomotionSafetyGate(SpeedCap(max_speed_scale=0.1, max_cmd=1.0), 0.5)
    p = gate.clamp_params(GaitParams.from_sequence([1.0]))
    assert p.speed_scale == pytest.approx(0.2)                # the gait's own floor
    cmd = gate.filter_cmd(parse_cmd_vel(1.0, 0.0, 1.0, 0.4))
    # strides = speed_scale x cmd: 0.2 x 0.5 == the configured 0.1
    assert p.speed_scale * cmd.vx == pytest.approx(0.1) and cmd.yaw == pytest.approx(0.2)
    assert LocomotionSafetyGate(SpeedCap(1.0, 1.0), 0.5).cmd_scale == 1.0


def test_safety_gate_holds_on_stale_qpos():
    gate = LocomotionSafetyGate(SpeedCap(), 0.5)
    assert not gate.hold_position(0.0)       # never seen: startup is not a hold
    gate.qpos_seen(1.0)
    assert not gate.hold_position(1.4)
    assert gate.hold_position(1.6)
    gate.qpos_seen(1.7)
    assert not gate.hold_position(1.75)
    assert gate.status(1.75)["holding_stale_qpos"] is False


def test_locomotion_status_includes_safety_when_given():
    sel = LocomotionModeSelector(False)
    d = json.loads(locomotion_status_json(sel, GaitParams(), False, {"estopped": True}))
    assert d["safety"] == {"estopped": True}
    assert "safety" not in json.loads(locomotion_status_json(sel, GaitParams(), False))
