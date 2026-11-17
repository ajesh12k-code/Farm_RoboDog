"""Pure (ROS-free, MuJoCo-free) locomotion logic for the Atharv Nandi locomotion service.

Everything here is plain Python so it can be unit-tested on CI without ROS 2:
    - cmd_vel interpretation per the /nandi/cmd_vel contract
    - hold dead-man (0.5 s) with an injected clock
    - gait_params clamping
    - trunk roll/pitch from the free-joint quaternion
    - truthful locomotion mode selection (RL only when preferred AND a policy is loaded)
    - /nandi/locomotion_status JSON
"""
import json
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# --- gait_params: [speed_scale, step_height_scale, freq_hz, balance_gain] -------------------
GAIT_PARAM_DEFAULTS = (1.0, 1.0, 1.8, 1.0)
GAIT_PARAM_LIMITS = ((0.2, 1.5), (0.5, 2.0), (0.8, 3.0), (0.0, 2.0))

HOLD_TIMEOUT_S = 0.5  # dead-man: a hold is released if no cmd_vel arrives for this long
CMD_EPS = 1e-3

MSG_NO_POLICY = "No RL policy loaded - using Tripod gait"
MSG_MODE_RL = "Mode: RL Balance Policy"
MSG_MODE_TRIPOD = "Mode: Tripod Gait"


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _finite(v, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


@dataclass(frozen=True)
class GaitParams:
    speed_scale: float = GAIT_PARAM_DEFAULTS[0]
    step_height_scale: float = GAIT_PARAM_DEFAULTS[1]
    freq_hz: float = GAIT_PARAM_DEFAULTS[2]
    balance_gain: float = GAIT_PARAM_DEFAULTS[3]

    @classmethod
    def from_sequence(cls, values: Sequence[float], current: Optional["GaitParams"] = None) -> "GaitParams":
        """Clamp a /nandi/gait_params array. Missing or non-finite entries keep the current value."""
        base = (current or cls()).as_list()
        out = []
        for i, (lo, hi) in enumerate(GAIT_PARAM_LIMITS):
            v = _finite(values[i], base[i]) if i < len(values) else base[i]
            out.append(clamp(v, lo, hi))
        return cls(*out)

    def as_list(self) -> list:
        return [self.speed_scale, self.step_height_scale, self.freq_hz, self.balance_gain]


# --- cmd_vel ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class MotionCommand:
    vx: float   # forward +, [-1, 1]
    vy: float   # left +, [-1, 1]
    yaw: float  # CCW (turn left) +, [-1, 1]
    hold: bool  # True: keep walking while commands keep arriving; False: one gait cycle

    @property
    def is_zero(self) -> bool:
        return abs(self.vx) < CMD_EPS and abs(self.vy) < CMD_EPS and abs(self.yaw) < CMD_EPS


def parse_cmd_vel(linear_x: float, linear_y: float, linear_z: float, angular_z: float) -> MotionCommand:
    """Map Twist fields to a MotionCommand: linear.x=vx, linear.y=vy, angular.z=yaw, linear.z>=0.5 => hold."""
    return MotionCommand(
        vx=clamp(_finite(linear_x, 0.0), -1.0, 1.0),
        vy=clamp(_finite(linear_y, 0.0), -1.0, 1.0),
        yaw=clamp(_finite(angular_z, 0.0), -1.0, 1.0),
        hold=_finite(linear_z, 0.0) >= 0.5,
    )


class HoldDeadman:
    """Releases a held command when cmd_vel stops arriving. Time is injected (seconds, monotonic)."""

    def __init__(self, timeout_s: float = HOLD_TIMEOUT_S):
        self.timeout_s = timeout_s
        self._last_hold_t: Optional[float] = None

    @property
    def armed(self) -> bool:
        return self._last_hold_t is not None

    def feed(self, cmd: MotionCommand, now: float):
        """Call for every cmd_vel. A non-zero hold (re)arms the timer; anything else disarms it."""
        self._last_hold_t = now if (cmd.hold and not cmd.is_zero) else None

    def disarm(self):
        self._last_hold_t = None

    def expired(self, now: float) -> bool:
        """True exactly once when an armed hold has gone stale; the caller must then release."""
        if self._last_hold_t is None or (now - self._last_hold_t) <= self.timeout_s:
            return False
        self._last_hold_t = None
        return True


# --- attitude ------------------------------------------------------------------------------
def quat_to_roll_pitch(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float]:
    """ZYX Euler roll/pitch (rad) of the trunk; both are independent of heading (yaw).

    Conventions (MuJoCo world, z up, quaternion w x y z):
        roll  > 0 : rotation about +x -> body +y (LEFT side) goes UP, right side goes down
        pitch > 0 : rotation about +y -> body +x (NOSE) goes DOWN
    """
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = math.asin(clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    return roll, pitch


def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


# --- mode selection ------------------------------------------------------------------------
class LocomotionModeSelector:
    """Truthful mode: 'rl' only when RL is preferred AND a policy is actually loaded."""

    def __init__(self, policy_loaded: bool, policy_path: Optional[str] = None):
        self.policy_loaded = bool(policy_loaded)
        self.policy_path = policy_path if self.policy_loaded else None
        self.prefer_rl = False  # start in tripod; the operator opts in to RL

    @property
    def mode(self) -> str:
        return "rl" if (self.prefer_rl and self.policy_loaded) else "tripod"

    def toggle(self) -> Tuple[bool, str]:
        """Handler body for /nandi/toggle_locomotion_mode -> (success, message)."""
        if not self.policy_loaded:
            self.prefer_rl = False
            return False, MSG_NO_POLICY
        self.prefer_rl = not self.prefer_rl
        return True, (MSG_MODE_RL if self.mode == "rl" else MSG_MODE_TRIPOD)


def locomotion_status_json(selector: LocomotionModeSelector, params: GaitParams, walking: bool) -> str:
    return json.dumps({
        "mode": selector.mode,
        "policy_loaded": selector.policy_loaded,
        "policy_path": selector.policy_path,
        "gait_params": [round(v, 4) for v in params.as_list()],
        "walking": bool(walking),
    })
