"""Pure (ROS-free, MuJoCo-free) locomotion logic for the Atharv Nandi locomotion service.

Everything here is plain Python so it can be unit-tested on CI without ROS 2:
    - cmd_vel interpretation per the /nandi/cmd_vel contract
    - hold dead-man (0.5 s) with an injected clock
    - gait_params clamping
    - trunk roll/pitch from the free-joint quaternion
    - truthful locomotion mode selection (RL only when preferred AND a policy is loaded)
    - /nandi/locomotion_status JSON
    - LocomotionSafetyGate: speed cap from config/safety.yaml, cmd_vel refused while e-stopped,
      hold-position watchdog when /nandi/sim_qpos goes stale
"""
import json
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from atharv_nandi.core.safety_logic import SpeedCap, StaleMonitor, load_safety_config

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


def locomotion_status_json(selector: LocomotionModeSelector, params: GaitParams, walking: bool,
                           safety: Optional[dict] = None, stop_seq: int = 0) -> str:
    """stop_seq counts /nandi/stop_motion calls since the node started, so a node that republishes a
    held cmd_vel (the mission executor) can see that someone else stopped the robot."""
    out = {
        "mode": selector.mode,
        "policy_loaded": selector.policy_loaded,
        "policy_path": selector.policy_path,
        "gait_params": [round(v, 4) for v in params.as_list()],
        "walking": bool(walking),
        "stop_seq": int(stop_seq),
    }
    if safety is not None:
        out["safety"] = safety
    return json.dumps(out)


# --- safety gate ---------------------------------------------------------------------------
QPOS_TOPIC = "/nandi/sim_qpos"


class LocomotionSafetyGate:
    """What locomotion lets through: capped commands, nothing while e-stopped, hold on stale qpos.

    Time is injected (seconds, monotonic). The e-stop state comes from /nandi/safety_status.
    """

    def __init__(self, speed_cap: Optional[SpeedCap] = None, stale_s: Optional[float] = None,
                 cfg: Optional[dict] = None):
        if speed_cap is None or stale_s is None:
            cfg = cfg or load_safety_config()
        self.speed_cap = speed_cap or SpeedCap.from_config(cfg)
        self.stale = StaleMonitor(stale_s if stale_s is not None else cfg["stale"]["max_age_s"])
        self.estopped = False
        self.reason: Optional[str] = None
        self.refused = 0

    def set_estop(self, estopped: bool, reason: Optional[str] = None) -> bool:
        """Returns True on the rising edge (the caller must stop the gait now)."""
        rising = bool(estopped) and not self.estopped
        self.estopped, self.reason = bool(estopped), (reason if estopped else None)
        return rising

    def filter_cmd(self, cmd: MotionCommand) -> Optional[MotionCommand]:
        """Capped command, or None if refused (e-stopped). A zero command always passes."""
        if self.estopped and not cmd.is_zero:
            self.refused += 1
            return None
        vx, vy, yaw = self.speed_cap.clamp_cmd(cmd.vx, cmd.vy, cmd.yaw)
        k = self.cmd_scale
        return MotionCommand(vx * k, vy * k, yaw * k, cmd.hold)

    @property
    def cmd_scale(self) -> float:
        """Strides are speed_scale x cmd: a cap below the gait's own minimum speed_scale (0.2) is
        enforced by scaling cmd_vel by cap / 0.2 on top of speed_scale = 0.2."""
        lo = GAIT_PARAM_LIMITS[0][0]
        cap = self.speed_cap.max_speed_scale
        return 1.0 if cap >= lo else cap / lo

    def clamp_params(self, params: GaitParams) -> GaitParams:
        p = params.as_list()
        p[0] = max(GAIT_PARAM_LIMITS[0][0], self.speed_cap.clamp_speed_scale(p[0]))
        return GaitParams(*p)

    def qpos_seen(self, now: float):
        self.stale.touch(QPOS_TOPIC, now)

    def hold_position(self, now: float) -> bool:
        """True while /nandi/sim_qpos is older than the stale limit (after it was first seen)."""
        return self.stale.is_stale(QPOS_TOPIC, now)

    def status(self, now: float) -> dict:
        return {
            "estopped": self.estopped,
            "speed_cap": self.speed_cap.max_speed_scale,
            "holding_stale_qpos": self.hold_position(now),
            "refused_cmds": self.refused,
        }


class MotionInhibitGate:
    """WP6 ``motion_inhibit`` (``bus.values["motion_inhibit"]`` / ``/nandi/motion_inhibit``) for the gait owner
    (``sim_driver``, ``locomotion_service``): ``stability.stance`` (stop into a 6-foot stance: the ordered wind
    response, ORCHARD-MVP-SPEC 2.3 rule 4), ``stability.walk`` / ``stability.stop`` / ``safety_fields.stop`` (no
    walking) halt the gait and refuse new walking commands while set; ``safety_fields.speed_scale`` (warning field:
    0.3) scales the command. A payload older than ``max_age_s`` (when time is given) or none at all inhibits
    nothing: the hooks are optional, the e-stop latch is the enforced path."""

    def __init__(self, max_age_s: float = 0.5):
        self.max_age_s = float(max_age_s)
        self.halt = False
        self.scale = 1.0
        self.reasons: Tuple[str, ...] = ()
        self._seen: Optional[float] = None

    def update(self, mi, now: Optional[float] = None) -> bool:
        """Feed one payload (dict or None). Returns True on the rising edge of ``halt`` (stop the gait now)."""
        was = self.halt
        if now is not None and isinstance(mi, dict):
            self._seen = float(now)
        st = (mi or {}).get("stability") if isinstance(mi, dict) else None
        fs = (mi or {}).get("safety_fields") if isinstance(mi, dict) else None
        st = st if isinstance(st, dict) else {}
        fs = fs if isinstance(fs, dict) else {}
        why = []
        if st.get("stance"):
            why.append("stance")
        if st.get("walk") or st.get("stop"):
            why += [str(r) for r in (st.get("reasons") or ["stability"])]
        if fs.get("stop"):
            why += [str(r) for r in (fs.get("reasons") or ["protective"])]
        self.halt = bool(why)
        self.reasons = tuple(dict.fromkeys(why))
        sc = fs.get("speed_scale", 1.0)
        self.scale = clamp(float(sc), 0.0, 1.0) if isinstance(sc, (int, float)) and math.isfinite(sc) else 1.0
        return self.halt and not was

    def expire(self, now: float) -> bool:
        """Drop a stale payload (``locomotion_service``: the topic went silent). Returns True if it was dropped."""
        if self._seen is not None and now - self._seen > self.max_age_s:
            self._seen = None
            self.halt, self.scale, self.reasons = False, 1.0, ()
            return True
        return False

    def filter_cmd(self, cmd: MotionCommand) -> Optional[MotionCommand]:
        """The command scaled by ``speed_scale``, or None (refused) while halted. A zero command always passes."""
        if cmd.is_zero:
            return cmd
        if self.halt:
            return None
        k = self.scale
        return MotionCommand(cmd.vx * k, cmd.vy * k, cmd.yaw * k, cmd.hold)
