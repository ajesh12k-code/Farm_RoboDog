"""Pure pose-to-goal navigation for Atharv Nandi missions (no ROS, no MuJoCo).

Outputs a normalised /nandi/cmd_vel (vx, yaw) for the tripod gait (no strafing, so a single
controller works for any heading):
    heading error e = wrap(atan2(goal - pose) - yaw)
    yaw command    = clamp(K_HEADING * e, -1, 1)
    forward command= 0 while |e| > TURN_IN_PLACE_RAD (turn on the spot first), else
                     clamp(K_DIST * d, VX_MIN, VX_MAX) * cos(e)
Arrival: horizontal distance <= ARRIVE_TOL_M (0.15 m).

CmdVelHold turns a stream of desired commands into what to publish at 10 Hz: a non-zero command is
sent with hold=1 every tick (well inside locomotion's 0.5 s dead-man), and a stop is sent as the
all-zero release (linear.z = 0) for a few ticks, then nothing.
"""
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

ARRIVE_TOL_M = 0.15
K_HEADING = 1.6           # yaw command per rad of heading error (P gain)
K_DIST = 1.2              # forward command per metre to go
VX_MAX = 1.0
VX_MIN = 0.35             # below this the tripod stride is too short to make progress on soil
TURN_IN_PLACE_RAD = 0.6   # rotate on the spot until the goal is within this of the nose
CMD_RATE_HZ = 10.0
RELEASE_TICKS = 3         # publish the all-zero release this many times when stopping


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass(frozen=True)
class NavCommand:
    vx: float
    yaw: float
    arrived: bool
    distance: float
    heading_error: float

    @property
    def is_zero(self) -> bool:
        return abs(self.vx) < 1e-6 and abs(self.yaw) < 1e-6


def goto(pose: Sequence[float], goal: Sequence[float], tol: float = ARRIVE_TOL_M,
         k_heading: float = K_HEADING, vx_max: float = VX_MAX) -> NavCommand:
    """pose = (x, y, yaw_rad), goal = (x, y). Pure function."""
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    dx, dy = float(goal[0]) - x, float(goal[1]) - y
    d = math.hypot(dx, dy)
    if d <= tol:
        return NavCommand(0.0, 0.0, True, d, 0.0)
    e = wrap_angle(math.atan2(dy, dx) - th)
    yaw = clamp(k_heading * e, -1.0, 1.0)
    if abs(e) > TURN_IN_PLACE_RAD:
        vx = 0.0
    else:
        vx = clamp(K_DIST * d, VX_MIN, vx_max) * math.cos(e)
    return NavCommand(vx, yaw, False, d, e)


def turn_to(yaw_now: float, yaw_goal: float, tol_rad: float = 0.1, k_heading: float = K_HEADING) -> NavCommand:
    """Rotate in place to a heading (used to face along the row before a pick)."""
    e = wrap_angle(yaw_goal - yaw_now)
    if abs(e) <= tol_rad:
        return NavCommand(0.0, 0.0, True, 0.0, e)
    return NavCommand(0.0, clamp(k_heading * e, -1.0, 1.0), False, 0.0, e)


class CmdVelHold:
    """What to publish on /nandi/cmd_vel at each 10 Hz tick.

    tick(desired) -> (vx, vy, yaw, hold_flag) or None (publish nothing).
    """

    def __init__(self, release_ticks: int = RELEASE_TICKS):
        self.release_ticks = release_ticks
        self._release_left = 0

    def tick(self, desired: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
        if desired is not None and (abs(desired[0]) > 1e-3 or abs(desired[1]) > 1e-3):
            self._release_left = self.release_ticks
            return (clamp(desired[0], -1, 1), 0.0, clamp(desired[1], -1, 1), 1.0)
        if self._release_left > 0:
            self._release_left -= 1
            return (0.0, 0.0, 0.0, 0.0)
        return None

    def force_release(self):
        """Next ticks publish the zero release even if nothing was being held (abort / e-stop)."""
        self._release_left = self.release_ticks


def unicycle_step(pose: Sequence[float], vx: float, yaw: float, dt: float,
                  v_max: float = 0.25, w_max: float = 0.6) -> Tuple[float, float, float]:
    """Kinematic toy model of the tripod gait (tests / planning): v = v_max vx, w = w_max yaw."""
    x, y, th = pose
    v, w = v_max * vx, w_max * yaw
    return (x + v * math.cos(th) * dt, y + v * math.sin(th) * dt, wrap_angle(th + w * dt))
