"""Pure scoring / fall / real-time logic for the Atharv Nandi harvest game (no ROS, no MuJoCo).

sim_runner feeds this module plain qpos-derived numbers; everything here is unit-tested in
tests/test_game_status.py with hand-computed values.

Conventions:
    quaternions are MuJoCo order (w, x, y, z); positions are world-frame metres.
    Basket geometry (trunk frame) comes from atharv_nandi.xml:
        dorsal_basket body at z=+0.055, basket_floor half-thickness 0.006 -> floor top z=0.061.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# In-basket footprint (trunk frame), per the /nandi/game_status contract.
BASKET_HALF_X = 0.31
BASKET_HALF_Y = 0.11
BASKET_FLOOR_Z = 0.061       # top surface of basket_floor
# Contract "z between basket floor and +0.12": trunk-frame z = +0.12, just under the basket wall tops
# (0.125). A crop resting on the floor sits at 0.096; one on the gantry rail (top 0.145) or still in
# the gripper above the walls is not "in" the basket.
BASKET_TOP_Z = 0.12

# Fall detection.
FALL_TILT_DEG = 50.0
FALL_MIN_TRUNK_Z = 0.12


def quat_to_mat(quat: Sequence[float]) -> np.ndarray:
    """3x3 rotation matrix (body -> world) of a (w, x, y, z) quaternion; normalises first."""
    q = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.eye(3)
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def world_to_trunk(point: Sequence[float], trunk_pos: Sequence[float], trunk_quat: Sequence[float]) -> np.ndarray:
    """Express a world point in the trunk frame: R^T (p - t)."""
    rot = quat_to_mat(trunk_quat)
    return rot.T @ (np.asarray(point, dtype=np.float64) - np.asarray(trunk_pos, dtype=np.float64))


def in_basket(crop_pos: Sequence[float], trunk_pos: Sequence[float], trunk_quat: Sequence[float]) -> bool:
    """True if the crop centre lies inside the dorsal basket volume (trunk frame)."""
    x, y, z = world_to_trunk(crop_pos, trunk_pos, trunk_quat)
    return bool(abs(x) < BASKET_HALF_X and abs(y) < BASKET_HALF_Y
                and BASKET_FLOOR_Z <= z <= BASKET_TOP_Z)


def tilt_deg(trunk_quat: Sequence[float]) -> float:
    """Angle between the trunk's up axis and world +z, in degrees (0 = level, 180 = upside down)."""
    up_z = quat_to_mat(trunk_quat)[2, 2]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(up_z)))))


def is_fallen(trunk_pos: Sequence[float], trunk_quat: Sequence[float]) -> bool:
    return tilt_deg(trunk_quat) > FALL_TILT_DEG or float(trunk_pos[2]) < FALL_MIN_TRUNK_Z


class FallCounter:
    """Counts falls on the rising edge of the fallen condition."""

    def __init__(self):
        self.falls = 0
        self.fallen = False

    def update(self, fallen_now: bool) -> bool:
        """Feed the current condition; returns True exactly when a new fall is counted."""
        rising = fallen_now and not self.fallen
        if rising:
            self.falls += 1
        self.fallen = bool(fallen_now)
        return rising

    def reset(self):
        self.falls = 0
        self.fallen = False


class RtfMeter:
    """Smoothed real-time factor: sim seconds advanced per wall second, EMA over >= window_s windows."""

    def __init__(self, alpha: float = 0.3, window_s: float = 0.25):
        self.alpha = alpha
        self.window_s = window_s
        self.rtf = 0.0
        self._primed = False
        self._anchor: Optional[Tuple[float, float]] = None

    def update(self, sim_time: float, wall_time: float) -> float:
        if self._anchor is None:
            self._anchor = (sim_time, wall_time)
            return self.rtf
        sim0, wall0 = self._anchor
        dwall = wall_time - wall0
        if dwall < self.window_s:
            return self.rtf
        inst = max(0.0, (sim_time - sim0) / dwall)
        self.rtf = inst if not self._primed else (1.0 - self.alpha) * self.rtf + self.alpha * inst
        self._primed = True
        self._anchor = (sim_time, wall_time)
        return self.rtf

    def rebase(self):
        """Forget the anchor (e.g. after sim time jumps on reset); keeps the smoothed value."""
        self._anchor = None


def plan_catch_up(behind_s: float, dt: float, max_steps: int) -> Tuple[int, float]:
    """How many mj_steps to run now to catch up with wall-clock.

    Returns (n_steps, dropped_s). If the sim is more than max_steps behind, run max_steps and
    report the remaining debt as dropped so the caller can forgive it (no spiral of death:
    a slow machine runs below real time and rtf reports it honestly).
    """
    if behind_s <= 0.0 or dt <= 0.0:
        return 0, 0.0
    n = int(behind_s / dt + 1e-9)
    if n > max_steps:
        return max_steps, behind_s - max_steps * dt
    return n, 0.0


def build_game_status(
    crops: Dict[str, Sequence[float]],
    trunk_pos: Sequence[float],
    trunk_quat: Sequence[float],
    falls: int,
    fallen: bool,
    elapsed_s: float,
    rtf: float,
) -> dict:
    """/nandi/game_status payload. `crops` maps name -> world centre, in crop order."""
    names: List[str] = [n for n, p in crops.items() if in_basket(p, trunk_pos, trunk_quat)]
    return {
        "crops_total": len(crops),
        "in_basket": len(names),
        "in_basket_names": names,
        "falls": int(falls),
        "fallen": bool(fallen),
        "elapsed_s": round(max(0.0, float(elapsed_s)), 2),
        "rtf": round(float(rtf), 3),
    }
