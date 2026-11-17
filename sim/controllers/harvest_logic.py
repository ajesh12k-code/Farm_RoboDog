"""Pure Atharv Nandi harvest logic: target choice, grasp/drop checks, status JSON and the pick-and-place sequence.

No ROS in here. HarvestSequence touches the world only through a small I/O object, so the same code
drives the live arm from harvest_ik_service.py (topics + wall-clock waits) and a bare MuJoCo model in
tests/test_harvest_logic.py (ctrl writes + mj_step).

I/O object (duck-typed):
    qpos() -> np.ndarray        latest full-world qpos (re-read at every step, never a stale snapshot)
    command(arm7) -> None       publish 7 arm targets; raises HarvestAborted once the run was aborted
    wait(seconds) -> None       let time pass; raises HarvestAborted once the run was aborted
    phase(name) -> None         report the current phase (goes into /nandi/harvest_status)
"""
import json
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from controllers.farm_scene import CROP_QPOS_START, crop_qpos_slice
from controllers.game_status import BASKET_FLOOR_Z, BASKET_HALF_X, BASKET_HALF_Y, BASKET_TOP_Z

REST_ARM = (0.0, 0.0, 0.1, -1.8, 1.4, 0.0, 0.0)  # gantry, yaw, shoulder, elbow, wrist, finger_l, finger_r
REACH_M = 1.08                # contract: crops farther than this (horizontal, from trunk) are not attempted
GRASP_MIN_RISE_M = 0.05       # grasp verified if the crop rose at least this much ...
GRASP_MAX_TIP_DIST_M = 0.06   # ... and is still this close to gripper_tip
# Basket footprint (trunk frame) is shared with /nandi/game_status scoring (controllers/game_status.py)
# so the drop check and the score can never disagree.
# A crop centre inside this trunk-frame xy box lies under the body (trunk_geom half-extents 0.34 x 0.14
# plus a crop radius): the arm cannot reach it without passing through the trunk and basket.
UNDER_BODY_HALF_X = 0.375
UNDER_BODY_HALF_Y = 0.175
# World-reset detection on consecutive /nandi/sim_qpos samples (20 ms of sim time apart): jumps
# larger than these are physically impossible (gantry slews <= 0.5 m/s, joints are servo-limited, the
# trunk moves well under 5 m/s), so they mean /nandi/reset_world teleported the world.
RESET_TRUNK_JUMP_M = 0.10
RESET_ARM_JUMP = (0.10, 0.6, 0.6, 0.6, 0.6)  # gantry (m), yaw, shoulder, elbow, wrist (rad)
RESET_CROP_JUMP_M = 0.15                     # a carried or falling crop moves < 0.05 m per sample

# Sequence geometry (metres). Gripper points straight down for pick and drop.
GRASP_TIP_DZ = 0.025    # tip above crop centre at grasp: pads straddle the equator, pad ends clear the soil
APPROACH_DZ = 0.10      # pre-grasp hover above the grasp pose
LIFT_DZ = 0.20          # vertical lift before the grasp check
CARRY_TIP_Z = 0.30      # trunk-frame tip height while swinging over the body (crop clears rail + walls)
DROP_Y = 0.065          # left channel between gantry rail (|y|<0.022) and basket wall (|y|>0.108)
DROP_TIP_Z = 0.255      # trunk-frame tip height at release (crop centre ~0.23, clear of the turret)
DROP_REACH = 0.30       # horizontal shoulder -> drop point distance
DROP_X_MAX = 0.22
SAG_MAX_FIX = 0.02      # largest gravity-sag correction applied to an IK target
SAG_TOL = 0.004


class HarvestAborted(Exception):
    """The run was aborted (release_arm / manual override): stop at once, leave the arm alone."""


class HarvestFailed(Exception):
    """The harvest cannot continue; str(exc) is the operator-facing result."""


# ---------------------------------------------------------------------- frames + crops
def quat_to_mat(q: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def trunk_pose(qpos: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(qpos[0:3], dtype=np.float64), quat_to_mat(qpos[3:7])


def to_trunk(p_world: Sequence[float], qpos: Sequence[float]) -> np.ndarray:
    t, r = trunk_pose(qpos)
    return r.T @ (np.asarray(p_world, dtype=np.float64) - t)


def to_world(p_trunk: Sequence[float], qpos: Sequence[float]) -> np.ndarray:
    t, r = trunk_pose(qpos)
    return t + r @ np.asarray(p_trunk, dtype=np.float64)


def crop_index(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def crop_pos(qpos: Sequence[float], name: str) -> np.ndarray:
    return np.array(qpos[crop_qpos_slice(crop_index(name))][0:3], dtype=np.float64)


def in_basket(p_trunk: Sequence[float]) -> bool:
    x, y, z = p_trunk
    return bool(abs(x) < BASKET_HALF_X and abs(y) < BASKET_HALF_Y and BASKET_FLOOR_Z <= z <= BASKET_TOP_Z)


def crop_in_basket(qpos: Sequence[float], name: str) -> bool:
    return in_basket(to_trunk(crop_pos(qpos, name), qpos))


def basket_contents(qpos: Sequence[float], names: Iterable[str]) -> List[str]:
    return [n for n in names if crop_in_basket(qpos, n)]


def horizontal_distance(qpos: Sequence[float], name: str) -> float:
    return float(np.linalg.norm(crop_pos(qpos, name)[:2] - np.asarray(qpos[0:2], dtype=np.float64)))


# ---------------------------------------------------------------------- decisions
def crop_under_body(qpos: Sequence[float], name: str) -> bool:
    x, y, _ = to_trunk(crop_pos(qpos, name), qpos)
    return bool(abs(x) < UNDER_BODY_HALF_X and abs(y) < UNDER_BODY_HALF_Y)


def select_target(qpos: Sequence[float], names: Iterable[str], reach: float = REACH_M) -> Tuple[Optional[str], str]:
    """Nearest crop not already in the basket (nor under the body), if within reach.

    Returns (name | None, message).
    """
    free = [n for n in names if not crop_in_basket(qpos, n)]
    if not free:
        return None, "No crop in reach (all crops already in basket)"
    under = [(horizontal_distance(qpos, n), n) for n in free if crop_under_body(qpos, n)]
    candidates = [(horizontal_distance(qpos, n), n) for n in free if not crop_under_body(qpos, n)]
    if candidates and min(candidates)[0] <= reach:
        dist, name = min(candidates)
        return name, f"Harvesting {name} ({dist:.2f} m)"
    if under:  # nearer than anything reachable: tell the operator to back off it
        dist, name = min(under)
        return None, f"No crop in reach (nearest {name} {dist:.2f} m, under the body - back up)"
    dist, name = min(candidates)
    return None, f"No crop in reach (nearest {name} {dist:.2f} m)"


def world_reset_detected(prev_qpos: Sequence[float], qpos: Sequence[float]) -> bool:
    """True if two consecutive sim_qpos samples differ by more than physics allows (a world reset)."""
    prev, cur = np.asarray(prev_qpos, dtype=np.float64), np.asarray(qpos, dtype=np.float64)
    if prev.shape != cur.shape or cur.shape[0] < 12:
        return False
    if np.linalg.norm(cur[0:3] - prev[0:3]) > RESET_TRUNK_JUMP_M:
        return True
    if np.any(np.abs(cur[7:12] - prev[7:12]) > np.asarray(RESET_ARM_JUMP)):
        return True
    n_crops = (cur.shape[0] - CROP_QPOS_START) // 7
    for k in range(1, n_crops + 1):
        s = crop_qpos_slice(k)
        if np.linalg.norm(cur[s][0:3] - prev[s][0:3]) > RESET_CROP_JUMP_M:
            return True
    return False


def grasp_ok(crop_before: Sequence[float], crop_after: Sequence[float], tip: Sequence[float]) -> bool:
    rose = crop_after[2] - crop_before[2]
    near = np.linalg.norm(np.asarray(crop_after, dtype=np.float64) - np.asarray(tip, dtype=np.float64))
    return bool(rose >= GRASP_MIN_RISE_M and near <= GRASP_MAX_TIP_DIST_M)


def status_json(mode: str, phase: str, target: Optional[str], last_result: str) -> str:
    return json.dumps({"mode": mode, "phase": phase, "target": target, "last_result": last_result})


def slew(current: Sequence[float], goal: Sequence[float], max_step: Sequence[float]) -> List[float]:
    """Move current toward goal by at most max_step per element."""
    c = np.asarray(current, dtype=np.float64)
    step = np.clip(np.asarray(goal, dtype=np.float64) - c, -np.asarray(max_step), np.asarray(max_step))
    return (c + step).tolist()


# ---------------------------------------------------------------------- sequence
class HarvestSequence:
    """Pick one crop, verify the grasp (one regrasp), drop it in the basket, verify, return to rest.

    ik is a controllers.arm_ik_solver.DorsalArmIKSolver (owned by this sequence's thread).
    """

    def __init__(self, ik, io, dt: float = 0.02):
        self.ik, self.io, self.dt = ik, io, dt
        self.q5 = np.array(REST_ARM[:5], dtype=np.float64)  # last commanded arm joints
        self.grip = float(REST_ARM[5])
        self.goal = np.zeros(3)    # last nominal tip goal (world)
        self.offset = np.zeros(3)  # gravity-sag correction added to IK targets

    def run(self, target: str, start_cmd: Sequence[float]) -> Tuple[bool, str]:
        self.q5 = np.array(start_cmd[:5], dtype=np.float64)
        self.grip = float(start_cmd[5])
        try:
            try:
                ok, msg = True, self._harvest(target)
            except HarvestFailed as exc:
                ok, msg = False, str(exc)
            self.io.phase("return_rest")
            self._joint_move(np.array(REST_ARM[:5]), self.ik.grip_open, 1.5)
        except HarvestAborted:
            return False, f"Aborted harvest of {target}"
        return ok, msg

    # -- phases
    def _harvest(self, target: str) -> str:
        for attempt in (1, 2):
            if self._pick(target, attempt):
                break
        else:
            raise HarvestFailed(f"Grasp failed on {target} (2 attempts)")
        self._carry_to_basket()
        self.io.phase("release")
        self._joint_move(self.q5, self.ik.grip_open, 0.5)
        self.io.phase("verify_drop")
        for _ in range(15):
            self.io.wait(0.1)
            if crop_in_basket(self.io.qpos(), target):
                return f"Harvested {target} into basket"
        raise HarvestFailed(f"Drop missed: {target} not in basket")

    def _pick(self, target: str, attempt: int) -> bool:
        self.io.phase("approach" if attempt == 1 else "regrasp")
        self.offset = np.zeros(3)
        qpos = self.io.qpos()
        hover = crop_pos(qpos, target) + [0.0, 0.0, GRASP_TIP_DZ + APPROACH_DZ]
        res = self.ik.solve(qpos, hover, seed_q=self.q5)
        if not res.ok:
            raise HarvestFailed(f"{target} out of arm reach ({horizontal_distance(qpos, target):.2f} m), move closer")
        self._joint_move(res.q, self.ik.grip_open, 1.5)
        self.goal = hover

        self.io.phase("descend")
        crop = crop_pos(self.io.qpos(), target)  # it may have been nudged since the approach began
        grasp = crop + [0.0, 0.0, GRASP_TIP_DZ]
        self._cartesian_move(crop + [0.0, 0.0, GRASP_TIP_DZ + APPROACH_DZ], grasp, self.ik.grip_open, 1.0)
        self._settle_at(grasp, self.ik.grip_open)

        self.io.phase("grasp")
        self._joint_move(self.q5, self.ik.grip_closed, 0.5)
        self.io.wait(0.3)
        before = crop_pos(self.io.qpos(), target)

        self.io.phase("lift")
        self._cartesian_move(grasp, grasp + [0.0, 0.0, LIFT_DZ], self.ik.grip_closed, 1.0)
        self.io.wait(0.3)

        self.io.phase("verify_grasp")
        qpos = self.io.qpos()
        if grasp_ok(before, crop_pos(qpos, target), self.ik.tip_pos(qpos)):
            return True
        self._joint_move(self.q5, self.ik.grip_open, 0.4)  # let go of whatever is half-held
        self.io.wait(0.6)                                   # let it drop and settle before re-reading it
        return False

    def _carry_to_basket(self):
        self.io.phase("carry")
        qpos = self.io.qpos()
        high = to_trunk(self.goal, qpos)
        high[2] = CARRY_TIP_Z
        self._cartesian_move(self.goal, to_world(high, qpos), self.ik.grip_closed, 1.2)

        # Swing over the left side on yaw alone: the tip circles at CARRY_TIP_Z, clear of the body.
        drop = self._drop_point()
        q = self.q5.copy()
        q[1] = math.atan2(drop[1], drop[0] - (self.ik.shoulder0[0] + q[0]))
        self._joint_move(q, self.ik.grip_closed, 2.0)

        qpos = self.io.qpos()
        self.offset = np.zeros(3)
        self.goal = self.ik.tip_pos(qpos, self.q5)
        drop_w = to_world(drop, qpos)
        self._cartesian_move(self.goal, drop_w, self.ik.grip_closed, 0.8)
        self._settle_at(drop_w, self.ik.grip_closed)

    def _drop_point(self) -> np.ndarray:
        """Trunk-frame release point above the left basket channel, DROP_REACH from the shoulder."""
        sx = self.ik.shoulder0[0] + self.q5[0]
        x = sx - DROP_REACH if sx - DROP_REACH >= -DROP_X_MAX else sx + DROP_REACH
        return np.array([float(np.clip(x, -DROP_X_MAX, DROP_X_MAX)), DROP_Y, DROP_TIP_Z])

    # -- motion primitives
    def _joint_move(self, q_target, grip: float, duration: float, ease: bool = True):
        q_start, g_start = self.q5.copy(), self.grip
        q_target = np.asarray(q_target, dtype=np.float64)
        steps = max(1, int(round(duration / self.dt)))
        for s in range(1, steps + 1):
            a = s / steps
            if ease:
                a = 0.5 * (1.0 - math.cos(math.pi * a))
            q = q_start + a * (q_target - q_start)
            g = g_start + a * (grip - g_start)
            self.io.command([*q.tolist(), g, g])
            self.io.wait(self.dt)
        self.q5, self.grip = q_target.copy(), float(grip)

    def _cartesian_move(self, p_from, p_to, grip: float, duration: float):
        """Straight tip line in the world, IK every 0.1 s against the latest qpos, gantry held."""
        p_from, p_to = np.asarray(p_from, dtype=np.float64), np.asarray(p_to, dtype=np.float64)
        n = max(2, int(round(duration / 0.1)))
        for i in range(1, n + 1):
            p = p_from + (p_to - p_from) * (i / n)
            res = self.ik.solve(self.io.qpos(), p + self.offset, seed_q=self.q5, gantry=self.q5[0])
            if not res.ok:
                raise HarvestFailed(f"Arm path out of reach ({res.pos_err * 100:.1f} cm off)")
            self._joint_move(res.q, grip, duration / n, ease=False)
        self.goal = p_to

    def _settle_at(self, goal, grip: float):
        """Wait out servo lag, then shift the IK target to cancel gravity sag at this pose.

        Errors above 4 cm mean contact (the pose is blocked), not sag: those are left alone.
        """
        for _ in range(2):
            self.io.wait(0.3)
            qpos = self.io.qpos()
            err = goal - self.ik.tip_pos(qpos)
            if np.linalg.norm(err) < SAG_TOL or np.linalg.norm(err) > 0.04:
                return
            self.offset = np.clip(self.offset + err, -SAG_MAX_FIX, SAG_MAX_FIX)
            res = self.ik.solve(qpos, goal + self.offset, seed_q=self.q5, gantry=self.q5[0])
            if not res.ok:
                return
            self._joint_move(res.q, grip, 0.3)
