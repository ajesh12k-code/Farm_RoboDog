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

from atharv_nandi.core import model_layout
from atharv_nandi.core.farm_scene import crop_qpos_slice
from atharv_nandi.core.game_status import BASKET_FLOOR_Z, BASKET_HALF_X, BASKET_HALF_Y, BASKET_TOP_Z

REST_ARM = (0.0, 0.0, 0.1, -1.8, 1.4, 0.0, 0.0)  # gantry, yaw, shoulder, elbow, wrist, finger_l, finger_r
REACH_M = 1.08                # contract: crops farther than this (horizontal, from trunk) are not attempted
# Grasp check (on the LIVE qpos after the lift): the crop must have risen at least GRASP_MIN_RISE_M,
# its centre must sit between the finger pads (pads span 2 cm above to 5 cm below gripper_tip along the
# tool axis; a held crop rides ~2.5 cm below the tip), and it must not be sliding out of the pads: over
# GRASP_HOLD_S its position relative to the tip may drift at most GRASP_MAX_SLIP_M. The old check
# (rise + 6 cm from the tip, one sample) passed a crop that was creeping out of the grip at ~8 cm/s.
GRASP_MIN_RISE_M = 0.05
GRASP_MAX_TIP_DIST_M = 0.045
GRASP_MAX_SLIP_M = 0.006
GRASP_HOLD_S = 0.3
HOLD_MAX_TIP_DIST_M = 0.06    # while carrying: farther than this from the tip = crop dropped
# Basket footprint (trunk frame) is shared with /nandi/game_status scoring (atharv_nandi/core/game_status.py)
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
# harvest_ik_service publishes /nandi/arm_targets every PUBLISH_DT, moving the command at most
# SLEW_PER_TICK per tick (gantry m/s, joints rad/s, fingers m/s) so mode switches never slam the arm.
# Kept here so the ROS-free closed-loop test drives the arm through exactly the same slew.
PUBLISH_DT = 0.02
# Rates (physics alignment v1, DERIVED from the mounting-audit reaction-torque estimate I*alpha ~ 2.4 Nm at the
# trunk): gantry 0.5 m/s, yaw 1.5 rad/s, shoulder/elbow/wrist 2.0 rad/s (were 3.0), fingers 0.1 m/s (unchanged).
SLEW_RATES = (0.5, 1.5, 2.0, 2.0, 2.0, 0.1, 0.1)
SLEW_PER_TICK = (np.array(SLEW_RATES) * PUBLISH_DT).tolist()

# Sequence geometry (metres). Gripper points straight down for pick and drop.
GRASP_TIP_DZ = 0.025    # tip above crop centre at grasp: pads straddle the equator, pad ends clear the soil
APPROACH_DZ = 0.10      # pre-grasp hover above the grasp pose
LIFT_DZ = 0.20          # vertical lift before the grasp check
CARRY_TIP_Z = 0.30      # trunk-frame tip height while swinging over the body (crop clears rail + walls)
DROP_Y = 0.065          # left channel between gantry rail (|y|<0.022) and basket wall (|y|>0.108)
DROP_TIP_Z = 0.255      # trunk-frame tip height at release (crop centre ~0.23, clear of the turret)
DROP_REACH = 0.30       # horizontal shoulder -> drop point distance
DROP_X_MAX = 0.22
SAG_MAX_FIX = 0.02      # largest gravity-sag correction applied to an IK target (the gravity feed-forward
                        # tau_g / kp of arm_ik_solver.gravity_offsets removes most sag; this closes the rest)
SAG_TOL = 0.004
# A Cartesian waypoint may miss its (sag-corrected) target by up to this much: the IK is solved with the
# gantry pinned and the sag offset can push an edge-of-envelope target a centimetre past what the pinned
# chain reaches (the old 3 mm limit aborted reachable crops with "Arm path out of reach (0.9 cm off)").
# Reachability itself is decided up front, strictly, for the whole grasp column (see _plan_column).
PATH_TOL = 0.015
# Scaled with the slower slew (3.0 -> 2.0 rad/s joints): 2.0 s -> 3.0 s.
SETTLE_TIMEOUT_S = 3.0
# The grasp column is planned for the crop where it lay when the approach began. If, at the hover, the
# crop has moved more than CROP_REPLAN_M (e.g. still rolling after a failed grasp), the column is
# re-planned for where it is now (at most CROP_REPLANS times) instead of descending a pinned-gantry
# column that no longer reaches it. A crop counts as still once it moves < CROP_STILL_M over 0.1 s.
CROP_REPLAN_M = 0.01
CROP_REPLANS = 2
CROP_STILL_M = 0.001
CROP_SETTLE_TIMEOUT_S = 3.0
# An out-of-reach report names the trunk tilt once it is at least this large: ~2 deg of nose-up pitch
# already costs a ground pick 0.62 m ahead its reach, so "move closer" alone would mislead on a ridge.
TILT_NOTE_DEG = 1.5


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


def world_reset_detected(prev_qpos: Sequence[float], qpos: Sequence[float],
                         layout: Optional[model_layout.Layout] = None) -> bool:
    """True if two consecutive sim_qpos samples differ by more than physics allows (a world reset).

    The legacy crop count comes from `layout` (default: this process's current layout when its nq matches),
    so joints appended by work-package modules are never read as crops (WP0)."""
    prev, cur = np.asarray(prev_qpos, dtype=np.float64), np.asarray(qpos, dtype=np.float64)
    if prev.shape != cur.shape or cur.shape[0] < 12:
        return False
    if np.linalg.norm(cur[0:3] - prev[0:3]) > RESET_TRUNK_JUMP_M:
        return True
    if np.any(np.abs(cur[7:12] - prev[7:12]) > np.asarray(RESET_ARM_JUMP)):
        return True
    n_crops = model_layout.n_legacy_crops_for(cur.shape[0], layout)
    for k in range(1, n_crops + 1):
        s = crop_qpos_slice(k)
        if np.linalg.norm(cur[s][0:3] - prev[s][0:3]) > RESET_CROP_JUMP_M:
            return True
    return False


def grasp_check(
    crop_before: Sequence[float],
    crop_after: Sequence[float],
    tip: Sequence[float],
    crop_later: Optional[Sequence[float]] = None,
    tip_later: Optional[Sequence[float]] = None,
) -> Tuple[bool, str]:
    """Is the crop really in the gripper? All positions are world frame, read from live qpos.

    crop_before: crop centre just before the lift; crop_after/tip: after the lift; crop_later/tip_later
    (optional): GRASP_HOLD_S later, to catch a crop that is sliding out of the pads.
    Returns (ok, detail) - detail is operator-facing ("rose 0.2 cm, 6.1 cm from the tip").
    """
    before, after, t = (np.asarray(v, dtype=np.float64) for v in (crop_before, crop_after, tip))
    rose = float(after[2] - before[2])
    near = float(np.linalg.norm(after - t))
    detail = f"rose {rose * 100:.1f} cm, {near * 100:.1f} cm from the tip"
    ok = rose >= GRASP_MIN_RISE_M and near <= GRASP_MAX_TIP_DIST_M
    if ok and crop_later is not None and tip_later is not None:
        later, t2 = np.asarray(crop_later, dtype=np.float64), np.asarray(tip_later, dtype=np.float64)
        slip = float(np.linalg.norm((later - t2) - (after - t)))
        detail += f", slipped {slip * 100:.1f} cm"
        ok = slip <= GRASP_MAX_SLIP_M and float(np.linalg.norm(later - t2)) <= GRASP_MAX_TIP_DIST_M
    return bool(ok), detail


def grasp_ok(crop_before, crop_after, tip, crop_later=None, tip_later=None) -> bool:
    return grasp_check(crop_before, crop_after, tip, crop_later, tip_later)[0]


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
        self.held = None           # crop being carried: every motion checks it is still in the gripper
        self.ff = np.zeros(5)      # gravity feed-forward currently added to the commanded joints

    def run(self, target: str, start_cmd: Sequence[float]) -> Tuple[bool, str]:
        self.q5 = np.array(start_cmd[:5], dtype=np.float64)
        self.grip = float(start_cmd[5])
        try:
            try:
                ok, msg = True, self._harvest(target)
            except HarvestFailed as exc:
                ok, msg = False, str(exc)
            self.held = None
            self.io.phase("return_rest")
            self._joint_move(np.array(REST_ARM[:5]), self.ik.grip_open, 1.5, gravity=False)
        except HarvestAborted:
            return False, f"Aborted harvest of {target}"
        return ok, msg

    # -- phases
    def _harvest(self, target: str) -> str:
        detail = ""
        for attempt in (1, 2):
            ok, detail = self._pick(target, attempt)
            if ok:
                break
        else:
            raise HarvestFailed(f"Grasp failed on {target} (2 attempts; last: {detail})")
        self.held = target
        self._carry_to_basket()
        self.held = None
        self.io.phase("release")
        self._joint_move(self.q5, self.ik.grip_open, 0.5)
        self.io.phase("verify_drop")
        for _ in range(15):
            self.io.wait(0.1)
            if crop_in_basket(self.io.qpos(), target):
                return f"Harvested {target} into basket"
        raise HarvestFailed(f"Drop missed: {target} not in basket")

    def _pick(self, target: str, attempt: int) -> Tuple[bool, str]:
        self.io.phase("approach" if attempt == 1 else "regrasp")
        self.offset = np.zeros(3)
        if attempt > 1:
            self._wait_crop_still(target)  # a crop let go by a failed grasp may still be rolling
        planned = crop_pos(self.io.qpos(), target)
        for replan in range(CROP_REPLANS + 1):
            qpos = self.io.qpos()
            q_hover = self._plan_column(qpos, planned, target)
            self._joint_move(q_hover, self.ik.grip_open, 1.5)
            self._wait_still()
            crop = crop_pos(self.io.qpos(), target)  # it may have been nudged since the approach began
            if np.linalg.norm(crop[:2] - planned[:2]) <= CROP_REPLAN_M:
                break
            if replan == CROP_REPLANS:
                raise HarvestFailed(f"{target} keeps moving ({np.linalg.norm(crop[:2] - planned[:2]) * 100:.1f} cm"
                                    " since planned), try again")
            # The column was planned for where the crop was; re-plan it for where it is now.
            self._wait_crop_still(target)
            planned = crop_pos(self.io.qpos(), target)
        self.goal = crop + [0.0, 0.0, GRASP_TIP_DZ + APPROACH_DZ]

        self.io.phase("descend")
        grasp = crop + [0.0, 0.0, GRASP_TIP_DZ]
        self._cartesian_move(crop + [0.0, 0.0, GRASP_TIP_DZ + APPROACH_DZ], grasp, self.ik.grip_open, 1.0)
        self._settle_at(grasp, self.ik.grip_open)

        self.io.phase("grasp")
        self._joint_move(self.q5, self.ik.grip_closed, 0.5)
        self.io.wait(0.3)
        before = crop_pos(self.io.qpos(), target)

        self.io.phase("lift")
        self._cartesian_move(grasp, grasp + [0.0, 0.0, LIFT_DZ], self.ik.grip_closed, 1.0)
        self._wait_still()

        # Judge the grasp on the live state only: where the crop actually is relative to the actual tip,
        # twice, GRASP_HOLD_S apart (a crop sliding out of the pads fails the second sample).
        self.io.phase("verify_grasp")
        qpos = self.io.qpos()
        after, tip = crop_pos(qpos, target), self.ik.tip_pos(qpos)
        self.io.wait(GRASP_HOLD_S)
        qpos = self.io.qpos()
        ok, detail = grasp_check(before, after, tip, crop_pos(qpos, target), self.ik.tip_pos(qpos))
        if ok:
            return True, detail
        self._joint_move(self.q5, self.ik.grip_open, 0.4)  # let go of whatever is half-held
        self.io.wait(0.6)                                   # let it drop and settle before re-reading it
        return False, detail

    def _plan_column(self, qpos, crop, target: str) -> np.ndarray:
        """Joints for the hover pose, with a gantry position from which the whole vertical pick column
        (hover, grasp, lift, carry height) is reachable with the gantry held still.

        Every column pose must also be free of arm-vs-robot contact (a crop tight against the legs or
        the trunk side cannot be picked from above: the forearm or gripper lands on the robot).
        Raises HarvestFailed('... out of arm reach ...') if no such gantry position exists, so a crop
        that is truly out of reach fails cleanly before the arm moves, and a reachable one never fails
        half-way down the column.
        """
        grasp = crop + [0.0, 0.0, GRASP_TIP_DZ]
        hover, lift = grasp + [0.0, 0.0, APPROACH_DZ], grasp + [0.0, 0.0, LIFT_DZ]
        high = to_trunk(lift, qpos)
        high[2] = max(high[2], CARRY_TIP_Z)
        column = (hover, grasp, lift, to_world(high, qpos))
        first = self.ik.solve(qpos, hover, seed_q=self.q5)
        if first.ok:
            lo, hi = self.ik.limits[0]
            g0 = float(first.q[0])
            for g in [g0, *sorted(np.linspace(lo, hi, 14), key=lambda x: abs(x - g0))]:
                sols = [self.ik.solve(qpos, p, seed_q=first.q, gantry=g) for p in column]
                if all(r.ok for r in sols) and not any(self._blocked(qpos, r.q) for r in sols):
                    return sols[0].q
        dist = horizontal_distance(qpos, target)
        if crop_under_body(qpos, target):  # e.g. rolled there after a failed grasp
            raise HarvestFailed(f"{target} out of arm reach ({dist:.2f} m, under the body), back up")
        _, r = trunk_pose(qpos)
        tilt = math.degrees(math.acos(float(np.clip(r[2, 2], -1.0, 1.0))))
        if tilt >= TILT_NOTE_DEG:  # on a ridge the arm's reach to the ground shrinks: say so
            pitch = math.degrees(math.asin(float(np.clip(r[2, 0], -1.0, 1.0))))
            roll = math.degrees(math.atan2(r[2, 1], r[2, 2]))
            raise HarvestFailed(f"{target} out of arm reach ({dist:.2f} m; trunk tilted {tilt:.1f} deg, pitch "
                                f"{pitch:+.1f} roll {roll:+.1f}), level the robot or move closer")
        raise HarvestFailed(f"{target} out of arm reach ({dist:.2f} m), move closer")

    def _blocked(self, qpos, q5) -> bool:
        check = getattr(self.ik, "robot_collisions", None)
        return bool(check(qpos, q5)) if check is not None else False

    def _carry_to_basket(self):
        self.io.phase("carry")
        qpos = self.io.qpos()
        high = to_trunk(self.goal, qpos)
        high[2] = max(high[2], CARRY_TIP_Z)
        self._cartesian_move(self.goal, to_world(high, qpos), self.ik.grip_closed, 1.2)

        # Swing over the left side on yaw alone: the tip circles at CARRY_TIP_Z, clear of the body.
        drop = self._drop_point()
        q = self.q5.copy()
        q[1] = math.atan2(drop[1], drop[0] - (self.ik.shoulder0[0] + q[0]))
        self._joint_move(q, self.ik.grip_closed, 2.0)
        self._wait_still()

        qpos = self.io.qpos()
        self.offset = np.zeros(3)
        self.goal = self.ik.tip_pos(qpos, self.q5)
        drop_w = to_world(drop, qpos)
        self._cartesian_move(self.goal, drop_w, self.ik.grip_closed, 0.8)
        self._settle_at(drop_w, self.ik.grip_closed)
        self._check_held()

    def _drop_point(self) -> np.ndarray:
        """Trunk-frame release point above the left basket channel, DROP_REACH from the shoulder."""
        sx = self.ik.shoulder0[0] + self.q5[0]
        x = sx - DROP_REACH if sx - DROP_REACH >= -DROP_X_MAX else sx + DROP_REACH
        return np.array([float(np.clip(x, -DROP_X_MAX, DROP_X_MAX)), DROP_Y, DROP_TIP_Z])

    # -- motion primitives
    def _check_held(self, qpos=None):
        if self.held is None:
            return
        qpos = self.io.qpos() if qpos is None else qpos
        if np.linalg.norm(crop_pos(qpos, self.held) - self.ik.tip_pos(qpos)) > HOLD_MAX_TIP_DIST_M:
            held, self.held = self.held, None
            raise HarvestFailed(f"Dropped {held} while carrying it")

    def _wait_still(self, timeout: float = SETTLE_TIMEOUT_S):
        """Poll the live tip until it has stopped moving (< 2 cm/s over 0.1 s), or timeout.

        Used instead of fixed sleeps so the sequence copes with servo lag, the command slew in
        harvest_ik_service and a sim running slower than real time.
        """
        prev, waited = None, 0.0
        while waited < timeout:
            self.io.wait(0.1)
            waited += 0.1
            qpos = self.io.qpos()
            self._check_held(qpos)
            tip = self.ik.tip_pos(qpos)
            if prev is not None and np.linalg.norm(tip - prev) < 0.002:
                return
            prev = tip

    def _wait_crop_still(self, target: str, timeout: float = CROP_SETTLE_TIMEOUT_S):
        """Poll the live crop until it has stopped rolling (< CROP_STILL_M over 0.1 s), or timeout."""
        prev, waited = crop_pos(self.io.qpos(), target), 0.0
        while waited < timeout:
            self.io.wait(0.1)
            waited += 0.1
            cur = crop_pos(self.io.qpos(), target)
            if np.linalg.norm(cur - prev) < CROP_STILL_M:
                return
            prev = cur

    def _joint_move(self, q_target, grip: float, duration: float, ease: bool = True, gravity: bool = True):
        """Interpolate the commanded joints to q_target (kinematic targets; the gravity feed-forward is
        added on top unless gravity=False, which ends exactly on q_target - used for REST_ARM)."""
        q_start, g_start, ff_start = self.q5.copy(), self.grip, self.ff.copy()
        q_target = np.asarray(q_target, dtype=np.float64)
        ff_target = np.zeros(5)
        if gravity and hasattr(self.ik, "gravity_offsets"):
            payload = self.ik.crop_mass if self.held is not None else 0.0
            ff_target = self.ik.gravity_offsets(self.io.qpos(), q_target, payload)
        steps = max(1, int(round(duration / self.dt)))
        for s in range(1, steps + 1):
            a = s / steps
            if ease:
                a = 0.5 * (1.0 - math.cos(math.pi * a))
            q = q_start + a * (q_target - q_start) + ff_start + a * (ff_target - ff_start)
            g = g_start + a * (grip - g_start)
            self.io.command([*q.tolist(), g, g])
            self.io.wait(self.dt)
            if self.held is not None and s % 5 == 0:
                self._check_held()
        self.q5, self.grip, self.ff = q_target.copy(), float(grip), ff_target

    def _cartesian_move(self, p_from, p_to, grip: float, duration: float):
        """Straight tip line in the world, IK every 0.1 s against the latest qpos, gantry held."""
        p_from, p_to = np.asarray(p_from, dtype=np.float64), np.asarray(p_to, dtype=np.float64)
        n = max(2, int(round(duration / 0.1)))
        for i in range(1, n + 1):
            p = p_from + (p_to - p_from) * (i / n)
            qpos = self.io.qpos()
            res = self.ik.solve(qpos, p + self.offset, seed_q=self.q5, gantry=self.q5[0])
            if not res.ok and np.any(self.offset):  # the sag offset may push an edge target out
                plain = self.ik.solve(qpos, p, seed_q=self.q5, gantry=self.q5[0])
                res = plain if plain.pos_err < res.pos_err else res
            if not res.ok and not (res.pos_err <= PATH_TOL and res.axis_err_deg <= 10.0):
                raise HarvestFailed(f"Arm path out of reach ({res.pos_err * 100:.1f} cm off)")
            self._joint_move(res.q, grip, duration / n, ease=False)
        self.goal = p_to

    def _settle_at(self, goal, grip: float):
        """Wait out servo lag, then shift the IK target to cancel gravity sag at this pose.

        Errors above 4 cm mean contact (the pose is blocked), not sag: those are left alone.
        """
        self._wait_still()
        for _ in range(2):
            qpos = self.io.qpos()
            err = goal - self.ik.tip_pos(qpos)
            if np.linalg.norm(err) < SAG_TOL or np.linalg.norm(err) > 0.04:
                return
            self.offset = np.clip(self.offset + err, -SAG_MAX_FIX, SAG_MAX_FIX)
            res = self.ik.solve(qpos, goal + self.offset, seed_q=self.q5, gantry=self.q5[0])
            if not res.ok and res.pos_err > PATH_TOL:
                return
            self._joint_move(res.q, grip, 0.3)
            self._wait_still()


# ---------------------------------------------------------------------- WP8: tool swap arm targets (WP2 change request)
def tool_arm_command(payload, ik, qpos, finger_default: Optional[Sequence[float]] = None) -> Optional[List[float]]:
    """/nandi/tool/arm_targets ``{q5, fingers, payload_kg, phase}`` (WP2, without gravity feed-forward) -> the 7 arm
    targets for /nandi/arm_targets: q5 + ``ik.gravity_offsets(qpos, q5, payload_kg)`` and the fingers. None for a
    malformed message, a message without q5 (the swap is between moves) or an idle phase. Pure; the harvest node
    forwards it while a swap runs (``mode`` ``tool_change``)."""
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except ValueError:
            return None
    if not isinstance(payload, dict) or payload.get("phase") in (None, "idle"):
        return None
    q5 = payload.get("q5")
    if not isinstance(q5, (list, tuple)) or len(q5) != 5:
        return None
    try:
        q = np.asarray([float(v) for v in q5], dtype=np.float64)
        kg = float(payload.get("payload_kg") or 0.0)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(q)) or not math.isfinite(kg):
        return None
    fingers = payload.get("fingers")
    if not isinstance(fingers, (list, tuple)) or len(fingers) != 2:
        fingers = list(finger_default) if finger_default is not None else [REST_ARM[5], REST_ARM[6]]
    off = ik.gravity_offsets(qpos, q, max(0.0, kg))
    return [*(q + off).tolist(), float(fingers[0]), float(fingers[1])]
