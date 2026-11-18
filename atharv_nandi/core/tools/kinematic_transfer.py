"""Kinematic transfers: desap cell tray, papaya cradle, ballast and crate unload (WP3; ORCHARD-MVP-SPEC 1.2 step 11,
2.4 "Basket", MISSION-PRESETS 9). ROS-free; MuJoCo + numpy. Implements ``interfaces.TransferControl``.

A harvested fruit is a WP23 pool body whose equality is off (clip cut or twist release). The transfer moves it
KINEMATICALLY (assisted register: ``kinematic_transfer``), one item at a time, only with the arm at PARK and the gait
stopped:

    place(d, fruit_id)   from the catcher at the PARK TCP (CoM within HANDOFF_RADIUS_M, else not_at_catcher) into the
                         next free tray cell, stalk-down,
                         one fruit per cell (mango, guava); a papaya lies along x in the rear cradle (one per trip).
                         Path: lift to the clearance height, carry above the target, lower at <= 0.3 m/s.
    seated (rider)       the fruit rides its cell: pose and velocity are written from the trunk pose every
                         ``RIDER_EVERY`` physics steps, gravity is cancelled with ``xfrc_applied`` and its contacts are
                         off (contype / conaffinity 0), so it never pushes the robot. Its mass goes into the WP1
                         ``ballast`` body instead (body_mass / body_ipos / inertia = delivered sum, cap 3 kg).
    unload(d)            cell by cell, then the cradle: carry each item above its planned crate spot, lower it to
                         ``RELEASE_GAP_M`` above the crate floor, restore its contacts and let physics drop it. The
                         impact speed (pre-contact speed) and the rest pose are measured: CoM inside the crate AABB
                         after settling -> ``delivered`` with per-tree attribution, else ``dropped``.

Interlocks (checked when a command starts and every physics step while busy): e-stop / runner hold -> freeze
(the item stays where it is, never dropped; stays paused until ``resume``); arm not at PARK, gait moving
or (during an unload) the WP24 crown not stowed -> interlock pause, resumed by itself when they hold again; while an
unload runs the WP24 mast also refuses raises (hook gate, ``hooks/transfer.py``). ``pause`` / ``resume`` /
``abort`` are the operator controls (``abort`` of a place finishes the place: the post-commit block is atomic;
``abort`` of an unload returns the item in the air to its cell and drops the queue).

Truth checks: ``tray_contact_pairs`` (fruit-on-fruit, ``mj_geomDistance`` < ``CONTACT_TOL_M`` between any two tray
items), counted for the moving item every step it moves and for the whole tray each time an item is seated.

State: ``to_dict`` / ``load_dict`` (JSON-ready, bit-exact), packed by the ``transfer`` sim-state provider of
``atharv_nandi/sim/hooks/transfer.py`` together with the ballast fields. The rider contact flags and the trimmed pedicel live in
pool-slot fields, which WP23's ``pool`` provider already packs.
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

from atharv_nandi.core import dock_logic
from atharv_nandi.core.interfaces import GateDecision
from atharv_nandi.sim.world.modules import fruit_pool as pool_mod

# ---------------------------------------------------------------------- constants (ASSUMED unless noted)
N_CELLS = 8
CELL_SITES = tuple(f"tray_cell_{k}" for k in range(N_CELLS))
CRADLE_SITE = "papaya_cradle"
CRADLE_RAILS = ("papaya_cradle_l", "papaya_cradle_r")
BASKET_BODY = "dorsal_basket"
BASKET_FLOOR = "basket_floor"
BALLAST_BODY = "ballast"
GANTRY_RAIL = "gantry_rail"
CRATE_SITE = "crate_aabb"
CRATE_BODY = "crate"
ARM_YAW_JOINT = "arm_yaw"

PAYLOAD_CAP_KG = 3.0            # OS 2.4 / WP3: ballast cap 3 kg
PAPAYA_PER_TRIP = 1             # OS 2.4: one papaya per trip, rear cradle
V_CARRY_MPS = 0.45              # carry speed of every kinematic segment (<= the 0.5 m/s papaya limit)
V_LOWER_MPS = 0.30              # last (lowering) segment into a cell, the cradle or the crate
PAPAYA_V_MAX_MPS = 0.5          # OS 2.4: colour-break papaya placement speed <= 0.5 m/s
IMPACT_MAX_MPS = 2.0            # OS 1.2 step 11 / O-20: impact < 2 m/s
PAPAYA_IMPACT_MAX_MPS = 0.5     # papaya <= 0.5 m/s
RELEASE_GAP_M = 0.008           # item bottom above the crate floor at release: free fall sqrt(2 g h) = 0.40 m/s
CLEAR_MARGIN_M = 0.02           # carry height margin above the highest obstacle (tray items, gantry rail, crate wall)
SEAT_GAP_M = 0.001              # fruit bottom above the tray floor when seated
CONTACT_TOL_M = 0.0             # fruit-on-fruit contact: geom distance below this (touch or penetration)
SETTLE_V_MPS = 0.05             # crate settle: speed below this ...
SETTLE_HOLD_S = 0.10            # ... for this long
SETTLE_TIMEOUT_S = 2.0          # after release: decided by then whatever the speed
ARM_PARK = (0.0, 0.0, 0.1, -1.8, 1.4)   # gantry, yaw, shoulder, elbow, wrist = harvest_logic.REST_ARM[:5]
ARM_PARK_JOINTS = ("gantry_slide", "arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist_pitch")
ARM_PARK_TOL = (0.01, 0.05, 0.05, 0.05, 0.05)   # m for the gantry, rad for the rest
GAIT_STOP_V_MPS = 0.03          # trunk horizontal speed below this = gait stopped
GAIT_STOP_W_RPS = 0.20          # and trunk angular rate below this
REACH_MAX_M = 0.78              # crate spots within 0.8 x 0.98 m (OS 2.2 reach at 0.4 m, 80 % rule) of the arm yaw axis
RIDER_EVERY = 5                 # rider pose/velocity writes every 5 physics steps (100 Hz at dt 0.002)
CRATE_PITCH_M = 0.095           # crate floor grid for mango / guava (one layer: pool fruit never touch each other)
ASSISTED_TAG = "kinematic_transfer"
HANDOFF_SITE = "tcp_range"      # the arm TCP site; at PARK the tool's catcher holds the cut fruit next to it
HANDOFF_FALLBACK_B = (0.14, 0.0, 0.846)   # PARK TCP in the basket frame (orchard_demo keyframe) if the site is missing
HANDOFF_RADIUS_M = 0.25         # fruit CoM within this of the PARK TCP: tool reach (clipper TCP 0.12 m past the
                                # handle) + catcher cup + the largest fruit half length (papaya 0.113 m)

INTERLOCK_PAUSES = ("arm_not_parked", "gait_moving", "crown_gate")
STATE_VERSION = 1
REASONS = ("estop", "arm_not_parked", "gait_moving", "busy", "payload_full", "still_attached", "unknown_item",
           "not_at_catcher", "wrong_target", "no_clearance", "no_crate", "crate_out_of_reach", "over_capacity",
           "empty", "crown_gate")
CROWN_STOWED_EXT_M = dock_logic.CROWN_STOWED_EXT_M   # MISSION-PRESETS 9 precondition: crown stowed = WP24 mast
                                # extension at most this; one threshold with WP13 (its park / unload decision)


class TransferError(RuntimeError):
    pass


# ---------------------------------------------------------------------- small pose helpers
def _qmul(a, b) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def _qconj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _qmat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def _qnorm(q) -> np.ndarray:
    q = np.asarray(q, float)
    return q / max(np.linalg.norm(q), 1e-12)


def _q_from_to(a, b) -> np.ndarray:
    """Shortest rotation taking unit vector a to unit vector b."""
    a = np.asarray(a, float) / max(np.linalg.norm(a), 1e-12)
    b = np.asarray(b, float) / max(np.linalg.norm(b), 1e-12)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-12:
        ax = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0.0, 1.0, 0.0])
        ax /= np.linalg.norm(ax)
        return np.array([0.0, *ax])
    ax = np.cross(a, b)
    q = np.array([1.0 + c, *ax])
    return _qnorm(q)


def _slerp(q0, q1, s: float) -> np.ndarray:
    q0, q1 = _qnorm(q0), _qnorm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return _qnorm(q0 + s * (q1 - q0))
    th = math.acos(min(1.0, dot))
    return (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)


def _q_angle_axis(q0, q1) -> Tuple[float, np.ndarray]:
    """Rotation from q0 to q1 expressed in q0's frame: (angle, axis)."""
    dq = _qmul(_qconj(_qnorm(q0)), _qnorm(q1))
    if dq[0] < 0:
        dq = -dq
    s = float(np.linalg.norm(dq[1:]))
    if s < 1e-12:
        return 0.0, np.array([0.0, 0.0, 1.0])
    return 2.0 * math.atan2(s, float(dq[0])), dq[1:] / s


def _fl(a) -> List[float]:
    return [float(v) for v in np.asarray(a, float).reshape(-1)]


def ellipsoid_half_extents(R: np.ndarray, semi) -> np.ndarray:
    """Half extents along the parent axes of an ellipsoid with semi-axes ``semi`` rotated by R (parent <- body)."""
    s = np.asarray(semi, float)
    return np.sqrt(((np.asarray(R, float) * s[None, :]) ** 2).sum(axis=1))


# ---------------------------------------------------------------------- gate (pure)
@dataclass(frozen=True)
class GateInputs:
    arm_parked: bool = True
    gait_stopped: bool = True
    holding: bool = False
    crown_stowed: bool = True     # WP24 mast extension <= CROWN_STOWED_EXT_M (True in models without the mast)


def interlock(inputs: GateInputs) -> GateDecision:
    """MISSION-PRESETS 9 interlocks: e-stop first, then arm PARK, then gait stopped."""
    if inputs.holding:
        return GateDecision.refuse("transfer", "estop")
    if not inputs.arm_parked:
        return GateDecision.refuse("transfer", "arm_not_parked")
    if not inputs.gait_stopped:
        return GateDecision.refuse("transfer", "gait_moving")
    return GateDecision.ok("transfer")


def busy_gate(inputs: GateInputs, unloading: bool) -> GateDecision:
    """The interlocks checked every physics step while busy: ``interlock`` and, during an unload, crown stowed
    (MISSION-PRESETS 9: no arm swing over the crate with the crown up; pauses, resumes by itself once stowed)."""
    gate = interlock(inputs)
    if gate and unloading and not inputs.crown_stowed:
        return GateDecision.refuse("transfer", "crown_gate")
    return gate


# ---------------------------------------------------------------------- kinematic path
class Path:
    """Piecewise-linear pose path with per-segment speed and orientation slerped over the path length.

    Waypoints and speeds are in one frame (the basket frame for every transfer). ``pose(s)`` / ``vel(s)`` at path
    time s in [0, duration]."""

    def __init__(self, points: Sequence[Sequence[float]], quats: Sequence[Sequence[float]], speeds: Sequence[float]):
        self.p = np.asarray(points, float).reshape(-1, 3)
        self.q = np.asarray([_qnorm(q) for q in quats], float).reshape(-1, 4)
        self.v = np.asarray(speeds, float).reshape(-1)
        if len(self.p) < 2 or len(self.q) != len(self.p) or len(self.v) != len(self.p) - 1:
            raise TransferError("path needs >= 2 points, one quat per point and one speed per segment")
        if np.any(self.v <= 0) or not np.all(np.isfinite(self.v)):
            raise TransferError("segment speeds must be > 0")
        self.len = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.dt = np.maximum(self.len / self.v, 0.02)         # >= 20 ms per segment (pure rotations too)
        self.t = np.concatenate([[0.0], np.cumsum(self.dt)])

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def _seg(self, s: float) -> Tuple[int, float]:
        s = min(max(float(s), 0.0), self.duration)
        i = int(min(np.searchsorted(self.t, s, side="right") - 1, len(self.dt) - 1))
        return i, (s - self.t[i]) / self.dt[i]

    def pose(self, s: float) -> Tuple[np.ndarray, np.ndarray]:
        i, u = self._seg(s)
        return self.p[i] + u * (self.p[i + 1] - self.p[i]), _slerp(self.q[i], self.q[i + 1], u)

    def vel(self, s: float) -> Tuple[np.ndarray, np.ndarray]:
        """(linear, angular) velocity in the path frame; zero at and after the end."""
        if s >= self.duration:
            return np.zeros(3), np.zeros(3)
        i, u = self._seg(s)
        lin = (self.p[i + 1] - self.p[i]) / self.dt[i]
        ang, ax = _q_angle_axis(self.q[i], self.q[i + 1])
        q = _slerp(self.q[i], self.q[i + 1], u)
        w = _qmat(q) @ (ax * ang / self.dt[i])            # axis is constant in the moving frame of a slerp
        return lin, w

    def max_speed(self) -> float:
        return float(np.max(self.len / self.dt))

    def last_speed(self) -> float:
        return float(self.len[-1] / self.dt[-1])

    def to_dict(self) -> dict:
        return {"p": _fl(self.p), "q": _fl(self.q), "v": _fl(self.v)}

    @classmethod
    def from_dict(cls, d: dict) -> "Path":
        return cls(np.asarray(d["p"], float).reshape(-1, 3), np.asarray(d["q"], float).reshape(-1, 4), d["v"])


# ---------------------------------------------------------------------- model layout
class TrayLayout:
    """Names -> ids and the compiled geometry of the tray, cradle, ballast and crate (basket frame)."""

    def __init__(self, m: mujoco.MjModel):
        def oid(kind, name, required=True):
            i = mujoco.mj_name2id(m, kind, name)
            if i < 0 and required:
                raise TransferError(f"model has no {name!r}: not an Atharv Nandi model with the WP1 tray")
            return i

        S, B, G = mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM
        self.basket = oid(B, BASKET_BODY)
        self.ballast = oid(B, BALLAST_BODY)
        self.cells = [oid(S, n) for n in CELL_SITES]
        self.cradle = oid(S, CRADLE_SITE)
        self.rails = [oid(G, n) for n in CRADLE_RAILS]
        self.floor = oid(G, BASKET_FLOOR)
        for sid in self.cells + [self.cradle]:
            if m.site_bodyid[sid] != self.basket:
                raise TransferError("tray sites must be on dorsal_basket")
        # chain basket -> root free body (fixed bodies in between)
        chain, b = [], self.basket
        while b > 0 and m.body_jntnum[b] == 0:
            chain.append(b)
            b = int(m.body_parentid[b])
        if b <= 0 or m.jnt_type[m.body_jntadr[b]] != mujoco.mjtJoint.mjJNT_FREE:
            raise TransferError("dorsal_basket must hang off the free trunk body through fixed bodies")
        self.trunk = b
        self.trunk_qadr = int(m.jnt_qposadr[m.body_jntadr[b]])
        self.trunk_dadr = int(m.jnt_dofadr[m.body_jntadr[b]])
        p, q = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        for bb in reversed(chain):                       # trunk -> ... -> basket
            p = p + _qmat(q) @ np.array(m.body_pos[bb])
            q = _qmul(q, m.body_quat[bb])
        self.basket_in_trunk = (p, q)
        self.ancestors = []                              # ballast and every ancestor up to the world (subtreemass)
        b = self.ballast
        while b > 0:
            self.ancestors.append(b)
            b = int(m.body_parentid[b])
        # floor and cell bounds (basket frame)
        self.floor_top = float(m.geom_pos[self.floor][2] + m.geom_size[self.floor][2])
        self.cell_pos = [np.array(m.site_pos[s]) for s in self.cells]
        self.cell_half = [self._cell_half(m, k) for k in range(N_CELLS)]
        ry = [float(m.geom_pos[g][1]) for g in self.rails]
        sy = [float(m.geom_size[g][1]) for g in self.rails]
        hi, lo = (0, 1) if ry[0] > ry[1] else (1, 0)
        self.rail_half_gap = 0.5 * ((ry[hi] - sy[hi]) - (ry[lo] + sy[lo]))
        self.rail_mid_y = 0.5 * (ry[0] + ry[1])
        self.rail_top = max(float(m.geom_pos[g][2] + m.geom_size[g][2]) for g in self.rails)
        self.cradle_pos = np.array(m.site_pos[self.cradle])
        self.cradle_half_len = max(float(m.geom_size[g][0]) for g in self.rails)
        # obstacle top in the basket frame: basket walls and the gantry rail (trunk geom)
        top = 0.0
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == self.basket and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                top = max(top, float(m.geom_pos[g][2] + m.geom_size[g][2]))
        rail = oid(G, GANTRY_RAIL, required=False)
        if rail >= 0:
            bp, bq = self.basket_in_trunk
            pr = _qmat(bq).T @ (np.array(m.geom_pos[rail]) - bp)
            top = max(top, float(pr[2] + m.geom_size[rail][2]))
        self.obstacle_top = top
        self.ballast_pos = np.array(m.body_pos[self.ballast])       # ballast frame origin in the basket frame
        self.crate_site = oid(S, CRATE_SITE, required=False)
        self.crate_body = oid(B, CRATE_BODY, required=False)
        self.crate_geoms = [g for g in range(m.ngeom) if self.crate_body >= 0 and m.geom_bodyid[g] == self.crate_body]
        self.arm_yaw_body = -1
        j = oid(mujoco.mjtObj.mjOBJ_JOINT, ARM_YAW_JOINT, required=False)
        if j >= 0:
            self.arm_yaw_body = int(m.jnt_bodyid[j])
        self.park_qadr = []
        for n in ARM_PARK_JOINTS:
            j = oid(mujoco.mjtObj.mjOBJ_JOINT, n, required=False)
            self.park_qadr.append(int(m.jnt_qposadr[j]) if j >= 0 else -1)
        self.handoff_site = oid(S, HANDOFF_SITE, required=False)

    def _cell_half(self, m, k: int) -> np.ndarray:
        """(half x, half y) of cell k from the nearest walls around its site (basket frame)."""
        c = np.array(m.site_pos[self.cells[k]])
        xs_lo, xs_hi, ys_lo, ys_hi = [], [], [], []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != self.basket or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX or g == self.floor:
                continue
            p, s = np.array(m.geom_pos[g]), np.array(m.geom_size[g])
            if abs(c[1] - p[1]) <= s[1] and s[0] < s[1]:        # x-wall spanning the cell's y
                (xs_lo if p[0] < c[0] else xs_hi).append((p[0] + s[0]) if p[0] < c[0] else (p[0] - s[0]))
            if abs(c[0] - p[0]) <= s[0] and s[1] < s[0]:        # y-wall spanning the cell's x
                (ys_lo if p[1] < c[1] else ys_hi).append((p[1] + s[1]) if p[1] < c[1] else (p[1] - s[1]))
        hx = min(c[0] - max(xs_lo), min(xs_hi) - c[0]) if xs_lo and xs_hi else 0.04
        hy = min(c[1] - max(ys_lo), min(ys_hi) - c[1]) if ys_lo and ys_hi else 0.04
        return np.array([hx, hy])

    # -------------------------------------------------------------- frames
    def basket_pose(self, d) -> Tuple[np.ndarray, np.ndarray]:
        """Basket frame in the world from the trunk qpos (exact for the current qpos, unlike xpos)."""
        a = self.trunk_qadr
        tp, tq = np.array(d.qpos[a:a + 3]), _qnorm(d.qpos[a + 3:a + 7])
        bp, bq = self.basket_in_trunk
        return tp + _qmat(tq) @ bp, _qmul(tq, bq)

    def trunk_twist(self, d) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(trunk position, linear velocity world, angular velocity world)."""
        a, v = self.trunk_qadr, self.trunk_dadr
        tq = _qnorm(d.qpos[a + 3:a + 7])
        return np.array(d.qpos[a:a + 3]), np.array(d.qvel[v:v + 3]), _qmat(tq) @ np.array(d.qvel[v + 3:v + 6])

    def arm_parked(self, d) -> bool:
        for adr, ref, tol in zip(self.park_qadr, ARM_PARK, ARM_PARK_TOL):
            if adr >= 0 and abs(float(d.qpos[adr]) - ref) > tol:
                return False
        return True

    def handoff_point(self, d) -> np.ndarray:
        """World position of the hand-over point: the arm TCP (the transfer runs only with the arm at PARK, so this
        is the PARK TCP, where the tool's catcher holds the cut fruit). Needs current site poses (after mj_forward /
        mj_step)."""
        if self.handoff_site >= 0:
            return np.array(d.site_xpos[self.handoff_site])
        bp, bq = self.basket_pose(d)
        return bp + _qmat(bq) @ np.asarray(HANDOFF_FALLBACK_B, float)

    def gait_stopped(self, d) -> bool:
        _, v, w = self.trunk_twist(d)
        return math.hypot(float(v[0]), float(v[1])) <= GAIT_STOP_V_MPS and float(np.linalg.norm(w)) <= GAIT_STOP_W_RPS


# ---------------------------------------------------------------------- controller
class TransferController:
    """Tray / cradle / unload transfers for one model (``TransferControl``). Drive it with ``substep(d, inputs)``
    before every ``mj_step`` (the hook does); commands are ``place``, ``unload``, ``pause``, ``resume``, ``abort``."""

    def __init__(self, m: mujoco.MjModel, pool):
        self.m = m
        self.pool = pool
        self.L = TrayLayout(m)
        self.dt = float(m.opt.timestep)
        self._ballast0 = {"mass": float(m.body_mass[self.L.ballast]), "ipos": np.array(m.body_ipos[self.L.ballast]),
                          "inertia": np.array(m.body_inertia[self.L.ballast]),
                          "iquat": np.array(m.body_iquat[self.L.ballast])}
        self._subtree0 = {b: float(m.body_subtreemass[b]) for b in self.L.ancestors}
        # compiled dof_M0 of every pool slot (MuJoCo uses dof_M0 for "simple" free bodies, see _sync_m0)
        self._m0_template = {} if pool is None else {
            s.k: np.array(m.dof_M0[s.dadr:s.dadr + 6]) for s in pool.slots}
        self._assign = {} if pool is None else pool_mod.model_state(m)["assign"]   # live pool assignment (no copy)
        self._seat_cache: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
        self.reset_state()

    # -------------------------------------------------------------- state
    def reset_state(self):
        self.items: Dict[str, dict] = {}       # fruit_id -> item record (tray, moving, falling, crate, dropped)
        self.cells: List[Optional[str]] = [None] * N_CELLS
        self.cradle: Optional[str] = None
        self.job: Optional[dict] = None        # current kinematic / falling job
        self.queue: List[str] = []             # unload queue (fruit ids)
        self.unloading = False
        self.paused: Optional[str] = None      # None | "operator" | "estop" | interlock reason
        self.events: List[dict] = []
        self.event_seq = getattr(self, "event_seq", 0)   # keeps counting over resets (publishers track it)
        self.fruit_contacts = 0                # truth: fruit-on-fruit contact events in the tray
        self.contact_pairs: List[List[str]] = []
        self.steps = 0
        self.crate_spots: Dict[str, List[float]] = {}   # fruit_id -> crate floor spot (crate frame)
        self.refusals = 0
        self.max_speed_mps = 0.0
        self.max_papaya_speed_mps = 0.0
        self._seat_cache = {}

    def reset(self, d=None):
        """World reset: contacts and gravity of touched slots back, ballast back to the compiled value, empty."""
        for fid, it in list(self.items.items()):
            self._restore_slot(d, it)
        for k in self._m0_template:
            self._restore_m0(k)
        self._write_ballast(self._ballast0["mass"], self._ballast0["ipos"], self._ballast0["inertia"],
                            self._ballast0["iquat"])
        self.reset_state()

    def _event(self, t: float, kind: str, **kw):
        self.event_seq += 1
        e = {**kw, "seq": self.event_seq, "t": round(float(t), 4), "event": kind}   # reserved keys win
        self.events.append(e)
        if len(self.events) > 200:
            del self.events[:-200]
        return e

    # -------------------------------------------------------------- TransferControl
    def pause(self, reason: str = "operator") -> None:
        if self.paused != "estop":
            self.paused = reason

    def resume(self) -> None:
        self.paused = None

    def busy(self) -> bool:
        return self.job is not None or bool(self.queue)

    def abort(self, d=None) -> GateDecision:
        """Drop the unload queue. A place is atomic after the commit and finishes; an unload item still carried is
        brought back to its cell; an item already released falls and is decided as usual."""
        for fid in self.queue:                           # queued items stay in the tray: their crate spots are free
            self.crate_spots.pop(fid, None)
        self.queue = []
        self.unloading = False
        j = self.job
        if j is not None and j["op"] == "unload" and j["phase"] == "carry" and d is not None:
            it = self.items[j["fruit_id"]]
            p_now, q_now = j["path_obj"].pose(j["s"])
            tgt_p, tgt_q = self._seat_pose(it)
            path = self._path_to(p_now, q_now, tgt_p, tgt_q, it, extra_top=0.0)
            self.job = {"op": "return", "fruit_id": it["fruit_id"], "phase": "carry", "s": 0.0,
                        "path": path.to_dict(), "path_obj": path}
            it["state"] = "moving"
            self.crate_spots.pop(it["fruit_id"], None)    # going back to its cell, not into the crate
        self.paused = None if self.paused in (None, "operator") else self.paused
        return GateDecision.ok("transfer_abort")

    # -------------------------------------------------------------- queries
    def payload_kg(self) -> float:
        """Mass on the robot: seated items and the item being carried."""
        return float(sum(it["mass_kg"] for it in self.items.values() if it["state"] in ("seated", "moving")))

    def tray_items(self) -> List[dict]:
        return [dict(self.items[f]) for f in self.cells if f] + ([dict(self.items[self.cradle])] if self.cradle else [])

    def free_cells(self) -> List[int]:
        return [k for k, f in enumerate(self.cells) if f is None]

    def per_tree(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for it in self.items.values():
            r = out.setdefault(it["tree_id"], {"tray_n": 0, "tray_kg": 0.0, "crate_n": 0, "crate_kg": 0.0,
                                               "dropped_n": 0, "fruit": []})
            r["fruit"].append(it["fruit_id"])
            if it["state"] in ("seated", "moving"):
                r["tray_n"] += 1
                r["tray_kg"] = round(r["tray_kg"] + it["mass_kg"], 4)
            elif it["state"] == "crate":
                r["crate_n"] += 1
                r["crate_kg"] = round(r["crate_kg"] + it["mass_kg"], 4)
            elif it["state"] == "dropped":
                r["dropped_n"] += 1
        return out

    def crate_summary(self) -> dict:
        crate = [it for it in self.items.values() if it["state"] == "crate"]
        return {"n": len(crate), "kg": round(sum(it["mass_kg"] for it in crate), 4),
                "dropped": sum(1 for it in self.items.values() if it["state"] == "dropped"),
                "max_impact_mps": round(max([it.get("impact_mps") or 0.0 for it in crate] or [0.0]), 4),
                "per_tree": {k: {"n": v["crate_n"], "kg": v["crate_kg"]} for k, v in self.per_tree().items()
                             if v["crate_n"]}}

    def release_crate_slots(self, d) -> List[str]:
        """Give the pool slots of unloaded items (crate or dropped) back to the pool as harvested; their records and
        per-tree attribution stay. Call before the next tree needs slots (10 slots for the whole orchard)."""
        if self.pool is None:
            return []
        out = []
        for it in self.items.values():
            if it["state"] in ("crate", "dropped") and not it.get("recycled") and not self._gone(it):
                self.pool.recycle(d, it["slot"], harvested=True, forward=False)
                self._restore_m0(it["slot"])
                it["recycled"] = True
                self.crate_spots.pop(it["fruit_id"], None)     # the body is gone: its spot is free again
                out.append(it["fruit_id"])
        if out:
            mujoco.mj_forward(self.m, d)
        return out

    # -------------------------------------------------------------- geometry of an item
    def _semi(self, slot: int) -> np.ndarray:
        return np.array(self.m.geom_size[self.pool.slots[slot].fruit], float)

    def _seat_local_quat(self, it: dict) -> np.ndarray:
        a = np.asarray(it["stem_axis_body"], float)
        if it["where"] == "cradle":
            return _q_from_to(a, [1.0, 0.0, 0.0])       # papaya lies along the basket x, stem to the front
        return _q_from_to(a, [0.0, 0.0, -1.0])          # stalk-down

    def _seat_pose(self, it: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Seated pose of the item in the basket frame (cached per item: fixed by its slot geometry and seat)."""
        key = (it["fruit_id"], it["where"], it["slot"])
        hit = self._seat_cache.get(key)
        if hit is None:
            p, q = self._seat_pose_uncached(it)
            hit = self._seat_cache[key] = (p, q, _qmat(q))
        return hit[0], hit[1]

    def _seat_pose_uncached(self, it: dict) -> Tuple[np.ndarray, np.ndarray]:
        L = self.L
        q = self._seat_local_quat(it)
        semi = self._semi(it["slot"])
        h = ellipsoid_half_extents(_qmat(q), semi)
        if it["where"] == "cradle":
            a, b, g = h[1], h[2], L.rail_half_gap
            z = L.rail_top + b * math.sqrt(max(0.0, 1.0 - (g / a) ** 2)) if a > g else L.floor_top + h[2]
            return np.array([L.cradle_pos[0], L.rail_mid_y, z]), q
        c = L.cell_pos[it["where"]]
        return np.array([c[0], c[1], L.floor_top + h[2] + SEAT_GAP_M]), q

    def _fits(self, slot: int, where) -> Optional[str]:
        """None when the fruit fits the cell / rests on the cradle rails, else a reason."""
        it = {"stem_axis_body": self._stem_axis_body(slot), "where": where, "slot": slot}
        h = ellipsoid_half_extents(_qmat(self._seat_local_quat(it)), self._semi(slot))
        if where == "cradle":
            return None if h[1] > self.L.rail_half_gap else "no_clearance"
        hx, hy = self.L.cell_half[where]
        return None if (h[0] <= hx and h[1] <= hy) else "no_clearance"

    def _stem_axis_body(self, slot: int) -> List[float]:
        fid = self.pool.assignments[slot]["fruit_id"]
        return [float(v) for v in self.pool.fruit[fid]["stem_axis"]]   # fruit frame = world axes at assignment

    # -------------------------------------------------------------- paths
    def _tray_top(self, exclude: Optional[str] = None) -> float:
        top = self.L.obstacle_top
        for f in [*self.cells, self.cradle]:
            if f and f != exclude and self.items[f]["state"] == "seated":
                it = self.items[f]
                p, q = self._seat_pose(it)
                top = max(top, float(p[2] + ellipsoid_half_extents(_qmat(q), self._semi(it["slot"]))[2]))
        return top

    def _path_to(self, p0, q0, p1, q1, it: dict, extra_top: float = 0.0, v_lower: float = V_LOWER_MPS) -> Path:
        """Lift from p0 to the carry height, carry above p1, lower to p1 (all in the basket frame)."""
        semi = self._semi(it["slot"])
        h0 = ellipsoid_half_extents(_qmat(q0), semi)[2]
        h1 = ellipsoid_half_extents(_qmat(q1), semi)[2]
        zc = max(self._tray_top(exclude=it["fruit_id"]), extra_top) + max(h0, h1) + CLEAR_MARGIN_M
        # start above the carry height (the catcher at PARK): the carry runs straight down to the carry height above
        # the target (both ends above every obstacle, so the whole segment is) and only the last segment is slow
        a = np.array([p0[0], p0[1], max(zc, float(p0[2]))])
        b = np.array([p1[0], p1[1], max(zc, float(p1[2]))])
        qm = q1                                            # orientation reached by the end of the carry
        return Path([p0, a, b, p1], [q0, _slerp(q0, q1, 0.5), qm, q1], [V_CARRY_MPS, V_CARRY_MPS, v_lower])

    # -------------------------------------------------------------- commands
    def _refuse(self, t: float, reason: str, **kw) -> GateDecision:
        self.refusals += 1
        self._event(t, "refused", reason=reason, **kw)
        return GateDecision.refuse("transfer", reason, **{k: v for k, v in kw.items() if isinstance(v, (int, float))})

    def place(self, d, fruit_id: str, target: Any = "auto", inputs: GateInputs = GateInputs()) -> GateDecision:
        """Start the kinematic transfer of a released pool fruit into the tray (``target`` auto | cell k | cradle)."""
        t = float(d.time)
        if self.pool is None:
            return self._refuse(t, "unknown_item", fruit_id=fruit_id, detail="no fruit pool")
        gate = interlock(inputs)
        if not gate:
            return self._refuse(t, gate.reason, fruit_id=fruit_id)
        if self.busy():
            return self._refuse(t, "busy", fruit_id=fruit_id)
        slot = self.pool.slot_of(fruit_id)
        if slot is None or fruit_id in self.items:
            return self._refuse(t, "unknown_item", fruit_id=fruit_id)
        if self.pool.is_attached(d, slot):
            return self._refuse(t, "still_attached", fruit_id=fruit_id)
        s = self.pool.slots[slot]
        # hand-over: the released fruit must be held at the catcher next to the PARK TCP (OS 1.2 step 11: never
        # faked; a fruit lying anywhere else is not carried through trees and the robot)
        p0w = np.array(d.qpos[s.qadr:s.qadr + 3])
        gap = float(np.linalg.norm(p0w - self.L.handoff_point(d)))
        if not math.isfinite(gap) or gap > HANDOFF_RADIUS_M:
            return self._refuse(t, "not_at_catcher", fruit_id=fruit_id, dist_m=round(gap, 4),
                                limit_m=HANDOFF_RADIUS_M)
        f = self.pool.fruit[fruit_id]
        mass = float(self.m.body_mass[s.body])
        papaya = f["crop"] == "papaya"
        if target == "auto":
            target = "cradle" if papaya else (self.free_cells()[0] if self.free_cells() else None)
            if target is None:
                return self._refuse(t, "payload_full", fruit_id=fruit_id, detail="no free cell")
        if target == "cradle":
            if not papaya:
                return self._refuse(t, "wrong_target", fruit_id=fruit_id, detail=f"{f['crop']} goes into a cell")
            if self.cradle is not None:
                return self._refuse(t, "payload_full", fruit_id=fruit_id, detail="papaya_per_trip")
        elif isinstance(target, int) and not isinstance(target, bool) and 0 <= target < N_CELLS:
            if papaya:
                return self._refuse(t, "wrong_target", fruit_id=fruit_id, detail="papaya goes into the cradle")
            if self.cells[target] is not None:
                return self._refuse(t, "payload_full", fruit_id=fruit_id, detail=f"cell {target}")
        else:
            return self._refuse(t, "wrong_target", fruit_id=fruit_id, detail=f"target {target!r}")
        if self.payload_kg() + mass > PAYLOAD_CAP_KG + 1e-9:
            return self._refuse(t, "payload_full", fruit_id=fruit_id, payload_kg=self.payload_kg(), mass_kg=mass)
        why = self._fits(slot, target)
        if why:
            return self._refuse(t, why, fruit_id=fruit_id, detail=f"does not fit {target}")
        it = {"fruit_id": fruit_id, "tree_id": f["tree_id"], "crop": f["crop"], "slot": slot, "mass_kg": round(mass, 6),
              "where": target, "state": "moving", "placed_t": None, "unloaded_t": None, "impact_mps": None,
              "stem_axis_body": self._stem_axis_body(slot), "assisted": [ASSISTED_TAG], "outcome": None,
              "place_speed_mps": None}
        self.items[fruit_id] = it
        if target == "cradle":
            self.cradle = fruit_id
        else:
            self.cells[target] = fruit_id
        self._trim_pedicel(slot)
        self._kinematic_slot(d, slot, on=True)
        bp, bq = self.L.basket_pose(d)
        q0w = _qnorm(d.qpos[s.qadr + 3:s.qadr + 7])
        p0 = _qmat(bq).T @ (p0w - bp)
        q0 = _qmul(_qconj(bq), q0w)
        p1, q1 = self._seat_pose(it)
        path = self._path_to(p0, q0, p1, q1, it)
        it["place_speed_mps"] = round(path.last_speed(), 4)
        self.job = {"op": "place", "fruit_id": fruit_id, "phase": "carry", "s": 0.0, "path": path.to_dict(),
                    "path_obj": path}
        self._note_speed(it, path)
        self._event(t, "place_start", fruit_id=fruit_id, where=target, tree_id=it["tree_id"])
        return GateDecision.ok("transfer", duration_s=path.duration)

    def unload(self, d, inputs: GateInputs = GateInputs()) -> GateDecision:
        """Unload every tray item into the crate, cell by cell, then the cradle (MISSION-PRESETS 9 step 2)."""
        t = float(d.time)
        gate = interlock(inputs)
        if not gate:
            return self._refuse(t, gate.reason)
        if not inputs.crown_stowed:          # MP 9 precondition: no arm swing over the crate with the crown up
            return self._refuse(t, "crown_gate")
        if self.busy():
            return self._refuse(t, "busy")
        if self.L.crate_site < 0:
            return self._refuse(t, "no_crate")
        order = [f for f in self.cells if f] + ([self.cradle] if self.cradle else [])
        if not order:
            return self._refuse(t, "empty")
        spots = self.plan_crate(d, order)
        if spots is None:
            return self._refuse(t, "over_capacity", n=len(order))
        reach = self.crate_reach_m(d, spots)
        if reach > REACH_MAX_M:
            return self._refuse(t, "crate_out_of_reach", reach_m=reach, limit_m=REACH_MAX_M)
        self.crate_spots.update(spots)
        self.queue = list(order)
        self.unloading = True
        self._event(t, "unload_start", n=len(order), reach_m=round(reach, 4))
        self._next_unload(d)
        return GateDecision.ok("transfer", n=len(order), reach_m=reach)

    # -------------------------------------------------------------- crate planning
    def crate_frame(self, d) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(centre world, rotation world <- crate, inner half extents) of the crate AABB site."""
        sid = self.L.crate_site
        return np.array(d.site_xpos[sid]), np.array(d.site_xmat[sid]).reshape(3, 3), np.array(self.m.site_size[sid])

    def crate_spots_local(self, d, side: float = 1.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Floor spots (crate frame, z = floor = -half_z): (small fruit grid, papaya strip). The papaya strip runs
        along the crate's long side ``side`` (+1 = +y), the small-fruit grid fills the rest; one layer, because pool
        fruit never collide with each other (stacked ones would interpenetrate)."""
        _, _, h = self.crate_frame(d)
        strip_w = 0.16
        sg = 1.0 if side >= 0 else -1.0
        small, papaya = [], []
        nx = int(math.floor(2 * h[0] / CRATE_PITCH_M + 1e-9))
        ny = int(math.floor((2 * h[1] - strip_w) / CRATE_PITCH_M + 1e-9))
        x0 = -h[0] + (2 * h[0] - nx * CRATE_PITCH_M) / 2
        for iy in range(ny):
            y = -sg * (h[1] - CRATE_PITCH_M * (iy + 0.5))
            for ix in range(nx):
                small.append(np.array([x0 + CRATE_PITCH_M * (ix + 0.5), y, -h[2]]))
        for x in (-h[0] / 2, h[0] / 2):
            papaya.append(np.array([x, sg * (h[1] - strip_w / 2), -h[2]]))
        return small, papaya

    def _arm_base(self, d) -> np.ndarray:
        return np.array(d.xpos[self.L.arm_yaw_body]) if self.L.arm_yaw_body >= 0 else self.L.basket_pose(d)[0]

    def plan_crate(self, d, order: Sequence[str]) -> Optional[Dict[str, List[float]]]:
        """Crate floor spot (crate frame) per item, nearest to the arm first; the papaya strip on the robot's side.
        None when the crate floor has no free spot left for an item."""
        c, R, _ = self.crate_frame(d)
        base = self._arm_base(d)
        side = float((R.T @ (base - c))[1])
        small, papaya = self.crate_spots_local(d, side)
        # spots taken by bodies in (or falling into) the crate; a spot of an item still on the robot is free
        used = [np.asarray(v, float) for k, v in self.crate_spots.items()
                if k in self.items and self.items[k]["state"] in ("falling", "crate", "dropped")]

        def dist(p):
            w = c + R @ p
            return math.hypot(w[0] - base[0], w[1] - base[1])

        out: Dict[str, List[float]] = {}
        for fid in order:
            cand = papaya if self.items[fid]["crop"] == "papaya" else small
            free = [p for p in cand if all(np.linalg.norm(p[:2] - u[:2]) > 1e-3 for u in used)]
            if not free:
                return None
            p = min(free, key=dist)
            out[fid] = _fl(p)
            used.append(p)
        return out

    def _spot_local(self, d, fid: str) -> np.ndarray:
        return np.asarray(self.crate_spots[fid], float)

    def _crate_release_pose(self, d, it: dict) -> Tuple[np.ndarray, np.ndarray]:
        """World release pose above the item's crate spot (seated orientation kept, bottom RELEASE_GAP_M high)."""
        c, R, _ = self.crate_frame(d)
        a = it["stem_axis_body"]
        # small fruit stay stalk-down; a papaya lies along the crate's long (x) axis
        qw = _q_from_to(a, R[:, 0]) if it["where"] == "cradle" else _q_from_to(a, [0.0, 0.0, -1.0])
        hz = ellipsoid_half_extents(_qmat(qw), self._semi(it["slot"]))[2]
        p = c + R @ self._spot_local(d, it["fruit_id"])
        p[2] += hz + RELEASE_GAP_M
        return p, qw

    def crate_reach_m(self, d, spots: Dict[str, List[float]]) -> float:
        """Worst horizontal distance from the arm yaw axis to a planned crate spot."""
        base = self._arm_base(d)
        c, R, _ = self.crate_frame(d)
        worst = 0.0
        for fid, k in spots.items():
            p = c + R @ np.asarray(k, float)
            worst = max(worst, float(math.hypot(p[0] - base[0], p[1] - base[1])))
        return worst

    def _next_unload(self, d):
        if not self.queue:
            self.unloading = False
            self._event(float(d.time), "unload_done", **self.crate_summary())
            return
        fid = self.queue.pop(0)
        it = self.items[fid]
        bp, bq = self.L.basket_pose(d)
        p0, q0 = self._seat_pose(it)
        pw, qw = self._crate_release_pose(d, it)
        p1 = _qmat(bq).T @ (pw - bp)
        q1 = _qmul(_qconj(bq), qw)
        c, R, h = self.crate_frame(d)
        crate_top_w = c[2] + h[2]
        crate_top_b = float((_qmat(bq).T @ (np.array([pw[0], pw[1], crate_top_w]) - bp))[2])
        path = self._path_to(p0, q0, p1, q1, it, extra_top=crate_top_b)
        it["state"] = "moving"
        self.job = {"op": "unload", "fruit_id": fid, "phase": "carry", "s": 0.0, "path": path.to_dict(),
                    "path_obj": path}
        self._note_speed(it, path)
        self._event(float(d.time), "unload_item", fruit_id=fid, tree_id=it["tree_id"])

    def _note_speed(self, it: dict, path: Path):
        self.max_speed_mps = max(self.max_speed_mps, path.max_speed())
        if it["crop"] == "papaya":
            self.max_papaya_speed_mps = max(self.max_papaya_speed_mps, path.max_speed())

    # -------------------------------------------------------------- model writes
    def _trim_pedicel(self, slot: int):
        """After the cut only the stalk from the stem end to cut_k is left on the fruit (visual capsule)."""
        s = self.pool.slots[slot]
        m = self.m
        a, b = np.array(m.site_pos[s.stem_end]), np.array(m.site_pos[s.cut_k])
        L = float(np.linalg.norm(b - a))
        r = float(m.geom_size[s.pedicel][0]) or 0.0025
        half = max(L / 2.0, 1e-4)
        m.geom_pos[s.pedicel] = (a + b) / 2.0
        m.geom_quat[s.pedicel] = _q_from_to([0, 0, 1], (b - a) if L > 1e-9 else [0, 0, 1])
        m.geom_size[s.pedicel] = [r, half, 0.0]
        m.geom_rbound[s.pedicel] = half + r
        m.geom_aabb[s.pedicel] = [0.0, 0.0, 0.0, r, r, half + r]

    def _kinematic_slot(self, d, slot: int, on: bool):
        s = self.pool.slots[slot]
        m = self.m
        if on:
            m.geom_contype[s.fruit] = m.geom_conaffinity[s.fruit] = 0
            m.body_contype[s.body] = m.body_conaffinity[s.body] = 0
        else:
            m.geom_contype[s.fruit], m.geom_conaffinity[s.fruit] = 1, 0
            m.body_contype[s.body], m.body_conaffinity[s.body] = 1, 0
            if d is not None:
                d.xfrc_applied[s.body] = 0.0

    def _sync_m0(self, slot: int):
        """Make the mass matrix of a pool slot match its fruit. The pool bodies are MuJoCo \"simple\" bodies, whose
        diagonal mass matrix comes from the compiled ``dof_M0``, which ``FruitPool.assign`` does not update (WP23
        change request): without this a 1.54 kg papaya falls at 100 m/s^2 (template 0.15 kg). Valid for a simple
        body (CoM at the body origin, principal axes = body axes)."""
        s = self.pool.slots[slot]
        m, a = self.m, s.dadr
        m.dof_M0[a:a + 3] = float(m.body_mass[s.body]) + m.dof_armature[a:a + 3]
        m.dof_M0[a + 3:a + 6] = m.body_inertia[s.body] + m.dof_armature[a + 3:a + 6]

    def _restore_m0(self, slot: int):
        if slot in self._m0_template:
            s = self.pool.slots[slot]
            self.m.dof_M0[s.dadr:s.dadr + 6] = self._m0_template[slot]

    def _restore_slot(self, d, it: dict):
        if self.pool is None:
            return
        if self.pool.assignments.get(it["slot"], {}).get("fruit_id") == it["fruit_id"]:
            self._kinematic_slot(d, it["slot"], on=False)

    def _write_ballast(self, mass, ipos, inertia, iquat):
        m, b = self.m, self.L.ballast
        m.body_mass[b] = float(mass)
        m.body_ipos[b] = ipos
        m.body_inertia[b] = inertia
        m.body_iquat[b] = iquat
        dm = float(mass) - self._ballast0["mass"]
        for a in self.L.ancestors:
            m.body_subtreemass[a] = self._subtree0[a] + dm

    def ballast_fields(self) -> dict:
        m, b = self.m, self.L.ballast
        return {"mass": float(m.body_mass[b]), "ipos": _fl(m.body_ipos[b]), "inertia": _fl(m.body_inertia[b]),
                "iquat": _fl(m.body_iquat[b])}

    def update_ballast(self):
        """Ballast = the seated tray items as point ellipsoids at their seats (ballast frame); 1 g base kept."""
        L = self.L
        m0 = self._ballast0["mass"]
        parts = [(m0, np.zeros(3), np.diag(self._ballast0["inertia"]))]
        for f in [*self.cells, self.cradle]:
            if not f or self.items[f]["state"] != "seated":
                continue
            it = self.items[f]
            p, q = self._seat_pose(it)
            R = _qmat(q)
            a, b, c = self._semi(it["slot"])
            mi = it["mass_kg"]
            I_body = np.diag([mi * (b * b + c * c) / 5.0, mi * (a * a + c * c) / 5.0, mi * (a * a + b * b) / 5.0])
            parts.append((mi, p - L.ballast_pos, R @ I_body @ R.T))
        M = sum(pm for pm, _, _ in parts)
        com = sum(pm * pr for pm, pr, _ in parts) / M
        I = np.zeros((3, 3))
        for pm, pr, Ic in parts:
            r = pr - com
            I += Ic + pm * (float(r @ r) * np.eye(3) - np.outer(r, r))
        w, V = np.linalg.eigh(0.5 * (I + I.T))
        if np.linalg.det(V) < 0:
            V[:, 0] = -V[:, 0]
        iq = np.zeros(4)
        mujoco.mju_mat2Quat(iq, V.reshape(-1))
        w = np.maximum(w, 1e-9)
        self._write_ballast(M, com, w, _qnorm(iq))

    # -------------------------------------------------------------- driving
    def _write_item(self, d, it: dict, p_b, q_b, v_rel_b=None, w_rel_b=None):
        """Write the item's free joint from a basket-frame pose (+ relative velocity) and the trunk twist."""
        s = self.pool.slots[it["slot"]]
        bp, bq = self.L.basket_pose(d)
        Rb = _qmat(bq)
        pw = bp + Rb @ np.asarray(p_b, float)
        qw = _qnorm(_qmul(bq, q_b))
        tp, tv, tw = self.L.trunk_twist(d)
        vw = tv + np.cross(tw, pw - tp)
        ww = tw.copy()
        if v_rel_b is not None:
            vw = vw + Rb @ v_rel_b
        if w_rel_b is not None:
            ww = ww + Rb @ w_rel_b
        d.qpos[s.qadr:s.qadr + 3] = pw
        d.qpos[s.qadr + 3:s.qadr + 7] = qw
        d.qvel[s.dadr:s.dadr + 3] = vw
        d.qvel[s.dadr + 3:s.dadr + 6] = _qmat(qw).T @ ww
        d.qacc_warmstart[s.dadr:s.dadr + 6] = 0.0
        d.xfrc_applied[s.body] = 0.0
        d.xfrc_applied[s.body, 2] = float(self.m.body_mass[s.body]) * -float(self.m.opt.gravity[2])

    def _gone(self, it: dict) -> bool:
        """The pool slot no longer holds this fruit (recycled by the task layer)."""
        if self.pool is None:
            return True
        a = self._assign.get(it["slot"])
        return a is None or a.get("fruit_id") != it["fruit_id"]

    def substep(self, d, inputs: GateInputs = GateInputs()) -> None:
        """Call before every ``mj_step``: interlocks, the current job, the riders (every ``RIDER_EVERY`` steps)."""
        if self.pool is None:
            return
        self.steps += 1
        t = float(d.time)
        cycle = (self.steps % RIDER_EVERY) == 0
        # interlocks while busy (e-stop latches the pause; arm / gait pauses clear by themselves)
        if self.busy():
            gate = busy_gate(inputs, self.unloading)
            if not gate:
                if gate.reason == "estop":
                    if self.paused != "estop":
                        self._event(t, "paused", reason="estop")
                        self.paused = "estop"
                elif self.paused is None or self.paused in INTERLOCK_PAUSES:
                    if self.paused != gate.reason:
                        self._event(t, "paused", reason=gate.reason)
                    self.paused = gate.reason
            elif self.paused in INTERLOCK_PAUSES:
                self._event(t, "resumed", reason=self.paused)
                self.paused = None
        j = self.job
        if j is not None:
            it = self.items.get(j["fruit_id"])
            if it is None or self._gone(it):
                self._drop_item(t, j["fruit_id"], "lost")
            elif j["phase"] == "carry":
                self._step_carry(d, j, it, t)
            elif j["phase"] == "fall":
                self._step_fall(d, j, it, t)
        if cycle:
            self._write_riders(d, t)

    def _write_riders(self, d, t):
        """Every seated item: pose and velocity from the trunk pose and twist (computed once per cycle)."""
        riders = []
        for f in [*self.cells, self.cradle]:
            if not f:
                continue
            it = self.items[f]
            if self._gone(it):
                self._drop_item(t, f, "lost")
                continue
            if it["state"] == "seated":
                riders.append(it)
        if not riders:
            return
        bp, bq = self.L.basket_pose(d)
        Rb = _qmat(bq)
        tp, tv, tw = self.L.trunk_twist(d)
        m, g = self.m, -float(self.m.opt.gravity[2])
        qw = np.zeros(4)
        wx, wy, wz = float(tw[0]), float(tw[1]), float(tw[2])
        w_b = Rb.T @ tw                                   # trunk rate in the basket frame
        for it in riders:
            s = self.pool.slots[it["slot"]]
            self._seat_pose(it)
            p_b, q_b, R_l = self._seat_cache[(it["fruit_id"], it["where"], it["slot"])]
            pw = bp + Rb @ p_b
            mujoco.mju_mulQuat(qw, bq, q_b)
            rx, ry, rz = float(pw[0] - tp[0]), float(pw[1] - tp[1]), float(pw[2] - tp[2])
            d.qpos[s.qadr:s.qadr + 3] = pw
            d.qpos[s.qadr + 3:s.qadr + 7] = qw
            d.qvel[s.dadr:s.dadr + 3] = (tv[0] + wy * rz - wz * ry, tv[1] + wz * rx - wx * rz, tv[2] + wx * ry - wy * rx)
            d.qvel[s.dadr + 3:s.dadr + 6] = R_l.T @ w_b
            d.qacc_warmstart[s.dadr:s.dadr + 6] = 0.0
            d.xfrc_applied[s.body] = (0.0, 0.0, float(m.body_mass[s.body]) * g, 0.0, 0.0, 0.0)

    def _step_carry(self, d, j, it, t):
        path = j["path_obj"]
        if self.paused is None:
            j["s"] = min(path.duration, j["s"] + self.dt)
        p, q = path.pose(j["s"])
        if self.paused is None:
            v, w = path.vel(j["s"])
        else:
            v, w = np.zeros(3), np.zeros(3)
        self._write_item(d, it, p, q, v, w)
        self._check_moving_contacts(d, it, t)
        if j["s"] >= path.duration - 1e-12 and self.paused is None:
            if j["op"] in ("place", "return"):
                it["state"] = "seated"
                if it["placed_t"] is None:
                    it["placed_t"] = round(t, 4)
                self.job = None
                self.update_ballast()
                self._check_tray_contacts(d, t)
                self._event(t, "seated", fruit_id=it["fruit_id"], where=it["where"], payload_kg=round(self.payload_kg(), 4))
            else:   # unload: release over the crate
                self._release(d, j, it, t)

    def _release(self, d, j, it, t):
        s = self.pool.slots[it["slot"]]
        if it["where"] == "cradle":
            self.cradle = None
        else:
            self.cells[it["where"]] = None
        it["state"] = "falling"
        self.update_ballast()
        self._kinematic_slot(d, it["slot"], on=False)
        self._sync_m0(it["slot"])
        d.qvel[s.dadr:s.dadr + 6] = 0.0                  # lowered to a stop, then let go
        j.update({"phase": "fall", "t_release": t, "touched": False, "v_max": 0.0, "slow_since": None})
        self._event(t, "released", fruit_id=it["fruit_id"])

    def _touching(self, d, gid: int) -> bool:
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == gid or c.geom2 == gid:
                return True
        return False

    def _step_fall(self, d, j, it, t):
        s = self.pool.slots[it["slot"]]
        self._sync_m0(it["slot"])
        v = float(np.linalg.norm(d.qvel[s.dadr:s.dadr + 3]))
        if not j["touched"]:
            j["v_max"] = max(j["v_max"], v)
            if self._touching(d, s.fruit):
                j["touched"] = True
                it["impact_mps"] = round(j["v_max"], 4)
        if j["touched"]:
            if v < SETTLE_V_MPS:
                j["slow_since"] = j["slow_since"] if j["slow_since"] is not None else t
            else:
                j["slow_since"] = None
        done = (j["slow_since"] is not None and t - j["slow_since"] >= SETTLE_HOLD_S) or \
            (t - j["t_release"] >= SETTLE_TIMEOUT_S)
        if not done:
            return
        if it["impact_mps"] is None:
            it["impact_mps"] = round(j["v_max"], 4)
        inside = self.in_crate(d, np.array(d.qpos[s.qadr:s.qadr + 3]))
        it["state"] = "crate" if inside else "dropped"
        it["outcome"] = "delivered" if inside else "dropped"
        it["unloaded_t"] = round(t, 4)
        lim = PAPAYA_IMPACT_MAX_MPS if it["crop"] == "papaya" else IMPACT_MAX_MPS
        it["impact_ok"] = bool(it["impact_mps"] < lim) if it["crop"] != "papaya" else bool(it["impact_mps"] <= lim)
        self.job = None
        self._event(t, "delivered" if inside else "dropped", fruit_id=it["fruit_id"], tree_id=it["tree_id"],
                    impact_mps=it["impact_mps"], kg=it["mass_kg"])
        if self.unloading:
            self._next_unload(d)

    def _drop_item(self, t, fid, why):
        it = self.items.pop(fid, None)
        self.cells = [None if c == fid else c for c in self.cells]
        if self.cradle == fid:
            self.cradle = None
        if self.job is not None and self.job["fruit_id"] == fid:
            self.job = None
        self.queue = [q for q in self.queue if q != fid]
        self.update_ballast()
        self._event(t, why, fruit_id=fid)
        return it

    def in_crate(self, d, p) -> bool:
        if self.L.crate_site < 0:
            return False
        c, R, h = self.crate_frame(d)
        loc = R.T @ (np.asarray(p, float) - c)
        return bool(np.all(np.abs(loc) <= h + 1e-9))

    # -------------------------------------------------------------- truth contact checks
    def _gid(self, fid: str) -> int:
        return self.pool.slots[self.items[fid]["slot"]].fruit

    def _dist(self, d, g1: int, g2: int) -> float:
        fromto = np.zeros(6)
        return float(mujoco.mj_geomDistance(self.m, d, g1, g2, 0.05, fromto))

    def tray_contact_pairs(self, d, tol: float = CONTACT_TOL_M) -> List[Tuple[str, str, float]]:
        """Every pair of tray items (seated or moving) closer than ``tol`` (truth, geom distance; needs current
        geom poses: call after ``mj_forward`` / ``mj_step``)."""
        ids = [f for f in [*self.cells, self.cradle] if f and self.items[f]["state"] in ("seated", "moving")]
        out = []
        for i in range(len(ids)):
            for k in range(i + 1, len(ids)):
                dist = self._dist(d, self._gid(ids[i]), self._gid(ids[k]))
                if dist < tol:
                    out.append((ids[i], ids[k], dist))
        return out

    def min_tray_gap_m(self, d) -> Optional[float]:
        ids = [f for f in [*self.cells, self.cradle] if f and self.items[f]["state"] in ("seated", "moving")]
        best = None
        for i in range(len(ids)):
            for k in range(i + 1, len(ids)):
                dist = self._dist(d, self._gid(ids[i]), self._gid(ids[k]))
                best = dist if best is None else min(best, dist)
        return best

    def _check_moving_contacts(self, d, it, t):
        g = self._gid(it["fruit_id"])
        for f in [*self.cells, self.cradle]:
            if f and f != it["fruit_id"] and self.items[f]["state"] == "seated":
                dist = self._dist(d, g, self._gid(f))
                if dist < CONTACT_TOL_M:
                    pair = sorted([f, it["fruit_id"]])
                    if pair not in self.contact_pairs:
                        self.contact_pairs.append(pair)
                        self.fruit_contacts += 1
                        self._event(t, "fruit_contact", pair=pair, dist_m=round(dist, 5))

    def _check_tray_contacts(self, d, t):
        mujoco.mj_kinematics(self.m, d)
        for a, b, dist in self.tray_contact_pairs(d):
            pair = sorted([a, b])
            if pair not in self.contact_pairs:
                self.contact_pairs.append(pair)
                self.fruit_contacts += 1
                self._event(t, "fruit_contact", pair=pair, dist_m=round(dist, 5))

    # -------------------------------------------------------------- serialisation
    def status(self) -> dict:
        j = self.job
        return {"busy": self.busy(), "paused": self.paused, "unloading": self.unloading,
                "job": None if j is None else {"op": j["op"], "fruit_id": j["fruit_id"], "phase": j["phase"],
                                               "s": round(j["s"], 4), "duration_s": round(j["path_obj"].duration, 4)},
                "cells": list(self.cells), "cradle": self.cradle, "payload_kg": round(self.payload_kg(), 4),
                "cap_kg": PAYLOAD_CAP_KG, "ballast_kg": round(float(self.m.body_mass[self.L.ballast]), 6),
                "queue": list(self.queue), "crate": self.crate_summary(), "per_tree": self.per_tree(),
                "fruit_contacts": self.fruit_contacts, "refusals": self.refusals,
                "max_speed_mps": round(self.max_speed_mps, 4), "max_papaya_speed_mps": round(self.max_papaya_speed_mps, 4),
                "assisted": [ASSISTED_TAG]}

    def to_dict(self) -> dict:
        j = None
        if self.job is not None:
            j = {k: v for k, v in self.job.items() if k != "path_obj"}
        return {"v": STATE_VERSION, "items": {k: dict(v) for k, v in self.items.items()}, "cells": list(self.cells),
                "cradle": self.cradle, "job": j, "queue": list(self.queue), "unloading": self.unloading,
                "paused": self.paused, "fruit_contacts": self.fruit_contacts,
                "contact_pairs": [list(p) for p in self.contact_pairs], "steps": self.steps,
                "crate_spots": {k: list(v) for k, v in self.crate_spots.items()}, "refusals": self.refusals,
                "max_speed_mps": self.max_speed_mps, "max_papaya_speed_mps": self.max_papaya_speed_mps}

    def m0_fields(self) -> List[List[float]]:
        """``dof_M0`` of every pool slot (6 per slot, slot order): WP3 writes it on release (``_sync_m0``)."""
        if self.pool is None:
            return []
        return [_fl(self.m.dof_M0[s.dadr:s.dadr + 6]) for s in self.pool.slots]

    def load_m0(self, rows: Sequence[Sequence[float]]) -> None:
        for s, row in zip(self.pool.slots, rows):
            self.m.dof_M0[s.dadr:s.dadr + 6] = np.asarray(row, float)

    def load_dict(self, st: dict) -> None:
        check_state_dict(st, n_slots=None if self.pool is None else len(self.pool.slots))
        self.items = {k: dict(v) for k, v in st["items"].items()}
        self.cells = list(st["cells"])
        self.cradle = st["cradle"]
        self.job = None
        if st["job"] is not None:
            self.job = dict(st["job"])
            self.job["path_obj"] = Path.from_dict(self.job["path"])
        self.queue = list(st["queue"])
        self.unloading = bool(st["unloading"])
        self.paused = st["paused"]
        self.fruit_contacts = int(st["fruit_contacts"])
        self.contact_pairs = [list(p) for p in st["contact_pairs"]]
        self.steps = int(st["steps"])
        self.crate_spots = {k: [float(x) for x in v] for k, v in st["crate_spots"].items()}
        self.refusals = int(st["refusals"])
        self.max_speed_mps = float(st["max_speed_mps"])
        self.max_papaya_speed_mps = float(st["max_papaya_speed_mps"])
        self._seat_cache = {}
        # mass matrix of the pool slots (see _sync_m0): the compiled template everywhere, the true one for the bodies
        # WP3 released into physics. The hook's provider then writes the packed values exactly (``load_m0``).
        synced = set()
        for it in self.items.values():
            if self.pool is not None and it["state"] in ("falling", "crate", "dropped") and not it.get("recycled") \
                    and not self._gone(it):
                synced.add(it["slot"])
        assigned = set(self._assign)           # WP23 assign writes the true dof_M0: keep it for every assigned slot
        for k in self._m0_template:
            if k not in synced and k not in assigned:
                self._restore_m0(k)
        for k in synced | (assigned & set(self._m0_template)):
            self._sync_m0(k)


_STATE_KEYS = {"v", "items", "cells", "cradle", "job", "queue", "unloading", "paused", "fruit_contacts",
               "contact_pairs", "steps", "crate_spots", "refusals", "max_speed_mps", "max_papaya_speed_mps"}
_ITEM_STATES = ("moving", "seated", "falling", "crate", "dropped")
_ITEM_KEYS = {"fruit_id", "tree_id", "crop", "slot", "mass_kg", "where", "state", "placed_t", "unloaded_t",
              "impact_mps", "stem_axis_body", "assisted", "outcome", "place_speed_mps"}
_ITEM_OPTIONAL = {"impact_ok", "recycled"}
_JOB_OPS = ("place", "unload", "return")
_JOB_KEYS = {"op", "fruit_id", "phase", "s", "path"}
_FALL_KEYS = {"t_release", "touched", "v_max", "slow_since"}


def _num(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def _opt_num(v) -> bool:
    return v is None or _num(v)


def check_state_dict(st: Any, n_slots: Optional[int] = None) -> None:
    """Full check of a ``to_dict`` payload (raises ValueError before anything is written): shapes and types of every
    field the controller reads, slot range (``n_slots`` = pool size, when known), and the consistency of cells, cradle,
    job and queue with the item states, so a loaded state can never fault the hooks."""
    if not isinstance(st, dict) or set(st) != _STATE_KEYS:
        raise ValueError(f"transfer state keys {sorted(st) if isinstance(st, dict) else type(st).__name__}")
    if st["v"] != STATE_VERSION:
        raise ValueError(f"transfer state version {st['v']!r}")
    items = st["items"]
    if not isinstance(items, dict):
        raise ValueError("transfer items must be an object")
    slots_live = {}
    for k, it in items.items():
        if not isinstance(it, dict) or it.get("fruit_id") != k or it.get("state") not in _ITEM_STATES:
            raise ValueError(f"transfer item {k!r} malformed")
        keys = set(it)
        if not _ITEM_KEYS <= keys or not keys <= _ITEM_KEYS | _ITEM_OPTIONAL:
            raise ValueError(f"transfer item {k!r} keys {sorted(keys)}")
        slot = it["slot"]
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0 or (n_slots is not None and slot >= n_slots):
            raise ValueError(f"transfer item {k!r}: slot {slot!r}")
        if not _num(it["mass_kg"]) or it["mass_kg"] < 0:
            raise ValueError(f"transfer item {k!r}: mass_kg")
        for key in ("tree_id", "crop"):
            if not isinstance(it[key], str) or not it[key]:
                raise ValueError(f"transfer item {k!r}: {key}")
        w = it["where"]
        if not (w == "cradle" or (isinstance(w, int) and not isinstance(w, bool) and 0 <= w < N_CELLS)):
            raise ValueError(f"transfer item {k!r}: where {w!r}")
        a = it["stem_axis_body"]
        if not isinstance(a, list) or len(a) != 3 or not all(_num(x) for x in a) or \
                math.sqrt(sum(x * x for x in a)) < 1e-6:
            raise ValueError(f"transfer item {k!r}: stem_axis_body")
        for key in ("placed_t", "unloaded_t", "impact_mps", "place_speed_mps"):
            if not _opt_num(it[key]):
                raise ValueError(f"transfer item {k!r}: {key}")
        if not isinstance(it["assisted"], list) or not all(isinstance(x, str) for x in it["assisted"]):
            raise ValueError(f"transfer item {k!r}: assisted")
        if it["outcome"] not in (None, "delivered", "dropped"):
            raise ValueError(f"transfer item {k!r}: outcome")
        for key in ("impact_ok", "recycled"):
            if key in it and not isinstance(it[key], bool):
                raise ValueError(f"transfer item {k!r}: {key}")
        if it["state"] in ("moving", "seated", "falling"):    # bodies WP3 drives (crate slots may be recycled)
            if slot in slots_live:
                raise ValueError(f"transfer items {slots_live[slot]!r} and {k!r} share pool slot {slot}")
            slots_live[slot] = k
    cells = st["cells"]
    if not isinstance(cells, list) or len(cells) != N_CELLS or any(c is not None and c not in items for c in cells):
        raise ValueError("transfer cells must list 8 item ids or null")
    if st["cradle"] is not None and st["cradle"] not in items:
        raise ValueError("transfer cradle must be an item id or null")
    # cells / cradle <-> items on the robot (seated or moving), each in exactly its own place
    placed = [c for c in cells if c is not None] + ([st["cradle"]] if st["cradle"] is not None else [])
    if len(set(placed)) != len(placed):
        raise ValueError("transfer item in two tray places")
    for k, c in enumerate(cells):
        if c is not None and (items[c]["where"] != k or items[c]["crop"] == "papaya"):
            raise ValueError(f"transfer cell {k} holds {c!r} (where {items[c]['where']!r})")
    if st["cradle"] is not None and items[st["cradle"]]["where"] != "cradle":
        raise ValueError("transfer cradle item is not a cradle item")
    for k, it in items.items():
        if (it["state"] in ("moving", "seated")) != (k in placed):
            raise ValueError(f"transfer item {k!r} ({it['state']}) and the tray places disagree")
    j = st["job"]
    moving = [k for k, it in items.items() if it["state"] in ("moving", "falling")]
    if j is not None:
        if not isinstance(j, dict) or j.get("fruit_id") not in items or j.get("phase") not in ("carry", "fall") \
                or j.get("op") not in _JOB_OPS:
            raise ValueError("transfer job malformed")
        need = _JOB_KEYS | (_FALL_KEYS if j["phase"] == "fall" else set())
        if set(j) != need:
            raise ValueError(f"transfer job keys {sorted(j)}")
        try:
            path = Path.from_dict(j["path"])
        except (KeyError, TypeError, ValueError, TransferError) as e:
            raise ValueError(f"transfer job path: {e}") from None
        if not (np.all(np.isfinite(path.p)) and np.all(np.isfinite(path.q))):
            raise ValueError("transfer job path must be finite")
        if not _num(j["s"]) or not 0.0 <= j["s"] <= path.duration + 1e-9:
            raise ValueError("transfer job s outside the path")
        want = "falling" if j["phase"] == "fall" else "moving"
        if items[j["fruit_id"]]["state"] != want or (j["phase"] == "fall" and j["op"] != "unload"):
            raise ValueError("transfer job phase and item state disagree")
        if j["phase"] == "fall":
            if not _num(j["t_release"]) or not isinstance(j["touched"], bool) or not _num(j["v_max"]) \
                    or not _opt_num(j["slow_since"]):
                raise ValueError("transfer job fall fields malformed")
        if moving != [j["fruit_id"]]:
            raise ValueError("transfer: exactly the job's item is moving or falling")
    elif moving:
        raise ValueError(f"transfer items {moving!r} move without a job")
    q = st["queue"]
    if not isinstance(q, list) or any(x not in items or items[x]["state"] != "seated" for x in q) \
            or len(set(q)) != len(q):
        raise ValueError("transfer queue must list seated item ids once")
    if not isinstance(st["unloading"], bool):
        raise ValueError("transfer unloading must be a boolean")
    if st["paused"] is not None and st["paused"] not in ("operator", "estop", *INTERLOCK_PAUSES):
        raise ValueError(f"transfer paused {st['paused']!r}")
    for key in ("fruit_contacts", "steps", "refusals"):
        if isinstance(st[key], bool) or not isinstance(st[key], int) or st[key] < 0:
            raise ValueError(f"transfer {key} must be a non-negative integer")
    for key in ("max_speed_mps", "max_papaya_speed_mps"):
        if not _num(st[key]):
            raise ValueError(f"transfer {key} must be a finite number")
    if not isinstance(st["crate_spots"], dict) or not isinstance(st["contact_pairs"], list):
        raise ValueError("transfer crate_spots / contact_pairs malformed")
    for p in st["contact_pairs"]:
        if not isinstance(p, list) or len(p) != 2 or not all(isinstance(x, str) for x in p):
            raise ValueError("transfer contact pair malformed")
    for k, v in st["crate_spots"].items():
        if k not in items or not isinstance(v, list) or len(v) != 3 or not all(_num(x) for x in v):
            raise ValueError(f"transfer crate spot {k!r} malformed")
    for x in q + ([j["fruit_id"]] if j is not None and j["op"] == "unload" else []):
        if x not in st["crate_spots"]:
            raise ValueError(f"transfer item {x!r} queued for the crate without a crate spot")
