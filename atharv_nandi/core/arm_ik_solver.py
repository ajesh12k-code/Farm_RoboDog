"""Kinematics of the Atharv Nandi dorsal gantry arm: IK for pick-and-place and the reach envelope.

Chain (all fixed in the trunk frame):
    gantry_slide (along trunk x) -> arm_yaw (about z) -> arm_shoulder, arm_elbow, arm_wrist_pitch
    (all about the local y axis) -> gripper_tip site.
Five joints, so we can pin the tip position (3) *and* point the gripper straight down (2), which is
what a top-down pick and a basket drop need.

solve() = closed-form elbow-up seed (planar 2-link in the arm plane) + damped-least-squares refinement
in the world frame, so a tilted or translated trunk is handled exactly. Link lengths and offsets are
read from the compiled model, never hard-coded.
"""
import math
from typing import NamedTuple, Optional, Sequence, Union

import mujoco
import numpy as np

from atharv_nandi.core.farm_scene import load_farm_model

ARM_JOINTS = ("gantry_slide", "arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist_pitch")
DOWN = np.array([0.0, 0.0, -1.0])
# Tool-aware TCP (WP8, ORCHARD-MVP-SPEC 2.2 / MVP-SPEC 2.2): distance from the wrist pivot to the tool TCP along the
# tool axis. T1 is the in-arm picker (gripper_tip site, read from the model); T2 / T5 are the WP0c frozen values and
# T4 is WP2's measured jaw centre (0.25 m, WP0c assumed 0.20: see the WP2 contract notes). Mirrors
# controllers.tools.registry.TOOLS (a test pins the two together).
TOOL_TCP_M = {"T2": 0.20, "T4": 0.25, "T5": 0.21}
# Mast keep-out (ORCHARD-MVP-SPEC 2.2): a vertical cylinder r 0.08 m about the mast axis at trunk x -0.30, y 0. The
# forearm, wrist and tool segment must stay KEEPOUT_R_M + KEEPOUT_MARGIN_M clear of the axis wherever they are
# above the trunk top (the mast rises from the trunk; below it the trunk / basket are the planner's `robot` class).
KEEPOUT_X_M = -0.30
KEEPOUT_R_M = 0.08
KEEPOUT_MARGIN_M = 0.01
KEEPOUT_Z0_M = 0.05           # trunk frame: the mast base (WP1 body `mast_base`)
SHOULDER_TORQUE_LIMIT_NM = 59.5   # 70 % of the 85 N.m forcerange (ORCHARD-MVP-SPEC 2.2, MISSION-PRESETS 5)
H_PREF = 0.30  # preferred horizontal shoulder-to-wrist distance when the gantry position is free
AXIS_W = 0.2   # weight of the tool-axis error vs. position error (metres per unit axis error)
DAMP = 0.02    # DLS damping
SELF_CLEARANCE = 0.001  # keep the upper arm this far (m) from the gantry carriage it is mounted on


class IKResult(NamedTuple):
    q: np.ndarray        # (gantry, yaw, shoulder, elbow, wrist) joint targets
    pos_err: float       # |tip - target| in metres at q
    axis_err_deg: float  # angle between the tool axis and straight down (0.0 if not requested)
    ok: bool


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _subtree(model: mujoco.MjModel, root: int) -> set:
    """Body ids of root and all its descendants (parents precede children in MuJoCo's body order)."""
    ids = {root}
    for b in range(root + 1, model.nbody):
        if model.body_parentid[b] in ids:
            ids.add(b)
    return ids


class DorsalArmIKSolver:
    """Tip-position (+ optional tool-down) IK for the 5-DOF dorsal arm. Not thread-safe (one MjData)."""

    def __init__(self, model: Union[mujoco.MjModel, str, None] = None, pos_tol: float = 0.003):
        if isinstance(model, str):  # legacy: path to an MJCF
            model = mujoco.MjModel.from_xml_path(model)
        self.model = model if model is not None else load_farm_model()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.pos_tol = pos_tol

        m = self.model
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        if min(jids) < 0:
            raise ValueError(f"Model is missing one of the arm joints {ARM_JOINTS}")
        self.qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([m.jnt_dofadr[j] for j in jids])
        self.limits = np.array([m.jnt_range[j] for j in jids], dtype=np.float64)

        def body(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)

        self.trunk_id = body("trunk")
        self.wrist_id = body("arm_wrist")
        self.forearm_id = body("arm_forearm")
        self.tip_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
        # Shoulder pivot in the trunk frame at gantry = 0 (carriage -> turret -> upper arm offsets).
        self.shoulder0 = sum(np.array(m.body_pos[body(n)]) for n in ("arm_carriage", "arm_turret", "arm_upper"))
        self.l1 = float(m.body_pos[body("arm_forearm")][2])  # upper arm runs along its local +z
        self.l2 = float(m.body_pos[body("arm_wrist")][0])    # forearm runs along its local +x
        self.l3 = float(m.site_pos[self.tip_site_id][0])     # wrist pivot -> gripper_tip along +x
        fa = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_left")
        self.grip_open, self.grip_closed = (float(v) for v in m.actuator_ctrlrange[fa])
        # Position-servo stiffness of each arm joint (for gravity feed-forward) and the crop mass.
        self.kp = np.zeros(5)
        for i, j in enumerate(jids):
            acts = [a for a in range(m.nu) if m.actuator_trnid[a, 0] == j]
            self.kp[i] = float(m.actuator_gainprm[acts[0], 0]) if acts else np.inf
        crop = body("crop_item_1")
        self.crop_mass = float(m.body_mass[crop]) if crop >= 0 else 0.0
        # The upper-arm capsule collides with the gantry carriage (its grandparent, so MuJoCo does not
        # filter the pair) once the shoulder swings ~8 deg below horizontal (1.71 rad). The joint range
        # allows 2.4 rad, so the IK used to command poses the arm physically cannot reach: the tip
        # stopped ~4 cm short and high and the pads missed the crop. The usable range is measured from
        # the model, so such a crop is reported out of reach instead.
        self.limits[2] = self._shoulder_clearance_range(body("arm_upper"), body("arm_carriage"))
        # Robot-vs-arm collision check for planned poses: geoms of the moving arm (turret and beyond)
        # against the rest of the robot (trunk, basket, rail, legs). Crops and the floor are ignored.
        arm = _subtree(m, body("arm_turret"))
        robot = _subtree(m, self.trunk_id) - _subtree(m, body("arm_carriage"))
        self._arm_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in arm}
        self._robot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in robot}
        # The shoulder limit above is measured at yaw 0, but the carriage is a box: at some yaws (e.g.
        # +-2.5 rad, rear-diagonal picks) the upper arm meets it earlier. Planned poses are therefore also
        # checked for upper-arm-vs-carriage clearance at their actual yaw.
        self._upper_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body("arm_upper")]
        self._carriage_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body("arm_carriage")]

    def robot_collisions(self, qpos: np.ndarray, q5: Sequence[float]) -> list:
        """(arm body, robot body) name pairs in contact if the arm stood at q5 (static check)."""
        m, d = self.model, self.data
        self._fk(qpos, q5)
        mujoco.mj_collision(m, d)
        hits = []
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if a in self._arm_geoms and (b in self._robot_geoms):
                    hits.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[a]),
                                 mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[b])))
        fromto = np.zeros(6)
        # Actual contact only (the SELF_CLEARANCE margin applies at yaw 0, via limits[2]): at yaw +-0.5 the
        # limit pose keeps ~0.35 mm, at +-0.75 and +-2.5 rad it would press 0.2-0.6 mm into the carriage.
        if any(mujoco.mj_geomDistance(m, d, a, b, 0.05, fromto) < 0.0
               for a in self._upper_geoms for b in self._carriage_geoms):
            hits.append(("arm_upper", "arm_carriage"))
        return hits

    def _shoulder_clearance_range(self, upper_id: int, carriage_id: int) -> np.ndarray:
        """Shoulder (lo, hi) within the joint range keeping the upper arm SELF_CLEARANCE off the carriage."""
        m, d = self.model, self.data
        lo, hi = (float(v) for v in self.limits[2])
        upper = [g for g in range(m.ngeom) if m.geom_bodyid[g] == upper_id]
        carriage = [g for g in range(m.ngeom) if m.geom_bodyid[g] == carriage_id]
        if not upper or not carriage:
            return np.array([lo, hi])
        q5 = np.array(d.qpos[self.qadr])
        q5[[0, 1, 3, 4]] = 0.0
        fromto = np.zeros(6)

        def clear(shoulder: float) -> bool:
            q5[2] = shoulder
            self._fk(d.qpos.copy(), q5)
            return all(mujoco.mj_geomDistance(m, d, a, b, 0.05, fromto) >= SELF_CLEARANCE
                       for a in upper for b in carriage)

        def edge(start: float, stop: float) -> float:
            prev = start
            for v in np.linspace(start, stop, int(abs(stop - start) / 0.005) + 1):
                if not clear(float(v)):
                    return prev
                prev = float(v)
            return stop

        rng = np.array([edge(0.0, lo), edge(0.0, hi)])
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        return rng

    # ------------------------------------------------------------------ forward kinematics
    def _fk(self, qpos: np.ndarray, q5: Optional[Sequence[float]] = None):
        self.data.qpos[:] = qpos
        if q5 is not None:
            self.data.qpos[self.qadr] = q5
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # refreshes cdof/subtree_com used by mj_jacSite

    def trunk_pose(self, qpos: np.ndarray):
        self._fk(qpos)
        return self.data.xpos[self.trunk_id].copy(), self.data.xmat[self.trunk_id].reshape(3, 3).copy()

    def tip_pos(self, qpos: np.ndarray, q5: Optional[Sequence[float]] = None) -> np.ndarray:
        self._fk(qpos, q5)
        return self.data.site_xpos[self.tip_site_id].copy()

    def tool_axis(self, qpos: np.ndarray, q5: Optional[Sequence[float]] = None) -> np.ndarray:
        """Unit vector from wrist to gripper tip (the direction the fingers point)."""
        self._fk(qpos, q5)
        return self.data.site_xmat[self.tip_site_id].reshape(3, 3)[:, 0].copy()

    def gravity_offsets(self, qpos: np.ndarray, q5: Sequence[float], payload: float = 0.0, loads=None) -> np.ndarray:
        """Joint offsets to add to position targets so the arm holds q5 against gravity.

        A position servo settles where kp * (ctrl - q) balances the gravity torque, so commanding q5
        alone leaves the tip sagging (5 cm at full reach with the shoulder's kp 70). The offset is
        qfrc_bias / kp at q5 (arm at rest, live trunk attitude), plus `payload` kg hanging at the tip, plus
        `loads` = [(kg, point in the wrist frame)] (WP8: a mounted tool and a caught fruit, each at its own CoM).
        """
        m, d = self.model, self.data
        self._fk(qpos, q5)
        d.qvel[:] = 0.0
        mujoco.mj_comVel(m, d)
        bias = np.zeros(m.nv)
        mujoco.mj_rne(m, d, 0, bias)
        tau = bias[self.dadr]
        # Any body gravcomp (none in the shipped MJCF, see the arm comment there) is already applied by MuJoCo,
        # so it is taken out of the offset; with gravcomp 0 everywhere this is a no-op.
        if np.any(m.body_gravcomp > 0.0):
            jac = np.zeros((3, m.nv))
            for b in np.nonzero(m.body_gravcomp > 0.0)[0]:
                mujoco.mj_jacBodyCom(m, d, jac, None, int(b))
                g_force = -m.body_mass[b] * m.body_gravcomp[b] * np.asarray(m.opt.gravity)
                tau = tau - jac[:, self.dadr].T @ g_force
        if payload > 0.0:
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jacp, None, self.tip_site_id)
            tau = tau + jacp[:, self.dadr].T @ np.array([0.0, 0.0, payload * -m.opt.gravity[2]])
        tau = tau + self._load_torque(loads)
        return tau / self.kp

    # ------------------------------------------------------------------ tool-aware (WP8)
    def _wrist_frame(self):
        """World pose (pos, rot) of the wrist pivot body at the last _fk."""
        b = self.wrist_id
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def _load_torque(self, loads) -> np.ndarray:
        """Joint torques (5) that point masses fixed in the wrist frame add under gravity: loads = [(kg, p_wrist)].

        Uses the state of the last _fk (call after it). A mounted tool and a caught fruit hang on welds, so their
        weight never shows up in the arm's own qfrc_bias: the feed-forward adds them here, each at its own CoM (the
        WP2 DriverArm lumped the tool at gripper_tip, which left 2.5 mm of TCP sag)."""
        if not loads:
            return np.zeros(5)
        m, d = self.model, self.data
        wp, wr = self._wrist_frame()
        jacp = np.zeros((3, m.nv))
        tau = np.zeros(5)
        for kg, p in loads:
            if kg <= 0.0:
                continue
            pw = wp + wr @ np.asarray(p, dtype=np.float64)
            mujoco.mj_jac(m, d, jacp, None, pw, self.wrist_id)
            tau = tau + jacp[:, self.dadr].T @ np.array([0.0, 0.0, kg * -m.opt.gravity[2]])
        return tau

    def gravity_offsets_loads(self, qpos: np.ndarray, q5: Sequence[float], loads) -> np.ndarray:
        """Feed-forward offsets (rad / m) for the arm at q5 carrying point masses ``loads`` = [(kg, p_wrist_frame)]."""
        return self.gravity_offsets(qpos, q5, 0.0, loads=loads)

    def tool_tcp_m(self, tool: Optional[str]) -> float:
        """Wrist pivot -> TCP along the tool axis for a tool id (None / 'T1' = the in-arm picker's gripper_tip)."""
        if tool in (None, "T1"):
            return self.l3
        return float(TOOL_TCP_M[tool])

    def tcp_solve(self, qpos: np.ndarray, target: Sequence[float], tool: Optional[str], pitch: float = 0.0,
                  gantry: Optional[float] = 0.0, keepout: bool = True):
        """Closed-form joints putting the TOOL TCP at ``target`` (world), the tool axis in the arm plane and pitched
        ``pitch`` rad below horizontal (0 level, pi/2 down). The arm is planar, so the axis yaw is the shoulder ->
        target bearing. ``gantry=None`` searches the rail (0 first) for a pose that passes the mast keep-out.
        Returns (q5, axis_world) or (None, None)."""
        tcp = self.tool_tcp_m(tool)
        t_pos, t_rot = self.trunk_pose(qpos)
        p = t_rot.T @ (np.asarray(target, dtype=np.float64) - t_pos)
        if gantry is None:
            lo, hi = self.limits[0]
            grid = [0.0, *sorted(np.linspace(lo, hi, 11), key=abs)]
        else:
            grid = [float(gantry)]
        for g in grid:
            s0 = self.shoulder0 + np.array([g, 0.0, 0.0])
            u = p[:2] - s0[:2]
            if float(np.linalg.norm(u)) < 1e-6:
                continue
            u = u / np.linalg.norm(u)
            axis = np.array([u[0] * math.cos(pitch), u[1] * math.cos(pitch), -math.sin(pitch)])
            q = self.planar_solution(p - tcp * axis, float(g))
            if q is None:
                continue
            q = np.array(q, dtype=np.float64)
            q[4] = q[4] - math.pi / 2.0 + pitch          # planar_solution aims the tool down (sum pi/2)
            if not (self.limits[4, 0] <= q[4] <= self.limits[4, 1]):
                continue
            if keepout and self.keepout_clearance(qpos, q, tool) < 0.0:
                continue
            return q, t_rot @ axis
        return None, None

    def keepout_clearance(self, qpos: np.ndarray, q5: Sequence[float], tool: Optional[str] = None) -> float:
        """Signed clearance (m) of the forearm, wrist and tool segment from the mast keep-out cylinder (radius
        KEEPOUT_R_M + KEEPOUT_MARGIN_M about the vertical axis at trunk x KEEPOUT_X_M, above KEEPOUT_Z0_M);
        negative = inside. Points below the mast base are not counted."""
        self._fk(qpos, q5)
        t_pos, t_rot = self.data.xpos[self.trunk_id].copy(), self.data.xmat[self.trunk_id].reshape(3, 3).copy()
        wp, wr = self._wrist_frame()
        elbow = self.data.xpos[self.forearm_id].copy()
        tcp = wp + wr[:, 0] * self.tool_tcp_m(tool)
        best = math.inf
        for a, b in ((elbow, wp), (wp, tcp)):
            for s in np.linspace(0.0, 1.0, 9):
                pt = t_rot.T @ (a + s * (b - a) - t_pos)
                if pt[2] < KEEPOUT_Z0_M:
                    continue
                best = min(best, math.hypot(pt[0] - KEEPOUT_X_M, pt[1]) - KEEPOUT_R_M - KEEPOUT_MARGIN_M)
        return float(best) if math.isfinite(best) else 1.0

    def shoulder_torque_nm(self, qpos: np.ndarray, q5: Sequence[float], loads=()) -> float:
        """Total static shoulder torque (arm self-weight + ``loads`` [(kg, p_wrist)], e.g. tool + fruit), N.m.
        MISSION-PRESETS 5 gates it at SHOULDER_TORQUE_LIMIT_NM at the worst target pose."""
        m, d = self.model, self.data
        self._fk(qpos, q5)
        d.qvel[:] = 0.0
        mujoco.mj_comVel(m, d)
        bias = np.zeros(m.nv)
        mujoco.mj_rne(m, d, 0, bias)
        tau = bias[self.dadr] + self._load_torque(list(loads))
        return float(abs(tau[2]))

    def predicted_com(self, qpos: np.ndarray, q5: Sequence[float], loads=()) -> np.ndarray:
        """Whole-robot CoM (world) with the arm at q5 plus point loads [(kg, p_wrist)] (tool, fruit). For the
        predicted tripod / CoM margin precondition (MISSION-PRESETS 5)."""
        m, d = self.model, self.data
        self._fk(qpos, q5)
        mujoco.mj_comPos(m, d)
        mass = float(m.body_subtreemass[self.trunk_id])
        com = np.array(d.subtree_com[self.trunk_id]) * mass
        wp, wr = self._wrist_frame()
        for kg, p in loads:
            com = com + kg * (wp + wr @ np.asarray(p, dtype=np.float64))
            mass += kg
        return com / mass

    def gripper_opening(self) -> float:
        """Gap (m) between the two finger pads with the fingers fully open (joint value 0)."""
        m, pads = self.model, []
        for name in ("finger_left", "finger_right"):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            g = m.body_geomadr[b]
            pads.append((m.body_pos[b][1] + m.geom_pos[g][1], m.geom_size[g][1]))
        (yl, hl), (yr, hr) = pads
        return float(abs(yl - yr) - hl - hr)

    # ------------------------------------------------------------------ inverse kinematics
    def preferred_gantry(self, wrist_trunk: np.ndarray) -> float:
        """Gantry value putting the shoulder ~H_PREF from the wrist point, on the robot-centre side."""
        dx = wrist_trunk[0] - self.shoulder0[0]
        along = math.sqrt(max(0.0, H_PREF ** 2 - wrist_trunk[1] ** 2))
        return float(np.clip(dx - math.copysign(along, dx), *self.limits[0]))

    def planar_solution(self, wrist_trunk: np.ndarray, gantry: float) -> Optional[np.ndarray]:
        """Closed-form elbow-up joints with the tool along -z_trunk, or None if infeasible/unsafe."""
        s = self.shoulder0 + np.array([gantry, 0.0, 0.0])
        v = np.asarray(wrist_trunk, dtype=np.float64) - s
        h = math.hypot(v[0], v[1])
        yaw = math.atan2(v[1], v[0]) if h > 1e-6 else 0.0
        z, d, l1, l2 = v[2], math.hypot(h, v[2]), self.l1, self.l2
        if not (abs(l1 - l2) + 1e-6 < d < l1 + l2 - 1e-6):
            return None
        # Arm-plane coordinates (h, z): upper arm at angle a1 above horizontal, forearm at a2.
        a1 = math.atan2(z, h) + math.acos(np.clip((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d), -1.0, 1.0))
        eh, ez = l1 * math.cos(a1), l1 * math.sin(a1)
        a2 = math.atan2(z - ez, h - eh)
        shoulder = math.pi / 2.0 - a1          # upper arm direction (sin q, cos q) in (h, z)
        elbow = -a2 - shoulder                  # forearm direction (cos(qs+qe), -sin(qs+qe))
        wrist = math.pi / 2.0 + a2              # qs+qe+qw = pi/2 -> tool straight down
        q = np.array([gantry, yaw, _wrap(shoulder), _wrap(elbow), _wrap(wrist)])
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            return None
        # Over the body footprint the elbow must stay above the basket rim and gantry rail.
        ex, ey = s[0] + eh * math.cos(yaw), eh * math.sin(yaw)
        if abs(ex) < 0.40 and abs(ey) < 0.20 and s[2] + ez < 0.16:
            return None
        return q

    def _seed(self, wrist_trunk: np.ndarray, gantry: Optional[float]) -> Optional[np.ndarray]:
        if gantry is not None:
            return self.planar_solution(wrist_trunk, float(np.clip(gantry, *self.limits[0])))
        pref = self.preferred_gantry(wrist_trunk)
        grid = np.linspace(self.limits[0, 0], self.limits[0, 1], 9)
        for g in [pref, *sorted(grid, key=lambda x: abs(x - pref))]:
            q = self.planar_solution(wrist_trunk, g)
            if q is not None:
                return q
        return None

    def solve(
        self,
        qpos: np.ndarray,
        target_pos: Sequence[float],
        seed_q: Optional[Sequence[float]] = None,
        tool_down: bool = True,
        gantry: Optional[float] = None,
        max_iters: int = 60,
    ) -> IKResult:
        """Joint targets putting gripper_tip at target_pos (world), optionally pointing straight down.

        qpos is the latest full-world state (robot + crops); only the arm joints are varied.
        gantry pins the gantry: it seeds the closed form AND is held fixed during the refinement. With
        the tool pointing down the chain is redundant by one DOF along the rail, so a free gantry let the
        damped-least-squares refinement wander along that null space from an exact seed (a side reach
        drifted the gantry 9 cm and yaw 0.4 rad within two waypoints, then missed by 1.7 cm).
        """
        target = np.asarray(target_pos, dtype=np.float64)
        q0 = None
        if tool_down:
            t_pos, t_rot = self.trunk_pose(qpos)
            wrist = t_rot.T @ (target - t_pos) + np.array([0.0, 0.0, self.l3])
            q0 = self._seed(wrist, gantry)
        if q0 is None:  # unreachable for the closed form: best-effort DLS from the caller's seed
            q0 = np.array(seed_q[:5] if seed_q is not None else self.data.qpos[self.qadr], dtype=np.float64)
            if gantry is not None:
                q0[0] = gantry
        q0 = np.clip(q0, self.limits[:, 0], self.limits[:, 1])
        return self._refine(qpos, target, q0, tool_down, max_iters, pin_gantry=gantry is not None)

    def _refine(self, qpos, target, q0, tool_down, max_iters, pin_gantry=False) -> IKResult:
        m, d, sid = self.model, self.data, self.tip_site_id
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        q = q0.copy()
        free = np.arange(1 if pin_gantry else 0, 5)
        dofs = self.dadr[free]

        def residual():
            self._fk(qpos, q)
            axis = d.site_xmat[sid].reshape(3, 3)[:, 0].copy()
            return target - d.site_xpos[sid], axis, DOWN - axis

        for _ in range(max_iters):
            err, axis, axis_err = residual()
            if np.linalg.norm(err) < 5e-4 and (not tool_down or np.linalg.norm(axis_err) < 0.01):
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, sid)
            J, e = jacp[:, dofs], err
            if tool_down:  # d(axis)/dt = omega x axis = -skew(axis) @ omega
                J = np.vstack([J, AXIS_W * (-_skew(axis) @ jacr[:, dofs])])
                e = np.concatenate([err, AXIS_W * axis_err])
            dq = J.T @ np.linalg.solve(J @ J.T + DAMP ** 2 * np.eye(len(e)), e)
            q[free] = np.clip(q[free] + dq, self.limits[free, 0], self.limits[free, 1])
        err, axis, _ = residual()  # residual at the q we actually return
        pos_err = float(np.linalg.norm(err))
        axis_deg = math.degrees(math.acos(np.clip(-axis[2], -1.0, 1.0))) if tool_down else 0.0
        return IKResult(q, pos_err, axis_deg, pos_err <= self.pos_tol and axis_deg <= 5.0)


# ---------------------------------------------------------------------- reach envelope
def discover_manipulator_joints(model: mujoco.MjModel, root_arm_body: str = "arm_carriage"):
    """All joints in the manipulator subtree, with qpos address and (fallback) limits."""
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_arm_body)
    if root_id == -1:
        raise ValueError(f"Manipulator root body '{root_arm_body}' not found in MJCF model.")
    arm_bodies = {root_id}
    for body_id in range(model.nbody):  # parents always precede children in MuJoCo body order
        if model.body_parentid[body_id] in arm_bodies and body_id != 0:
            arm_bodies.add(body_id)
    joints = []
    for jnt_id in range(model.njnt):
        if model.jnt_bodyid[jnt_id] not in arm_bodies:
            continue
        jtype = model.jnt_type[jnt_id]
        low, high = float(model.jnt_range[jnt_id, 0]), float(model.jnt_range[jnt_id, 1])
        if not model.jnt_limited[jnt_id]:
            low, high = (-np.pi, np.pi) if jtype == mujoco.mjtJoint.mjJNT_HINGE else (-0.25, 0.25)
        joints.append({
            "id": jnt_id,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or f"joint_{jnt_id}",
            "type": int(jtype),
            "qpos_adr": int(model.jnt_qposadr[jnt_id]),
            "low": low,
            "high": high,
        })
    return joints


def compute_work_envelope(
    model: mujoco.MjModel,
    base_body_name: str = "trunk",
    tip_site_name: str = "gripper_tip",
    arm_root_name: str = "arm_carriage",
    n_yaw: int = 12,
    n_shoulder: int = 12,
    n_gantry: int = 8,
    decimals: int = 3,
    shoulder_max: Optional[float] = None,
) -> dict:
    """Wireframe of the tip's reach (elbow straight), as segments in the base-body frame.

    Resolution and rounding are chosen so the JSON stays ~30 KB: the old 1.5k-segment, full-precision
    payload (~210 KB) was too large for a std_srvs/Trigger response to come back reliably.

    The shoulder sweep stops at the planner's usable limit (shoulder_max; by default
    DorsalArmIKSolver(model).limits[2][1], where the upper arm meets its gantry carriage), not the
    joint's 2.4 rad, so the wireframe shows what auto-harvest will actually attempt.
    """
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_site_name)
    joints = discover_manipulator_joints(model, arm_root_name)
    slides = [j for j in joints if j["type"] == mujoco.mjtJoint.mjJNT_SLIDE and "finger" not in j["name"]]
    hinges = [j for j in joints if j["type"] == mujoco.mjtJoint.mjJNT_HINGE]
    if len(hinges) < 2:
        raise RuntimeError("Need at least 2 hinge joints in the arm to sweep a 3D envelope.")
    yaw_j, sh_j = hinges[0], hinges[1]
    el_j = hinges[2] if len(hinges) > 2 else None
    gantry_j = slides[0] if slides else None
    sh_high = sh_j["high"]
    if shoulder_max is None and model.nkey > 0:
        try:
            shoulder_max = float(DorsalArmIKSolver(model).limits[2][1])
        except ValueError:  # not the Atharv Nandi arm: sweep the joint range
            shoulder_max = None
    if shoulder_max is not None:
        sh_high = min(sh_high, float(shoulder_max))

    base_pos = data.xpos[base_id].copy()
    base_rot = data.xmat[base_id].reshape(3, 3).copy()
    floor_z = 0.015
    floor_thr = (floor_z - base_pos[2]) + 0.035

    def tip(g, yaw, sh):
        if gantry_j is not None:
            data.qpos[gantry_j["qpos_adr"]] = g
        data.qpos[yaw_j["qpos_adr"]] = yaw
        data.qpos[sh_j["qpos_adr"]] = sh
        if el_j is not None:
            data.qpos[el_j["qpos_adr"]] = 0.0
        mujoco.mj_kinematics(model, data)
        p = data.site_xpos[tip_id].copy()
        p[2] = max(floor_z, p[2])
        return np.round((p - base_pos) @ base_rot, decimals).tolist()  # base-local, follows the chassis

    canopy, floor = [], []

    def add(p1, p2, is_floor):
        (floor if is_floor else canopy).append([p1, p2])

    gantry_ext = [gantry_j["low"], gantry_j["high"]] if gantry_j else [0.0]
    sh_arc = np.linspace(max(-0.3, sh_j["low"]), sh_high, n_shoulder)
    yaw_fan = np.linspace(yaw_j["low"], yaw_j["high"], n_yaw + 1)[:-1]
    ring_ks = sorted(set(range(0, n_shoulder, 2)) | {n_shoulder - 1})

    # 1. Meridian ribs + latitude rings at each gantry extreme.
    for g in gantry_ext:
        grid = [[tip(g, yaw, sh) for sh in sh_arc] for yaw in yaw_fan]
        for rib in grid:
            for a, b in zip(rib, rib[1:]):
                add(a, b, a[2] <= floor_thr and b[2] <= floor_thr)
        for k in ring_ks:
            for i in range(len(grid)):
                add(grid[i][k], grid[(i + 1) % len(grid)][k], k >= n_shoulder - 2)

    # 2. Longitudinal ridges along the gantry rail.
    if gantry_j is not None:
        g_steps = np.linspace(gantry_j["low"], gantry_j["high"], n_gantry)
        sh_ridges = np.linspace(max(0.0, sh_j["low"]), sh_high, 3)
        for yaw in np.linspace(yaw_j["low"] * 0.5, yaw_j["high"] * 0.5, 5):
            for sh in sh_ridges:
                ridge = [tip(g, yaw, sh) for g in g_steps]
                for a, b in zip(ridge, ridge[1:]):
                    add(a, b, sh >= sh_ridges[-1] - 0.05)

    return {
        "joints_discovered": [j["name"] for j in joints],
        "canopy_segments": canopy,
        "floor_segments": floor,
    }
