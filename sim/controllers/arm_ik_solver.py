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

from controllers.farm_scene import load_farm_model

ARM_JOINTS = ("gantry_slide", "arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist_pitch")
DOWN = np.array([0.0, 0.0, -1.0])
H_PREF = 0.30  # preferred horizontal shoulder-to-wrist distance when the gantry position is free
AXIS_W = 0.2   # weight of the tool-axis error vs. position error (metres per unit axis error)
DAMP = 0.02    # DLS damping


class IKResult(NamedTuple):
    q: np.ndarray        # (gantry, yaw, shoulder, elbow, wrist) joint targets
    pos_err: float       # |tip - target| in metres at q
    axis_err_deg: float  # angle between the tool axis and straight down (0.0 if not requested)
    ok: bool


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


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
        self.tip_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
        # Shoulder pivot in the trunk frame at gantry = 0 (carriage -> turret -> upper arm offsets).
        self.shoulder0 = sum(np.array(m.body_pos[body(n)]) for n in ("arm_carriage", "arm_turret", "arm_upper"))
        self.l1 = float(m.body_pos[body("arm_forearm")][2])  # upper arm runs along its local +z
        self.l2 = float(m.body_pos[body("arm_wrist")][0])    # forearm runs along its local +x
        self.l3 = float(m.site_pos[self.tip_site_id][0])     # wrist pivot -> gripper_tip along +x
        fa = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_left")
        self.grip_open, self.grip_closed = (float(v) for v in m.actuator_ctrlrange[fa])

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
        gantry pins the gantry for the closed-form seed (keeps Cartesian paths continuous).
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
        return self._refine(qpos, target, np.clip(q0, self.limits[:, 0], self.limits[:, 1]), tool_down, max_iters)

    def _refine(self, qpos, target, q0, tool_down, max_iters) -> IKResult:
        m, d, sid = self.model, self.data, self.tip_site_id
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        q = q0.copy()

        def residual():
            self._fk(qpos, q)
            axis = d.site_xmat[sid].reshape(3, 3)[:, 0].copy()
            return target - d.site_xpos[sid], axis, DOWN - axis

        for _ in range(max_iters):
            err, axis, axis_err = residual()
            if np.linalg.norm(err) < 5e-4 and (not tool_down or np.linalg.norm(axis_err) < 0.01):
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, sid)
            J, e = jacp[:, self.dadr], err
            if tool_down:  # d(axis)/dt = omega x axis = -skew(axis) @ omega
                J = np.vstack([J, AXIS_W * (-_skew(axis) @ jacr[:, self.dadr])])
                e = np.concatenate([err, AXIS_W * axis_err])
            dq = J.T @ np.linalg.solve(J @ J.T + DAMP ** 2 * np.eye(len(e)), e)
            q = np.clip(q + dq, self.limits[:, 0], self.limits[:, 1])
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
) -> dict:
    """Wireframe of the tip's reach (elbow straight), as segments in the base-body frame.

    Resolution and rounding are chosen so the JSON stays ~30 KB: the old 1.5k-segment, full-precision
    payload (~210 KB) was too large for a std_srvs/Trigger response to come back reliably.
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
    sh_arc = np.linspace(max(-0.3, sh_j["low"]), sh_j["high"], n_shoulder)
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
        sh_ridges = np.linspace(max(0.0, sh_j["low"]), sh_j["high"], 3)
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
