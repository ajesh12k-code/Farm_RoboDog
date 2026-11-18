"""Stability probe: reads a MuJoCo model + qpos and feeds atharv_nandi/core/stability.py.

MuJoCo, no ROS. The probe owns its own MjData, runs mj_forward on it (kinematics, contacts,
constraint forces, sensors) and NEVER steps, so it can run beside the live sim on a copy of
/nandi/sim_qpos without touching the physics. One probe costs well under 2 ms (tests measure it).

It extracts:
    CoM          subtree_com of the trunk (whole robot, incl. welded payload / mast mocks), plus
                 optional extra point loads (e.g. crops lying in the basket)
    contacts     geom-to-world contacts of the 6 *_foot geoms with dist <= stability.CONTACT_DIST_M
    feet         foot geom centres
    IMU          site 'imu' orientation if present, else the trunk quaternion
    mast head    site 'mast_head_ring' if present (NANDI_MAST_H mock), and the mast_base_f/_t sensors
    arm reach    horizontal distance shoulder -> site 'tool_flange' (or 'gripper_tip' if absent)

Sites from the alignment spec section 4 are read if present and skipped if absent, so this module
works before and after the Package A model change.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

from atharv_nandi.core import load_model as lm
from atharv_nandi.core import stability as st

LEGS = ("FR", "FL", "MR", "ML", "RR", "RL")
# Tripods of the alternating gait (= tripod_gait.TRIPOD_PHASE: A = FR, ML, RR; B = FL, MR, RL).
TRIPODS = (("FR", "ML", "RR"), ("FL", "MR", "RL"))
FOOT_GEOMS = tuple(f"{leg}_foot" for leg in LEGS)
HIP_JOINTS = tuple(f"{leg}_hip_joint" for leg in LEGS)
TRUNK_BODY = "trunk"
SHOULDER_BODY = "arm_upper"
IMU_SITE = "imu"
MAST_HEAD_SITE = "mast_head_ring"
MAST_FORCE_SENSOR = "mast_base_f"
MAST_TORQUE_SENSOR = "mast_base_t"
TOOL_SITES = ("tool_flange", "gripper_tip")


def _id(model, objtype, name: str) -> int:
    return mujoco.mj_name2id(model, objtype, name)


class StabilityProbe:
    """Static stability snapshot of a model at a given qpos (mj_forward only)."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        O = mujoco.mjtObj
        self.trunk = _id(model, O.mjOBJ_BODY, TRUNK_BODY)
        if self.trunk < 0:
            raise ValueError("model has no 'trunk' body")
        self.foot_ids: Dict[int, str] = {}
        for name in FOOT_GEOMS:
            gid = _id(model, O.mjOBJ_GEOM, name)
            if gid < 0:
                raise ValueError(f"model has no foot geom {name!r}")
            self.foot_ids[gid] = name
        self.foot_gid = {name: gid for gid, name in self.foot_ids.items()}
        self.foot_radius = {name: float(model.geom_size[gid][0]) for gid, name in self.foot_ids.items()}
        self.hip_qadr = [int(model.jnt_qposadr[_id(model, O.mjOBJ_JOINT, j)]) for j in HIP_JOINTS
                         if _id(model, O.mjOBJ_JOINT, j) >= 0]
        self.imu_site = _id(model, O.mjOBJ_SITE, IMU_SITE)
        self.head_site = _id(model, O.mjOBJ_SITE, MAST_HEAD_SITE)
        self.shoulder = _id(model, O.mjOBJ_BODY, SHOULDER_BODY)
        self.tool_site = next((s for s in (_id(model, O.mjOBJ_SITE, n) for n in TOOL_SITES) if s >= 0), -1)
        self.mast_f = self._sensor_adr(MAST_FORCE_SENSOR)
        self.mast_t = self._sensor_adr(MAST_TORQUE_SENSOR)
        self._f6 = np.zeros(6)

    def _read_gravity(self):
        """Gravity is read on every probe, so a tilted-gravity (slope) test sees the current value."""
        g = np.asarray(self.model.opt.gravity, dtype=np.float64)
        n = float(np.linalg.norm(g))
        self.gravity_dir = g / n if n > 0 else np.array([0.0, 0.0, -1.0])
        self.g = n or st.G

    def _sensor_adr(self, name: str) -> Optional[int]:
        sid = _id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return None if sid < 0 else int(self.model.sensor_adr[sid])

    def foot_clearance(self, name: str) -> Tuple[float, Optional[np.ndarray]]:
        """Gap (m) between a foot sphere and the world geometry straight below it (along gravity), and
        the ground point hit. (inf, None) when the ray hits nothing or a non-world body first (a crop).
        Needs mj_forward to have run on self.data."""
        m, d = self.model, self.data
        gid = self.foot_gid[name]
        p = d.geom_xpos[gid].copy()
        hit = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(m, d, p, self.gravity_dir, None, 1, int(m.geom_bodyid[gid]), hit)
        if dist < 0 or hit[0] < 0 or m.geom_bodyid[hit[0]] != 0:
            return math.inf, None
        return max(0.0, float(dist) - self.foot_radius[name]), p + float(dist) * self.gravity_dir

    def stance_tripod(self, in_contact: Sequence[str]) -> Optional[dict]:
        """The tripod nearest the ground: {"legs", "clearance_m" (max over its 3 feet), "points"}.
        Feet already in contact count with clearance 0 at their contact point (given by the caller)."""
        best = None
        for legs in TRIPODS:
            clear, pts = [], []
            for leg in legs:
                c, g = self.foot_clearance(f"{leg}_foot")
                clear.append(0.0 if f"{leg}_foot" in in_contact else c)
                pts.append(g)
            worst = max(clear)
            if best is None or worst < best["clearance_m"]:
                best = {"legs": legs, "clearance_m": worst, "per_foot_m": clear, "points": pts}
        return best

    @property
    def robot_mass_kg(self) -> float:
        return float(self.model.body_subtreemass[self.trunk])

    def probe(self, qpos: Sequence[float], extra_loads: Sequence[Tuple[float, Sequence[float]]] = ()) -> dict:
        """Snapshot at qpos. extra_loads: [(kg, world xyz)] point masses carried by the robot."""
        m, d = self.model, self.data
        q = np.asarray(qpos, dtype=np.float64)
        if q.shape[0] != m.nq:
            raise ValueError(f"qpos has {q.shape[0]} values, model nq={m.nq}")
        d.qpos[:] = q
        d.qvel[:] = 0.0
        d.act[:] = 0.0
        mujoco.mj_forward(m, d)
        self._read_gravity()
        up = -self.gravity_dir

        mass = self.robot_mass_kg
        comb = lm.total_mass_and_com_shift(mass, d.subtree_com[self.trunk].copy(), extra_loads)
        mass, com = comb["total_kg"], np.asarray(comb["com"])

        # foot contacts with the world body (floor, ridges, clods, orchard bunds)
        pts: List[np.ndarray] = []
        fn: List[float] = []
        per_foot: Dict[str, List[np.ndarray]] = {}
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > st.CONTACT_DIST_M:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self.foot_ids and m.geom_bodyid[g2] == 0:
                foot = self.foot_ids[g1]
            elif g2 in self.foot_ids and m.geom_bodyid[g1] == 0:
                foot = self.foot_ids[g2]
            else:
                continue
            mujoco.mj_contactForce(m, d, i, self._f6)
            p = c.pos.copy()
            pts.append(p)
            fn.append(abs(float(self._f6[0])))
            per_foot.setdefault(foot, []).append(p)

        n_feet = len(per_foot)
        support = st.SUPPORT_CONTACT if n_feet else st.SUPPORT_NONE
        stance = None
        sup_pts = list(pts)
        if len(st.support_polygon_3d(pts, self.gravity_dir)) < 3:
            # review F3: < 3 feet touching mid-gait; fall back to the stance tripod if it is near the ground
            stance = self.stance_tripod(per_foot)
            if stance is not None and stance["clearance_m"] <= st.STANCE_NEAR_M:
                support = st.SUPPORT_STANCE_TRIPOD
                for leg, g in zip(stance["legs"], stance["points"]):
                    if f"{leg}_foot" not in per_foot and g is not None:
                        sup_pts.append(g)
            else:
                support = st.SUPPORT_NONE

        poly3 = st.support_polygon_3d(sup_pts, self.gravity_dir)
        R = st.gravity_frame(self.gravity_dir)
        poly2 = [tuple((R @ p)[:2]) for p in poly3]
        com2 = (R @ com)[:2]
        margin = st.com_margin(com2, poly2) if len(poly2) >= 3 else -math.inf
        edges = st.edge_tip_angles(com, poly3, self.gravity_dir) if len(poly3) >= 3 else []
        tip = min(e["tip_deg"] for e in edges) if edges else -math.inf
        cop = st.cop(pts, fn)

        imu_quat = np.zeros(4)
        if self.imu_site >= 0:
            mujoco.mju_mat2Quat(imu_quat, d.site_xmat[self.imu_site])
        else:
            imu_quat[:] = d.xquat[self.trunk]
        # slopes are relative to gravity (R maps world -> gravity frame; identity for gravity -z)
        body_g = R @ st.quat_to_mat(imu_quat)
        yaw = math.atan2(body_g[1, 0], body_g[0, 0])
        foot_pts = [R @ np.mean(v, axis=0) for v in per_foot.values()]
        slope_pts = st.slope_from_points(foot_pts, yaw) if len(foot_pts) >= 3 else None
        slope_imu = st.slope_from_mat(body_g)

        ground_h = float(np.mean([np.dot(p, up) for p in sup_pts])) if sup_pts else 0.0
        head = mast_h = head_push = head_push_rigid = None
        if self.head_site >= 0:
            head = d.site_xpos[self.head_site].copy()
            mast_h = float(np.dot(head, up)) - ground_h
            head_push_rigid = (st.min_head_push_to_tip(mass, com, poly3, head, self.gravity_dir, self.g)
                               if edges else -math.inf)
            head_push = head_push_rigid * st.HEAD_PUSH_COMPLIANCE if math.isfinite(head_push_rigid) else head_push_rigid
        mast_f = None if self.mast_f is None else float(np.linalg.norm(d.sensordata[self.mast_f:self.mast_f + 3]))
        mast_t = None if self.mast_t is None else float(np.linalg.norm(d.sensordata[self.mast_t:self.mast_t + 3]))

        reach = None
        if self.shoulder >= 0 and self.tool_site >= 0:
            v = d.site_xpos[self.tool_site] - d.xpos[self.shoulder]
            reach = float(np.linalg.norm(v - np.dot(v, up) * up))

        hips = [float(q[a]) for a in self.hip_qadr]
        sup_feet = set(per_foot)
        if support == st.SUPPORT_STANCE_TRIPOD:
            sup_feet |= {f"{leg}_foot" for leg in stance["legs"]}
        n_support = len(sup_feet)
        mode = st.infer_mode(n_support if len(poly2) >= 3 else min(n_support, 2), mast_h, hips)
        return {
            "mass_kg": mass, "com": com.tolist(), "n_contacts": n_feet, "n_contact_points": len(pts),
            "support": support, "n_support": n_support, "support_feet": sorted(sup_feet),
            "stance_clearance_m": None if stance is None else stance["clearance_m"],
            "feet_in_contact": sorted(per_foot), "contact_points": [p.tolist() for p in pts],
            "normal_forces_n": fn, "polygon": [p.tolist() for p in poly3], "margin_m": margin,
            "tip_angle_deg": tip, "edges": edges, "cop": None if cop is None else cop.tolist(),
            "imu_quat": imu_quat.tolist(), "slope_imu_deg": slope_imu, "slope_points_deg": slope_pts,
            "mast_head": None if head is None else head.tolist(), "mast_h_m": mast_h,
            "head_push_to_tip_n": head_push, "head_push_rigid_n": head_push_rigid,
            "mast_base_force_n": mast_f, "mast_base_moment_nm": mast_t,
            "arm_reach_m": reach, "hip_rad": hips, "mode": mode,
            "foot_pos": {self.foot_ids[g]: d.geom_xpos[g].tolist() for g in self.foot_ids},
        }

    def tip_angle_toward(self, result: dict, direction_xy: Sequence[float]) -> float:
        """Tip angle (deg) of the polygon edge whose outward normal best matches direction_xy
        (world horizontal, e.g. the downhill direction of a tilt test)."""
        dvec = np.array([direction_xy[0], direction_xy[1], 0.0])
        best = max(result["edges"], key=lambda e: float(np.dot(e["outward"], dvec)))
        return best["tip_deg"]


def basket_load(model: mujoco.MjModel, qpos: Sequence[float], in_basket_names: Sequence[str],
                payload_kg: float = 0.0) -> Tuple[float, List[Tuple[float, List[float]]]]:
    """Mass in the basket: the welded payload mock (already in the robot subtree) plus the crops
    game_status reports in the basket. Returns (basket_kg, crop point loads for the CoM)."""
    loads = []
    for name in in_basket_names:
        bid = _id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0 or model.body_jntnum[bid] < 1:
            continue
        a = int(model.jnt_qposadr[model.body_jntadr[bid]])
        loads.append((float(model.body_subtreemass[bid]), [float(v) for v in qpos[a:a + 3]]))
    return float(payload_kg) + sum(kg for kg, _ in loads), loads


def assess(result: dict, basket_kg: float = 0.0, head_kg: Optional[float] = None,
           leg_rms_nm_max: Optional[float] = None, t: float = 0.0, walking: bool = True,
           speed_scale: float = 1.0) -> dict:
    """Combine a probe result with the load model: classify() + the /nandi/stability payload."""
    pitch, roll = result["slope_imu_deg"]
    if result["slope_points_deg"] is not None and result["n_contacts"] >= 3:
        pitch, roll = result["slope_points_deg"]  # terrain under the feet beats body attitude
    cap = lm.basket_capacity(walking=walking, speed_scale=speed_scale)
    reach = result["arm_reach_m"]
    arm_limit = None if reach is None else lm.arm_payload_limit(reach)
    inputs = {
        "n_contacts": result["n_contacts"], "margin_m": result["margin_m"],
        "tip_angle_deg": result["tip_angle_deg"], "pitch_deg": pitch, "roll_deg": roll,
        "basket_kg": basket_kg, "basket_limit_kg": cap["limit_kg"], "mast_h_m": result["mast_h_m"],
        "head_kg": head_kg, "head_push_to_tip_n": result["head_push_to_tip_n"],
        # mast_base_t at mj_forward with qvel = 0 is the static (weight) moment only: dynamic push and
        # sway are not seen by a probe. head_force_n has no sensor yet (mast_base_f includes the weight).
        "mast_base_moment_nm": result["mast_base_moment_nm"],
    }
    verdict = st.classify(result["mode"], inputs)
    if leg_rms_nm_max is not None and leg_rms_nm_max > lm.LEG_CONTINUOUS_NM:
        verdict["reasons"].append(f"leg RMS torque {leg_rms_nm_max:.1f} Nm > {lm.LEG_CONTINUOUS_NM:.0f} Nm continuous")
    status = st.make_status(
        t=t, mode=result["mode"], n_contacts=result["n_contacts"], com=result["com"],
        margin_m=result["margin_m"], tip_angle_deg=result["tip_angle_deg"], pitch_deg=pitch, roll_deg=roll,
        head_push_to_tip_n=result["head_push_to_tip_n"], basket_kg=basket_kg,
        basket_limit_kg=cap["limit_kg"], basket_binding=cap["binding"], arm_reach_m=reach,
        arm_limit_kg=arm_limit, leg_rms_nm_max=leg_rms_nm_max,
        inhibit=verdict["inhibit"], reasons=verdict["reasons"])
    return status


LEG_JOINT_SUFFIXES = ("_hip_joint", "_thigh_joint", "_calf_joint")


def env_float(env, name: str, default: float) -> float:
    try:
        return float(env.get(name, default))
    except (TypeError, ValueError):
        return default


class MonitorCore:
    """ROS-free body of atharv_nandi/nodes/stability_monitor.py: keeps the latest inputs, builds one status on demand.

    payload_kg: the NANDI_PAYLOAD_KG basket mock (already welded into the robot's mass and CoM, so it
    only adds to the reported basket_kg). head_kg: the mast head mass for the rating check
    (NANDI_MAST_HEAD_KG, default 1.5 when a mast mock is present).

    Debounce (review F3): status() publishes an inhibit flag only once it has been raised in every status
    for the last debounce_s (stability.INHIBIT_DEBOUNCE_S), and holds the last supported mode until
    UNSUPPORTED has lasted that long, so tripod-gait transients do not block walking. mast_trip is
    immediate. The instantaneous verdict is kept as "mode_raw" / "inhibit_raw"; "reasons" are the raw
    (instantaneous) findings.
    """

    def __init__(self, model: mujoco.MjModel, payload_kg: float = 0.0, head_kg: Optional[float] = None,
                 debounce_s: float = st.INHIBIT_DEBOUNCE_S):
        self.model = model
        self.probe = StabilityProbe(model)
        self.payload_kg = max(0.0, float(payload_kg))
        self.head_kg = head_kg if self.probe.head_site >= 0 else None
        self.qpos: Optional[np.ndarray] = None
        self.in_basket: List[str] = []
        self.rms = lm.TorqueRmsWindow()
        self.bad_nq = 0
        self.debounce_s = max(0.0, float(debounce_s))
        self._raised_since: Dict[str, float] = {}
        self._unsupported_since: Optional[float] = None
        self._last_supported_mode = st.MODE_STAND

    def on_qpos(self, q: Sequence[float]) -> bool:
        if len(q) != self.model.nq:
            self.bad_nq += 1
            return False
        self.qpos = np.asarray(q, dtype=np.float64)
        return True

    def on_game(self, game: dict):
        names = game.get("in_basket_names") if isinstance(game, dict) else None
        if isinstance(names, list):
            self.in_basket = [str(n) for n in names]

    def on_joint_states(self, t: float, names: Sequence[str], efforts: Sequence[float]):
        legs = [float(e) for n, e in zip(names, efforts) if str(n).endswith(LEG_JOINT_SUFFIXES)]
        if legs:
            self.rms.add(t, legs)

    def status(self, t: float) -> Optional[dict]:
        if self.qpos is None:
            return None
        kg, loads = basket_load(self.model, self.qpos, self.in_basket, self.payload_kg)
        result = self.probe.probe(self.qpos, extra_loads=loads)
        rms = self.rms.summary()["max_rms_nm"]
        return self._debounce(t, assess(result, basket_kg=kg, head_kg=self.head_kg, leg_rms_nm_max=rms, t=t))

    def _debounce(self, t: float, status: dict) -> dict:
        raw_mode, raw_inhibit = status["mode"], list(status["inhibit"])
        self._raised_since = {f: self._raised_since.get(f, t) for f in raw_inhibit}
        held = [f for f in raw_inhibit
                if f in st.UNDEBOUNCED_INHIBITS or t - self._raised_since[f] >= self.debounce_s - 1e-9]
        mode = raw_mode
        if raw_mode == st.MODE_UNSUPPORTED:
            if self._unsupported_since is None:
                self._unsupported_since = t
            if t - self._unsupported_since < self.debounce_s - 1e-9:
                mode = self._last_supported_mode
        else:
            self._unsupported_since = None
            self._last_supported_mode = raw_mode
        status.update(mode=mode, inhibit=held, mode_raw=raw_mode, inhibit_raw=raw_inhibit)
        return status
