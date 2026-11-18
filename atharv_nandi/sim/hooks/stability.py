"""WP6 safety and stability hooks (ORCHARD-MVP-SPEC 2.3, 5.2, 5.3; MVP-SPEC 5.3, 5.6).

Two hooks share one ``StabilityRuntime`` (a fault in one leaves the other running; a hook fault also puts the runner
into safe hold, the safe direction):

    stability      50 Hz: stance set (foot contacts + 60 ms hysteresis), CoM, 6-foot and stance margins,
                   ``predicted_tripod_margin`` (worse cross-slope side, rack downhill by default, + carried payload: welded WP2 tools and basket
                   fruit + arm at PARK), local
                   slope estimate, filtered tilt, the crown supervisor (ordered wind response, margin, tilt,
                   link-loss and low-battery retracts as ``safety`` requests to the WP24 arbiter), the WP24 raise
                   gate (``hooks.mast.register_gate``), the e-stop actions (tools safe, crown brake) and e-stop stop
                   timing. Publishes ``/nandi/stability/crown`` at 10 Hz and on change.
    safety_fields  10 Hz: safety rays (mast LiDAR from ``mast_lidar``: 7 elevation fans, not the MVP-SPEC 5.3 "3 beams",
                   which leave a forearm at 0.6 m height unseen between 0.66 and 2.15 m; 2 low planes at 0.18 m;
                   ``bumper`` touch) in geom groups {0, 1} (the robot is group 2, never seen); map-masked from the model's
                   static geoms by name (``assisted: map_masked_lidar``); hull-relative protective / warning
                   fields with crown scaling, orbit pause, animal stop, WORK arm stop. Publishes
                   ``/nandi/safety_fields`` (polygons + state) at 10 Hz.

Shared on the bus (read by WP24 / WP25 / WP27 / WP28, the console, and test harnesses):

    values["stability"]       the crown / stance state (same as /nandi/stability/crown)
    values["safety_fields"]   the field state (same as /nandi/safety_fields, without polygons)
    values["motion_inhibit"]  {"stability": {...}, "safety_fields": {...}}: what motion must do now:
                              ``stance`` (stop the gait into a 6-foot stance: rule 4), ``walk`` (no walking),
                              ``stop`` (protective latch), ``speed_scale``, ``orbit_pause``, ``arm_stop``,
                              ``blade_block``, with reasons. The hooks cannot drive the gait or the hold
                              (sim_runner / sim_driver own them): ``sim_driver`` honours it on the gait, and
                              ``/nandi/motion_inhibit`` (the same dict, 10 Hz + on change) is honoured by
                              ``locomotion_service`` (``locomotion_logic.MotionInhibitGate``). In the live stack
                              ``atharv_nandi/nodes/safety_supervisor.py`` adopts a field latch as an e-stop latch (reason
                              protective / animal / bumper / stability: the MVP-SPEC 5.3 stability latch rides
                              ``/nandi/safety_fields``), which sim_runner enforces as the safe hold.

Inputs read from the bus: ``mast_control`` / ``mast`` (WP24), ``wind``, ``link``, ``platform`` (WP7),
``orbit`` ({active, arc: [[x, y, yaw], ...] in the base frame}, WP25), ``work`` ({active}, WP8 / WP29),
``tool_actuation`` (a WP2 ``ToolActuation`` or a list of them).
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from atharv_nandi.core import safety_fields as sf
from atharv_nandi.core import stability as st
from atharv_nandi.core.game_status import BASKET_FLOOR_Z, BASKET_HALF_X, BASKET_HALF_Y, BASKET_TOP_Z
from atharv_nandi.core.interfaces import GateDecision, MastRequest
from atharv_nandi.core.safety_logic import STOP_SPEED_MPS, EstopActions

LEGS = st.LEG_NAMES
FOOT_GEOMS = {leg: f"{leg}_foot" for leg in LEGS}
TRUNK, ARM_ROOT, HEAD_BODY, LIDAR_SITE, BUMPER_SENSOR = "trunk", "arm_carriage", "crown_head", "mast_lidar", "bumper"
SECTIONS = ("mast_s1", "mast_s2", "mast_s3")
ARM_POINT_SITES = ("tool_flange", "gripper_tip")
ARM_POINT_BODIES = ("arm_upper", "arm_forearm", "arm_wrist")
CROWN_TOPIC = "/nandi/stability/crown"
FIELDS_TOPIC = "/nandi/safety_fields"
ALERT_TOPIC = "/nandi/stability/alert"
INHIBIT_TOPIC = "/nandi/motion_inhibit"   # bus.values["motion_inhibit"] for locomotion_service (10 Hz + on change)
CAMERA = "crown_cam"
RATE_HZ = 50.0
FIELDS_HZ = 10.0
PUBLISH_PERIOD_S = 0.1
CONTACT_DIST_M = 0.004               # foot-ground contact (dist <= 4 mm counts: 2 x the stability CONTACT_DIST_M)
WALK_SPEED_MPS = 0.03                # trunk horizontal speed above this (EMA) = walking
SPEED_TAU_S = 0.15
ARM_ACTIVE_RADPS = 0.05
ESTOP_STILL_S = 0.2                  # e-stop timing: the trunk counts as stopped after 0.2 s below 0.02 m/s
SLOPE_MIN_AREA_M2 = 0.05             # a foot set must span this much to define the local plane (tripod: ~0.16 m2)
SLOPE_TAU_S = 0.3
# Safety rays (MVP-SPEC 5.3: the mast's 3 beams, the 2 low planes at 0.18 m, the bumper)
MAST_BEAMS_DEG = (0.0, -8.0, -16.0, -24.0, -32.0, -40.0, -50.0)   # see the module docstring: 7 fans, not 3
LOW_PLANE_Z_M = 0.18
LOW_PLANES_X = (0.36, -0.36)         # front and rear of the trunk (base frame)
RAY_STEP_DEG = 2.5                   # 6.5 cm between rays at 1.5 m: a 9 cm forearm is crossed
RAY_RANGE_M = 6.0
RAY_GROUPS = (1, 1, 0, 0, 0, 0)      # geom groups 0 and 1 only (robot 2, decals 4, envelopes 5 never)
# Geoms attached to the world body itself (ground, terrain hfield, crate, dock, laterals, patches) are left out of the
# ray test (bodyexclude 0): every one is a static map geom the mask drops anyway (tests/test_safety_fields.py: no
# unmasked static geom in legacy / orchard_lite / orchard_demo / monsoon), and the terrain hfield made one 1620-ray tick
# cost 51 ms on orchard_demo. Actors (mocap bodies) and tree bodies are still tested.
RAY_BODY_EXCLUDE = 0
POLYGON_PERIOD_S = 0.5               # display polygons are rebuilt at 2 Hz (and on a mode or latch change)
STATE_MAX_AGE_S = 0.1                # the raise gate refuses on a stability state older than this
SLOW_HZ = 10.0                       # local-slope fit and 6-foot margin rate (terrain changes slowly; SLOPE_TAU_S 0.3 s)
FIELD_REACH_M = 4.5                  # returns farther than this from the trunk centre cannot be in any field (hull <= 0.9 m
                                     # from the centre + the largest offset, 2.7 m crown-scaled warning, + margin)
LEG_MOVING_RADPS = 0.5               # any leg joint faster than this = the gait is running
ROBOT_GROUP = 2                      # WP1: every robot geom (and the WP2 tools) is in group 2, never a safety return
# Carried payload in the CoM: free bodies welded to the robot by an active equality (WP2 tools on the rack or the
# wrist, a fruit caught in a tool head) and free bodies lying in the basket (harvested fruit). A body tied to the
# world by an active equality (a fruit still on its stem) is not carried. Basket box in the trunk frame, z band
# widened for stacked fruit.
BASKET_Z_BAND_M = (BASKET_FLOOR_Z - 0.03, BASKET_TOP_Z + 0.15)
STAB_CLEAR_S = sf.CLEAR_HOLD_S       # the stability latch clears (during the hold) once the margin stayed ok this long
_HAS_MULTIRAY = hasattr(mujoco, "mj_multiRay")


def _id(m, kind, name) -> int:
    return mujoco.mj_name2id(m, kind, name)


def _name(m, kind, i) -> str:
    return mujoco.mj_id2name(m, kind, i) or ""


def _fin(v, default=0.0) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else default


class StabilityRuntime:
    def __init__(self):
        self.m = self.d = None
        self.bus = None
        self.ok = False
        self.table = st.CrownGateTable.load()
        self.crown = st.CrownSupervisor()
        self.fields = sf.FieldMonitor()
        self.stance = st.StanceSet()
        self.latch = st.StabilityLatch()
        self.estop = EstopActions()
        self.reset_runtime()

    # ------------------------------------------------------------------ binding
    def bind(self, m, d):
        if m is self.m and d is self.d:
            return
        self.m, self.d = m, d
        O = mujoco.mjtObj
        self.trunk = _id(m, O.mjOBJ_BODY, TRUNK)
        self.feet = {leg: _id(m, O.mjOBJ_GEOM, g) for leg, g in FOOT_GEOMS.items()}
        self.ok = self.trunk >= 0 and all(g >= 0 for g in self.feet.values())
        if not self.ok:
            return
        self.foot_set = {g: leg for leg, g in self.feet.items()}
        self.is_foot = np.zeros(m.ngeom, dtype=bool)
        self.is_foot[list(self.foot_set)] = True
        self.root_of = np.asarray(m.body_rootid)
        self.arm_root = _id(m, O.mjOBJ_BODY, ARM_ROOT)
        self.arm_sites = [s for s in (_id(m, O.mjOBJ_SITE, n) for n in ARM_POINT_SITES) if s >= 0]
        self.arm_bodies = [b for b in (_id(m, O.mjOBJ_BODY, n) for n in ARM_POINT_BODIES) if b >= 0]
        self.leg_dofs = [int(m.jnt_dofadr[m.actuator_trnid[a, 0]]) for a in range(7, min(25, m.nu))]
        self.head = _id(m, O.mjOBJ_BODY, HEAD_BODY)
        self.sec_kg = [float(m.body_mass[b]) for b in (_id(m, O.mjOBJ_BODY, n) for n in SECTIONS) if b >= 0]
        self.head_kg = float(m.body_subtreemass[self.head]) if self.head >= 0 else 0.0
        self.lidar = _id(m, O.mjOBJ_SITE, LIDAR_SITE)
        self.cam = _id(m, O.mjOBJ_CAMERA, CAMERA)
        self.foot_r = max(float(m.geom_size[g][0]) for g in self.feet.values())
        robot = int(self.root_of[self.trunk])
        self.eq_pairs = [(e, int(self.root_of[m.eq_obj1id[e]]), int(self.root_of[m.eq_obj2id[e]]))
                         for e in range(m.neq)
                         if int(m.eq_type[e]) in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD))
                         and int(m.eq_objtype[e]) == int(O.mjOBJ_BODY)]
        self.free_roots = np.array([b for b in range(1, m.nbody)
                                    if self.root_of[b] == b and b != robot and m.body_jntnum[b] > 0
                                    and int(m.jnt_type[m.body_jntadr[b]]) == int(mujoco.mjtJoint.mjJNT_FREE)],
                                   dtype=np.int64)
        self.robot_kg = float(m.body_subtreemass[self.trunk])
        self.total_kg = self.robot_kg
        self.carried: List[int] = []
        self._eq_key = None
        self._world_tied = {0}
        self._welded: List[int] = []
        sid = _id(m, O.mjOBJ_SENSOR, BUMPER_SENSOR)
        self.bumper_adr = int(m.sensor_adr[sid]) if sid >= 0 else -1
        # arm subtree CoM at PARK (home keyframe), in the trunk frame
        self.arm_park_b = None
        if self.arm_root >= 0 and m.nkey:
            d2 = mujoco.MjData(m)
            mujoco.mj_resetDataKeyframe(m, d2, 0)
            mujoco.mj_forward(m, d2)
            R = d2.xmat[self.trunk].reshape(3, 3)
            self.arm_park_b = R.T @ (d2.subtree_com[self.arm_root] - d2.xpos[self.trunk])
        # static map mask: class per geom (None = masked)
        self.geom_cls: List[Optional[str]] = []
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if (b and self.root_of[b] == self.root_of[self.trunk]) or int(m.geom_group[g]) == ROBOT_GROUP:
                self.geom_cls.append(None)           # the robot itself, incl. the WP2 tools (free bodies, group 2)
                continue
            self.geom_cls.append(sf.classify_geom(_name(m, O.mjOBJ_GEOM, g), _name(m, O.mjOBJ_BODY, b)))
        self.unmasked_static = [ _name(m, O.mjOBJ_GEOM, g) for g, c in enumerate(self.geom_cls)
                                 if c == sf.CLS_OBSTACLE and not _name(m, O.mjOBJ_GEOM, g).startswith("actor_")]
        # geoms a safety ray can return as an obstacle (unmasked, ray groups 0 / 1): when none is within
        # FIELD_REACH_M of the trunk the cast is skipped (exact: a ray only ever returns a geom it hits)
        self.obstacle_gids = np.array([g for g, c in enumerate(self.geom_cls)
                                       if c is not None and int(m.geom_group[g]) in (0, 1)], dtype=np.int64)
        self._build_rays()

    def _build_rays(self):
        az = np.radians(np.arange(0.0, 360.0, RAY_STEP_DEG))
        dirs = []
        srcs = []
        for el in MAST_BEAMS_DEG:
            e = math.radians(el)
            dirs.append(np.stack([np.cos(az) * math.cos(e), np.sin(az) * math.cos(e), np.full_like(az, math.sin(e))], 1))
            srcs.append(f"mast_{int(abs(el))}")
        self.mast_dirs = np.concatenate(dirs)                     # base frame (yawed with the trunk)
        self.mast_src = np.repeat(np.arange(len(MAST_BEAMS_DEG)), len(az))
        self.low_dirs = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], 1)
        self.src_names = srcs + [f"low_{i}" for i in range(len(LOW_PLANES_X))]
        self.n_az = len(az)

    def reset_runtime(self):
        self.crown.reset()
        self.fields.reset()
        self.stance.reset()
        self.latch = st.StabilityLatch()
        self.estop = EstopActions()
        self.v_ema = 0.0
        self.last_xy: Optional[np.ndarray] = None
        self.last_t: Optional[float] = None
        self.state: Dict[str, Any] = {}
        self.field_state: Dict[str, Any] = {}
        self.slope = (0.0, 0.0)
        self._slope_t: Optional[float] = None
        self.was_holding = False
        self.estop_runs: List[dict] = []
        self._estop_cur: Optional[dict] = None
        self.retract_log: List[dict] = []
        self.submitted: List[dict] = []
        self._next_pub = 0.0
        self._pub_key = None
        self.min_predicted = math.inf
        self.ray_returns = 0
        self._next_slow = 0.0
        self._hexa = math.inf
        self._next_near = 0.0
        self._near: List[str] = []
        self._gates = None
        self._poly = None
        self._poly_key = None
        self._next_poly = 0.0
        self.v_fwd_ema = 0.0
        self.stab_ok_since: Optional[float] = None
        self.stab_handed = False          # the stability latch was set during this hold (its clear gates the reset)
        self.bumper_seen = False          # bumper touch sampled at 50 Hz, consumed by the 10 Hz field tick
        self._inhibit_key = None
        self._next_inhibit = 0.0

    def reset(self, m, d):
        self.reset_runtime()

    # ------------------------------------------------------------------ measurements
    def frame(self) -> Tuple[np.ndarray, np.ndarray]:
        d = self.d
        return d.xmat[self.trunk].reshape(3, 3), d.xpos[self.trunk].copy()

    def feet_in_contact(self) -> List[str]:
        m, d = self.m, self.d
        n = int(d.ncon)
        if n == 0:
            return []
        c = d.contact
        g1, g2 = np.asarray(c.geom1[:n]), np.asarray(c.geom2[:n])
        ok = np.asarray(c.dist[:n]) <= CONTACT_DIST_M
        robot = self.root_of[self.trunk]
        out = set()
        for ga, gb in ((g1, g2), (g2, g1)):
            sel = ok & self.is_foot[ga] & (self.root_of[m.geom_bodyid[gb]] != robot)
            out.update(self.foot_set[int(g)] for g in ga[sel])
        return sorted(out)

    def feet_near_ground(self) -> List[str]:
        """Feet whose sphere is within STANCE_NEAR_M of the ground straight below (along gravity): on rough ground
        a stiff-legged stance leaves a foot a few mm up (stability.STANCE_NEAR_M, review F3); it still carries
        the robot the moment it is loaded, so it counts for the stance set."""
        m, d = self.m, self.d
        gdir = np.asarray(m.opt.gravity, dtype=np.float64)
        gdir = gdir / max(float(np.linalg.norm(gdir)), 1e-9)
        groups = np.asarray((1, 0, 0, 0, 0, 0), dtype=np.uint8)
        out = []
        g1 = np.zeros(1, dtype=np.int32)
        for leg, g in self.feet.items():
            r = float(m.geom_size[g][0])
            dist = mujoco.mj_ray(m, d, d.geom_xpos[g].copy(), gdir, groups, 1, -1, g1)
            if 0.0 <= dist - r <= st.STANCE_NEAR_M:
                out.append(leg)
        return out

    def carried_roots(self, R: np.ndarray, p0: np.ndarray) -> List[int]:
        """Free root bodies the robot carries now (module constants): welded to the robot through a chain of active
        equalities and not tied to the world, or lying in the basket box."""
        m, d = self.m, self.d
        robot = int(self.root_of[self.trunk])
        key = bytes(np.asarray(d.eq_active, dtype=np.uint8)) if m.neq else b""
        if key != self._eq_key:                            # the weld graph only changes when an equality toggles
            act = [(a, b) for e, a, b in self.eq_pairs if d.eq_active[e]]
            world, held = {0}, {robot}
            for grp in (world, held):
                grow = True
                while grow:
                    grow = False
                    for a, b in act:
                        for x, y in ((a, b), (b, a)):
                            if x in grp and y not in grp and not (grp is held and y in world):
                                grp.add(y)
                                grow = True
            self._eq_key = key
            self._world_tied = world
            self._welded = sorted(b for b in held if b != robot and b not in world)
        world = self._world_tied
        out = list(self._welded)
        if len(self.free_roots):
            P = (d.subtree_com[self.free_roots] - p0) @ R               # trunk frame (R.T @ v per row)
            inb = ((np.abs(P[:, 0]) < BASKET_HALF_X) & (np.abs(P[:, 1]) < BASKET_HALF_Y)
                   & (P[:, 2] >= BASKET_Z_BAND_M[0]) & (P[:, 2] <= BASKET_Z_BAND_M[1]))
            out += [int(b) for b in self.free_roots[inb] if int(b) not in world and int(b) not in out]
        return sorted(out)

    def com_b(self) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Whole-robot CoM in the trunk frame, carried payload included (WP2 tools, basket fruit), feet, gravity and
        the arm-at-PARK shift."""
        R, p0 = self.frame()
        d, m = self.d, self.m
        self.carried = self.carried_roots(R, p0)
        M = self.robot_kg
        c = d.subtree_com[self.trunk] * M
        for b in self.carried:
            mb = float(m.body_subtreemass[b])
            c = c + d.subtree_com[b] * mb
            M += mb
        self.total_kg = M
        com_b = R.T @ (c / M - p0)
        feet_b = {leg: R.T @ (d.geom_xpos[g] - p0) for leg, g in self.feet.items()}
        g_b = R.T @ np.asarray(m.opt.gravity, dtype=np.float64)
        shift = np.zeros(3)
        if self.arm_park_b is not None:
            arm_b = R.T @ (d.subtree_com[self.arm_root] - p0)
            shift = float(m.body_subtreemass[self.arm_root]) / M * (self.arm_park_b - arm_b)
        return com_b, feet_b, g_b, shift

    def lens_height(self) -> Optional[float]:
        """Crown camera height above the local ground (the lowest foot contact point), along gravity: not the world z
        (a basin or a dip lowers that) and right on tilted-gravity test planes."""
        if self.cam < 0:
            return None
        d = self.d
        up = -np.asarray(self.m.opt.gravity, dtype=np.float64)
        up = up / max(float(np.linalg.norm(up)), 1e-9)
        ground = min(float(d.geom_xpos[g] @ up) for g in self.feet.values()) - self.foot_r
        return float(d.cam_xpos[self.cam] @ up) - ground

    def gravity_rot(self) -> np.ndarray:
        """World -> gravity-aligned frame (z = up = -g): slopes and tilts are relative to gravity, so a tilted-gravity
        test plane reads like real terrain."""
        return st.gravity_frame(np.asarray(self.m.opt.gravity, dtype=np.float64))

    def trunk_tilt(self) -> Tuple[float, float]:
        """(roll_deg, pitch_deg) of the trunk relative to gravity: roll > 0 left side up, pitch > 0 nose up."""
        R = self.gravity_rot() @ self.d.xmat[self.trunk].reshape(3, 3)
        pitch, roll = st.slope_from_mat(R)
        return roll, pitch

    def local_slope(self, t: float, legs: List[str]) -> Tuple[float, float]:
        """Plane of the stance feet relative to gravity, in the heading frame: (roll_deg, pitch_deg) with roll > 0
        left side up and pitch > 0 nose up. Only a foot set spanning >= SLOPE_MIN_AREA_M2 counts (three feet on one
        side give no plane); low-passed over SLOPE_TAU_S (terrain does not change faster); kept otherwise."""
        if len(legs) >= 3:
            Rg = self.gravity_rot()
            Rt = Rg @ self.d.xmat[self.trunk].reshape(3, 3)
            yaw = math.atan2(Rt[1, 0], Rt[0, 0])
            pts = [Rg @ self.d.geom_xpos[self.feet[k]] for k in legs]
            hull = st.convex_hull([(p[0], p[1]) for p in pts])
            area = 0.5 * abs(sum(hull[i][0] * hull[(i + 1) % len(hull)][1] - hull[(i + 1) % len(hull)][0] * hull[i][1]
                                 for i in range(len(hull)))) if len(hull) >= 3 else 0.0
            pr = st.slope_from_points(pts, yaw) if area >= SLOPE_MIN_AREA_M2 else None
            if pr is not None:
                new = (pr[1], pr[0])
                if self._slope_t is None:
                    self.slope = new
                else:
                    a = min(1.0, max(0.0, t - self._slope_t) / SLOPE_TAU_S)
                    self.slope = (self.slope[0] + a * (new[0] - self.slope[0]), self.slope[1] + a * (new[1] - self.slope[1]))
                self._slope_t = t
        return self.slope

    def _speed(self, t: float) -> float:
        xy = self.d.qpos[0:2].copy()
        if self.last_xy is not None and self.last_t is not None and t > self.last_t:
            dxy = xy - self.last_xy
            v = float(np.linalg.norm(dxy)) / (t - self.last_t)
            yaw = st.yaw_from_quat(self.d.qpos[3:7])
            v_fwd = float(dxy[0] * math.cos(yaw) + dxy[1] * math.sin(yaw)) / (t - self.last_t)
            a = min(1.0, (t - self.last_t) / SPEED_TAU_S)
            self.v_ema += a * (v - self.v_ema)
            self.v_fwd_ema += a * (v_fwd - self.v_fwd_ema)
        self.last_xy, self.last_t = xy, t
        return self.v_ema

    def _mast(self):
        mc = self.bus.values.get("mast_control") if self.bus is not None else None
        ms = self.bus.values.get("mast") if self.bus is not None else None
        ext = _fin((ms or {}).get("ext_m"), 0.0) if isinstance(ms, dict) else 0.0
        lens = (ms or {}).get("lens_z") if isinstance(ms, dict) else None
        return mc, ext, (_fin(lens, None) if lens is not None else None), ms or {}

    def _wind(self) -> Tuple[float, bool]:
        w = self.bus.values.get("wind") or {}
        if not isinstance(w, dict):
            return 0.0, False
        return _fin(w.get("speed_mps"), 0.0), bool(w.get("gust"))

    def _links(self) -> Tuple[bool, bool]:
        lk = self.bus.values.get("link") or {}
        local_ok = not (isinstance(lk, dict) and isinstance(lk.get("local"), dict) and lk["local"].get("state") == "lost")
        pf = self.bus.values.get("platform") or {}
        low = isinstance(pf, dict) and pf.get("energy_action") == "return_and_wait"
        return local_ok, low

    # ------------------------------------------------------------------ WP24 raise gate
    def raise_gate(self, ext_from: float, ext_to: float, ctx: dict) -> GateDecision:
        """``hooks.mast.register_gate``: OS 2.3 rule 1 (WP6 part) on the live state. Applies only to the model this
        runtime's ``stability`` hook ticks (the gate list is process-wide; another driver's model is not ours), and
        only with a fresh tick (<= 0.1 s old); a stale state refuses (fail safe)."""
        m, d = ctx.get("m"), ctx.get("d")
        if m is None or d is None:
            return GateDecision.refuse(st.GATE_RAISE, st.R_STABILITY, detail="stability_no_model")
        if d is not self.d or m is not self.m:
            return GateDecision.ok(st.GATE_RAISE)            # another driver's model (the gate list is process-wide)
        if not self.ok:
            return GateDecision.refuse(st.GATE_RAISE, st.R_STABILITY, detail="stability_unbound")
        if not self.state or abs(float(d.time) - float(self.state.get("t", -1.0))) > STATE_MAX_AGE_S:
            return GateDecision.refuse(st.GATE_RAISE, st.R_STABILITY, detail="stability_state_stale")
        if ctx.get("bus") is not None:
            self.bus = ctx["bus"]
        s = self.state or {}
        com_b, feet_b, g_b, shift = self.com_b()
        dz = st.mast_com_shift(ext_to - ext_from, self.sec_kg, self.head_kg, self.total_kg)
        hexa = st.hexa_margin(com_b, feet_b, g_b, com_shift_b=shift + dz)
        wind, gust = self._wind()
        roll, pitch = s.get("slope_deg", {}).get("roll", 0.0), s.get("slope_deg", {}).get("pitch", 0.0)
        n = len(s.get("stance", [])) if s else len(self.feet_in_contact())
        walking = bool(s.get("walking", False))
        dec = st.gate_raise(ext_from, ext_to, hexa, roll, wind, gust, not walking, n,
                            estop=bool(self.bus is not None and self.bus.holding()),
                            tilt_deg=st.slope_tilt_deg(roll, pitch))
        if dec and s.get("stance_request"):
            return GateDecision.refuse(st.GATE_RAISE, st.R_WIND, detail="wind_stance")
        if dec and s.get("walk_inhibit"):
            return GateDecision.refuse(st.GATE_RAISE, st.R_STABILITY, detail="walk_inhibit")
        return dec

    # ------------------------------------------------------------------ 50 Hz
    def tick(self, m, d, t, bus):
        self.bind(m, d)
        self.bus = bus
        if not self.ok:
            bus.values["stability"] = {"ok": False}
            return
        holding = bus.holding()
        mc, ext, lens, ms = self._mast()
        self._estop(t, holding, mc, ext)
        contacts = self.feet_in_contact()
        if len(contacts) >= 6:
            near = []
        else:                                              # ray tests on the terrain: at SLOW_HZ, held in between
            if t + 1e-9 >= self._next_near:
                self._near = self.feet_near_ground()
                self._next_near = t + 1.0 / SLOW_HZ
            near = self._near
        stance = self.stance.update(t, sorted(set(contacts) | set(near)))
        com_b, feet_b, g_b, shift = self.com_b()
        roll, pitch = self.trunk_tilt()
        slow = t + 1e-9 >= self._next_slow                  # slope fit and 6-foot margin at SLOW_HZ (state, gates)
        if slow:
            self._next_slow = t + 1.0 / SLOW_HZ
            self.local_slope(t, sorted(stance))
            self._hexa = st.hexa_margin(com_b, feet_b, g_b)
        s_roll, s_pitch = self.slope
        hexa = self._hexa
        pred = st.predicted_tripod_margin(com_b, feet_b, g_b, com_shift_b=shift, terrain_roll_deg=s_roll)
        Rgb = st.gravity_frame(g_b)
        live = (st.com_margin((Rgb @ com_b)[:2], st.convex_hull([(Rgb @ feet_b[k])[:2] for k in stance]))
                if len(stance) >= 3 else -math.inf)
        v = self._speed(t)
        legs_moving = bool(np.max(np.abs(d.qvel[self.leg_dofs])) > LEG_MOVING_RADPS) if self.leg_dofs else False
        walking = v > WALK_SPEED_MPS or legs_moving
        wind, gust = self._wind()
        local_ok, low_bat = self._links()
        arm_active = bool(np.max(np.abs(d.qvel[[int(m.jnt_dofadr[m.actuator_trnid[a, 0]]) for a in range(min(5, m.nu))]]))
                          > ARM_ACTIVE_RADPS) if m.nu >= 5 else False
        work = bus.values.get("work") or {}
        state = "WALK" if walking else ("WORK" if (arm_active or (isinstance(work, dict) and work.get("active")))
                                        else "STOPPED")
        retracting = bool(isinstance(ms, dict) and ms.get("arbiter_source") == "safety"
                          and _fin(ms.get("target_ext"), 1.0) <= 1e-6)
        # during the hold no retract can be sent (e-stop beats safety): the crown supervisor keeps its event for the
        # first tick after the release (a wind retract re-fires then, review WP6 F6)
        out = self.crown.update(t, ext, walking, len(stance) >= 6 and not walking, pred["margin_m"], roll, pitch,
                                s_roll, s_pitch, wind, gust, local_ok, low_bat, retracting=retracting or holding)
        if self.bumper_adr >= 0 and float(d.sensordata[self.bumper_adr]) > 1e-6:
            self.bumper_seen = True
        if out["retract"] and mc is not None and not holding:
            dec = mc.submit(MastRequest("safety", 0.0, out["retract"], t=float(t)))
            ev = {"t": round(float(t), 4), "reason": out["retract"], "ext_m": round(ext, 4), "accepted": bool(dec)}
            self.retract_log.append(ev)
            bus.publish(ALERT_TOPIC, {"id": "crown_retract", **ev})
        latched = self.latch.update(t, state, live, len(stance))
        # MVP-SPEC 5.3 stability latch -> e-stop latch: the latch rides /nandi/safety_fields (reason stability), the
        # supervisor adopts it and sim_runner safe-holds. During the hold the latch is handed over and clears once
        # the live stance margin stayed >= LATCH_MARGIN_M for STAB_CLEAR_S; /nandi/estop_reset is refused until then.
        margin_ok = len(stance) >= 3 and math.isfinite(live) and live >= st.LATCH_MARGIN_M
        if margin_ok:
            if self.stab_ok_since is None:
                self.stab_ok_since = float(t)
        else:
            self.stab_ok_since = None
        if latched and holding:
            self.stab_handed = True
            if self.stability_clear(t):
                self.latch.reset(True)
                latched = False
        if not holding:
            self.stab_handed = False
        if walking and ext > st.CROWN_UP_EXT_M:
            self.min_predicted = min(self.min_predicted, pred["margin_m"])
        if slow or self._gates is None:                    # display only (the gates act through their own calls)
            tilt = st.slope_tilt_deg(s_roll, s_pitch)
            self._gates = {"walk": st.gate_crown_walk(ext, pred["margin_m"], v, wind, gust, tilt,
                                                      starting=not walking).allowed,
                           "slope": st.gate_walk_slope(s_roll, s_pitch).allowed,
                           "raise_hexa_ok": hexa >= st.RAISE_MIN_HEXA_M}
        gates = self._gates
        self.state = {
            "t": round(float(t), 4), "ext_m": round(ext, 4), "lens_z": lens, "walking": walking, "state": state,
            "speed_mps": round(v, 4), "speed_fwd_mps": round(self.v_fwd_ema, 4),
            "mass_kg": round(self.total_kg, 3), "carried": len(self.carried), "stance": sorted(stance), "contacts": contacts, "degenerate": len(stance) < 3,
            # margin_m: the predicted worst-case margin other packages gate on (WP2 tool swap and blade close, WP8 IK:
            # MVP-SPEC 5.3 0.03 m); stance_margin_m: the live CoM margin to the stance-set polygon
            "margin_m": round(pred["margin_m"], 4) if math.isfinite(pred["margin_m"]) else None,
            "stance_margin_m": round(live, 4) if math.isfinite(live) else None,
            "hexa_m": round(hexa, 4) if math.isfinite(hexa) else None,
            "predicted_tripod_m": round(pred["margin_m"], 4) if math.isfinite(pred["margin_m"]) else None,
            "tripod_A_m": round(pred["A"], 4), "tripod_B_m": round(pred["B"], 4),
            "worst_case": "both_sides+park", "worst_side": pred["side"],
            "com_b": [round(float(x), 4) for x in com_b], "tilt_deg": {"roll": round(roll, 3), "pitch": round(pitch, 3)},
            "slope_deg": {"roll": round(s_roll, 3), "pitch": round(s_pitch, 3)},
            "tilt_filtered_deg": out["tilt_filtered_deg"], "wind_state": out["wind_state"],
            "stance_request": out["stance_request"], "walk_inhibit": out["walk_inhibit"],
            "retract_reason": out["retract_reason"], "retracts": len(self.retract_log), "gates": gates,
            "latched": latched, "latch_reason": self.latch.reason, "stability_clear": self.stability_clear(t),
            "estop": self._estop_cur or
            (self.estop_runs[-1] if self.estop_runs else None),
            "table_run_id": self.table.run_id}
        bus.values["stability"] = self.state
        mi = bus.values.setdefault("motion_inhibit", {})
        mi["stability"] = {"stance": out["stance_request"], "walk": bool(out["walk_inhibit"]) or latched,
                           "stop": latched, "reasons": out["walk_inhibit"] + (["stability"] if latched else [])}
        self._publish_inhibit(t, mi)
        key =(out["wind_state"], out["stance_request"], len(self.retract_log), latched, state, bool(holding))
        if key != self._pub_key or t + 1e-9 >= self._next_pub:
            bus.publish(CROWN_TOPIC, self.state)
            self._pub_key = key
            self._next_pub = t + PUBLISH_PERIOD_S

    def stability_clear(self, t: float) -> bool:
        """The live stance margin stayed >= LATCH_MARGIN_M for STAB_CLEAR_S (the stability cause is gone)."""
        return self.stab_ok_since is not None and t - self.stab_ok_since >= STAB_CLEAR_S - 1e-9

    def _publish_inhibit(self, t: float, mi: dict):
        """``/nandi/motion_inhibit`` (locomotion_service honours it): 10 Hz and on every change."""
        sf_ = mi.get("safety_fields") or {}
        s_ = mi.get("stability") or {}
        key = (bool(s_.get("stance")), bool(s_.get("walk")), bool(s_.get("stop")), bool(sf_.get("stop")),
               sf_.get("speed_scale"))
        if key != self._inhibit_key or t + 1e-9 >= self._next_inhibit:
            self.bus.publish(INHIBIT_TOPIC, {"t": round(float(t), 4), **{k: dict(v) for k, v in mi.items()}})
            self._inhibit_key, self._next_inhibit = key, t + PUBLISH_PERIOD_S

    def _estop(self, t: float, holding: bool, mc, ext: float):
        """Rising edge of the runner hold: tools safe + crown brake (EstopActions) and stop timing."""
        if holding and not self.was_holding:
            tools = self.bus.values.get("tool_actuation")
            tools = list(tools) if isinstance(tools, (list, tuple)) else ([tools] if tools is not None else [])
            self.estop = EstopActions(tools, mc)
            self.estop.engage(t, str((self.bus.values.get("hold") or {}).get("reason", "estop")))
            self._estop_cur = {"t_engage": round(float(t), 4), "ext_at_engage_m": round(ext, 4),
                               "actions": self.estop.log[-1]["actions"], "stopped": None,
                               "v_engage_mps": round(float(np.linalg.norm(self.d.qvel[0:2])), 4)}
            self._estop_xy0 = self.d.qpos[0:2].copy()
            self._estop_below: Optional[float] = None
        elif not holding and self.was_holding:
            self.estop.release()
            if self._estop_cur is not None:
                self.estop_runs.append(self._estop_cur)
                self._estop_cur = None
        if holding and self._estop_cur is not None:
            # stopped = trunk speed < 0.02 m/s (safety_logic.STOP_SPEED_MPS) continuously for ESTOP_STILL_S
            v = float(np.linalg.norm(self.d.qvel[0:2]))
            if self._estop_cur["stopped"] is None:
                if v < STOP_SPEED_MPS:
                    if self._estop_below is None:
                        self._estop_below = float(t)
                    if t - self._estop_below >= ESTOP_STILL_S - 1e-9:
                        self._estop_cur["stopped"] = {
                            "t_s": round(self._estop_below - self._estop_cur["t_engage"], 3),
                            "dist_m": round(float(np.linalg.norm(self.d.qpos[0:2] - self._estop_xy0)), 4)}
                else:
                    self._estop_below = None
            self._estop_cur["ext_now_m"] = round(ext, 4)
        self.was_holding = holding

    # ------------------------------------------------------------------ 10 Hz fields
    def cast(self) -> List[sf.Return]:
        m, d = self.m, self.d
        R, p0 = self.frame()
        yaw = st.yaw_from_quat(d.qpos[3:7])
        c, s = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        origins, dirs, src = [], [], []
        if self.lidar >= 0:
            o = d.site_xpos[self.lidar].copy()
            dw = self.mast_dirs @ Rz.T
            origins.append(np.repeat(o[None, :], len(dw), 0))
            dirs.append(dw)
            src.append(self.mast_src)
        z_ground = float(min(d.geom_xpos[g][2] for g in self.feet.values()))
        for i, x in enumerate(LOW_PLANES_X):
            o = p0 + Rz @ np.array([x, 0.0, 0.0])
            o[2] = z_ground + LOW_PLANE_Z_M
            origins.append(np.repeat(o[None, :], self.n_az, 0))
            dirs.append(self.low_dirs @ Rz.T)
            src.append(np.full(self.n_az, len(MAST_BEAMS_DEG) + i))
        if not origins:
            return []
        P = np.concatenate(origins)
        V = np.concatenate(dirs)
        S = np.concatenate(src)
        gid, dist = self._rays(P, V, [len(x) for x in origins])
        hit = (gid >= 0) & (dist >= 0.0) & (dist <= RAY_RANGE_M)
        self.ray_returns = int(hit.sum())
        out = []
        if hit.any():
            W = P[hit] + V[hit] * dist[hit][:, None]
            B = sf.to_base_frame(W, p0[:2], yaw)
            near = np.hypot(B[:, 0], B[:, 1]) <= FIELD_REACH_M     # beyond this no field can reach
            for (x, y, _z), w, g, s_ in zip(B[near], W[near], gid[hit][near], S[hit][near]):
                cls = self.geom_cls[int(g)] if 0 <= g < len(self.geom_cls) else sf.CLS_OBSTACLE
                if cls is None:
                    continue
                out.append(sf.Return(float(x), float(y), float(w[2]), cls, _name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)),
                                     self.src_names[int(s_)]))
        return out

    def obstacle_within_reach(self) -> bool:
        if len(self.obstacle_gids) == 0:
            return False
        d = self.d
        p = d.xpos[self.trunk]
        q = d.geom_xpos[self.obstacle_gids]
        r = np.hypot(q[:, 0] - p[0], q[:, 1] - p[1]) - self.m.geom_rbound[self.obstacle_gids]
        return bool(np.min(r) <= FIELD_REACH_M)

    def _rays(self, P: np.ndarray, V: np.ndarray, sizes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """One mj_multiRay per fan (same origin), mj_ray per ray where multiRay is missing."""
        m, d = self.m, self.d
        groups = np.asarray(RAY_GROUPS, dtype=np.uint8)
        n = len(P)
        gid = np.full(n, -1, dtype=np.int32)
        dist = np.full(n, -1.0)
        i = 0
        for k in sizes:
            if _HAS_MULTIRAY:
                g = np.full(k, -1, dtype=np.int32)
                r = np.full(k, -1.0)
                mujoco.mj_multiRay(m, d, np.ascontiguousarray(P[i]), np.ascontiguousarray(V[i:i + k].ravel()),
                                   groups, 1, RAY_BODY_EXCLUDE, g, r, k, RAY_RANGE_M)
                gid[i:i + k], dist[i:i + k] = g, r
            else:  # pragma: no cover
                for j in range(i, i + k):
                    g1 = np.zeros(1, dtype=np.int32)
                    dist[j] = mujoco.mj_ray(m, d, P[j], V[j], groups, 1, RAY_BODY_EXCLUDE, g1)
                    gid[j] = g1[0]
            i += k
        return gid, dist

    def hull(self, ext: float) -> List[Tuple[float, float]]:
        d = self.d
        p0 = d.xpos[self.trunk]
        yaw = st.yaw_from_quat(d.qpos[3:7])

        def b(pts):
            return sf.to_base_frame(np.asarray(pts).reshape(-1, 3), p0[:2], yaw)[:, :2]

        feet = b([d.geom_xpos[g] for g in self.feet.values()])
        arm = b([d.site_xpos[s] for s in self.arm_sites] + [d.xpos[x] for x in self.arm_bodies]) if (
            self.arm_sites or self.arm_bodies) else []
        head = b([d.xpos[self.head]])[0] if (self.head >= 0 and ext > st.CROWN_UP_EXT_M) else None
        return sf.robot_hull(feet, arm, head)

    def fields_tick(self, m, d, t, bus):
        self.bind(m, d)
        self.bus = bus
        if not self.ok:
            bus.values["safety_fields"] = {"ok": False}
            return
        _mc, ext, lens, _ms = self._mast()
        s = self.state or {}
        orbit = bus.values.get("orbit") or {}
        arc = None
        if isinstance(orbit, dict) and orbit.get("active"):
            mode = sf.MODE_ORBIT
            arc = sf.OrbitArc([tuple(float(x) for x in p[:3]) for p in (orbit.get("arc") or [])])
        elif s.get("walking"):
            mode = sf.MODE_WALK
        elif s.get("state") == "WORK":
            mode = sf.MODE_WORK
        else:
            mode = sf.MODE_STOPPED
        # bumper: any touch since the last field tick (sampled at 50 Hz by the stability tick) or now
        bumper = self.bumper_seen or bool(self.bumper_adr >= 0 and float(d.sensordata[self.bumper_adr]) > 1e-6)
        self.bumper_seen = False
        holding = bus.holding()
        rets = self.cast() if self.obstacle_within_reach() else []
        lh = self.lens_height()
        lens_h = lh if lh is not None else lens
        # signed speed: the d_stop(v) term goes to the sector in the direction of travel (front, or rear in reverse)
        v_mag = _fin(s.get("speed_mps"), 0.0)
        v_sgn = -v_mag if _fin(s.get("speed_fwd_mps"), 0.0) < 0.0 else v_mag
        fs_ = self.fields.update(t, mode, rets, self.hull(ext), v=v_sgn,
                                 lens_z=lens_h, ext=ext, arc=arc, bumper=bumper)
        if holding and self.fields.latched:
            # The runner holds (the supervisor adopted the latch as an e-stop latch): the hook latch is handed over
            # and clears once the field stayed clear CLEAR_HOLD_S; /nandi/estop_reset is still refused until then.
            if self.fields.clear(t)[0]:
                fs_.latched, fs_.latch_reason = False, ""
        body = fs_.as_dict(polygons=False)
        detail = self.fields.events[-1]["source"] if self.fields.events else ""
        stab_clear = self.stability_clear(t)
        prot_clear = self.fields.protective_clear(t)
        if self.latch.latched or self.stab_handed:
            prot_clear = prot_clear and stab_clear           # the stability cause gates the e-stop reset too
        if self.latch.latched and not fs_.latched:
            # the MVP-SPEC 5.3 stability latch rides the field latch: the supervisor adopts reason 'stability'
            body.update({"latched": True, "latch_reason": "stability", "stop": True, "speed_scale": 0.0})
            detail = self.latch.reason
        body.update({"protective_clear": prot_clear, "stability_clear": stab_clear, "ext_m": round(ext, 4),
                     "lens_z": lens, "lens_h": None if lens_h is None else round(lens_h, 4), "returns": len(rets),
                     "rays_hit": self.ray_returns, "detail": detail, "events": len(self.fields.events)})
        self.field_state = body
        bus.values["safety_fields"] = body
        # WP2 interlock input (the WP2 contract notes): anything but an animal inside the WORK warning field (hull +
        # 1.0 m, crown-scaled) freezes the arm and blocks blade close / share lowering, whatever the motion mode
        work_off = sf.field_offsets(sf.MODE_WORK, 0.0, lens_h, ext)
        wp, ww = self.fields.classify(rets, fs_.hull, work_off)
        bus.values["work_field"] = {"t": round(float(t), 4), "person": any(h.cls != sf.CLS_ANIMAL for h in wp + ww),
                                    "animal": any(h.cls == sf.CLS_ANIMAL for h in wp + ww), "n": len(wp) + len(ww)}
        mi = bus.values.setdefault("motion_inhibit", {})
        mi["safety_fields"] ={"stop": fs_.latched, "speed_scale": fs_.speed_scale, "orbit_pause": fs_.orbit_pause,
                               "arm_stop": fs_.arm_stop, "blade_block": fs_.blade_block,
                               "reasons": [fs_.latch_reason] if fs_.latched else []}
        if self._poly is None or t + 1e-9 >= self._next_poly or self._poly_key != (mode, fs_.latched):
            self._poly = {k: sf.field_polygon(fs_.hull, o, n=24) for k, o in fs_.offsets.items()}
            self._poly_key, self._next_poly = (mode, fs_.latched), t + POLYGON_PERIOD_S
        bus.publish(FIELDS_TOPIC, {**body, "hull": [[round(p[0], 3), round(p[1], 3)] for p in fs_.hull],
                                   "polygons": self._poly})


_RT: List[StabilityRuntime] = []


def register(hooks):
    """``stability`` (50 Hz) and ``safety_fields`` (10 Hz); registers the WP6 raise gate with the WP24 hook."""
    rt = StabilityRuntime()
    hooks.register("stability", rt.tick, rate_hz=RATE_HZ, reset=rt.reset)
    hooks.register("safety_fields", rt.fields_tick, rate_hz=FIELDS_HZ)
    try:
        from atharv_nandi.sim.hooks import mast as hm
        for old in _RT:
            hm.unregister_gate(old.raise_gate)
        hm.register_gate(rt.raise_gate)
    except ImportError:  # pragma: no cover - the mast hook is always present on the pilot branch
        pass
    _RT[:] = [rt]
    return rt
