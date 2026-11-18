"""MuJoCo collision scene for planning the 5-DOF dorsal arm (WPCAP).

The scene owns a private copy of the compiled world (``copy.deepcopy(model)``) and its own ``MjData``, so a
validity check never touches the live sim. A check is one ``mj_kinematics`` + one ``mj_collision`` in C:

* **Moving set**: every collidable geom of the arm subtree (``arm_turret`` and below: turret, upper arm,
  forearm, palm, finger pads, and any tool body mounted under the wrist) plus the geoms of *attached* bodies
  (a held fruit or tool, carried rigidly with ``arm_wrist``, together with every body below its free root: a
  mounted tool's head, blades and jaws ride on hinge / slide joints under the handle).
* **Obstacle set**: every other geom that can physically touch the moving set (contype/conaffinity rule of the
  live model), plus *keep-out* robot parts that are contype 0 in the sim but solid on hardware (the crown mast
  sections and head, the desap tray). Each obstacle has a class with its own clearance margin.

The private model gets one planning bit: moving geoms ``contype = BIT, conaffinity = 0``, obstacles
``contype = 0, conaffinity = BIT``, everything else 0/0, so MuJoCo's broad phase only produces moving-vs-
obstacle pairs (legs on the ground, fruit on wood etc. are never generated). The clearance of a class is its
geom margin: any contact the narrow phase reports (dist < margin) makes the state invalid. MuJoCo's parent /
child filter and the model's ``<exclude>`` pairs still apply (turret vs carriage is a parent-child pair).

Explicit ``<pair>`` elements (the WPCAP finger x pool-fruit pairs) are checked by MuJoCo regardless of contype /
conaffinity, so the masks cannot switch them off. The scene therefore (a) gives an explicit pair the obstacle
class margin when it joins a moving geom to an obstacle geom, and (b) counts only moving x obstacle contacts:
pairs between two moving geoms (pad vs the held fruit), or with an ignored / non-collidable geom (the pick target,
a parked contype-0 slot), never make a state invalid.

Classes and default margins (m; override per scene or in ``config/hal.yaml`` ``planning.margins``):

    robot     0.005   trunk, rail, basket, legs, rack, bumper (WP0c ROBOT_CLEAR_M)
    carriage  0.000   gantry carriage vs upper arm: actual contact only (as DorsalArmIKSolver.robot_collisions)
    keepout   0.010   crown mast + head, tray (contype 0 in the sim, solid on the robot)
    wood      0.010   tree wood (contype 1 / conaffinity 0 tree geoms)
    fruit     0.010   assigned pool fruit and legacy crops (the pick target is ignored per plan)
    fixture   0.010   crate, dock, props
    ground    0.020   floor / orchard terrain (WP0c GROUND_CLEAR_M)

The live model mutates at run time (fruit pool assign/release toggles contype and sizes, the mast writes
``body_pos``), so ``sync(model, data)`` copies those fields and the state before every plan.
"""
import copy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

ARM_JOINTS = ("gantry_slide", "arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist_pitch")
ARM_ROOT = "arm_turret"
WRIST_BODY = "arm_wrist"
CARRIAGE_BODY = "arm_carriage"
PLAN_BIT = 1 << 7
DEFAULT_MARGINS: Dict[str, float] = {"robot": 0.005, "carriage": 0.0, "keepout": 0.010, "wood": 0.010,
                                     "fruit": 0.010, "fixture": 0.010, "ground": 0.020}
# contype-0 robot geoms that are solid on hardware (WP1 mast, crown head, desap tray / cradle pads)
KEEPOUT_PREFIXES = ("mast_", "crown_head", "tray_", "cradle_")
FIXTURE_PREFIXES = ("crate", "dock", "prop", "rope", "lateral", "bamboo")
FRUIT_PREFIXES = ("pool_", "crop_item_")


def _subtree(model: mujoco.MjModel, root: int) -> set:
    ids = {root}
    for b in range(root + 1, model.nbody):
        if model.body_parentid[b] in ids:
            ids.add(b)
    return ids


def _name(model, obj, i) -> str:
    return mujoco.mj_id2name(model, obj, i) or ""


class CollisionScene:
    """Collision checks for arm configurations q5 = (gantry, yaw, shoulder, elbow, wrist) in a composed world."""

    def __init__(self, model: mujoco.MjModel, margins: Optional[Dict[str, float]] = None,
                 keepout_prefixes: Sequence[str] = KEEPOUT_PREFIXES):
        self.src = model
        self.m = copy.deepcopy(model)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.margins = dict(DEFAULT_MARGINS)
        self.margins.update(margins or {})
        self.keepout_prefixes = tuple(keepout_prefixes)
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        if min(jids) < 0:
            raise ValueError(f"model is missing one of the arm joints {ARM_JOINTS}")
        self.qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self.limits = np.array([m.jnt_range[j] for j in jids], dtype=np.float64)
        self.wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY)
        self.carriage_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CARRIAGE_BODY)
        self.robot_bodies = _subtree(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk"))
        self.arm_bodies = _subtree(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ARM_ROOT))
        self._attached: Dict[int, Tuple[int, np.ndarray, np.ndarray]] = {}   # body -> (qposadr, rel pos, rel quat)
        self._carried: set = set()                 # attached roots and every body below them (geoms ride along)
        self._ignore_geoms: set = set()
        self._ignore_classes: set = set()
        self.n_checks = 0
        self._base_contype = np.array(model.geom_contype)
        self._base_conaff = np.array(model.geom_conaffinity)
        self.geom_class: List[str] = [self._classify(g) for g in range(m.ngeom)]
        self._apply_masks()

    # ------------------------------------------------------------------ classification and masks
    def _classify(self, g: int) -> str:
        m = self.m
        b = int(m.geom_bodyid[g])
        bname, gname = _name(m, mujoco.mjtObj.mjOBJ_BODY, b), _name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        if b in self.arm_bodies:
            return "arm"
        if b == self.carriage_id:
            return "carriage"
        if b in self.robot_bodies:
            if gname.startswith(self.keepout_prefixes) or bname.startswith(self.keepout_prefixes):
                return "keepout"
            return "robot"
        if bname.startswith(FRUIT_PREFIXES):
            return "fruit"
        if bname.startswith("tree_") or gname.startswith("tree_"):
            return "wood"
        if bname.startswith(FIXTURE_PREFIXES) or gname.startswith(FIXTURE_PREFIXES):
            return "fixture"
        if b == 0:
            return "ground"
        return "fixture"

    def _collidable_with_arm(self, g: int, arm_ct: int, arm_ca: int) -> bool:
        ct, ca = int(self._base_contype[g]), int(self._base_conaff[g])
        return bool((ct & arm_ca) or (arm_ct & ca))

    def _apply_masks(self) -> None:
        m = self.m
        self._carried = set()
        for root in self._attached.keys():
            self._carried |= _subtree(m, int(root))
        moving = [g for g in range(m.ngeom) if self.geom_class[g] == "arm" or m.geom_bodyid[g] in self._carried]
        moving = [g for g in moving if self._base_contype[g] or self._base_conaff[g] or m.geom_bodyid[g] in self._carried]
        arm_ct = int(np.bitwise_or.reduce(self._base_contype[moving])) if moving else 1
        arm_ca = int(np.bitwise_or.reduce(self._base_conaff[moving])) if moving else 1
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 0
        m.geom_margin[:] = 0.0
        m.geom_gap[:] = 0.0
        self.moving_geoms = sorted(moving)
        self.obstacle_geoms: List[int] = []
        for g in self.moving_geoms:
            m.geom_contype[g] = PLAN_BIT
        for g in range(m.ngeom):
            cls = self.geom_class[g]
            if g in self._ignore_geoms or cls in self._ignore_classes or m.geom_bodyid[g] in self._carried:
                continue
            if cls == "arm":
                continue
            if cls == "keepout":
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                    continue
            elif not self._collidable_with_arm(g, arm_ct, arm_ca):
                continue
            m.geom_conaffinity[g] = PLAN_BIT
            m.geom_margin[g] = float(self.margins.get(cls, 0.0))
            self.obstacle_geoms.append(g)
        self._is_moving = np.zeros(m.ngeom, dtype=bool)
        self._is_moving[self.moving_geoms] = True
        self._is_obstacle = np.zeros(m.ngeom, dtype=bool)
        self._is_obstacle[self.obstacle_geoms] = True
        # explicit pairs bypass the masks: margin of the obstacle class for moving x obstacle pairs, 0 otherwise
        # (those contacts are dropped by _violations)
        if m.npair:
            g1, g2 = m.pair_geom1, m.pair_geom2
            for p in range(m.npair):
                a, b = int(g1[p]), int(g2[p])
                o = b if self._is_moving[a] else a
                live = (self._is_moving[a] and self._is_obstacle[b]) or (self._is_moving[b] and self._is_obstacle[a])
                m.pair_margin[p] = float(self.margins.get(self.geom_class[o], 0.0)) if live else 0.0
                m.pair_gap[p] = 0.0
        # MuJoCo filters body pairs first with body_contype / body_conaffinity (OR over the body's geoms)
        m.body_contype[:] = 0
        m.body_conaffinity[:] = 0
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            m.body_contype[b] |= m.geom_contype[g]
            m.body_conaffinity[b] |= m.geom_conaffinity[g]
        # margins changed: the broad phase uses body/geom bounding volumes inflated by the margin

    def obstacle_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for g in self.obstacle_geoms:
            out[self.geom_class[g]] = out.get(self.geom_class[g], 0) + 1
        return out

    def set_ignore(self, geoms: Iterable = (), classes: Iterable[str] = ()) -> None:
        """Drop geoms (ids or names, or body names: all their geoms) and whole classes from the obstacle set."""
        ids = set()
        for g in geoms:
            if isinstance(g, str):
                gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, g)
                if gid >= 0:
                    ids.add(gid)
                    continue
                bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, g)
                if bid < 0:
                    raise KeyError(f"no geom or body named {g!r}")
                ids.update(i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] == bid)
            else:
                ids.add(int(g))
        self._ignore_geoms, self._ignore_classes = ids, set(classes)
        self._apply_masks()

    # ------------------------------------------------------------------ live state
    def sync(self, model: Optional[mujoco.MjModel] = None, data: Optional[mujoco.MjData] = None,
             qpos: Optional[np.ndarray] = None) -> None:
        """Copy the mutable fields of the live model (pool contype/size/pos, mast body_pos) and the live state."""
        src = model if model is not None else self.src
        m = self.m
        for f in ("geom_size", "geom_pos", "geom_quat", "geom_rbound", "geom_aabb", "body_pos", "body_quat"):
            getattr(m, f)[:] = getattr(src, f)
        if not (np.array_equal(self._base_contype, src.geom_contype)
                and np.array_equal(self._base_conaff, src.geom_conaffinity)):
            self._base_contype = np.array(src.geom_contype)
            self._base_conaff = np.array(src.geom_conaffinity)
            self._apply_masks()
        if data is not None:
            self.d.qpos[:] = data.qpos
            self.d.mocap_pos[:] = data.mocap_pos
            self.d.mocap_quat[:] = data.mocap_quat
        if qpos is not None:
            self.d.qpos[:] = qpos
        mujoco.mj_kinematics(m, self.d)
        for b in list(self._attached):
            self._attached[b] = self._relative(b)

    def _relative(self, body: int):
        d = self.d
        jnt = int(self.m.body_jntadr[body])
        if jnt < 0 or self.m.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"attached body {_name(self.m, mujoco.mjtObj.mjOBJ_BODY, body)!r} has no free joint")
        wr, wq = d.xpos[self.wrist_id], d.xquat[self.wrist_id]
        inv = np.zeros(4)
        mujoco.mju_negQuat(inv, wq)
        rel_p = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_p, d.xpos[body] - wr, inv)
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, inv, d.xquat[body])
        return int(self.m.jnt_qposadr[jnt]), rel_p, rel_q

    def attach(self, body: str) -> None:
        """Carry a free body (held fruit or tool) rigidly with the wrist; its geoms and those of every body below it
        (tool head, blades, jaws on their own joints, kept at their current joint values) join the moving set."""
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            raise KeyError(body)
        mujoco.mj_kinematics(self.m, self.d)
        self._attached[bid] = self._relative(bid)
        self._apply_masks()

    def detach_all(self) -> None:
        self._attached.clear()
        self._apply_masks()

    # ------------------------------------------------------------------ checks
    def _set(self, q5: Sequence[float]) -> None:
        d = self.d
        d.qpos[self.qadr] = q5
        if self._attached:
            mujoco.mj_kinematics(self.m, d)
            wp, wq = d.xpos[self.wrist_id], d.xquat[self.wrist_id]
            for adr, rp, rq in self._attached.values():
                p = np.zeros(3)
                mujoco.mju_rotVecQuat(p, rp, wq)
                q = np.zeros(4)
                mujoco.mju_mulQuat(q, wq, rq)
                d.qpos[adr:adr + 3] = wp + p
                d.qpos[adr + 3:adr + 7] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_collision(self.m, d)

    def in_limits(self, q5: Sequence[float]) -> bool:
        q = np.asarray(q5)
        return bool(np.all(q >= self.limits[:, 0] - 1e-9) and np.all(q <= self.limits[:, 1] + 1e-9))

    def is_valid(self, q5: Sequence[float]) -> bool:
        """True if q5 is inside the limits and no moving geom is within its obstacle's clearance."""
        self.n_checks += 1
        if not self.in_limits(q5):
            return False
        self._set(q5)
        return self.d.ncon == 0 or not self._violations()

    def _violations(self) -> List[Tuple[int, int, float]]:
        """(moving geom, obstacle geom, dist) of every contact in d that joins the moving set to an obstacle."""
        out = []
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self._is_moving[g1] and self._is_obstacle[g2]:
                out.append((g1, g2, float(c.dist)))
            elif self._is_moving[g2] and self._is_obstacle[g1]:
                out.append((g2, g1, float(c.dist)))
        return out

    def contacts(self, q5: Sequence[float]) -> List[dict]:
        """Every clearance violation at q5: moving geom, obstacle geom, its class and the distance (m)."""
        self._set(q5)
        m, out = self.m, []
        for a, o, dist in self._violations():
            out.append({"arm": _name(m, mujoco.mjtObj.mjOBJ_GEOM, a) or _name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[a]),
                        "obstacle": _name(m, mujoco.mjtObj.mjOBJ_GEOM, o) or _name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[o]),
                        "class": self.geom_class[o], "dist": dist})
        return out

    def clearance(self, q5: Sequence[float], cap: float = 0.10) -> Dict[str, float]:
        """Smallest distance (m, capped at ``cap``) from the moving set to each obstacle class at q5."""
        m = self.m
        saved, saved_pair = m.geom_margin.copy(), m.pair_margin.copy()
        m.geom_margin[self.obstacle_geoms] = cap
        m.pair_margin[:] = cap
        try:
            self._set(q5)
            out = {c: cap for c in set(self.geom_class[g] for g in self.obstacle_geoms)}
            for _, o, dist in self._violations():
                cls = self.geom_class[o]
                out[cls] = min(out[cls], dist)
            return out
        finally:
            m.geom_margin[:] = saved
            m.pair_margin[:] = saved_pair

    def motion_valid(self, qa: Sequence[float], qb: Sequence[float], step: float) -> bool:
        """Straight joint-space segment qa -> qb checked every ``step`` rad (max-norm) including both ends."""
        qa, qb = np.asarray(qa, dtype=np.float64), np.asarray(qb, dtype=np.float64)
        n = max(1, int(np.ceil(np.max(np.abs(qb - qa)) / step)))
        for k in range(n + 1):
            if not self.is_valid(qa + (qb - qa) * (k / n)):
                return False
        return True

    def tip(self, q5: Sequence[float], site: str = "gripper_tip") -> np.ndarray:
        self.d.qpos[self.qadr] = q5
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, site)].copy()
