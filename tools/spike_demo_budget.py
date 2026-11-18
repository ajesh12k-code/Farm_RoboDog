#!/usr/bin/env python3
"""WP11a CPU spike for the Atharv Nandi orchard MVP (ORCHARD-MVP-SPEC 3.7, MVP-SPEC 3.8). No ROS.

Builds the **orchard stub**: the live farm world (``farm_scene._build_spec``, flat terrain, 6 legacy
crops) plus the 17 ``orchard_demo`` trees of ORCHARD-MVP-SPEC 4.2 (static bodies, merged foliage
meshes, <= 70 render geoms per tree), 3 rack tools (free body + 2 jaw slides, welded), a 10-slot
fruit pool (5 connect + 5 weld on tree bodies), 10 mocap actors (4 people, 2 animals, 4 debris),
4 ring cameras, a crown camera and the mast LiDAR site. Then it measures, section by section:

    stub       model counts: nv, nq, neq, equality rows at rest, nmocap, render geoms per tree
    physics    ms per mj_step (stub vs robot-only farm), mj_setConst per payload event, LiDAR and
               stability hooks per call
    render     per capture RGB + depth (one pass, two readbacks) + segmentation pass, ring and
               crown at 320 x 240, mjvScene camera-distance post-filter (> 4 m dropped) vs none,
               scene geom counts, model-unchanged check
    rays       mast LiDAR 1,620 rays with the safety mask {0, 1} and the coverage mask {0, 1, 3}
               (mesh foliage included); TREE 360 coverage rays (72 points x 3 bands per station)
    viewer     passive MuJoCo viewer at 10 Hz sync (Xvfb + glfw) vs none: process CPU cores
    ack        two processes (sim paced at RTF 0.9 + perception renderer): perception ack latency
               per TREE 360 capture at 8 OT-1 stations, in sim seconds against the 0.5 s window
    livestack  the current ROS stack (launch/launch_nandi.sh, headless) for a window: CPU per process
    verdict    go / no-go from the JSON outputs of render + ack (pure; also used by the tests)

Every run prints one JSON document with a run id and a device id (run-identity rule). Run each section in
the capped runner (2 CPU / 2 GB / 180 s), e.g.

    nandi-sim <worktree> python3 tools/spike_demo_budget.py render --out runs/wp11a/render.json

GL sections start their own Xvfb display when none is reachable (MUJOCO_GL=glfw). LP_NUM_THREADS
defaults to 1 (ORCHARD-MVP-SPEC 3.7 rule). CI-runner numbers are pending: CI runs are not
authorised yet, so every figure this prints on a Mac runner is a local measurement.
"""
import argparse
import datetime as _dt
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

# Env defaults (LP_NUM_THREADS=1, NANDI_TERRAIN=flat) are applied in main() only, so importing this
# module (the tests do) does not change the environment of the importing process.
ENV_DEFAULTS = {"LP_NUM_THREADS": "1", "NANDI_TERRAIN": "flat"}

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from atharv_nandi.core import farm_scene  # noqa: E402
from atharv_nandi.launch_plan import proc_label  # noqa: E402

G = mujoco.mjtGeom

# ------------------------------------------------------------------ spec constants (ORCHARD-MVP-SPEC)
# Geom groups (2.5 "Ray groups"): 0 wood and ground, 1 fruit, 2 robot, 3 foliage, 4 decals, 5 query-only.
GRP_WOOD, GRP_FRUIT, GRP_ROBOT, GRP_FOLIAGE, GRP_DECAL, GRP_QUERY = 0, 1, 2, 3, 4, 5
SAFETY_MASK = (1, 1, 0, 0, 0, 0)       # safety rays: {0, 1} plus actors (actors are in group 0)
COVERAGE_MASK = (1, 1, 0, 1, 0, 0)     # coverage rays: {0, 1, 3}
RENDER_GROUPS = (1, 1, 1, 1, 1, 0)     # cameras: 0-4 visible, 5 query-only hidden
CULL_DIST_M = 4.0                      # 3.7: scene geoms farther than 4 m from the camera are dropped
RENDER_GEOMS_PER_TREE_MAX = 70         # 3.7 rule
SCENE_GEOMS_AFTER_CULL_MAX = 700       # 3.7 rule
CAPTURE_W, CAPTURE_H = 320, 240        # ring and crown captures
PASSES_PER_CAPTURE_BUDGET = 3          # RGB+depth (1) + seg (1) = 2, budgeted as 3
CAPTURES_PER_STATION = 3               # low (ring left), mid (crown 1.70 m), high (crown 2.70 m)
STATIONS = 8                           # TREE 360 stations per orbit
ORBIT_SIM_S = 165.0                    # O-06 / scene 2: orbit <= 165 s sim
ACK_WINDOW_SIM_S = 0.5                 # 3.7: budget 0.5 s sim per capture window
RTF_TARGET = 0.9
RENDER_BUDGET_FRAC = 0.35              # MVP-SPEC 3.8: render_budget_frac 0.20 -> 0.35
STANDOFF_M = 0.9                       # lens-to-canopy stand-off d (r1 = canopy_r + d)
LENS_Z_M = 0.98                        # ring lens / mast LiDAR height above ground
CROWN_MID = (1.70, -10.0)              # crown lens height (m) and gimbal pitch (deg, negative = down)
CROWN_HIGH = (2.70, -20.0)
RING_PITCH_DOWN_DEG = 25.0
RING_FOVY_DEG = 83.6                   # 100 deg HFOV at 4:3
CROWN_FOVY_DEG = 60.0                  # assumption (crown camera not built yet; WP1)
MAST_X = -0.30                         # mast_base x on the trunk (2.3)
LIDAR_BEAMS_DEG = (0.0, -15.0, -30.0)  # MVP-SPEC 2.4: 360 rays x 3 beams
LIDAR_RAYS_PER_BEAM = 360
LOW_PLANE_Z_M = 0.18                   # low safety planes
LOW_PLANE_RAYS = 270                   # 2 planes x 270 = 540; total 1,620 rays (MVP-SPEC 3.8)
LIDAR_RANGE_M = 8.0
COVERAGE_POINTS_PER_BAND = 72          # O-01
CI_PENDING = "pending: CI-runner runs not authorised yet (2026-11-17)"

# orchard_demo layout (ORCHARD-MVP-SPEC 4.2 with OQ-02 B: NHB 5 x 2.5 m mango).
ROW_Y = {"R1": 2.5, "R2": 6.5, "R3": 10.0, "R4": 13.5}
IN_ROW_M = {"R1": 2.5, "R2": 1.5, "R3": 1.8, "R4": 1.8}
ROW_SPECIES = {"R1": "mango", "R2": "guava", "R3": "papaya", "R4": "banana"}
ROW_HEIGHTS = {"R1": (1.8, 2.2, 2.5, 2.8), "R2": (1.8, 2.0, 2.0, 2.2),
               "R3": (2.2, 2.4, 2.6, 2.6), "R4": (2.4, 2.6, 2.6, 2.8)}
ROW_X0 = 2.0
OT1 = {"id": "OT-1", "species": "mango", "x": 5.0, "y": -3.5, "H": 2.2, "canopy_r": 1.0}

# What the stub appends (ORCHARD-MVP-SPEC 2.6): nv = 31 robot + 36 legacy crops + 24 tools + 60 pool.
N_TOOLS, N_POOL, N_POOL_CONNECT = 3, 10, 5
ACTORS = ([("person", (6.5, -1.0)), ("person", (8.0, 4.5)), ("person", (3.0, 8.5)), ("person", (-2.0, 3.0))]
          + [("animal", (9.5, 11.5)), ("animal", (-3.0, -4.5))]
          + [("debris", (2.2, 2.0)), ("debris", (4.6, 2.0)), ("debris", (2.0, 6.0)), ("debris", (3.4, 6.0))])
NV_TARGET, EQ_ROWS_MAX, NMOCAP_TARGET, N_TREES = 151, 63, 10, 17


# ------------------------------------------------------------------ small geometry helpers
def _quat_from_mat(R: np.ndarray) -> List[float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(-1))
    return q.tolist()


def _rot_z(a: float) -> List[float]:
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def cam_quat(yaw_deg: float, pitch_deg: float) -> List[float]:
    """Camera quaternion (parent frame) looking along yaw (from parent +x, CCW) with pitch (deg, + up).
    MuJoCo cameras look along local -z with +y up."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    up = np.cross(right, fwd)
    R = np.stack([right, up, -fwd], axis=1)
    return _quat_from_mat(R)


def _quat_from_to(v) -> List[float]:
    v = np.asarray(v, float) / np.linalg.norm(v)
    z = np.array([0, 0, 1.0])
    c = float(np.dot(z, v))
    if c > 0.9999:
        return [1.0, 0.0, 0.0, 0.0]
    ax = np.cross(z, v)
    s = np.linalg.norm(ax)
    ax /= s
    ang = math.atan2(s, c)
    return [math.cos(ang / 2), *(ax * math.sin(ang / 2)).tolist()]


# ------------------------------------------------------------------ layout
def orchard_demo_layout() -> List[dict]:
    """The 17 orchard_demo trees: OT-1 plus 4 trees in each of R1-R4 (positions are candidates)."""
    trees = [dict(OT1)]
    for row in ("R1", "R2", "R3", "R4"):
        for k, h in enumerate(ROW_HEIGHTS[row]):
            sp = ROW_SPECIES[row]
            cr = 0.45 * h if sp in ("mango", "guava") else (0.75 if sp == "papaya" else 1.0)
            if sp == "guava":
                cr = min(cr, 0.7)   # hedge, in-row 1.5 m
            trees.append({"id": f"{row}-{k + 1}", "species": sp, "x": ROW_X0 + k * IN_ROW_M[row],
                          "y": ROW_Y[row], "H": float(h), "canopy_r": round(cr, 3)})
    return trees


def ot1_stations(n: int = STATIONS, standoff: float = STANDOFF_M) -> List[Tuple[float, float, float]]:
    """(x, y, heading) of the TREE 360 stations on OT-1: CCW, tree on the robot's left."""
    r1 = OT1["canopy_r"] + standoff
    out = []
    for k in range(n):
        th = 2 * math.pi * k / n
        out.append((OT1["x"] + r1 * math.cos(th), OT1["y"] + r1 * math.sin(th), th + math.pi / 2))
    return out


# ------------------------------------------------------------------ stub builder
class _Tree:
    """Procedural tree on one static body (adapted from tools/probes/treesim_proto.py, merged foliage)."""

    def __init__(self, spec: mujoco.MjSpec, t: dict, idx: int):
        self.spec, self.t, self.idx = spec, t, idx
        self.rng = np.random.default_rng(1000 + idx)
        self.body = spec.worldbody.add_body(name=f"tree_{idx}", pos=[t["x"], t["y"], 0.0])
        self.verts: List[float] = []
        self.faces: List[int] = []
        self.render_geoms = 0
        self.fruit_anchor_w: List[np.ndarray] = []

    def geom(self, group: int, **kw):
        kw.setdefault("contype", 0)
        kw.setdefault("conaffinity", 0)
        if "mass" not in kw and kw["contype"] == 0:
            kw["mass"] = 0
        self.body.add_geom(group=group, **kw)
        if group <= GRP_DECAL:
            self.render_geoms += 1

    def wood(self, p0, p1, r, rgba=(0.45, 0.3, 0.18, 1)):
        self.geom(GRP_WOOD, type=G.mjGEOM_CAPSULE, fromto=[*p0, *p1], size=[r, 0, 0], rgba=list(rgba),
                  contype=1, conaffinity=0)

    def blob(self, c, hs, quat=(1, 0, 0, 0)):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(quat, float))
        R = R.reshape(3, 3)
        V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float) * hs
        F = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
        base = len(self.verts) // 3
        self.verts.extend((V @ R.T + np.asarray(c, float)).ravel().tolist())
        self.faces.extend((np.asarray(F) + base).ravel().tolist())

    def flush_foliage(self, rgba=(0.15, 0.45, 0.15, 1)):
        if not self.verts:
            return
        name = f"canopy_{self.idx}"
        self.spec.add_mesh(name=name, uservert=self.verts, userface=self.faces,
                           inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        self.geom(GRP_FOLIAGE, type=G.mjGEOM_MESH, meshname=name, rgba=list(rgba))
        self.verts, self.faces = [], []

    def basin(self, r=0.9, segs=6, h=0.10):
        for k in range(segs):
            a0, a1 = 2 * math.pi * k / segs, 2 * math.pi * (k + 1) / segs
            p0 = [r * math.cos(a0), r * math.sin(a0), 0.0]
            p1 = [r * math.cos(a1), r * math.sin(a1), 0.0]
            self.geom(GRP_WOOD, type=G.mjGEOM_CAPSULE, fromto=[*p0, *p1], size=[h, 0, 0],
                      rgba=[0.30, 0.22, 0.15, 1], contype=1, conaffinity=1)

    def leaf_targets(self, n, radius, z_lo=0.5, z_hi=1.4):
        for k in range(n):
            th = 2 * math.pi * k / n + self.rng.uniform(-0.2, 0.2)
            p = [radius * math.cos(th), radius * math.sin(th), self.rng.uniform(z_lo, z_hi)]
            self.geom(GRP_FOLIAGE, type=G.mjGEOM_BOX, pos=p, size=[0.06, 0.025, 0.002], quat=_rot_z(th),
                      rgba=[0.2, 0.5, 0.2, 1])
            self.body.add_site(name=f"leaf_{self.idx}_{k}", pos=p, size=[0.005, 0, 0], group=GRP_QUERY)

    def decals(self, n, radius, z_lo=0.4, z_hi=1.6):
        for k in range(n):
            th = self.rng.uniform(0, 2 * math.pi)
            p = [radius * math.cos(th), radius * math.sin(th), self.rng.uniform(z_lo, z_hi)]
            self.geom(GRP_DECAL, type=G.mjGEOM_CYLINDER, pos=p, size=[0.03, 0.001, 0],
                      quat=_quat_from_to([math.cos(th), math.sin(th), 0]), rgba=[0.25, 0.2, 0.1, 1])

    def static_fruit(self, n, crown_r, size, rgba, z_lo=0.5, z_hi=2.0):
        for k in range(n):
            th = self.rng.uniform(0, 2 * math.pi)
            r = crown_r * self.rng.uniform(0.75, 1.0)
            z = self.rng.uniform(z_lo, min(z_hi, self.t["H"] - 0.2))
            anchor = np.array([r * math.cos(th), r * math.sin(th), z])
            self.fruit_anchor_w.append(anchor + np.array([self.t["x"], self.t["y"], 0.0]))
            self.geom(GRP_FRUIT, type=G.mjGEOM_ELLIPSOID, pos=(anchor - [0, 0, size[2] + 0.02]).tolist(),
                      size=list(size), rgba=list(rgba))

    def envelope(self, c, size):
        self.geom(GRP_QUERY, type=G.mjGEOM_ELLIPSOID, pos=list(c), size=list(size), rgba=[0, 1, 0, 0.1])

    # species ---------------------------------------------------------------
    def canopy_tree(self, fruit_n, fruit_size, fruit_rgba):
        t, rng, H = self.t, self.rng, self.t["H"]
        trunk_h = 0.35 * H
        self.wood([0, 0, 0], [0, 0, trunk_h], 0.05 + 0.01 * H)
        cr = t["canopy_r"]
        n_sc = int(rng.integers(3, 5))
        for i in range(n_sc):
            a = 2 * math.pi * i / n_sc + rng.uniform(-0.3, 0.3)
            el = rng.uniform(0.6, 0.9)
            L = 0.5 * H
            p1 = np.array([math.cos(a) * L * math.cos(el) * 0.8, math.sin(a) * L * math.cos(el) * 0.8,
                           trunk_h + L * math.sin(el)])
            self.wood([0, 0, trunk_h], p1, 0.03)
            for _ in range(2):
                a2 = a + rng.uniform(-0.7, 0.7)
                p2 = p1 + 0.25 * H * np.array([math.cos(a2) * 0.7, math.sin(a2) * 0.7, rng.uniform(0.1, 0.5)])
                self.wood(p1, p2, 0.015)
        cz = trunk_h + 0.4 * H
        for _ in range(40):
            th, ph = rng.uniform(0, 2 * math.pi), rng.uniform(-0.3, 1.3)
            rr = cr * rng.uniform(0.7, 1.0)
            p = [rr * math.cos(th) * math.cos(ph), rr * math.sin(th) * math.cos(ph), cz + 0.6 * cr * math.sin(ph)]
            self.blob(p, [0.18, 0.18, 0.06], _rot_z(th))
        self.flush_foliage()
        self.leaf_targets(12, cr * 0.95)
        self.decals(4, cr * 0.97)
        self.static_fruit(fruit_n, cr, fruit_size, fruit_rgba)
        self.envelope([0, 0, cz], [cr, cr, 0.6 * cr + 0.15])
        self.basin()

    def mango(self):
        self.canopy_tree(20, [0.04, 0.035, 0.06], (0.55, 0.65, 0.2, 1))

    def guava(self):
        self.canopy_tree(20, [0.035, 0.035, 0.04], (0.75, 0.85, 0.35, 1))

    def papaya(self):
        rng, H = self.rng, self.t["H"]
        self.wood([0, 0, 0], [0, 0, H], 0.08, rgba=(0.55, 0.55, 0.45, 1))
        for k in range(14):
            a = 2 * math.pi * k / 14 * 2.4
            d = np.array([math.cos(a), math.sin(a), rng.uniform(0.2, 0.8)])
            d /= np.linalg.norm(d)
            p0 = np.array([0, 0, H - 0.05])
            p1 = p0 + 0.6 * d
            self.geom(GRP_FOLIAGE, type=G.mjGEOM_CAPSULE, fromto=[*p0, *p1], size=[0.008, 0, 0],
                      rgba=[0.5, 0.7, 0.3, 1])
            self.blob(p1 + 0.15 * d, [0.25, 0.25, 0.01], _rot_z(a))
        self.flush_foliage()
        for k in range(12):
            a = rng.uniform(0, 2 * math.pi)
            z = H - 0.15 - 0.06 * k
            c = [0.17 * math.cos(a), 0.17 * math.sin(a), z - 0.05]
            self.fruit_anchor_w.append(np.array([self.t["x"] + 0.08 * math.cos(a), self.t["y"] + 0.08 * math.sin(a), z]))
            self.geom(GRP_FRUIT, type=G.mjGEOM_ELLIPSOID, pos=c, size=[0.07, 0.07, 0.11],
                      rgba=[0.95, 0.6, 0.1, 1] if k < 3 else [0.3, 0.6, 0.2, 1])
        self.leaf_targets(8, 0.6, H - 0.6, H)
        self.decals(2, 0.09, 0.4, H - 0.4)
        self.envelope([0, 0, H], [0.75, 0.75, 0.4])
        self.basin(r=0.7)

    def banana(self):
        rng, H = self.rng, self.t["H"]
        self.geom(GRP_WOOD, type=G.mjGEOM_CYLINDER, pos=[0, 0, H / 2], size=[0.11, H / 2, 0],
                  rgba=[0.45, 0.6, 0.3, 1], contype=1, conaffinity=0)
        for k in range(9):
            a = 2 * math.pi * k / 9 * 1.6 + rng.uniform(-0.2, 0.2)
            d = np.array([math.cos(a), math.sin(a), rng.uniform(0.0, 0.9)])
            d /= np.linalg.norm(d)
            self.geom(GRP_FOLIAGE, type=G.mjGEOM_BOX, pos=list(np.array([0, 0, H]) + 0.8 * d),
                      size=[0.22, 0.004, 0.75], quat=_quat_from_to(d),
                      rgba=[0.2, 0.55, 0.2, 1] if k > 1 else [0.55, 0.45, 0.2, 1])
        self.geom(GRP_WOOD, type=G.mjGEOM_CAPSULE, fromto=[0.12, 0, H - 0.1, 0.25, 0, H - 0.9],
                  size=[0.025, 0, 0], rgba=[0.4, 0.5, 0.2, 1])
        for h in range(6):
            self.geom(GRP_FRUIT, type=G.mjGEOM_ELLIPSOID, pos=[0.2, 0, H - 0.25 - 0.1 * h],
                      size=[0.14, 0.14, 0.05], rgba=[0.4, 0.65, 0.2, 1], contype=1, conaffinity=0, mass=1)
        for _ in range(2):
            a = rng.uniform(0, 2 * math.pi)
            x, y = 0.3 * math.cos(a), 0.3 * math.sin(a)
            self.geom(GRP_WOOD, type=G.mjGEOM_CAPSULE, fromto=[x, y, 0, x, y, rng.uniform(0.3, 0.8)],
                      size=[0.04, 0, 0], rgba=[0.4, 0.6, 0.3, 1], contype=1, conaffinity=0)
        self.leaf_targets(8, 0.9, H - 0.4, H + 0.4)
        self.decals(2, 0.12, 0.4, H - 0.4)
        self.envelope([0, 0, H], [1.0, 1.0, 0.6])
        self.basin(r=0.8)
        for k in range(3):
            self.fruit_anchor_w.append(np.array([self.t["x"] + 0.2, self.t["y"], H - 0.3 - 0.1 * k]))


def _robot_body_ids(m: mujoco.MjModel) -> set:
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    return {b for b in range(m.nbody) if b == trunk or _is_desc(m, b, trunk)}


def _is_desc(m, b, root) -> bool:
    while b > 0:
        if b == root:
            return True
        b = int(m.body_parentid[b])
    return False


def build_stub(trees: Optional[List[dict]] = None, pool: bool = True, tools: bool = True,
               actors: bool = True,
               terrain: Optional[str] = None) -> Tuple[mujoco.MjModel, dict]:
    """Compile the orchard stub. Returns (model, info) with per-tree render-geom counts.

    terrain: base world terrain (default: NANDI_TERRAIN if set, else flat). Set only for the build and
    restored afterwards, so the caller's environment is unchanged.
    """
    trees = orchard_demo_layout() if trees is None else trees
    terrain = terrain or os.environ.get("NANDI_TERRAIN") or "flat"
    prev = os.environ.get("NANDI_TERRAIN")
    os.environ["NANDI_TERRAIN"] = terrain
    try:
        spec = farm_scene._build_spec()  # noqa: SLF001 - shared builder, read-only use
    finally:
        if prev is None:
            os.environ.pop("NANDI_TERRAIN", None)
        else:
            os.environ["NANDI_TERRAIN"] = prev
    # Robot geoms -> group 2 (visible) or 5 (mass-only, never rendered or ray-hit); stub convention,
    # WP1 owns the real assignment.
    def _regroup(body):
        for g in body.geoms:
            g.group = GRP_QUERY if g.group >= 3 else GRP_ROBOT
        for c in body.bodies:
            _regroup(c)
    trunk = spec.body("trunk")
    _regroup(trunk)
    key = spec.keys[0]
    key_q = np.asarray(key.qpos, dtype=np.float64)
    tz = float(key_q[2])
    # WP1 ships ring_0..ring_3, crown_cam (on the crown head) and mast_lidar in the robot MJCF. The stub keeps its
    # own probe cameras and LiDAR site on the trunk (the poses and camera count its recorded numbers used), so
    # drop WP1's first.
    for el in [spec.camera(f"ring_{i}") for i in range(4)] + [spec.camera("crown_cam"), spec.site("mast_lidar")]:
        if el is not None:
            spec.delete(el)
    # cameras and LiDAR on the trunk (lens height above ground)
    for yaw in (0, 90, 180, 270):
        a = math.radians(yaw)
        trunk.add_camera(name=f"ring_{yaw}", pos=[0.05 * math.cos(a), 0.05 * math.sin(a), LENS_Z_M - tz],
                         quat=cam_quat(yaw, -RING_PITCH_DOWN_DEG), fovy=RING_FOVY_DEG)
    trunk.add_camera(name="crown_cam", pos=[MAST_X, 0.0, CROWN_MID[0] - tz],
                     quat=cam_quat(90, CROWN_MID[1]), fovy=CROWN_FOVY_DEG)
    trunk.add_site(name="mast_lidar", pos=[MAST_X, 0.0, LENS_Z_M - tz], size=[0.01, 0, 0], group=GRP_QUERY)
    for sx in (0.36, -0.36):
        trunk.add_site(name=f"low_plane_{'f' if sx > 0 else 'r'}", pos=[sx, 0.0, LOW_PLANE_Z_M - tz],
                       size=[0.01, 0, 0], group=GRP_QUERY)
    spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), 640)
    spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), 480)

    per_tree, anchors = [], []
    for i, t in enumerate(trees):
        tr = _Tree(spec, t, i)
        getattr(tr, t["species"])()
        per_tree.append({"id": t["id"], "species": t["species"], "render_geoms": tr.render_geoms})
        anchors.append(tr.fruit_anchor_w)
    # drip laterals (one per row) + OT-1 ring lateral (8 chords)
    for row, y in ROW_Y.items():
        x0, x1 = ROW_X0 - 2.0, ROW_X0 + 3 * IN_ROW_M[row] + 2.0
        spec.worldbody.add_geom(name=f"lateral_{row}", type=G.mjGEOM_CAPSULE, size=[0.008, 0, 0],
                                fromto=[x0, y - 0.4, 0.008, x1, y - 0.4, 0.008], rgba=[0.05, 0.05, 0.05, 1],
                                group=GRP_WOOD, contype=0, conaffinity=0)
    for k in range(8):
        a0, a1 = 2 * math.pi * k / 8, 2 * math.pi * (k + 1) / 8
        spec.worldbody.add_geom(name=f"lateral_ot1_{k}", type=G.mjGEOM_CAPSULE, size=[0.008, 0, 0],
                                fromto=[OT1["x"] + 0.8 * math.cos(a0), OT1["y"] + 0.8 * math.sin(a0), 0.008,
                                        OT1["x"] + 0.8 * math.cos(a1), OT1["y"] + 0.8 * math.sin(a1), 0.008],
                                rgba=[0.05, 0.05, 0.05, 1], group=GRP_WOOD, contype=0, conaffinity=0)

    extra_q: List[float] = []
    if tools:
        for k in range(N_TOOLS):
            pos = [0.05 - 0.12 * k, -0.17, tz + 0.06]
            b = spec.worldbody.add_body(name=f"tool_{k}", pos=pos)
            b.add_freejoint(name=f"tool_{k}_fj")
            b.add_geom(type=G.mjGEOM_BOX, size=[0.04, 0.02, 0.05], mass=0.2, rgba=[0.2, 0.2, 0.7, 1],
                       contype=0, conaffinity=0, group=GRP_ROBOT)
            for s in (-1, 1):
                jb = b.add_body(name=f"tool_{k}_jaw_{'l' if s < 0 else 'r'}", pos=[0, 0.01 * s, -0.06])
                jb.add_joint(name=f"tool_{k}_jaw_{'l' if s < 0 else 'r'}_j", type=mujoco.mjtJoint.mjJNT_SLIDE,
                             axis=[0, s, 0], range=[0, 0.02], stiffness=50, damping=1, limited=True)
                jb.add_geom(type=G.mjGEOM_BOX, size=[0.005, 0.01, 0.02], mass=0.05, contype=0, conaffinity=0,
                            group=GRP_ROBOT, rgba=[0.3, 0.3, 0.3, 1])
            e = spec.add_equality(name=f"rack_weld_{k}", objtype=mujoco.mjtObj.mjOBJ_BODY,
                                  name1=f"tool_{k}", name2="trunk")
            e.type = mujoco.mjtEq.mjEQ_WELD
            # anchor at the tool origin and relpose all zero (= taken from qpos0): MjSpec's default data
            # puts the anchor at (0, 1, 0), 1 m to the side, and the tool then swings on a soft pendulum.
            e.data[0:10] = 0.0
            e.data[10] = 1.0
            e.solref = [0.006, 1]     # as stiff as the pool welds: a racked tool must not sag
            extra_q += pos + [1, 0, 0, 0, 0.0, 0.0]
    if pool:
        mango_idx = [i for i, t in enumerate(trees) if t["species"] == "mango"]
        guava_idx = [i for i, t in enumerate(trees) if t["species"] == "guava"] or mango_idx
        for k in range(N_POOL):
            connect = k < N_POOL_CONNECT
            ti = (mango_idx if connect else guava_idx)[k % len(mango_idx if connect else guava_idx)]
            t = trees[ti]
            anchor = anchors[ti][k % len(anchors[ti])] if anchors[ti] else np.array([t["x"] + 0.5, t["y"], 1.2])
            size = [0.04, 0.035, 0.06] if connect else [0.035, 0.035, 0.04]
            centre = anchor - np.array([0.0, 0.0, size[2] + 0.02])
            b = spec.worldbody.add_body(name=f"pool_{k}", pos=centre.tolist())
            b.add_freejoint(name=f"pool_{k}_fj")
            b.add_geom(type=G.mjGEOM_ELLIPSOID, size=size, mass=0.3 if connect else 0.15,
                       rgba=[0.95, 0.75, 0.1, 1], contype=1, conaffinity=0, group=GRP_FRUIT)
            e = spec.add_equality(name=f"pool_eq_{k}", objtype=mujoco.mjtObj.mjOBJ_BODY,
                                  name1=f"pool_{k}", name2=f"tree_{ti}")
            if connect:
                e.type = mujoco.mjtEq.mjEQ_CONNECT
                e.data[0:3] = (anchor - centre).tolist()
            else:
                e.type = mujoco.mjtEq.mjEQ_WELD
                e.data[0:3] = (anchor - np.array([t["x"], t["y"], 0.0])).tolist()
                e.data[10] = 1.0
            e.solref = [0.006, 1]
            extra_q += centre.tolist() + [1, 0, 0, 0]
    if actors:
        for k, (kind, (x, y)) in enumerate(ACTORS):
            b = spec.worldbody.add_body(name=f"actor_{k}_{kind}", pos=[x, y, 0.0], mocap=True)
            if kind == "person":
                b.add_geom(type=G.mjGEOM_CAPSULE, fromto=[0, 0, 0.2, 0, 0, 1.55], size=[0.18, 0, 0],
                           rgba=[0.8, 0.5, 0.4, 1], contype=0, conaffinity=0, group=GRP_WOOD)
            elif kind == "animal":
                b.add_geom(type=G.mjGEOM_BOX, pos=[0, 0, 0.45], size=[0.45, 0.15, 0.25],
                           rgba=[0.6, 0.5, 0.4, 1], contype=0, conaffinity=0, group=GRP_WOOD)
            else:
                b.add_geom(type=G.mjGEOM_CAPSULE, fromto=[0, 0, 0.02, 0.3, 0, 0.02], size=[0.01, 0, 0],
                           rgba=[0.4, 0.3, 0.2, 1], contype=0, conaffinity=0, group=GRP_WOOD)
    key.qpos = np.concatenate([key_q, np.asarray(extra_q, dtype=np.float64)])
    t0 = time.perf_counter()
    m = spec.compile()
    compile_s = time.perf_counter() - t0
    info = {"trees": per_tree, "compile_s": round(compile_s, 3), "n_trees": len(trees)}
    return m, info


def reset(m: mujoco.MjModel, d: mujoco.MjData):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)


def stub_counts(m: mujoco.MjModel, info: dict) -> dict:
    d = mujoco.MjData(m)
    reset(m, d)
    eq_rows = int(np.sum(d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY))
    robot = _robot_body_ids(m)
    rg = [t["render_geoms"] for t in info["trees"]]
    visible = int(np.sum(m.geom_group <= GRP_DECAL))
    return {"nv": m.nv, "nq": m.nq, "nu": m.nu, "nbody": m.nbody, "ngeom": m.ngeom, "neq": m.neq,
            "eq_rows_at_rest": eq_rows, "nmocap": m.nmocap, "n_trees": info["n_trees"],
            "render_geoms_per_tree_max": max(rg) if rg else 0, "render_geoms_trees_total": int(sum(rg)),
            "render_geoms_model_total": visible,
            "robot_render_geoms": int(sum(1 for g in range(m.ngeom)
                                          if m.geom_bodyid[g] in robot and m.geom_group[g] <= GRP_DECAL)),
            "per_tree": info["trees"], "compile_s": info["compile_s"], "timestep": m.opt.timestep,
            "targets": {"nv": NV_TARGET, "eq_rows_max": EQ_ROWS_MAX, "nmocap": NMOCAP_TARGET,
                        "render_geoms_per_tree_max": RENDER_GEOMS_PER_TREE_MAX}}


# ------------------------------------------------------------------ mjvScene post-filter cull
class SceneCuller:
    """Drops mjvScene geoms farther than ``max_dist`` from the camera after ``mjv_updateScene`` and
    compacts ``ngeom`` (ORCHARD-MVP-SPEC 3.7). No model field is touched.

    Distance = |geom pos - camera| - geom_rbound (bounding sphere), so a tree whose canopy reaches into
    the 4 m sphere is kept. Always kept, at any distance: planes and height fields, non-geom scene
    objects (sites, decor), and every geom of a mocap body (the people, animals and debris actors):
    person and animal pixels are privacy truth (MVP-SPEC 5.4 masking) and safety-relevant, so the cull
    never removes them. What the cull does remove (static scenery beyond 4 m, incl. a parcel boundary)
    is the assisted-register entry `scene_cull_4m` (ORCHARD-MVP-SPEC 3.9).
    The compaction runs on a raw byte view of the mjvGeom array (vectorised); the layout is verified at
    construction and a per-field copy is the fallback.
    """

    _INT_OFF = {"type": 0, "objtype": 8, "objid": 12, "segid": 28}

    def __init__(self, model: mujoco.MjModel, scene: mujoco.MjvScene, max_dist: float = CULL_DIST_M):
        self.model, self.scene, self.max_dist = model, scene, float(max_dist)
        self.rbound = np.asarray(model.geom_rbound, dtype=np.float64)
        self.always = np.isin(np.asarray(model.geom_type), [int(G.mjGEOM_PLANE), int(G.mjGEOM_HFIELD)])
        # actors (mocap bodies: people, animals, debris) are never culled
        self.always |= np.asarray(model.body_mocapid)[np.asarray(model.geom_bodyid)] >= 0
        self.raw = None
        self.fast = self._map()

    def _map(self) -> bool:
        import ctypes
        sc = self.scene
        if sc.maxgeom < 2:
            return False
        g0, g1 = sc.geoms[0], sc.geoms[1]
        a0 = g0.size.ctypes.data
        stride = g1.size.ctypes.data - a0
        base = a0 - 32                      # 8 int32 fields precede size[3] in mjvGeom
        self.pos_off = g0.pos.ctypes.data - base
        if stride <= 0 or self.pos_off != 44:
            return False
        buf = (ctypes.c_uint8 * (stride * sc.maxgeom)).from_address(base)
        self.raw = np.ctypeslib.as_array(buf).reshape(sc.maxgeom, stride)
        self.stride = stride
        # verify against the bindings on a sentinel write/read
        old = (g1.type, g1.objtype, g1.objid, g1.segid)
        g1.type, g1.objtype, g1.objid, g1.segid = 5, 7, 12345, 4321
        ok = (self._ints("type")[1] == 5 and self._ints("objtype")[1] == 7 and self._ints("objid")[1] == 12345
              and self._ints("segid")[1] == 4321)
        g1.type, g1.objtype, g1.objid, g1.segid = old
        return bool(ok)

    def _ints(self, field: str) -> np.ndarray:
        o = self._INT_OFF[field]
        return self.raw[:, o:o + 4].copy().view(np.int32).ravel()

    def keep_mask(self, cam_pos: Sequence[float]) -> np.ndarray:
        sc, n = self.scene, self.scene.ngeom
        if self.fast:
            pos = self.raw[:n, self.pos_off:self.pos_off + 12].copy().view(np.float32).reshape(n, 3)
            objtype = self._ints("objtype")[:n]
            objid = self._ints("objid")[:n]
        else:
            pos = np.array([sc.geoms[i].pos for i in range(n)], dtype=np.float32).reshape(n, 3)
            objtype = np.array([sc.geoms[i].objtype for i in range(n)], dtype=np.int32)
            objid = np.array([sc.geoms[i].objid for i in range(n)], dtype=np.int32)
        is_geom = (objtype == int(mujoco.mjtObj.mjOBJ_GEOM)) & (objid >= 0) & (objid < self.model.ngeom)
        dist = np.linalg.norm(pos.astype(np.float64) - np.asarray(cam_pos, dtype=np.float64), axis=1)
        gid = np.where(is_geom, objid, 0)
        far = is_geom & ~self.always[gid] & ((dist - self.rbound[gid]) > self.max_dist)
        return ~far

    def cull(self, cam_pos: Sequence[float]) -> Tuple[int, int]:
        """Filter the scene in place; returns (ngeom_before, ngeom_after)."""
        sc = self.scene
        n = sc.ngeom
        keep = self.keep_mask(cam_pos)
        idx = np.nonzero(keep)[0]
        k = int(idx.size)
        if k == n:
            return n, n
        if self.fast:
            self.raw[:k] = self.raw[idx]
            # segid must stay a scene index (mujoco.Renderer decodes segmentation by segid < ngeom)
            self.raw[:k, 28:32] = np.arange(k, dtype=np.int32).view(np.uint8).reshape(k, 4)
        else:
            fields = ("type", "dataid", "objtype", "objid", "category", "matid", "texcoord", "segid", "size",
                      "pos", "mat", "rgba", "emission", "specular", "shininess", "reflectance", "label",
                      "camdist", "modelrbound", "transparent")
            for dst, src in enumerate(idx):
                if dst == src:
                    continue
                gs, gd = sc.geoms[int(src)], sc.geoms[dst]
                for f in fields:
                    if hasattr(gs, f):
                        v = getattr(gs, f)
                        if isinstance(v, np.ndarray):
                            getattr(gd, f)[...] = v
                        else:
                            setattr(gd, f, v)
                gd.segid = dst
        sc.ngeom = k
        return n, k


def model_fingerprint(m: mujoco.MjModel) -> str:
    import hashlib
    h = hashlib.sha1()
    for a in (m.geom_rgba, m.geom_group, m.geom_size, m.geom_pos, m.geom_quat, m.geom_type, m.geom_contype,
              m.geom_conaffinity, m.geom_matid, m.geom_dataid, m.geom_rbound, m.body_pos, m.body_quat,
              m.cam_pos, m.cam_quat, m.cam_fovy, m.mat_rgba, m.vis.global_.fovy, m.nmocap):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ rays
def lidar_directions_local() -> Tuple[np.ndarray, np.ndarray]:
    """(mast directions 1080x3, low-plane directions 270x3) in the trunk frame."""
    mast = []
    for el in LIDAR_BEAMS_DEG:
        e = math.radians(el)
        for k in range(LIDAR_RAYS_PER_BEAM):
            a = 2 * math.pi * k / LIDAR_RAYS_PER_BEAM
            mast.append([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    low = []
    for k in range(LOW_PLANE_RAYS):
        a = math.radians(-135 + 270.0 * k / (LOW_PLANE_RAYS - 1))
        low.append([math.cos(a), math.sin(a), 0.0])
    return np.asarray(mast), np.asarray(low)


class Lidar:
    """1,620-ray sweep: 3 x 360 from mast_lidar + 2 low planes x 270 (front fan, rear fan mirrored)."""

    def __init__(self, m: mujoco.MjModel):
        self.m = m
        self.trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.sites = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
                      for n in ("mast_lidar", "low_plane_f", "low_plane_r")]
        mast, low = lidar_directions_local()
        self.local = [mast, low, low * np.array([-1.0, -1.0, 1.0])]
        self.n = sum(len(v) for v in self.local)
        self.geomid = [np.zeros(len(v), dtype=np.int32) for v in self.local]
        self.dist = [np.zeros(len(v), dtype=np.float64) for v in self.local]

    def sweep(self, d: mujoco.MjData, mask: Sequence[int], count_groups: bool = False) -> Dict[str, float]:
        R = d.xmat[self.trunk].reshape(3, 3)
        gg = np.asarray(mask, dtype=np.uint8)
        hits, fol = 0, 0
        for sid, loc, gid, dist in zip(self.sites, self.local, self.geomid, self.dist):
            vec = (loc @ R.T).ravel()
            mujoco.mj_multiRay(self.m, d, d.site_xpos[sid], vec, gg, 1, self.trunk, gid, dist, len(loc),
                               LIDAR_RANGE_M)
            hits += int(np.sum(gid >= 0))
            if count_groups:
                fol += int(np.sum(self.m.geom_group[gid[gid >= 0]] == GRP_FOLIAGE))
        return {"hits": hits, "foliage_hits": fol}


def coverage_points(tree: dict, n: int = COVERAGE_POINTS_PER_BAND) -> Dict[str, np.ndarray]:
    """72 canopy-surface sample points per band (low / mid / high) on the canopy ellipsoid."""
    H, cr = tree["H"], tree["canopy_r"]
    cz, rz = 0.35 * H + 0.4 * H, 0.6 * cr + 0.15
    bands = {"low": (0.2, 0.9), "mid": (0.9, 1.6), "high": (1.6, H)}
    out = {}
    for name, (z0, z1) in bands.items():
        pts = []
        for k in range(n):
            a = 2 * math.pi * k / n
            z = z0 + (z1 - z0) * ((k * 7) % n) / n
            if name == "low":
                r = 0.12 + (cr * 0.6) * (z - z0) / (z1 - z0)   # trunk / skirt band
            else:
                s = max(0.0, 1.0 - ((z - cz) / rz) ** 2)
                r = cr * math.sqrt(s)
            pts.append([tree["x"] + r * math.cos(a), tree["y"] + r * math.sin(a), z])
        out[name] = np.asarray(pts)
    return out


def coverage_visible(m, d, origin, pts, mask, surface_tol=0.15) -> np.ndarray:
    vec = pts - np.asarray(origin)
    rng = np.linalg.norm(vec, axis=1)
    unit = (vec / rng[:, None]).ravel()
    gid = np.zeros(len(pts), dtype=np.int32)
    dist = np.zeros(len(pts), dtype=np.float64)
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    mujoco.mj_multiRay(m, d, np.asarray(origin, dtype=np.float64), unit, np.asarray(mask, dtype=np.uint8),
                       1, trunk, gid, dist, len(pts), float(np.max(rng)) + 1.0)
    return (gid < 0) | (dist >= rng - surface_tol)


# ------------------------------------------------------------------ station poses
def _robot_carried_free_joints(m: mujoco.MjModel) -> List[int]:
    """qpos addresses of the free joints welded to the trunk (the rack tools). mj_kinematics does not
    enforce welds, so a station pose must carry them with the base. Pool fruit hang on trees: not moved."""
    out = []
    for k in range(N_TOOLS):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"tool_{k}_fj")
        if j >= 0:
            out.append(int(m.jnt_qposadr[j]))
    return out


def station_qpos(m: mujoco.MjModel, q: np.ndarray, station: Tuple[float, float, float]) -> np.ndarray:
    """``q`` with the trunk moved to ``station`` (x, y, heading) and the welded rack tools moved rigidly
    with it (same base transform)."""
    x, y, hd = station
    q = np.array(q, dtype=np.float64, copy=True)
    p0, q0 = q[0:3].copy(), q[3:7].copy()
    p1 = np.array([x, y, p0[2]])
    q1 = np.asarray(_rot_z(hd), dtype=np.float64)
    q0_inv = np.empty(4)
    mujoco.mju_negQuat(q0_inv, q0)
    dq = np.empty(4)
    mujoco.mju_mulQuat(dq, q1, q0_inv)          # world-frame rotation old base -> new base
    for a in _robot_carried_free_joints(m):
        rel = q[a:a + 3] - p0
        rot = np.empty(3)
        mujoco.mju_rotVecQuat(rot, rel, dq)
        q[a:a + 3] = p1 + rot
        qt = np.empty(4)
        mujoco.mju_mulQuat(qt, dq, q[a + 3:a + 7])
        q[a + 3:a + 7] = qt
    q[0:3], q[3:7] = p1, q1
    return q


def set_station_pose(m: mujoco.MjModel, d: mujoco.MjData, station: Tuple[float, float, float],
                     key_qpos: np.ndarray):
    d.qpos[:] = station_qpos(m, key_qpos, station)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_camlight(m, d)


def set_crown(m: mujoco.MjModel, d: mujoco.MjData, cid: int, height_m: float, pitch_deg: float, tz: float):
    m.cam_pos[cid] = [MAST_X, 0.0, height_m - tz]
    m.cam_quat[cid] = cam_quat(90, pitch_deg)
    mujoco.mj_camlight(m, d)


def station_captures() -> List[Tuple[str, str, Optional[Tuple[float, float]]]]:
    return [("low", "ring_90", None), ("mid", "crown_cam", CROWN_MID), ("high", "crown_cam", CROWN_HIGH)]


# ------------------------------------------------------------------ GL capture
class Capturer:
    """RGB + depth in one GL pass (two readbacks) + a segmentation pass, optional scene cull."""

    def __init__(self, m: mujoco.MjModel, width=CAPTURE_W, height=CAPTURE_H, cull: bool = True):
        self.m = m
        self.r = mujoco.Renderer(m, height=height, width=width, max_geom=20000)
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.opt = mujoco.MjvOption()
        for gi, on in enumerate(RENDER_GROUPS):
            self.opt.geomgroup[gi] = on
        self.cull = cull
        self.culler = SceneCuller(m, self.r.scene) if cull else None
        self.rgb = np.empty((height, width, 3), dtype=np.uint8)
        self.zbuf = np.empty((height, width), dtype=np.float32)

    def _update(self, d, cid) -> Tuple[int, int]:
        self.r.update_scene(d, camera=cid, scene_option=self.opt)
        n = self.r.scene.ngeom
        if self.culler is not None:
            return self.culler.cull(d.cam_xpos[cid])
        return n, n

    def capture(self, d: mujoco.MjData, cid: int, keep_images: bool = False) -> dict:
        r = self.r
        t0 = time.perf_counter()
        before, after = self._update(d, cid)
        t_upd = time.perf_counter() - t0
        if getattr(r, "_gl_context", None) is not None:
            r._gl_context.make_current()
        mujoco.mjr_render(r._rect, r.scene, r._mjr_context)
        mujoco.mjr_readPixels(self.rgb, self.zbuf, r._rect, r._mjr_context)
        t1 = time.perf_counter()
        r.enable_segmentation_rendering()
        try:
            self._update(d, cid)
            seg = r.render()
        finally:
            r.disable_segmentation_rendering()
        t2 = time.perf_counter()
        if keep_images:
            self.last = (np.flipud(self.rgb).copy(), np.flipud(r_depth(self.zbuf, self.m)).copy(), seg.copy())
        return {"rgbd_ms": (t1 - t0) * 1e3, "seg_ms": (t2 - t1) * 1e3, "capture_ms": (t2 - t0) * 1e3,
                "update_cull_ms": t_upd * 1e3, "ngeom_before": before, "ngeom_after": after,
                "seg_objects": int(len(np.unique(seg[..., 0])))}

    def close(self):
        try:
            self.r.close()
        except Exception:
            pass


def r_depth(zbuf: np.ndarray, m: mujoco.MjModel) -> np.ndarray:
    from atharv_nandi.core.camera_rig import CameraRenderer
    return CameraRenderer._linear_depth(zbuf, m)


def write_png(path: str, img: np.ndarray):
    """Minimal PNG writer (8-bit grey or RGB), no PIL needed."""
    import struct
    import zlib
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape[:2]
    ch = 1 if img.ndim == 2 else img.shape[2]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0 if ch == 1 else 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _save_samples(m, d, key_q, tz, crown, cull: bool, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cap = Capturer(m, cull=cull)
    set_station_pose(m, d, ot1_stations()[0], key_q)
    for band, cam, crown_cfg in station_captures():
        if crown_cfg is not None:
            set_crown(m, d, crown, crown_cfg[0], crown_cfg[1], tz)
        cap.capture(d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam), keep_images=True)
        rgb, depth, seg = cap.last
        tag = f"{band}_{'cull' if cull else 'nocull'}"
        write_png(os.path.join(out_dir, f"{tag}_rgb.png"), rgb)
        dn = np.clip(depth / 8.0, 0, 1)
        write_png(os.path.join(out_dir, f"{tag}_depth.png"), (255 * (1 - dn)).astype(np.uint8))
        sid = seg[..., 0].astype(np.int64)
        pal = (np.stack([(sid * 67) % 256, (sid * 141) % 256, (sid * 29) % 256], -1) * (sid[..., None] >= 0))
        write_png(os.path.join(out_dir, f"{tag}_seg.png"), pal.astype(np.uint8))
    set_crown(m, d, crown, CROWN_MID[0], CROWN_MID[1], tz)
    cap.close()


def ensure_display() -> Optional[subprocess.Popen]:
    """Start Xvfb when MUJOCO_GL=glfw and no X display answers; returns the process (or None)."""
    gl = os.environ.get("MUJOCO_GL", "")
    if gl not in ("", "glfw") or sys.platform == "darwin":
        return None
    disp = os.environ.get("DISPLAY")
    if disp and shutil.which("xdpyinfo"):
        if subprocess.run(["xdpyinfo", "-display", disp], capture_output=True).returncode == 0:
            return None
    if not shutil.which("Xvfb"):
        return None
    os.environ["DISPLAY"] = ":97"
    os.environ["MUJOCO_GL"] = "glfw"
    p = subprocess.Popen(["Xvfb", ":97", "-screen", "0", "1280x800x24"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    return p


# ------------------------------------------------------------------ identity (run-identity rule)
_GIT_ARG_RE = r"^[0-9a-f]{7,40}(-dirty)?$"


def check_git_arg(value: str) -> str:
    """``--git`` must be a hex sha with an optional ``-dirty`` suffix (the host computes both; the
    container cannot read a worktree .git). Anything else ("unknown", "", "HEAD") is refused."""
    import re
    v = (value or "").strip()
    if not re.match(_GIT_ARG_RE, v):
        raise SystemExit(f"--git {value!r}: expected <hex sha>[-dirty], e.g. "
                         "$(git rev-parse --short=12 HEAD)$(git diff --quiet HEAD || echo -dirty)")
    return v


def tool_blob_sha(path: str = __file__) -> str:
    """git blob id of this tool file as run (``git rev-parse <commit>:tools/spike_demo_budget.py`` must
    equal it for a run id to name that commit's code)."""
    import hashlib
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _git_short(repo: str = REPO) -> str:
    if os.environ.get("NANDI_GIT_SHA"):
        return check_git_arg(os.environ["NANDI_GIT_SHA"])
    try:
        from atharv_nandi.core import run_log
        sha = run_log.git_sha(repo) or "nogit"
        dirty = run_log.git_dirty(repo)
        return sha[:12] + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def cgroup_cpus() -> Optional[float]:
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if q == "max" else round(int(q) / int(p), 2)
    except Exception:
        return None


def device_id() -> str:
    return os.environ.get("NANDI_DEVICE_ID") or f"mac-capped-runner-{cgroup_cpus() or os.cpu_count()}cpu"


def identity(section: str) -> dict:
    utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = _git_short()
    return {"run_id": f"wp11a-{section}-{utc}-{sha}", "device_id": device_id(),
            "git": sha, "tool_blob": tool_blob_sha(), "utc": utc, "mujoco": mujoco.__version__, "python": platform.python_version(),
            "machine": platform.machine(), "cgroup_cpus": cgroup_cpus(), "os_cpu_count": os.cpu_count(),
            "env": {k: os.environ.get(k) for k in ("LP_NUM_THREADS", "MUJOCO_GL", "NANDI_TERRAIN",
                                                    "NANDI_CROPS", "NANDI_DT")},
            "ci_runner": CI_PENDING}


def _stats(v: Sequence[float]) -> dict:
    a = np.asarray(v, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median": round(float(np.median(a)), 3), "mean": round(float(np.mean(a)), 3),
            "p90": round(float(np.percentile(a, 90)), 3), "max": round(float(np.max(a)), 3),
            "min": round(float(np.min(a)), 3)}


# ------------------------------------------------------------------ sections
ALLEY_POSES = [(x, y, 0.0) for y in (4.5, 8.25, 11.75) for x in (2.0, 4.0, 6.0, 8.0)]


def culled_scene_counts(m: mujoco.MjModel, poses, cameras=("ring_0", "ring_90", "crown_cam")) -> dict:
    """mjvScene geom count before / after the 4 m post-filter at the given robot poses (no GL needed)."""
    d = mujoco.MjData(m)
    reset(m, d)
    key_q = d.qpos.copy()
    scn = mujoco.MjvScene(m, maxgeom=20000)
    opt = mujoco.MjvOption()
    for gi, on in enumerate(RENDER_GROUPS):
        opt.geomgroup[gi] = on
    culler = SceneCuller(m, scn)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    before, after = [], []
    for pose in poses:
        set_station_pose(m, d, pose, key_q)
        for c in cameras:
            cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c)
            mujoco.mjv_updateScene(m, d, opt, mujoco.MjvPerturb(), cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
            b, a = culler.cull(d.cam_xpos[cam.fixedcamid])
            before.append(b)
            after.append(a)
    return {"before": _stats(before), "after": _stats(after)}


def sec_stub(args) -> dict:
    m, info = build_stub()
    return {"stub": stub_counts(m, info),
            "scene_cull_ot1_stations": culled_scene_counts(m, ot1_stations()),
            "scene_cull_alleys": culled_scene_counts(m, ALLEY_POSES)}


def _bench_steps(m, steps, hook=None, hook_every=0) -> dict:
    d = mujoco.MjData(m)
    reset(m, d)
    for _ in range(300):
        mujoco.mj_step(m, d)
    t0 = time.perf_counter()
    hook_t = []
    for i in range(steps):
        mujoco.mj_step(m, d)
        if hook is not None and i % hook_every == 0:
            h0 = time.perf_counter()
            hook(d)
            hook_t.append((time.perf_counter() - h0) * 1e3)
    wall = time.perf_counter() - t0
    phys_ms = (wall - sum(hook_t) / 1e3) / steps * 1e3
    out = {"ms_per_step": round(phys_ms, 4), "steps": steps, "rtf_physics_only": round(m.opt.timestep / (phys_ms / 1e3), 2),
           "physics_core_frac_at_rtf_0_9": round(RTF_TARGET * phys_ms / 1e3 / m.opt.timestep, 4),
           "nefc_end": int(d.nefc), "ncon_end": int(d.ncon), "trunk_z_end": round(float(d.qpos[2]), 3),
           "badqacc": int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)}
    if hook_t:
        out["hook_ms"] = _stats(hook_t)
    return out


def sec_physics(args) -> dict:
    steps = int(args.steps)
    base = farm_scene.load_farm_model()
    res = {"robot_farm_flat": _bench_steps(base, steps)}
    m, info = build_stub()
    res["stub_counts"] = {k: v for k, v in stub_counts(m, info).items() if k != "per_tree"}
    res["orchard_stub"] = _bench_steps(m, steps)
    lid = Lidar(m)
    every10 = int(round(0.1 / m.opt.timestep))
    res["orchard_stub_lidar_10hz_safety"] = _bench_steps(m, steps, lambda d: lid.sweep(d, SAFETY_MASK), every10)
    res["orchard_stub_lidar_10hz_coverage"] = _bench_steps(m, steps, lambda d: lid.sweep(d, COVERAGE_MASK), every10)
    for k in ("orchard_stub_lidar_10hz_safety", "orchard_stub_lidar_10hz_coverage"):
        res[k]["lidar_rays"] = lid.n
        res[k]["lidar_ms_per_s_at_10hz"] = round(res[k]["hook_ms"]["mean"] * 10, 2)
    try:
        from atharv_nandi.core.stability_probe import MonitorCore
        mon = MonitorCore(m)
        tick = [0.0]

        def stab(d):
            tick[0] += 0.02
            mon.on_qpos(d.qpos)
            mon.status(tick[0])
        every50 = int(round(0.02 / m.opt.timestep))
        r = _bench_steps(m, steps, stab, every50)
        r["stability_ms_per_s_at_50hz"] = round(r["hook_ms"]["mean"] * 50, 2)
        res["orchard_stub_stability_50hz"] = r
    except Exception as exc:  # pragma: no cover - informational
        res["orchard_stub_stability_50hz"] = {"error": f"{type(exc).__name__}: {exc}"}
    # mj_setConst per payload event (on a copy: it rewrites qpos0-derived constants)
    import copy
    m2 = copy.deepcopy(m)
    d2 = mujoco.MjData(m2)
    reset(m2, d2)
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        mujoco.mj_setConst(m2, d2)
        ts.append((time.perf_counter() - t0) * 1e3)
    res["mj_setConst_ms"] = _stats(ts)
    return res


def sec_render(args) -> dict:
    xv = ensure_display()
    try:
        m, info = build_stub()
        d = mujoco.MjData(m)
        reset(m, d)
        key_q = d.qpos.copy()
        tz = float(key_q[2])
        crown = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "crown_cam")
        out: Dict[str, dict] = {"counts": {k: v for k, v in stub_counts(m, info).items() if k != "per_tree"}}
        stations = ot1_stations()
        fp_change: Dict[str, bool] = {}
        for cull in (False, True):
            fp_start = model_fingerprint(m)   # per mode: a change is attributed to this loop only
            cap = Capturer(m, cull=cull)
            # warm-up (context, shaders, mesh upload)
            set_station_pose(m, d, stations[0], key_q)
            for _ in range(3):
                cap.capture(d, crown)
            rows = {"ring": [], "crown_mid": [], "crown_high": []}
            for rep in range(int(args.reps)):
                for st in stations:
                    set_station_pose(m, d, st, key_q)
                    for band, cam, crown_cfg in station_captures():
                        if crown_cfg is not None:
                            set_crown(m, d, crown, crown_cfg[0], crown_cfg[1], tz)
                        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
                        rows["ring" if band == "low" else f"crown_{band}"].append(cap.capture(d, cid))
                set_crown(m, d, crown, CROWN_MID[0], CROWN_MID[1], tz)
            alley = []
            for pose in ALLEY_POSES:     # in-row views: the densest scenes (3 rows within 4 m)
                set_station_pose(m, d, pose, key_q)
                alley.append(cap.capture(d, crown))
            cap.close()
            tag = "cull_4m" if cull else "no_cull"
            fp_change[tag] = (fp_start != model_fingerprint(m))   # after orbit + alleys + close
            res = {}
            res["alley_crown_mid"] = {f: _stats([r[f] for r in alley]) for f in ("rgbd_ms", "seg_ms", "capture_ms", "ngeom_after")}
            for k, v in rows.items():
                res[k] = {f: _stats([r[f] for r in v]) for f in ("rgbd_ms", "seg_ms", "capture_ms",
                                                                   "update_cull_ms", "ngeom_before", "ngeom_after")}
            st_ms = [sum(rows[k][i]["capture_ms"] for k in rows) for i in range(len(rows["ring"]))]
            res["station_9pass_equiv_ms"] = _stats([s * PASSES_PER_CAPTURE_BUDGET / 2.0 for s in st_ms])
            res["station_measured_ms"] = _stats(st_ms)
            if cull:
                res["culler_fast_path"] = bool(cap.culler.fast)
            if args.save_dir:
                _save_samples(m, d, key_q, tz, crown, cull, args.save_dir)
            out[tag] = res
        # crown restored to mid == build pose before each fingerprint, so a difference is a model write
        out["model_unchanged_by_cull"] = not fp_change["cull_4m"]
        out["model_unchanged_no_cull"] = not fp_change["no_cull"]
        out["scene_geoms_after_cull_max"] = max(out["cull_4m"][k]["ngeom_after"]["max"] for k in ("ring", "crown_mid", "crown_high", "alley_crown_mid"))
        out["scene_geoms_no_cull_max"] = max(out["no_cull"][k]["ngeom_after"]["max"] for k in ("ring", "crown_mid", "crown_high", "alley_crown_mid"))
        out["rules"] = {"scene_geoms_after_cull_max": SCENE_GEOMS_AFTER_CULL_MAX,
                        "render_geoms_per_tree_max": RENDER_GEOMS_PER_TREE_MAX}
        return out
    finally:
        if xv:
            xv.terminate()


def sec_rays(args) -> dict:
    m, info = build_stub()
    d = mujoco.MjData(m)
    reset(m, d)
    key_q = d.qpos.copy()
    tz = float(key_q[2])
    lid = Lidar(m)
    out: Dict[str, dict] = {"lidar_rays": lid.n}
    # OT-1 orbit stations + R1/R2 alley poses (y 4.5, between mango R1 and guava R2)
    stations = ot1_stations() + [(x, 4.5, 0.0) for x in (2.0, 4.0, 6.0, 8.0)]
    for name, mask in (("safety_mask_0_1", SAFETY_MASK), ("coverage_mask_0_1_3", COVERAGE_MASK)):
        ts, hits, fol = [], [], []
        for rep in range(int(args.reps)):
            for st in stations:
                set_station_pose(m, d, st, key_q)
                t0 = time.perf_counter()
                h = lid.sweep(d, mask)
                ts.append((time.perf_counter() - t0) * 1e3)
                hits.append(h["hits"])
                fol.append(lid.sweep(d, mask, count_groups=True)["foliage_hits"])
        out[name] = {"ms_per_sweep": _stats(ts), "ms_per_s_at_10hz": round(float(np.mean(ts)) * 10, 2),
                     "hits_per_sweep": _stats(hits), "foliage_hits_per_sweep": _stats(fol)}
    # TREE 360 coverage rays from the capture lens positions to 72 points x 3 bands
    pts = coverage_points(OT1)
    crown = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "crown_cam")
    ring = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ring_90")
    cov = {}
    for name, mask in (("no_foliage_0_1", SAFETY_MASK), ("coverage_0_1_3", COVERAGE_MASK)):
        ts, seen = [], {b: np.zeros(COVERAGE_POINTS_PER_BAND, bool) for b in pts}
        per_cap = {b: [] for b in pts}
        for st in ot1_stations():
            set_station_pose(m, d, st, key_q)
            for band, cam, crown_cfg in station_captures():
                if crown_cfg is not None:
                    set_crown(m, d, crown, crown_cfg[0], crown_cfg[1], tz)
                cid = ring if cam == "ring_90" else crown
                t0 = time.perf_counter()
                vis = coverage_visible(m, d, d.cam_xpos[cid].copy(), pts[band], mask)
                ts.append((time.perf_counter() - t0) * 1e3)
                seen[band] |= vis
                per_cap[band].append(float(np.mean(vis)))
        cov[name] = {"ms_per_capture_72_rays": _stats(ts),
                     "seen_frac_by_band_ray_only": {b: round(float(np.mean(v)), 3) for b, v in seen.items()},
                     "visible_frac_per_capture_mean": {b: round(float(np.mean(v)), 3) for b, v in per_cap.items()}}
    set_crown(m, d, crown, CROWN_MID[0], CROWN_MID[1], tz)
    out["coverage_rays"] = cov
    out["note"] = ("seen_frac is ray occlusion only (no frustum, no footprint): it shows the coverage mask "
                   "counting foliage, not O-01 coverage")
    return out


def sec_viewer(args) -> dict:
    xv = ensure_display()
    import resource
    try:
        m, _ = build_stub()
        d = mujoco.MjData(m)
        reset(m, d)
        dur = float(args.seconds)

        def run(viewer) -> dict:
            ru0 = resource.getrusage(resource.RUSAGE_SELF)
            t0 = time.perf_counter()
            sim0 = d.time
            next_sync, syncs, sync_ms = t0, 0, []
            while time.perf_counter() - t0 < dur:
                target = t0 + (d.time - sim0) / 1.0     # paced at RTF 1.0
                now = time.perf_counter()
                if now < target:
                    time.sleep(min(0.002, target - now))
                    continue
                mujoco.mj_step(m, d)
                if viewer is not None and now >= next_sync:
                    s0 = time.perf_counter()
                    viewer.sync()
                    sync_ms.append((time.perf_counter() - s0) * 1e3)
                    syncs += 1
                    next_sync += 0.1
            wall = time.perf_counter() - t0
            ru1 = resource.getrusage(resource.RUSAGE_SELF)
            cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
            return {"wall_s": round(wall, 2), "sim_s": round(d.time - sim0, 2), "rtf": round((d.time - sim0) / wall, 3),
                    "cpu_cores": round(cpu / wall, 3), "syncs": syncs, "sync_ms": _stats(sync_ms)}

        out = {"no_viewer": run(None)}
        try:
            import mujoco.viewer as mv
            with mv.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
                time.sleep(1.0)
                out["passive_viewer_10hz"] = run(v)
        except Exception as exc:
            out["passive_viewer_10hz"] = {"error": f"{type(exc).__name__}: {exc}"}
        if "cpu_cores" in out.get("passive_viewer_10hz", {}):
            out["viewer_cost_cores"] = round(out["passive_viewer_10hz"]["cpu_cores"] - out["no_viewer"]["cpu_cores"], 3)
        return out
    finally:
        if xv:
            xv.terminate()


# --- ack: two processes ---------------------------------------------------
def _perception_worker(conn, cull: bool, passes: int = 2):
    xv = ensure_display()
    try:
        m, _ = build_stub()
        d = mujoco.MjData(m)
        reset(m, d)
        tz = float(d.qpos[2])
        crown = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "crown_cam")
        cap = Capturer(m, cull=cull)
        for _ in range(3):
            cap.capture(d, crown)
        conn.send({"ready": True})
        import resource
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        render_s = 0.0
        while True:
            msg = conn.recv()
            if msg is None:
                break
            d.qpos[:] = msg["qpos"]
            d.mocap_pos[:] = msg["mocap_pos"]
            d.mocap_quat[:] = msg["mocap_quat"]
            d.eq_active[:] = msg["eq_active"]
            if msg["crown"] is not None:
                set_crown(m, d, crown, msg["crown"][0], msg["crown"][1], tz)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_camlight(m, d)
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, msg["cam"])
            r = cap.capture(d, cid)
            if passes >= 3:   # budgeted third pass: one more RGB+depth render of the same view
                t3 = time.perf_counter()
                cap.r.update_scene(d, camera=cid, scene_option=cap.opt)
                if cap.culler is not None:
                    cap.culler.cull(d.cam_xpos[cid])
                mujoco.mjr_render(cap.r._rect, cap.r.scene, cap.r._mjr_context)
                mujoco.mjr_readPixels(cap.rgb, cap.zbuf, cap.r._rect, cap.r._mjr_context)
                r["capture_ms"] += (time.perf_counter() - t3) * 1e3
            render_s += r["capture_ms"] / 1e3
            conn.send({"ack": msg["seq"], "capture_ms": r["capture_ms"], "ngeom_after": r["ngeom_after"]})
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        conn.send({"cpu_s": (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime), "render_s": render_s})
        cap.close()
    finally:
        if xv:
            xv.terminate()


def sec_ack(args) -> dict:
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_perception_worker, args=(child, not args.no_cull, int(args.passes)), daemon=True)
    proc.start()
    m, _ = build_stub()
    d = mujoco.MjData(m)
    reset(m, d)
    key_q = d.qpos.copy()
    lid = Lidar(m)
    if not parent.poll(120):
        raise RuntimeError("perception worker did not start")
    parent.recv()
    dt = m.opt.timestep
    lidar_every = int(round(0.1 / dt))
    settle_s, walk_s = 1.0, float(args.walk_s)
    rows = []
    seq = 0
    t_wall0 = time.perf_counter()
    sim0 = d.time
    step_i = 0

    def step(n=1):
        nonlocal step_i
        for _ in range(n):
            target = t_wall0 + (d.time - sim0) / RTF_TARGET
            now = time.perf_counter()
            if now < target:
                time.sleep(target - now)
            mujoco.mj_step(m, d)
            step_i += 1
            if step_i % lidar_every == 0:
                lid.sweep(d, SAFETY_MASK)

    for si, st in enumerate(ot1_stations()):
        step(int(walk_s / dt))
        step(int(settle_s / dt))
        for band, cam, crown_cfg in station_captures():
            q = station_qpos(m, d.qpos, st)
            seq += 1
            t_req_sim, t_req_wall = d.time, time.perf_counter()
            parent.send({"seq": seq, "qpos": q, "mocap_pos": d.mocap_pos.copy(), "mocap_quat": d.mocap_quat.copy(),
                         "eq_active": d.eq_active.copy(), "cam": cam, "crown": crown_cfg})
            while not parent.poll(0):
                step(1)
                if d.time - t_req_sim > 10.0:
                    raise RuntimeError("ack timeout")
            ack = parent.recv()
            ack_sim = d.time - t_req_sim
            ack_wall = time.perf_counter() - t_req_wall
            rows.append({"station": si, "band": band, "ack_sim_s": round(ack_sim, 4), "ack_wall_s": round(ack_wall, 4),
                         "capture_ms": round(ack["capture_ms"], 2), "within_window": ack_sim <= ACK_WINDOW_SIM_S})
            # the rest of the 0.5 s sim capture window (if any) is spent dwelling
            rest = ACK_WINDOW_SIM_S - ack_sim
            if rest > 0:
                step(int(rest / dt))
    wall = time.perf_counter() - t_wall0
    sim = d.time - sim0
    parent.send(None)
    tail = parent.recv() if parent.poll(30) else {}
    proc.join(10)
    capture_wall = sum(r["ack_wall_s"] for r in rows)
    return {"acks": rows, "ack_sim_s": _stats([r["ack_sim_s"] for r in rows]),
            "ack_wall_s": _stats([r["ack_wall_s"] for r in rows]),
            "capture_ms": _stats([r["capture_ms"] for r in rows]),
            "acks_within_window": int(sum(r["within_window"] for r in rows)), "n_acks": len(rows),
            "rtf_overall": round(sim / wall, 3), "wall_s": round(wall, 2), "sim_s": round(sim, 2),
            "rtf_during_captures": round(sum(r["ack_sim_s"] for r in rows) / max(capture_wall, 1e-9), 3),
            "perception_cpu_s": round(tail.get("cpu_s", float("nan")), 2),
            "perception_render_s": round(tail.get("render_s", float("nan")), 2),
            "cull": not args.no_cull, "passes_per_capture": int(args.passes), "walk_s_between_stations": walk_s, "lidar_10hz_in_sim": True}


def sec_livestack(args) -> dict:
    """Run launch/launch_nandi.sh (current legacy world, NANDI_HEADLESS=1) and sample CPU per process."""
    xv = ensure_display()
    env = dict(os.environ, NANDI_HEADLESS="1")
    if not args.keep_terrain:
        env.pop("NANDI_TERRAIN", None)
    p = subprocess.Popen(["bash", os.path.join(REPO, "launch", "launch_nandi.sh")], env=env, cwd=REPO,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    hz = os.sysconf("SC_CLK_TCK")

    def snap() -> Dict[int, Tuple[str, float]]:
        out = {}
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                stat = open(f"/proc/{pid}/stat").read()
                cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ").strip()
            except OSError:
                continue
            fields = stat[stat.rfind(")") + 2:].split()
            cpu = (int(fields[11]) + int(fields[12])) / hz
            out[int(pid)] = (cmd, cpu)
        return out

    try:
        time.sleep(float(args.warmup_s))
        s0, t0 = snap(), time.perf_counter()
        time.sleep(float(args.seconds))
        s1, t1 = snap(), time.perf_counter()
        wall = t1 - t0
        procs = []
        for pid, (cmd, cpu) in s1.items():
            if pid == os.getpid() or not cmd:
                continue
            c0 = s0.get(pid, (cmd, 0.0))[1]
            cores = (cpu - c0) / wall
            if cores < 0.005:
                continue
            name = proc_label(cmd)   # `<node>.py` also for `python3 -m atharv_nandi.nodes <node>` (WPR)
            procs.append({"pid": pid, "proc": name, "cores": round(cores, 3)})
        procs.sort(key=lambda r: -r["cores"])
        rtf = None
        try:   # best effort: one /nandi/game_status sample (std_msgs/String JSON with "rtf")
            txt = subprocess.run(["ros2", "topic", "echo", "--once", "--full-length", "/nandi/game_status"],
                                 capture_output=True, text=True, timeout=15, env=env).stdout
            body = txt.split("data:", 1)[1].strip().strip("-").strip()
            body = body[1:-1] if body[:1] in ("'", '"') else body
            rtf = json.loads(body.replace("\\'", "'")).get("rtf")
        except Exception:
            rtf = None
        return {"game_status_rtf": rtf,"window_s": round(wall, 1), "warmup_s": float(args.warmup_s), "processes": procs,
                "total_cores": round(sum(r["cores"] for r in procs), 3),
                "world": "legacy farm world (NANDI_TERRAIN default); the orchard world is not built yet (WP0/WP23)",
                "stack_alive": p.poll() is None}
    finally:
        try:
            os.killpg(p.pid, 15)
        except Exception:
            pass
        if xv:
            xv.terminate()


# ------------------------------------------------------------------ verdict (pure)
RENDER_BUDGET_BURST_S = 0.3            # camera_rig.RenderBudget default burst (perception_node uses it)
RTF_TOLERANCE = 0.02                   # criterion C: RTF in the capture windows >= rtf - tolerance


def _replay(capture_ms: Sequence[float], window_wall_s: float, frac: float, burst_s: float) -> int:
    from atharv_nandi.core.camera_rig import RenderBudget
    b = RenderBudget(frac, burst_s)
    now = 0.0
    skips = 0
    for c in capture_ms:
        if b.allow(now):
            b.charge(c / 1e3)
        else:
            skips += 1
        now += window_wall_s
    return skips


def token_bucket_skips(capture_ms: Sequence[float], window_wall_s: float, frac: float = RENDER_BUDGET_FRAC,
                       burst_s: float = RENDER_BUDGET_BURST_S) -> dict:
    """Replay camera_rig.RenderBudget over one station: captures start every ``window_wall_s`` and each
    costs ``capture_ms`` (all passes). Returns the skips at ``burst_s`` and the smallest burst (to 1 ms,
    rounded up) for which the replay itself skips nothing. Zero skips is monotone in the burst (a larger
    cap never lowers the balance while every capture is charged), so a bisection on the replay is exact,
    refill cap included."""
    skips = _replay(capture_ms, window_wall_s, frac, burst_s)
    total = sum(capture_ms) / 1e3
    span = window_wall_s * max(0, len(capture_ms) - 1)
    lo, hi = 0.0, total + 1e-3            # burst > total never runs dry
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _replay(capture_ms, window_wall_s, frac, mid) == 0:
            hi = mid
        else:
            lo = mid
    need = math.ceil(hi * 1e3) / 1e3
    if _replay(capture_ms, window_wall_s, frac, need) != 0:   # guard the rounding
        need += 1e-3
    return {"skips": skips, "burst_s": burst_s, "burst_s_for_zero_skips": round(need, 3),
            "render_s": round(total, 3), "span_s": round(span, 3)}


def verdict(render: dict, ack: Optional[dict] = None, rtf: float = RTF_TARGET,
            frac: float = RENDER_BUDGET_FRAC, burst_s: float = RENDER_BUDGET_BURST_S) -> dict:
    """GO / NO-GO for 3 captures x 3 passes per station within RenderBudget(frac, burst_s) at RTF ``rtf``.

    Gates (all must hold):
      B  every capture ack arrives within ACK_WINDOW_SIM_S of sim time (ack section, if given).
      C  physics keeps RTF >= rtf - RTF_TOLERANCE during the capture windows (ack section).
      D  scene geoms after the cull <= SCENE_GEOMS_AFTER_CULL_MAX.
      E  camera_rig.RenderBudget(frac, burst_s), replayed on the p90 3-pass station captures (one per
         ACK_WINDOW_SIM_S / rtf of wall), skips no capture. This is the binding render-fraction gate.
    Report-only: A, the station mean render fraction (9 passes over ORBIT_SIM_S / STATIONS / rtf of wall,
    which includes the idle arc, so it is loose: it fails only at ~1.8 s per capture), the in-window
    peak fraction and the burst the bucket needs for zero skips.
    """
    cul = render["cull_4m"]
    st9 = cul["station_9pass_equiv_ms"]["p90"] / 1e3
    station_wall = ORBIT_SIM_S / STATIONS / rtf
    frac_station = st9 / station_wall
    cap_p90 = [cul[k]["capture_ms"]["p90"] * PASSES_PER_CAPTURE_BUDGET / 2.0 for k in ("ring", "crown_mid", "crown_high")]
    window_wall = ACK_WINDOW_SIM_S / rtf
    peak = max(cap_p90) / 1e3 / window_wall
    tb = token_bucket_skips(cap_p90, window_wall, frac, burst_s)
    crit = {"D_scene_geoms_after_cull": {"value": render.get("scene_geoms_after_cull_max"),
                                         "limit": SCENE_GEOMS_AFTER_CULL_MAX,
                                         "ok": (render.get("scene_geoms_after_cull_max") or 0) <= SCENE_GEOMS_AFTER_CULL_MAX},
            "E_render_budget_no_skips": {"value": tb["skips"], "limit": 0, "burst_s": burst_s, "frac": frac,
                                         "burst_s_for_zero_skips": tb["burst_s_for_zero_skips"],
                                         "ok": tb["skips"] == 0}}
    if ack is not None:
        crit["B_ack_within_0_5_s_sim"] = {"value": f"{ack['acks_within_window']}/{ack['n_acks']}",
                                          "p90_ack_sim_s": ack["ack_sim_s"]["p90"],
                                          "ok": ack["acks_within_window"] == ack["n_acks"]}
        crit["C_rtf_during_captures"] = {"value": ack["rtf_during_captures"], "limit": round(rtf - RTF_TOLERANCE, 3),
                                         "target": rtf, "tolerance": RTF_TOLERANCE,
                                         "ok": ack["rtf_during_captures"] >= rtf - RTF_TOLERANCE}
    ok = all(c["ok"] for c in crit.values())
    return {"go": bool(ok), "criteria": crit,
            "report_only": {"A_station_mean_frac": {"value": round(frac_station, 4), "limit": frac,
                                                    "ok": frac_station <= frac},
                            "capture_window_peak_frac_p90": round(peak, 3),
                            "render_budget_token_bucket": tb,
                            "station_wall_s": round(station_wall, 2), "station_9pass_p90_s": round(st9, 3)},
            "complete": ack is not None}


def sec_verdict(args) -> dict:
    render = json.load(open(args.render))
    render = render.get("render", render)
    ack = None
    if args.ack:
        a = json.load(open(args.ack))
        ack = a.get("ack", a)
    return verdict(render, ack, burst_s=float(args.burst))   # main() nests it under out["verdict"]


SECTIONS = {"stub": sec_stub, "physics": sec_physics, "render": sec_render, "rays": sec_rays,
            "viewer": sec_viewer, "ack": sec_ack, "livestack": sec_livestack, "verdict": sec_verdict}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("section", choices=sorted(SECTIONS))
    ap.add_argument("--out", help="also write the JSON here")
    ap.add_argument("--git", help="git short sha for the run id (the container cannot read a worktree .git)")
    ap.add_argument("--steps", type=int, default=2000, help="physics: steps per case")
    ap.add_argument("--reps", type=int, default=2, help="render/rays: orbits")
    ap.add_argument("--seconds", type=float, default=15.0, help="viewer: seconds per case; livestack: window")
    ap.add_argument("--warmup-s", type=float, default=45.0, help="livestack: start-up wait")
    ap.add_argument("--keep-terrain", action="store_true", help="livestack: keep NANDI_TERRAIN")
    ap.add_argument("--walk-s", type=float, default=2.0, help="ack: sim seconds walked between stations")
    ap.add_argument("--no-cull", action="store_true", help="ack: render without the scene cull")
    ap.add_argument("--passes", type=int, default=2, choices=(2, 3), help="ack: GL passes per capture (3 = budget)")
    ap.add_argument("--render", help="verdict: render JSON")
    ap.add_argument("--save-dir", help="render: write sample RGB / depth / seg images of station 0 here")
    ap.add_argument("--ack", help="verdict: ack JSON")
    ap.add_argument("--burst", type=float, default=RENDER_BUDGET_BURST_S,
                    help="verdict: RenderBudget burst_s for gate E (default: camera_rig default)")
    args = ap.parse_args(argv)
    for k, v in ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)
    if args.git:
        os.environ["NANDI_GIT_SHA"] = check_git_arg(args.git)
    t0 = time.perf_counter()
    out = {"identity": identity(args.section), "section": args.section}
    out[args.section] = SECTIONS[args.section](args)
    out["identity"]["wall_s"] = round(time.perf_counter() - t0, 1)
    text = json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if args.out:   # before printing: a GL thread may abort the process at X shutdown
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
