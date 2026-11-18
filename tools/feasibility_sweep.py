#!/usr/bin/env python3
"""WP0c: orchard reach, load and feasibility sweep for Atharv Nandi (ROS-free, headless MuJoCo).

    python3 tools/feasibility_sweep.py candidates [--write]   # (re)generate the candidate block of the JSON
    python3 tools/feasibility_sweep.py sweep [--write]        # reach map, targets, leaves, rack, crate, mast, 5x3 variant
    python3 tools/feasibility_sweep.py load [--write]         # T4 at 1.2 / 1.8 kg: sag, joint torques vs both gates
    python3 tools/feasibility_sweep.py creep [--write]        # standing creep 0-10 deg, every fall-line direction
    python3 tools/feasibility_sweep.py matrix [--write]       # stability matrix: roll x +-pitch grid + heading sweep

Spec: ORCHARD-MVP-SPEC 2.1-2.4 and 4.2, acceptance in WP0c.
Inputs and outputs live in config/fields/orchard_candidates.json: `candidates` is generated here, the
`frozen`, `load`, `creep` and `matrix` blocks are written by the matching sub-command with a run id.

Model: the pre-WP1 MJCF (atharv_nandi/sim/models/AtharvNandi/atharv_nandi_v1.xml = physics alignment v1) plus the WP0c full
layout built with MjSpec, every addition joint-free and non-colliding (contype 0): the sectioned crown mast
at mast_base, the 3-cradle rack with its racked tools, the mounted tool on arm_wrist, the fruit at the tool
TCP and a basket payload. Nothing in atharv_nandi/ is edited; WP1 has since built the real geometry in atharv_nandi.xml.

Every number this script writes is (measured, sweep) with a run_id and a device_id (run-identity rule). Tool
geometry, tool masses, crate size and tree shapes are assumptions and carry that label in the JSON.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from atharv_nandi import paths  # noqa: E402

# The pre-WP1 robot (frozen copy, identical physics to the MJCF this sweep measured): WP1 now ships the crown
# mast, rack and tray in atharv_nandi.xml itself, with the same names this tool adds (mast_s1..s3, crown_head,
# crown_ref) and different masses (head 0.4 kg, battery moved), so building on it would double the mast. The
# recorded run ids stay reproducible on this file; a re-run on the WP1 layout is a WP6 matrix job.
ROBOT_XML = os.path.join(paths.ROBOT_MODEL_DIR, "atharv_nandi_v1.xml")
CANDIDATES_JSON = os.path.join(REPO, "config", "fields", "orchard_candidates.json")

# ---------------------------------------------------------------------------------------------- rules
PICK_BAND_Z = 0.80            # OS 2.2: contact targets at z <= 0.8 m
REACH_FRACTION = 0.80         # OS 2.2: within <= 80 % of the measured reach in their direction
EXIT_M = 0.05                 # WP0c: approach + 5 cm exit
EXIT_STEPS = 5
ROBOT_CLEAR_M = 0.005         # arm/tool vs robot (trunk, basket, rail, legs, mast, rack): no collision + 5 mm
WOOD_CLEAR_M = 0.005          # arm/tool vs tree wood
JAW_CLEAR_M = 0.015           # OS 2.4 jaw_clearance: 15 mm per side vs trunk and neighbouring fruit
# T4 twist roll (OS 2.4 / MP 5; atharv_nandi/core/tasks/twist_pick.py): the open jaws are pre-rolled 20 deg against the
# twist, close, then roll the other way until the release fires (crop twist_min_roll_deg 30). The jaw clearance is
# checked over that stroke plus a 15 deg overshoot, per twist direction: "+" = pre-roll -20 then roll toward +,
# "-" = mirrored (twist_pick's retry). Roll is about the tool axis (wrist +x = the twist_roll hinge axis of
# atharv_nandi/sim/models/AtharvNandi/tools/twist_picker.xml), right-handed. A pose freezes when one direction holds 15 mm at
# every checked roll; frozen T4 records carry twist.dirs_ok and the clearance per roll.
TWIST_PRE_ROLL_DEG = 20.0
TWIST_MIN_RELEASE_DEG = 30.0
TWIST_OVERSHOOT_DEG = 15.0
TWIST_CHECK_ROLLS_DEG = (-TWIST_PRE_ROLL_DEG, 0.0, 15.0, TWIST_MIN_RELEASE_DEG,
                         TWIST_MIN_RELEASE_DEG + TWIST_OVERSHOOT_DEG)   # direction "+"; "-" mirrors the signs
TWIST_DIRS = {"+": 1.0, "-": -1.0}
LEAF_CLEAR_M = 0.02           # MP 3: wrist-leaf clearance >= 2 cm
GROUND_CLEAR_M = 0.02         # arm/tool vs ground
KEEP_OUT_R = 0.08             # OS 2.2: mast keep-out cylinder r 0.08 m
SHOULDER_SPEC_FORCERANGE = 85.0
SHOULDER_GATE_NM = 0.7 * SHOULDER_SPEC_FORCERANGE   # 59.5 N.m (WP0c acceptance as written)
SAG_GATE_M = 0.005            # TCP sag < 5 mm with feed-forward
UNDERSIDE_MIN_FRACTION = 0.40 # >= 40 % of in-band leaf quads
CREEP_DWELL_LIMIT_M = 0.02    # OS 2.1 base pose loop: > 2 cm -> pause + re-approach; dwell slope = creep <= 2 cm / 20 s
CREEP_WINDOW_S = 20.0         # one pick cycle: the dwell limit is stated for this window
CREEP_REF_S = 0.1             # pose-loop reference taken 0.1 s after arrival (feet in contact); creep counts from here
LONG_DWELL_S = 60.0           # long dwells (LEAF SCAN, TREE 360 station, WORK assist): creep budget checked here too
CREEP_LONG_WINDOW_S = 120.0   # long creep runs: time to the 2 cm pose-loop threshold is measured inside this
TRIPOD_START_GATE_M = 0.04    # OS 2.3 rule 2: crown-up walking start gate (predicted worst tripod margin)
TRIPOD_CONTINUE_GATE_M = 0.03 # OS 2.3 rule 2: continue gate
MATRIX_TILTS_DEG = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)   # heading sweep tilts (OS 4.2 has a 10 deg patch)
MATRIX_DOWNHILL_DEG = tuple(float(a) for a in range(0, 360, 15))   # downhill direction in the robot frame
MATRIX_PITCH_DEG = (-10.0, -5.0, 0.0, 5.0, 10.0)    # + = nose downhill, - = rear (mast side) downhill
MATRIX_ROLL_DEG = (0.0, 2.0, -2.0, 3.0, -3.0, 5.0, -5.0)
MATRIX_EXT = (0.0, 0.6, 1.6)
WALK_SLOPE_LIMIT_DEG = 5.0    # OS 5.2: walking any preset <= 5 deg cross-slope (a dwell station must be walked to)
SWEEP_BUDGET_S = 120.0
MODEL_HEADROOM = 0.70         # joints at <= 70 % of the shipped MJCF forcerange (load gate next to the spec's)
# OS 2.2 arm reach as quoted in the spec (probe, pre-alignment): distance ahead at height. The measured map differs.
SPEC_OS22_REACH_FWD = {"0.035": 0.83, "0.400": 0.98, "0.800": 0.89}
MANGO_5X3_IN_ROW_M = 3.0      # OQ-02 B alternative mango spacing (5 x 3 m), swept as a variant of R1
OBS_N = 32                    # pooled obstacle capsules / spheres in the sweep model

# Station rules (robot frame: origin on the ground under the trunk centre, x forward, y left, z up).
STATION_PY = (0.24, 0.30, 0.36, 0.42, 0.48, 0.54, 0.60, 0.66)   # target lateral offset to the robot centre
STATION_PX = (-0.10, 0.05, 0.20, 0.35)                           # target along-x offset to the robot centre
# Face-in stations (robot heading toward the trunk, mast 0.30 m behind the centre): target ahead of the robot.
FACE_PX = (0.40, 0.50, 0.60, 0.70)
FACE_PY = (-0.10, 0.0, 0.10)
STATION_WOOD_CLEAR_M = 0.10   # feet and body >= 10 cm from any wood
CROWN_FOLIAGE_CLEAR_M = 0.05  # OS 2.3 rule 3: head < 0.05 m from wood / foliage envelope stops the mast; a stowed
                              # head parked closer would raise strike events, so stations keep >= 0.05 m

# Arm grids for the closed-form IK.
GANTRY_GRID = np.round(np.linspace(-0.26, 0.26, 27), 4)
PHI_GRID = np.radians(np.arange(-180.0, 180.0, 3.0))

# Mast (OS 2.3) and full layout (OS 2.6). Masses are assumptions; head at the 0.6 kg hard cap = worst case.
MAST_XY = (-0.30, 0.0)
MAST_BASE_Z = 0.05            # trunk frame (top of the trunk box)
MAST_RING_Z = 0.59            # mast-local ring height -> ring lens 0.98 m world at home
MAST_HEAD_LOCAL_Z = 0.71      # mast-local crown head centre stowed -> lens 1.10 m world
MAST_STROKE = 1.60
MAST_FIXED = (("mast_drive", 0.40), ("mast_outer", 0.35), ("mast_ring", 0.85))   # 1.6 kg fixed
MAST_SECTIONS = (("mast_s1", 0.018, 0.15), ("mast_s2", 0.015, 0.13), ("mast_s3", 0.012, 0.12))
HEAD_KG = 0.6
MAST_SYSTEM_BUDGET_KG = 2.4   # OS 2.3 mass budget: whole mast system with ring
MAST_MOVING_BUDGET_KG = 0.8   # OS 2.3: moving part (sections + head at its 0.4 kg target)
RACK_STRUCT_KG = 0.2
BASKET_KG = 3.0

# Tools (OS 2.4; geometry and TCP are assumptions until WP2 lands). TCP in the arm_wrist frame.
TOOLS = {
    "T1": {"mass": 0.0, "tcp": [0.105, 0.0, 0.0], "len": 0.0, "r": 0.0, "jaw": 0.0},
    "T2": {"mass": 0.28, "tcp": [0.20, 0.0, 0.0], "len": 0.12, "r": 0.025, "jaw": 0.0},
    "T4": {"mass": 0.35, "tcp": [0.20, 0.0, 0.0], "len": 0.12, "r": 0.03, "jaw": 0.16},
    "T5": {"mass": 0.30, "tcp": [0.21, 0.0, 0.0], "len": 0.13, "r": 0.02, "jaw": 0.0},
    "CAM": {"mass": 0.0, "tcp": [0.02, 0.0, 0.03], "len": 0.0, "r": 0.0, "jaw": 0.0},  # wrist_cam
}
FLANGE_X = 0.09
RACK_TCP = [FLANGE_X, 0.0, 0.0]


# ---------------------------------------------------------------------------------------------- helpers
def run_meta(kind: str) -> dict:
    # NANDI_GIT_SHA: set by the caller when .git is not visible (container runs of a git worktree)
    sha = os.environ.get("NANDI_GIT_SHA", "").strip()
    if not sha:
        try:
            sha = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
        except Exception:
            sha = ""
    sha = sha or "unknown"
    now = datetime.datetime.now(datetime.timezone.utc)
    return {
        "run_id": f"wp0c-{kind}-{now.strftime('%Y%m%dT%H%M%SZ')}-{sha}",
        "device_id": os.environ.get("NANDI_DEVICE_ID", f"capped-runner-{platform.machine()}-2cpu-2gb"),
        "git_sha": sha, "mujoco": mujoco.__version__, "utc": now.isoformat(timespec="seconds"),
    }


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def rotx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def twist_rolls_deg(direction: str) -> Tuple[float, ...]:
    """The jaw rolls (deg, about the tool axis) checked for one twist direction."""
    return tuple(TWIST_DIRS[direction] * r + 0.0 for r in TWIST_CHECK_ROLLS_DEG)   # + 0.0: no -0.0


def rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def load_json(path: str = CANDIDATES_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(doc: dict, path: str = CANDIDATES_JSON):
    """Atomic write: a sibling temp file, fsync, then os.replace (a run killed mid-write leaves the old file)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=1, sort_keys=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


JSON_LOCK_STALE_S = 600.0     # a lock older than this (a killed writer) is broken


@contextlib.contextmanager
def json_lock(path: str = CANDIDATES_JSON, timeout_s: float = 900.0):
    """Cross-process lock for the read-modify-write of the shared JSON. mkdir is atomic on every filesystem,
    including a bind mount shared by several runner containers (flock is not reliable there)."""
    lock = os.path.join(os.path.dirname(path) or ".", "." + os.path.basename(path) + ".lock")
    t0 = time.time()
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > JSON_LOCK_STALE_S:
                    os.rmdir(lock)
                    continue
            except FileNotFoundError:
                continue
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"{lock} held for > {timeout_s:g} s")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            os.rmdir(lock)
        except FileNotFoundError:
            pass


def candidates_digest(cand: dict) -> str:
    return hashlib.sha256(json.dumps(cand, sort_keys=True).encode()).hexdigest()[:16]


def block_inputs(doc: dict, kind: str) -> dict:
    """What a measured block was computed from: the candidates digest (sweep) or the frozen run id (load, creep,
    matrix, which take the rack and the worst T4 pose from `frozen`)."""
    if kind == "frozen":
        return {"candidates_digest": candidates_digest(doc.get("candidates") or {})}
    return {"frozen_run_id": ((doc.get("frozen") or {}).get("meta") or {}).get("run_id")}


def stale_blocks(doc: dict) -> List[str]:
    """Measured blocks whose recorded inputs no longer match the file (regenerated candidates, re-run sweep)."""
    out = []
    for k in ("frozen", "load", "creep", "matrix"):
        b = doc.get(k)
        if b is not None and b.get("inputs") != block_inputs(doc, k):
            out.append(k)
    return out


def update_json(updates: Dict[str, object], path: str = CANDIDATES_JSON) -> dict:
    """Locked read-modify-write: re-read the file under the lock, set the given top-level keys, refresh the
    `stale_blocks` marker, write atomically. Concurrent sub-commands keep each other's blocks."""
    with json_lock(path):
        doc = load_json(path) if os.path.exists(path) else {}
        doc.update(updates)
        doc["stale_blocks"] = stale_blocks(doc)
        save_json(doc, path)
    if doc["stale_blocks"]:
        print(f"WARNING: stale blocks in {os.path.basename(path)}: {doc['stale_blocks']} (re-run them)",
              file=sys.stderr)
    return doc

def r3(v, nd=4):
    return [round(float(x), nd) for x in v]


def _id(m, t, name):
    return mujoco.mj_name2id(m, t, name)


GEOM_DIST_BISECT_STEPS = 14   # 0.25 m / 2^14 = 0.015 mm


def geom_distance(m, d, a: int, b: int, cap: float, ft: np.ndarray) -> float:
    """mj_geomDistance with a guard: on MuJoCo 3.3.7 a box-box pair inside distmax sometimes returns exactly 0.0
    while fromto holds two points that are not the closest pair (seen for finger pads vs the trunk box and vs a
    leaf quad). A genuine penetration is negative and a genuine touch has coincident points. An exact 0.0 with
    separated points is resolved by bisecting distmax: a query that returns distmax proves the distance is at
    least distmax, a non-zero value below it is taken as the distance, another 0.0 lowers the bracket. The result
    is the proven lower bound (never the fromto length, which overstated clearances by up to 7 mm)."""
    v = mujoco.mj_geomDistance(m, d, a, b, cap, ft)
    if v != 0.0:
        return v
    sep = float(np.linalg.norm(ft[3:6] - ft[0:3]))
    if sep <= 1e-6:
        return v
    probe = np.zeros(6)
    lo, hi = 0.0, min(sep, cap)
    for _ in range(GEOM_DIST_BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        r = mujoco.mj_geomDistance(m, d, a, b, mid, probe)
        if r >= mid - 1e-12:
            lo = mid
        elif r != 0.0:
            ft[:] = probe
            return r
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------------------------- full layout
def build_layout(ext: float = 0.0, mast_xy: Sequence[float] = MAST_XY, head_kg: float = HEAD_KG,
                 mounted: Optional[str] = None, fruit_kg: float = 0.0, basket_kg: float = BASKET_KG,
                 cradles: Optional[List[dict]] = None, racked: Sequence[str] = ("T2", "T4", "T5"),
                 kinematic_tools: bool = False, obstacles: int = 0) -> mujoco.MjModel:
    """Shipped robot + WP0c full layout. All additions contype 0 / conaffinity 0 and joint-free.

    kinematic_tools: put every tool's collision proxy on arm_wrist with negligible mass (sweep model).
    obstacles: number of pooled world capsules/spheres/boxes for per-station tree geometry (sweep model).
    """
    spec = mujoco.MjSpec.from_file(ROBOT_XML)
    for n in ("ridge_1", "ridge_2"):
        g = spec.geom(n)
        if g is not None:
            spec.delete(g)
    # Park the loose crop far away (its qpos slot stays, so the frozen layout is unchanged).
    crop = spec.body("crop_item_1")
    crop.pos = [-4.0, 4.0, 0.035]
    key = spec.keys[0]
    q = np.asarray(key.qpos, dtype=np.float64).copy()
    q[32:35] = [-4.0, 4.0, 0.035]
    key.qpos = q.tolist()
    nc = dict(contype=0, conaffinity=0, group=5)
    trunk = spec.body("trunk")
    # Sectioned crown mast.
    mx, my = float(mast_xy[0]), float(mast_xy[1])
    mast = trunk.add_body(name="wp0c_mast", pos=[mx, my, MAST_BASE_Z])
    mast.add_geom(name="mast_drive", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0.03], size=[0.04, 0.04, 0.03],
                  mass=MAST_FIXED[0][1], rgba=[0.3, 0.3, 0.3, 1], **nc)
    mast.add_geom(name="mast_outer", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0.06, 0, 0, MAST_RING_Z],
                  size=[0.02, 0, 0], mass=MAST_FIXED[1][1], rgba=[0.5, 0.5, 0.5, 1], **nc)
    mast.add_geom(name="mast_ring", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, MAST_RING_Z],
                  size=[0.06, 0.04, 0], mass=MAST_FIXED[2][1], rgba=[0.1, 0.1, 0.1, 1], **nc)
    mast.add_geom(name="mast_keepout", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=[0, 0, 0.0, 0, 0, MAST_HEAD_LOCAL_Z + MAST_STROKE + 0.1], size=[KEEP_OUT_R, 0, 0],
                  mass=0.0, rgba=[1, 0, 0, 0.0], contype=0, conaffinity=0, group=5)
    parent = mast
    for name, r, kg in MAST_SECTIONS:
        parent = parent.add_body(name=name, pos=[0, 0, ext / 3.0])
        parent.add_geom(name=name + "_g", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0.10, 0, 0, 0.70],
                        size=[r, 0, 0], mass=kg, rgba=[0.7, 0.7, 0.7, 1], **nc)
    head = parent.add_body(name="crown_head", pos=[0, 0, MAST_HEAD_LOCAL_Z])
    head.add_geom(name="crown_head_g", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.06, 0.06],
                  mass=max(1e-4, head_kg), rgba=[0.9, 0.2, 0.2, 1], **nc)
    head.add_site(name="crown_ref", pos=[0, 0, 0])
    # Rack: cradles + racked tools hanging below each cradle point.
    for i, c in enumerate(cradles or []):
        p = c["pos"]
        trunk.add_geom(name=f"rack_cradle_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[p[0], p[1] - 0.01, p[2] - 0.02],
                       size=[0.035, 0.012, 0.02], mass=RACK_STRUCT_KG / max(1, len(cradles)),
                       rgba=[0.2, 0.2, 0.25, 1], **nc)
        tname = c.get("tool")
        if tname in racked and tname != mounted and tname in TOOLS:
            t = TOOLS[tname]
            trunk.add_geom(name=f"racked_{tname}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                           fromto=[p[0], p[1], p[2] - 0.01, p[0], p[1], p[2] - t["len"]],
                           size=[max(t["r"], 0.02), 0, 0], mass=t["mass"], rgba=[0.8, 0.6, 0.1, 1], **nc)
    # Basket payload (3 kg default) at site basket_payload.
    if basket_kg > 0:
        b = spec.body("dorsal_basket")
        b.add_geom(name="wp0c_payload", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0.041 + 0.03],
                   size=[0.15, 0.10, 0.03], mass=basket_kg, rgba=[0.55, 0.35, 0.15, 0.6], **nc)
    # Tools on the wrist.
    wrist = spec.body("arm_wrist")
    names = [k for k in ("T2", "T4", "T5") if kinematic_tools or k == mounted]
    for k in names:
        t = TOOLS[k]
        kg = 1e-5 if kinematic_tools else t["mass"]
        x0, x1 = FLANGE_X + 0.01, t["tcp"][0] - (0.04 if k == "T4" else 0.02)
        wrist.add_geom(name=f"tool_{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[x0, 0, 0, x1, 0, 0],
                       size=[t["r"], 0, 0], mass=kg, rgba=[0.9, 0.7, 0.1, 1], **nc)
        if t["jaw"] > 0:   # two jaw fingers at full opening (+ pad), straddling the TCP
            half = t["jaw"] / 2.0 + 0.012
            for s, side in ((1, "l"), (-1, "r")):
                wrist.add_geom(name=f"tool_{k}_jaw_{side}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                               fromto=[x1, s * half, 0, t["tcp"][0] + 0.05, s * half, 0], size=[0.01, 0, 0],
                               mass=1e-5, rgba=[0.2, 0.2, 0.2, 1], **nc)
                wrist.add_geom(name=f"tool_{k}_bridge_{side}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                               fromto=[x1, 0, 0, x1, s * half, 0], size=[0.01, 0, 0], mass=1e-5,
                               rgba=[0.2, 0.2, 0.2, 1], **nc)
        wrist.add_site(name=f"tcp_{k}", pos=list(t["tcp"]))
    if mounted and fruit_kg > 0:
        wrist.add_geom(name="wp0c_fruit", type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=list(TOOLS[mounted]["tcp"]),
                       size=[0.06, 0, 0], mass=fruit_kg, rgba=[0.95, 0.55, 0.1, 1], **nc)
    # Pooled world obstacles for the kinematic sweep.
    for i in range(obstacles):
        spec.worldbody.add_geom(name=f"obs_cap_{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, pos=[0, 0, -10],
                                size=[0.01, 0.01, 0], **nc)
        spec.worldbody.add_geom(name=f"obs_sph_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0, 0, -10],
                                size=[0.01, 0, 0], **nc)
    if obstacles:
        spec.worldbody.add_geom(name="obs_leaf", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -10],
                                size=[0.04, 0.025, 0.002], **nc)
        for i in range(4):
            spec.worldbody.add_geom(name=f"obs_box_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -10],
                                    size=[0.01, 0.01, 0.01], **nc)
    m = spec.compile()
    # The pooled obstacles are compiled with an identity quat, so MuJoCo marks them geom_sameframe = BODYROT and
    # mj_kinematics then copies the world rotation into geom_xmat, ignoring the geom_quat the sweep writes at run
    # time: every oblique wood capsule would stand vertical and every leaf quad would lie flat. Force the full
    # local-to-global transform for them.
    for g in range(m.ngeom):
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("obs_"):
            m.geom_sameframe[g] = 0
    return m


def settle(m: mujoco.MjModel, seconds: float = 1.0, arm_q: Optional[Sequence[float]] = None,
           ff: bool = True) -> mujoco.MjData:
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = m.key_ctrl[0]
    if arm_q is not None:
        d.qpos[7:12] = arm_q
        d.ctrl[0:5] = arm_q
        if ff:
            d.ctrl[0:5] = np.asarray(arm_q) + gravity_offsets(m, d)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return d


def gravity_offsets(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    """tau_g / kp for the 5 arm actuators at the current qpos (static), incl. any mass on arm_wrist."""
    d2 = mujoco.MjData(m)
    d2.qpos[:] = d.qpos
    mujoco.mj_forward(m, d2)
    bias = d2.qfrc_bias.copy()
    tau = np.array([bias[m.jnt_dofadr[m.actuator_trnid[a, 0]]] for a in range(5)])
    kp = m.actuator_gainprm[0:5, 0]
    return tau / kp


MARGIN_THRESH = {"robot": ROBOT_CLEAR_M, "wood": WOOD_CLEAR_M, "ground": GROUND_CLEAR_M, "keepout": 0.0,
                 "jaw": JAW_CLEAR_M, "leaf": LEAF_CLEAR_M}


def margin_of(chk: dict) -> float:
    """Smallest clearance above its threshold (carriage self-clearance is enforced by the IK limits)."""
    return min(chk[k] - v for k, v in MARGIN_THRESH.items() if k in chk)


# ---------------------------------------------------------------------------------------------- kinematic sweep
class ArmSweep:
    """Closed-form arm IK (all tool angles in the arm plane) + MuJoCo FK and distance verification."""

    ARM_BODIES = ("arm_upper", "arm_forearm", "arm_wrist", "finger_left", "finger_right")

    def __init__(self, cradles: Optional[List[dict]] = None, mast_xy: Sequence[float] = MAST_XY):
        from atharv_nandi.core.arm_ik_solver import DorsalArmIKSolver
        self.m = build_layout(cradles=cradles, mast_xy=mast_xy, kinematic_tools=True, obstacles=OBS_N,
                              racked=("T2", "T4", "T5"), mounted=None)
        m = self.m
        d0 = settle(m, 1.0)
        self.q_base = d0.qpos.copy()
        self.d = mujoco.MjData(m)
        self.d.qpos[:] = self.q_base
        mujoco.mj_kinematics(m, self.d)
        tid = _id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.trunk_pos = self.d.xpos[tid].copy()
        self.trunk_R = self.d.xmat[tid].reshape(3, 3).copy()
        self.trunk_yaw = math.atan2(self.trunk_R[1, 0], self.trunk_R[0, 0])
        self.ground_origin = np.array([self.trunk_pos[0], self.trunk_pos[1], 0.0])
        self.R_robot = rotz(self.trunk_yaw)          # robot frame (level) -> sim world
        ik = DorsalArmIKSolver(m)
        self.limits = ik.limits.copy()
        self.l1, self.l2 = ik.l1, ik.l2
        self.shoulder0 = ik.shoulder0.copy()
        self.qadr = ik.qadr.copy()
        self.wrist_id = _id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_wrist")

        def geoms_of(bodies):
            ids = {_id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies}
            return [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids]

        gname = lambda g: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)  # noqa: E731
                           or "~" + (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""))
        self.gname = gname
        # the shipped (collidable) arm geoms: upper arm, forearm, palm, finger pads
        self.arm_core = [g for g in geoms_of(self.ARM_BODIES)
                         if (m.geom_contype[g] or m.geom_conaffinity[g]) and not gname(g).startswith("tool_")]
        self.tool_geoms = {k: [g for g in range(m.ngeom) if gname(g).startswith(f"tool_{k}")] for k in TOOLS}
        self.jaw_geoms = {k: [g for g in v if "_jaw_" in gname(g)] for k, v in self.tool_geoms.items()}
        self._jaw_local = {}                      # geom -> (pos, mat) in the arm_wrist frame (roll about wrist x)
        for g in (g for v in self.jaw_geoms.values() for g in v):
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            self._jaw_local[g] = (m.geom_pos[g].copy(), R.reshape(3, 3))
        self.tcp_site = {k: _id(m, mujoco.mjtObj.mjOBJ_SITE, f"tcp_{k}") for k in ("T2", "T4", "T5")}
        carriage = geoms_of(("arm_carriage", "arm_turret"))
        self.carriage_geoms = carriage
        arm_tree = set(geoms_of(("arm_carriage", "arm_turret") + self.ARM_BODIES))
        trunk_b = _id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        robot_bodies = set()
        for b in range(1, m.nbody):   # robot = trunk subtree minus the arm
            p = b
            while p not in (0, trunk_b):
                p = m.body_parentid[p]
            if p == trunk_b:
                robot_bodies.add(b)
        robot_bodies |= {trunk_b}
        self.robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in robot_bodies and g not in arm_tree
                            and gname(g) != "mast_keepout" and not gname(g).startswith("tool_")
                            and (m.geom_contype[g] or gname(g).startswith(("mast_", "crown_", "rack_", "racked_")))]
        self.rack_geoms = {gname(g): g for g in self.robot_geoms if gname(g).startswith(("rack_", "racked_"))}
        self.keepout = _id(m, mujoco.mjtObj.mjOBJ_GEOM, "mast_keepout")
        self.floor = _id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.obs_cap = [_id(m, mujoco.mjtObj.mjOBJ_GEOM, f"obs_cap_{i}") for i in range(OBS_N)]
        self.obs_sph = [_id(m, mujoco.mjtObj.mjOBJ_GEOM, f"obs_sph_{i}") for i in range(OBS_N)]
        self.obs_leaf = _id(m, mujoco.mjtObj.mjOBJ_GEOM, "obs_leaf")
        self.obs_box = [_id(m, mujoco.mjtObj.mjOBJ_GEOM, f"obs_box_{i}") for i in range(4)]
        self._ft = np.zeros(6)
        self.fail_stats: Dict[str, int] = {}
        self.clear_obstacles()

    # -------------------------------------------------------------- frames
    def robot_to_sim(self, p_r) -> np.ndarray:
        return self.ground_origin + self.R_robot @ np.asarray(p_r, dtype=np.float64)

    def robot_dir_to_trunk(self, v_r) -> np.ndarray:
        return self.trunk_R.T @ (self.R_robot @ np.asarray(v_r, dtype=np.float64))

    def sim_to_trunk(self, p_sim) -> np.ndarray:
        return self.trunk_R.T @ (np.asarray(p_sim, dtype=np.float64) - self.trunk_pos)

    # -------------------------------------------------------------- obstacles
    def clear_obstacles(self):
        m = self.m
        for g in self.obs_cap + self.obs_sph + self.obs_box + [self.obs_leaf]:
            m.geom_pos[g] = [0, 0, -10.0]
        self.active_obs: Dict[str, List[Tuple[int, float, str]]] = {"wood": [], "jaw": [], "leaf": [], "box": []}

    def add_capsule(self, a_r, b_r, r, kind="wood", label="wood"):
        i = len([1 for k in self.active_obs.values() for o in k if o[0] in self.obs_cap])
        g = self.obs_cap[i]
        a, b = self.robot_to_sim(a_r), self.robot_to_sim(b_r)
        c, ax = (a + b) / 2.0, b - a
        half = float(np.linalg.norm(ax)) / 2.0
        q = np.zeros(4)
        mujoco.mju_quatZ2Vec(q, unit(ax) if half > 1e-9 else np.array([0, 0, 1.0]))
        m = self.m
        m.geom_pos[g], m.geom_quat[g] = c, q
        m.geom_size[g] = [r, max(half, 1e-4), 0]
        m.geom_rbound[g] = r + half
        self.active_obs[kind].append((g, 0.0, label))

    def add_sphere(self, c_r, r, kind="wood", label="fruit"):
        i = len([1 for k in self.active_obs.values() for o in k if o[0] in self.obs_sph])
        g = self.obs_sph[i]
        m = self.m
        m.geom_pos[g] = self.robot_to_sim(c_r)
        m.geom_size[g] = [r, 0, 0]
        m.geom_rbound[g] = r
        self.active_obs[kind].append((g, 0.0, label))

    def add_box(self, c_r, half, label="crate"):
        i = len(self.active_obs["box"])
        g = self.obs_box[i]
        m = self.m
        m.geom_pos[g] = self.robot_to_sim(c_r)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.R_robot.flatten())
        m.geom_quat[g] = q
        m.geom_size[g] = half
        m.geom_rbound[g] = float(np.linalg.norm(half))
        self.active_obs["box"].append((g, 0.0, label))

    def set_leaf(self, c_r, n_r):
        m, g = self.m, self.obs_leaf
        m.geom_pos[g] = self.robot_to_sim(c_r)
        q = np.zeros(4)
        mujoco.mju_quatZ2Vec(q, self.R_robot @ unit(n_r))
        m.geom_quat[g] = q
        m.geom_rbound[g] = 0.05
        self.active_obs["leaf"] = [(g, 0.0, "leaf")]

    # -------------------------------------------------------------- closed-form IK
    def candidates(self, p_r, tcp, cone_axis_r, cone_deg, gantry=None, max_n=60, coarse=False):
        """All closed-form arm solutions putting the TCP at p_r (robot frame) with the tool axis inside the
        cone. Returns an array (n, 5) of joint vectors sorted by a centring cost, and the tool axes."""
        p = self.sim_to_trunk(self.robot_to_sim(p_r))
        c = unit(self.robot_dir_to_trunk(cone_axis_r))
        cos_c = math.cos(math.radians(cone_deg))
        lx, lz = float(tcp[0]), float(tcp[2])
        G = np.asarray([gantry] if gantry is not None else (GANTRY_GRID[::3] if coarse else GANTRY_GRID),
                       dtype=np.float64)
        phis = PHI_GRID[::2] if coarse else PHI_GRID
        S = self.shoulder0[None, :] + np.stack([G, np.zeros_like(G), np.zeros_like(G)], axis=1)
        dxy = p[None, :2] - S[:, :2]
        hh = np.linalg.norm(dxy, axis=1)
        u = dxy / np.maximum(hh, 1e-9)[:, None]
        out_q, out_a = [], []
        for f in (1.0, -1.0):
            uf = u * f
            yaw = np.arctan2(uf[:, 1], uf[:, 0])
            ht = hh * f
            zt = p[2] - S[:, 2]
            ph = phis[None, :]
            ax = np.cos(ph)[..., None] * np.concatenate([uf, np.zeros((len(G), 1))], axis=1)[:, None, :]
            ax[..., 2] = -np.sin(ph)
            ok_cone = (ax @ c) >= cos_c                                     # (nG, nPhi)
            wh = ht[:, None] - lx * np.cos(ph) - lz * np.sin(ph)
            wz = zt[:, None] + lx * np.sin(ph) - lz * np.cos(ph)
            dd = np.hypot(wh, wz)
            ok_d = (dd > abs(self.l1 - self.l2) + 1e-6) & (dd < self.l1 + self.l2 - 1e-6)
            cos_t = np.clip((self.l1 ** 2 + dd ** 2 - self.l2 ** 2) / (2.0 * self.l1 * np.maximum(dd, 1e-9)), -1, 1)
            for b in (1.0, -1.0):
                a1 = np.arctan2(wz, wh) + b * np.arccos(cos_t)
                eh, ez = self.l1 * np.cos(a1), self.l1 * np.sin(a1)
                a2 = np.arctan2(wz - ez, wh - eh)
                qs = wrap(np.pi / 2.0 - a1)
                qe = wrap(-a2 - qs)
                qw = wrap(ph + a2)
                qy = np.broadcast_to(yaw[:, None], qs.shape)
                qg = np.broadcast_to(G[:, None], qs.shape)
                Q = np.stack([qg, qy, qs, qe, qw], axis=-1)
                lim = self.limits
                ok = ok_cone & ok_d
                for j in range(5):
                    ok &= (Q[..., j] >= lim[j, 0] - 1e-9) & (Q[..., j] <= lim[j, 1] + 1e-9)
                if np.any(ok):
                    out_q.append(Q[ok])
                    out_a.append(ax[ok])
        if not out_q:
            return np.zeros((0, 5)), np.zeros((0, 3))
        Q, A = np.concatenate(out_q), np.concatenate(out_a)
        # cost: joints away from their limits, gantry near 0, tool axis near the cone centre
        mid = self.limits.mean(axis=1)
        span = np.diff(self.limits, axis=1)[:, 0]
        cost = np.sum(((Q - mid) / span) ** 2, axis=1) + 1.5 * (1.0 - A @ c)
        order = np.argsort(cost)
        idx = order[:: max(1, len(order) // max_n)][:max_n] if len(order) > max_n else order
        return Q[idx], A[idx]

    # -------------------------------------------------------------- FK + distances
    def fk(self, q5):
        d = self.d
        d.qpos[:] = self.q_base
        d.qpos[self.qadr] = q5
        mujoco.mj_kinematics(self.m, d)

    def tcp_of(self, tcp) -> Tuple[np.ndarray, np.ndarray]:
        R = self.d.xmat[self.wrist_id].reshape(3, 3)
        return self.d.xpos[self.wrist_id] + R @ np.asarray(tcp, dtype=np.float64), R[:, 0].copy()

    def dist(self, ga: Sequence[int], gb: Sequence[int], cap=0.25) -> Tuple[float, Tuple[int, int]]:
        best, pair = cap, (-1, -1)
        for a in ga:
            for b in gb:
                v = geom_distance(self.m, self.d, a, b, cap, self._ft)
                if v < best:
                    best, pair = v, (a, b)
        return best, pair

    def jaw_dist_rolled(self, tool: str, obs: Sequence[int], roll_deg: float) -> float:
        """Jaw clearance to obs with the jaws rolled roll_deg about the tool axis (wrist x) at the current FK.
        Overrides the jaw geoms' world frames in the shared MjData, measures, then restores them."""
        jaws = self.jaw_geoms.get(tool, [])
        if not jaws or not obs:
            return 0.25
        d = self.d
        Rw = d.xmat[self.wrist_id].reshape(3, 3)
        pw = d.xpos[self.wrist_id]
        Rr = Rw @ rotx(math.radians(roll_deg))
        saved = [(d.geom_xpos[g].copy(), d.geom_xmat[g].copy()) for g in jaws]
        try:
            for g in jaws:
                p, R = self._jaw_local[g]
                d.geom_xpos[g] = pw + Rr @ p
                d.geom_xmat[g] = (Rr @ R).ravel()
            return self.dist(jaws, obs)[0]
        finally:
            for g, (p, R) in zip(jaws, saved):
                d.geom_xpos[g], d.geom_xmat[g] = p, R

    def jaw_twist(self, tool: str, obs: Sequence[int], dirs: Sequence[str] = tuple(TWIST_DIRS)) -> dict:
        """Jaw clearance at every checked roll of each twist direction: {by_roll_deg, min_by_dir, dirs_ok}."""
        rolls = sorted({r for k in dirs for r in twist_rolls_deg(k)})
        by = {r: self.jaw_dist_rolled(tool, obs, r) for r in rolls}
        mins = {k: min(by[r] for r in twist_rolls_deg(k)) for k in dirs}
        return {"by_roll_deg": by, "min_by_dir": mins, "dirs_ok": [k for k in dirs if mins[k] >= JAW_CLEAR_M]}

    def check_pose(self, q5, tool: str, jaw_check: bool = False, leaf_check: bool = False,
                   exclude_robot: Sequence[int] = (), twist_dirs: Sequence[str] = tuple(TWIST_DIRS)) -> dict:
        """Clearances of the arm (+ tool) at q5. Pass = every clearance >= its threshold.

        jaw_check (T4): the jaws keep JAW_CLEAR_M from wood and neighbouring fruit at every roll of the twist
        stroke (TWIST_CHECK_ROLLS_DEG) for at least one of twist_dirs. res["jaw"] = the smallest clearance over
        the stroke of the best direction, res["jaw_roll0"] = the clearance with the jaws level, res["twist"] =
        jaw_twist() (non-float, so it stays out of the clearance dict)."""
        self.fk(q5)
        arm = list(self.arm_core) + list(self.tool_geoms.get(tool, []))
        robot = [g for g in self.robot_geoms if g not in exclude_robot]
        r_robot, pr = self.dist(arm, robot)
        upper = [g for g in self.arm_core if self.m.geom_bodyid[g] == _id(self.m, mujoco.mjtObj.mjOBJ_BODY, "arm_upper")]
        r_carr, _ = self.dist(upper, [g for g in self.carriage_geoms
                                      if self.m.geom_bodyid[g] == _id(self.m, mujoco.mjtObj.mjOBJ_BODY, "arm_carriage")])
        r_keep, _ = self.dist(arm + list(self.carriage_geoms), [self.keepout], cap=0.3)
        r_floor, _ = self.dist(arm, [self.floor])
        wood = [o[0] for o in self.active_obs["wood"]] + [o[0] for o in self.active_obs["box"]]
        r_wood, pw = self.dist(arm, wood) if wood else (0.25, (-1, -1))
        res = {"robot": r_robot, "carriage": r_carr, "keepout": r_keep, "ground": r_floor, "wood": r_wood,
               "robot_pair": [self.gname(pr[0]), self.gname(pr[1])] if pr[0] >= 0 else None,
               "wood_pair": [self.gname(pw[0]), self.gname(pw[1])] if pw[0] >= 0 else None}
        ok = (r_robot >= ROBOT_CLEAR_M and r_carr >= 0.0 and r_keep >= 0.0 and r_floor >= GROUND_CLEAR_M
              and r_wood >= WOOD_CLEAR_M)
        if jaw_check:
            jaw_obs = wood + [o[0] for o in self.active_obs["jaw"]]
            tw = self.jaw_twist(tool, jaw_obs, twist_dirs)
            r_jaw = max(tw["min_by_dir"].values()) if tw["min_by_dir"] else 0.25
            res["jaw"] = r_jaw
            res["jaw_roll0"] = tw["by_roll_deg"].get(0.0, 0.25)
            res["twist"] = tw
            if ok and r_jaw < JAW_CLEAR_M:
                self.fail_stats["jaw_only"] = self.fail_stats.get("jaw_only", 0) + 1
            ok = ok and r_jaw >= JAW_CLEAR_M
        if leaf_check and self.active_obs["leaf"]:
            r_leaf, _ = self.dist(arm, [self.obs_leaf])
            res["leaf"] = r_leaf
            ok = ok and r_leaf >= LEAF_CLEAR_M
        res["ok"] = bool(ok)
        return res

    def solve(self, p_r, tool, cone_axis_r, cone_deg, exit_r=None, jaw_check=False, leaf_check=False,
              exclude_robot=(), max_verify=24, exit_cone_slack_deg=12.0, tcp=None, enough=3) -> Optional[dict]:
        """Best verified pose (max min-clearance) for a TCP target, incl. the exit path when exit_r is given.

        exit_r: a robot-frame vector, or "retreat" = back out 5 cm along the actual tool axis.
        jaw_check: the twist direction has to hold at the target and at every exit step (the jaws stay at
        the release roll on the retreat); the record carries `twist` with the surviving directions."""
        tcp = tcp if tcp is not None else (TOOLS[tool]["tcp"] if tool in TOOLS else RACK_TCP)
        Q, A = self.candidates(p_r, tcp, cone_axis_r, cone_deg, max_n=max_verify)
        best, n_ok = None, 0
        target_sim = self.robot_to_sim(p_r)
        for q5, a in zip(Q, A):
            chk = self.check_pose(q5, tool, jaw_check, leaf_check, exclude_robot)
            if not chk["ok"]:
                continue
            tpos, taxis = self.tcp_of(tcp)
            if np.linalg.norm(tpos - target_sim) > 0.002:
                continue
            axis_r = self.R_robot.T @ taxis
            dirs = list(chk["twist"]["dirs_ok"]) if jaw_check else list(TWIST_DIRS)
            ex = None
            if exit_r is not None:
                e = -axis_r if isinstance(exit_r, str) else np.asarray(exit_r, dtype=np.float64)
                ex = self._exit_path(p_r, tool, q5, e, axis_r, exit_cone_slack_deg, jaw_check, exclude_robot, tcp,
                                     dirs)
                if ex is None:
                    continue
                if jaw_check:
                    dirs = ex["twist_dirs"]
            if jaw_check:                   # the target's jaw clearance over the directions that survived the exit
                chk["jaw"] = max(chk["twist"]["min_by_dir"][k] for k in dirs)
            clear = min(v for k, v in chk.items() if isinstance(v, float) and k != "carriage")
            marg = margin_of(chk)
            n_ok += 1
            if ex is not None:
                marg = min(marg, ex["margin"])
            if best is None or marg > best["margin"]:
                best = {"q": r3(q5), "axis_r": r3(axis_r), "min_clear": round(clear, 4), "margin": round(marg, 4),
                        "clear": {k: round(v, 4) for k, v in chk.items() if isinstance(v, float)},
                        "exit": ex}
                if jaw_check:
                    tw = chk["twist"]
                    best["twist"] = {
                        "dirs_ok": dirs,
                        "preferred": max(dirs, key=lambda k: tw["min_by_dir"][k]),
                        "rolls_deg": {k: list(twist_rolls_deg(k)) for k in TWIST_DIRS},
                        "jaw_by_roll_deg": {f"{r:g}": round(v, 4) for r, v in sorted(tw["by_roll_deg"].items())},
                        "jaw_min_by_dir": {k: round(v, 4) for k, v in tw["min_by_dir"].items()},
                        "exit_jaw_min_by_dir": ex["jaw_min_by_dir"] if ex else None}
            if n_ok >= enough:
                break
        return best

    def _exit_path(self, p_r, tool, q0, exit_r, cone_axis_r, cone_deg, jaw_check, exclude_robot, tcp,
                   twist_dirs: Sequence[str] = tuple(TWIST_DIRS)):
        e = unit(exit_r)
        q_prev = np.asarray(q0, dtype=np.float64)
        dirs = list(twist_dirs)
        steps = []
        for k in range(1, EXIT_STEPS + 1):
            p = np.asarray(p_r) + e * EXIT_M * k / EXIT_STEPS
            Q, _A = self.candidates(p, tcp, cone_axis_r, cone_deg, gantry=float(q_prev[0]), max_n=400)
            if len(Q) == 0:
                return None
            dq = np.abs(wrap(Q - q_prev[None, :]))
            j = int(np.argmin(dq.max(axis=1)))
            if dq[j].max() > 0.35:      # no branch flip within one 1 cm step
                return None
            chk = self.check_pose(Q[j], tool, jaw_check, False, exclude_robot, twist_dirs=dirs)
            if not chk["ok"]:
                return None
            if jaw_check:
                dirs = list(chk["twist"]["dirs_ok"])
            steps.append(chk)
            q_prev = Q[j]
        worst = 1.0
        jaw_min = {k: 1.0 for k in dirs}
        for chk in steps:               # margins over the directions that hold on the whole exit
            if jaw_check:
                for k in dirs:
                    jaw_min[k] = min(jaw_min[k], chk["twist"]["min_by_dir"][k])
                chk["jaw"] = max(chk["twist"]["min_by_dir"][k] for k in dirs)
            worst = min(worst, margin_of(chk))
        out = {"q_end": r3(q_prev), "margin": round(worst, 4)}
        if jaw_check:
            out["twist_dirs"] = dirs
            out["jaw_min_by_dir"] = {k: round(v, 4) for k, v in jaw_min.items()}
        return out

    # -------------------------------------------------------------- reach map
    def _reachable(self, p, tool) -> bool:
        Q, _A = self.candidates(p, TOOLS[tool]["tcp"], [0, 0, 1.0], 180.0, max_n=30, coarse=True)
        for q5 in Q:
            c = self.check_pose(q5, tool)
            if c["robot"] >= 0.0 and c["carriage"] >= 0.0 and c["ground"] >= 0.0:
                return True
        return False

    def reach(self, azimuth_rad: float, z: float, tool: str, rho_max=1.3) -> float:
        """Measured reach: largest horizontal distance from the trunk centre axis (robot frame) along the
        azimuth at height z where a collision-free pose exists (any tool angle). Coarse 5 cm steps inward
        from rho_max, then 1 cm refinement outward."""
        u = np.array([math.cos(azimuth_rad), math.sin(azimuth_rad), 0.0])
        pt = lambda r: u * r + np.array([0.0, 0.0, z])  # noqa: E731
        rho = rho_max
        while rho > 0.05 and not self._reachable(pt(rho), tool):
            rho -= 0.05
        if rho <= 0.05:
            return 0.0
        while rho + 0.01 <= rho_max + 1e-9 and self._reachable(pt(rho + 0.01), tool):
            rho += 0.01
        return round(rho, 3)


# ---------------------------------------------------------------------------------------------- reach map
REACH_AZ_DEG = tuple(range(-180, 180, 15))
REACH_Z = tuple(round(0.05 * i, 2) for i in range(1, 17))      # 0.05 .. 0.80


def reach_map(sw: ArmSweep, tool: str) -> dict:
    sw.clear_obstacles()
    table = []
    for az in REACH_AZ_DEG:
        row = []
        prev = 1.3
        for z in REACH_Z:
            row.append(sw.reach(math.radians(az), z, tool, rho_max=min(1.3, prev + 0.12)))
            prev = row[-1] if row[-1] > 0 else 1.3
        table.append(row)
    return {"tool": tool, "az_deg": list(REACH_AZ_DEG), "z": list(REACH_Z), "rho": table}


def reach_lookup(rm: dict, azimuth_rad: float, z: float) -> float:
    """Bilinear interpolation of the reach map (azimuth wraps, z clamps)."""
    az = list(rm["az_deg"])
    zs = list(rm["z"])
    tab = np.asarray(rm["rho"], dtype=np.float64)
    a = (math.degrees(azimuth_rad) - az[0]) % 360.0
    step = az[1] - az[0]
    i0 = int(a // step) % len(az)
    i1 = (i0 + 1) % len(az)
    ta = (a - i0 * step) / step
    zc = min(max(z, zs[0]), zs[-1])
    j0 = min(int((zc - zs[0]) // (zs[1] - zs[0])), len(zs) - 2)
    tz = (zc - zs[j0]) / (zs[1] - zs[0])
    v0 = tab[i0, j0] * (1 - tz) + tab[i0, j0 + 1] * tz
    v1 = tab[i1, j0] * (1 - tz) + tab[i1, j0 + 1] * tz
    return float(v0 * (1 - ta) + v1 * ta)


# ---------------------------------------------------------------------------------------------- candidates
def _tree_wood(t: dict) -> List[dict]:
    """Wood capsules in the tree frame (trunk at the origin): trunk + 3 scaffolds for mango / guava."""
    sp = t["species"]
    wood = [{"a": [0, 0, 0], "b": [0, 0, t["trunk_h"]], "r": t["trunk_r"]}]
    if sp in ("mango", "guava"):
        for k, az in enumerate((30.0, 150.0, 270.0)):
            a = math.radians(az + 17.0 * (t["seed"] % 3))
            z0 = t["trunk_h"]
            L = 0.55 if sp == "mango" else 0.45
            wood.append({"a": [0, 0, z0], "b": [L * 0.7 * math.cos(a), L * 0.7 * math.sin(a), z0 + L * 0.7],
                         "r": 0.035 if sp == "mango" else 0.025})
    return wood


def generate_candidates() -> dict:
    """Deterministic candidate set for orchard_demo (OS 4.2 with the OQ-02 B spacing frozen)."""
    trees = []

    def tree(tid, sp, x, y, H, canopy_r, trunk_r, trunk_h, basin_r, alley_sides, seed, **kw):
        t = {"id": tid, "species": sp, "x": x, "y": y, "H": H, "canopy_r": canopy_r, "trunk_r": trunk_r,
             "trunk_h": trunk_h, "basin_r": basin_r, "alley_sides": alley_sides, "seed": seed}
        t.update(kw)
        t["wood"] = _tree_wood(t)
        trees.append(t)
        return t

    # OT-1 isolated dwarf mango (full orbit), R1 mango 5 x 2.5 m (OQ-02 B), R2 guava, R3 papaya, R4 banana.
    tree("OT-1", "mango", 6.0, -3.5, 2.2, 1.0, 0.06, 0.77, 0.8, ["all"], 1)
    for i, (H, x) in enumerate(zip((1.8, 2.2, 2.5, 2.8), (2.0, 4.5, 7.0, 9.5))):
        tree(f"R1-{i + 1}", "mango", x, 2.5, H, min(0.45 * H, 1.1), 0.05 + 0.005 * i, round(0.35 * H, 3), 0.8,
             ["-y", "+y"], 10 + i)
    for i, x in enumerate((2.0, 3.5, 5.0, 6.5)):
        tree(f"R2-{i + 1}", "guava", x, 7.5, (1.6, 1.9, 2.2, 2.5)[i], 0.7, 0.045, 0.5, 0.6, ["-y", "+y"], 20 + i)
    for i, x in enumerate((2.0, 3.8, 5.6, 7.4)):
        tree(f"R3-{i + 1}", "papaya", x, 11.0, (1.6, 1.9, 2.2, 2.6)[i], 0.9, 0.07, (1.6, 1.9, 2.2, 2.6)[i] - 0.3,
             0.6, ["-y", "+y"], 30 + i)
    for i, x in enumerate((2.0, 3.8, 5.6, 7.4)):
        tree(f"R4-{i + 1}", "banana", x, 14.5, 2.2, 1.4, 0.118, 2.2, 0.6, ["-y", "+y"], 40 + i,
             alley_clear_height_m=1.3)

    targets, leaves = [], []

    def radial_in(az):     # horizontal unit vector pointing from outside toward the trunk axis (tree frame)
        return [-math.cos(az), -math.sin(az), 0.0]

    def tgt(t, task, tool, p, axis, cone, kind_exit="retreat", standoff=(0.0, 0.0), **kw):
        d = {"id": f"{t['id']}:{task}:{sum(1 for x in targets if x['tree'] == t['id'] and x['task'] == task) + 1}",
             "tree": t["id"], "task": task, "tool": tool, "p": r3(p, 3), "axis": r3(axis, 3), "cone_deg": cone,
             "standoff_m": list(standoff), "exit": kind_exit}
        d.update(kw)
        targets.append(d)

    for t in trees:
        sp, rng = t["species"], np.random.default_rng(1000 + t["seed"])
        az_alley = [-90.0] if t["alley_sides"] != ["all"] else [-90.0, 0.0, 90.0, 180.0]
        if sp in ("mango", "guava"):
            rc = t["canopy_r"]
            zs = (0.50, 0.58, 0.66, 0.74, 0.79, 0.95, 1.40) if sp == "mango" else (0.48, 0.56, 0.64, 0.72, 0.78, 1.10)
            for k, z in enumerate(zs):
                az = math.radians(-90.0 + (-50, -20, 10, 40, -35, 0, 25)[k % 7] + float(rng.uniform(-5, 5)))
                if k == 3 and t["id"] in ("R1-2", "R2-2"):
                    az = math.radians(90.0)      # far side of a hedgerow (reached from the other alley)
                rr = rc * (0.80 + 0.12 * ((k * 7) % 3) / 2.0)
                p = [rr * math.cos(az), rr * math.sin(az), z]
                if sp == "mango":
                    tgt(t, "mango_pedicel_cut", "T2", p, radial_in(az), 50.0, fruit_kg=0.25,
                        note="cut_k = stem end + 2.0 cm (pedicel_leave_cm)")
                else:
                    tgt(t, "guava_clip", "T2", p, radial_in(az), 50.0, fruit_kg=0.15)
                    if k in (0, 2, 4):
                        tgt(t, "guava_cradle", "T4", [p[0], p[1], p[2] - 0.05], radial_in(az), 40.0, fruit_kg=0.15)
            # interior fruit, 0.35 canopy_r from the trunk
            az = math.radians(-80.0)
            tgt(t, "mango_pedicel_cut" if sp == "mango" else "guava_clip", "T2",
                [0.35 * rc * math.cos(az), 0.35 * rc * math.sin(az), 0.70], radial_in(az), 50.0, fruit_kg=0.2,
                note="interior fruit")
            # prune: rootstock sprouts (alley side + far side), skirt shoots, dead twig
            for k, (az_d, z) in enumerate(((-70.0, 0.22), (60.0, 0.28))):
                az = math.radians(az_d)
                rr = t["trunk_r"] + 0.05
                tgt(t, "prune_rootstock_sprout", "T5", [rr * math.cos(az), rr * math.sin(az), z],
                    [-math.sin(az), math.cos(az), -0.3], 70.0, stem_mm=6)
            for k, (az_d, z, f) in enumerate(((-110.0, 0.45, 0.7), (-75.0, 0.60, 0.85))):
                az = math.radians(az_d)
                tgt(t, "prune_skirt_shoot", "T5", [f * rc * math.cos(az), f * rc * math.sin(az), z],
                    radial_in(az), 70.0, stem_mm=10)
            # leaves: 12 quads (4 azimuths x young/mature x upper/under), half in band on the alley sides
            for k in range(12):
                az = math.radians((-90.0, -45.0, -135.0, 90.0)[k % 4] + float(rng.uniform(-10, 10)))
                z = (0.55, 0.65, 0.75, 1.10, 0.60, 1.5)[k % 6]
                rr = rc * (0.85 + 0.1 * (k % 2))
                tilt = math.radians(35.0 + 10.0 * (k % 3))        # leaf normal tilted outward from vertical
                n = [math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), math.cos(tilt)]
                leaves.append({"id": f"{t['id']}:leaf:{k + 1}", "tree": t["id"], "p": r3([rr * math.cos(az),
                               rr * math.sin(az), z], 3), "n": r3(n, 3), "kind": ("young", "mature")[k % 2]})
        elif sp == "papaya":
            fruit_r = 0.075
            col = []
            # column around the trunk, spaced so neighbours keep >= 15 mm off the jaw path (except the seeded one)
            for k, (az_d, z) in enumerate(((-90.0, 0.64), (-40.0, 0.80), (-150.0, 0.78), (90.0, 0.70), (-90.0, 0.95))):
                az = math.radians(az_d + float(rng.uniform(-5, 5)))
                rr = t["trunk_r"] + fruit_r + 0.005
                col.append([rr * math.cos(az), rr * math.sin(az), z])
            if t["id"] == "R3-3":   # seeded no-clearance fruit: a neighbour within 15 mm of the jaw path
                az = math.radians(-90.0 + 40.0)
                rr = t["trunk_r"] + fruit_r + 0.005
                col.append([rr * math.cos(az), rr * math.sin(az), col[0][2] + 0.03])
            for k, p in enumerate(col[:5]):
                az = math.atan2(p[1], p[0])
                neigh = [q for j, q in enumerate(col) if j != k]
                tgt(t, "papaya_cradle", "T4", p, radial_in(az), 30.0, fruit_kg=1.8, neighbours=r3l(neigh),
                    neighbour_r=fruit_r)
            # trunk-scan only; papaya has no leaves in the band (n/a: crown-only)
        else:   # banana: bunch MARK only; tidy a dead blade hanging into the band
            for k, (az_d, z) in enumerate(((-100.0, 0.72), (-60.0, 0.78))):
                az = math.radians(az_d)
                rr = 0.55 + 0.1 * k
                tgt(t, "banana_tidy", "T5", [rr * math.cos(az), rr * math.sin(az), z], radial_in(az), 70.0,
                    stem_mm=8)
            az = math.radians(-90.0)
            tgt(t, "banana_bunch", "T4", [0.3 * math.cos(az), 0.3 * math.sin(az), 1.5], radial_in(az), 30.0,
                fruit_kg=26.7, note="MARK only (OQ-03): human cut")
            for k in range(2):     # in-band drooping dead blades are the only banana leaf targets
                az = math.radians(-95.0 + 20 * k)
                n = [0.6 * math.cos(az), 0.6 * math.sin(az), 0.8]
                leaves.append({"id": f"{t['id']}:leaf:{k + 1}", "tree": t["id"],
                               "p": r3([0.75 * math.cos(az), 0.75 * math.sin(az), 0.70 + 0.05 * k], 3),
                               "n": r3(unit(n), 3), "kind": "dead_blade"})
        # trunk / base scan: heights 0.2 / 0.5 / 0.8 on the alley-side azimuths (4 for OT-1)
        for az_d in az_alley:
            az = math.radians(az_d)
            for z in (0.2, 0.5, 0.8):
                surf = [t["trunk_r"] * math.cos(az), t["trunk_r"] * math.sin(az), z]
                tgt(t, "trunk_scan", "CAM", surf, radial_in(az), 50.0, kind_exit=None, standoff=(0.15, 0.25))
    return {"trees": trees, "targets": targets, "leaves": leaves}


def r3l(lst, nd=3):
    return [r3(x, nd) for x in lst]


# ---------------------------------------------------------------------------------------------- stations
def canopy_halfwidth(t: dict, z: float) -> float:
    """Horizontal radius of the tree's foliage envelope at height z (0 outside its vertical extent)."""
    sp = t["species"]
    if sp == "papaya":
        z0, z1 = t["H"] - 0.6, t["H"] + 0.1
    elif sp == "banana":
        z0, z1 = t.get("alley_clear_height_m", 1.3), t["H"] + 0.4
    else:
        z0, z1 = t["trunk_h"] * 0.75, t["H"]
    if not (z0 < z < z1):
        return 0.0
    zc, hz = (z0 + z1) / 2.0, (z1 - z0) / 2.0
    return t["canopy_r"] * math.sqrt(max(0.0, 1.0 - ((z - zc) / hz) ** 2))


FOOT_R = np.array([[0.26, -0.245], [0.26, 0.245], [0.0, -0.255], [0.0, 0.255], [-0.26, -0.245], [-0.26, 0.245]])
BODY_HALF = (0.34, 0.14)
HEAD_STOWED_Z = 1.10


def station_check(st_xy, heading, trees: List[dict], mast_xy=MAST_XY) -> Tuple[bool, str]:
    """Robot standing at st_xy (world) with heading: feet/body >= 10 cm from wood, stowed crown head >= 0.10 m
    outside every foliage envelope."""
    R = rotz(heading)[:2, :2]
    feet = st_xy[None, :] + FOOT_R @ R.T
    head = st_xy + R @ np.asarray(mast_xy)
    for t in trees:
        c = np.array([t["x"], t["y"]])
        if np.linalg.norm(c - st_xy) > 3.5:
            continue
        rt = t["trunk_r"]
        if np.min(np.linalg.norm(feet - c, axis=1)) < rt + 0.03 + STATION_WOOD_CLEAR_M:
            return False, f"feet_wood:{t['id']}"
        # body box vs trunk circle
        loc = R.T @ (c - st_xy)
        dx = max(abs(loc[0]) - BODY_HALF[0], 0.0)
        dy = max(abs(loc[1]) - BODY_HALF[1], 0.0)
        if math.hypot(dx, dy) < rt + STATION_WOOD_CLEAR_M:
            return False, f"body_wood:{t['id']}"
        hw = canopy_halfwidth(t, HEAD_STOWED_Z)
        if hw > 0 and np.linalg.norm(head - c) < hw + 0.06 + CROWN_FOLIAGE_CLEAR_M:
            return False, f"crown_foliage:{t['id']}"
    return True, ""


def station_options(t: dict, p_tree, trees):
    """Robot stations (world xy, heading) placing the target at (px, py) in the robot frame.

    Side stations: tree on the robot's left (the rack side faces away). Face-in stations: robot heading toward
    the trunk (perpendicular to the row in a hedgerow alley), target ahead; the mast then sits 0.30 m further
    from the canopy, which is what lets the stowed head stay outside a mango skirt."""
    p = np.asarray(p_tree, dtype=np.float64)
    pw = np.array([t["x"] + p[0], t["y"] + p[1]])
    az = math.atan2(p[1], p[0]) if np.linalg.norm(p[:2]) > 1e-6 else -math.pi / 2
    if t["alley_sides"] == ["all"]:
        side = az + math.pi / 2.0                # robot left (+y_R) points toward the trunk
        face = az + math.pi
    else:
        side = 0.0 if p[1] <= 0 else math.pi
        face = math.pi / 2.0 if p[1] <= 0 else -math.pi / 2.0
    out = []
    for h, pxs, pys, mode in ((side, STATION_PX, STATION_PY, "side"), (face, FACE_PX, FACE_PY, "face_in")):
        R = rotz(h)[:2, :2]
        for py in pys:
            for px in pxs:
                st = pw - R @ np.array([px, py])
                ok, why = station_check(st, h, trees)
                out.append((st, h, px, py, ok, why, mode))
    return out


def to_robot(t: dict, st, heading, p_tree) -> np.ndarray:
    pw = np.array([t["x"], t["y"], 0.0]) + np.asarray(p_tree, dtype=np.float64)
    R = rotz(heading)
    return R.T @ (pw - np.array([st[0], st[1], 0.0]))


def load_world(sw: ArmSweep, t: dict, st, heading, trees, target=None):
    """Wood of this tree and its neighbours (<= 2.5 m) into the obstacle pool, in the robot frame."""
    sw.clear_obstacles()
    R = rotz(heading)
    for tt in trees:
        if math.hypot(tt["x"] - st[0], tt["y"] - st[1]) > 2.5:
            continue
        for w in tt["wood"]:
            a = R.T @ (np.array([tt["x"], tt["y"], 0.0]) + np.asarray(w["a"]) - np.array([st[0], st[1], 0.0]))
            b = R.T @ (np.array([tt["x"], tt["y"], 0.0]) + np.asarray(w["b"]) - np.array([st[0], st[1], 0.0]))
            sw.add_capsule(a, b, w["r"], label=tt["id"])
    if target and target.get("neighbours"):
        for q in target["neighbours"]:
            sw.add_sphere(to_robot(t, st, heading, q), target.get("neighbour_r", 0.075), kind="jaw", label="neighbour")


# ---------------------------------------------------------------------------------------------- target sweep
def _rot_about(v, axis, ang):
    """Rodrigues rotation of v about a unit axis."""
    v, k = np.asarray(v, dtype=np.float64), unit(axis)
    return v * math.cos(ang) + np.cross(k, v) * math.sin(ang) + k * np.dot(k, v) * (1 - math.cos(ang))


def view_axes(a_pref, elev_deg=(0.0, 20.0, 40.0, -20.0), az_deg=(-30.0, 0.0, 30.0)):
    """Camera viewing axes around a preferred axis: tilted down by elev, swung by az about vertical."""
    a = unit(a_pref)
    side = np.cross([0, 0, 1.0], a)
    side = unit(side) if np.linalg.norm(side) > 1e-6 else np.array([0, 1.0, 0])
    out = []
    for e in elev_deg:
        for s in az_deg:
            b = _rot_about(a, side, math.radians(e))
            out.append(unit(_rot_about(b, [0, 0, 1.0], math.radians(s))))
    return out


def cone_axes(n, half_deg=30.0, k=6):
    """n itself plus k axes tilted half_deg away from n."""
    n = unit(n)
    p = unit(np.cross(n, [1.0, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1.0, 0]))
    out = [n]
    for i in range(k):
        out.append(unit(_rot_about(_rot_about(n, p, math.radians(half_deg)), n, 2 * math.pi * i / k)))
    return out


CAM_CONE_DEG = 10.0


def _placements(t, p_tree, trees):
    opts = station_options(t, p_tree, trees)
    return sorted(opts, key=lambda o: (not o[4], o[6] != "side",
                                       abs(o[3] - 0.42) + 0.5 * abs(o[2] - 0.12) if o[6] == "side" else abs(o[2] - 0.5)))


def sweep_target(sw: ArmSweep, tgt: dict, t: dict, trees, reach_maps) -> dict:
    p_tree = np.asarray(tgt["p"], dtype=np.float64)
    rec = {"id": tgt["id"], "tree": tgt["tree"], "task": tgt["task"], "tool": tgt["tool"], "p": tgt["p"]}
    so = tgt.get("standoff_m", [0.0, 0.0])
    if tgt["task"] == "banana_bunch":
        return {**rec, "status": "MARK", "reason": "human_cut_only (OQ-03)"}
    if p_tree[2] > PICK_BAND_Z:
        return {**rec, "status": "MARK", "reason": "above_pick_band"}
    rm = reach_maps["T1" if tgt["tool"] == "T1" else ("CAM" if tgt["tool"] == "CAM" else "T4")]
    axis_tree = np.asarray(tgt["axis"], dtype=np.float64)
    is_cam = tgt["tool"] == "CAM"
    best, why_fail = None, {}
    sw.fail_stats = {}
    standoffs = [0.0] if so[1] <= 0 else list(np.round(np.linspace(so[0], so[1], 3), 3))
    for st, h, px, py, ok, why, mode in _placements(t, p_tree, trees):
        if not ok:
            k = why.split(":")[0]
            why_fail[k] = why_fail.get(k, 0) + 1
            continue
        R = rotz(h)
        load_world(sw, t, st, h, trees, tgt)
        p_r0 = to_robot(t, st, h, p_tree)
        axes = view_axes(R.T @ axis_tree) if is_cam else [R.T @ axis_tree]
        for a_r in axes:
            for s in standoffs:
                p_r = p_r0 - a_r * s        # camera sits s back along its viewing axis
                if p_r[2] > PICK_BAND_Z + 1e-9 or p_r[2] < GROUND_CLEAR_M:
                    why_fail["band"] = why_fail.get("band", 0) + 1
                    continue
                rho = math.hypot(p_r[0], p_r[1])
                lim = REACH_FRACTION * reach_lookup(rm, math.atan2(p_r[1], p_r[0]), float(p_r[2]))
                if rho > lim:
                    why_fail["reach80"] = why_fail.get("reach80", 0) + 1
                    continue
                sol = sw.solve(p_r, tgt["tool"], a_r, CAM_CONE_DEG if is_cam else tgt["cone_deg"],
                               exit_r="retreat" if tgt.get("exit") == "retreat" else None,
                               jaw_check=tgt["tool"] == "T4")
                if sol is None:
                    why_fail["ik_or_collision"] = why_fail.get("ik_or_collision", 0) + 1
                    continue
                score = sol["margin"]
                if best is None or score > best["margin"]:
                    best = {"station": {"xy": r3(st, 3), "heading_deg": round(math.degrees(h), 1), "mode": mode,
                                        "target_px": px, "target_py": py,
                                        "inside_basin": bool(math.hypot(st[0] - t["x"], st[1] - t["y"])
                                                             < t["basin_r"] + 0.25)},
                            "p_robot": r3(p_r), "standoff": s, "rho": round(rho, 3), "reach80": round(lim, 3), **sol}
            if best is not None and best["margin"] >= 0.01:
                break
        if best is not None and best["margin"] >= 0.01:
            break
    if best is None:
        if why_fail.get("ik_or_collision"):
            reason = "no_clearance" if sw.fail_stats.get("jaw_only") else "ik_or_collision"
        elif why_fail.get("reach80"):
            reason = "out_of_reach_80"
        else:
            reason = "no_station"
        return {**rec, "status": "MARK", "reason": reason, "fail_counts": why_fail}
    return {**rec, "status": "FROZEN", **best}


def sweep_leaf(sw: ArmSweep, leaf: dict, t: dict, trees, reach_maps) -> dict:
    p_tree = np.asarray(leaf["p"], dtype=np.float64)
    n_tree = unit(leaf["n"])
    rec = {"id": leaf["id"], "tree": leaf["tree"], "p": leaf["p"], "kind": leaf["kind"]}
    if p_tree[2] > PICK_BAND_Z:
        return {**rec, "in_band": False, "underside": False, "reason": "above_pick_band"}
    rm = reach_maps["CAM"]
    best = None
    for st, h, px, py, ok, why, mode in _placements(t, p_tree, trees):
        if not ok:
            continue
        R = rotz(h)
        load_world(sw, t, st, h, trees)
        c_r = to_robot(t, st, h, p_tree)
        n_r = R.T @ n_tree
        sw.set_leaf(c_r, n_r)
        for a_r in cone_axes(n_r, 35.0, 6):           # view axes within 35 deg of the leaf normal
            if a_r[2] <= 0.2:                          # must look up at the underside
                continue
            for s in (0.12, 0.16, 0.20):
                cam = c_r - a_r * s                    # camera below the leaf, looking up
                if cam[2] < GROUND_CLEAR_M:
                    continue
                rho = math.hypot(cam[0], cam[1])
                if rho > REACH_FRACTION * reach_lookup(rm, math.atan2(cam[1], cam[0]), float(cam[2])):
                    continue
                sol = sw.solve(cam, "CAM", a_r, CAM_CONE_DEG, leaf_check=True)
                if sol and (best is None or sol["margin"] > best["margin"]):
                    best = {**sol, "station": {"xy": r3(st, 3), "heading_deg": round(math.degrees(h), 1), "mode": mode},
                            "standoff": s, "wrist_pitch": sol["q"][4], "cam_robot": r3(cam)}
            if best is not None:
                break
        if best is not None:
            break
    return {**rec, "in_band": True, "underside": best is not None, **({"pose": best} if best else {})}


# ---------------------------------------------------------------------------------------------- rack, crate, mast
RACK_X = tuple(round(-0.20 + 0.05 * i, 2) for i in range(10))     # -0.20 .. 0.25
RACK_Y = (-0.17, -0.19, -0.21)
RACK_Z = (0.14, 0.18, 0.22, 0.26, 0.30)
RACK_ORDER = ("T2", "T4", "T5")
RACK_LEG_HIP_RAD = (0.0, 0.45, -0.45)   # stand, SCAN_WIDE abduction both ways (right legs)
RACKED_LEN, RACKED_R = 0.13, 0.03       # worst racked tool envelope below the cradle point


def _leg_geoms(m):
    leg_b = {b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").endswith(
        ("_hip", "_thigh", "_calf"))}
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] in leg_b]


def _racked_leg_clear(sw: "ArmSweep", geoms) -> dict:
    """Distance of the given racked-tool geoms to every leg geom with the right hips at 0 / +-0.45 rad."""
    m, d = sw.m, sw.d
    legs = _leg_geoms(m)
    hips = [m.jnt_qposadr[_id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_hip_joint")] for n in ("FR", "MR", "RR")]
    out = {}
    for a in RACK_LEG_HIP_RAD:
        d.qpos[:] = sw.q_base
        d.qpos[hips] = a
        mujoco.mj_kinematics(m, d)
        out[f"{a:+.2f}"] = round(sw.dist(geoms, legs)[0], 4)
    sw.fk(sw.q_base[sw.qadr])
    return out


def rack_sweep(mast_xy=MAST_XY) -> dict:
    """Pick 3 cradle positions on one right-side rail (same y, z): tool-down docking with a 5 cm vertical
    approach/exit, every tool reachable with the other two racked, racked tools clear of the right legs in the
    SCAN_WIDE stance; max margin, then the lowest rail (lower CoM)."""
    sw = ArmSweep(cradles=[], mast_xy=mast_xy)
    single = {}
    for x in RACK_X:
        for y in RACK_Y:
            for z in RACK_Z:
                p_r = sw.R_robot.T @ (sw.trunk_pos + sw.trunk_R @ np.array([x, y, z]) - sw.ground_origin)
                sol = sw.solve(p_r, "T4", [0, 0, -1.0], 8.0, exit_r=[0, 0, 1.0], tcp=RACK_TCP)
                if not sol:
                    continue
                sw.clear_obstacles()
                a_r = sw.R_robot.T @ (sw.trunk_pos + sw.trunk_R @ np.array([x, y, z - 0.01]) - sw.ground_origin)
                b_r = sw.R_robot.T @ (sw.trunk_pos + sw.trunk_R @ np.array([x, y, z - RACKED_LEN]) - sw.ground_origin)
                sw.add_capsule(a_r, b_r, RACKED_R, label="racked_probe")
                legc = min(_racked_leg_clear(sw, [sw.active_obs["wood"][0][0]]).values())
                sw.clear_obstacles()
                if legc < ROBOT_CLEAR_M:
                    continue
                single[(x, y, z)] = min(sol["margin"], legc - ROBOT_CLEAR_M)
    best = None
    for y in RACK_Y:
        for z in RACK_Z:
            xs = sorted(x for (x, yy, zz) in single if yy == y and zz == z)
            for i in range(len(xs)):
                for j in range(i + 1, len(xs)):
                    for k in range(j + 1, len(xs)):
                        if xs[j] - xs[i] < 0.12 - 1e-9 or xs[k] - xs[j] < 0.12 - 1e-9:
                            continue
                        sc = min(single[(xs[i], y, z)], single[(xs[j], y, z)], single[(xs[k], y, z)])
                        key = (round(sc, 3), -z, -abs(y))
                        if best is None or key > best[0]:
                            best = (key, [(xs[i], y, z), (xs[j], y, z), (xs[k], y, z)])
    if best is None:
        return {"ok": False, "single_ok": len(single)}
    cradles = [{"tool": RACK_ORDER[i], "pos": list(p)} for i, p in enumerate(best[1])]
    # verify with the full rack built (other two tools racked as obstacles; own cradle excluded)
    sw2 = ArmSweep(cradles=cradles, mast_xy=mast_xy)
    out = []
    for i, c in enumerate(cradles):
        p = np.array(c["pos"])
        p_r = sw2.R_robot.T @ (sw2.trunk_pos + sw2.trunk_R @ p - sw2.ground_origin)
        own = [g for n, g in sw2.rack_geoms.items() if n == f"racked_{c['tool']}" or n == f"rack_cradle_{i}"]
        sol = sw2.solve(p_r, c["tool"], [0, 0, -1.0], 8.0, exit_r=[0, 0, 1.0], exclude_robot=own, tcp=RACK_TCP)
        out.append({**c, "reachable": sol is not None,
                    **({"q": sol["q"], "margin": sol["margin"], "clear": sol["clear"]} if sol else {})})
    racked = [g for n, g in sw2.rack_geoms.items() if n.startswith("racked_")]
    leg_clear = _racked_leg_clear(sw2, racked)
    return {"ok": all(c["reachable"] for c in out) and min(leg_clear.values()) >= ROBOT_CLEAR_M,
            "leg_clear_by_right_hip_rad": leg_clear, "cradles": out, "single_ok": len(single),
            "grid": {"x": list(RACK_X), "y": list(RACK_Y), "z": list(RACK_Z)}}


CRATE_HALF = (0.30, 0.20, 0.15)      # 0.60 x 0.40 x 0.30 m field crate (assumption)
CRATE_WALL = 0.01


CRATE_GAPS_M = (0.06, 0.10, 0.14)


def crate_world(sw: ArmSweep, gap: float) -> Tuple[float, float]:
    """Crate walls (4 boxes) on the robot's left with the near wall gap m outside the feet. Returns (y_near, hz)."""
    y_near = 0.255 + 0.03 + gap
    cy = y_near + CRATE_HALF[1]
    sw.clear_obstacles()
    hz = CRATE_HALF[2]
    sw.add_box([0, y_near + CRATE_WALL, hz], [CRATE_HALF[0], CRATE_WALL, hz], "crate_near")
    sw.add_box([0, y_near + 2 * CRATE_HALF[1] - CRATE_WALL, hz], [CRATE_HALF[0], CRATE_WALL, hz], "crate_far")
    sw.add_box([-CRATE_HALF[0] + CRATE_WALL, cy, hz], [CRATE_WALL, CRATE_HALF[1], hz], "crate_rear")
    sw.add_box([CRATE_HALF[0] - CRATE_WALL, cy, hz], [CRATE_WALL, CRATE_HALF[1], hz], "crate_front")
    return y_near, hz


def crate_sweep(sw: ArmSweep) -> dict:
    """Robot parked with the crate on its left: the T2 (fruit in the catcher) must reach drop points over the
    crate interior with the tool down and clear the rim, plus a 5 cm lift exit."""
    res = []
    for gap in CRATE_GAPS_M:
        y_near, hz = crate_world(sw, gap)
        pts = []
        for x in (-0.18, 0.0, 0.18):
            for dy in (0.10, 0.20):
                p = [x, y_near + dy, 2 * hz + 0.02]
                sol = sw.solve(p, "T2", [0, 0, -1.0], 25.0, exit_r=[0, 0, 1.0])
                pts.append({"p": r3(p, 3), "ok": sol is not None,
                            **({"q": sol["q"], "min_clear": sol["min_clear"], "exit_q_end": sol["exit"]["q_end"]}
                               if sol else {})})
        res.append({"gap_m": gap, "crate_near_wall_y": round(y_near, 3), "points": pts,
                    "n_ok": sum(p["ok"] for p in pts)})
    sw.clear_obstacles()
    best = max(res, key=lambda r: (r["n_ok"], r["gap_m"]))     # all points, then the widest foot gap
    return {"options": res, "frozen": {"gap_m": best["gap_m"], "crate_near_wall_y": best["crate_near_wall_y"],
                                       "n_ok": best["n_ok"], "n": len(best["points"])},
            "crate_half_m": list(CRATE_HALF)}


def mast_check(mast_xy=MAST_XY, frozen_q: Sequence[Sequence[float]] = ()) -> dict:
    """Mast at (x, y): static clearance to the trunk-body parts (gantry rail, pillars, basket, carriage
    travel), and the r 0.08 keep-out vs the arm over the gantry travel (PARK) and over the frozen poses."""
    m = build_layout(mast_xy=mast_xy, kinematic_tools=True)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_kinematics(m, d)
    ft = np.zeros(6)
    gid = lambda n: _id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
    mast_geoms = [gid(n) for n in ("mast_drive", "mast_outer", "mast_ring", "mast_s1_g", "mast_s2_g", "mast_s3_g")]
    body_parts = {n: gid(n) for n in ("gantry_rail", "gantry_pillar_rear", "gantry_pillar_front", "trunk_geom")}
    basket_b = _id(m, mujoco.mjtObj.mjOBJ_BODY, "dorsal_basket")
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == basket_b and m.geom_contype[g]:
            body_parts[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)] = g
    static = {}
    for n, g in body_parts.items():
        static[n] = round(min(geom_distance(m, d, a, g, 0.5, ft) for a in mast_geoms), 4)
    # the mast sits on top of the trunk box: contact with trunk_geom at its base is by design
    static_ok = all(v >= 0.005 for n, v in static.items() if n != "trunk_geom")
    keep = gid("mast_keepout")
    car_b = [_id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("arm_carriage", "arm_turret", "arm_upper", "arm_forearm",
                                                            "arm_wrist", "finger_left", "finger_right")]
    # shipped collidable geoms of carriage, turret, links, palm and fingers (tools are checked in the sweep)
    arm_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in car_b and (m.geom_contype[g] or m.geom_conaffinity[g])]
    qadr = [m.jnt_qposadr[_id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in
            ("gantry_slide", "arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist_pitch")]
    park = list(m.key_qpos[0][qadr])
    travel = []
    for g in np.round(np.linspace(-0.26, 0.26, 53), 3):
        d.qpos[:] = m.key_qpos[0]
        d.qpos[qadr] = [g] + park[1:]
        mujoco.mj_kinematics(m, d)
        dist = min(geom_distance(m, d, a, keep, 0.5, ft) for a in arm_geoms)
        travel.append((float(g), dist + KEEP_OUT_R))     # distance to the mast axis
    g_min_ok = min((g for g, dd in travel if dd >= KEEP_OUT_R), default=None)
    worst_pose = None
    for q in frozen_q:
        d.qpos[:] = m.key_qpos[0]
        d.qpos[qadr] = q
        mujoco.mj_kinematics(m, d)
        dist = min(geom_distance(m, d, a, keep, 0.5, ft) for a in arm_geoms)
        worst_pose = dist if worst_pose is None else min(worst_pose, dist)
    return {"mast_xy": list(mast_xy), "static_clear": static, "static_ok": static_ok,
            "park_axis_dist_at_gantry_min": round(travel[0][1], 4),
            "gantry_min_for_keepout": g_min_ok, "keepout_ok_full_travel": travel[0][1] >= KEEP_OUT_R,
            "frozen_poses_keepout_margin": None if worst_pose is None else round(worst_pose, 4)}


# ---------------------------------------------------------------------------------------------- physics: load
def _arm_static_torques(m, d) -> np.ndarray:
    d2 = mujoco.MjData(m)
    d2.qpos[:] = d.qpos
    mujoco.mj_forward(m, d2)
    return np.array([d2.qfrc_bias[m.jnt_dofadr[m.actuator_trnid[a, 0]]] for a in range(5)])


def load_case(q5, fruit_kg: float, ff: bool = True, seconds: float = 3.0, cradles=None,
              ff_fruit_kg: Optional[float] = None) -> dict:
    """T4 mounted with fruit_kg at the TCP, arm at q5, standing on flat ground with the full layout.

    With ff the feed-forward is tau_g / kp. ff_fruit_kg None = ideal feed-forward: qfrc_bias of this same model,
    so sub-mm sag holds by construction (it only shows the position servo holds with an exact FF). A value is the
    fruit mass the feed-forward assumes (e.g. 0.0 = payload unknown / not yet weighed): the FF then comes from a
    model with that fruit, which measures the sag a payload estimate error causes."""
    m = build_layout(mounted="T4", fruit_kg=fruit_kg, cradles=cradles, racked=("T2", "T5"))
    m_ff = m if ff_fruit_kg is None else build_layout(mounted="T4", fruit_kg=ff_fruit_kg, cradles=cradles,
                                                      racked=("T2", "T5"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = m.key_ctrl[0]
    d.qpos[7:12] = q5
    tau_static = _arm_static_torques(m, d)
    d.ctrl[0:5] = np.asarray(q5) + (gravity_offsets(m_ff, d) if ff else 0.0)
    site = _id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp_T4")
    tid = _id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    n = int(seconds / m.opt.timestep)
    acc_t, acc_w, acc_f, cnt = np.zeros(3), np.zeros(3), np.zeros(5), 0
    for k in range(n):
        if ff and k == int(1.0 / m.opt.timestep):          # re-evaluate at the live trunk attitude once
            dq = mujoco.MjData(m)
            dq.qpos[:] = d.qpos
            dq.qpos[7:12] = q5
            d.ctrl[0:5] = np.asarray(q5) + gravity_offsets(m_ff, dq)
        mujoco.mj_step(m, d)
        if k >= n - int(0.5 / m.opt.timestep):
            R = d.xmat[tid].reshape(3, 3)
            acc_t += R.T @ (d.site_xpos[site] - d.xpos[tid])
            acc_w += d.site_xpos[site]
            acc_f += d.actuator_force[0:5]
            cnt += 1
    tcp_trunk = acc_t / cnt
    # commanded TCP: same trunk pose, arm exactly at q5
    dk = mujoco.MjData(m)
    dk.qpos[:] = d.qpos
    dk.qpos[7:12] = q5
    mujoco.mj_kinematics(m, dk)
    R = dk.xmat[tid].reshape(3, 3)
    cmd_trunk = R.T @ (dk.site_xpos[site] - dk.xpos[tid])
    sag = tcp_trunk - cmd_trunk
    fr = m.actuator_forcerange[0:5, 1]
    force = acc_f / cnt
    return {"fruit_kg": fruit_kg, "ff": ff, "ff_mode": ("none" if not ff else
                                                        ("ideal" if ff_fruit_kg is None else f"fruit_{ff_fruit_kg:g}kg")),
            "q": r3(q5), "sag_m": round(float(np.linalg.norm(sag)), 5),
            "sag_z_m": round(float(sag[2]), 5),
            "static_torque_nm": {"shoulder": round(abs(tau_static[2]), 3), "elbow": round(abs(tau_static[3]), 3),
                                 "wrist": round(abs(tau_static[4]), 3)},
            "actuator_force_nm": {"shoulder": round(abs(force[2]), 3), "elbow": round(abs(force[3]), 3),
                                  "wrist": round(abs(force[4]), 3)},
            "forcerange_nm": {"shoulder": float(fr[2]), "elbow": float(fr[3]), "wrist": float(fr[4])},
            "saturated": [n for n, i in (("shoulder", 2), ("elbow", 3), ("wrist", 4)) if abs(force[i]) >= 0.98 * fr[i]],
            "mass_kg": round(float(m.body_subtreemass[1]), 3)}


ARM_STRETCH_Q = (0.0, math.pi / 2.0, math.pi / 2.0, -math.pi / 2.0, 0.0)   # links + tool in one horizontal line


def frozen_t4_torques(frozen: dict) -> dict:
    """Static joint torques of every frozen T4 pose (target and exit end) with 1.2 and 1.8 kg fruit."""
    out = {}
    for kg in (1.2, 1.8):
        m = build_layout(mounted="T4", fruit_kg=kg)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        fr = m.actuator_forcerange[0:5, 1]
        worst = np.zeros(3)
        over70 = 0
        n = 0
        for t in frozen.get("targets", []):
            if t.get("status") != "FROZEN" or t["tool"] != "T4":
                continue
            for q in (t["q"], (t.get("exit") or {}).get("q_end")):
                if q is None:
                    continue
                d.qpos[7:12] = q
                tau = np.abs(_arm_static_torques(m, d)[2:5])
                worst = np.maximum(worst, tau)
                over70 += int(np.any(tau > MODEL_HEADROOM * fr[2:5]))
                n += 1
        pct = worst / fr[2:5]
        out[f"{kg:g}"] = {"poses": n, "max_shoulder_nm": round(float(worst[0]), 3), "max_elbow_nm": round(float(worst[1]), 3),
                          "max_wrist_nm": round(float(worst[2]), 3), "poses_over_70pct_of_model_forcerange": over70,
                          "max_pct_of_model_forcerange": {"shoulder": round(float(pct[0]), 3),
                                                          "elbow": round(float(pct[1]), 3),
                                                          "wrist": round(float(pct[2]), 3)}}
    return out


def worst_t4_pose(frozen: dict, fruit_kg: float = 1.8) -> Tuple[Optional[list], Optional[str]]:
    """The frozen T4 target pose with the largest static shoulder torque at fruit_kg."""
    m = build_layout(mounted="T4", fruit_kg=fruit_kg)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    best = (None, None, -1.0)
    for t in frozen.get("targets", []):
        if t.get("status") != "FROZEN" or t["tool"] != "T4":
            continue
        for q in (t["q"], (t.get("exit") or {}).get("q_end")):
            if q is None:
                continue
            d.qpos[7:12] = q
            tau = _arm_static_torques(m, d)
            if abs(tau[2]) > best[2]:
                best = (q, t["id"], abs(tau[2]))
    return best[0], best[1]


# ---------------------------------------------------------------------------------------------- physics: slope
def set_slope(m: mujoco.MjModel, roll_deg: float = 0.0, pitch_deg: float = 0.0):
    """Rotated gravity = standing on a plane. roll > 0: ground falls to -y (right / rack side downhill)."""
    g = 9.81
    a, b = math.radians(roll_deg), math.radians(pitch_deg)
    gv = np.array([g * math.sin(b), -g * math.sin(a) * math.cos(b), -g * math.cos(a) * math.cos(b)])
    m.opt.gravity[:] = gv


def set_tilt(m: mujoco.MjModel, tilt_deg: float, downhill_deg: float):
    """Rotated gravity for a plane tilted by tilt_deg whose fall line points along downhill_deg in the robot frame
    (0 = nose downhill, 90 = left downhill, 180 = rear / mast downhill, 270 = right / rack downhill)."""
    g = 9.81
    t, a = math.radians(tilt_deg), math.radians(downhill_deg)
    m.opt.gravity[:] = np.array([g * math.sin(t) * math.cos(a), g * math.sin(t) * math.sin(a), -g * math.cos(t)])


def tilt_of(m: mujoco.MjModel) -> Tuple[float, float]:
    """(tilt_deg, downhill_deg in [0, 360)) of the current gravity vector."""
    gv = np.asarray(m.opt.gravity, dtype=np.float64)
    gn = float(np.linalg.norm(gv))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -gv[2] / gn))))
    dh = math.degrees(math.atan2(gv[1], gv[0])) % 360.0 if math.hypot(gv[0], gv[1]) > 1e-9 else 0.0
    return round(tilt, 3), round(dh, 1)


def rack_side(downhill_deg: float, tilt_deg: float) -> str:
    """Rack (right, -y) side relative to the fall line."""
    s = math.sin(math.radians(downhill_deg))
    if tilt_deg < 1e-9 or abs(s) < 1e-6:
        return "level"
    return "downhill" if s < 0 else "uphill"


def _apply_slope(m, roll_deg, pitch_deg, tilt_deg, downhill_deg):
    if tilt_deg is not None:
        set_tilt(m, tilt_deg, downhill_deg or 0.0)
    else:
        set_slope(m, roll_deg, pitch_deg)


FEET = ("FR_foot", "FL_foot", "MR_foot", "ML_foot", "RR_foot", "RL_foot")
TRI = {"A": ("FR_foot", "ML_foot", "RR_foot"), "B": ("FL_foot", "MR_foot", "RL_foot")}


def margins(m: mujoco.MjModel, d: mujoco.MjData) -> dict:
    """Static margins of the whole-robot CoM projected along gravity: 6-foot hull and tripods A/B."""
    from atharv_nandi.core.stability import com_margin, convex_hull
    gh = unit(m.opt.gravity)
    x = np.array([1.0, 0, 0])
    u = unit(x - np.dot(x, gh) * gh)
    v = unit(np.cross(-gh, u))
    P = lambda p: (float(np.dot(p, u)), float(np.dot(p, v)))  # noqa: E731
    feet = {f: d.geom_xpos[_id(m, mujoco.mjtObj.mjOBJ_GEOM, f)].copy() for f in FEET}
    com = P(d.subtree_com[1])
    hexa = com_margin(com, convex_hull([P(p) for p in feet.values()]))
    tri = {k: com_margin(com, convex_hull([P(feet[f]) for f in fs])) for k, fs in TRI.items()}
    return {"hexa": round(hexa, 4), "tri_A": round(tri["A"], 4), "tri_B": round(tri["B"], 4),
            "tripod": round(min(tri.values()), 4)}


def creep_case(roll_deg: float, pitch_deg: float = 0.0, ext: float = 0.0, arm_q=None, fruit_kg=0.0,
               cradles=None, ref_s: float = CREEP_REF_S, window_s: float = CREEP_WINDOW_S,
               tilt_deg: Optional[float] = None, downhill_deg: Optional[float] = None) -> dict:
    """Standing creep. The slope is (roll, pitch) via set_slope, or (tilt, downhill) via set_tilt when tilt_deg
    is given. The pose-loop reference (OS 2.1) is the base pose ref_s after arrival (CREEP_REF_S = 0.1 s: feet
    in contact, the keyframe drop over); every number counts from there, so the early slip on the slope is in.
    creep_m: in-plane base displacement over window_s. time_to_2cm_s: first time (from the reference) the
    in-plane base displacement exceeds the 2 cm threshold, None if it does not within the window.
    drift_first_1s_m: the part of creep_m in the first second (the share a 1 s-settle reference would drop)."""
    mounted = "T4" if arm_q is not None else None
    m = build_layout(ext=ext, cradles=cradles, mounted=mounted, fruit_kg=fruit_kg,
                     racked=("T2", "T5") if mounted else ("T2", "T4", "T5"))
    _apply_slope(m, roll_deg, pitch_deg, tilt_deg, downhill_deg)
    tilt, fall_deg = tilt_of(m)
    gh = unit(m.opt.gravity)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = m.key_ctrl[0]
    if arm_q is not None:
        d.qpos[7:12] = arm_q
        d.ctrl[0:5] = np.asarray(arm_q) + gravity_offsets(m, d)
    for _ in range(int(round(ref_s / m.opt.timestep))):
        mujoco.mj_step(m, d)
    inplane = lambda v: float(np.linalg.norm(v - np.dot(v, gh) * gh))  # noqa: E731
    p0 = d.qpos[0:3].copy()
    feet0 = np.array([d.geom_xpos[_id(m, mujoco.mjtObj.mjOBJ_GEOM, f)] for f in FEET])
    n = int(round(window_s / m.opt.timestep))
    every = max(1, int(round(0.1 / m.opt.timestep)))
    k_1s = int(round(1.0 / m.opt.timestep)) - 1
    p_half = p0
    t_cross = None
    drift_1s = None
    for k in range(n):
        mujoco.mj_step(m, d)
        if k == n // 2:
            p_half = d.qpos[0:3].copy()
        if k == k_1s:
            drift_1s = inplane(d.qpos[0:3] - p0)
        if t_cross is None and (k + 1) % every == 0:
            if inplane(d.qpos[0:3] - p0) > CREEP_DWELL_LIMIT_M:
                t_cross = round((k + 1) * m.opt.timestep, 1)
    p1 = d.qpos[0:3].copy()
    feet1 = np.array([d.geom_xpos[_id(m, mujoco.mjtObj.mjOBJ_GEOM, f)] for f in FEET])
    rate = inplane(p1 - p_half) / (window_s / 2.0)
    df = feet1 - feet0
    df_plane = df - (df @ gh)[:, None] * gh[None, :]
    return {"roll_deg": roll_deg if tilt_deg is None else None, "pitch_deg": pitch_deg if tilt_deg is None else None,
            "tilt_deg": tilt, "downhill_deg": fall_deg, "ext": ext, "arm": "T4_worst" if arm_q is not None else "PARK",
            "fruit_kg": fruit_kg, "ref_s": ref_s, "window_s": window_s, "creep_m": round(inplane(p1 - p0), 4),
            "drift_first_1s_m": round(drift_1s if drift_1s is not None else inplane(p1 - p0), 4),
            "rate_2nd_half_mm_s": round(rate * 1000.0, 3), "time_to_2cm_s": t_cross,
            "feet_slip_max_m": round(float(np.max(np.linalg.norm(df_plane, axis=1))), 4),
            "mass_kg": round(float(m.body_subtreemass[1]), 3)}


def mast_budget(head_kg: float = HEAD_KG) -> dict:
    """Measured masses of the modelled WP0c mast against the OS 2.3 budget, and which robot the matrix describes.
    The head at the 0.6 kg hard cap is the stability worst case, but with it the modelled mast breaks both mass
    budgets; the shipped WP1 robot differs (see `robot`), so the gate envelope here is not the shipped one."""
    m = build_layout(head_kg=head_kg)
    sys_kg = float(m.body_subtreemass[_id(m, mujoco.mjtObj.mjOBJ_BODY, "wp0c_mast")])
    mov_kg = float(m.body_subtreemass[_id(m, mujoco.mjtObj.mjOBJ_BODY, MAST_SECTIONS[0][0])])
    return {"head_kg": head_kg, "mast_system_kg": round(sys_kg, 3), "mast_system_budget_kg": MAST_SYSTEM_BUDGET_KG,
            "mast_moving_kg": round(mov_kg, 3), "mast_moving_budget_kg": MAST_MOVING_BUDGET_KG,
            "within_budget": sys_kg <= MAST_SYSTEM_BUDGET_KG + 1e-9 and mov_kg <= MAST_MOVING_BUDGET_KG + 1e-9,
            "robot": f"pre-WP1 {os.path.relpath(ROBOT_XML, paths.PACKAGE_DIR)} + the WP0c lumped mast/rack (not the shipped WP1 "
                     f"robot: head 0.40 kg = 0.36 box + sensors, ring 0.745 kg, battery moved)",
            "superseded_by": "WP6 tools/crown_gate_matrix.py -> config/crown_gates.json (the matrix re-run on the "
                             "shipped WP1 robot, head raised to the 0.6 kg cap); the OS 2.3 table is replaced from "
                             "that block, not from this one"}


def matrix_case(ext: float, roll_deg: float, pitch_deg: float, arm_q=None, fruit_kg=0.0, cradles=None,
                settle_s: float = 1.5, model_cache: Optional[dict] = None,
                tilt_deg: Optional[float] = None, downhill_deg: Optional[float] = None) -> dict:
    """Static stability of the full layout. Slope as (roll, pitch) via set_slope (pitch > 0 nose downhill,
    pitch < 0 rear / mast downhill), or as (tilt, downhill) via set_tilt when tilt_deg is given."""
    key = (arm_q is not None, fruit_kg)
    mounted = "T4" if arm_q is not None else None
    if model_cache is not None and key in model_cache:
        m = model_cache[key]
    else:
        m = build_layout(ext=0.0, cradles=cradles, mounted=mounted, fruit_kg=fruit_kg,
                         racked=("T2", "T5") if mounted else ("T2", "T4", "T5"))
        if model_cache is not None:
            model_cache[key] = m
    for name in ("mast_s1", "mast_s2", "mast_s3"):
        m.body_pos[_id(m, mujoco.mjtObj.mjOBJ_BODY, name)][2] = ext / 3.0
    _apply_slope(m, roll_deg, pitch_deg, tilt_deg, downhill_deg)
    tilt, dh = tilt_of(m)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = m.key_ctrl[0]
    if arm_q is not None:
        d.qpos[7:12] = arm_q
        d.ctrl[0:5] = np.asarray(arm_q) + gravity_offsets(m, d)
    fell = False
    for _ in range(int(settle_s / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if d.qpos[2] < 0.12:
            fell = True
            break
    mg = margins(m, d)
    head = d.xpos[_id(m, mujoco.mjtObj.mjOBJ_BODY, "crown_head")]
    if tilt_deg is None:
        rack = "downhill" if roll_deg > 0 else ("uphill" if roll_deg < 0 else "level")
    else:
        rack = rack_side(dh, tilt)
    return {"ext": ext, "roll_deg": roll_deg if tilt_deg is None else None,
            "pitch_deg": pitch_deg if tilt_deg is None else None, "tilt_deg": tilt, "downhill_deg": dh,
            "rack": rack,
            "arm": "T4_worst" if arm_q is not None else "PARK", "fruit_kg": fruit_kg, **mg,
            "lens_z": round(float(head[2]) - float(d.qpos[2]) + 0.34, 3), "fell": fell,
            "mass_kg": round(float(m.body_subtreemass[1]), 3)}


# ---------------------------------------------------------------------------------------------- commands
def cmd_candidates(args):
    gen = generate_candidates()
    upd = {
        "schema": "nandi.orchard_candidates.v1",
        "classification": "Public",
        "owner": "WP0c (tools/feasibility_sweep.py)",
        "basis": {
            "spec": "ORCHARD-MVP-SPEC 2.1-2.4, 4.2; WP0c",
            "spacing": "OQ-02 B (2026-11-17): mango rows 5 x 2.5 m frozen (R1 in-row 2.5 m, row pitch 5 m "
                       "to R2); 5 x 3 m is the alternative mango spacing: `sweep` re-runs every R1 target with R1 "
                       "at 3 m in-row (layout.variants.mango_5x3) and records it in frozen.variant_5x3",
            "frames": "tree frame = trunk axis at the origin, world axes; robot frame = ground under the trunk "
                      "centre, x forward, y left",
            "labels": "tree shapes, tool geometry/masses and crate size are assumptions; every number in "
                      "`frozen`, `load`, `creep`, `matrix` is (measured, sweep) with its run_id",
        },
        "rules": {"pick_band_z_m": PICK_BAND_Z, "reach_fraction": REACH_FRACTION, "exit_m": EXIT_M,
                  "robot_clear_m": ROBOT_CLEAR_M, "wood_clear_m": WOOD_CLEAR_M, "jaw_clear_m": JAW_CLEAR_M,
                  "leaf_clear_m": LEAF_CLEAR_M, "keep_out_r_m": KEEP_OUT_R, "station_wood_clear_m": STATION_WOOD_CLEAR_M,
                  "crown_foliage_clear_m": CROWN_FOLIAGE_CLEAR_M},
        "tools": {k: {kk: vv for kk, vv in v.items()} for k, v in TOOLS.items()},
        "layout": {"home": [0.0, 0.0], "dock": [0.0, 0.0], "crate": [0.5, -1.5], "mud_patch": [-3.0, -2.0],
                   "rows_y": {"R1": 2.5, "R2": 7.5, "R3": 11.0, "R4": 14.5}, "mango_5x3_in_row_m": MANGO_5X3_IN_ROW_M,
                   "variants": {"mango_5x3": {"row": "R1", "in_row_m": MANGO_5X3_IN_ROW_M,
                                              "x": [t["x"] for t in variant_5x3_trees(gen["trees"])
                                                    if t["id"].startswith("R1-")]}}},
        "candidates": gen,
    }
    stale = []
    if args.write:
        stale = update_json(upd)["stale_blocks"]      # frozen/load/creep/matrix keep their data, flagged if stale
    print(json.dumps({"trees": len(gen["trees"]), "targets": len(gen["targets"]), "leaves": len(gen["leaves"]),
                      "stale_blocks": stale}))


def variant_5x3_trees(trees: List[dict], in_row: float = MANGO_5X3_IN_ROW_M) -> List[dict]:
    """The candidate trees with the R1 mango row re-spaced to in_row (first tree fixed at its x). Tree-frame
    targets, wood and canopies are unchanged; only the neighbour distances (stations, wood) change."""
    r1 = sorted((t for t in trees if t["id"].startswith("R1-")), key=lambda t: t["x"])
    x0 = r1[0]["x"] if r1 else 0.0
    newx = {t["id"]: round(x0 + i * in_row, 3) for i, t in enumerate(r1)}
    return [{**t, "x": newx[t["id"]]} if t["id"] in newx else dict(t) for t in trees]


def sweep_variant_5x3(sw: "ArmSweep", doc: dict, base_targets: List[dict], reach_maps: dict) -> dict:
    """Every R1 target swept again on the 5 x 3 m variant layout; compared with the 5 x 2.5 m result."""
    t0 = time.time()
    trees = variant_5x3_trees(doc["candidates"]["trees"])
    tmap = {t["id"]: t for t in trees}
    base = {t["id"]: t for t in base_targets}
    res = [sweep_target(sw, tg, tmap[tg["tree"]], trees, reach_maps)
           for tg in doc["candidates"]["targets"] if tg["tree"].startswith("R1-")]
    by_task: Dict[str, List[int]] = {}
    marked: Dict[str, int] = {}
    for r in res:
        b = base[r["id"]]
        e = by_task.setdefault(r["task"], [0, 0])
        e[0] += int(b["status"] == "FROZEN")
        e[1] += int(r["status"] == "FROZEN")
        if r["status"] != "FROZEN":
            marked[r["reason"]] = marked.get(r["reason"], 0) + 1
    diff = [{"id": r["id"], "5x2.5": base[r["id"]]["status"], "5x3": r["status"], "reason_5x3": r.get("reason")}
            for r in res if r["status"] != base[r["id"]]["status"]]
    return {"row": "R1", "in_row_m": MANGO_5X3_IN_ROW_M,
            "x": [t["x"] for t in trees if t["id"].startswith("R1-")],
            "targets": len(res), "frozen_5x2.5": sum(v[0] for v in by_task.values()),
            "frozen_5x3": sum(v[1] for v in by_task.values()),
            "frozen_by_task_5x2.5_vs_5x3": {k: v for k, v in by_task.items()}, "marked_by_reason_5x3": marked,
            "status_changes": diff, "wall_s": round(time.time() - t0, 1),
            "targets_5x3": [{k: r.get(k) for k in ("id", "status", "reason", "margin", "station")} for r in res]}


def run_sweep(doc: dict, progress=True) -> dict:
    t0 = time.time()
    meta = run_meta("sweep")
    trees = doc["candidates"]["trees"]
    tmap = {t["id"]: t for t in trees}
    rack = rack_sweep()
    if progress:
        print(f"[{time.time() - t0:5.1f}s] rack", file=sys.stderr, flush=True)
    cradles = [{"tool": c["tool"], "pos": c["pos"]} for c in rack.get("cradles", [])]
    sw = ArmSweep(cradles=cradles)
    reach_maps = {k: reach_map(sw, k) for k in ("T1", "T4", "CAM")}
    if progress:
        print(f"[{time.time() - t0:5.1f}s] rack + reach maps", file=sys.stderr, flush=True)
    rm = reach_maps["T1"]
    reach_fwd = {f"{z:.3f}": round(max(reach_lookup(rm, 0.0, z), 0.0), 3) for z in (0.05, 0.40, 0.80)}
    reach_side = {f"{z:.3f}": round(reach_lookup(rm, math.pi / 2, z), 3) for z in (0.05, 0.40, 0.80)}
    targets = []
    for tg in doc["candidates"]["targets"]:
        t1 = time.time()
        targets.append(sweep_target(sw, tg, tmap[tg["tree"]], trees, reach_maps))
        targets[-1]["t_s"] = round(time.time() - t1, 2)
    if progress:
        print(f"[{time.time() - t0:5.1f}s] targets", file=sys.stderr, flush=True)
    leaves = [sweep_leaf(sw, lf, tmap[lf["tree"]], trees, reach_maps) for lf in doc["candidates"]["leaves"]]
    if progress:
        print(f"[{time.time() - t0:5.1f}s] leaves", file=sys.stderr, flush=True)
    crate = crate_sweep(sw)
    frozen_q = [t["q"] for t in targets if t["status"] == "FROZEN"] + [c["q"] for c in rack.get("cradles", []) if c.get("q")]
    mast = [mast_check((x, y), frozen_q if (x, y) == MAST_XY else ()) for x, y in
            ((-0.36, 0.0), (-0.33, 0.0), (-0.30, 0.0), (-0.30, 0.07), (-0.30, 0.09))]
    in_band = [lf for lf in leaves if lf["in_band"]]
    under = sum(1 for lf in in_band if lf["underside"])
    fz = [t for t in targets if t["status"] == "FROZEN"]
    summary = {
        "targets": len(targets), "frozen": len(fz), "marked": len(targets) - len(fz),
        "frozen_by_task": {}, "marked_by_reason": {},
        "leaf_in_band": len(in_band), "leaf_underside": under,
        "underside_fraction": round(under / len(in_band), 3) if in_band else None,
        "rack_ok": rack.get("ok", False), "crate_points_ok": f"{crate['frozen']['n_ok']}/{crate['frozen']['n']}",
        "reach_T1_forward": reach_fwd, "reach_T1_left": reach_side,
        "reach_spec_os22_forward": SPEC_OS22_REACH_FWD,
        "reach_note": "measured = largest horizontal distance from the trunk-centre axis with a collision-free T1 "
                      "pose at any tool angle; 0.0 at z 0.05 because the T1 fingers reach the ground (ground "
                      "clearance < 0) at every candidate there. OS 2.2 quotes 0.83 / 0.98 / 0.89 m at 0.035 / 0.4 / "
                      "0.8 m (probe); the 80 % rule uses the measured map",
        "max_frozen_z": max((t["p_robot"][2] for t in fz), default=None),
        "max_rho_over_reach": round(max((t["rho"] / (t["reach80"] / REACH_FRACTION) for t in fz), default=0), 3),
        "min_frozen_clearance": min((t["min_clear"] for t in fz), default=None),
        "min_frozen_margin": min((t["margin"] for t in fz), default=None),
    }
    for t in fz:
        summary["frozen_by_task"][t["task"]] = summary["frozen_by_task"].get(t["task"], 0) + 1
    for t in targets:
        if t["status"] != "FROZEN":
            summary["marked_by_reason"][t["reason"]] = summary["marked_by_reason"].get(t["reason"], 0) + 1
    meta["wall_s"] = round(time.time() - t0, 1)
    meta["budget_s"] = SWEEP_BUDGET_S
    variant = sweep_variant_5x3(sw, doc, targets, reach_maps)
    if progress:
        print(f"[{time.time() - t0:5.1f}s] 5 x 3 m variant", file=sys.stderr, flush=True)
    summary["variant_5x3_frozen"] = f"{variant['frozen_5x3']}/{variant['targets']} (5 x 2.5 m: {variant['frozen_5x2.5']})"
    return {"meta": meta, "summary": summary, "rack": rack, "crate": crate, "mast": mast,
            "reach_maps": reach_maps, "targets": targets, "leaves": leaves, "variant_5x3": variant}


def cmd_sweep(args):
    doc = load_json()
    res = run_sweep(doc)
    print(json.dumps({"meta": res["meta"], "summary": res["summary"], "rack_ok": res["rack"].get("ok"),
                      "mast": res["mast"]}, indent=1))
    if args.write:
        res["inputs"] = block_inputs(doc, "frozen")
        update_json({"frozen": res})


def load_summary(load: dict) -> dict:
    """Owner-facing status of the load gates. The sag gate as written uses the ideal feed-forward (the same
    model's qfrc_bias), so it cannot fail: it is reported as vacuous, next to the sag a fruit-blind feed-forward
    gives, until the runtime feed-forward carries crop_mass before the hold."""
    blind = load.get("ff_mismatch_cases") or []
    sag_blind = {f"{c['fruit_kg']:g}": c["sag_m"] for c in blind}
    shoulder_ok = all(c["static_torque_nm"]["shoulder"] <= SHOULDER_GATE_NM for c in load["cases"] if c["ff"])
    head = load["pass_model_headroom_by_fruit_kg"]
    return {
        "spec_gate": "pass" if load["pass_spec_gate"] else "fail",
        "shoulder_gate": "pass" if shoulder_ok else "fail",
        "sag_gate": "vacuous (ideal feed-forward, holds by construction)",
        "sag_fruit_blind_ff_m": sag_blind,
        "sag_gate_with_fruit_blind_ff": ("pass" if blind and all(v < SAG_GATE_M for v in sag_blind.values())
                                         else "fail"),
        "model_headroom_gate": {k: "pass" if v else "fail" for k, v in head.items()},
        "overall": "pass" if load["pass"] else "fail",
        "text": (f"spec gate {'passes' if load['pass_spec_gate'] else 'fails'}; model-headroom gate "
                 + ", ".join(f"{'passes' if v else 'fails'} at {k} kg" for k, v in head.items())
                 + "; the sag gate is vacuous until the runtime feed-forward carries crop_mass (fruit-blind "
                 + "feed-forward: " + " / ".join(f"{v * 1000:.1f} mm at {k} kg" for k, v in sag_blind.items())
                 + f", gate {SAG_GATE_M * 1000:g} mm)"),
    }


def cmd_load(args):
    doc = load_json()
    fr = doc.get("frozen")
    if not fr:
        raise SystemExit("run `sweep --write` first")
    q, tid = worst_t4_pose(fr, 1.8)
    cradles = [{"tool": c["tool"], "pos": c["pos"]} for c in fr["rack"].get("cradles", [])]
    out = {"meta": run_meta("load"), "inputs": block_inputs(doc, "load"), "pose_from": tid, "cases": []}
    for kg in (1.2, 1.8):
        for ff in (True, False):
            out["cases"].append(load_case(q, kg, ff=ff, cradles=cradles))
    # payload-estimate error: the FF assumes no fruit (not yet weighed). Informative, not gated.
    out["ff_mismatch_cases"] = [load_case(q, kg, ff=True, cradles=cradles, ff_fruit_kg=0.0) for kg in (1.2, 1.8)]
    out["frozen_t4_static"] = frozen_t4_torques(fr)
    out["envelope"] = []
    for kg in (1.2, 1.8):
        m = build_layout(mounted="T4", fruit_kg=kg)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[7:12] = ARM_STRETCH_Q          # upper arm + forearm + tool horizontal, sideways: lever max
        tau = _arm_static_torques(m, d)
        out["envelope"].append({"fruit_kg": kg, "q": list(ARM_STRETCH_Q), "shoulder_nm": round(abs(tau[2]), 3),
                                "elbow_nm": round(abs(tau[3]), 3), "wrist_nm": round(abs(tau[4]), 3)})
    out["gate"] = {"sag_m_lt": SAG_GATE_M, "shoulder_total_nm_le": SHOULDER_GATE_NM,
                   "note": "59.5 = 70 % of the 85 N.m forerange the spec quotes; the shipped MJCF (physics alignment "
                           "v1) has arm_shoulder forcerange 30 N.m, so 70 % of the model is 21 N.m (reported too)",
                   "model_shoulder_70pct_nm": 0.7 * 30.0,
                   "model_headroom": f"every frozen T4 pose (target + exit end): shoulder, elbow and wrist static "
                                     f"torque <= {MODEL_HEADROOM:g} x the shipped MJCF forcerange",
                   "sag_note": "the gated sag uses the ideal feed-forward (qfrc_bias of the same model at the "
                               "commanded q), so it only shows the position servo holds; ff_mismatch_cases give "
                               "the sag when the FF assumes no fruit (payload estimate error), not gated"}
    out["pass_spec_gate"] = all(c["sag_m"] < SAG_GATE_M and c["static_torque_nm"]["shoulder"] <= SHOULDER_GATE_NM
                                for c in out["cases"] if c["ff"])
    out["pass_model_headroom_by_fruit_kg"] = {k: v["poses_over_70pct_of_model_forcerange"] == 0
                                              for k, v in out["frozen_t4_static"].items()}
    out["pass"] = out["pass_spec_gate"] and all(out["pass_model_headroom_by_fruit_kg"].values())
    out["load_summary"] = load_summary(out)
    print(json.dumps(out, indent=1))
    if args.write:
        update_json({"load": out})


def _tilt_key(c: dict) -> float:
    return round(float(c["tilt_deg"]), 1)


def dwell_limit(cases: List[dict]) -> Optional[float]:
    """Largest tilt (deg, any direction: cross, pitch either sign, oblique) at which every measured creep (all
    directions, all arm states) is <= 2 cm over its window (CREEP_WINDOW_S = one pick cycle), with every smaller
    measured tilt also passing."""
    slopes = sorted({_tilt_key(c) for c in cases})
    lim = None
    for s in slopes:
        worst = max(c["creep_m"] for c in cases if _tilt_key(c) == s)
        if worst <= CREEP_DWELL_LIMIT_M:
            lim = s
        else:
            break
    return lim


def dwell_time_by_tilt(long_cases: List[dict]) -> Dict[str, Optional[float]]:
    """Per tilt: the shortest measured time (s) to the 2 cm pose-loop threshold over all directions and arm
    states of the long runs; None if no long run at that tilt crossed it within its window."""
    out: Dict[str, Optional[float]] = {}
    for s in sorted({_tilt_key(c) for c in long_cases}):
        ts = [c["time_to_2cm_s"] for c in long_cases if _tilt_key(c) == s and c["time_to_2cm_s"] is not None]
        out[f"{s:g}"] = min(ts) if ts else None
    return out


def long_dwell_limit(long_cases: List[dict], dwell_s: float = LONG_DWELL_S) -> Optional[float]:
    """Largest tilt whose long runs all stay under 2 cm for at least dwell_s (every smaller tilt also passing)."""
    lim = None
    for s, t in sorted(((float(k), v) for k, v in dwell_time_by_tilt(long_cases).items())):
        if t is None or t >= dwell_s:
            lim = s
        else:
            break
    return lim


def cmd_creep(args):
    doc = load_json()
    fr = doc.get("frozen") or {}
    cradles = [{"tool": c["tool"], "pos": c["pos"]} for c in (fr.get("rack") or {}).get("cradles", [])]
    q, _ = worst_t4_pose(fr, 1.8) if fr else (None, None)
    cases = []
    for roll in (0.0, 3.0, -3.0, 5.0, -5.0, 10.0, -10.0):
        cases.append(creep_case(roll, cradles=cradles))
    if q is not None:
        for roll in (3.0, -3.0, 5.0, -5.0):
            cases.append(creep_case(roll, arm_q=q, fruit_kg=1.8, cradles=cradles))
        for pitch in (5.0, -5.0):
            cases.append(creep_case(0.0, pitch, arm_q=q, fruit_kg=1.8, cradles=cradles))
    for pitch in (3.0, -3.0, 5.0, -5.0, 10.0, -10.0):          # + nose downhill, - rear (mast) downhill
        cases.append(creep_case(0.0, pitch, cradles=cradles))
    for tilt in (3.0, 5.0):                                      # oblique fall lines (every 45 deg)
        for dh in (45.0, 135.0, 225.0, 315.0):
            cases.append(creep_case(0.0, cradles=cradles, tilt_deg=tilt, downhill_deg=dh))
    long_cases = []
    for tilt in (3.0, 5.0):
        for dh in (0.0, 90.0, 180.0, 270.0):
            long_cases.append(creep_case(0.0, cradles=cradles, window_s=CREEP_LONG_WINDOW_S, tilt_deg=tilt,
                                         downhill_deg=dh))
        if q is not None:
            long_cases.append(creep_case(0.0, arm_q=q, fruit_kg=1.8, cradles=cradles, window_s=CREEP_LONG_WINDOW_S,
                                         tilt_deg=tilt, downhill_deg=270.0))
    lim = dwell_limit(cases)
    dwell_lim = min(lim, WALK_SLOPE_LIMIT_DEG) if lim is not None else None
    t_by = dwell_time_by_tilt(long_cases)
    out = {"meta": run_meta("creep"), "inputs": block_inputs(doc, "creep"), "cases": cases,
           "long_cases": long_cases, "dwell_limit_creep_deg": lim,
           "dwell_limit_deg": dwell_lim,
           "binding": "walking slope rule (OS 5.2, <= 5 deg)" if lim is not None and lim >= WALK_SLOPE_LIMIT_DEG
           else "creep",
           "dwell_window_s": CREEP_WINDOW_S,
           "time_to_2cm_s_by_tilt": t_by,
           "time_to_2cm_s_at_dwell_limit": t_by.get(f"{dwell_lim:g}") if dwell_lim is not None else None,
           "long_dwell_s": LONG_DWELL_S,
           "long_dwell_limit_deg": long_dwell_limit(long_cases),
           "rule": f"dwell limit = largest tilt (any direction) with creep <= {CREEP_DWELL_LIMIT_M} m in one "
                   f"{CREEP_WINDOW_S:g} s pick cycle (PARK and worst T4 pose), capped by the walking rule. Dwells "
                   f"longer than time_to_2cm_s_at_dwell_limit (LEAF SCAN, TREE 360 station, WORK) trigger one OS 2.1 "
                   f"pose-loop re-approach per that interval; long_dwell_limit_deg is the largest tilt that holds "
                   f"{LONG_DWELL_S:g} s without one. Reference convention: every creep, slip and time_to_2cm_s counts "
                   f"from the base pose {CREEP_REF_S:g} s after arrival (feet in contact), so the early slip on the "
                   f"slope is included; the pose-loop reference must be taken that early (drift_first_1s_m is the "
                   f"share a 1 s-settle reference would drop)",
           "ref_s": CREEP_REF_S}
    print(json.dumps(out, indent=1))
    if args.write:
        update_json({"creep": out})


TILT_BOUNDS_DEG = (0.0,) + MATRIX_TILTS_DEG


def _row_key(r: dict) -> dict:
    return {k: r[k] for k in ("arm", "ext", "roll_deg", "pitch_deg", "tilt_deg", "downhill_deg", "rack", "tripod",
                              "hexa")}


def matrix_summary(rows: List[dict]) -> dict:
    """Worst rows and the crown-up start/continue gate envelope over every slope direction.

    envelope[arm][ext][T] = min tripod over all rows (grid and heading sweep) with tilt <= T, any direction.
    start_gate_max_tilt_deg[arm][ext] = largest T in TILT_BOUNDS_DEG with envelope >= 0.04 m (every smaller T too);
    None when even flat ground fails."""
    worst = min(rows, key=lambda r: r["tripod"])
    env: Dict[str, Dict[str, Dict[str, float]]] = {}
    start: Dict[str, Dict[str, Optional[float]]] = {}
    cont: Dict[str, Dict[str, Optional[float]]] = {}
    for arm in sorted({r["arm"] for r in rows}):
        env[arm], start[arm], cont[arm] = {}, {}, {}
        for ext in sorted({r["ext"] for r in rows}):
            sub = [r for r in rows if r["arm"] == arm and r["ext"] == ext]
            e = {}
            for T in TILT_BOUNDS_DEG:
                inb = [r for r in sub if r["tilt_deg"] <= T + 0.05]
                if inb:
                    w = min(inb, key=lambda r: r["tripod"])
                    e[f"{T:g}"] = {"min_tripod": w["tripod"], "downhill_deg": w["downhill_deg"],
                                   "tilt_deg": w["tilt_deg"]}
            env[arm][f"{ext:g}"] = e
            for gate, dst in ((TRIPOD_START_GATE_M, start), (TRIPOD_CONTINUE_GATE_M, cont)):
                lim = None
                for T in TILT_BOUNDS_DEG:
                    if f"{T:g}" in e and e[f"{T:g}"]["min_tripod"] >= gate:
                        lim = T
                    else:
                        break
                dst[arm][f"{ext:g}"] = lim
    park = [v for v in start.get("PARK", {}).values()]
    park_c = [v for v in cont.get("PARK", {}).values()]
    return {"worst_row": _row_key(worst),
            "unstable_tripod_rows": sum(1 for r in rows if r["tripod"] < 0.0),
            "rows_below_start_gate": sum(1 for r in rows if r["tripod"] < TRIPOD_START_GATE_M),
            "envelope": env, "start_gate_max_tilt_deg": start, "continue_gate_max_tilt_deg": cont,
            "start_gate_max_tilt_park_deg": (None if not park or any(v is None for v in park) else min(park)),
            "continue_gate_max_tilt_park_deg": (None if not park_c or any(v is None for v in park_c) else min(park_c))}


def cmd_matrix(args):
    doc = load_json()
    fr = doc.get("frozen") or {}
    cradles = [{"tool": c["tool"], "pos": c["pos"]} for c in (fr.get("rack") or {}).get("cradles", [])]
    q, _ = worst_t4_pose(fr, 1.8) if fr else (None, None)
    cache: dict = {}
    rows = []
    t0 = time.time()
    heading_rows = []
    arms = [a for a in ("PARK", "T4") if a == "PARK" or q is not None]
    for arm in arms:
        kw = {"arm_q": q if arm == "T4" else None, "fruit_kg": 1.8 if arm == "T4" else 0.0, "cradles": cradles,
              "model_cache": cache}
        for ext in MATRIX_EXT:
            for pitch in MATRIX_PITCH_DEG:
                for roll in MATRIX_ROLL_DEG:
                    rows.append(matrix_case(ext, roll, pitch, **kw))
            for tilt in MATRIX_TILTS_DEG:
                for dh in MATRIX_DOWNHILL_DEG:
                    heading_rows.append(matrix_case(ext, 0.0, 0.0, tilt_deg=tilt, downhill_deg=dh, **kw))
    # shipped robot alone (no mast, rack, tools, basket), flat: the reference the OS 2.1 probe quoted (0.083 m)
    base_spec = mujoco.MjSpec.from_file(ROBOT_XML)
    for n in ("ridge_1", "ridge_2"):
        base_spec.delete(base_spec.geom(n))
    m0 = base_spec.compile()
    d0 = settle(m0, 1.5)
    baseline = {**margins(m0, d0), "mass_kg": round(float(m0.body_subtreemass[1]), 3)}
    flat_up = [r for r in rows if r["roll_deg"] == 0 and r["pitch_deg"] == 0 and r["ext"] > 0 and r["arm"] == "PARK"]
    every = rows + heading_rows
    out = {"meta": run_meta("matrix"), "inputs": block_inputs(doc, "matrix"), "rows": rows,
           "heading_rows": heading_rows,
           "wall_s": round(time.time() - t0, 1), "baseline_robot": baseline,
           "convention": "pitch > 0 = nose downhill (gravity +x), pitch < 0 = rear / mast downhill; roll > 0 = rack "
                         "(right) downhill; downhill_deg = fall line in the robot frame (0 nose, 90 left, 180 rear, "
                         "270 right / rack)",
           "flat_crown_up_min_tripod": min(r["tripod"] for r in flat_up) if flat_up else None,
           "min_hexa": min(r["hexa"] for r in every), "any_fell": any(r["fell"] for r in every)}
    out.update(matrix_summary(every))
    out["layout_vs_budget"] = mast_budget()
    out["crown_up_walking"] = ("allowed per gate table" if (out["flat_crown_up_min_tripod"] or 0) >= TRIPOD_START_GATE_M
                               else "dropped everywhere: rule 7 stop-and-go is the default (OS 2.3)")
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "heading_rows")}, indent=1))
    if args.write:
        update_json({"matrix": out})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("candidates", "sweep", "load", "creep", "matrix"))
    ap.add_argument("--write", action="store_true", help="write the result block into the candidates JSON")
    args = ap.parse_args(argv)
    {"candidates": cmd_candidates, "sweep": cmd_sweep, "load": cmd_load, "creep": cmd_creep,
     "matrix": cmd_matrix}[args.cmd](args)


if __name__ == "__main__":
    main()
