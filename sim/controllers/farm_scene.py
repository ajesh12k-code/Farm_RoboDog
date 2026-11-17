"""Single source of truth for the live Atharv Nandi world (robot + crop field + terrain).

Every live node (sim_runner, locomotion, harvest IK, work envelope, RL inference, web bridge)
MUST build its MjModel through ``load_farm_model()`` so that ``nq`` and the qpos layout agree
across processes that exchange ``/nandi/sim_qpos``.

qpos layout (stable prefix, identical to the raw robot MJCF so the RL policy keeps working):
    [0:7]    trunk free joint (x y z qw qx qy qz)
    [7:14]   gantry, arm_yaw, shoulder, elbow, wrist, finger_l, finger_r
    [14:32]  18 leg joints (FR, FL, MR, ML, RR, RL) x (hip, thigh, calf)
    [32:39]  crop_item_1 free joint
    [39:...] crop_item_2..crop_item_N free joints, 7 each, in order

Environment knobs (read at load time; all processes in one container see the same values):
    NANDI_CROPS    total crop count incl. crop_item_1 (default 6, min 1)
    NANDI_TERRAIN  flat | ridges | rough   (default rough)
    NANDI_SEED     integer seed for crop scatter + rough bumps (default 7)
"""
import math
import os
from typing import List, Tuple

import mujoco
import numpy as np

ROBOT_XML = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "AtharvNandi", "atharv_nandi.xml")
)
CROP_RADIUS = 0.035
CROP_QPOS_START = 32  # crop_item_1 free joint starts here; crop k (1-based) at 32 + 7*(k-1)


def scene_config() -> Tuple[int, str, int]:
    n = max(1, int(os.environ.get("NANDI_CROPS", "6")))
    terrain = os.environ.get("NANDI_TERRAIN", "rough").strip().lower()
    if terrain not in ("flat", "ridges", "rough"):
        terrain = "rough"
    seed = int(os.environ.get("NANDI_SEED", "7"))
    return n, terrain, seed


def crop_layout(n: int, seed: int) -> List[Tuple[float, float]]:
    """Deterministic (x, y) for crops 2..n: two loose rows ahead of the robot, avoiding ridges."""
    rng = np.random.default_rng(seed)
    pts = []
    for k in range(n - 1):
        row_y = 0.45 if k % 2 == 0 else -0.45
        x = 1.0 + 0.55 * k + rng.uniform(-0.12, 0.12)
        # keep crops off the two fixed ridges (x≈1.4 and x≈2.2) so they sit flat
        for ridge_x in (1.4, 2.2):
            if abs(x - ridge_x) < 0.25:
                x = ridge_x + 0.3
        pts.append((float(x), float(row_y + rng.uniform(-0.1, 0.1))))
    return pts


# Walkability budget (tripod gait, step_height_scale 1.0): the 0.42 rad calf flex of
# controllers/tripod_gait.py (LIFT_CALF_RAD, lift ~ cos(phase)) raises the foot ~7 cm at mid-swing,
# only ~3.5 cm a third of the swing either side, and ~0 at lift-off/touch-down. The foot is a 3 cm sphere, so any vertical edge taller than 3 cm is a
# wall the foot stubs on. Hence: ridges are smooth mounds (slope <= ~25 deg, which the sphere rolls
# up with friction 1.2), and clods stay below the foot radius.
RIDGES = (  # (geom name in the robot MJCF, centre x, crest height, half width)
    ("ridge_1", 1.4, 0.030, 0.15),
    ("ridge_2", 2.2, 0.040, 0.18),
)
CLOD_HEIGHT_RANGE = (0.010, 0.024)   # full clod height (m); < foot radius 0.03
CLOD_CROP_CLEARANCE = 0.08           # keep clods this far (m) from any crop footprint


def _shape_ridges(spec: mujoco.MjSpec):
    """Turn the robot MJCF's square ridge boxes into rounded mounds of the same footprint.

    Each mound is a long cylinder along y whose axis sits below ground so only a shallow cap
    pokes out: radius R = (w^2 + h^2) / (2h) for crest height h and half width w.
    """
    for name, x, h, w in RIDGES:
        g = spec.geom(name)
        if g is None:
            continue
        radius = (w * w + h * h) / (2.0 * h)
        half_len = float(g.size[1])
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [radius, half_len, 0.0]
        g.pos = [x, 0.0, h - radius]
        g.quat = [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]  # cylinder axis z -> world y


def _add_clods(spec: mujoco.MjSpec, seed: int, crop_xy: List[Tuple[float, float]]):
    """Static low clods in a patch beyond the crop rows: geoms only, no qpos impact."""
    rng = np.random.default_rng(seed + 1000)
    lo, hi = CLOD_HEIGHT_RANGE
    for i in range(40):
        # draw every value first so the layout is independent of which clods get skipped
        x, y = rng.uniform(3.6, 6.5), rng.uniform(-1.2, 1.2)
        half_h = rng.uniform(lo, hi) / 2.0
        sx, sy = rng.uniform(0.06, 0.16), rng.uniform(0.06, 0.16)
        yaw = rng.uniform(0, np.pi)
        reach = math.hypot(sx, sy) + CROP_RADIUS + CLOD_CROP_CLEARANCE
        if any(math.hypot(x - cx, y - cy) < reach for cx, cy in crop_xy):
            continue
        spec.worldbody.add_geom(
            name=f"clod_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, y, half_h], size=[sx, sy, half_h],
            quat=[math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
            rgba=[0.32, 0.23, 0.15, 1],
        )


def _build_spec() -> mujoco.MjSpec:
    n, terrain, seed = scene_config()
    spec = mujoco.MjSpec.from_file(ROBOT_XML)
    world = spec.worldbody

    if terrain == "flat":
        for name, _x, _h, _w in RIDGES:
            g = spec.geom(name)
            if g is not None:
                spec.delete(g)
    else:
        _shape_ridges(spec)

    layout = crop_layout(n, seed)
    extra_qpos = []
    for k, (x, y) in enumerate(layout, start=2):
        b = world.add_body(name=f"crop_item_{k}", pos=[x, y, CROP_RADIUS])
        b.add_freejoint(name=f"crop_freejoint_{k}")
        b.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[CROP_RADIUS, 0, 0], mass=0.15,
            rgba=[0.9, 0.15, 0.15, 1], friction=[1.2, 0.05, 0.01],
        )
        extra_qpos += [x, y, CROP_RADIUS, 1, 0, 0, 0]

    if terrain == "rough":
        _add_clods(spec, seed, layout)

    if extra_qpos:
        key = spec.keys[0]
        key.qpos = np.concatenate([np.asarray(key.qpos, dtype=np.float64), np.asarray(extra_qpos)])
    return spec


def load_farm_model() -> mujoco.MjModel:
    """Compile the full live world. Call once per process."""
    return _build_spec().compile()


def crop_body_names(model: mujoco.MjModel) -> List[str]:
    names, k = [], 1
    while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"crop_item_{k}") != -1:
        names.append(f"crop_item_{k}")
        k += 1
    return names


def crop_qpos_slice(k: int) -> slice:
    """qpos slice of crop k's free joint (k is 1-based)."""
    s = CROP_QPOS_START + 7 * (k - 1)
    return slice(s, s + 7)
