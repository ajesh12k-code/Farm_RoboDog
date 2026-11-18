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
    [...]    joints of work-package modules (atharv_nandi/sim/world/modules/*.py), appended in compose ORDER; look every
             index up by name (atharv_nandi/core/model_layout.py), never by literal

Composition (WP0): _build_spec() is world.compose.build(): robot, then the built-in modules "robot" (0),
"legacy_crops" (10) and "legacy_terrain" (40) defined below, then every registered atharv_nandi/sim/world/modules/*.py in
ORDER, then the generated home keyframe. The legacy world compiles byte-identical to the pre-WP0 builder.

Environment knobs (read at load time; all processes in one container see the same values):
    NANDI_WORLD        world name (default legacy); selects config/fields/<world>.json when present (WP4)
    NANDI_CROPS        total crop count incl. crop_item_1 (default 6, min 1)
    NANDI_TERRAIN      flat | ridges | rough | orchard   (default rough)
    NANDI_SEED         integer seed for crop scatter + rough bumps + orchard basins (default 7)
    NANDI_DT           physics timestep override (default: the MJCF 0.002 s). Only 0.002..0.004 is accepted;
                       0.004 is for STANDING and stop-and-shoot scans only: it is not
                       gated for walking (MEASURED 2026-11-17, vx 1 / yaw 1, 10 kg: 1-5 calf-floor ticks, foot
                       penetration up to 10.2 mm; 0 ticks and <= 8.4 mm at 0.002). Walking / RL runs use 0.002.
    NANDI_PAYLOAD_KG   > 0: welded basket payload mock (box 0.30 x 0.20 x 0.06 m at site basket_payload,
                       no collision, no joint) of that mass (default off)
    NANDI_MAST_H       mast height in m (0 = none, default): welded mast mock at site mast_base, tube
                       MAST_TUBE_KG_PER_M (ASSUMED) + a head body carrying site mast_head_ring; sensors
                       mast_base_f / mast_base_t on site mast_base_ft inside the mast body (no collision, no joint)
    NANDI_MAST_HEAD_KG mast head mass in kg (default 1.5, the rated head load)
    NANDI_BUND_H       orchard only: override every basin bund height (m); default seeded 0.10..0.15
None of the mocks adds a joint: nq, the qpos layout and crop_item_1 at qpos[32] never change.
"""
import math
import os
from typing import List, Optional, Tuple

import mujoco
import numpy as np

from atharv_nandi import paths
from atharv_nandi.core import model_layout
from atharv_nandi.sim.world import compose

ROBOT_XML = os.path.join(paths.ROBOT_MODEL_DIR, "atharv_nandi.xml")
CROP_RADIUS = 0.035
CROP_QPOS_START = 32  # crop_item_1 free joint starts here; crop k (1-based) at 32 + 7*(k-1)


# Crop contact (ground placeholder sphere): condim 6 adds rolling/torsional friction so a released crop
# stops instead of rolling away (MEASURED: free-rolling drift 200 mm -> 2.5 mm, contact audit c98754d).
CROP_MASS_KG = 0.15
CROP_CONDIM = 6
CROP_FRICTION = (0.8, 0.05, 0.02)   # ASSUMED sliding / torsional / rolling

# Payload mock (NANDI_PAYLOAD_KG): a welded, non-colliding box at site basket_payload.
PAYLOAD_HALF = (0.15, 0.10, 0.03)   # 0.30 x 0.20 x 0.06 m (spec section 4)
# Mast mock (NANDI_MAST_H / NANDI_MAST_HEAD_KG).
MAST_TUBE_KG_PER_M = 1.0            # ASSUMED (the audits bracketed 0.45 and 1.5 kg/m)
MAST_TUBE_RADIUS = 0.025            # ASSUMED 50 mm tube
MAST_HEAD_KG_DEFAULT = 1.5          # rated mast-head load (spec section 5)
DT_RANGE = (0.002, 0.004)           # NANDI_DT accepted range (s)
GROUND_SOLREF = (0.01, 1.0)         # hard orchard ground (MEASURED 0.70 mm standing penetration, contact audit)

# Orchard scene (NANDI_TERRAIN=orchard). Rows run along world x; the robot starts in the alley at y = 0.
# Spacings: mango high-density 3 x 2.5 to 5 x 5 m (CITED krishisewa.com "High density and meadow orchard
# planting system in fruit crops", 2026-11-17); guava 3-6 m ASSUMED; papaya 1.5 x 1.5 m and banana 1.8 x 1.8 m
# conventional (CITED ICAR Indian Farming, epubs.icar.org.in IndFarm article 114154, 2026-11-17); the project
# brief ranges (mango 2.5-5, guava 3-6, papaya 1.8-2.5, banana 1.5-2 m) are used as ASSUMED defaults below.
ORCHARD_ROW_SPACING = 5.0           # m between tree rows (mango HDP 5 x 2.5 m, ASSUMED default)
ORCHARD_PLANT_SPACING = 2.5         # m between trees in a row
ORCHARD_ROWS_Y = (-2.5, 2.5)        # the two rows either side of the starting alley
ORCHARD_TREES_X = tuple(-2.5 + ORCHARD_PLANT_SPACING * k for k in range(6))
TREE_TRUNK_RADIUS = 0.10            # spec placeholder trunk: cylinder r 0.10, h 1.0
TREE_TRUNK_HEIGHT = 1.0
BASIN_RADIUS_RANGE = (1.0, 1.5)     # circular basin (bund ring centre line) radius, ASSUMED
BUND_HEIGHT_RANGE = (0.10, 0.15)    # bund crest height, ASSUMED
# Bund profile: a circular-cap mound whose base edge slope is BUND_EDGE_DEG (ASSUMED until cited: farm basin bunds
# are earth mounds with gentle sides). The first version used a fixed 0.20 m half width, so the edge slope was
# asin(2hw / (w^2 + h^2)) = 53 deg at h 0.10 and 74 deg at 0.15 (DERIVED), steeper than the foot friction cone
# atan(0.8) = 38.7 deg: AT9 graded the gait against a wall (review F7). Half width now follows the height.
BUND_EDGE_DEG = 30.0


def bund_half_width(h: float, edge_deg: float = BUND_EDGE_DEG) -> float:
    """Half base width (m) of a circular-cap bund of crest height h whose base edge slope is edge_deg.

    From sin(edge) = 2 h w / (w^2 + h^2): w = h (1 + cos edge) / sin edge (0.373 m at h 0.10, 0.560 m at 0.15)."""
    a = math.radians(edge_deg)
    return h * (1.0 + math.cos(a)) / math.sin(a)


BUND_SEGMENTS = 24
LATERAL_RADIUS = 0.008              # 16 mm drip lateral
LATERAL_OFFSET = 0.5                # lateral runs along the row this far (m) on the alley side of the trunks, ASSUMED
CROSS_SLOPE_DEG = 3.0               # orchard cross-slope (rises toward +y), ASSUMED


def scene_config() -> Tuple[int, str, int]:
    n = max(1, int(os.environ.get("NANDI_CROPS", "6")))
    terrain = os.environ.get("NANDI_TERRAIN", "rough").strip().lower()
    if terrain not in ("flat", "ridges", "rough", "orchard"):
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


# Walkability budget (tripod gait, step_height_scale 1.0): the 0.44 rad calf flex of
# atharv_nandi/core/tripod_gait.py (LIFT_CALF_RAD, lift ~ cos(phase)) raises the foot ~7 cm at mid-swing,
# only ~3.5 cm a third of the swing either side, and ~0 at lift-off/touch-down. The foot is a 3 cm sphere, so any vertical edge taller than 3 cm is a
# wall the foot stubs on. Hence: ridges are smooth mounds (slope <= ~25 deg, which the sphere rolls
# up with friction 0.8, atan 0.8 = 39 deg), and clods stay below the foot radius.
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


def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default
    return v if math.isfinite(v) else default


def _site(spec: mujoco.MjSpec, name: str):
    for s in spec.sites:
        if s.name == name:
            return s
    return None


def _add_payload_mock(spec: mujoco.MjSpec, kg: float):
    """Welded basket payload: child body of dorsal_basket at site basket_payload, no joint, no collision."""
    site = _site(spec, "basket_payload")
    b = spec.body("dorsal_basket").add_body(name="payload_mock", pos=list(site.pos))
    b.add_geom(name="payload_mock_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(PAYLOAD_HALF), mass=kg,
               contype=0, conaffinity=0, rgba=[0.55, 0.35, 0.15, 0.6])


def _add_mast_mock(spec: mujoco.MjSpec, height: float, head_kg: float):
    """Welded mast: tube body at site mast_base + head body with site mast_head_ring. The base force/torque
    sensors sit on a site inside the mast body (a trunk site cannot carry them: the trunk's parent joint is the
    freejoint), so they read the mast-to-trunk load."""
    base = _site(spec, "mast_base")
    mast = spec.body("trunk").add_body(name="mast_mock", pos=list(base.pos))
    mast.add_geom(name="mast_tube", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[MAST_TUBE_RADIUS, height / 2.0, 0],
                  pos=[0, 0, height / 2.0], mass=max(1e-3, MAST_TUBE_KG_PER_M * height),
                  contype=0, conaffinity=0, rgba=[0.6, 0.6, 0.62, 1])
    mast.add_site(name="mast_base_ft", pos=[0, 0, 0], size=[0.006, 0, 0], rgba=[1, 1, 1, 0])
    head = mast.add_body(name="mast_head", pos=[0, 0, height])
    head.add_geom(name="mast_head_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.06, 0, 0],
                  mass=max(1e-3, head_kg), contype=0, conaffinity=0, rgba=[0.1, 0.1, 0.1, 1])
    head.add_site(name="mast_head_ring", pos=[0, 0, 0], size=[0.006, 0, 0], rgba=[1, 1, 1, 0])
    spec.add_sensor(name="mast_base_f", type=mujoco.mjtSensor.mjSENS_FORCE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="mast_base_ft")
    spec.add_sensor(name="mast_base_t", type=mujoco.mjtSensor.mjSENS_TORQUE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="mast_base_ft")


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return out


def orchard_slope_quat() -> np.ndarray:
    a = math.radians(CROSS_SLOPE_DEG)
    return np.array([math.cos(a / 2.0), math.sin(a / 2.0), 0.0, 0.0])  # about +x: ground rises toward +y


def orchard_ground_z(y: float) -> float:
    """Ground height of the tilted orchard plane at world y (the plane passes through the origin)."""
    return y * math.tan(math.radians(CROSS_SLOPE_DEG))


def orchard_layout(seed: int, bund_h: Optional[float] = None) -> List[dict]:
    """Trees as dicts {x, y, basin_r, bund_h} (flat local frame; y across the slope)."""
    rng = np.random.default_rng(seed + 2000)
    trees = []
    for ry in ORCHARD_ROWS_Y:
        for tx in ORCHARD_TREES_X:
            r = float(rng.uniform(*BASIN_RADIUS_RANGE))
            h = float(rng.uniform(*BUND_HEIGHT_RANGE)) if bund_h is None else float(bund_h)
            trees.append({"x": tx, "y": ry, "basin_r": r, "bund_h": h})
    return trees


def _add_orchard(spec: mujoco.MjSpec, seed: int):
    """Static orchard: tilted floor, tree trunks, circular basin bunds, drip laterals. World geoms only."""
    q_slope = orchard_slope_quat()
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, q_slope)
    rot = rot.reshape(3, 3)
    floor = spec.geom("floor")
    floor.quat = q_slope.tolist()
    world = spec.worldbody
    bund_h = os.environ.get("NANDI_BUND_H", "").strip()
    trees = orchard_layout(seed, float(bund_h) if bund_h else None)

    def add(name, local_pos, local_quat=(1, 0, 0, 0), **kw):
        world.add_geom(name=name, pos=(rot @ np.asarray(local_pos, dtype=np.float64)).tolist(),
                       quat=_quat_mul(q_slope, local_quat).tolist(), **kw)

    soil = [0.30, 0.22, 0.15, 1]
    for i, t in enumerate(trees):
        add(f"tree_{i}", [t["x"], t["y"], TREE_TRUNK_HEIGHT / 2.0], type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[TREE_TRUNK_RADIUS, TREE_TRUNK_HEIGHT / 2.0, 0], rgba=[0.35, 0.24, 0.12, 1])
        # bund ring: rounded mounds (cylinder caps, like the ridges) along tangential chords
        h, r = t["bund_h"], t["basin_r"]
        w = bund_half_width(h)
        radius = (w * w + h * h) / (2.0 * h)
        half_len = math.pi * r / BUND_SEGMENTS * 1.08
        for k in range(BUND_SEGMENTS):
            th = 2.0 * math.pi * k / BUND_SEGMENTS
            cx, cy = t["x"] + r * math.cos(th), t["y"] + r * math.sin(th)
            # cylinder axis (local z) along the tangent (-sin th, cos th, 0)
            tang = th + math.pi / 2.0
            q_tan = [math.cos(tang / 2.0), 0.0, 0.0, math.sin(tang / 2.0)]
            q_lay = [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]  # local z -> x
            add(f"bund_{i}_{k}", [cx, cy, h - radius], _quat_mul(q_tan, q_lay),
                type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, half_len, 0], rgba=soil)
    x_lo, x_hi = ORCHARD_TREES_X[0] - 3.0, ORCHARD_TREES_X[-1] + 3.0
    for j, ry in enumerate(ORCHARD_ROWS_Y):
        ly = ry - math.copysign(LATERAL_OFFSET, ry)  # alley side of the trunks
        add(f"drip_lateral_{j}", [(x_lo + x_hi) / 2.0, ly, LATERAL_RADIUS],
            [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[LATERAL_RADIUS, (x_hi - x_lo) / 2.0, 0], rgba=[0.05, 0.05, 0.05, 1])


# ---------------------------------------------------------------------- composition (WP0)
# The legacy world is three built-in compose modules (ORCHARD-MVP-SPEC 3.1 ORDER bands). Work packages add
# their own modules in atharv_nandi/sim/world/modules/<name>.py and never edit this file.
def _robot_options() -> dict:
    dt = os.environ.get("NANDI_DT", "").strip()
    return {"dt": dt, "payload_kg": _env_float("NANDI_PAYLOAD_KG", 0.0), "mast_h": _env_float("NANDI_MAST_H", 0.0),
            "mast_head_kg": _env_float("NANDI_MAST_HEAD_KG", MAST_HEAD_KG_DEFAULT),
            "bund_h": os.environ.get("NANDI_BUND_H", "").strip()}


def world_config() -> dict:
    """Every knob that changes the composed world: NANDI_WORLD, NANDI_CROPS, NANDI_TERRAIN, NANDI_SEED + mocks."""
    n, terrain, seed = scene_config()
    cfg = {"n_crops": n, "terrain": terrain, "seed": seed}
    cfg.update(_robot_options())
    cfg["world"] = compose.world_name()
    return cfg


def _attach_robot_options(spec: mujoco.MjSpec, field: dict, ctx) -> None:
    """Order 0 (robot): timestep override and the joint-less payload / mast mocks (no qpos impact)."""
    cfg = ctx.config
    dt = cfg.get("dt", "")
    if dt:
        v = float(dt)
        if not (DT_RANGE[0] <= v <= DT_RANGE[1]):
            raise ValueError(f"NANDI_DT={dt} outside {DT_RANGE}")
        spec.option.timestep = v
    if cfg.get("payload_kg", 0.0) > 0.0:
        _add_payload_mock(spec, cfg["payload_kg"])
    if cfg.get("mast_h", 0.0) > 0.0:
        _add_mast_mock(spec, cfg["mast_h"], cfg.get("mast_head_kg", MAST_HEAD_KG_DEFAULT))


def _attach_legacy_crops(spec: mujoco.MjSpec, field: dict, ctx) -> None:
    """Order 10: crop_item_2..N free bodies (crop_item_1 is in the robot MJCF) + their home-key qpos."""
    cfg = ctx.config
    n, seed, terrain = int(cfg["n_crops"]), int(cfg["seed"]), cfg["terrain"]
    extra_qpos = []
    for k, (x, y) in enumerate(crop_layout(n, seed), start=2):
        z = CROP_RADIUS + (orchard_ground_z(y) if terrain == "orchard" else 0.0)
        b = spec.worldbody.add_body(name=f"crop_item_{k}", pos=[x, y, z])
        b.add_freejoint(name=f"crop_freejoint_{k}")
        b.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[CROP_RADIUS, 0, 0], mass=CROP_MASS_KG,
            rgba=[0.9, 0.15, 0.15, 1], condim=CROP_CONDIM, friction=list(CROP_FRICTION),
        )
        extra_qpos += [x, y, z, 1, 0, 0, 0]
    if extra_qpos:
        key = spec.keys[0]
        key.qpos = np.concatenate([np.asarray(key.qpos, dtype=np.float64), np.asarray(extra_qpos)])


def _attach_legacy_terrain(spec: mujoco.MjSpec, field: dict, ctx) -> None:
    """Order 40: ridges (shaped or removed), clods or the orchard scene, then hard-ground contact on every
    static world geom that exists at this point (later modules set their own contact)."""
    cfg = ctx.config
    n, seed, terrain = int(cfg["n_crops"]), int(cfg["seed"]), cfg["terrain"]
    if terrain in ("flat", "orchard"):
        for name, _x, _h, _w in RIDGES:
            g = spec.geom(name)
            if g is not None:
                spec.delete(g)
    else:
        _shape_ridges(spec)
    if terrain == "rough":
        _add_clods(spec, seed, crop_layout(n, seed))
    elif terrain == "orchard":
        _add_orchard(spec, seed)
    for g in spec.worldbody.geoms:
        g.solref = list(GROUND_SOLREF)


compose.register_module("robot", compose.ORDER["robot"], _attach_robot_options, version="1",
                        fingerprint=lambda field, ctx: "dt={dt},payload={payload_kg:g},mast={mast_h:g}/{mast_head_kg:g}"
                        .format(**ctx.config))
compose.register_module("legacy_crops", compose.ORDER["legacy_crops"], _attach_legacy_crops, version="1",
                        fingerprint=lambda field, ctx: f"n={ctx.config['n_crops']}")
compose.register_module("legacy_terrain", compose.ORDER["terrain"], _attach_legacy_terrain, version="1",
                        fingerprint=lambda field, ctx: f"terrain={ctx.config['terrain']},bund={ctx.config['bund_h']}")


def build_world() -> "compose.Composed":
    """Compose the live world from the environment (robot, legacy modules, every atharv_nandi/sim/world/modules/*.py)."""
    cfg = world_config()
    return compose.build(cfg["world"], cfg["seed"], ROBOT_XML, cfg)


def world_identity() -> Tuple[str, str]:
    """(world, world_id) of the world this environment composes, without building it."""
    cfg = world_config()
    wid = compose.identify(cfg["world"], cfg["seed"], cfg)[0]
    return cfg["world"], wid


def _build_spec() -> mujoco.MjSpec:
    return build_world().spec


def load_farm_model() -> mujoco.MjModel:
    """Compile the full live world. Call once per process.

    Also records the world's ``Layout`` as this process's current layout (``model_layout.current()``) and
    refuses a model that breaks the frozen robot table (design rule 1)."""
    w = build_world()
    model = w.spec.compile()
    lay = model_layout.assert_frozen(model_layout.Layout.from_model(model, world=w.world, world_id=w.world_id))
    model_layout.set_current(lay)
    return model


def crop_body_names(model: mujoco.MjModel) -> List[str]:
    names, k = [], 1
    while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"crop_item_{k}") != -1:
        names.append(f"crop_item_{k}")
        k += 1
    return names


def crop_qpos_slice(k: int) -> slice:
    """qpos slice of crop k's free joint (k is 1-based). Valid in every world: compose keeps the legacy crops
    contiguous right after the robot (``model_layout.frozen_violations`` enforces it)."""
    s = CROP_QPOS_START + 7 * (k - 1)
    return slice(s, s + 7)
