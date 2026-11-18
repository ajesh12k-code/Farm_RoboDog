"""Pure (ROS-free) logic behind the Atharv Nandi browser bridge (atharv_nandi/nodes/web_bridge.py).

Scene + pose serialization, client-message validation, cmd_vel mapping, the held-walk
dead-man, and safe static-file resolution. Nothing here imports rclpy, so CI can test it.

WebSocket wire format (JSON text frames on /ws):
  server -> client
    {"type": "scene", ...}       once per connection, see build_scene()
    {"type": "state", "poses": [x y z qw qx qy qz per streamed body] | null,
     "telemetry": {...} | null, "game": {...} | null, "harvest": {...} | null,
     "loco": {...} | null, "sim_age_s": float | null, "hold": bool,
     "hidden": [geom id], "styles": [[id, rgba, size, pos, quat]] | null, "relay": bool}     ~30 Hz
     (poses cover the moving bodies only; static ones are placed once from scene["static_bodies"])
    {"type": "envelope", "data": {"canopy_segments": [...], "floor_segments": [...]} | null}
    {"type": "result", "name": str, "success": bool, "message": str}
    {"type": "error", "message": str}
    {"type": "pong", "t": float}
  client -> server
    {"type": "cmd", "vx": f, "vy": f, "yaw": f, "hold": bool}     each in [-1, 1]
    {"type": "service", "name": one of SERVICES}
    {"type": "gait_params", "values": [speed, step_height, freq_hz, balance_gain]}
    {"type": "manual_arm", "values": [gantry, yaw, shoulder, elbow, wrist, finger_l, finger_r]}
    {"type": "envelope", "show": bool}
    {"type": "ping", "t": f}
"""
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlsplit

import mujoco
import numpy as np

from atharv_nandi.core.farm_scene import CROP_RADIUS, crop_body_names
from atharv_nandi.core.telemetry import parse_qpos_telemetry

# MuJoCo geom types the browser can draw, with MuJoCo's size conventions:
#   plane: [half_x, half_y, grid] (0 = infinite)   sphere: [r]   ellipsoid: [rx, ry, rz]
#   capsule / cylinder: [r, half_length] along local z   box: [half_x, half_y, half_z]
#   mesh: vertices / faces in scene["meshes"][<geom "mesh">]   hfield: grid in scene["hfields"][<geom "hfield">]
GEOM_TYPES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}
# Geom groups drawn in the browser (WP10): 0 wood, terrain and fixtures, 1 crops, fruit and actors, 2 the robot,
# 3 foliage (WP23), 4 decals, drip laterals and rope lines (WP4 / WP23). Group 5 is query-only (foliage envelope,
# the safety-invisible copies of laterals and ropes) and is never drawn.
VISIBLE_GROUPS = 5
_DEFAULT_GEOM_RGBA = (0.5, 0.5, 0.5, 1.0)
HFIELD_MAX_SIDE = 128   # heightfield samples per side sent to the browser (max-pooled, so no bump is lost)
MESH_DIGITS = 4

# qpos[7:14] / ctrl[0:7]; the gripper fingers are the last two
ARM_JOINTS = (
    "gantry_slide", "arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist_pitch",
    "finger_left_joint", "finger_right_joint",
)
GAIT_LIMITS = ((0.2, 1.5), (0.5, 2.0), (0.8, 3.0), (0.0, 2.0))
GAIT_DEFAULTS = (1.0, 1.0, 1.8, 1.0)
GAIT_NAMES = ("speed_scale", "step_height_scale", "freq_hz", "balance_gain")

# Browser-facing service name -> std_srvs/Trigger service on the ROS graph
SERVICE_TOPICS = {
    "stop": "/nandi/stop_motion",
    "release_arm": "/nandi/release_arm",
    "auto_harvest": "/nandi/auto_harvest",
    "toggle_mode": "/nandi/toggle_locomotion_mode",
    "reset_world": "/nandi/reset_world",
    "work_envelope": "/compute_work_envelope",
}
SERVICES = tuple(SERVICE_TOPICS)

MAX_MSG_CHARS = 4096
POSE_DIGITS = 4  # 0.1 mm / 1e-4 quaternion units: plenty for display, keeps frames small
ZERO_CMD_EPS = 1e-3

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}


class BadMessage(ValueError):
    """A client message that fails validation; the reason is safe to send back."""


# ---------------------------------------------------------------- scene / poses

def _r(values, digits: int) -> List[float]:
    return [round(float(v), digits) for v in values]


def _geom_rgba(model: mujoco.MjModel, g: int) -> np.ndarray:
    """MuJoCo's rule: a material colours the geom unless the geom's own rgba was changed."""
    rgba = model.geom_rgba[g]
    matid = int(model.geom_matid[g])
    if matid >= 0 and np.allclose(rgba, _DEFAULT_GEOM_RGBA):
        return model.mat_rgba[matid]
    return rgba


def arm_spec(model: mujoco.MjModel) -> Dict:
    """Joint names, limits and home pose of the 7 arm DOF (slider ranges for the browser)."""
    low, high = [], []
    for name in ARM_JOINTS:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise ValueError(f"arm joint '{name}' missing from model")
        low.append(float(model.jnt_range[j, 0]))
        high.append(float(model.jnt_range[j, 1]))
    home = model.key_qpos[0][7:14] if model.nkey > 0 else model.qpos0[7:14]
    return {"names": list(ARM_JOINTS), "low": low, "high": high, "home": _r(home, 4)}


def body_moves(model: mujoco.MjModel, b: int) -> bool:
    """True if body b can move: it or an ancestor has a joint or is a mocap body (actors), or it is a robot
    body moved by a sim-state provider (the WP24 mast sections ride the trunk's free joint, so they count)."""
    while b > 0:
        if int(model.body_jntnum[b]) > 0 or int(model.body_mocapid[b]) >= 0:
            return True
        b = int(model.body_parentid[b])
    return False


def _mesh_block(model: mujoco.MjModel, mid: int) -> Dict:
    v0, nv = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    f0, nf = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    verts = np.round(model.mesh_vert[v0:v0 + nv], MESH_DIGITS).ravel().tolist()
    faces = model.mesh_face[f0:f0 + nf].astype(int).ravel().tolist()
    return {"v": verts, "f": faces}


def _hfield_block(model: mujoco.MjModel, hid: int, max_side: int = HFIELD_MAX_SIDE) -> Dict:
    """Decimated heightfield: ``z = data * size[2]`` on a (rows x cols) grid over [-sx, sx] x [-sy, sy] in the geom
    frame (MuJoCo's convention, row = y). Each output sample is the max of its source cell, so basins and bunds
    keep their rims."""
    nrow, ncol = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
    adr = int(model.hfield_adr[hid])
    h = np.asarray(model.hfield_data[adr:adr + nrow * ncol], dtype=np.float64).reshape(nrow, ncol)
    step = max(1, int(math.ceil(max(nrow, ncol) / float(max_side))))
    rows = list(range(0, nrow, step))
    cols = list(range(0, ncol, step))
    if rows[-1] != nrow - 1:
        rows.append(nrow - 1)
    if cols[-1] != ncol - 1:
        cols.append(ncol - 1)
    out = np.empty((len(rows), len(cols)))
    for i, r in enumerate(rows):
        r1 = min(nrow, r + step)
        for j, c in enumerate(cols):
            out[i, j] = h[r:r1, c:min(ncol, c + step)].max()
    size = model.hfield_size[hid]
    return {"rows": len(rows), "cols": len(cols), "size": _r(size, 5),
            "z": np.round(out * float(size[2]), 4).ravel().tolist()}


def scene_bodies(model: mujoco.MjModel, scene: Dict) -> Tuple[List[int], List[int]]:
    """(streamed body ids, static body ids) of a scene."""
    return [b["id"] for b in scene["bodies"]], [b["id"] for b in scene.get("static_bodies", [])]


def build_scene(model: mujoco.MjModel) -> Dict:
    """Static scene description: every drawable geom (groups 0-4) in its body's local frame.

    Bodies with drawable geoms that can move are listed in ``bodies``; pose_frame() streams their world poses in
    exactly that order. Bodies that never move (trees, dock, crate: no joint and no mocap in their chain) are in
    ``static_bodies`` with their world pose, so the 30 Hz stream carries only what moves. Mesh geoms (WP23
    foliage) reference ``meshes[name] = {v: [x y z ...], f: [i j k ...]}``; heightfield geoms (WP4 terrain)
    reference ``hfields[id]``. Every geom carries its model ``id`` (``state.hidden`` lists hidden geom ids).
    """
    geoms, used = [], set()
    meshes, hfields = {}, {}
    for g in range(model.ngeom):
        gtype = GEOM_TYPES.get(int(model.geom_type[g]))
        if gtype is None or int(model.geom_group[g]) >= VISIBLE_GROUPS:
            continue
        rgba = _geom_rgba(model, g)
        styled = False
        if float(rgba[3]) <= 0.0:
            # alpha 0 outside the robot (group 2: mass-only parts) = a provider-managed geom (WP23 fruit pool slot,
            # debris): drawn hidden, its colour / size / pose follow state.styles
            if int(model.geom_group[g]) == 2 or gtype in ("plane", "hfield"):
                continue
            styled = True
        bid = int(model.geom_bodyid[g])
        geom = {
            "id": g,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or None,
            "body": bid,
            "type": gtype,
            "size": _r(model.geom_size[g], 5),
            "pos": _r(model.geom_pos[g], 5),
            "quat": _r(model.geom_quat[g], 6),  # w, x, y, z
            "rgba": _r(rgba, 3),
            "group": int(model.geom_group[g]),
        }
        if styled:
            geom["styled"] = True
        if gtype == "mesh":
            mid = int(model.geom_dataid[g])
            key = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid) or f"mesh_{mid}"
            if key not in meshes:
                meshes[key] = _mesh_block(model, mid)
            geom["mesh"] = key
        elif gtype == "hfield":
            hid = int(model.geom_dataid[g])
            if str(hid) not in hfields:
                hfields[str(hid)] = _hfield_block(model, hid)
            geom["hfield"] = str(hid)
        geoms.append(geom)
        if bid != 0:
            used.add(bid)

    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    bodies, static = [], []
    for b in sorted(used):
        row = {"id": b, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or None}
        if body_moves(model, b):
            bodies.append(row)
        else:
            row["pose"] = _r(np.hstack([data.xpos[b], data.xquat[b]]), POSE_DIGITS)
            static.append(row)
    return {
        "type": "scene",
        "nq": int(model.nq),
        "bodies": bodies,
        "static_bodies": static,
        "geoms": geoms,
        "meshes": meshes,
        "hfields": hfields,
        "styled": [g["id"] for g in geoms if g.get("styled")],
        "trunk": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk"),
        "crops": crop_body_names(model),
        "crop_radius": CROP_RADIUS,
        "arm": arm_spec(model),
        "gait": {"names": list(GAIT_NAMES), "low": [lo for lo, _ in GAIT_LIMITS],
                 "high": [hi for _, hi in GAIT_LIMITS], "default": list(GAIT_DEFAULTS)},
        "services": list(SERVICES),
    }


def pose_frame(model: mujoco.MjModel, data: mujoco.MjData, body_ids: Sequence[int],
               qpos: Sequence[float]) -> Optional[List[float]]:
    """Flat [x y z qw qx qy qz] * len(body_ids) world poses for qpos, or None if qpos is unusable."""
    try:
        q = np.asarray(qpos, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if q.shape != (model.nq,) or not np.all(np.isfinite(q)):
        return None
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    return poses_of(data, body_ids)


def poses_of(data: mujoco.MjData, body_ids: Sequence[int]) -> List[float]:
    """Flat [x y z qw qx qy qz] per body from data whose kinematics are current (e.g. after a sim_state apply)."""
    ids = np.asarray(body_ids, dtype=int)
    frame = np.hstack([data.xpos[ids], data.xquat[ids]])
    return np.round(frame, POSE_DIGITS).ravel().tolist()


def geom_styles(model: mujoco.MjModel, geom_ids: Sequence[int]) -> List[List[float]]:
    """[id, r, g, b, a, sx, sy, sz, px, py, pz, qw, qx, qy, qz] per provider-managed geom (scene["styled"])."""
    out = []
    for g in geom_ids:
        row = [int(g)] + _r(model.geom_rgba[g], 3) + _r(model.geom_size[g], 4) + _r(model.geom_pos[g], 4)
        out.append(row + _r(model.geom_quat[g], 4))
    return out


def hidden_geoms(model: mujoco.MjModel, geom_ids: Sequence[int]) -> List[int]:
    """Scene geoms a sim-state provider has hidden (alpha 0: picked fruit, cut shoots, WP23 ``hidden_geoms``)."""
    return [int(g) for g in geom_ids if float(model.geom_rgba[g][3]) <= 0.0]


def telemetry(qpos: Sequence[float]) -> Optional[Dict]:
    """Trunk position, tilt and heading (deg, + = CCW from world +x) for the HUD."""
    if len(qpos) < 7 or not all(math.isfinite(float(v)) for v in qpos[:7]):
        return None
    x, z, tilt = parse_qpos_telemetry(qpos)
    w, qx, qy, qz = (float(v) for v in qpos[3:7])
    heading = math.degrees(math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    return {"x": round(x, 3), "y": round(float(qpos[1]), 3), "z": round(z, 3),
            "tilt_deg": round(tilt, 1), "heading_deg": round(heading, 1)}


# ---------------------------------------------------------------- JSON in/out

def _finite(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def encode_message(msg: Dict) -> str:
    """Compact JSON. NaN/inf (which the browser's JSON.parse rejects) become null."""
    return json.dumps(_finite(msg), separators=(",", ":"), allow_nan=False)


def parse_status(text: str) -> Optional[Dict]:
    """A /nandi/*_status JSON string -> dict, or None if it is not a JSON object."""
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None


def _is_point(p) -> bool:
    return (isinstance(p, list) and len(p) == 3 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p))


def parse_envelope(text: str) -> Optional[Dict]:
    """/compute_work_envelope or /nandi/workspace_wireframe JSON -> segments (trunk frame).

    Returns None for "" (hidden) or malformed data; malformed individual segments are dropped.
    """
    if not text:
        return None
    obj = parse_status(text)
    if obj is None:
        return None
    out = {}
    for key in ("canopy_segments", "floor_segments"):
        segs = obj.get(key, [])
        if not isinstance(segs, list):
            return None
        out[key] = [s for s in segs if isinstance(s, list) and len(s) == 2 and all(map(_is_point, s))]
    if not out["canopy_segments"] and not out["floor_segments"]:
        return None
    return out


def _number(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise BadMessage(f"{what} must be a finite number")
    return float(v)


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, v))


def _vector(msg: Dict, n: int, low: Sequence[float], high: Sequence[float]) -> List[float]:
    vals = msg.get("values")
    if not isinstance(vals, list) or len(vals) != n:
        raise BadMessage(f"'values' must be a list of {n} numbers")
    return [_clamp(_number(v, f"values[{i}]"), low[i], high[i]) for i, v in enumerate(vals)]


def parse_client_message(raw, arm_low: Sequence[float], arm_high: Sequence[float]) -> Dict:
    """Validate + normalise one client frame. Raises BadMessage; never anything else."""
    if not isinstance(raw, str):
        raise BadMessage("binary frames are not supported")
    if len(raw) > MAX_MSG_CHARS:
        raise BadMessage("message too large")
    try:
        msg = json.loads(raw)
    except (ValueError, RecursionError):
        raise BadMessage("invalid JSON") from None
    if not isinstance(msg, dict):
        raise BadMessage("message must be a JSON object")
    kind = msg.get("type")

    if kind == "cmd":
        hold = msg.get("hold", False)
        if not isinstance(hold, bool):
            raise BadMessage("'hold' must be true or false")
        out = {k: _clamp(_number(msg.get(k, 0.0), f"'{k}'"), -1.0, 1.0) for k in ("vx", "vy", "yaw")}
        return {"type": "cmd", **out, "hold": hold}
    if kind == "service":
        name = msg.get("name")
        if name not in SERVICE_TOPICS:
            raise BadMessage(f"unknown service; expected one of {', '.join(SERVICES)}")
        return {"type": "service", "name": name}
    if kind == "gait_params":
        lows, highs = zip(*GAIT_LIMITS)
        return {"type": "gait_params", "values": _vector(msg, 4, lows, highs)}
    if kind == "manual_arm":
        return {"type": "manual_arm", "values": _vector(msg, len(ARM_JOINTS), arm_low, arm_high)}
    if kind == "envelope":
        show = msg.get("show")
        if not isinstance(show, bool):
            raise BadMessage("'show' must be true or false")
        return {"type": "envelope", "show": show}
    if kind == "ping":
        return {"type": "ping", "t": _number(msg.get("t", 0.0), "'t'")}
    raise BadMessage("unknown message type")


# ---------------------------------------------------------------- cmd_vel / hold

def cmd_to_twist(vx: float, vy: float, yaw: float, hold: bool) -> Tuple[float, float, float, float]:
    """/nandi/cmd_vel contract -> (linear.x, linear.y, linear.z, angular.z)."""
    return float(vx), float(vy), (1.0 if hold else 0.0), float(yaw)


def is_zero_cmd(vx: float, vy: float, yaw: float) -> bool:
    return max(abs(vx), abs(vy), abs(yaw)) < ZERO_CMD_EPS


RELEASE = (0.0, 0.0, 0.0, False)


class HoldTracker:
    """Owns the 'keep walking' command for the bridge. Pure: callers pass the clock in.

    Every method returns the (vx, vy, yaw, hold) to publish on cmd_vel now, or None.
    A held command is re-published by tick() until the browser releases it, its owner
    disconnects, or the browser stops refreshing it for ``refresh_timeout_s`` (dead-man
    for a frozen tab whose socket is still open).
    """

    def __init__(self, refresh_timeout_s: float = 0.5):
        self.timeout = refresh_timeout_s
        self.cmd: Optional[Tuple[float, float, float]] = None
        self.owner = None
        self.refreshed = 0.0

    @property
    def active(self) -> bool:
        return self.cmd is not None

    def _clear(self):
        self.cmd, self.owner = None, None

    def on_cmd(self, owner, vx: float, vy: float, yaw: float, hold: bool, now: float):
        if hold and not is_zero_cmd(vx, vy, yaw):
            self.cmd, self.owner, self.refreshed = (vx, vy, yaw), owner, now
            return vx, vy, yaw, True
        self._clear()
        return RELEASE if is_zero_cmd(vx, vy, yaw) else (vx, vy, yaw, False)

    def release(self):
        self._clear()
        return RELEASE

    def tick(self, now: float):
        if self.cmd is None:
            return None
        if now - self.refreshed > self.timeout:
            return self.release()
        return (*self.cmd, True)

    def on_disconnect(self, owner):
        if self.cmd is not None and self.owner is owner:
            return self.release()
        return None


# ---------------------------------------------------------------- WebSocket origin

LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def allowed_hostnames(env_value: Optional[str] = None) -> frozenset:
    """Hostnames a browser may use to reach the bridge: loopback plus NANDI_WEB_ALLOWED_HOSTS (comma list)."""
    extra = {h.strip().lower() for h in (env_value or "").split(",") if h.strip()}
    return LOCAL_HOSTNAMES | frozenset(extra)


def origin_allowed(origin: Optional[str], host: Optional[str],
                   allowed_hosts: frozenset = LOCAL_HOSTNAMES) -> bool:
    """Same-origin + host-allowlist check for the /ws handshake.

    Browsers always send Origin on a WebSocket handshake, so a page on another site that opens
    ws://localhost:8080/ws is refused. The Host must also name an allowed hostname: a DNS-rebinding
    page (evil.example resolving to 127.0.0.1) sends matching Origin and Host, which the hostname
    allowlist refuses. Non-browser clients (scripts, tests) send no Origin and are allowed.
    """
    if not origin:
        return True
    if not host:
        return False
    try:
        netloc = urlsplit(origin.strip()).netloc
        hostname = urlsplit("//" + host.strip()).hostname
    except ValueError:
        return False
    if not netloc or netloc.lower() != host.strip().lower():
        return False
    return bool(hostname) and hostname.lower() in allowed_hosts


# ---------------------------------------------------------------- static files

def resolve_static_path(web_root: str, request_path: str) -> Optional[str]:
    """Map an HTTP request path to a file under web_root, or None (404).

    Rejects anything resolving outside web_root (../, encoded ../, symlinks out),
    NUL bytes, backslashes, directories and unknown file types.
    """
    path = unquote(request_path.split("?", 1)[0].split("#", 1)[0])
    if "\x00" in path or "\\" in path:
        return None
    if path == "" or path.endswith("/"):
        path += "index.html"
    root = os.path.realpath(web_root)
    full = os.path.realpath(os.path.join(root, path.lstrip("/")))
    if os.path.commonpath([root, full]) != root or not os.path.isfile(full):
        return None
    if os.path.splitext(full)[1].lower() not in MIME_TYPES:
        return None
    return full


def mime_type(path: str) -> str:
    return MIME_TYPES.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
