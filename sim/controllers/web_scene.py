"""Pure (ROS-free) logic behind the Atharv Nandi browser bridge (sim/web_bridge.py).

Scene + pose serialization, client-message validation, cmd_vel mapping, the held-walk
dead-man, and safe static-file resolution. Nothing here imports rclpy, so CI can test it.

WebSocket wire format (JSON text frames on /ws):
  server -> client
    {"type": "scene", ...}       once per connection, see build_scene()
    {"type": "state", "poses": [x y z qw qx qy qz per streamed body] | null,
     "telemetry": {...} | null, "game": {...} | null, "harvest": {...} | null,
     "loco": {...} | null, "sim_age_s": float | null, "hold": bool}           ~30 Hz
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

from controllers.farm_scene import CROP_RADIUS, crop_body_names
from controllers.telemetry import parse_qpos_telemetry

# MuJoCo geom types the browser can draw, with MuJoCo's size conventions:
#   plane: [half_x, half_y, grid] (0 = infinite)   sphere: [r]   ellipsoid: [rx, ry, rz]
#   capsule / cylinder: [r, half_length] along local z   box: [half_x, half_y, half_z]
GEOM_TYPES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
}
VISIBLE_GROUPS = 3  # MuJoCo's viewer shows geom groups 0..2 by default
_DEFAULT_GEOM_RGBA = (0.5, 0.5, 0.5, 1.0)

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


def build_scene(model: mujoco.MjModel) -> Dict:
    """Static scene description: every drawable geom in its body's local frame.

    Bodies with drawable geoms (except the world body 0, which never moves) are listed in
    ``bodies``; pose_frame() streams their world poses in exactly that order.
    """
    geoms, moving = [], set()
    for g in range(model.ngeom):
        gtype = GEOM_TYPES.get(int(model.geom_type[g]))
        if gtype is None or int(model.geom_group[g]) >= VISIBLE_GROUPS:
            continue
        rgba = _geom_rgba(model, g)
        if float(rgba[3]) <= 0.0:
            continue
        bid = int(model.geom_bodyid[g])
        geoms.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or None,
            "body": bid,
            "type": gtype,
            "size": _r(model.geom_size[g], 5),
            "pos": _r(model.geom_pos[g], 5),
            "quat": _r(model.geom_quat[g], 6),  # w, x, y, z
            "rgba": _r(rgba, 3),
        })
        if bid != 0:
            moving.add(bid)

    bodies = [{"id": b, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or None}
              for b in sorted(moving)]
    return {
        "type": "scene",
        "nq": int(model.nq),
        "bodies": bodies,
        "geoms": geoms,
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
    ids = np.asarray(body_ids, dtype=int)
    frame = np.hstack([data.xpos[ids], data.xquat[ids]])
    return np.round(frame, POSE_DIGITS).ravel().tolist()


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
