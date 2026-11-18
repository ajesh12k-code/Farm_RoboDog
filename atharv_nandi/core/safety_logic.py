"""Pure (ROS-free, MuJoCo-free) safety logic for Atharv Nandi.

Used by atharv_nandi/nodes/safety_supervisor.py (latch, geofence, fall, liveness), atharv_nandi/nodes/sim_runner.py (stop
metrics) and atharv_nandi/nodes/locomotion_service.py (speed cap, stale-input watchdog). Every clock is injected
(seconds, monotonic or sim time), so everything here is unit-tested in tests/test_safety_logic.py.

Contract (see README, "Safety"):
    /nandi/estop        std_srvs/Trigger  engage the latch (reason 'operator')
    /nandi/estop_reset  std_srvs/Trigger  re-arm; refused while any cause (outside the geofence,
                                          tilted / fallen) is still present, whatever reason latched
    /nandi/safety_status  std_msgs/String JSON >= 5 Hz, built by build_safety_status()
    /nandi/heartbeat      std_msgs/String JSON {node, t, ok, info}, built by heartbeat_json()
"""
import copy
import json
import math
import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from atharv_nandi import paths

Point = Tuple[float, float]

# Reasons the latch can hold. Every reset checks every physical cause (geofence, fall), whatever
# reason latched first; 'operator' (the /nandi/estop service) and 'supervisor_lost' (sim_runner lost
# the supervisor's status and safe-held on its own) have no physical cause and clear on
# /nandi/estop_reset only. Nothing clears by itself.
REASON_OPERATOR = "operator"
REASON_GEOFENCE = "geofence"
REASON_FALL = "fall"
REASON_SUPERVISOR_LOST = "supervisor_lost"
# WP6 field latches (atharv_nandi/sim/hooks/stability.py publishes /nandi/safety_fields; the supervisor adopts its latch as an
# e-stop latch). Their cause is the field: a reset is refused until the hook reports protective_clear.
REASON_PROTECTIVE = "protective"
REASON_ANIMAL = "animal"
REASON_BUMPER = "bumper"
REASON_STABILITY = "stability"
FIELD_REASONS = (REASON_PROTECTIVE, REASON_ANIMAL, REASON_BUMPER, REASON_STABILITY)
FIELDS_MAX_AGE_S = 1.0          # a /nandi/safety_fields older than this is stale: its clear does not count
# WP6 handheld e-stop latch (ORCHARD-MVP-SPEC 1.4, WP1 contract): /nandi/handheld_estop/state (hook
# handheld_estop, atharv_nandi/sim/hooks/beacon.py) carries estop_request = the paired handheld reports its mushroom pressed.
# The cause is the mushroom: a reset is refused while it is still pressed, or while the state is stale (WP1
# publishes it at 2 Hz once the handheld has been heard). Local-link loss is WP28's LOCAL SAFETY LINK LOSS
# response, not a latch.
REASON_HANDHELD = "handheld"
HANDHELD_MAX_AGE_S = 1.0

HOLD_MODE_SAFE = "safe_hold"   # actuators hold the joint angles of the moment of engage (stiffened, braked)
HOLD_MODE_NONE = "none"

# Where the supervisor keeps the latch so a restarted supervisor comes back latched. /tmp lives as
# long as the container: a node restart keeps the latch, a fresh container (or launch_nandi.sh,
# which deletes it) starts un-latched.
DEFAULT_STATE_FILE = "/tmp/nandi_estop_latch.json"

STOP_SPEED_MPS = 0.02          # stop metrics: the robot counts as stopped below this trunk speed

DEFAULT_CONFIG = {
    "estop": {
        # Safe hold, applied to every actuator except the fingers (they keep their commanded grip):
        # kp x hold_stiffness_scale, kv x hold_damping_scale, plus a joint brake (MuJoCo dry
        # friction, N m; arm joints get half). WP6 (P-08, stationary <= 0.5 s): (2, 10, 4.0) stops the
        # trunk (< 0.02 m/s for 0.2 s) <= 0.42 s after engage over 162 flat runs (crown ext 0 / 0.6 /
        # 1.6 x cmd 0.3 / 0.62 / 1.0 x 12 gait phases) and <= 0.44 s on +-5 deg cross-slope (48 runs);
        # the former (2, 2, 2.0) took up to 0.64 s; higher kp is worse (kp x4 rings: no stop in 1.2 s).
        "hold_stiffness_scale": 2.0,
        "hold_damping_scale": 10.0,
        "hold_brake_nm": 4.0,
        # estop_reset blends every actuator from the hold pose to the live targets (smoothstep);
        # the blend lasts max_i |delta_i| / ctrlrange_i / rearm_rate seconds, clipped to
        # [rearm_min_s, rearm_max_s]: at most ~0.4 ctrlrange/s instead of a full-kp snap.
        "rearm_rate": 0.25,
        "rearm_min_s": 1.0,
        "rearm_max_s": 3.0,
        "state_file": DEFAULT_STATE_FILE,
    },
    "speed": {
        "max_speed_scale": 1.0,   # cap on gait_params[0] (the gait's own limit is 1.5)
        "max_cmd": 1.0,           # cap on |(vx, vy)| and |yaw| of /nandi/cmd_vel (contract range is [-1, 1])
    },
    "geofence": {
        "enabled": True,
        # Keep-in polygon, world/local metres (x forward from the home pose, y left). Default:
        # a rectangle around the farm plot (crops x 1.0..3.6, clod field x 3.6..6.5, |y| <= 1.2).
        "polygon": [[-1.5, -2.0], [7.0, -2.0], [7.0, 2.0], [-1.5, 2.0]],
        # estop_reset after a geofence latch needs the trunk at least this far inside: the trunk
        # sways back a few cm after the stop, which must not count as "inside again".
        "reset_margin_m": 0.1,
    },
    "fall": {
        "tilt_deg": 35.0,
        "sustain_s": 0.3,
    },
    "stale": {
        "max_age_s": 0.5,
        "topics": ["/nandi/sim_qpos", "/nandi/leg_targets"],
    },
    "liveness": {
        "max_age_s": 2.0,         # a node whose heartbeat is older than this is reported not ok
    },
}


# --- small YAML subset -------------------------------------------------------------------------
def _parse_scalar(text: str):
    text = text.strip()
    if text == "":
        return None
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "~"):
        return None
    try:
        return json.loads(text)  # numbers and flow lists like [[0, 1], [2, 3]]
    except ValueError:
        return text.strip("'\"")


def parse_simple_yaml(text: str) -> dict:
    """Parse the nested 'key: value' subset config/safety.yaml uses (comments, flow lists).

    PyYAML is used when available; this fallback keeps the ROS-free CI free of extra deps.
    """
    root: dict = {}
    stack: List[Tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"not a 'key: value' line: {raw!r}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _parse_scalar(value)
    return root


def _merge(base: dict, over: dict) -> dict:
    """Deep merge into a fresh copy: the result never aliases ``base`` (WPCOV review: a caller that edited the loaded
    config, e.g. its geofence polygon, edited DEFAULT_CONFIG for the whole process)."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else copy.deepcopy(v)
    return out


def default_config_path() -> str:
    return paths.config_path("safety.yaml")


def load_safety_config(path: Optional[str] = None) -> dict:
    """Defaults merged with config/safety.yaml (or $NANDI_SAFETY_CONFIG). A missing file = defaults."""
    path = path or os.environ.get("NANDI_SAFETY_CONFIG") or default_config_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return _merge(DEFAULT_CONFIG, {})
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
    except ImportError:
        data = parse_simple_yaml(text)
    return _merge(DEFAULT_CONFIG, data)


# --- e-stop latch ------------------------------------------------------------------------------
class EstopLatch:
    """Latched e-stop. The first reason sticks until a successful reset."""

    def __init__(self):
        self.estopped = False
        self.reason: Optional[str] = None
        self.engaged_t: Optional[float] = None

    def engage(self, reason: str, now: float) -> bool:
        """Latch with `reason`. Returns True on the rising edge (False if already latched)."""
        if self.estopped:
            return False
        self.estopped, self.reason, self.engaged_t = True, str(reason), float(now)
        return True

    def can_reset(self, cause_clear: bool) -> bool:
        return (not self.estopped) or bool(cause_clear)

    def reset(self, cause_clear: bool) -> Tuple[bool, str]:
        """Handler body for /nandi/estop_reset -> (success, message)."""
        if not self.estopped:
            return True, "Not e-stopped"
        if not cause_clear:
            return False, f"Refused: cause '{self.reason}' still present"
        reason = self.reason
        self.estopped, self.reason, self.engaged_t = False, None, None
        return True, f"E-stop reset (was '{reason}')"


# --- geofence ----------------------------------------------------------------------------------
def _seg_dist(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 <= 0.0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_edge(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> float:
    """Unsigned distance (m) from `point` to the nearest polygon edge."""
    p = (float(point[0]), float(point[1]))
    n = len(polygon)
    return min(_seg_dist(p, tuple(polygon[i]), tuple(polygon[(i + 1) % n])) for i in range(n))


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]], edge_eps: float = 1e-9) -> bool:
    """Even-odd ray cast. Points on an edge (within edge_eps) count as inside (keep-in fence)."""
    if len(polygon) < 3:
        return False
    if distance_to_edge(point, polygon) <= edge_eps:
        return True
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def geofence_margin(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> float:
    """Signed distance to the fence: + inside, - outside (m)."""
    d = distance_to_edge(point, polygon)
    return d if point_in_polygon(point, polygon) else -d


# --- speed cap ---------------------------------------------------------------------------------
class SpeedCap:
    """Clamps operator commands: planar |(vx, vy)| <= max_cmd, |yaw| <= max_cmd, speed_scale <= max."""

    def __init__(self, max_speed_scale: float = 1.0, max_cmd: float = 1.0):
        self.max_speed_scale = max(0.0, float(max_speed_scale))
        self.max_cmd = max(0.0, float(max_cmd))

    @classmethod
    def from_config(cls, cfg: dict) -> "SpeedCap":
        s = cfg.get("speed", {})
        return cls(s.get("max_speed_scale", 1.0), s.get("max_cmd", 1.0))

    def clamp_cmd(self, vx: float, vy: float, yaw: float) -> Tuple[float, float, float]:
        """Clamp an operator command. A non-finite component fails closed to 0 (WPCOV review: a NaN yaw came out
        as +max_cmd through min/max and an infinite vx as NaN after scaling)."""
        vx, vy, yaw = (float(c) if math.isfinite(c) else 0.0 for c in (vx, vy, yaw))
        mag = math.hypot(vx, vy)
        if mag > self.max_cmd and mag > 0.0:
            k = self.max_cmd / mag
            vx, vy = vx * k, vy * k
        yaw = max(-self.max_cmd, min(self.max_cmd, yaw))
        return vx, vy, yaw

    def clamp_speed_scale(self, speed_scale: float) -> float:
        s = float(speed_scale)
        return min(s, self.max_speed_scale) if math.isfinite(s) else 0.0      # NaN passed min() unchanged


# --- stale-input watchdog ----------------------------------------------------------------------
class StaleMonitor:
    """Tracks last-seen times per topic. A topic never seen is not stale until expect()ed."""

    def __init__(self, max_age_s: float = 0.5):
        self.max_age_s = float(max_age_s)
        self._last: Dict[str, float] = {}

    def expect(self, topic: str, now: float):
        """Start watching `topic` as if it had just arrived (grace period of max_age_s)."""
        self._last.setdefault(topic, float(now))

    def touch(self, topic: str, now: float):
        self._last[topic] = float(now)

    def age(self, topic: str, now: float) -> Optional[float]:
        t = self._last.get(topic)
        return None if t is None else max(0.0, float(now) - t)

    def is_stale(self, topic: str, now: float) -> bool:
        a = self.age(topic, now)
        return a is not None and a > self.max_age_s

    def stale(self, now: float, topics: Optional[Iterable[str]] = None) -> List[str]:
        names = sorted(self._last) if topics is None else list(topics)
        return [t for t in names if self.is_stale(t, now)]


# --- fall detection ----------------------------------------------------------------------------
class TiltFallDetector:
    """Fall = `fallen` flag (game_status.is_fallen) now, or tilt above tilt_deg for sustain_s."""

    def __init__(self, tilt_deg: float = 35.0, sustain_s: float = 0.3):
        self.tilt_deg = float(tilt_deg)
        self.sustain_s = float(sustain_s)
        self._over_since: Optional[float] = None

    @classmethod
    def from_config(cls, cfg: dict) -> "TiltFallDetector":
        f = cfg.get("fall", {})
        return cls(f.get("tilt_deg", 35.0), f.get("sustain_s", 0.3))

    def update(self, tilt_deg: float, fallen: bool, now: float) -> bool:
        if self._over(tilt_deg):
            if self._over_since is None:
                self._over_since = float(now)
        else:
            self._over_since = None
        sustained = self._over_since is not None and (now - self._over_since) >= self.sustain_s
        return bool(fallen) or sustained

    def cause_present(self, tilt_deg: float, fallen: bool) -> bool:
        """For estop_reset: the robot must be upright again (no sustain needed to refuse)."""
        return bool(fallen) or self._over(tilt_deg)

    def _over(self, tilt_deg: float) -> bool:
        """Tilt above the limit. A non-finite tilt (IMU fault) counts as over: fail closed (WPCOV review)."""
        t = float(tilt_deg)
        return not math.isfinite(t) or t > self.tilt_deg

    def reset(self):
        self._over_since = None


# --- stop metrics ------------------------------------------------------------------------------
class StopMetrics:
    """Time and trunk travel from e-stop engage until trunk speed < STOP_SPEED_MPS."""

    def __init__(self, stop_speed: float = STOP_SPEED_MPS):
        self.stop_speed = float(stop_speed)
        self._t0: Optional[float] = None
        self._prev: Optional[Point] = None
        self._dist = 0.0
        self.last: Optional[dict] = None   # {"t_s", "dist_m"} of the last completed stop

    @property
    def measuring(self) -> bool:
        return self._t0 is not None

    def start(self, now: float, xy: Sequence[float]):
        self._t0, self._prev, self._dist = float(now), (float(xy[0]), float(xy[1])), 0.0

    def cancel(self):
        self._t0 = self._prev = None

    def update(self, now: float, xy: Sequence[float], speed: float) -> Optional[dict]:
        """Feed a sample; returns the result dict once, when the robot has stopped."""
        if self._t0 is None:
            return None
        p = (float(xy[0]), float(xy[1]))
        self._dist += math.hypot(p[0] - self._prev[0], p[1] - self._prev[1])
        self._prev = p
        if speed < self.stop_speed:
            self.last = {"t_s": round(float(now) - self._t0, 3), "dist_m": round(self._dist, 4)}
            self._t0 = self._prev = None
            return self.last
        return None


# --- heartbeats / liveness ---------------------------------------------------------------------
def heartbeat_json(node: str, t: float, ok: bool = True, info: Optional[dict] = None) -> str:
    return json.dumps({"node": str(node), "t": round(float(t), 3), "ok": bool(ok), "info": info or {}})


def parse_heartbeat(text: str) -> Optional[dict]:
    try:
        d = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(d, dict) or not isinstance(d.get("node"), str) or not d["node"]:
        return None
    info = d.get("info")
    return {"node": d["node"], "ok": bool(d.get("ok", True)), "info": info if isinstance(info, dict) else {}}


class LivenessTracker:
    """Per-node heartbeat age (receiver clock) and ok flag."""

    def __init__(self, max_age_s: float = 2.0):
        self.max_age_s = float(max_age_s)
        self._seen: Dict[str, Tuple[float, bool]] = {}

    def beat(self, node: str, now: float, ok: bool = True):
        self._seen[node] = (float(now), bool(ok))

    def nodes(self, now: float) -> Dict[str, dict]:
        out = {}
        for name in sorted(self._seen):
            t, ok = self._seen[name]
            age = max(0.0, float(now) - t)
            out[name] = {"age_s": round(age, 2), "ok": bool(ok and age <= self.max_age_s)}
        return out


# --- /nandi/safety_status ----------------------------------------------------------------------
def build_safety_status(
    latch: EstopLatch,
    speed_cap: float,
    geofence_ok: bool,
    fallen: bool,
    stale: Sequence[str],
    last_stop: Optional[dict],
    nodes: Dict[str, dict],
    hold_mode: str = HOLD_MODE_NONE,
    geofence_margin_m: Optional[float] = None,
    tilt_deg: Optional[float] = None,
    engaged_at: Optional[float] = None,
    rearmed_at: Optional[float] = None,
) -> dict:
    return {
        "estopped": bool(latch.estopped),
        "latched_reason": latch.reason,
        "speed_cap": float(speed_cap),
        "geofence_ok": bool(geofence_ok),
        "fallen": bool(fallen),
        "stale": list(stale),
        "last_stop": dict(last_stop) if last_stop else None,
        "nodes": dict(nodes),
        "hold_mode": hold_mode,
        "geofence_margin_m": None if geofence_margin_m is None else round(float(geofence_margin_m), 3),
        "tilt_deg": None if tilt_deg is None else round(float(tilt_deg), 1),
        "engaged_at": None if engaged_at is None else float(engaged_at),
        "rearmed_at": None if rearmed_at is None else float(rearmed_at),
    }


def parse_safety_status(text: str) -> Optional[dict]:
    """Tolerant reader for subscribers (sim_runner, locomotion). None if unparseable."""
    try:
        d = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(d, dict):
        return None

    def _num(v):
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    return {"estopped": bool(d.get("estopped", False)), "latched_reason": d.get("latched_reason"),
            "engaged_at": _num(d.get("engaged_at")), "rearmed_at": _num(d.get("rearmed_at"))}


def rearm_duration(delta: Sequence[float], ctrl_range: Sequence[float], rate: float,
                   min_s: float, max_s: float) -> float:
    """Blend time from the hold pose to the live targets: the largest move in ctrlrange units / rate."""
    worst = 0.0
    for d, r in zip(delta, ctrl_range):
        if r > 0.0:
            worst = max(worst, abs(float(d)) / float(r))
    t = worst / rate if rate > 0.0 else max_s
    return max(float(min_s), min(float(max_s), t))


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def hold_should_release(estopped: bool, supervisor_lost: bool, rearmed_at: Optional[float],
                        hold_since: Optional[float]) -> bool:
    """sim_runner: leave safe hold only for a re-arm stamped at or after the hold started."""
    if estopped or supervisor_lost:
        return False
    return rearmed_at is not None and (hold_since is None or rearmed_at >= hold_since)


def save_latch_state(path: str, state: dict):
    """Atomic write of SafetyCore.state() (tmp file + rename)."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    os.replace(tmp, path)


def load_latch_state(path: str) -> Optional[dict]:
    """The saved state, or None if missing / unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


class SafetyCore:
    """Supervisor brain: latch + geofence + fall, fed with trunk pose samples. ROS-free.

    `wall_clock` stamps engaged_at / rearmed_at (seconds since the epoch, shared across processes):
    sim_runner leaves safe hold only for a re-arm stamped after its hold started, so a restarted
    supervisor that merely says estopped=false does not release it.
    """

    def __init__(self, cfg: Optional[dict] = None, wall_clock=time.time):
        self.cfg = cfg or load_safety_config()
        gf = self.cfg["geofence"]
        self.geofence_enabled = bool(gf.get("enabled", True))
        self.polygon = [(float(p[0]), float(p[1])) for p in gf["polygon"]]
        self.reset_margin = max(0.0, float(gf.get("reset_margin_m", 0.0)))
        self.latch = EstopLatch()
        self.fall = TiltFallDetector.from_config(self.cfg)
        self.speed_cap = SpeedCap.from_config(self.cfg)
        self.geofence_ok = True
        self.margin: Optional[float] = None
        self.tilt = 0.0
        self.fallen_flag = False
        self.fallen = False
        self.events: List[dict] = []   # latch events for logging: {"t", "reason", "detail"}
        self.wall_clock = wall_clock
        self.engaged_at: Optional[float] = None   # wall time of the current latch
        self.rearmed_at: Optional[float] = None   # wall time of the last successful estop_reset
        self.field_latched = False               # WP6 /nandi/safety_fields: hook latch, its reason, its clear
        self.field_reason = ""
        self.field_clear = True
        self.field_seen: Optional[float] = None
        self.handheld_pressed = False            # WP6 /nandi/handheld_estop/state: estop_request, its freshness
        self.handheld_seen: Optional[float] = None

    def _engage(self, reason: str, now: float, detail: str) -> bool:
        if self.latch.engage(reason, now):
            self.engaged_at = float(self.wall_clock())
            self.events.append({"t": round(float(now), 3), "reason": reason, "detail": detail})
            return True
        return False

    def engage_operator(self, now: float) -> bool:
        return self._engage(REASON_OPERATOR, now, "operator /nandi/estop")

    def engage_supervisor_lost(self, now: float, detail: str = "sim_runner safe-held without status") -> bool:
        return self._engage(REASON_SUPERVISOR_LOST, now, detail)

    # --- persistence --------------------------------------------------------------------------
    def state(self) -> dict:
        return {"estopped": self.latch.estopped, "reason": self.latch.reason,
                "engaged_at": self.engaged_at, "rearmed_at": self.rearmed_at}

    def restore(self, state: Optional[dict], now: float) -> bool:
        """Re-latch from a saved state (supervisor restart). Returns True if it restored a latch."""
        if not state:
            return False
        ra = state.get("rearmed_at")
        self.rearmed_at = float(ra) if isinstance(ra, (int, float)) else None
        if not state.get("estopped"):
            return False
        reason = str(state.get("reason") or REASON_OPERATOR)
        if reason in FIELD_REASONS:
            # a field latch: the field counts as occupied until a fresh /nandi/safety_fields payload says clear
            self.field_latched, self.field_reason, self.field_clear = True, reason, False
        if reason == REASON_HANDHELD:
            # the mushroom counts as pressed until a fresh /nandi/handheld_estop/state says released
            self.handheld_pressed = True
        if not self._engage(reason, now, "restored from the saved latch (supervisor restart)"):
            return False
        ea = state.get("engaged_at")
        if isinstance(ea, (int, float)):
            self.engaged_at = float(ea)
        return True

    def update_pose(self, now: float, xy: Sequence[float], tilt_deg: float, fallen_flag: bool) -> Optional[str]:
        """Feed one trunk sample. Returns the reason if this sample latched the e-stop."""
        self.tilt, self.fallen_flag = float(tilt_deg), bool(fallen_flag)
        if self.geofence_enabled:
            self.margin = geofence_margin(xy, self.polygon)
            self.geofence_ok = self.margin >= 0.0
        else:
            self.margin, self.geofence_ok = None, True
        self.fallen = self.fall.update(tilt_deg, fallen_flag, now)
        if self.fallen and self._engage(REASON_FALL, now, f"tilt {tilt_deg:.1f} deg, fallen={fallen_flag}"):
            return REASON_FALL
        if not self.geofence_ok and self._engage(
                REASON_GEOFENCE, now, f"trunk at ({xy[0]:.2f}, {xy[1]:.2f}) is {-self.margin:.2f} m outside"):
            return REASON_GEOFENCE
        return None

    def update_fields(self, now: float, fields: dict) -> Optional[str]:
        """Feed one /nandi/safety_fields payload (WP6 hook). A hook latch (protective, animal, bumper, stability)
        engages the e-stop latch with that reason. Returns the reason if this payload latched it."""
        self.field_seen = float(now)
        self.field_latched = bool(fields.get("latched"))
        self.field_reason = str(fields.get("latch_reason") or "")
        self.field_clear = bool(fields.get("protective_clear", not self.field_latched))
        if self.field_latched:
            reason = self.field_reason if self.field_reason in FIELD_REASONS else REASON_PROTECTIVE
            if self._engage(reason, now, f"safety field: {fields.get('detail') or reason}"):
                return reason
        return None

    def update_handheld(self, now: float, state: dict) -> Optional[str]:
        """Feed one /nandi/handheld_estop/state payload (WP1 hook). ``estop_request`` (the paired handheld reports
        its mushroom pressed) engages the e-stop latch with reason ``handheld``. Returns the reason if this payload
        latched it."""
        self.handheld_seen = float(now)
        self.handheld_pressed = bool(state.get("estop_request"))
        if self.handheld_pressed and self._engage(
                REASON_HANDHELD, now, f"handheld e-stop {state.get('device') or ''} seq {state.get('seq')}".strip()):
            return REASON_HANDHELD
        return None

    def causes_present(self, now: Optional[float] = None) -> List[str]:
        """Every physical cause that forbids a re-arm now, whatever reason latched first."""
        out = []
        if self.handheld_pressed:
            out.append(REASON_HANDHELD)
        elif self.latch.reason == REASON_HANDHELD:
            # released, but only a fresh state counts (no payload since a restart counts as still pressed)
            if self.handheld_seen is None or (now is not None and now - self.handheld_seen > HANDHELD_MAX_AGE_S):
                out.append(REASON_HANDHELD)
        if self.latch.reason in FIELD_REASONS or self.field_latched:
            # no payload at all counts as stale (fail safe: a restarted supervisor has not seen the field yet)
            stale = self.field_seen is None or (now is not None and now - self.field_seen > FIELDS_MAX_AGE_S)
            if not self.field_clear or stale:
                out.append(self.field_reason if self.field_reason in FIELD_REASONS else REASON_PROTECTIVE)
        if self.geofence_enabled and self.margin is not None:
            # After a geofence latch the trunk must be reset_margin back inside (it sways back a
            # few cm after the stop); otherwise anywhere inside the fence will do.
            need = self.reset_margin if self.latch.reason == REASON_GEOFENCE else 0.0
            if not self.margin >= need:          # NaN margin (bad pose) refuses the re-arm (WPCOV review)
                out.append(REASON_GEOFENCE)
        if self.fall.cause_present(self.tilt, self.fallen_flag):
            out.append(REASON_FALL)
        return out

    def cause_clear(self) -> bool:
        return not self.causes_present()

    def reset(self, now: Optional[float] = None) -> Tuple[bool, str]:
        causes = self.causes_present(now)
        if causes and self.latch.estopped:
            return False, f"Refused: cause {', '.join(repr(c) for c in causes)} still present (latched '{self.latch.reason}')"
        ok, msg = self.latch.reset(not causes)
        if ok:
            # Also stamped when not e-stopped: it releases a sim_runner 'supervisor_lost' hold.
            self.rearmed_at = float(self.wall_clock())
        return ok, msg


# --- WP6 e-stop actions --------------------------------------------------------------------------
class EstopActions:
    """What an e-stop does beyond the joint hold (MVP-SPEC 5.6, ORCHARD-MVP-SPEC 2.3 rule 6), coded against the WP0
    interfaces: every ``ToolActuation`` opens its blade and stops; the crown brakes in place (``MastControl.brake``;
    the sections hold, they retract only during e-stop recovery after both confirmations, WP28). ``engage`` runs
    once per latch (rising edge); ``release`` re-arms the edge detector. Timings are kept for the report."""

    def __init__(self, tools: Sequence = (), mast=None):
        self.tools = list(tools)
        self.mast = mast
        self.engaged_t: Optional[float] = None
        self.log: List[dict] = []

    def engage(self, now: float, reason: str = REASON_OPERATOR) -> bool:
        if self.engaged_t is not None:
            return False
        self.engaged_t = float(now)
        done = []
        for tool in self.tools:
            try:
                tool.open_blade()
                tool.stop()
                done.append("tool_safe")
            except Exception as exc:  # noqa: BLE001 - one bad tool must not stop the others
                done.append(f"tool_error:{type(exc).__name__}")
        if self.mast is not None:
            try:
                self.mast.brake()
                done.append("crown_brake")
            except Exception as exc:  # noqa: BLE001
                done.append(f"crown_error:{type(exc).__name__}")
        self.log.append({"t": round(float(now), 4), "reason": reason, "actions": done})
        return True

    def release(self):
        self.engaged_t = None
