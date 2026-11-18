"""Pure (ROS-free) pilot-console extensions on top of controllers.web_scene.

web_scene owns the base WebSocket protocol (scene/state/cmd/service/...). This module adds the
operator-console pieces without editing it:

  client -> server (new)
    {"type": "auth", "token": str}                      first message when auth is required
    {"type": "service", "name": "estop" | "estop_reset" | "mission_abort"}
    {"type": "mission_start", "zone": "zone_a"}         -> /nandi/mission/command
  server -> client (new)
    {"type": "hello", "auth_required": bool, "dev_mode": bool, "role": str|null}
    {"type": "auth", "ok": bool, "role": str|null, "message": str}
    {"type": "camera", "format": "jpeg"|"png", "data": base64, "t": float}   <= 2 Hz
    state frames gain "ext": {safety, mission, battery, telemetry, detections} (each dict|null)
    and "health": {node: "ok"|"warn"|"fail"|"n/a"}

Anything else is delegated unchanged to web_scene.parse_client_message.
"""
import base64
import glob
import hmac
import json
import math
import os
import re
import secrets
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs

from atharv_nandi.core import web_scene

BadMessage = web_scene.BadMessage

EXTRA_SERVICES = {
    "estop": "/nandi/estop",
    "estop_reset": "/nandi/estop_reset",
    "mission_abort": "/nandi/mission/abort",
}
SERVICE_TOPICS = {**web_scene.SERVICE_TOPICS, **EXTRA_SERVICES}
SERVICES = tuple(SERVICE_TOPICS)

MISSION_COMMAND_TOPIC = "/nandi/mission/command"
ZONE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
MAX_MISSION_BYTES = 256 * 1024

# Optional topics from the parallel safety / mission / perception / telemetry packages.
# All are std_msgs/String JSON except the camera (sensor_msgs/CompressedImage).
EXT_TOPICS = {
    "safety": "/nandi/safety_status",
    "mission": "/nandi/mission/status",
    "battery": "/nandi/battery_status",
    "telemetry": "/nandi/telemetry",
    "detections": "/nandi/crop_detections",
}
CAMERA_TOPIC = "/nandi/camera/head/compressed"
CAMERA_MAX_HZ = 2.0
CAMERA_MAX_BYTES = 512 * 1024
MAX_DETECTIONS = 64

# Topic freshness -> health level. sim_qpos (50 Hz) is the liveness signal for /healthz.
HEALTH_WARN_S = 2.0
HEALTH_FAIL_S = 5.0
QPOS_STALE_S = 2.0
NODE_LEVELS = ("ok", "warn", "fail", "n/a")


# ---------------------------------------------------------------- client messages

def parse_client_message(raw, arm_low: Sequence[float], arm_high: Sequence[float]) -> Dict:
    """Console message types first; everything else goes to web_scene. Raises BadMessage only."""
    if not isinstance(raw, str):
        raise BadMessage("binary frames are not supported")
    if len(raw) > web_scene.MAX_MSG_CHARS:
        raise BadMessage("message too large")
    try:
        msg = json.loads(raw)
    except (ValueError, RecursionError):
        raise BadMessage("invalid JSON") from None
    if not isinstance(msg, dict):
        raise BadMessage("message must be a JSON object")
    kind = msg.get("type")
    if kind == "auth":
        token = msg.get("token")
        if not isinstance(token, str) or not token:
            raise BadMessage("'token' must be a non-empty string")
        return {"type": "auth", "token": token}
    if kind == "mission_start":
        zone = msg.get("zone", "zone_a")
        if not isinstance(zone, str) or not ZONE_RE.match(zone):
            raise BadMessage("'zone' must match [a-z0-9][a-z0-9_-]{0,31}")
        return {"type": "mission_start", "zone": zone}
    if kind == "service" and msg.get("name") in EXTRA_SERVICES:
        return {"type": "service", "name": msg["name"]}
    if kind == "service" and msg.get("name") not in SERVICE_TOPICS:
        raise BadMessage(f"unknown service; expected one of {', '.join(SERVICES)}")
    return web_scene.parse_client_message(raw, arm_low, arm_high)


def extend_scene(scene: Dict) -> Dict:
    """Base scene + the console's service list (the browser hides buttons it cannot use)."""
    out = dict(scene)
    out["services"] = list(SERVICES)
    out["console"] = {"mission_zones": [], "ext_topics": dict(EXT_TOPICS), "camera_topic": CAMERA_TOPIC}
    return out


# ---------------------------------------------------------------- missions

def mission_files(missions_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(missions_dir, "*.geojson")))


def _zone_of(path: str, obj) -> Iterable[str]:
    yield os.path.splitext(os.path.basename(path))[0]
    if isinstance(obj, dict):
        props = obj.get("properties")
        if isinstance(props, dict):
            for key in ("zone", "id", "name"):
                if isinstance(props.get(key), str):
                    yield props[key]
        for feat in obj.get("features", []) if isinstance(obj.get("features"), list) else []:
            p = feat.get("properties") if isinstance(feat, dict) else None
            if isinstance(p, dict) and isinstance(p.get("zone"), str):
                yield p["zone"]


def _load_geojson(path: str):
    try:
        if os.path.getsize(path) > MAX_MISSION_BYTES:
            return None
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, ValueError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None


def list_zones(missions_dir: str) -> List[str]:
    zones = []
    for path in mission_files(missions_dir):
        obj = _load_geojson(path)
        if obj is None:
            continue
        for z in _zone_of(path, obj):
            if ZONE_RE.match(z) and z not in zones:
                zones.append(z)
    return zones


def mission_command(zone: str, missions_dir: str, repo_root: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Stock start command for /nandi/mission/command -> (json_text | None, message).

    Looks for <missions_dir>/<zone>.geojson, then any *.geojson whose properties (or a feature's
    properties) name the zone. The command embeds the GeoJSON so the executor needs no file access:
      {"action": "start", "zone": str, "source": "config/missions/<file>", "mission": {GeoJSON}}
    """
    if not isinstance(zone, str) or not ZONE_RE.match(zone):
        return None, "invalid zone name"
    paths = mission_files(missions_dir)
    if not paths:
        return None, f"no mission files in {_rel(missions_dir, repo_root)} (mission package not installed yet)"
    exact = os.path.join(missions_dir, f"{zone}.geojson")
    ordered = ([exact] if exact in paths else []) + [p for p in paths if p != exact]
    for path in ordered:
        obj = _load_geojson(path)
        if obj is None:
            continue
        if zone in _zone_of(path, obj):
            cmd = {"action": "start", "zone": zone, "source": _rel(path, repo_root), "mission": obj}
            return json.dumps(cmd, separators=(",", ":")), f"Mission start sent: {zone} ({_rel(path, repo_root)})"
    return None, f"no mission for zone '{zone}' in {_rel(missions_dir, repo_root)}"


def _rel(path: str, root: Optional[str]) -> str:
    if root:
        try:
            return os.path.relpath(path, root)
        except ValueError:
            pass
    return path


# ---------------------------------------------------------------- optional topics

def parse_ext(key: str, text: str) -> Optional[Dict]:
    """JSON text from one of EXT_TOPICS -> dict for the state frame (None if unusable)."""
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError, TypeError):
        return None
    if key == "detections":
        items = obj.get("detections") if isinstance(obj, dict) else obj
        if not isinstance(items, list):
            return None
        out = [d for d in (_detection(x) for x in items[:MAX_DETECTIONS]) if d is not None]
        return {"items": out, "frame": obj.get("frame", "world") if isinstance(obj, dict) else "world"}
    return obj if isinstance(obj, dict) else None


def _num(v) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def _detection(d) -> Optional[Dict]:
    """{x,y,z} | {position|pos|xyz: [x,y,z]} + optional label/confidence -> normalised dict."""
    if not isinstance(d, dict):
        return None
    p = d.get("position", d.get("pos", d.get("xyz")))
    if isinstance(p, dict):
        p = [p.get("x"), p.get("y"), p.get("z", 0.0)]
    if not isinstance(p, list):
        p = [d.get("x"), d.get("y"), d.get("z", 0.0)]
    if len(p) < 2:
        return None
    xyz = [_num(v) for v in (list(p) + [0.0])[:3]]
    if any(v is None for v in xyz):
        return None
    conf = _num(d.get("confidence", d.get("score")))
    label = d.get("label", d.get("class", d.get("name")))
    return {"xyz": [round(v, 3) for v in xyz], "label": str(label)[:32] if label is not None else None,
            "confidence": round(conf, 3) if conf is not None else None}


def camera_frame(fmt: str, data: bytes, t: float) -> Optional[Dict]:
    """sensor_msgs/CompressedImage -> camera event (None if unsupported / too large)."""
    f = (fmt or "").lower()
    kind = "png" if "png" in f else "jpeg" if ("jpeg" in f or "jpg" in f or f == "") else None
    if kind is None or not data or len(data) > CAMERA_MAX_BYTES:
        return None
    return {"type": "camera", "format": kind, "data": base64.b64encode(bytes(data)).decode("ascii"),
            "t": round(float(t), 3)}


class RateLimiter:
    """At most `hz` accepts per second; pure, caller passes the clock."""

    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self.last = -math.inf

    def accept(self, now: float) -> bool:
        if now - self.last < self.period:
            return False
        self.last = now
        return True


# ---------------------------------------------------------------- health

def age_level(age_s: Optional[float]) -> str:
    if age_s is None:
        return "n/a"
    if age_s < HEALTH_WARN_S:
        return "ok"
    if age_s < HEALTH_FAIL_S:
        return "warn"
    return "fail"


def _node_level(v) -> str:
    """safety_status.nodes values: "ok"/"warn"/"fail"/..., bool, or {"ok"|"state"|"level"|"age_s"}."""
    if isinstance(v, bool):
        return "ok" if v else "fail"
    if isinstance(v, str):
        s = v.lower()
        if s in ("ok", "alive", "up", "green", "healthy"):
            return "ok"
        if s in ("warn", "warning", "stale", "amber", "degraded", "slow"):
            return "warn"
        if s in ("fail", "failed", "dead", "down", "red", "error", "lost", "timeout"):
            return "fail"
        return "n/a"
    if isinstance(v, dict):
        for key in ("level", "state", "status"):
            if key in v:
                return _node_level(v[key])
        if "ok" in v:
            return _node_level(bool(v["ok"]))
        if "age_s" in v:
            return age_level(_num(v["age_s"]))
    return "n/a"


def node_health(topic_ages: Mapping[str, Optional[float]], safety: Optional[Dict] = None) -> Dict[str, str]:
    """Topic freshness, overridden by the safety supervisor's own per-node verdicts when present.

    The verdicts only count while /nandi/safety_status itself is fresh (age_level 'ok'): a dead
    supervisor's last map must not keep every chip green. The 'safety' chip is always the
    supervisor topic's own freshness (it cannot vouch for itself).
    """
    out = {name: age_level(age) for name, age in topic_ages.items()}
    nodes = safety.get("nodes") if isinstance(safety, dict) else None
    if isinstance(nodes, dict) and age_level(topic_ages.get("safety")) == "ok":
        for name, v in list(nodes.items())[:32]:
            key = str(name)[:40]
            if key != "safety":
                out[key] = _node_level(v)
    return out


# Optional-topic payloads older than this are dropped from state frames (shown as absent),
# so a dead publisher's last safety / mission / battery message does not freeze the UI.
EXT_STALE_S = HEALTH_FAIL_S


def fresh_ext(ext: Mapping[str, Optional[Dict]], topic_ages: Mapping[str, Optional[float]]) -> Dict:
    """Copy of `ext` with every entry whose topic is older than EXT_STALE_S (or never seen) set to None."""
    out = {}
    for key, value in ext.items():
        age = topic_ages.get(key)
        out[key] = value if age is not None and age < EXT_STALE_S else None
    return out


def health_report(qpos_age_s: Optional[float], topic_ages: Mapping[str, Optional[float]],
                  safety: Optional[Dict] = None) -> Tuple[int, Dict]:
    """GET /healthz -> (HTTP status, JSON body). 503 when sim_qpos is missing or older than 2 s."""
    ok = qpos_age_s is not None and qpos_age_s <= QPOS_STALE_S
    body = {"ok": ok, "nodes": node_health(topic_ages, safety),
            "last_qpos_age_s": round(qpos_age_s, 3) if qpos_age_s is not None else None}
    return (200 if ok else 503), body


# ---------------------------------------------------------------- E-STOP latch

# While the console's E-STOP latch is set, only these may move the robot again: nothing.
# Stop/release/reset/status requests stay allowed so the operator can recover.
ESTOP_BLOCKED_SERVICES = frozenset({"auto_harvest"})


def blocked_by_estop(msg: Dict) -> bool:
    """True if `msg` would make the robot move while the E-STOP latch is set."""
    kind = msg.get("type")
    if kind == "cmd":
        return not web_scene.is_zero_cmd(msg.get("vx", 0.0), msg.get("vy", 0.0), msg.get("yaw", 0.0))
    if kind in ("manual_arm", "mission_start"):
        return True
    return kind == "service" and msg.get("name") in ESTOP_BLOCKED_SERVICES


# While the console relay walks the robot (TREE 360 / recovery / work drive) these are refused
# (MP 0.3 / MISSION-PRESETS: arm PARKed while walking; no second motion source).
# Stop / release / E-STOP / console commands stay allowed.
RELAY_BLOCKED_SERVICES = frozenset({"auto_harvest"})


def blocked_by_relay(msg: Dict) -> bool:
    """True if `msg` would move the arm, change the gait or start a second motion source while the relay drives."""
    kind = msg.get("type")
    if kind == "cmd":
        return not web_scene.is_zero_cmd(msg.get("vx", 0.0), msg.get("vy", 0.0), msg.get("yaw", 0.0))
    if kind in ("manual_arm", "gait_params", "mission_start"):
        return True
    return kind == "service" and msg.get("name") in RELAY_BLOCKED_SERVICES


# Pilot services that stop the gait (/nandi/stop_motion first): the console relay must stay stopped after them.
STOPPING_SERVICES = frozenset({"stop", "auto_harvest", "reset_world", "estop", "mission_abort"})
STOP_PAUSE_REASONS = (STOPPING_SERVICES - {"estop"}) | {"pause"}   # queue pauses an operator Resume / start ends


def sim_local_link_on(raw: Optional[str], field_mode: bool) -> bool:
    """NANDI_SIM_LOCAL_LINK: default on outside field mode; never on in field mode, whatever the variable says."""
    if field_mode:
        return False
    v = (raw or "").strip().lower()
    return True if not v else v in ("1", "true", "yes", "on")


class EstopLatch:
    """The console's E-STOP latch, persisted to `path` so a bridge restart cannot clear it.

    Each press bumps `seq`. A reset clears the latch only if no press happened after the reset
    started (clear(seq_at_start)), so an E-STOP racing an in-flight reset always wins.
    A missing state file means unlatched; an unreadable or corrupt one latches (fail safe).
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.seq = 0
        self.errors = 0
        self.state = self._unlatched()
        self._load()

    @staticmethod
    def _unlatched() -> Dict:
        return {"latched": False, "t": None, "by": None, "message": ""}

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                raise ValueError("not an object")
        except (OSError, ValueError, RecursionError):
            self.state = {"latched": True, "t": None, "by": "bridge",
                          "message": f"E-STOP state file {self.path} unreadable: latched for safety"}
            return
        if obj.get("latched"):
            t = obj.get("t")
            self.state = {"latched": True, "t": t if isinstance(t, (int, float)) else None,
                          "by": str(obj.get("by") or "")[:80] or None,
                          "message": "E-STOP latched before the bridge restarted: "
                                     + str(obj.get("message") or "")[:300]}

    def _save(self):
        if not self.path:
            return
        tmp = f"{self.path}.tmp"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.state, f, separators=(",", ":"))
            os.replace(tmp, self.path)
        except OSError:
            self.errors += 1

    @property
    def latched(self) -> bool:
        return bool(self.state["latched"])

    def as_dict(self) -> Dict:
        return dict(self.state)

    def latch(self, by: str, t: float, message: str = "E-STOP pressed") -> int:
        self.seq += 1
        self.state = {"latched": True, "t": t, "by": by, "message": message}
        self._save()
        return self.seq

    def set_message(self, seq: int, message: str) -> bool:
        """Attach the outcome text to press `seq`, unless a newer press or a reset replaced it."""
        if seq != self.seq or not self.latched:
            return False
        self.state["message"] = message
        self._save()
        return True

    def clear(self, seq_at_start: int) -> bool:
        """Clear the latch unless an E-STOP was pressed after the reset began (seq moved on)."""
        if seq_at_start != self.seq:
            return False
        self.state = self._unlatched()
        self._save()
        return True


def audit_values(msg: Dict):
    """The parsed command's payload for the audit row (None when the type/name says it all)."""
    kind = msg.get("type")
    if kind == "cmd":
        return [msg.get("vx"), msg.get("vy"), msg.get("yaw"), bool(msg.get("hold"))]
    if kind in ("manual_arm", "gait_params"):
        return list(msg.get("values") or [])
    if kind == "mission_start":
        return msg.get("zone")
    if kind == "envelope":
        return bool(msg.get("show"))
    return None


# ================================================================ WP10 orchard console (MP 0.7, OS 5.5)
#
# client -> server (operator role only, audit-logged; the viewer role is refused by web_auth.permit)
#   {"type": "preset_start", "preset": id, "targets": {tree_ids | row_ids | block_id | block_sample}, "params": {},
#    "confirm": bool}                                       Ready check -> cmd/mission -> PresetQueue
#   {"type": "preset_pause" | "preset_resume"}              the running item (queue resume after a Halt)
#   {"type": "preset_abort", "qid"?: str}                   the running item, or a queued one
#   {"type": "tree_mark", "tree_id", "kind", "clock"?, "h_class" | "h_m", "side"?, "tool_hint"?, "note"?}
#   {"type": "assist_reply", "reply": "take_over" | "skip" | "halt"}
#   {"type": "queue_reorder", "order": [qid, ...]}          a permutation of the queued ids
#   {"type": "jobsheet_request", "row_id", "lang": "hi" | "en"}
#   extensions (WP28 / WP24 / WP7 change requests to WP10):
#   {"type": "recovery", "op": run_safe_park | run_return_home | abort | confirm_operator | ack_resume | ack_halt |
#    arm | check, "item"?: <FALLEN checklist item>}       -> /nandi/recovery/cmd (never /nandi/recovery/local)
#   {"type": "crown", "band": stow | mid | high}            -> /nandi/mast/cmd (source preset, reason operator)
#   {"type": "link_test", "link": "local" | "cloud", "dur_s": 1-30}   sim only: WP7 link-loss injection
# server -> client (each stream <= 2 Hz and only on change; events once)
#   preset_status   {tiles[], queue{items[], eta_s, soc_end, disk_free_end_mb, n}, active, history[], paused, chips[],
#                    ready{rtk, soc, local_link, cloud_link, ...}, assist}
#   map             {world, frame, trees[], rows[], blocks[], laterals[], ropes[], geofence, home, dock, annuli{}}
#   coverage        TREE 360 progress {tree_id, phase, station, stations, coverage_pct{low, mid, high}, ...}
#   mast            crown gauge {ext_m, target_ext, band, gated, reason, detail, arbiter_source, wind{}, gates{}}
#   recovery        {armed, active, mode, step, prompt, outcome, checklist, confirms, links{local, cloud}}
#   tree_record     one TreeRecord summary per tree visit (event)       mark   one MarkRecord (event)
#   jobsheet        the answer to jobsheet_request {row_id, lang, sheet, html, voice[]} (event, to the requester)

from collections import deque  # noqa: E402

from atharv_nandi.core import job_sheet as js  # noqa: E402
from atharv_nandi.core import mission_logic as ml  # noqa: E402
from atharv_nandi.core import presets as ps  # noqa: E402
from atharv_nandi.core import ready_check as rc  # noqa: E402
from atharv_nandi.core import records as rec  # noqa: E402
from atharv_nandi.core import web_auth  # noqa: E402

STREAM_MAX_HZ = 2.0                   # MP 0.7: <= 2 Hz per stream
BUDGET_BYTES_PER_S = 1024 * 1024      # MP 0.7: <= 1 MiB/s total per client
CONSOLE_STREAMS = ("preset_status", "map", "coverage", "mast", "recovery")
CONSOLE_EVENTS = ("tree_record", "mark", "jobsheet")
CONSOLE_TYPES = ("preset_start", "preset_pause", "preset_resume", "preset_abort", "tree_mark", "assist_reply",
                 "queue_reorder", "jobsheet_request", "recovery", "crown", "link_test")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,47}$")
ASSIST_REPLIES = ("take_over", "skip", "halt")
RECOVERY_OPS = {
    "run_safe_park": {"run": "safe_park"}, "run_return_home": {"run": "return_home"}, "abort": {"abort": True},
    "confirm_operator": {"confirm": "operator"}, "ack_resume": {"ack": "resume"}, "ack_halt": {"ack": "halt"},
    "arm": {"arm": True},
}
# WP28 outcomes that end a recovery normally (resumed after the hold, home, recovered, ready to stand, checklist done)
RECOVERY_DONE = (None, "", "done", "resumed", "home", "recovered", "ready", "manual")
FALLEN_CHECKS = ("crown_lock", "lift_points", "two_people", "laterals_tools", "mast_inspected", "person_clear")
CROWN_BANDS = ("stow", "mid", "high")
MAX_PARAMS = 24
MAX_TARGETS = 32

# Inputs the console reads (std_msgs/String JSON). Person positions never leave this module: world/state is reduced
# to the light level and the lateral counter (DP-13: the console overlay only shows the robot's own fields). The 3D view
# draws the simulator's own bodies (mocap actors = simulated people) from /nandi/sim_state: sim ground truth, not
# console data (the WP10 contract notes).
HUB_TOPICS_IN = {
    "mast": "/nandi/mast/state", "tree360": "/nandi/tree360/progress", "recovery": "/nandi/recovery/state",
    "recovery_alert": "/nandi/recovery/alert", "link_local": "/nandi/link/local", "link_cloud": "/nandi/link/cloud",
    "gnss": "/nandi/gnss/fix", "battery": "/nandi/platform/battery", "wind": "/nandi/wind",
    "weather": "/nandi/weather", "stability": "/nandi/stability/crown", "tools": "/nandi/tool/state",
    "transfer": "/nandi/transfer/state", "world": "/nandi/world/state", "work": "/nandi/work/state",
    "work_events": "/nandi/work/events", "dock": "/nandi/dock/state", "demo": "/nandi/demo/state",
    "demo_events": "/nandi/demo/events", "tree_record": "/nandi/tree/record", "mark": "/nandi/mark",
    "jobsheet": "/nandi/jobsheet", "assist": "/nandi/assist",
    "drive_tree360": "/nandi/tree360/drive", "drive_recovery": "/nandi/recovery/motion", "drive_work": "/nandi/work/drive",
}
HUB_TOPICS_OUT = {
    "tree360_cmd": "/nandi/tree360/cmd", "recovery_cmd": "/nandi/recovery/cmd", "mast_cmd": "/nandi/mast/cmd",
    "work_cmd": "/nandi/work/cmd", "dock_cmd": "/nandi/dock/cmd", "demo_run": "/nandi/demo/run",
    "preset_run": "/nandi/preset/run", "preset_state": "/nandi/preset/state", "mark": "/nandi/mark",
    "link_cmd": "/nandi/link/cmd", "cloud_beat": "/nandi/link/cloud/heartbeat",
    "local_beat": "/nandi/link/local/heartbeat",
}
# How a preset runs in the live stack. tree360 / recovery / work have hooks inside sim_runner (WP25 / WP28 / WP29);
# dock_unload needs the WP13 hook (PROPOSED); the contact and scan presets run only in the WP9 demo-tour world
# (atharv_nandi/nodes/demo_tour_node.py, its own SimDriver) when that node is up and the tour has a matching step, else the
# command is only published on /nandi/preset/run for an executor (none in this stack: WP8 / WP9 / WP38 CR).
EXECUTOR_OF_MODE = {
    "tree360": "tree360", "safe_park": "recovery", "return_home": "recovery", "estop_recover": "recovery_confirm",
    "tote_carry": "work", "lead_to_marks": "work", "follow_me": "work", "dock_unload": "dock",
}
WORK_MODE = {"tote_carry": "tote_carry", "lead_to_marks": "lead_to_marks", "follow_me": "follow_me"}
FRESH_S = 3.0                 # a status topic older than this is treated as absent
DRIVE_STALE_S = 0.5           # the cmd_vel dead-man: a drive older than this releases the walk
TREE360_START_TIMEOUT_S = 6.0
EXEC_START_TIMEOUT_S = 6.0
CLOUD_BEAT_HZ = 1.0
LOCAL_BEAT_HZ = 10.0          # local link timeout is 0.5 s (WP7)
CLIENT_ALIVE_S = 5.0          # a browser that has not pinged for this long no longer keeps the cloud link up
TILES_PERIOD_S = 1.0
TERMINAL_T360 = ("done", "aborted")


class StreamLimiter:
    """At most ``hz`` sends per stream, and only when the payload text changed (or every ``keepalive_s``)."""

    def __init__(self, hz: float = STREAM_MAX_HZ, keepalive_s: float = 5.0):
        self.period = 1.1 / float(hz)   # 10 % margin: tick jitter never yields 3 messages in 1 s
        self.keepalive = keepalive_s
        self.last_t: Dict[str, float] = {}
        self.last_text: Dict[str, str] = {}

    def offer(self, stream: str, text: str, now: float) -> bool:
        last = self.last_t.get(stream, -math.inf)
        if now - last < self.period - 1e-9:
            return False
        if text == self.last_text.get(stream) and now - last < self.keepalive:
            return False
        self.last_t[stream], self.last_text[stream] = now, text
        return True


class ByteBudget:
    """Sliding 1 s window per client: a droppable frame (state, camera) is sent only if the bytes sent in the last
    second plus it stay <= ``rate``. Events (``must``: scene, E-STOP results, records) always go and count, so
    droppable frames give way after a big event."""

    def __init__(self, rate: float = BUDGET_BYTES_PER_S, window_s: float = 1.0):
        self.rate = float(rate)
        self.window = float(window_s)
        self.log: deque = deque()
        self.in_window = 0
        self.sent = 0
        self.dropped = 0

    def _trim(self, now: float):
        while self.log and now - self.log[0][0] >= self.window:
            self.in_window -= self.log.popleft()[1]

    def allow(self, nbytes: int, now: float, must: bool = False) -> bool:
        self._trim(now)
        if must or self.in_window + nbytes <= self.rate:
            self.log.append((now, nbytes))
            self.in_window += nbytes
            self.sent += nbytes
            return True
        self.dropped += 1
        return False


class ConsoleAuditLog(web_auth.AuditLog):
    """web_auth.AuditLog that also records every console (MP 0.7) command."""

    def should_log(self, client, msg_type, name, t, values=None) -> bool:
        if msg_type in CONSOLE_TYPES:
            last = self._last.get(client)
            return not (last is not None and last[0] == msg_type and last[1] == name
                        and last[2] == self._key(values) and t - last[3] < web_auth.DEDUPE_WINDOW_S)
        return super().should_log(client, msg_type, name, t, values)


def console_audit_values(msg: Dict):
    kind = msg.get("type")
    if kind == "preset_start":
        return [msg.get("preset"), json.dumps(msg.get("targets"), sort_keys=True)[:120], bool(msg.get("confirm"))]
    if kind == "preset_abort":
        return msg.get("qid")
    if kind == "tree_mark":
        return [msg.get("tree_id"), msg.get("kind"), msg.get("clock"), msg.get("h_class")]
    if kind == "assist_reply":
        return msg.get("reply")
    if kind == "queue_reorder":
        return list(msg.get("order") or [])
    if kind == "jobsheet_request":
        return [msg.get("row_id"), msg.get("lang")]
    if kind == "recovery":
        return [msg.get("op"), msg.get("item")]
    if kind == "crown":
        return msg.get("band")
    if kind == "link_test":
        return [msg.get("link"), msg.get("dur_s")]
    return audit_values(msg)


# ---------------------------------------------------------------- client schemas

def _id(v, what: str) -> str:
    if not isinstance(v, str) or not ID_RE.match(v):
        raise BadMessage(f"'{what}' must be an id ([A-Za-z0-9][A-Za-z0-9_.:-]{{0,47}})")
    return v


def _id_list(v, what: str, max_n: int = MAX_TARGETS) -> List[str]:
    if not isinstance(v, list) or not 1 <= len(v) <= max_n:
        raise BadMessage(f"'{what}' must be a list of 1-{max_n} ids")
    return [_id(x, what) for x in v]


def _scalar(v, depth: int = 0):
    if isinstance(v, bool) or v is None or isinstance(v, str) and len(v) <= 64:
        return v
    if isinstance(v, (int, float)) and math.isfinite(v):
        return v
    if isinstance(v, list) and depth == 0 and len(v) <= 8:
        return [_scalar(x, 1) for x in v]
    if isinstance(v, dict) and depth == 0 and len(v) <= 8:
        return {str(k)[:32]: _scalar(x, 1) for k, x in v.items()}
    raise BadMessage("param values must be numbers, strings (<= 64 chars), booleans or short lists")


def _targets(t) -> Dict:
    if t is None:
        return {}
    if not isinstance(t, dict):
        raise BadMessage("'targets' must be an object")
    keys = set(t) & {"tree_ids", "row_ids", "block_id", "block_sample"}
    if set(t) - keys:
        raise BadMessage("'targets' takes one of tree_ids, row_ids, block_id, block_sample")
    if len(keys) > 1:
        raise BadMessage("'targets' takes exactly one of tree_ids, row_ids, block_id, block_sample")
    out = {}
    if "tree_ids" in t:
        out["tree_ids"] = _id_list(t["tree_ids"], "tree_ids")
    if "row_ids" in t:
        out["row_ids"] = _id_list(t["row_ids"], "row_ids", 8)
    if "block_id" in t:
        out["block_id"] = _id(t["block_id"], "block_id")
    if "block_sample" in t:
        bs = t["block_sample"]
        if not isinstance(bs, dict):
            raise BadMessage("'block_sample' must be an object")
        out["block_sample"] = {str(k)[:32]: _scalar(v, 1) for k, v in list(bs.items())[:4]}
    return out


def parse_console_message(msg: Dict) -> Optional[Dict]:
    """MP 0.7 console messages (already JSON-decoded) -> normalised dict; None if ``msg`` is not a console type.
    Raises BadMessage with a reason the browser may show."""
    kind = msg.get("type")
    if kind not in CONSOLE_TYPES:
        return None
    if kind == "preset_start":
        params = msg.get("params", {})
        if not isinstance(params, dict) or len(params) > MAX_PARAMS:
            raise BadMessage(f"'params' must be an object with <= {MAX_PARAMS} keys")
        confirm = msg.get("confirm", False)
        if not isinstance(confirm, bool):
            raise BadMessage("'confirm' must be true or false")
        return {"type": kind, "preset": _id(msg.get("preset"), "preset"), "targets": _targets(msg.get("targets")),
                "params": {_id(k, "param"): _scalar(v) for k, v in params.items()}, "confirm": confirm}
    if kind in ("preset_pause", "preset_resume"):
        return {"type": kind}
    if kind == "preset_abort":
        qid = msg.get("qid")
        return {"type": kind, "qid": None if qid is None else _id(qid, "qid")}
    if kind == "tree_mark":
        mk = msg.get("kind")
        if mk not in rec.MARK_KINDS:
            raise BadMessage(f"'kind' must be one of {', '.join(rec.MARK_KINDS)}")
        clock = msg.get("clock", 6)
        if isinstance(clock, bool) or not isinstance(clock, int) or not 1 <= clock <= 12:
            raise BadMessage("'clock' must be an integer 1-12")
        h_m = msg.get("h_m")
        h_class = msg.get("h_class")
        if h_m is not None:
            h_m = _number(h_m, "'h_m'")
            if not 0.0 <= h_m <= 4.0:
                raise BadMessage("'h_m' must be within 0-4 m")
        elif h_class not in rec.H_CLASSES:
            raise BadMessage(f"give 'h_m' or 'h_class' ({', '.join(rec.H_CLASSES)})")
        side = msg.get("side")
        if side is not None and side not in rec.SIDES:
            raise BadMessage("'side' must be N, E, S or W")
        tool = msg.get("tool_hint")
        if tool is not None and tool not in rec.TOOL_HINTS:
            raise BadMessage(f"'tool_hint' must be one of {', '.join(rec.TOOL_HINTS)}")
        note = msg.get("note", "")
        if not isinstance(note, str) or len(note) > 120:
            raise BadMessage("'note' must be a string of <= 120 chars")
        return {"type": kind, "tree_id": _id(msg.get("tree_id"), "tree_id"), "kind": mk, "clock": clock,
                "h_m": h_m, "h_class": h_class, "side": side, "tool_hint": tool, "note": note}
    if kind == "assist_reply":
        reply = msg.get("reply")
        if reply not in ASSIST_REPLIES:
            raise BadMessage("'reply' must be take_over, skip or halt")
        return {"type": kind, "reply": reply}
    if kind == "queue_reorder":
        order = msg.get("order")
        if not isinstance(order, list) or len(order) > ml.QUEUE_MAX:
            raise BadMessage(f"'order' must be a list of <= {ml.QUEUE_MAX} queue ids")
        return {"type": kind, "order": [_id(q, "order") for q in order]}
    if kind == "jobsheet_request":
        lang = msg.get("lang", "hi")
        if lang not in ("hi", "en"):
            raise BadMessage("'lang' must be hi or en")
        return {"type": kind, "row_id": _id(msg.get("row_id"), "row_id"), "lang": lang}
    if kind == "recovery":
        op = msg.get("op")
        if op == "check":
            item = msg.get("item")
            if item not in FALLEN_CHECKS:
                raise BadMessage(f"'item' must be one of {', '.join(FALLEN_CHECKS)}")
            return {"type": kind, "op": op, "item": item}
        if op not in RECOVERY_OPS:
            raise BadMessage(f"'op' must be one of {', '.join(list(RECOVERY_OPS) + ['check'])}")
        return {"type": kind, "op": op, "item": None}
    if kind == "crown":
        band = msg.get("band")
        if band not in CROWN_BANDS:
            raise BadMessage("'band' must be stow, mid or high")
        return {"type": kind, "band": band}
    if kind == "link_test":
        link = msg.get("link", "local")
        if link not in ("local", "cloud"):
            raise BadMessage("'link' must be local or cloud")
        dur = _number(msg.get("dur_s", 6.0), "'dur_s'")
        if not 1.0 <= dur <= 30.0:
            raise BadMessage("'dur_s' must be within 1-30 s")
        return {"type": kind, "link": link, "dur_s": dur}
    raise BadMessage("unknown console message")  # pragma: no cover - CONSOLE_TYPES is exhaustive


def parse_any(raw, arm_low: Sequence[float], arm_high: Sequence[float]) -> Dict:
    """One client frame: console (MP 0.7) types first, then the pilot console / base protocol."""
    if isinstance(raw, str) and len(raw) <= web_scene.MAX_MSG_CHARS:
        try:
            msg = json.loads(raw)
        except (ValueError, RecursionError):
            msg = None
        if isinstance(msg, dict) and msg.get("type") in CONSOLE_TYPES:
            return parse_console_message(msg)
    return parse_client_message(raw, arm_low, arm_high)


# ---------------------------------------------------------------- server schemas

_T = {"str": str, "num": (int, float), "int": int, "bool": bool, "list": list, "dict": dict}
SERVER_SCHEMAS = {
    "preset_status": {"tiles": "list", "queue": "dict", "active": ("dict", None), "history": "list",
                      "paused": "bool", "chips": "list", "ready": "dict", "assist": ("dict", None)},
    "map": {"world": "str", "trees": "list", "rows": "list", "blocks": "list", "laterals": "list", "ropes": "list",
            "geofence": "list", "home": ("list", None), "dock": ("dict", None), "annuli": "dict"},
    "coverage": {"tree_id": ("str", None), "phase": ("str", None), "coverage_pct": "dict", "station": ("int", None),
                 "stations": ("int", None), "live": "bool"},
    "mast": {"ext_m": ("num", None), "target_ext": ("num", None), "band": "str", "gated": "bool",
             "arbiter_source": ("str", None), "wind": "dict", "gates": "dict", "live": "bool"},
    "recovery": {"armed": "bool", "active": ("str", None), "prompt": ("str", None), "links": "dict", "live": "bool"},
    "tree_record": {"tree_id": "str", "record": "dict"},
    "mark": {"mark": "dict"},
    "jobsheet": {"row_id": "str", "lang": "str", "sheet": "dict", "html": "str", "voice": "list"},
}
TILE_KEYS = ("id", "tile", "hi", "en", "mode", "enabled", "reason", "level", "executor", "live")
QUEUE_ITEM_KEYS = ("qid", "preset", "eta_start_s", "eta_end_s", "soc_end", "disk_free_end_mb", "blocked")


def _type_ok(v, spec) -> bool:
    specs = spec if isinstance(spec, tuple) else (spec,)
    for s in specs:
        if s is None and v is None:
            return True
        if s is not None and isinstance(v, _T[s]) and not (s in ("num", "int") and isinstance(v, bool)):
            return True
    return False


def validate_server_message(msg: Dict) -> List[str]:
    """Schema errors of one server -> client console message ([] = valid)."""
    if not isinstance(msg, dict):
        return ["not an object"]
    schema = SERVER_SCHEMAS.get(msg.get("type"))
    if schema is None:
        return [f"unknown type {msg.get('type')!r}"]
    errs = [f"{k}: {type(msg.get(k)).__name__}" for k, spec in schema.items() if not _type_ok(msg.get(k), spec)]
    if msg["type"] == "preset_status":
        for tile in msg.get("tiles") or []:
            errs += [f"tile.{k}" for k in TILE_KEYS if k not in tile]
        for it in (msg.get("queue") or {}).get("items") or []:
            errs += [f"queue.{k}" for k in QUEUE_ITEM_KEYS if k not in it]
    try:
        web_scene.encode_message(msg)
    except (TypeError, ValueError) as exc:
        errs.append(f"not JSON: {exc}")
    return errs


# ---------------------------------------------------------------- orchard map

def _xy(v) -> Optional[List[float]]:
    if isinstance(v, dict):
        v = [v.get("x"), v.get("y")]
    if isinstance(v, (list, tuple)) and len(v) >= 2 and all(_num(c) is not None for c in v[:2]):
        return [round(float(v[0]), 3), round(float(v[1]), 3)]
    return None


def tree_orbit_r(tree: Dict) -> float:
    """TREE 360 inner orbit radius r1 = clamp(canopy_r + standoff, 1.2, 3.0) (MP 0.8 default)."""
    cr = _num(tree.get("canopy_r_m")) or 0.0
    so = _num(tree.get("orbit_standoff_m")) or 0.9
    return round(min(3.0, max(1.2, cr + so)), 3)


def orchard_map(field_doc: Optional[Dict], registry: Optional[ps.Registry] = None) -> Dict:
    """The console ``map`` message from the WP4 field registry (MP 0.1). A field without trees (legacy) gives an
    empty map with world name, so the browser shows "no orchard registry"."""
    f = field_doc or {}
    trees = []
    for t in f.get("trees") or []:
        if not isinstance(t, dict) or "tree_id" not in t:
            continue
        xy = _xy(t.get("geo"))
        if xy is None:
            continue
        presets = []
        for p in registry or []:
            crops = list(getattr(p, "crops", ()) or [])
            if (not crops or t.get("crop") in crops) and p.kind not in ("recovery", "work", "dock"):
                presets.append(p.id)
        trees.append({
            "tree_id": t["tree_id"], "tag": t.get("physical_tag") or t["tree_id"], "row_id": t.get("row_id"),
            "block_id": t.get("block_id"), "crop": t.get("crop"), "cultivar": t.get("cultivar"),
            "form": t.get("form"), "xy": xy, "height_m": _num(t.get("height_m")),
            "canopy_r_m": _num(t.get("canopy_r_m")), "trunk_r_m": _num(t.get("trunk_r_m")),
            "orbit_clear": bool(t.get("orbit_clear", True)), "flags": list(t.get("flags") or [])[:8],
            "last_visit": t.get("last_visit"), "presets": presets,
            "ropes": [r for r in (t.get("rope_lines") or []) if isinstance(r, dict)][:4],
            "props": len(t.get("props") or []), "r1": tree_orbit_r(t),
        })
    by_id = {t["tree_id"]: t for t in trees}
    laterals = []
    fx = f.get("fixtures") or {}
    for lat in fx.get("laterals") or []:
        if not isinstance(lat, dict):
            continue
        pts = None
        if isinstance(lat.get("points"), list):
            pts = [p for p in (_xy(q) for q in lat["points"]) if p is not None]
        elif isinstance(lat.get("ring"), dict) and lat["ring"].get("tree_id") in by_id:
            c = by_id[lat["ring"]["tree_id"]]["xy"]
            r = _num(lat["ring"].get("r_m")) or 0.8
            n = 16
            pts = [[round(c[0] + r * math.cos(2 * math.pi * k / n), 3),
                    round(c[1] + r * math.sin(2 * math.pi * k / n), 3)] for k in range(n + 1)]
        if pts and len(pts) >= 2:
            laterals.append({"id": str(lat.get("id", f"lateral_{len(laterals)}")), "points": pts})
    ropes = []
    for t in trees:
        for r in t.pop("ropes"):
            a, b = _xy(r.get("a") or r.get("from") or r.get("p0")), _xy(r.get("b") or r.get("to") or r.get("p1"))
            if a and b:
                ropes.append({"tree_id": t["tree_id"], "points": [a, b], "h_m": _num(r.get("h_m") or r.get("height_m"))})
    blocks = {}
    for t in trees:
        if t["block_id"]:
            blocks.setdefault(t["block_id"], {"block_id": t["block_id"], "crop": t["crop"], "trees": []})["trees"].append(
                t["tree_id"])
    rows = [{"row_id": r.get("row_id"), "block_id": r.get("block_id"), "crop": r.get("crop"), "y": _num(r.get("y")),
             "trees": [t["tree_id"] for t in trees if t["row_id"] == r.get("row_id")]}
            for r in (f.get("rows") or []) if isinstance(r, dict)]
    home = f.get("home") or {}
    dock = f.get("dock") if isinstance(f.get("dock"), dict) else None
    return {"type": "map", "world": str(f.get("world") or "legacy"), "frame": str(f.get("frame") or ""),
            "season": f.get("season"), "trees": trees, "rows": rows, "blocks": list(blocks.values()),
            "laterals": laterals, "ropes": ropes,
            "geofence": [p for p in (_xy(q) for q in (f.get("geofence") or [])) if p is not None],
            "home": _xy(home.get("xy")) if isinstance(home, dict) else None,
            "dock": {"xy": _xy(dock.get("xy")), "yaw_deg": _num(dock.get("yaw_deg")),
                     "size_m": dock.get("size_m")} if dock else None,
            "annuli": {t["tree_id"]: {"r1": t["r1"], "clear": t["orbit_clear"]} for t in trees}}


def tree_record_summary(r: Dict, media_key: Optional[str] = None) -> Dict:
    """What the tree card shows of a TreeRecord (no geo, no person data; leads stay leads). With ``media_key`` every
    /media URL carries ``?k=<key>`` (the bridge serves media only with that per-process capability)."""
    keep = ("schema", "run_id", "mission_id", "tree_id", "physical_tag", "crop", "cultivar", "block_id", "row_id",
            "visit_t", "presets", "status", "coverage", "canopy", "trunk", "leaf", "fruit", "harvest", "prune",
            "next_actions", "marks_n", "detector", "assisted", "flags", "label", "trimmed")
    out = {k: r.get(k) for k in keep if k in r}
    syms = r.get("symptoms") or []
    out["symptoms_n"] = r.get("symptoms_n", len(syms))
    out["leads"] = [{"class": s.get("class"), "band": s.get("band"), "clock": s.get("clock"), "stage": s.get("stage"),
                     "count": s.get("count")} for s in syms[:12] if isinstance(s, dict)]
    media = r.get("media") or {}
    run, tree = r.get("run_id"), r.get("tree_id")
    strips, frames = [], []
    if isinstance(run, str) and isinstance(tree, str) and ID_RE.match(run) and ID_RE.match(tree):
        base = f"/media/{run}/tree360/{tree}/"
        q = f"?k={media_key}" if media_key else ""
        for b in ("low", "mid", "high"):
            strips.append({"band": b, "url": base + f"strip_{b}.jpg" + q})
        for p in (media.get("keyframe_paths") or [])[:8]:
            if isinstance(p, str) and MEDIA_REL_RE.match(p):
                frames.append(base + p + q)
    out["media"] = {"manifest_root": media.get("manifest_root"), "strips": strips, "keyframes": frames,
                    "keyframes_n": media.get("keyframes_n", len(media.get("keyframe_paths") or []))}
    return out


MEDIA_REL_RE = re.compile(r"^(?:(?:low|mid|high)/[0-9]{1,6}\.jpg|strip_(?:low|mid|high)\.jpg)$")
MEDIA_EXT = {".jpg": "image/jpeg"}


def media_key_ok(request_path: str, key: Optional[str]) -> bool:
    """True if ``request_path`` carries ``?k=<key>`` (constant-time). A falsy ``key`` refuses everything."""
    if not key or "?" not in request_path:
        return False
    try:
        given = parse_qs(request_path.split("?", 1)[1], max_num_fields=4).get("k") or [""]
    except ValueError:
        return False
    return hmac.compare_digest(given[0].encode(), key.encode())


def resolve_media_path(runs_root: str, request_path: str, allowed: Iterable[Tuple[str, str]],
                       key: Optional[str] = None) -> Optional[str]:
    """``/media/<run_id>/tree360/<tree_id>/<file>`` -> a file under ``runs_root``, or None. Only TREE 360 strips and
    keyframes (FilteredFrames written by WP25's recorder) of a (run_id, tree_id) a TreeRecord named are served, and
    with ``key`` only when the URL carries that capability (``?k=``; the run / tree ids alone are guessable)."""
    if key is not None and not media_key_ok(request_path, key):
        return None
    path = request_path.split("?", 1)[0]
    parts = path.split("/")
    if len(parts) < 6 or parts[0] != "" or parts[1] != "media" or parts[3] != "tree360":
        return None
    run, tree, rel = parts[2], parts[4], "/".join(parts[5:])
    if not (ID_RE.match(run) and ID_RE.match(tree) and MEDIA_REL_RE.match(rel)):
        return None
    if (run, tree) not in set(allowed):
        return None
    root = os.path.realpath(runs_root)
    full = os.path.realpath(os.path.join(root, run, "tree360", tree, rel))
    if os.path.commonpath([root, full]) != root or not os.path.isfile(full):
        return None
    return full


_number = web_scene._number


def _loads(text) -> Optional[Dict]:
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def tour_step_index(doc: Optional[Dict]) -> Dict[Tuple[str, str], str]:
    """(preset id, canonical targets JSON) -> step id of WP27's orchard_tour (the demo-tour executor)."""
    out = {}
    for s in (doc or {}).get("steps") or []:
        if isinstance(s, dict) and s.get("preset") and s.get("id"):
            out[(s["preset"], json.dumps(s.get("targets") or {}, sort_keys=True))] = s["id"]
    return out


def side_of_clock(clock: int) -> str:
    """Seen from the alley (MP 0.6 convention, viewer due south): 12 far = N, 3 = E, 6 = S, 9 = W."""
    return "NESW"[int(((int(clock) % 12) + 1.5) // 3) % 4]


H_CLASS_M = {"neeche": 0.5, "beech": 1.3, "upar": 2.2}


class ConsoleHub:
    """Everything the WP10 console does between the browsers and the hooks, ROS-free (the bridge glues ROS).

    Inputs arrive as JSON text with ``ingest(key, text, now)`` (keys of HUB_TOPICS_IN). Client commands go through
    ``handle``; ``tick`` dispatches the queue, follows the running item's executor, beats the console link and
    collects what to publish in ``outbox`` ((HUB_TOPICS_OUT key, payload)); ``drive`` is the live gait relay of the
    hooks that cannot drive the gait inside sim_runner (TREE 360, recovery, work assist) onto /nandi/cmd_vel;
    ``streams`` returns the <= 2 Hz MP 0.7 server messages; ``events`` holds one-off messages for every client.
    """

    def __init__(self, field_doc: Optional[Dict], registry: Optional[ps.Registry] = None, *, device: str = "nandi-sim",
                 sim_local_link: bool = False, sim_tools: bool = True, novnc: bool = False,
                 tour_doc: Optional[Dict] = None, run_id: Optional[str] = None):
        self.field = field_doc if (field_doc or {}).get("trees") else None
        self.world = str((field_doc or {}).get("world") or "legacy")
        self.registry = registry if registry is not None else ps.Registry.load()
        self.device = device
        self.sim_tools = bool(sim_tools)
        self.sim_local_link = bool(sim_local_link) and self.sim_tools   # sim_tools is off in field mode
        self.media_key = secrets.token_urlsafe(18)   # capability in every /media URL (sent to authenticated clients)
        self.novnc = bool(novnc)
        self.run_id = run_id or f"console-{rec.mission_id(device)}"
        self.queue = ml.PresetQueue()
        self.map = orchard_map(field_doc, self.registry)
        self.tour_steps = tour_step_index(tour_doc)
        self.data: Dict[str, Dict] = {}
        self.stamps: Dict[str, float] = {}
        self.active: Optional[Dict] = None
        self.history: deque = deque(maxlen=12)
        self.paused = False
        self.paused_reason = ""
        self.blocked: Optional[str] = None
        self.records: Dict[str, Dict] = {}
        self.media_allowed: set = set()
        self.marks: List[Dict] = []
        self.assist: Optional[Dict] = None
        self.outbox: List[Tuple[str, Dict]] = []
        self.events: List[Dict] = []
        self.limiter = StreamLimiter()
        self._tiles: List[Dict] = []
        self._tiles_t = -math.inf
        self._ready: Dict[str, Dict] = {}
        self._beat = {"cloud": -math.inf, "local": -math.inf}
        self._recovery_pending: Optional[Dict] = None
        self._estop_seen = False
        self._relaying = False
        self.drive_hold: Optional[str] = None   # STOP / pause: relay nothing until an operator resume or start
        self._hold_rec_active = None
        self.context_fault: Optional[str] = None
        self._stop_seq: Optional[int] = None
        self._mark_n = 0
        self._last_tree360_t = None

    # ------------------------------------------------------------ inputs
    def ingest(self, key: str, text, now: float):
        obj = _loads(text)
        if obj is None or key not in HUB_TOPICS_IN:
            return
        if key == "world":   # DP-13: never keep actor (person) positions
            actors = [a for a in (obj.get("actors") or []) if isinstance(a, dict)]
            obj = {"light": obj.get("light"), "laterals": obj.get("laterals"), "world": obj.get("world"),
                   "actors_n": len(actors), "persons_n": sum(1 for a in actors if a.get("class") == "person")}
        self.data[key], self.stamps[key] = obj, now
        if key == "tree_record" and isinstance(obj.get("tree_id"), str):
            s = tree_record_summary(obj, self.media_key)
            self.records[obj["tree_id"]] = s
            if isinstance(obj.get("run_id"), str):
                self.media_allowed.add((obj["run_id"], obj["tree_id"]))
            self.events.append({"type": "tree_record", "tree_id": obj["tree_id"], "record": s})
        elif key == "mark":
            self._add_mark(obj)
        elif key == "assist" or (key == "tree360" and isinstance(obj.get("assist"), dict)):
            a = obj if key == "assist" else dict(obj["assist"], source="tree360", tree_id=obj.get("tree_id"))
            if isinstance(a, dict):
                self.assist = {"source": str(a.get("source") or "perception"), "reason": str(a.get("reason") or ""),
                               "tree_id": a.get("tree_id"), "t": now,
                               "options": list(ASSIST_REPLIES)}
        if key == "tree360" and self.assist and self.assist.get("source") == "tree360" and not obj.get("assist"):
            self.assist = None

    def get(self, key: str, now: float, fresh: Optional[float] = FRESH_S) -> Optional[Dict]:
        t = self.stamps.get(key)
        if t is None or (fresh is not None and now - t > fresh):
            return None
        return self.data.get(key)

    def _add_mark(self, m: Dict) -> bool:
        if rec.validate_mark(m):
            return False
        key = rec.mark_key(m)
        if any(rec.mark_key(x) == key for x in self.marks):
            return False
        self.marks.append(m)
        del self.marks[:-500]
        self.events.append({"type": "mark", "mark": m})
        return True

    # ------------------------------------------------------------ Ready context
    def ready_context(self, now: float, robot_xy=(0.0, 0.0), estop: bool = False, soc: Optional[float] = None,
                      disk_free_mb: float = 4096.0, fallen: bool = False) -> rc.ReadyContext:
        """The WP27 Ready context from the live topics. A malformed status payload never raises out of here (that
        would stop the 10 Hz hub tick, its heartbeats and the relay): it fails closed (rtk NONE, stability latched,
        reason ``sensor_fault`` in ``self.context_fault``)."""
        try:
            ctx = self._ready_context(now, robot_xy, estop, soc, disk_free_mb, fallen, ignore=False)
            self.context_fault = None
            return ctx
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            self.context_fault = f"sensor_fault: {type(exc).__name__}: {exc}"[:160]
            return self._ready_context(now, robot_xy, estop, soc, disk_free_mb, fallen, ignore=True)

    def _ready_context(self, now, robot_xy, estop, soc, disk_free_mb, fallen, ignore: bool) -> rc.ReadyContext:
        def g(k):
            v = None if ignore else self.get(k, now)
            return v if isinstance(v, dict) else {}
        gnss, bat, wind, weather, stab = g("gnss"), g("battery"), g("wind"), g("weather"), g("stability")
        mast, tools, tr, world, rs, work = g("mast"), g("tools"), g("transfer"), g("world"), g("recovery"), g("work")
        soc_v = _num(bat.get("soc")) if bat else None
        if soc_v is None and bat:
            soc_v = _num(bat.get("percentage"))
        if soc_v is None:
            soc_v = soc if soc is not None else 1.0
        slope = stab.get("slope_deg") if isinstance(stab.get("slope_deg"), dict) else {}
        light = world.get("light") if isinstance(world.get("light"), dict) else {}
        cells = tr.get("cells") if isinstance(tr.get("cells"), list) else None
        beh = rs.get("behaviour") if isinstance(rs.get("behaviour"), dict) else {}
        confirms = beh.get("confirms") if isinstance(beh.get("confirms"), dict) else {}
        weather_ok = weather and all(k in weather for k in ("month", "rain", "temp_c"))
        kw = dict(
            rtk=str(gnss.get("rtk") or "NONE"), soc=float(soc_v), disk_free_mb=float(disk_free_mb),
            robot_xy=tuple(robot_xy), home_xy=tuple(self.map["home"]) if self.map.get("home") else None,
            estop=bool(estop),
            local_link=str((self.get("link_local", now) or {}).get("state") or "unknown"),
            cloud_link=str((self.get("link_cloud", now) or {}).get("state") or "unknown"),
            stability_ok=not ignore and not bool(stab.get("latched")),
            cross_slope_deg=abs(_num(slope.get("roll")) or 0.0), pitch_deg=abs(_num(slope.get("pitch")) or 0.0),
            wind_mps=_num(wind.get("mean_mps")) or _num(wind.get("speed_mps")) or 0.0, gust=bool(wind.get("gust")),
            weather=dict(weather) if weather_ok else None,
            season_profile=str((self.field or {}).get("season") or "up"),
            tool_mounted=tools.get("mounted") or None,
            holding=bool(tools.get("caught")), payload_kg=_num(tr.get("payload_kg")) or 0.0,
            tray_cells_free=sum(1 for c in cells if not c) if cells is not None else 8,
            cradle_free=not tr.get("cradle"), novnc=self.novnc,
            # person actors are composed only with noVNC off (WP4); unknown world state = assume they exist
            actors=bool(world.get("persons_n", 1)) if world else not self.novnc,
            marks_n=len(self.marks), tag_paired=isinstance(work.get("tag"), dict) and bool(work["tag"].get("issued")),
            ext_m=_num(mast.get("ext_m")) or 0.0, fallen=bool(fallen or rs.get("active") == "fallen"),
            estop_confirmed=(bool(confirms.get("local")), bool(confirms.get("operator"))),
        )
        if tools and isinstance(tools.get("parked"), list):
            kw["tools_available"] = tuple(str(t) for t in tools["parked"])
        lux = _num(light.get("lux"))
        if lux is not None:
            kw["light_lux"] = lux
        return rc.ReadyContext(**kw)

    def executor_of(self, preset: ps.Preset, targets: Optional[Dict] = None) -> str:
        ex = EXECUTOR_OF_MODE.get(preset.mode)
        if ex:
            return ex
        if targets is not None and (preset.id, json.dumps(targets, sort_keys=True)) in self.tour_steps:
            return "demo"
        return "demo" if any(p == preset.id for p, _ in self.tour_steps) else "plan_only"

    def executor_live(self, ex: str, now: float) -> bool:
        if ex in ("tree360", "recovery", "recovery_confirm", "work"):
            return self.field is not None
        if ex == "dock":
            return self.get("dock", now, 10.0) is not None
        if ex == "demo":
            return self.get("demo", now, 10.0) is not None
        return False

    def tiles(self, now: float, ctx: rc.ReadyContext) -> List[Dict]:
        if now - self._tiles_t < TILES_PERIOD_S and self._tiles:
            return self._tiles
        ready = rc.check_all(self.registry, ctx, self.field)
        self._ready = {k: v.to_dict() for k, v in ready.items()}
        tiles = ps.console_tiles(self.registry, ready)
        for t in tiles:
            p = self.registry.get(t["id"])
            ex = self.executor_of(p)
            t["executor"], t["live"] = ex, self.executor_live(ex, now)
            t["crops"] = list(p.crops)
            t["kind"] = p.kind
            r = self._ready.get(t["id"]) or {}
            t["reasons"] = list(r.get("reasons") or [])[:6]
            t["chips"] = list(r.get("chips") or [])[:4]
        self._tiles, self._tiles_t = tiles, now
        return tiles

    # ------------------------------------------------------------ client commands
    def handle(self, msg: Dict, now: float, ctx: rc.ReadyContext, estop: bool = False) -> Tuple[bool, str, List[Dict]]:
        """One parsed console message -> (ok, message for the toast, replies for the requester only)."""
        kind = msg["type"]
        try:
            fn = getattr(self, f"_on_{kind}")
            return fn(msg, now, ctx, estop)
        except ml.Rejection as exc:
            return False, f"{exc.reason}: {exc.detail}", []
        except ps.PresetError as exc:
            return False, f"{getattr(exc, 'reason', 'param_invalid')}: {exc}", []

    def _on_preset_start(self, msg, now, ctx, estop):
        if self.field is None:
            return False, f"registry_mismatch: world {self.world} has no orchard tree registry", []
        try:
            preset = self.registry.get(msg["preset"])
        except ps.PresetError as exc:
            return False, f"preset_unknown: {exc}", []
        if estop and preset.mode != "estop_recover":
            return False, "estop: E-STOP is latched: only E-STOP RECOVERY may start", []
        targets = msg["targets"] or None
        res = rc.check(preset, ctx, targets, msg["params"] or None, self.field)
        if res.level == "no_go":
            return False, f"{', '.join(res.reasons) or 'no_go'}: Ready check refused {preset.id}", []
        if res.level == "degraded" and not msg["confirm"]:
            return False, f"confirm_required: {', '.join(res.reasons)} (degraded: confirm to start)", []
        start = {"preset": preset.id, "targets": msg["targets"], "params": msg["params"]}
        cmd = ml.from_preset_start(start, self.registry, self.field)
        trees = cmd["plan"].get("tree_ids") or []
        xy = self._tree_xy(trees)
        qid = self.queue.add(cmd, res.estimate, xy[0] if xy else ctx.robot_xy, xy[-1] if xy else ctx.robot_xy,
                             res.level, res.reasons, msg["confirm"])
        ex = self.executor_of(preset, msg["targets"])
        for it in self.queue.items:
            if it.qid == qid:
                it.cmd.setdefault("console", {}).update(executor=ex, adjust=dict(res.adjust))
        if self.active is None:
            self._release_hold()   # an explicit start after a STOP: the queue may run again
        note = "" if self.executor_live(ex, now) else f" (executor {ex}: not live in this stack)"
        return True, f"{preset.id} queued as {qid}{note}", []

    def _tree_xy(self, tree_ids) -> List[Tuple[float, float]]:
        by = {t["tree_id"]: t["xy"] for t in self.map["trees"]}
        return [tuple(by[t]) for t in tree_ids if t in by]

    def _recovery_drives(self, now: float) -> bool:
        r = self.get("drive_recovery", now, DRIVE_STALE_S)
        return r is not None and not r.get("release") and not r.get("stop")

    def stop_motion(self, now: float, reason: str = "stop") -> Optional[str]:
        """The pilot STOP (/nandi/stop_motion, also before a harvest, a reset, a mission abort, the E-STOP) or a
        Pause: hold the relay (one release, then nothing) until an operator Resume, a preset start or a recovery
        run from the menu, pause the running item (a recovery behaviour has no pause: it is aborted) and pause the
        queue, so the next 10 Hz relay tick cannot restart the walk. None when nothing runs, relays or waits."""
        a = self.active
        if a is None and not self._relaying and not self.queue.items:
            return None
        self.drive_hold = reason
        self._recovery_pending = None   # a recovery run still waiting for the arm must not start after the STOP
        self._hold_rec_active = (self.get("recovery", now) or {}).get("active")
        note = "relay stopped"
        if a is not None and not a.get("aborting"):
            if a["executor"] in ("recovery", "recovery_confirm"):
                self._abort_active(now, reason)
                note = f"{a['preset']} aborted (a recovery behaviour has no pause)"
            elif a.get("paused") or self._exec_cmd(a, "pause"):
                a["paused"] = True
                note = f"{a['preset']} paused"
        elif a is None and self._recovery_drives(now):
            self.outbox.append(("recovery_cmd", {"abort": True}))
            note = "recovery motion stopped (aborted: a recovery behaviour has no pause)"
        if not self.paused:
            self.paused, self.paused_reason = True, reason
        return note

    def observe_stop_seq(self, seq, now: float) -> Optional[str]:
        """``stop_seq`` of /nandi/locomotion_status: every /nandi/stop_motion call bumps it, whoever made it (Tk
        panel, API, another console, a mission). A change holds the relay like the console's own STOP."""
        if not isinstance(seq, int) or isinstance(seq, bool):
            return None
        prev, self._stop_seq = self._stop_seq, seq
        if prev is None or seq == prev:
            return None
        return self.stop_motion(now, "stop")

    def _release_hold(self, clear_stop_pause: bool = True):
        self.drive_hold = None
        if clear_stop_pause and self.paused and self.paused_reason in STOP_PAUSE_REASONS:
            self.paused, self.paused_reason = False, ""

    def _on_preset_pause(self, msg, now, ctx, estop):
        a = self.active
        if a is None or a["executor"] in ("recovery", "recovery_confirm"):
            note = self.stop_motion(now, "pause") if (a is not None or self._relaying) else None
            return (True, note, []) if note else (False, "nothing is running", [])
        if not self._exec_cmd(a, "pause"):
            return False, f"{a['preset']} ({a['executor']}) cannot pause", []
        a["paused"] = True
        self.drive_hold = "pause"   # the relay waits for Resume even if the executor's pause is lost
        self._hold_rec_active = (self.get("recovery", now) or {}).get("active")
        return True, f"{a['preset']} paused", []

    def _on_preset_resume(self, msg, now, ctx, estop):
        a = self.active
        if estop:   # also with nothing running: the E-STOP's queue pause ends only after the reset
            return False, "estop: reset the E-STOP first", []
        if a is None:
            was = self.paused or self.drive_hold is not None
            self.paused, self.paused_reason, self.drive_hold = False, "", None
            return (True, "queue resumed", []) if was else (False, "nothing is paused", [])
        if not a.get("paused"):
            if self.drive_hold is not None:
                self._release_hold()
                return True, f"{a['preset']}: relay resumed", []
            return False, f"{a['preset']} is not paused", []
        if not self._exec_cmd(a, "resume"):
            return False, f"{a['preset']} cannot resume", []
        a["paused"] = False
        self._release_hold()
        return True, f"{a['preset']} resumed", []

    def _on_preset_abort(self, msg, now, ctx, estop):
        qid = msg.get("qid")
        if qid and (self.active is None or self.active["qid"] != qid):
            if self.queue.remove(qid):
                return True, f"{qid} removed from the queue", []
            return False, f"{qid} is not queued", []
        a = self.active
        if a is None:
            return False, "nothing is running", []
        self._abort_active(now, "operator")
        return True, f"{a['preset']} aborting", []

    def _on_tree_mark(self, msg, now, ctx, estop):
        tree = next((t for t in self.map["trees"] if t["tree_id"] == msg["tree_id"]), None)
        if tree is None:
            return False, f"registry_mismatch: no tree {msg['tree_id']}", []
        h_m = msg["h_m"] if msg["h_m"] is not None else H_CLASS_M[msg["h_class"]]
        self._mark_n += 1
        m = {"schema": rec.MARK_SCHEMA, "tree_id": tree["tree_id"], "physical_tag": tree["tag"], "kind": msg["kind"],
             "target_id": f"console-{self._mark_n}", "clock": msg["clock"],
             "side": msg["side"] or side_of_clock(msg["clock"]),
             "h_class": "neeche" if h_m < 0.8 else ("beech" if h_m <= 1.8 else "upar"), "h_m": round(h_m, 2),
             "stage": None, "reason": "operator", "diameter_mm": None, "paste_required": False,
             "tool_hint": msg["tool_hint"], "source": "console"}
        if msg["note"]:
            m["note"] = msg["note"]
        errs = rec.validate_mark(m)
        if errs:
            return False, f"mark refused: {', '.join(errs)}", []
        self._add_mark(m)
        self.outbox.append(("mark", m))
        return True, f"MARK {m['kind']} on {tree['tag']} ({m['clock']} o'clock, {m['h_class']})", []

    def _on_assist_reply(self, msg, now, ctx, estop):
        a, reply = self.active, msg["reply"]
        if self.assist is None and a is None:
            return False, "no assist request is open", []
        self.assist = None
        if a is None:
            return True, f"assist {reply} noted", []
        if reply == "take_over":
            self._exec_cmd(a, "pause")
            a["paused"] = True
            return True, f"{a['preset']} paused: you have control (drive, then Resume or Abort)", []
        self._abort_active(now, "assist_skip" if reply == "skip" else "assist_halt")
        if reply == "halt":
            self.paused, self.paused_reason = True, "assist_halt"
            return True, f"{a['preset']} halted; queue paused (Resume to continue)", []
        return True, f"{a['preset']}: target skipped", []

    def _on_queue_reorder(self, msg, now, ctx, estop):
        self.queue.reorder(msg["order"])
        return True, "queue reordered", []

    def _on_jobsheet_request(self, msg, now, ctx, estop):
        row = msg["row_id"]
        rows = {r["row_id"] for r in self.map["rows"]} | {t["row_id"] for t in self.map["trees"]}
        if row not in rows:
            return False, f"no row {row}", []
        ident = rec.identity(self.run_id, self.device)
        sheets = js.build_job_sheets(self.marks, (self.field or {}).get("trees") or [], ident, rows=[row])
        sheet = sheets.get(row)
        if sheet is None:
            return False, f"no job sheet for row {row}", []
        reply = {"type": "jobsheet", "row_id": row, "lang": msg["lang"], "sheet": sheet,
                 "html": js.render_html(sheet), "voice": js.voice_lines(sheet),
                 "missing": js.missing_marks([m for m in self.marks if self._row_of(m) == row], sheets)}
        return True, f"job sheet {row}: {sheet['n_marks']} jobs", [reply]

    def _row_of(self, m: Dict) -> Optional[str]:
        return next((t["row_id"] for t in self.map["trees"] if t["tree_id"] == m.get("tree_id")), None)

    def _on_recovery(self, msg, now, ctx, estop):
        if self.field is None:
            return False, "recovery behaviours run in the orchard worlds only", []
        op = msg["op"]
        payload = {"check": msg["item"]} if op == "check" else dict(RECOVERY_OPS[op])
        rs = self.get("recovery", now) or {}
        if op.startswith("run_") or op == "confirm_operator":
            self._release_hold(clear_stop_pause=False)   # an explicit recovery run from the menu may move
            if self.active and self.active["executor"] not in ("recovery", "recovery_confirm"):
                self._abort_active(now, "recovery")
            if not rs.get("armed"):
                self._recovery_pending = payload
                self.outbox.append(("recovery_cmd", {"arm": True}))
                return True, f"recovery: arming, then {op}", []
        self.outbox.append(("recovery_cmd", payload))
        return True, f"recovery: {op}", []

    def _on_crown(self, msg, now, ctx, estop):
        if estop:
            return False, "estop: crown requests are refused while the E-STOP is latched", []
        if self.active and self.active["executor"] == "tree360":
            return False, "crown_gate: TREE 360 controls the crown while it runs", []
        self.outbox.append(("mast_cmd", {"source": "preset", "band": msg["band"], "reason": "operator"}))
        return True, f"crown request: {msg['band']} (WP24 arbiter decides)", []

    def _on_link_test(self, msg, now, ctx, estop):
        if not self.sim_tools:
            return False, "link_test is a sim tool (off in field mode)", []
        self.outbox.append(("link_cmd", {"drop": {"link": msg["link"], "dur": msg["dur_s"]}}))
        return True, f"sim: {msg['link']} link dropped for {msg['dur_s']:.0f} s", []

    # ------------------------------------------------------------ executors
    def _exec_cmd(self, a: Dict, op: str) -> bool:
        ex = a["executor"]
        if ex == "tree360":
            payload = {"cmd": op, "reason": "operator"} if op == "abort" else {"cmd": op}
            self.outbox.append(("tree360_cmd", payload))
        elif ex == "work":
            self.outbox.append(("work_cmd", {"op": op}))
        elif ex == "demo":
            self.outbox.append(("demo_run", {"cmd": op}))
        elif ex == "dock" and op == "abort":
            self.outbox.append(("dock_cmd", {"op": "abort"}))
        elif ex in ("recovery", "recovery_confirm") and op == "abort":
            self.outbox.append(("recovery_cmd", {"abort": True}))
        else:
            return False
        return True

    def _abort_active(self, now: float, reason: str):
        a = self.active
        if a is None or a.get("aborting"):
            return
        a["aborting"], a["abort_reason"], a["t_abort"] = True, reason, now
        if not self._exec_cmd(a, "abort") or a["executor"] == "plan_only":
            self._finish(now, "aborted", reason)

    def _finish(self, now: float, status: str, reason: str = ""):
        a = self.active
        if a is None:
            return
        self.history.appendleft({"qid": a["qid"], "preset": a["preset"], "tree_ids": a["tree_ids"],
                                 "executor": a["executor"], "status": status, "reason": reason,
                                 "dur_s": round(now - a["t_start"], 1)})
        self.events.append({"type": "result", "name": "preset", "success": status == "done",
                            "message": f"{a['preset']} {status}{': ' + reason if reason else ''}"})
        self.active = None

    def _start(self, item, now: float):
        cmd = item.cmd
        preset = self.registry.get(item.preset)
        ex = (cmd.get("console") or {}).get("executor") or self.executor_of(preset)
        trees = list(cmd["plan"].get("tree_ids") or [])
        a = {"qid": item.qid, "preset": item.preset, "mode": cmd["mode"], "executor": ex, "tree_ids": trees,
             "pending": list(trees), "mission_id": cmd.get("mission_id"), "t_start": now, "t_phase": now,
             "state": "starting", "paused": False, "cmd": cmd, "seen": False}
        self.active = a
        if ex == "tree360":
            self._tree360_next(a, now)
        elif ex in ("recovery", "recovery_confirm"):
            run = {"confirm": "operator"} if ex == "recovery_confirm" else {"run": cmd["mode"]}
            rs = self.get("recovery", now) or {}
            if rs.get("armed"):
                self.outbox.append(("recovery_cmd", run))
            else:
                self._recovery_pending = run
                self.outbox.append(("recovery_cmd", {"arm": True}))
        elif ex == "work":
            self.outbox.append(("work_cmd", {"op": "start", "mode": WORK_MODE[cmd["mode"]],
                                             "params": self._work_params(cmd)}))
        elif ex == "dock":
            self.outbox.append(("dock_cmd", {"op": "start", "sync": "on"}))
        elif ex == "demo" and self.executor_live("demo", now):
            step = self.tour_steps.get((item.preset, json.dumps(cmd["plan"].get("targets") or {}, sort_keys=True)))
            if step is None:
                self.outbox.append(("preset_run", cmd))
                self._finish(now, "no_live_executor", "no orchard_tour step for these targets")
                return
            self.outbox.append(("demo_run", {"steps": [step], "program": "orchard_tour", "transit": "teleport"}))
            a["step"] = step
        else:
            self.outbox.append(("preset_run", cmd))
            self._finish(now, "no_live_executor",
                         "published on /nandi/preset/run; no live executor for this preset in this stack")

    @staticmethod
    def _work_params(cmd: Dict) -> Dict:
        p = dict(cmd["plan"].get("params") or {})
        if cmd["mode"] == "tote_carry":
            return {"tote_kg": float(p.get("tote_kg", 2.0)), "dest": {"to": "dock"}}
        if cmd["mode"] == "lead_to_marks":
            return {"stop_m": float(p.get("stop_distance_m", 1.5))}
        return {}

    def _tree360_params(self, a: Dict) -> Dict:
        r = dict(a["cmd"]["plan"].get("resolved") or {})
        keep = ("arc_mode", "standoff_m", "stations", "bands", "settle_s", "capture_s", "v_mps", "crown_walk")
        out = {k: r[k] for k in keep if k in r}
        adj = (a["cmd"].get("console") or {}).get("adjust") or {}
        cw = adj.get("crown_walk") or out.get("crown_walk")
        # WP27 'snake_mid' (arcs at mid) has no TREE 360 mode: run stop-and-go (crown up only standing)
        out["crown_walk"] = "stop_and_go" if cw in ("snake_mid", "stop_and_go") else "snake"
        out["run_id"] = re.sub(r"[^A-Za-z0-9_.-]", "-", str(a.get("mission_id") or self.run_id))[:60]
        out["reason"] = "operator"
        return out

    def _tree360_next(self, a: Dict, now: float):
        tree = a["pending"].pop(0)
        a["tree"], a["t_phase"], a["seen"] = tree, now, False
        self.outbox.append(("tree360_cmd", {"cmd": "start", "tree_id": tree, "params": self._tree360_params(a)}))

    def _follow(self, now: float):
        """Advance the running item from its executor's state."""
        a = self.active
        if a is None:
            return
        ex = a["executor"]
        if a.get("aborting") and now - a["t_abort"] > 15.0:
            self._finish(now, "aborted", a.get("abort_reason", "") + " (no executor answer)")
            return
        if ex == "tree360":
            t = self.stamps.get("tree360")
            p = self.data.get("tree360") or {}
            if t is not None and t >= a["t_phase"] and (p.get("tree_id") == a.get("tree") or p.get("error")):
                if p.get("error") and not a["seen"]:
                    self._finish(now, "refused", str(p["error"])[:200])
                    return
                a["seen"], a["state"] = True, "running"
                if p.get("phase") in TERMINAL_T360:
                    if p["phase"] == "aborted" or a.get("aborting"):
                        self._finish(now, "aborted", str(p.get("abort_reason") or a.get("abort_reason") or ""))
                    elif a["pending"]:
                        self._tree360_next(a, now)
                    else:
                        self._finish(now, "done")
            elif not a["seen"] and now - a["t_phase"] > TREE360_START_TIMEOUT_S:
                self._finish(now, "no_live_executor", "no /nandi/tree360/progress (tree360 hook not running?)")
        elif ex in ("recovery", "recovery_confirm"):
            rs = self.get("recovery", now) or {}
            name = "estop_recover" if ex == "recovery_confirm" else a["mode"]
            if rs.get("active") == name:
                a["seen"], a["state"] = True, "running"
            elif a["seen"] and rs.get("active") != name:
                beh = rs.get("behaviour") or {}
                hist = rs.get("history") or []
                out = (hist[-1] if hist and isinstance(hist[-1], dict) else beh) or {}
                oc = out.get("outcome")
                status = ("done" if oc in RECOVERY_DONE else "halted" if oc == "halted"
                          else "assist" if oc == "assist" else "failed")
                self._finish(now, status, str(oc or out.get("reason") or ""))
            elif not a["seen"] and now - a["t_start"] > EXEC_START_TIMEOUT_S + 4.0:
                self._finish(now, "refused", "recovery hook did not start the behaviour (see its cmd_result)")
        elif ex == "work":
            w = self.get("work", now) or {}
            if w.get("mode") == WORK_MODE.get(a["mode"]) and w.get("phase") not in (None, "idle"):
                a["seen"], a["state"] = True, "running"
                if w.get("phase") in ("done", "aborted", "stopped"):
                    self._finish(now, "done" if w["phase"] == "done" else "aborted", str(w.get("hold") or ""))
            elif a["seen"]:
                self._finish(now, "done")
            elif now - a["t_start"] > EXEC_START_TIMEOUT_S:
                ev = self.data.get("work_events") or {}
                why = ",".join(ev.get("reasons") or []) if ev.get("event") == "refused" else "no /nandi/work/state"
                self._finish(now, "refused", why)
        elif ex == "dock":
            d = self.get("dock", now) or {}
            if d.get("step") not in (None, "idle"):
                a["seen"], a["state"] = True, "running"
                if d.get("step") in ("done", "assist"):
                    self._finish(now, "done" if d["step"] == "done" else "assist", str(d.get("reason") or ""))
            elif now - a["t_start"] > EXEC_START_TIMEOUT_S:
                self._finish(now, "no_live_executor", "no /nandi/dock/state (WP13 dock hook not in this stack)")
        elif ex == "demo":
            d = self.get("demo", now, 10.0) or {}
            if d.get("state") in ("loading", "running", "finishing"):
                a["seen"], a["state"] = True, "running"
            elif a["seen"] and d.get("state") in ("done", "error", "idle"):
                steps = [s for s in d.get("steps") or [] if s.get("id") == a.get("step")]
                st_ = steps[0].get("status") if steps else d.get("state")
                self._finish(now, "done" if st_ == "done" else str(st_), str((steps[0] if steps else {}).get("reason") or ""))
            elif not a["seen"] and now - a["t_start"] > 20.0:
                self._finish(now, "no_live_executor", "demo tour node did not start")

    # ------------------------------------------------------------ periodic
    def tick(self, now: float, ctx: rc.ReadyContext, estop: bool = False, clients_alive: bool = False):
        # Heartbeats first: a fault further down (a malformed executor state) must never stop the link beats.
        if clients_alive and now - self._beat["cloud"] >= 1.0 / CLOUD_BEAT_HZ:
            self._beat["cloud"] = now
            self.outbox.append(("cloud_beat", {"src": "console"}))
        if self.sim_local_link and self.field is not None and now - self._beat["local"] >= 1.0 / LOCAL_BEAT_HZ - 1e-6:
            self._beat["local"] = now
            self.outbox.append(("local_beat", {"src": "sim_handheld"}))
        if estop and not self._estop_seen:
            if self.active is not None:
                self._abort_active(now, "estop")
            self.paused, self.paused_reason = True, "estop"
        self._estop_seen = estop
        rs = self.get("recovery", now) or {}
        if self.drive_hold is not None:
            # a recovery behaviour that starts after the STOP (link loss, low SoC: WP28 safety) may move the robot
            cur = rs.get("active")
            if cur and cur != self._hold_rec_active:
                self.drive_hold = None
            self._hold_rec_active = cur
        if self._recovery_pending is not None and rs.get("armed"):
            self.outbox.append(("recovery_cmd", self._recovery_pending))
            self._recovery_pending = None
        self._follow(now)
        self.blocked = None
        if self.active is None and not self.paused and not estop and self.queue.items:
            self._dispatch(now, ctx)

    def _dispatch(self, now: float, ctx: rc.ReadyContext):
        def recheck(item):
            p = self.registry.get(item.preset)
            return rc.check(p, ctx, item.cmd["plan"].get("targets") or None, item.cmd["plan"].get("params") or None,
                            self.field)
        try:
            item = self.queue.pop(ctx.soc, ctx.disk_free_mb, ctx.robot_xy, ctx.home_xy or (0.0, 0.0), recheck)
        except ml.Rejection as exc:
            self.blocked = f"{exc.reason}: {exc.detail}"
            return
        if item is not None:
            self._start(item, now)

    def drive(self, now: float, estop: bool = False) -> Optional[Tuple[float, float, float, bool]]:
        """The gait command to relay now ((vx, vy, yaw, hold) for /nandi/cmd_vel) or None (nothing to send).
        Recovery motion beats TREE 360 beats work assist. A stale drive (> 0.5 s), a stand / stop, the end of the
        run, a pilot STOP / Pause (``drive_hold``) or the E-STOP latch release the walk once."""
        cmd = None if (estop or self.drive_hold is not None) else self._drive_source(now)
        if cmd is not None:
            self._relaying = True
            return cmd
        if self._relaying:
            self._relaying = False
            return web_scene.RELEASE
        return None

    @property
    def relaying(self) -> bool:
        return self._relaying

    def _drive_source(self, now: float):
        r = self.get("drive_recovery", now, DRIVE_STALE_S)
        if r is not None and (r.get("lock") or not r.get("release")):
            # recovery drives, or holds the robot locked (no other source may move it)
            if r.get("release") or r.get("stop"):
                return None
            return self._vec(r.get("vx"), r.get("vy"), r.get("yaw"))
        a = self.active
        p = self.get("tree360", now, 2.0)
        d = self.get("drive_tree360", now, DRIVE_STALE_S)   # the hook republishes every 0.1 s (sim time)
        t_d = self.stamps.get("drive_tree360")
        # only this tree's drive (never the last drive of the previous tree of a multi-tree item)
        if (d is not None and p is not None and a is not None and a["executor"] == "tree360"
                and t_d is not None and t_d >= a["t_phase"] and p.get("tree_id") == a.get("tree")
                and p.get("phase") not in TERMINAL_T360 and not p.get("paused") and not d.get("stand")):
            return self._vec(d.get("vx"), 0.0, d.get("yaw"))
        w = self.get("drive_work", now, DRIVE_STALE_S)
        if w is not None and not w.get("stop") and isinstance(w.get("cmd"), list) and len(w["cmd"]) == 3:
            return self._vec(*w["cmd"])
        return None

    @staticmethod
    def _vec(vx, vy, yaw):
        vals = [max(-1.0, min(1.0, _num(v) or 0.0)) for v in (vx, vy, yaw)]
        if web_scene.is_zero_cmd(*vals):
            return None
        return vals[0], vals[1], vals[2], True

    # ------------------------------------------------------------ server messages
    def preset_status(self, now: float, ctx: rc.ReadyContext) -> Dict:
        tiles = self.tiles(now, ctx)
        q = self.queue.status(ctx.soc, ctx.disk_free_mb, ctx.robot_xy, ctx.home_xy or (0.0, 0.0))
        for it in q.get("items") or []:
            it.setdefault("blocked", [])
            src = next((x for x in self.queue.items if x.qid == it.get("qid")), None)
            if src is not None:
                it["tree_ids"] = list(src.cmd["plan"].get("tree_ids") or [])
                it["executor"] = (src.cmd.get("console") or {}).get("executor")
        a = self.active
        active = None
        if a is not None:
            active = {k: a.get(k) for k in ("qid", "preset", "mode", "executor", "tree_ids", "tree", "state", "paused",
                                            "aborting", "mission_id", "step")}
            active["elapsed_s"] = round(now - a["t_start"], 1)
            active["live"] = self.executor_live(a["executor"], now)
        chips = []
        for k, lbl in (("rtk", ctx.rtk), ("local_link", ctx.local_link), ("cloud_link", ctx.cloud_link)):
            chips.append({"id": k, "value": lbl, "level": "ok" if lbl in ("FIXED", "up") else (
                "warn" if lbl in ("FLOAT", "unknown") else "fail")})
        chips.append({"id": "soc", "value": round(ctx.soc * 100), "level": "ok" if ctx.soc > 0.4 else (
            "warn" if ctx.soc > 0.2 else "fail")})
        chips.append({"id": "wind", "value": round(ctx.wind_mps, 1), "level": "ok" if ctx.wind_mps < 8 else "warn"})
        chips.append({"id": "slope", "value": round(ctx.cross_slope_deg, 1),
                      "level": "ok" if ctx.cross_slope_deg <= 3 else ("warn" if ctx.cross_slope_deg <= 5 else "fail")})
        if self.sim_local_link:
            chips.append({"id": "sim_handheld", "value": "sim", "level": "warn"})
        if self.blocked:
            chips.append({"id": "blocked", "value": self.blocked[:80], "level": "fail"})
        if self.context_fault:
            chips.append({"id": "sensor_fault", "value": self.context_fault[:80], "level": "fail"})
        return {"type": "preset_status", "tiles": tiles, "queue": q, "active": active, "history": list(self.history),
                "paused": self.paused, "paused_reason": self.paused_reason, "relay_hold": self.drive_hold,
                "chips": chips,
                "ready": {"rtk": ctx.rtk, "soc": round(ctx.soc, 3), "local_link": ctx.local_link,
                          "cloud_link": ctx.cloud_link, "wind_mps": round(ctx.wind_mps, 2),
                          "cross_slope_deg": round(ctx.cross_slope_deg, 2), "estop": ctx.estop,
                          "season": (self.field or {}).get("season"), "month": ctx.month_now(),
                          "disk_free_mb": round(ctx.disk_free_mb)},
                "assist": self.assist, "world": self.world, "marks_n": len(self.marks),
                "records": sorted(self.records)}

    def coverage(self, now: float) -> Dict:
        p = self.get("tree360", now, 5.0)
        live = p is not None and not p.get("error")
        p = p if live else {}
        keep = ("tree_id", "phase", "status", "mode", "crown_walk", "station", "stations", "R_m", "standoff_m",
                "gap_deg", "paused", "pauses", "operator_paused", "aborting", "frames", "masked_pct", "r_err_m",
                "reentries", "abort_reason", "orbit_s")
        out = {"type": "coverage", "live": live, **{k: p.get(k) for k in keep}}
        cov = p.get("coverage_pct") if isinstance(p.get("coverage_pct"), dict) else {}
        out["coverage_pct"] = {b: _num(cov.get(b)) for b in ("low", "mid", "high")}
        for k in ("station", "stations"):
            if not isinstance(out[k], int) or isinstance(out[k], bool):
                out[k] = None
        return out

    def mast(self, now: float) -> Dict:
        m = self.get("mast", now) or {}
        st_ = self.get("stability", now) or {}
        w = self.get("wind", now) or {}
        ext = _num(m.get("ext_m"))
        band = "n/a" if ext is None else ("stow" if ext < 0.02 else ("mid" if ext < 1.0 else "high"))
        gates = st_.get("gates") if isinstance(st_.get("gates"), dict) else {}
        return {"type": "mast", "live": bool(m) and bool(m.get("has_mast", True)), "ext_m": ext,
                "target_ext": _num(m.get("target_ext")), "lens_z": _num(m.get("lens_z")), "band": band,
                "max_ext_m": 1.6, "gated": bool(m.get("gated")), "reason": m.get("reason") or "",
                "detail": str(m.get("detail") or "")[:80], "arbiter_source": m.get("arbiter_source"),
                "arbiter_reason": m.get("arbiter_reason"), "braked": bool(m.get("braked")),
                "moving": bool(m.get("moving")), "held_by": m.get("held_by") or "",
                "clearance_m": _num(m.get("clearance_m")), "strike_count": m.get("strike_count") or 0,
                "wind": {"speed_mps": _num(w.get("speed_mps")), "mean_mps": _num(w.get("mean_mps")),
                         "gust": bool(w.get("gust")), "limits": w.get("limits") or {}},
                "gates": {"walk": gates.get("walk"), "slope": gates.get("slope"), "raise": gates.get("raise_hexa_ok"),
                          "margin_m": _num(st_.get("margin_m")), "wind_state": st_.get("wind_state")}}

    def recovery(self, now: float) -> Dict:
        rs = self.get("recovery", now) or {}
        beh = rs.get("behaviour") if isinstance(rs.get("behaviour"), dict) else {}
        links = {}
        for k in ("local", "cloud"):
            l_ = self.get(f"link_{k}", now) or {}
            links[k] = {"state": l_.get("state") or "unknown", "age_s": _num(l_.get("age_s")),
                        "needs_confirm": bool(l_.get("needs_confirm")), "action": l_.get("action"),
                        "sim": k == "local" and self.sim_local_link}
        return {"type": "recovery", "live": bool(rs), "armed": bool(rs.get("armed")), "active": rs.get("active"),
                "mode": rs.get("mode"), "step": beh.get("step"), "outcome": beh.get("outcome"),
                "reason": beh.get("reason"), "prompt": rs.get("prompt") if isinstance(rs.get("prompt"), str) else None,
                "checklist": beh.get("checklist") if isinstance(beh.get("checklist"), dict) else {},
                "confirms": beh.get("confirms") if isinstance(beh.get("confirms"), dict) else {},
                "lock": bool(rs.get("lock")), "queue_pause": bool(rs.get("queue_pause")), "links": links,
                "fallen_checks": list(FALLEN_CHECKS), "pending": self._recovery_pending is not None}

    def streams(self, now: float, ctx: rc.ReadyContext) -> List[Dict]:
        """MP 0.7 server streams due now (<= 2 Hz each, only on change or as a 5 s keep-alive)."""
        out = []
        for name in ("preset_status", "coverage", "mast", "recovery"):
            msg = getattr(self, name)(now, ctx) if name == "preset_status" else getattr(self, name)(now)
            if self.limiter.offer(name, web_scene.encode_message(msg), now):
                out.append(msg)
        return out

    def snapshot(self, now: float, ctx: rc.ReadyContext) -> List[Dict]:
        """Everything a newly connected browser needs at once (map, streams, records, marks)."""
        out = [self.map, self.preset_status(now, ctx), self.coverage(now), self.mast(now), self.recovery(now)]
        out += [{"type": "tree_record", "tree_id": k, "record": v} for k, v in self.records.items()]
        out += [{"type": "mark", "mark": m} for m in self.marks[-200:]]
        return out

    def take_outbox(self) -> List[Tuple[str, Dict]]:
        out, self.outbox = self.outbox, []
        return out

    def take_events(self) -> List[Dict]:
        out, self.events = self.events, []
        return out
