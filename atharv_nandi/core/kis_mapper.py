"""Pure mapper: Atharv Nandi /nandi/* status -> Kisaanu ``kis/v1`` MQTT topics and payloads.

No ROS, no MQTT, no I/O. telemetry_bridge.py feeds it parsed status dicts and hands the returned
(topic, payload) pairs to a sink (JSONL file by default, MQTT only when enabled in config).

Rules followed (the Kisaanu kis/v1 broker topic contract, read 2026-11-17; cited, not copied):
    topic tree  kis/v1/{tenant}/{site}/{device}/tele/{stream} | /state (retained) | /status (retained, will)
    slugs       tenant/site [a-z0-9][a-z0-9-]{0,39}, device [a-z0-9][a-z0-9-]{0,63},
                stream [a-z0-9][a-z0-9_-]{0,31}
    envelope    every tele/state payload carries v, ts (epoch ms UTC), ts_src, boot, seq
                (seq per stream, starts at 0 on every boot, +1 per message)
    provenance  "data_provenance": "simulated" at the top level of every JSON payload (section 6.2)
                (the envelope is applied last: a body or state extra can never overwrite these keys)
    sim.t       tele/pose carries sim.t = the simulated instant in epoch ms (section 7 additive field)
    json        strict: NaN/Infinity are refused (NonFinite), never written
    sizes       tele/* <= 16 KiB, state <= 8 KiB; status is the plain string online/offline
    simulation  tenant must be sim-* ; the mapper refuses any other (as services/sim-devices does)

Deviations, stated rather than hidden:
    ts_src is "sim" (the sim's wall clock, not NTP-disciplined). broker contract section 2 only defines
    "ntp" and "boot", so a real ingest would treat "sim" as unknown; that is deliberate for now:
    a simulated record must never pass as an NTP-timed measurement.

Stream status (STREAMS below). broker contract section 3 says a tele/{other} stream "must be named here
before it is used". None of the robot streams is named there yet, so all of them are PROPOSED:
    pose, batt, mission, detect   PROPOSED - from the robot stream design, pending a
                                  broker contract section 3 change
    harvest                       PROPOSED - this repo only; not part of any Kisaanu contract
Only the topic tree, the envelope, state and status are the contract today.
"""
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = "kis/v1"
PAYLOAD_VERSION = 1
TS_SRC = "sim"
DATA_PROVENANCE = "simulated"

TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
SITE_RE = TENANT_RE
DEVICE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
STREAM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
SIM_TENANT_RE = re.compile(r"^sim-")

MAX_TELE_BYTES = 16 * 1024
MAX_STATE_BYTES = 8 * 1024

# stream -> (status, origin). Everything here is PROPOSED until broker contract section 3 names it.
STREAMS: Dict[str, Tuple[str, str]] = {
    "pose": ("PROPOSED", "robot-stream-design"),
    "batt": ("PROPOSED", "robot-stream-design"),
    "mission": ("PROPOSED", "robot-stream-design"),
    "detect": ("PROPOSED", "robot-stream-design"),
    "harvest": ("PROPOSED", "atharv-nandi only"),  # PROPOSED: per-attempt harvest outcome
}

EARTH_RADIUS_M = 6378137.0  # WGS-84 equatorial radius


class TenantRefused(ValueError):
    """The tenant is not a simulation tenant (sim-*): a sim must never publish as a real tenant."""


class PayloadTooLarge(ValueError):
    """An encoded payload exceeds the broker contract section 5 limit for its kind."""


class BadSlug(ValueError):
    """A topic level does not match its broker contract section 1 shape."""


def check_slug(value: str, pattern: "re.Pattern", what: str) -> str:
    if not isinstance(value, str) or not pattern.match(value):
        raise BadSlug(f"{what} {value!r} does not match {pattern.pattern}")
    return value


def enu_to_latlon(east_m: float, north_m: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    """Small-offset local-tangent-plane conversion (equirectangular; < 1 mm error over a field)."""
    lat = origin_lat + math.degrees(north_m / EARTH_RADIUS_M)
    coslat = max(1e-9, math.cos(math.radians(origin_lat)))
    lon = origin_lon + math.degrees(east_m / (EARTH_RADIUS_M * coslat))
    return lat, lon


def yaw_from_quat(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _r(v: float, nd: int) -> float:
    return round(float(v), nd)


def encode(payload: Any) -> bytes:
    """Compact UTF-8 JSON (status strings are sent as-is). Strict JSON: NaN/Infinity raise ValueError."""
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False).encode("utf-8")


class NonFinite(ValueError):
    """A payload would carry NaN/Infinity (not valid JSON), e.g. qpos after a MuJoCo divergence."""


class KisMapper:
    """Builds kis/v1 (topic, payload) pairs for one simulated device and one boot."""

    def __init__(self, tenant: str, site: str, device: str, boot: int,
                 origin: Optional[Dict[str, float]] = None, sw: str = "atharv-nandi unknown"):
        check_slug(tenant, TENANT_RE, "tenant")
        if not SIM_TENANT_RE.match(tenant):
            raise TenantRefused(f"tenant {tenant!r} refused: the simulator only publishes under sim-* tenants")
        self.tenant = tenant
        self.site = check_slug(site, SITE_RE, "site")
        self.device = check_slug(device, DEVICE_RE, "device")
        origin = origin or {}
        self.origin_lat = float(origin.get("lat", 0.0))
        self.origin_lon = float(origin.get("lon", 0.0))
        self.origin_alt = float(origin.get("alt", 0.0))
        self.sw = sw
        self.boot = int(boot)
        self._seq: Dict[str, int] = {}

    # ------------------------------------------------------------------ topics + envelope
    @property
    def base(self) -> str:
        return f"{ROOT}/{self.tenant}/{self.site}/{self.device}"

    def tele_topic(self, stream: str) -> str:
        check_slug(stream, STREAM_RE, "stream")
        return f"{self.base}/tele/{stream}"

    def state_topic(self) -> str:
        return f"{self.base}/state"

    def status_topic(self) -> str:
        return f"{self.base}/status"

    def new_boot(self, boot: int):
        """A new boot: new boot id, every stream's seq restarts at 0."""
        self.boot = int(boot)
        self._seq.clear()

    def peek_seq(self, stream: str) -> int:
        """The seq the next message on `stream` will carry (not consumed)."""
        return self._seq.get(stream, -1) + 1

    def next_seq(self, stream: str) -> int:
        seq = self.peek_seq(stream)
        self._seq[stream] = seq
        return seq

    def envelope(self, stream: str, ts_ms: int, seq: Optional[int] = None) -> Dict[str, Any]:
        return {"v": PAYLOAD_VERSION, "ts": int(ts_ms), "ts_src": TS_SRC, "boot": self.boot,
                "seq": self.peek_seq(stream) if seq is None else int(seq), "data_provenance": DATA_PROVENANCE}

    def _build(self, stream: str, ts_ms: int, body: Dict[str, Any], fixed: Optional[Dict[str, Any]],
               limit: int, topic: str) -> Tuple[str, Dict[str, Any]]:
        """Body first, then the fixed keys and the envelope on top: a body can never overwrite v, ts,
        ts_src, boot, seq or data_provenance. The seq is consumed only once the payload is accepted
        (strict JSON, within the size limit), so a refused payload leaves no gap in the stream."""
        payload = dict(body)
        if fixed:
            payload.update(fixed)
        payload.update(self.envelope(stream, ts_ms))
        try:
            check_size(payload, limit, topic)
        except ValueError as exc:
            if isinstance(exc, PayloadTooLarge):
                raise
            raise NonFinite(f"{topic}: {exc}") from exc
        self.next_seq(stream)
        return topic, payload

    def _tele(self, stream: str, ts_ms: int, body: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        return self._build(stream, ts_ms, body, None, MAX_TELE_BYTES, self.tele_topic(stream))

    # ------------------------------------------------------------------ tele streams
    def pose(self, ts_ms: int, qpos: Sequence[float], sim_t_ms: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """PROPOSED tele/pose from the trunk free joint (world x = east, y = north, z = up).

        sim_t_ms: the simulated instant in epoch ms (broker contract section 7 additive field `sim.t`),
        sent beside the real `ts`. Raises NonFinite for a NaN/Inf trunk pose (diverged physics)."""
        if len(qpos) < 7:
            raise ValueError("qpos needs the 7-value trunk free joint")
        if not all(math.isfinite(float(v)) for v in qpos[0:7]):
            raise NonFinite("tele/pose: non-finite trunk qpos (physics diverged?)")
        e, n, u = (float(v) for v in qpos[0:3])
        lat, lon = enu_to_latlon(e, n, self.origin_lat, self.origin_lon)
        yaw = yaw_from_quat(*(float(v) for v in qpos[3:7]))
        body = {
            "frame": "enu",
            "enu": {"e": _r(e, 4), "n": _r(n, 4), "u": _r(u, 4)},
            "lat": _r(lat, 9), "lon": _r(lon, 9), "alt": _r(self.origin_alt + u, 3),
            # ENU heading convention: 0 = east, CCW positive (degrees)
            "yaw_deg": _r(math.degrees(yaw), 2),
            "quat_wxyz": [_r(v, 5) for v in qpos[3:7]],
        }
        if sim_t_ms is not None and math.isfinite(float(sim_t_ms)):
            body["sim"] = {"t": int(sim_t_ms)}
        return self._tele("pose", ts_ms, body)

    def batt(self, ts_ms: int, battery: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """PROPOSED tele/batt from /nandi/battery (fields passed through, numbers only)."""
        keys = ("soc", "voltage_v", "current_a", "power_w", "energy_wh", "capacity_wh", "temperature_c")
        body = {k: battery[k] for k in keys if _is_num(battery.get(k))}
        return self._tele("batt", ts_ms, body)

    def mission(self, ts_ms: int, mission: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """PROPOSED tele/mission: mission/arm state and score (see mission_summary)."""
        return self._tele("mission", ts_ms, dict(mission))

    def detect(self, ts_ms: int, detections: Iterable[Dict[str, Any]], model: Optional[str] = None
               ) -> Tuple[str, Dict[str, Any]]:
        """PROPOSED tele/detect from /nandi/crop_detections. Raises PayloadTooLarge past 16 KiB."""
        body: Dict[str, Any] = {"detections": [dict(d) for d in detections]}
        if model:
            body["model"] = model
        return self._tele("detect", ts_ms, body)

    def harvest(self, ts_ms: int, attempt: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """PROPOSED tele/harvest: one completed auto_harvest attempt (not part of the kis/v1 contract)."""
        return self._tele("harvest", ts_ms, {"attempt": dict(attempt)})

    # ------------------------------------------------------------------ state + status
    def state(self, ts_ms: int, mode: str, extra: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """Retained state (contract): what the device IS. seq counts on its own 'state' key."""
        fixed = {"fw": self.sw, "model": "kisaanu/atharv-nandi-sim@1", "mode": mode}
        return self._build("state", ts_ms, dict(extra or {}), fixed, MAX_STATE_BYTES, self.state_topic())

    def status(self, online: bool) -> Tuple[str, str]:
        """Retained status (contract): the plain strings online / offline, not JSON."""
        return self.status_topic(), "online" if online else "offline"

    def will(self) -> Dict[str, Any]:
        """Last-will settings to pass to the MQTT client on CONNECT (broker contract section 4)."""
        return {"topic": self.status_topic(), "payload": "offline", "qos": 1, "retain": True,
                "will_delay_s": 10}


def check_size(payload: Any, limit: int, topic: str = "") -> int:
    n = len(encode(payload))
    if n > limit:
        raise PayloadTooLarge(f"{topic or 'payload'}: {n} bytes > {limit} byte limit")
    return n


def _is_num(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def retain_for(topic: str) -> bool:
    """Retained: state and status only (broker contract section 5)."""
    return topic.endswith("/state") or topic.endswith("/status")


def mission_summary(harvest: Dict[str, Any], game: Dict[str, Any],
                    mission_status: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None
                    ) -> Dict[str, Any]:
    """tele/mission body from parsed harvest/game status (atharv_nandi/core/telemetry.py parsers)."""
    arm_mode = harvest.get("mode", "rl_balance")
    out: Dict[str, Any] = {
        "state": "harvesting" if arm_mode == "ik_harvest" else ("manual" if arm_mode == "manual" else "idle"),
        "arm_mode": arm_mode,
        "phase": harvest.get("phase", "idle"),
        "target": harvest.get("target"),
        "last_result": (harvest.get("last_result") or "")[:200],
        "picked": int(game.get("in_basket", 0)),
        "crops_total": int(game.get("crops_total", 0)),
        "falls": int(game.get("falls", 0)),
        "fallen": bool(game.get("fallen", False)),
    }
    if run_id:
        out["run_id"] = run_id
    if mission_status:
        # /nandi/mission_status (mission package) wins for the mission state when present
        if isinstance(mission_status.get("state"), str):
            out["state"] = mission_status["state"][:32]
        for k in ("mission_id", "plan", "waypoint", "progress"):
            if k in mission_status:
                out[k] = mission_status[k]
    return out


def lines_for(records: List[Tuple[str, Any]]) -> List[str]:
    """JSONL lines for the file sink: {"topic", "retain", "payload"} per message."""
    return [json.dumps({"topic": t, "retain": retain_for(t), "payload": p}, separators=(",", ":"))
            for t, p in records]
