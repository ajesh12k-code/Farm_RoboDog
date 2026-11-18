"""World events and the foot-on-lateral counter (WP4; ORCHARD-MVP-SPEC 1.2, 3.6, 6). Pure Python, ROS-free.

``EventScheduler`` fires the scripted events of a field (``config/fields/<world>.json`` ``events[]``) and starts
the actor scripts (``actors[].start``). An event is::

    {"id": str, "type": wind | rain | weather | light | fog | actor | marker,
     "trigger": {"mode": at_s | robot_within | cmd | always, "t": s, "xy": [x, y], "r": m,
                 "after_s": delay after the condition, "repeat_s": period (0 = once)},
     ...type fields: wind {script: WindScript}, rain {dur_s, mm_h}, weather {cmd}, light {preset}, fog {density},
                     actor {actor, action: start | stop | park | home}, marker {note}}

``wind`` and ``rain`` are handed to the WP7 platform hook (wind drag ½ρv²CdA and the weather stub are WP7's) as
in-process commands on ``/nandi/wind/cmd`` / ``/nandi/weather/cmd``; WP4 schedules, WP7 applies. ``light`` and
``fog`` set the lighting preset (``world.lighting``). ``cmd`` events fire only on ``/nandi/world/cmd``
``{"fire": id}``.

``LateralCounter`` counts **foot placements on drip laterals** (OS 3.6): a foot whose contact point (foot centre
projected to the ground, foot in contact with the ground) lies within ``near_m`` = 2 cm of a lateral's axis.
One count per placement (rising edge), released when the foot lifts or moves beyond ``release_m``. Laterals are
non-colliding (query + visual), so a foot passes through and the count is the only effect: no footstep planner
avoids them (OS 7.1, risk 10).
"""
import json
import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EVENT_TYPES = ("wind", "rain", "weather", "light", "fog", "actor", "marker")
TRIGGER_MODES = ("at_s", "robot_within", "cmd", "always", "never")
WIND_CMD_TOPIC = "/nandi/wind/cmd"        # WP7 platform hook (drag owner)
WEATHER_CMD_TOPIC = "/nandi/weather/cmd"  # WP7 weather stub
LATERAL_NEAR_M = 0.02                     # OS 3.6: contact point within 2 cm of a lateral
LATERAL_RELEASE_M = 0.03
FOOT_CONTACT_TOL_M = 0.01                 # foot bottom within 1 cm of the ground surface = in contact


@dataclass
class Trigger:
    mode: str = "cmd"
    t: float = 0.0
    xy: Tuple[float, float] = (0.0, 0.0)
    r: float = 0.0
    after_s: float = 0.0
    repeat_s: float = 0.0

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Trigger":
        d = dict(d or {})
        mode = d.get("mode", "cmd")
        if mode not in TRIGGER_MODES:
            raise ValueError(f"trigger mode {mode!r} not in {TRIGGER_MODES}")
        xy = d.get("xy", (0.0, 0.0))
        return cls(mode, float(d.get("t", 0.0)), (float(xy[0]), float(xy[1])), float(d.get("r", 0.0)),
                   float(d.get("after_s", 0.0)), float(d.get("repeat_s", 0.0)))


@dataclass
class Event:
    id: str
    type: str
    trigger: Trigger
    data: dict = dc_field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        if d.get("type") not in EVENT_TYPES:
            raise ValueError(f"event {d.get('id')!r}: type {d.get('type')!r} not in {EVENT_TYPES}")
        data = {k: v for k, v in d.items() if k not in ("id", "type", "trigger")}
        return cls(str(d["id"]), d["type"], Trigger.from_dict(d.get("trigger")), data)


@dataclass
class _Arm:
    """Runtime state of one trigger."""
    armed_at: Optional[float] = None     # condition first met (robot_within) / fire time pending
    fired: int = 0
    last_fire: Optional[float] = None


def actor_start_events(actor_entries: Iterable[dict]) -> List[Event]:
    """One synthetic ``actor`` event per actor whose ``start`` is not ``cmd`` (ids ``actor:<id>:start``)."""
    out = []
    for a in actor_entries:
        st = dict(a.get("start") or {"mode": "cmd"})
        if st.get("mode", "cmd") in ("cmd", "never"):
            continue
        out.append(Event(f"actor:{a['id']}:start", "actor", Trigger.from_dict(st),
                         {"actor": a["id"], "action": "start"}))
    return out


class EventScheduler:
    def __init__(self, events: Sequence[dict] = (), actors: Sequence[dict] = ()):
        self.events: List[Event] = [Event.from_dict(e) for e in events] + actor_start_events(actors)
        ids = [e.id for e in self.events]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate event ids: {ids}")
        self.by_id = {e.id: e for e in self.events}
        self.reset()

    def reset(self):
        self.arms: Dict[str, _Arm] = {e.id: _Arm() for e in self.events}
        self.pending: List[Tuple[float, Event]] = []    # (due time, event) from cmd fires
        self.log: List[dict] = []

    def fire(self, event_id: str, t: float):
        """Queue an event now (``/nandi/world/cmd`` ``{"fire": id}``)."""
        if event_id not in self.by_id:
            raise KeyError(f"no event {event_id!r}")
        ev = self.by_id[event_id]
        self.pending.append((float(t) + ev.trigger.after_s, ev))

    def _due(self, ev: Event, arm: _Arm, t: float, robot_xy: Optional[Tuple[float, float]]) -> bool:
        tr = ev.trigger
        if tr.mode in ("cmd", "never"):
            return False
        if arm.fired and tr.repeat_s <= 0:
            return False
        if arm.fired and tr.repeat_s > 0:
            return t + 1e-9 >= arm.last_fire + tr.repeat_s and self._cond(tr, t, robot_xy)
        if tr.mode == "always":
            return True
        if tr.mode == "at_s":
            return t + 1e-9 >= tr.t
        # robot_within: condition, then after_s delay
        if arm.armed_at is None:
            if self._cond(tr, t, robot_xy):
                arm.armed_at = t
            else:
                return False
        return t + 1e-9 >= arm.armed_at + tr.after_s

    @staticmethod
    def _cond(tr: Trigger, t: float, robot_xy) -> bool:
        if tr.mode == "robot_within":
            return robot_xy is not None and math.dist(robot_xy, tr.xy) <= tr.r
        if tr.mode == "at_s":
            return t + 1e-9 >= tr.t
        return True

    def update(self, t: float, robot_xy: Optional[Tuple[float, float]] = None) -> List[Event]:
        """Events that fire at sim time t, in declaration order (queued cmd fires first)."""
        out: List[Event] = []
        keep = []
        for due, ev in sorted(self.pending, key=lambda p: p[0]):
            if t + 1e-9 >= due:
                out.append(ev)
            else:
                keep.append((due, ev))
        self.pending = keep
        for ev in self.events:
            arm = self.arms[ev.id]
            if self._due(ev, arm, float(t), robot_xy):
                out.append(ev)
        for ev in out:
            arm = self.arms[ev.id]
            arm.fired += 1
            arm.last_fire = float(t)
            self.log.append({"t": float(t), "id": ev.id, "type": ev.type})
        return out

    def fired_ids(self) -> List[str]:
        return [e["id"] for e in self.log]


# ---------------------------------------------------------------------- command parsing
ACTOR_ACTIONS = ("start", "stop", "park", "home")
CMD_KEYS = ("fire", "actor", "action", "light", "fog", "reset_counters", "seq")


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_command(raw: Any) -> Tuple[Optional[dict], str]:
    """A ``/nandi/world/cmd`` payload (JSON text or dict) → (command, "ok") or (None, reason).

    Keys and value types are checked so a malformed message is refused, never raised: ``fire`` / ``actor`` /
    ``light`` non-empty strings, ``action`` one of ``ACTOR_ACTIONS`` (only with ``actor``), ``fog`` a finite
    number clamped to [0, 1], ``reset_counters`` a JSON boolean, ``seq`` an integer or string. The returned
    dict is a new object (``fog`` clamped)."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None, "not JSON"
    if not isinstance(raw, dict):
        return None, "not a JSON object"
    keys = set(raw)
    if not keys <= set(CMD_KEYS):
        return None, f"unknown keys {sorted(str(k) for k in keys - set(CMD_KEYS))}"
    if not keys - {"seq"}:
        return None, "no command"
    out = dict(raw)
    for k in ("fire", "actor", "light"):
        if k in out and not (isinstance(out[k], str) and out[k]):
            return None, f"{k} must be a non-empty string"
    if "action" in out:
        if "actor" not in out:
            return None, "action without actor"
        if out["action"] not in ACTOR_ACTIONS:
            return None, f"action must be one of {list(ACTOR_ACTIONS)}"
    if "fog" in out:
        f = out["fog"]
        if not _is_num(f) or not math.isfinite(f):
            return None, "fog must be a finite number"
        out["fog"] = min(1.0, max(0.0, float(f)))
    if "reset_counters" in out and not isinstance(out["reset_counters"], bool):
        return None, "reset_counters must be true or false"
    if "seq" in out and not (isinstance(out["seq"], str) or (isinstance(out["seq"], int)
                                                              and not isinstance(out["seq"], bool))):
        return None, "seq must be an integer or string"
    return out, "ok"


def parse_command(raw: Any) -> Optional[dict]:
    """``check_command`` without the reason: the command dict, or None if unusable."""
    return check_command(raw)[0]


# ---------------------------------------------------------------------- laterals
def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Planar distances from points p (N, 2) to segments (a, b) (M, 2) → (N, M)."""
    p = np.asarray(p, dtype=np.float64).reshape(-1, 2)[:, None, :]
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)[None, :, :]
    b = np.asarray(b, dtype=np.float64).reshape(-1, 2)[None, :, :]
    d = b - a
    L2 = np.maximum(np.sum(d * d, axis=-1), 1e-12)
    t = np.clip(np.sum((p - a) * d, axis=-1) / L2, 0.0, 1.0)
    q = a + t[..., None] * d
    return np.linalg.norm(p - q, axis=-1)


class LateralCounter:
    """Rising-edge count of foot placements within ``near_m`` of a lateral axis."""

    def __init__(self, segments: Sequence[Tuple[str, Sequence[float], Sequence[float]]],
                 near_m: float = LATERAL_NEAR_M, release_m: float = LATERAL_RELEASE_M):
        self.ids = [s[0] for s in segments]
        self.a = np.array([s[1] for s in segments], dtype=np.float64).reshape(-1, 2)
        self.b = np.array([s[2] for s in segments], dtype=np.float64).reshape(-1, 2)
        self.near_m, self.release_m = float(near_m), float(release_m)
        self.reset()

    def reset(self):
        self.count = 0
        self.per_lateral: Dict[str, int] = {}
        self.per_foot: Dict[str, int] = {}
        self.on: Dict[str, Optional[str]] = {}
        self.events: List[dict] = []

    def nearest(self, xy: Sequence[float]) -> Tuple[Optional[str], float]:
        if not len(self.ids):
            return None, math.inf
        d = point_segment_distance(np.asarray(xy, dtype=np.float64)[None, :], self.a, self.b)[0]
        k = int(np.argmin(d))
        return self.ids[k], float(d[k])

    def update(self, t: float, feet: Iterable[Tuple[str, float, float, bool]]) -> List[dict]:
        """feet: (foot name, contact x, contact y, in_contact). Returns the new placement events."""
        new = []
        for name, x, y, contact in feet:
            lid, dist = self.nearest((x, y))
            cur = self.on.get(name)
            if cur is not None:
                if not contact or dist > self.release_m:
                    self.on[name] = None
                continue
            if contact and dist <= self.near_m:
                self.on[name] = lid
                self.count += 1
                self.per_lateral[lid] = self.per_lateral.get(lid, 0) + 1
                self.per_foot[name] = self.per_foot.get(name, 0) + 1
                ev = {"t": float(t), "type": "lateral_step", "foot": name, "lateral": lid,
                      "xy": [round(float(x), 4), round(float(y), 4)], "dist_m": round(dist, 4), "count": self.count}
                self.events.append(ev)
                new.append(ev)
        return new

    def status(self) -> dict:
        return {"count": self.count, "per_lateral": dict(self.per_lateral), "per_foot": dict(self.per_foot),
                "near_m": self.near_m}


# ---------------------------------------------------------------------- field validation
def _finite(v: Any) -> bool:
    return _is_num(v) and math.isfinite(v)


def validate_events(events: Any, actor_ids: Iterable[str] = (),
                    presets: Optional[Iterable[str]] = None) -> List[str]:
    """Schema errors of a field's ``events[]`` (empty = valid). ``field_spec.validate`` calls it, so a typo in
    an authored event (unknown light preset, missing ``preset`` / ``actor``, non-numeric fog density, unknown
    trigger mode) fails at load instead of raising inside ``WorldRuntime.apply_event`` during a tick (a hook
    fault safe-holds the robot). ``presets`` defaults to ``world.lighting.PRESETS``."""
    if events is None:
        return []
    if not isinstance(events, list):
        return ["events must be a list"]
    if presets is None:
        from atharv_nandi.sim.world import lighting
        presets = lighting.PRESETS
    presets, actor_ids = set(presets), set(actor_ids)
    errs, seen = [], set()
    for i, e in enumerate(events):
        if not isinstance(e, dict):
            errs.append(f"event #{i}: not an object")
            continue
        eid = e.get("id")
        tag = f"event {eid!r}"
        if not (isinstance(eid, str) and eid):
            errs.append(f"event #{i}: id must be a non-empty string")
        elif eid in seen:
            errs.append(f"duplicate event id {eid!r}")
        seen.add(eid)
        typ = e.get("type")
        if typ not in EVENT_TYPES:
            errs.append(f"{tag}: type {typ!r} not in {EVENT_TYPES}")
            continue
        tr = e.get("trigger")
        if tr is not None and not isinstance(tr, dict):
            errs.append(f"{tag}: trigger must be an object")
        else:
            tr = tr or {}
            try:
                Trigger.from_dict(tr)
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                errs.append(f"{tag}: trigger {exc}")
            for k in ("t", "r", "after_s", "repeat_s"):
                if k in tr and not (_finite(tr[k]) and tr[k] >= 0):
                    errs.append(f"{tag}: trigger.{k} must be a finite number >= 0")
            if "xy" in tr and not (isinstance(tr["xy"], (list, tuple)) and len(tr["xy"]) == 2
                                   and all(_finite(v) for v in tr["xy"])):
                errs.append(f"{tag}: trigger.xy must be [x, y]")
        if typ == "light" and e.get("preset") not in presets:
            errs.append(f"{tag}: light preset {e.get('preset')!r} not in {sorted(presets)}")
        elif typ == "fog" and "density" in e and not _finite(e["density"]):
            errs.append(f"{tag}: fog density must be a finite number")
        elif typ == "rain":
            for k in ("dur_s", "mm_h"):
                if k in e and not (_finite(e[k]) and e[k] >= 0):
                    errs.append(f"{tag}: rain {k} must be a finite number >= 0")
        elif typ == "wind" and not isinstance(e.get("script", {}), dict):
            errs.append(f"{tag}: wind script must be an object")
        elif typ == "weather" and not isinstance(e.get("cmd", {}), dict):
            errs.append(f"{tag}: weather cmd must be an object")
        elif typ == "actor":
            if e.get("actor") not in actor_ids:
                errs.append(f"{tag}: actor {e.get('actor')!r} not in actors[] {sorted(actor_ids)}")
            if e.get("action", "start") not in ACTOR_ACTIONS:
                errs.append(f"{tag}: action {e.get('action')!r} not in {ACTOR_ACTIONS}")
    return errs
