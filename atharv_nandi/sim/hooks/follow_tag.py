"""WP29 hook ``follow_tag`` (20 Hz): simulated UWB tag + the WORK assist controller (MISSION-PRESETS 7; OS 5.6).

Per tick:
    1. ``/nandi/work/tag/cmd`` (every message, in order):
       ``{"op": "issue", "actor": id, "adult": true, "shift": "s1", "type"?: "worker"}`` issues the tag to a person
       actor (adult workers only: a ``child`` actor, a missing ``adult`` attestation, a body that is not a person
       (``actor_<cls>_`` with cls other than ``person``: an animal or a prop, whatever type is declared) or a body of
       unknown type (not a WP4 field actor and no declared ``"type"``) is refused ``tag_not_adult``, OQ-17 A; the WP4
       type is looked up by the actor id whether it is given as ``w1`` or ``actor_person_w1``);
       ``{"op": "revoke"}``; ``{"op": "rotate", "shift": "s2"}`` (new pseudonymous id);
       ``{"op": "drop", "dur_s": 3.0}`` (no readings for that long: occlusion / flat tag battery);
       ``{"op": "walk", "waypoints": [[x, y], ...], "speed_mps": 0.15}`` scripts the tag holder's walk (field actors
       through the WP4 ``world_events`` actor runtime, other person bodies by this hook's own mocap writes).
    2. UWB tag model at 10 Hz (ASSUMED, a DW3000-class two-antenna anchor on the trunk): range and bearing of the
       holder's body from the trunk centre, sigma 0.05 m and 2 deg (seeded by NANDI_SEED), 0..20 m, no reading inside
       a ``drop`` window. ``assisted: truth_tag_pose`` (the reading is truth + noise, no RF multipath model).
    3. ``/nandi/work/cmd`` (every message, in order): ``{"op": "start", "mode": "tote_carry" | "lead_to_marks" |
       "follow_me" | "spot_spray", "params": {...}}``, ``{"op": "pause" | "resume" | "next" | "ack" | "abort" | "spray"}``.
       ``lead_to_marks`` takes ``params.marks`` (MarkRecords) or, when absent, the ``/nandi/mark`` records received.
    4. ``controllers.work_assist.WorkAssist.update(Obs)`` with: truth base pose (``assisted: truth_base_pose``), the
       safe hold, the WP6 field latch and field returns (``hooks.stability`` FieldMonitor, read-only), the crown
       extension (WP24 ``values["mast"]``), the local safety link (WP7 ``values["link"]``), SoC, the field geofence.
       The WP6 field returns must be present and fresh (``FieldMonitor.last`` <= 0.25 s old, ``safety_fields`` not
       ``ok: false``): otherwise ``Obs.fields_ok`` is false and every mode refuses to start / stops
       (``fields_unavailable``): person detection fails closed.
    5. Out: ``bus.values["work_drive"]`` ``{t, stop, cmd: [vx, vy, yaw] (normalised as /nandi/cmd_vel), reason}`` (the
       SimDriver task ``work_assist.driver_task`` walks it), ``/nandi/work/drive`` (10 Hz while active: the live relay
       to ``/nandi/cmd_vel`` is a change request to WP27), ``/nandi/work/state`` (2 Hz + on change; also
       ``bus.values["work_assist"]``), ``/nandi/work/events``. None of them carries a coordinate or the tag id; the
       track log stays in ``WorkAssist.track`` (device memory, 7 days).

A malformed command is answered on ``/nandi/work/events`` (``bad_command``) and never faults the hook. Without an
orchard field (legacy model) the hook still runs (no geofence, no tree registry: lead and tree tote refuse).
"""
import math
import os
import random
from typing import Dict, List, Optional, Tuple

from atharv_nandi.core import work_assist as wa

CMD_TOPIC = "/nandi/work/cmd"
TAG_CMD_TOPIC = "/nandi/work/tag/cmd"
MARK_TOPIC = "/nandi/mark"
STATE_TOPIC = "/nandi/work/state"
DRIVE_TOPIC = "/nandi/work/drive"
EVENTS_TOPIC = "/nandi/work/events"
RATE_HZ = 20.0
TAG_HZ = 10.0
STATE_PERIOD_S = 0.5
DRIVE_PERIOD_S = 0.1
TAG_SIGMA_R_M = 0.05
TAG_SIGMA_B_RAD = math.radians(2.0)
TAG_MAX_RANGE_M = wa.TAG_RANGE_MAX_M
PERSON_PREFIX = "actor_person_"
SETTLE_SPEED_MPS = 0.01
SETTLE_S = 1.0
ASSISTED = ("truth_base_pose", "truth_tag_pose", "map_masked_lidar")
MAX_MARKS = 500


def _yaw(q) -> float:
    return math.atan2(2.0 * (q[3] * q[6] + q[4] * q[5]), 1.0 - 2.0 * (q[5] * q[5] + q[6] * q[6]))


class FollowTagRuntime:
    def __init__(self, m):
        import mujoco
        from atharv_nandi.sim.world import actors as wact
        self.m = m
        self.field = None
        self.actor_types: Dict[str, str] = {}
        try:
            from atharv_nandi.sim.world import compose
            from atharv_nandi.sim.world.modules import terrain as terrain_mod
            ident = terrain_mod.parse_world_text(m)
            if ident is not None and ident[0]:
                self.field = compose.load_field(ident[0], ident[1])
                for e in self.field.get("actors") or []:
                    self.actor_types[str(e["id"])] = str(e["type"])
        except (ImportError, OSError, ValueError, KeyError):
            self.field = None
        self.bodies = wact.actor_bodies(m)                   # body name -> mocap id
        self.body_ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.bodies}
        self.trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.geofence = [tuple(p) for p in (self.field or {}).get("geofence") or []] or None
        self.seed = int(os.environ.get("NANDI_SEED", "7") or 7)
        self.counts: Dict[str, int] = {}     # command topic -> arrivals handled (kept across world resets)
        self.reset()

    def reset(self):
        self.tags = wa.TagRegistry(salt=f"nandi-sim-{self.seed}")
        self.assist = wa.WorkAssist(self.field, self.tags)
        self.rng = random.Random(self.seed * 7919 + 29)
        self.reading: Optional[wa.TagReading] = None
        self.next_tag_t = 0.0
        self.drop_until = -1.0
        self.walk = None                 # own scripted walk for a non-field holder: (path, speed, t0)
        self.marks: List[dict] = []
        self.mark_count = 0
        self.ev_sent = 0
        self.next_state_t = 0.0
        self.next_drive_t = 0.0
        self.last_key = None
        self.last_xy: Optional[Tuple[float, float, float]] = None
        self.slow_since: Optional[float] = None
        self.settled_for = None
        self.local_events: List[dict] = []

    # ------------------------------------------------------------------ helpers
    def holder_body(self) -> Optional[str]:
        h = self.tags.holder
        if h is None:
            return None
        name = h if h.startswith(PERSON_PREFIX) else f"{PERSON_PREFIX}{h}"
        return name if name in self.bodies else None

    def pose(self, d) -> wa.Pose:
        return wa.Pose(float(d.qpos[0]), float(d.qpos[1]), _yaw(d.qpos))

    def _emit(self, t, kind, **kw):
        self.local_events.append({"t": round(float(t), 3), "event": kind, **kw})

    # ------------------------------------------------------------------ tag commands and model
    def tag_command(self, raw, t, d, bus):
        if not isinstance(raw, dict) or not isinstance(raw.get("op"), str):
            self._emit(t, "bad_command", topic="tag", why="expected {op: ...}")
            return
        op = raw["op"]
        if op == "issue":
            aid = raw.get("actor")
            if not isinstance(aid, str) or not aid:
                self._emit(t, "bad_command", topic="tag", why="actor id missing")
                return
            name = aid if aid.startswith("actor_") else f"{PERSON_PREFIX}{aid}"
            if name not in self.bodies:
                self._emit(t, "tag_refused", reason="unknown_target")
                return
            if not name.startswith(PERSON_PREFIX):
                # an animal or a prop body: never a tag holder, whatever type the operator declares
                self._emit(t, "tag_refused", reason="tag_not_adult")
                return
            short = name[len(PERSON_PREFIX):]
            # the WP4 field type wins; a body outside the field (a test / scripted body) needs the operator's
            # declared role; an unknown role is refused (fail closed, OQ-17 A: adult workers only)
            atype = self.actor_types.get(short) or (raw.get("type") if isinstance(raw.get("type"), str) else None)
            if atype is None:
                self._emit(t, "tag_refused", reason="tag_not_adult")
                return
            ok, res = self.tags.issue(short, atype, raw.get("adult") is True, raw.get("shift"), t)
            if ok:
                self._emit(t, "tag_issued", shift=self.tags.shift)
                self.reading = None
            else:
                self._emit(t, "tag_refused", reason=res)
        elif op == "revoke":
            self.tags.revoke()
            self.reading = None
            self._emit(t, "tag_revoked")
        elif op == "rotate":
            self.tags.rotate(str(raw.get("shift", "s1")))
            self._emit(t, "tag_rotated", shift=self.tags.shift)
        elif op == "drop":
            dur = wa._fin(raw.get("dur_s"), None)
            if dur is None or dur < 0:
                self._emit(t, "bad_command", topic="tag", why="dur_s")
                return
            self.drop_until = t + dur
            self._emit(t, "tag_drop", dur_s=round(dur, 2))
        elif op == "walk":
            self._walk_cmd(raw, t, d)
        else:
            self._emit(t, "bad_command", topic="tag", why=f"unknown op {op!r}")

    def _walk_cmd(self, raw, t, d):
        body = self.holder_body()
        wps = raw.get("waypoints")
        sp = wa._fin(raw.get("speed_mps"), 0.15)
        if body is None or not isinstance(wps, list) or not wps or sp is None or sp <= 0:
            self._emit(t, "bad_command", topic="tag", why="walk needs an issued tag, waypoints and speed_mps")
            return
        try:
            pts = [(float(p[0]), float(p[1])) for p in wps]
        except (TypeError, ValueError, IndexError, KeyError):
            pts = None
        if pts is None or not all(math.isfinite(c) for q in pts for c in q):
            self._emit(t, "bad_command", topic="tag", why="waypoints")
            return
        short = body[len("actor_person_"):]
        if short in self.actor_types:
            # a field actor: hand the script to the WP4 world_events runtime (it writes the mocap pose)
            try:
                from atharv_nandi.sim.hooks import world_events as hwe
                from atharv_nandi.sim.world import actors as wact
                rt = hwe.runtime_for(self.m)
                a = rt.actors[short] if rt is not None else None
            except (ImportError, KeyError):
                a = None
            if a is not None:
                a.spec.script = wact.Script(pts, "world", float(sp), [], False, "stay", "path")
                a.start(t)
                rt.dirty = True
                self._emit(t, "tag_walk", n=len(pts), via="world_events")
                return
        mid = self.bodies[body]
        x0, y0 = float(d.mocap_pos[mid][0]), float(d.mocap_pos[mid][1])
        self.walk = ([(x0, y0)] + pts, float(sp), float(t))
        self._emit(t, "tag_walk", n=len(pts), via="follow_tag")

    def _write_walk(self, d, t):
        if self.walk is None:
            return
        body = self.holder_body()
        if body is None:
            self.walk = None
            return
        path, sp, t0 = self.walk
        s = (t - t0) * sp
        for a, b in zip(path, path[1:]):
            L = math.dist(a, b)
            if s <= L:
                f = s / L if L > 1e-9 else 1.0
                x, y = a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
                yaw = math.atan2(b[1] - a[1], b[0] - a[0])
                break
            s -= L
        else:
            x, y = path[-1]
            yaw = None
            self.walk = None
        mid = self.bodies[body]
        d.mocap_pos[mid][0], d.mocap_pos[mid][1] = x, y
        if yaw is not None:
            d.mocap_quat[mid] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def tag_model(self, d, t):
        if t + 1e-9 < self.next_tag_t:
            return
        self.next_tag_t = t + 1.0 / TAG_HZ
        body = self.holder_body()
        if body is None or t < self.drop_until:
            return
        bid = self.body_ids[body]
        hx, hy = float(d.xpos[bid][0]), float(d.xpos[bid][1])
        p = self.pose(d)
        dx, dy = hx - p.x, hy - p.y
        r = math.hypot(dx, dy)
        if r > TAG_MAX_RANGE_M:
            return
        b = wa._wrap(math.atan2(dy, dx) - p.yaw)
        self.reading = wa.TagReading(float(t), max(0.0, r + self.rng.gauss(0.0, TAG_SIGMA_R_M)),
                                     wa._wrap(b + self.rng.gauss(0.0, TAG_SIGMA_B_RAD)))

    # ------------------------------------------------------------------ observation
    def obs(self, d, t, bus) -> wa.Obs:
        sfv = bus.values.get("safety_fields") or {}
        latched = str(sfv.get("latch_reason") or "") if sfv.get("latched") else ""
        hits: List[wa.FieldHit] = []
        try:
            from atharv_nandi.sim.hooks import stability as hs
            last = hs._RT[0].fields.last if hs._RT else None
        except ImportError:  # pragma: no cover
            last = None
        fields_ok = bool(sfv) and sfv.get("ok", True) is not False and last is not None \
            and 0.0 <= t - float(last.t) <= wa.FIELDS_STALE_S
        if fields_ok:
            for fh in list(last.protective) + list(last.warning):
                hits.append(wa.FieldHit(float(fh.ret.x), float(fh.ret.y), fh.field, fh.cls, fh.sector))
        mast = bus.values.get("mast") or {}
        ext = wa._fin(mast.get("ext_m"), 0.0) if isinstance(mast, dict) else 0.0
        link = (bus.values.get("link") or {}).get("local") or {}
        local_ok = not (isinstance(link, dict) and link.get("state") == "lost")
        pf = bus.values.get("platform") or {}
        soc = wa._fin(pf.get("soc"), None) if isinstance(pf, dict) else None
        return wa.Obs(t=float(t), pose=self.pose(d), holding=bus.holding(), latched=latched, hits=hits,
                      tag=self.reading if self.tags.issued else None, crown_ext_m=ext or 0.0, local_link_ok=local_ok,
                      soc=soc, geofence=self.geofence, fields_ok=fields_ok)

    # ------------------------------------------------------------------ work commands
    def work_command(self, raw, ob: wa.Obs):
        if not isinstance(raw, dict) or not isinstance(raw.get("op"), str):
            self._emit(ob.t, "bad_command", topic="work", why="expected {op: ...}")
            return
        op = raw["op"]
        if op == "start":
            mode = raw.get("mode")
            params = raw.get("params") or {}
            if not isinstance(mode, str) or not isinstance(params, dict):
                self._emit(ob.t, "bad_command", topic="work", why="start needs mode and params {}")
                return
            if mode == "lead_to_marks" and params.get("marks") is None and self.marks:
                params = {**params, "marks": list(self.marks)}
            try:
                ok, why = self.assist.start(mode, params, ob)
            except Exception as exc:   # noqa: BLE001 - a malformed command never faults the hook (contract)
                self.assist.events.append({"t": round(float(ob.t), 3), "event": "refused", "mode": None,
                                           "requested": mode, "reasons": ["bad_command"]})
                self._emit(ob.t, "bad_command", topic="work", why=f"start: {type(exc).__name__}")
                return
            self.settled_for = None
            if not ok and "bad_command" in why:
                self._emit(ob.t, "bad_command", topic="work", why="start params")
        else:
            try:
                ok, msg = self.assist.command(op, ob)
            except Exception as exc:   # noqa: BLE001
                self._emit(ob.t, "bad_command", topic="work", why=f"{op}: {type(exc).__name__}")
                return
            self._emit(ob.t, "command", op=op, ok=ok, msg=msg)

    def _new(self, bus, topic, t):
        """Every message delivered on ``topic`` since the last tick, oldest first (a lost one is reported)."""
        n = bus.arrival_count(topic)
        k = self.counts.get(topic, 0)
        if n <= k:
            return []
        got, lost = bus.since(topic, k)
        self.counts[topic] = n
        if lost:
            self._emit(t, "cmd_lost", topic=topic, n=lost)
        return got

    # ------------------------------------------------------------------ tick
    def tick(self, m, d, t, bus):
        if bus is not getattr(self, "_bus", None):      # a new driver / runner bus: its arrival counts start at 0
            self._bus = bus
            self.counts = {}
            self.mark_count = 0
        for raw in self._new(bus, TAG_CMD_TOPIC, t):
            self.tag_command(raw, t, d, bus)
        self._write_walk(d, t)
        n = bus.arrival_count(MARK_TOPIC)
        if n > self.mark_count:
            got, _lost = bus.since(MARK_TOPIC, self.mark_count)
            self.mark_count = n
            self.marks += [mk for mk in got if isinstance(mk, dict) and "tree_id" in mk]
            del self.marks[:-MAX_MARKS]
        self.tag_model(d, t)
        ob = self.obs(d, t, bus)
        for raw in self._new(bus, CMD_TOPIC, t):
            self.work_command(raw, ob)
        drive = self.assist.update(ob)
        self._settle(ob)
        self._holder_hint(ob, bus)
        wd = {"t": round(float(t), 4), **drive.as_dict(), "mode": self.assist.mode}
        bus.values["work_drive"] = wd
        st = self.assist.state(t)
        bus.values["work_assist"] = st
        if self.assist.active and t + 1e-9 >= self.next_drive_t:
            bus.publish(DRIVE_TOPIC, wd)
            self.next_drive_t = t + DRIVE_PERIOD_S
        key = (st["mode"], st["phase"], st["hold"], st["latched"], st["tag"]["issued"], st["tag"]["seen"])
        evs = self.assist.events[self.ev_sent:] + self.local_events
        self.ev_sent = len(self.assist.events)
        self.local_events = []
        for e in evs:
            bus.publish(EVENTS_TOPIC, e)
        if evs or key != self.last_key or t + 1e-9 >= self.next_state_t:
            bus.publish(STATE_TOPIC, {**st, "assisted": list(ASSISTED)})
            self.next_state_t = t + STATE_PERIOD_S
            self.last_key = key

    def _holder_hint(self, ob: wa.Obs, bus):
        """``bus.values["work_tag_holder"]`` (device-local, never published): where the tag holder is in the base frame
        while FOLLOW-ME runs with a fresh reading, so WP6 can leave the holder out of the warning-field speed cap
        (change request, the WP29 contract notes). The protective field is never exempt. Absent otherwise."""
        a, r = self.assist, ob.tag
        if a.mode == "follow_me" and a.phase == "running" and r is not None and ob.t - r.t <= wa.TAG_STALE_S:
            bus.values["work_tag_holder"] = {"t": round(float(r.t), 3),
                                             "x": round(r.range_m * math.cos(r.bearing_rad), 3),
                                             "y": round(r.range_m * math.sin(r.bearing_rad), 3),
                                             "r_m": wa.TAG_ASSOC_M}
        else:
            bus.values.pop("work_tag_holder", None)

    def _settle(self, ob: wa.Obs):
        """Record the stop error once the gait has settled at a lead stop or tote delivery."""
        a = self.assist
        p = ob.pose
        if self.last_xy is not None:
            dt = max(1e-6, ob.t - self.last_xy[2])
            v = math.hypot(p.x - self.last_xy[0], p.y - self.last_xy[1]) / dt
            self.slow_since = (self.slow_since if self.slow_since is not None else ob.t) if v < SETTLE_SPEED_MPS \
                else None
        self.last_xy = (p.x, p.y, ob.t)
        key = (a.mode, a.phase, a.stop_k)
        if a.phase in ("at_stop", "delivered") and self.settled_for != key and self.slow_since is not None \
                and ob.t - self.slow_since >= SETTLE_S:
            err = a.settle_error(p)
            if err is not None:
                self.settled_for = key
                self._emit(ob.t, "stop_settled", mode=a.mode, stop=a.stop_k, stop_err_m=err)


_RUNTIMES: Dict[int, FollowTagRuntime] = {}


def runtime_for(m) -> FollowTagRuntime:
    rt = _RUNTIMES.get(id(m))
    if rt is None or rt.m is not m:
        if len(_RUNTIMES) > 4:
            _RUNTIMES.clear()
        rt = _RUNTIMES[id(m)] = FollowTagRuntime(m)
    return rt


def _tick(m, d, t, bus):
    runtime_for(m).tick(m, d, t, bus)


def _reset(m, d):
    if m is None:
        return
    rt = _RUNTIMES.get(id(m))
    if rt is not None and rt.m is m:
        rt.reset()


def register(hooks):
    hooks.register("follow_tag", _tick, rate_hz=RATE_HZ, topics_in=(CMD_TOPIC, TAG_CMD_TOPIC, MARK_TOPIC),
                   reset=_reset)
