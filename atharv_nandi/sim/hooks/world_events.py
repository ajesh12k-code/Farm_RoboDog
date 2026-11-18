"""WP4 hook ``world_events`` (50 Hz): scripted world events, actor motion, lighting, foot-on-lateral counter.

Orchard worlds only: the hook reads the compiled model's ``nandi_world`` text (``world/modules/terrain.py``) and
loads that field; in a model without it (legacy, test models) every call returns at once and nothing is
published, so the legacy world runs byte-identical.

Per tick (after ``mj_step``):
    1. ``/nandi/world/cmd`` (every message delivered since the last tick, oldest first, via ``bus.since``; any that
       fell out of the 32-message bus window give one ``cmd_lost`` event ``{n}``): ``{"fire": event_id}``,
       ``{"actor": id, "action": start|stop|park|home}``, ``{"light": preset, "fog": density 0-1}``,
       ``{"reset_counters": true}`` (+ ``seq``, echoed). Types are checked (``check_command``); a malformed
       message gives a ``cmd_refused`` event, never a hook fault
    2. ``EventScheduler.update`` → wind / rain go to the WP7 platform hook as in-process commands
       (``bus.deliver("/nandi/wind/cmd" | "/nandi/weather/cmd", ...)``: WP7 owns the drag and the weather stub),
       light / fog write the model lighting, actor events start / stop scripts
    3. actor scripts → ``data.mocap_pos`` / ``mocap_quat`` (z follows the terrain)
    4. foot-on-lateral counter (``LateralCounter``) from the six foot geoms
    5. publish ``/nandi/world/events`` (each fired event, each lateral step) and ``/nandi/world/state`` (1 Hz and
       on change); ``bus.values["world"]`` holds the same state for WP6 / WP24 / WP27 / WP28

Model mutation: the lighting preset (sun light, headlight, fog), reported by the sim-state provider ``lighting``
(``world.lighting.register_provider``, registered when an orchard world is composed). ``reset`` restores the
compiled lighting and clears scripts, schedule and counter.
"""
import math
from typing import Dict, List, Optional

from atharv_nandi.core import world_events as wev

STATE_TOPIC = "/nandi/world/state"
EVENTS_TOPIC = "/nandi/world/events"
CMD_TOPIC = "/nandi/world/cmd"
RATE_HZ = 50.0
STATE_PERIOD_S = 1.0
FEET = ("FR_foot", "FL_foot", "MR_foot", "ML_foot", "RR_foot", "RL_foot")
FOOT_R = 0.03


class WorldRuntime:
    """Per-model state of the hook (one per compiled orchard model)."""

    def __init__(self, m, world: str, seed: int):
        import mujoco
        from atharv_nandi.sim.world import actors, compose, field_spec, lighting, terrain_orchard
        self.m = m
        self.world, self.seed = world, int(seed)
        self.field = compose.load_field(world, seed)
        self.grid = terrain_orchard.grid_for(self.field, seed)
        self.light_base = lighting.snapshot(m)
        self.light_default = (self.field.get("lighting") or {}).get("default", lighting.DEFAULT_PRESET)
        bodies = actors.actor_bodies(m)
        self.actors: Dict[str, "actors.ActorRuntime"] = {}
        self.mocap: Dict[str, int] = {}
        self.prop_mocap: Dict[str, int] = {}
        entries = []
        for e in self.field.get("actors") or []:
            spec = actors.ActorSpec.from_dict(e)
            name = actors.body_name(spec)
            if name not in bodies:          # left out by the privacy policy (person actors with noVNC up)
                continue
            self.actors[spec.id] = actors.ActorRuntime(spec)
            self.mocap[spec.id] = bodies[name]
            pn = actors.prop_body_name(spec.id)
            if pn in bodies:
                self.prop_mocap[spec.id] = bodies[pn]
            entries.append(e)
        self.scheduler = wev.EventScheduler(self.field.get("events") or [], entries)
        self.counter = wev.LateralCounter(field_spec.lateral_segments(self.field))
        self.feet = [(n, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)) for n in FEET]
        self.feet = [(n, g) for n, g in self.feet if g >= 0]
        self.reset()

    def reset(self):
        from atharv_nandi.sim.world import lighting
        lighting.restore(self.m, self.light_base)
        self.light = {"preset": self.light_default, "fog": 0.0}
        for a in self.actors.values():
            a.reset()
        self.scheduler.reset()
        self.counter.reset()
        # /nandi/world/cmd: commands delivered before the reset are not replayed after it (the seen count jumps to
        # the bus count; a runtime that has not ticked yet starts at 0 and applies what was already delivered)
        bus = getattr(self, "bus", None)
        self.cmd_seen = bus.arrival_count(CMD_TOPIC) if bus is not None else getattr(self, "cmd_seen", 0)
        self.cmd_lost = getattr(self, "cmd_lost", 0)
        self.next_state_t = 0.0
        self.dirty = True
        self.fired: List[dict] = []

    # ------------------------------------------------------------------ actions
    def _robot_pose(self, d):
        q = d.qpos
        yaw = math.atan2(2.0 * (q[3] * q[6] + q[4] * q[5]), 1.0 - 2.0 * (q[5] * q[5] + q[6] * q[6]))
        return float(q[0]), float(q[1]), float(yaw)

    def _set_light(self, preset: Optional[str] = None, fog: Optional[float] = None):
        from atharv_nandi.sim.world import lighting
        p = preset if preset is not None else self.light["preset"]
        f = float(fog) if fog is not None else self.light["fog"]
        if not math.isfinite(f):
            raise ValueError(f"fog density {fog!r} is not finite")
        f = min(1.0, max(0.0, f))
        lighting.preset(p)
        lighting.restore(self.m, self.light_base)
        lighting.apply_model(self.m, p, f)
        self.light = {"preset": p, "fog": f}
        self.dirty = True

    def _actor(self, aid: str, action: str, t: float, pose):
        a = self.actors.get(aid)
        if a is None:
            return False
        if action == "start":
            a.start(t, pose)
        elif action == "stop":
            a.stop()
        elif action == "park":
            a.park()
        elif action == "home":
            a.reset()
        else:
            return False
        self.dirty = True
        return True

    def apply_event(self, ev: "wev.Event", t: float, pose, bus) -> dict:
        rec = {"t": round(float(t), 4), "type": ev.type, "id": ev.id}
        data = ev.data
        t_fire = round(float(t), 4)
        # WP7 hand-off: every payload carries the sim time it fired at (t_fire, s); the t0 values inside are
        # seconds after t_fire, so the timing does not depend on when WP7 reads its inbox
        if ev.type == "wind":
            script = dict(data.get("script", {}))
            bus.deliver(wev.WIND_CMD_TOPIC, {**script, "t_fire": t_fire, "event": ev.id})
            rec["script"] = script
        elif ev.type == "rain":
            cmd = {"rain": {"t0": 0.0, "dur": float(data.get("dur_s", 600.0)), "mm_h": float(data.get("mm_h", 10.0))},
                   "t_fire": t_fire, "event": ev.id}
            bus.deliver(wev.WEATHER_CMD_TOPIC, cmd)
            rec["cmd"] = cmd
        elif ev.type == "weather":
            bus.deliver(wev.WEATHER_CMD_TOPIC, {**dict(data.get("cmd", {})), "t_fire": t_fire, "event": ev.id})
        elif ev.type == "light":
            self._set_light(preset=data["preset"])
            rec["preset"] = data["preset"]
        elif ev.type == "fog":
            self._set_light(fog=float(data.get("density", 0.5)))
            rec["fog"] = self.light["fog"]
        elif ev.type == "actor":
            rec["actor"], rec["action"] = data["actor"], data.get("action", "start")
            rec["ok"] = self._actor(data["actor"], data.get("action", "start"), t, pose)
        elif ev.type == "marker":
            rec["note"] = data.get("note", "")
        self.fired.append(rec)
        return rec

    def command(self, raw, t: float, pose, bus) -> List[dict]:
        """Apply one ``/nandi/world/cmd`` message. A malformed or unknown command is answered with a
        ``cmd_refused`` event and never raises (an exception would fault the hook and safe-hold the robot)."""
        cmd, why = wev.check_command(raw)
        if cmd is None:
            return [{"t": round(t, 4), "type": "cmd_refused", "reason": why, "raw": str(raw)[:200]}]
        try:
            return self._apply_command(cmd, t, pose)
        except (KeyError, ValueError, TypeError, AttributeError) as exc:     # backstop: typed checks above
            return [{"t": round(t, 4), "type": "cmd_refused", "reason": f"{type(exc).__name__}: {exc}"[:200],
                     "raw": str(raw)[:200]}]

    def _apply_command(self, cmd: dict, t: float, pose) -> List[dict]:
        out = []
        seq = {"seq": cmd["seq"]} if "seq" in cmd else {}
        if "fire" in cmd:
            try:
                self.scheduler.fire(cmd["fire"], t)
            except KeyError as exc:
                out.append({"t": round(t, 4), "type": "cmd_refused", "reason": str(exc), **seq})
        if "actor" in cmd:
            ok = self._actor(cmd["actor"], cmd.get("action", "start"), t, pose)
            out.append({"t": round(t, 4), "type": "actor", "id": "cmd", "actor": cmd["actor"],
                        "action": cmd.get("action", "start"), "ok": ok, **seq})
        if "light" in cmd or "fog" in cmd:
            try:
                self._set_light(cmd.get("light"), cmd.get("fog"))
                out.append({"t": round(t, 4), "type": "light", "id": "cmd", **self.light, **seq})
            except KeyError as exc:
                out.append({"t": round(t, 4), "type": "cmd_refused", "reason": str(exc), **seq})
        if cmd.get("reset_counters") is True:
            self.counter.reset()
            self.dirty = True
        return out

    def _write_actors(self, d, t: float):
        from atharv_nandi.sim.world import actors
        for aid, a in self.actors.items():
            x, y, yaw = a.update(t)
            mid = self.mocap[aid]
            d.mocap_pos[mid] = (x, y, self.grid.surface_at(x, y) if a.state != "parked" else 0.0)
            d.mocap_quat[mid] = actors.yaw_quat(yaw)
            if aid in self.prop_mocap:
                fp = a.fruit_drop(t)
                if fp is not None:
                    fx, fy, fz, _falling = fp
                    d.mocap_pos[self.prop_mocap[aid]] = (fx, fy, fz + (self.grid.surface_at(fx, fy)
                                                                       if a.state != "parked" else 0.0))

    def _feet(self, d):
        out = []
        for n, g in self.feet:
            x, y, z = (float(v) for v in d.geom_xpos[g])
            ground = self.grid.surface_at(x, y)
            out.append((n, x, y, (z - FOOT_R) - ground <= wev.FOOT_CONTACT_TOL_M))
        return out

    def state(self, t: float, d) -> dict:
        from atharv_nandi.sim.world import lighting
        acts = []
        for aid, a in self.actors.items():
            mid = self.mocap[aid]
            acts.append({"id": aid, "type": a.spec.type, "class": a.spec.cls, "state": a.state,
                         "xy": [round(float(d.mocap_pos[mid][0]), 3), round(float(d.mocap_pos[mid][1]), 3)]})
        light = lighting.state(self.light["preset"], self.light["fog"])
        return {"t": round(float(t), 3), "world": self.world, "light": light,
                "actors": acts, "laterals": self.counter.status(), "events_fired": self.scheduler.fired_ids(),
                "source": "sim_truth"}

    def tick(self, m, d, t: float, bus):
        pose = self._robot_pose(d)
        events = []
        # every /nandi/world/cmd delivered since the last tick, in delivery order (bus.since: two commands within
        # one 20 ms tick are both applied, a repeat of the same object counts again); messages that fell out of
        # the bus window are reported in one cmd_lost event, never dropped silently
        self.bus = bus
        if bus.arrival_count(CMD_TOPIC) != self.cmd_seen:
            msgs, lost = bus.since(CMD_TOPIC, self.cmd_seen)
            self.cmd_seen += len(msgs) + lost      # exactly what since() covered (a racing delivery waits a tick)
            if lost:
                self.cmd_lost += lost
                events.append({"t": round(t, 4), "type": "cmd_lost", "n": int(lost)})
            for raw in msgs:
                events += self.command(raw, t, pose, bus)
        for ev in self.scheduler.update(t, pose[:2]):
            try:
                events.append(self.apply_event(ev, t, pose, bus))
            except (KeyError, ValueError, TypeError) as exc:   # backstop: field_spec.validate checks events[]
                events.append({"t": round(t, 4), "type": "event_refused", "id": ev.id,
                               "reason": f"{type(exc).__name__}: {exc}"[:200]})
        if self.actors:
            self._write_actors(d, t)
        steps = self.counter.update(t, self._feet(d))
        events += steps
        for e in events:
            bus.publish(EVENTS_TOPIC, e)
        if events or self.dirty or t + 1e-9 >= self.next_state_t:
            st = self.state(t, d)
            bus.values["world"] = st
            bus.publish(STATE_TOPIC, st)
            self.next_state_t = t + STATE_PERIOD_S
            self.dirty = False


_RUNTIMES: Dict[int, WorldRuntime] = {}
_NOT_ORCHARD: Dict[int, object] = {}     # id -> model, for models without a field (legacy): checked once


def runtime_for(m) -> Optional[WorldRuntime]:
    """The hook state for model m (None in a model that is not an orchard world)."""
    rt = _RUNTIMES.get(id(m))
    if rt is not None and rt.m is m:
        return rt
    if _NOT_ORCHARD.get(id(m)) is m:
        return None
    from atharv_nandi.sim.world.modules import terrain as terrain_mod
    ident = terrain_mod.parse_world_text(m)
    if ident is None or not ident[0]:
        if len(_NOT_ORCHARD) > 8:
            _NOT_ORCHARD.clear()
        _NOT_ORCHARD[id(m)] = m
        return None
    if len(_RUNTIMES) > 4:
        _RUNTIMES.clear()
    rt = _RUNTIMES[id(m)] = WorldRuntime(m, ident[0], ident[1])
    return rt


def _tick(m, d, t, bus):
    rt = runtime_for(m)
    if rt is not None:
        rt.tick(m, d, t, bus)


def _reset(m, d):
    if m is None:
        return
    rt = runtime_for(m)
    if rt is not None:
        rt.reset()


def register(hooks):
    hooks.register("world_events", _tick, rate_hz=RATE_HZ, mutates_model=True, topics_in=(CMD_TOPIC,), reset=_reset)
