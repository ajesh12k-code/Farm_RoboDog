"""WP28 hook ``recovery`` (50 Hz; reads the model, writes nothing in it): the recovery behaviours of
``controllers.recovery_behaviours`` on the live bus (ORCHARD-MVP-SPEC 5.3; MISSION-PRESETS 8).

Inputs (bus values, all optional; a missing one is "unknown", never a trigger):
    link        WP7 ``{local, cloud: {state: unknown / up / lost, needs_confirm, ...}}``
    platform    WP7 ``{pose [x, y], rtk, energy_action, ...}``: the navigation pose is the WP7 pose estimate
    mast        WP24 state (``ext_m``, ``arbiter_source``); ``mast_control`` the arbiter (recovery requests)
    stability, safety_fields, motion_inhibit   WP6 (stance, walk inhibit, protective clear, stability clear)
    tools       WP2 tool state (mounted, value) -> tools safe
    hold        the runner's safe hold (e-stop, fall, hook fault)
    work / mission_progress   WP8 / WP27 / WP9 [PROPOSED]: ``{active}``, ``{item_active, current_item, last_completed}``
    gait_cmd    the gait's (vx, vy, yaw) command (``RecoveryRunner`` writes it; the live stack has none yet: the stall
                detector then assumes forward walking)
Commands: ``/nandi/recovery/cmd`` (JSON, every message since the last tick, oldest first)::

    {"arm": true | false}                      act (default: observe only, see below); a disarm is refused while a
                                               behaviour holds the robot (it would drop the motion lock)
    {"run": "safe_park" | "return_home"}       console recovery menu
    {"confirm": "operator"}                    e-stop recovery: the operator confirmation (console only)
    {"ack": "resume" | "halt"}                 link-loss reconnect prompt, SAFE PARK hold, assist; Halt keeps the
                                               robot parked and locked until a later Resume
    {"check": "<item>"}                        FALLEN checklist (crown_lock, lift_points, two_people,
                                               laterals_tools, mast_inspected, person_clear)
    {"abort": true}                            stop SAFE PARK / RETURN HOME / STUCK

``/nandi/recovery/local`` (the local safety link transport: handheld / dock bridge, never exposed to the console)
takes only ``{"confirm": "local"}`` and ``{"ack": ...}`` (a local supervisor ack, which may resume a console-link
loss while the console is still lost); with a paired WP1 handheld it must carry ``"device"`` = the paired name. The
console topic refuses the local confirm and ``"local": true``, and the local topic refuses the operator confirm, so
one sender cannot give both e-stop confirmations (provided the transports keep the topics apart).

Observe-only by default: every other package's tests load every hook file, and a recovery that stows the crown or
opens a tool by itself would change them. The hook acts once armed: ``NANDI_RECOVERY=1`` (live stack; change request
to launch_nandi.sh / WP10), ``{"arm": true}`` on the command topic, or ``bus.values["recovery_arm"] = True``
(``RecoveryRunner`` sets it). Unarmed it still publishes the state (with ``armed: false``) and its triggers.

Armed, it acts on what a hook may touch: WP24 ``recovery`` requests through ``mast_control``; tools open through
``/nandi/tool/actuate {"open": true}`` (delivered in-process); the WP1 beacon through ``/nandi/beacon/request``; the
WP7 link ack (``/nandi/link/cmd {"ack": link}``) after an operator Resume / Halt. Motion it cannot drive (the
runner and ``locomotion_service`` own the gait): it leaves ``bus.values["recovery"]["actions"]["motion"]`` for the
``RecoveryRunner`` (SimDriver) and ``bus.values["motion_inhibit"]["recovery"] = {stop, walk, reasons}`` for
``locomotion_service`` (it rides ``/nandi/motion_inhibit`` with WP6's keys; honouring it is a change request to
``locomotion_logic.MotionInhibitGate``), and publishes ``/nandi/recovery/motion`` (10 Hz while it drives).

Publishes ``/nandi/recovery/state`` (2 Hz + on change: active behaviour, step, outcome, prompt, KPIs) and
``/nandi/recovery/alert`` (OS 5.4 tele/alert ids ``local_link_lost_hold``, ``cloud_link_lost_hold``, ``stuck``,
``low_battery_return``, ``fallen``, ``tool_fault``, ``rth_failed``; command results ``cmd_result``).

Assisted register (sim only, labelled in the state ``assisted``): the stall detector's progress pose is the trunk
truth (``progress_from_truth``; the robot would use RTK FIXED + visual odometry); row-following landmarks are the
registry trunks within 3 m, observed from truth with 3 cm noise (``registry_trunk_obs_from_truth``; WP5's low-plane
trunk fit on the robot). Sink depth is the MuJoCo contact penetration of the feet on soft patches.
"""
import math
import os
from typing import Any, Dict, Optional

import mujoco

from atharv_nandi.core import recovery_behaviours as rb

TOPIC_CMD = "/nandi/recovery/cmd"
TOPIC_LOCAL = "/nandi/recovery/local"      # the local safety link transport (handheld / dock), never the console
TOPIC_STATE = "/nandi/recovery/state"
TOPIC_ALERT = "/nandi/recovery/alert"
TOPIC_MOTION = "/nandi/recovery/motion"
RATE_HZ = 50.0
STATE_PERIOD_S = 0.5
MOTION_PERIOD_S = 0.1
CROWN_RESUBMIT_S = 0.5
TOOLS_RESEND_S = 1.0
LEG_MOVING_RADPS = 0.5            # = stability.LEG_MOVING_RADPS: any leg joint faster = the gait runs
SOFT_PREFIXES = ("patch_sink", "patch_flood", "patch_mud")
FEET = ("FR", "FL", "MR", "ML", "RR", "RL")
ASSISTED = ["progress_from_truth", "registry_trunk_obs_from_truth"]

_STATE: Dict[int, "RecoveryRuntime"] = {}


def _field_for(m) -> Optional[dict]:
    try:
        from atharv_nandi.sim.world import field_spec
        from atharv_nandi.sim.world.modules.terrain import parse_world_text
        wt = parse_world_text(m)
        if wt is None:
            return None
        return field_spec.load(wt[0])
    except Exception:  # noqa: BLE001 - a world without a field file has no map (SAFE PARK sits in place)
        return None


class RecoveryRuntime:
    def __init__(self, m, d):
        self.m, self.d = m, d
        fld = _field_for(m)
        self.field = fld
        self.fmap = rb.FieldMap.from_field(fld) if fld is not None and fld.get("trees") else None
        home = (fld or {}).get("home", {}).get("xy", [0.0, 0.0]) if fld else [0.0, 0.0]
        self.home = (float(home[0]), float(home[1]))
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        self.feet = [g for g in (gid(f"{k}_foot") for k in FEET) if g >= 0]
        self.soft = set()
        for i in range(m.ngeom):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if n.startswith(SOFT_PREFIXES):
                self.soft.add(i)
        self.leg_dofs = []
        for i in range(min(m.nu, 25)):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if any(n.startswith(k) for k in FEET):
                self.leg_dofs.append(int(m.jnt_dofadr[m.actuator_trnid[i, 0]]))
        self.seed = int(os.environ.get("NANDI_SEED", "7") or 7)
        self.armed_env = os.environ.get("NANDI_RECOVERY", "").strip().lower() in ("1", "on", "true", "yes")
        self.reset()

    def reset(self):
        self.mgr = rb.RecoveryManager(self.fmap, self.home)
        self.armed = self.armed_env
        self.noise = rb.seeded_noise(self.seed)
        self.cmd_seen: Dict[str, int] = {}
        self.seq = 0
        self._last_key = None
        self._next_state = 0.0
        self._next_motion = 0.0
        self._crown_sent: Optional[float] = None
        self._tools_sent: Optional[float] = None
        self._beacon: Optional[str] = None
        self.results: list = []
        self.kpi: Dict[str, Any] = {"local_drift_truth_m": 0.0}
        self._truth_stop: Optional[tuple] = None
        self.last_obs: Optional[rb.Obs] = None
        self.last_actions: Optional[rb.Actions] = None

    # ------------------------------------------------------------------ observation
    def _sink_m(self, d) -> float:
        if not self.soft or not self.feet:
            return 0.0
        pen: Dict[int, float] = {}
        feet = set(self.feet)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            f = g1 if g1 in feet else (g2 if g2 in feet else -1)
            o = g2 if f == g1 else g1
            if f >= 0 and o in self.soft:
                pen[f] = max(pen.get(f, 0.0), -float(c.dist))
        return sum(pen.values()) / len(pen) if pen else 0.0

    def obs(self, m, d, t, bus) -> rb.Obs:
        v = bus.values
        w, x, y, z = (float(q) for q in d.qpos[3:7])
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))
        tx, ty = float(d.qpos[0]), float(d.qpos[1])
        pf = v.get("platform") if isinstance(v.get("platform"), dict) else {}
        pose = pf.get("pose")
        fresh = isinstance(pose, (list, tuple)) and len(pose) == 2 and abs(float(pf.get("t", -9)) - t) <= 0.1
        px, py = (float(pose[0]), float(pose[1])) if fresh else (tx, ty)
        rtk = str(pf.get("rtk") or "FIXED") if fresh else "FIXED"
        lk = v.get("link") if isinstance(v.get("link"), dict) else {}
        ll = (lk.get("local") or {}).get("state", "unknown") if isinstance(lk.get("local"), dict) else "unknown"
        lc = (lk.get("cloud") or {}).get("state", "unknown") if isinstance(lk.get("cloud"), dict) else "unknown"
        ms = v.get("mast") if isinstance(v.get("mast"), dict) else {}
        try:
            ext = float(ms.get("ext_m", 0.0))
        except (TypeError, ValueError):
            ext = 0.0
        mi = v.get("motion_inhibit") if isinstance(v.get("motion_inhibit"), dict) else {}
        sti = mi.get("stability") if isinstance(mi.get("stability"), dict) else {}
        sfi = mi.get("safety_fields") if isinstance(mi.get("safety_fields"), dict) else {}
        walk_ok = not (sti.get("stance") or sti.get("walk") or sti.get("stop") or sfi.get("stop"))
        sfs = v.get("safety_fields") if isinstance(v.get("safety_fields"), dict) else {}
        stab = v.get("stability") if isinstance(v.get("stability"), dict) else {}
        tools = v.get("tools") if isinstance(v.get("tools"), dict) else None
        tools_safe = None
        if tools is not None and tools.get("mounted"):
            val = tools.get("value")
            busy = bool(tools.get("busy"))
            tools_safe = (not busy) and (val is None or (isinstance(val, (int, float)) and
                                                        (val < 0.5 if tools.get("mounted") != "T4" else val >= 0.15)))
        hold = v.get("hold")
        work = v.get("work") if isinstance(v.get("work"), dict) else {}
        mp = v.get("mission_progress") if isinstance(v.get("mission_progress"), dict) else {}
        legs = bool(self.leg_dofs) and max(abs(float(d.qvel[k])) for k in self.leg_dofs) > LEG_MOVING_RADPS
        speed = math.hypot(float(d.qvel[0]), float(d.qvel[1]))
        gc = v.get("gait_cmd")
        cvx = cvy = None
        if isinstance(gc, (list, tuple)) and len(gc) >= 2:
            cvx, cvy = float(gc[0]), float(gc[1])
        lms = []
        if self.fmap is not None:
            c, s = math.cos(yaw), math.sin(yaw)
            for tr in self.fmap.nearest_trees((tx, ty)):
                dx, dy = tr.x - tx, tr.y - ty
                lms.append((tr.x, tr.y, c * dx + s * dy + self.noise(), -s * dx + c * dy + self.noise()))
        return rb.Obs(
            t=float(t), x=px, y=py, yaw=yaw, speed=speed, walking=legs or speed > 0.03,
            holding=hold is not None, hold_reason=str((hold or {}).get("reason", "")) if isinstance(hold, dict) else "",
            ext_m=ext, mast_source=ms.get("arbiter_source"), tools_safe=tools_safe, link_local=ll, link_cloud=lc,
            rtk=rtk, energy_action=str(pf.get("energy_action") or "continue"), tilt_deg=tilt,
            sink_m=self._sink_m(d), item_active=bool(work.get("active") or mp.get("item_active")),
            walk_ok=walk_ok, fields_clear=not bool(sfs.get("latched")) and sfs.get("protective_clear", True) is not False,
            stability_ok=not bool(stab.get("latched")), prog_x=tx, prog_y=ty, landmarks=tuple(lms),
            last_completed=mp.get("last_completed"), current_item=mp.get("current_item"), cmd_vx=cvx, cmd_vy=cvy)

    # ------------------------------------------------------------------ tick
    def tick(self, m, d, t, bus):
        if bus.values.get("recovery_arm"):
            self.armed = True
        o = self.obs(m, d, t, bus)
        self.last_obs = o
        self._commands(t, bus, o)
        self.mgr.auto = self.armed
        a = self.mgr.update(o)
        self.last_actions = a
        self.seq += 1
        if self.armed:
            self._apply(t, bus, o, a)
        for al in a.alerts:
            bus.publish(TOPIC_ALERT, {**al, "armed": self.armed})
        self._kpis(o, d)
        self._share(t, bus, o, a)

    def _local_device_ok(self, bus, msg) -> tuple:
        """A paired handheld (WP1 ``handheld_estop`` state) binds the local channel to its device name."""
        hh = bus.values.get("handheld_estop")
        dev = hh.get("device") if isinstance(hh, dict) and hh.get("seen") else None
        if dev is None:
            return True, ""
        if not isinstance(msg, dict) or str(msg.get("device", "")) != str(dev):
            return False, f"refused: local channel message not from the paired handheld {dev!r}"
        return True, ""

    def _commands(self, t, bus, o):
        for topic, source in ((TOPIC_CMD, "console"), (TOPIC_LOCAL, "local")):
            msgs, lost = bus.since(topic, self.cmd_seen.get(topic, 0))
            self.cmd_seen[topic] = bus.arrival_count(topic)
            for msg in msgs:
                try:
                    if isinstance(msg, str):
                        import json
                        msg = json.loads(msg)
                    if source == "local":
                        ok, why = self._local_device_ok(bus, msg)
                        if ok:
                            ok, why = self.mgr.command(t, msg, o, source="local")
                    elif isinstance(msg, dict) and "arm" in msg:
                        ok, why = self._arm(bool(msg["arm"]))
                    else:
                        ok, why = self.mgr.command(t, msg, o, source="console")
                except Exception as exc:  # noqa: BLE001 - malformed input never faults the hook
                    ok, why = False, f"{type(exc).__name__}: {exc}"
                res = {"t": round(t, 3), "id": "cmd_result", "source": source,
                       "cmd": msg if isinstance(msg, (dict, str)) else str(msg), "ok": ok, "detail": why}
                self.results.append(res)
                bus.publish(TOPIC_ALERT, res)
            if lost:
                bus.publish(TOPIC_ALERT, {"t": round(t, 3), "id": "cmd_lost", "n": lost, "source": source})
        for k in self.mgr.pending_ack_link:
            bus.deliver("/nandi/link/cmd", {"ack": k})       # WP7: clears needs_confirm on the link status
        self.mgr.pending_ack_link.clear()

    def _arm(self, want: bool) -> tuple:
        """Disarm only when no behaviour holds the robot: a disarm would drop ``motion_inhibit.recovery`` (the only
        motion stop on a local-link loss; WP6 retracts the crown only) without the Resume / Halt answer."""
        b = self.mgr.active
        if not want and self.armed and b is not None:
            return False, f"refused: {b.name} holds the robot (step {b.step}); finish, answer or abort it first"
        self.armed = want
        return True, f"armed={self.armed}"

    def _apply(self, t, bus, o, a: rb.Actions):
        mc = bus.values.get("mast_control")
        if a.crown is not None and mc is not None and not o.holding:
            self._ensure_crown(t, bus, mc, a.crown, a.crown_reason or "recovery")
        if a.tools_open and o.tools_safe is False and not o.holding:
            if self._tools_sent is None or t - self._tools_sent >= TOOLS_RESEND_S:
                bus.deliver("/nandi/tool/actuate", {"open": True})
                self._tools_sent = t
        if not a.tools_open:
            self._tools_sent = None
        if a.beacon != self._beacon:
            if a.beacon is not None:
                bus.deliver("/nandi/beacon/request", {"state": a.beacon, "source": "recovery"})
            elif self._beacon is not None:
                bus.deliver("/nandi/beacon/request", {"state": "idle", "source": "recovery"})
            self._beacon = a.beacon

    def _ensure_crown(self, t, bus, mc, target: float, reason: str):
        from atharv_nandi.core.interfaces import MastRequest
        ms = bus.values.get("mast") if isinstance(bus.values.get("mast"), dict) else {}
        if ms.get("has_mast") is False:
            return
        try:
            ext = float(ms.get("ext_m", 0.0))
            tgt = float(ms.get("target_ext", ext))
        except (TypeError, ValueError):
            return
        if ext <= target + rb.CROWN_STOWED_M:
            return
        src = ms.get("arbiter_source")
        if src in ("estop", "safety") and tgt <= target + 1e-6:
            return                              # a higher-priority retract is already on its way down
        if src == "recovery" and tgt <= target + 1e-6:
            return
        if self._crown_sent is not None and t - self._crown_sent < CROWN_RESUBMIT_S:
            return
        self._crown_sent = t
        mc.submit(MastRequest("recovery", float(target), reason, t=float(t)))

    def _kpis(self, o: rb.Obs, d):
        b = self.mgr.active
        if isinstance(b, rb.LocalLinkLoss) and b.sw.t_stopped is not None and b.t_restored is None:
            if self._truth_stop is None:
                self._truth_stop = (float(d.qpos[0]), float(d.qpos[1]))
            dd = math.hypot(float(d.qpos[0]) - self._truth_stop[0], float(d.qpos[1]) - self._truth_stop[1])
            self.kpi["local_drift_truth_m"] = max(self.kpi["local_drift_truth_m"], dd)
        elif not isinstance(b, rb.LocalLinkLoss):
            self._truth_stop = None

    def _share(self, t, bus, o, a: rb.Actions):
        b = self.mgr.active
        st = {"t": round(t, 3), "armed": self.armed, **self.mgr.status(), "prompt": a.prompt,
              "lock": a.lock, "queue_pause": a.queue_pause, "new_targets": a.new_targets,
              "kpi": {k: rb._r(v, 4) for k, v in self.kpi.items()}, "assisted": ASSISTED,
              "obs": {"link_local": o.link_local, "link_cloud": o.link_cloud, "rtk": o.rtk, "sink_m": rb._r(o.sink_m, 4),
                      "walking": o.walking, "ext_m": rb._r(o.ext_m), "energy_action": o.energy_action}}
        acts = {"motion": a.motion.to_dict() if (a.motion is not None and self.armed) else None,
                "lock": a.lock and self.armed, "arm_park": a.arm_park and self.armed,
                "power_off": a.power_off and self.armed, "stand_up": a.stand_up and self.armed}
        bus.values["recovery"] = {**st, "actions": acts, "seq": self.seq}
        mi = bus.values.setdefault("motion_inhibit", {})
        locked = bool(acts["lock"])
        mi["recovery"] = {"stop": locked, "walk": locked, "reasons": [b.name] if (locked and b is not None) else []}
        key = (st["active"], (st["behaviour"] or {}).get("step"), (st["behaviour"] or {}).get("outcome"),
               a.prompt, self.armed, a.lock)
        if key != self._last_key or t + 1e-9 >= self._next_state:
            bus.publish(TOPIC_STATE, bus.values["recovery"])
            self._last_key = key
            self._next_state = t + STATE_PERIOD_S
        if acts["motion"] is not None and t + 1e-9 >= self._next_motion:
            bus.publish(TOPIC_MOTION, {"t": round(t, 3), **acts["motion"], "lock": locked})
            self._next_motion = t + MOTION_PERIOD_S


def runtime_for(m) -> Optional[RecoveryRuntime]:
    return _STATE.get(id(m))


def register(hooks):
    def tick(m, d, t, bus):
        rt = _STATE.get(id(m))
        if rt is None or rt.d is not d:
            rt = _STATE[id(m)] = RecoveryRuntime(m, d)
        rt.tick(m, d, t, bus)

    def reset(m, d):
        rt = _STATE.get(id(m)) if m is not None else None
        if rt is not None:
            rt.reset()

    hooks.register("recovery", tick, rate_hz=RATE_HZ, topics_in=(TOPIC_CMD, TOPIC_LOCAL), reset=reset)
