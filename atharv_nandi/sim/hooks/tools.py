"""WP2 hook ``tools`` (50 Hz, mutates the model): rack swap, tool actuation, cut rule, twist release, debris
(ORCHARD-MVP-SPEC 2.4, 3.4; MVP-SPEC 2.3, 3.4, 5.7). ROS-free; ``sim_runner`` and ``sim_driver`` drive it.

``ToolRuntime`` binds to one (MjModel, MjData) of a world composed with tools (``world.modules.tools``):

    change     ``/nandi/tool/change`` {"tool": "T2"|"T4"|"T5"|"T1"|null, "req_id"?} (or a bare tool id string):
               ``ToolChanger`` jobs (dock the mounted tool, mount the target) after ``swap_gate`` (holding, e-stop,
               crown up, margin, person; all but holding re-checked every tick while the job runs). Also
               {"ack_fault": true} (operator assist done: a faulted changer back to idle) and {"release": true}
               (let go of the caught fruit). Anything else is refused ``bad_request``, never a dock
    actuate    ``/nandi/tool/actuate`` {"roll"?: rad, "value"?: blade 0..1 (1 = closed) | jaw opening m,
               "open"?: true, "stop"?: true} on the MOUNTED tool (``ToolActuation``); refused while held. A
               blade / jaw CLOSE (blade value >= 0.5, jaw opening < 0.15 m) also needs the MVP-SPEC 2.3
               blade-close interlock: a target within 0.15 m of the TCP (sim stand-in for the wrist-camera
               track: an assigned pool fruit / prune target), no person in the WORK field, stability OK
    cut        T2 / T5 mounted: every tick the ``CutRule`` checks the assigned pool fruit (``cut_k``, T2) or the
               in-reach prune targets (tree truth, T5) against the blade segment; on fire the pool equality goes
               off and the fruit is caught (assisted catch weld), or the shoot is hidden and a debris body drops.
               ``NANDI_CUT`` = ``logical`` (default, 6 mm window) | ``physics`` (4 mm) | ``resist`` (4 mm + an
               opposing blade torque F_cut x lever while the target is in the window, MVP-SPEC 3.4)
    twist      T4 mounted: ``TwistRelease`` per assigned weld slot (baseline at first sight, roll armed when
               both jaw pads press the fruit); on release the pool equality goes off and the jaws' catch weld on
    e-stop     ``bus.holding()``: actuation refused and the tool actuators held at their position; the cut rule
               and the twist release do not run (a closed blade needs a fresh closing after the hold, the roll
               count restarts); a running swap ends in ``fault`` (reason ``estop``) and safe-parks once the
               hold clears. While a swap runs the WP24 mast refuses raises (``register_gate``)

Publishes ``/nandi/tool/state`` (2 Hz + on change) ``{mounted, parked, mount, phase, reason, busy, tcp_frame,
assisted, roll, value, sensors, last}`` (``sensors``: OS 2.4 blade / roll jointpos, T4 pads, T2 TCP range; ``mount``
``rigid`` for a ``NANDI_TOOL_MOUNT=rigid:<tool>`` build, where every change is refused ``rigid_mount``), ``/nandi/tool/event`` (one message per event: ``tool_change_done``, ``tool_change_fault``,
``tool_change_refused``, ``tool_locked``, ``cut``, ``cut_miss``, ``prune_refused``, ``twist_release``,
``actuate_refused``) and, while a swap runs, ``/nandi/tool/arm_targets`` ``{q5, fingers, payload_kg, phase}``:
the hook cannot drive robot actuators 0..24 (WP0 rule 1), so the arm owner forwards these (``SimDriver``:
``install(driver)``; live stack: change request to WP8 ``harvest_ik_service``). ``bus.values["tools"]`` holds the
state dict for other hooks (WP6 / WP8 / WP24).
"""
import json
import math
import os
import weakref
from typing import Any, Callable, Dict, List, Optional

import mujoco
import numpy as np

from atharv_nandi.core.tools import cut_logic, grasp_sense, registry as reg, tool_changer as tc, twist_logic
from atharv_nandi.core.tools.tool_changer import (BLADE_CLOSE_CTRL, CLOSE_TARGET_M, JAW_OPEN_CTRL_M, MIN_MARGIN_M,  # noqa: F401
                                                    Interlocks, MjToolPlant, RigidMount, ToolChanger)

CHANGE_TOPIC = "/nandi/tool/change"
ACTUATE_TOPIC = "/nandi/tool/actuate"
STATE_TOPIC = "/nandi/tool/state"
EVENT_TOPIC = "/nandi/tool/event"
ARM_TOPIC = "/nandi/tool/arm_targets"
RATE_HZ = 50.0
STATE_PERIOD_S = 0.5
CUT_TARGET_RANGE_M = 0.03       # a target is considered for the cut rule within 3 cm of the blade segment
MISS_AFTER_S = 0.2              # closed this long without a cut -> one cut_miss event per closing
TWIST_TOUCH_N = 0.5            # both T4 pads above this -> the jaws closed on the fruit (roll counted from here)
RESEAT_M = 0.05                # a tool this far from its active weld pose = the robot was teleported: re-seat
ASSISTED = ("weld_tool_mount", "logical_cut", "logical_twist_release", "weld_grasp", "kinematic_debris")


def cut_mode_from_env() -> str:
    v = os.environ.get("NANDI_CUT", "logical").strip().lower() or "logical"
    if v not in ("logical", "physics", "resist"):
        raise ValueError(f"NANDI_CUT={v!r}: use logical, physics or resist")
    return v


class MountedToolActuation:
    """``interfaces.ToolActuation`` for the mounted tool (WP6 codes the e-stop against ``open_blade`` / ``stop``)."""

    def __init__(self, rt: "ToolRuntime"):
        self.rt = rt
        self.stopped = False

    def _ids(self):
        k = self.rt.changer.mounted
        return None if k is None else self.rt.h[k]

    def open_blade(self):
        ids = self._ids()
        if ids is None:
            return
        t = ids.spec
        self.rt.d.ctrl[ids.act_act] = t.ctrl_range[1] if t.kind == "twist" else 0.0

    def stop(self):
        """Hold every tool actuator where it is now (brake in place) and refuse new commands until ``resume``."""
        self.stopped = True
        for ids in self.rt.h.tools.values():
            d = self.rt.d
            d.ctrl[ids.roll_act] = d.qpos[ids.roll_qadr]
            gear = float(self.rt.m.actuator_gear[ids.act_act, 0])
            d.ctrl[ids.act_act] = d.qpos[ids.act_qadr] * gear

    def resume(self):
        self.stopped = False

    def set(self, value: float) -> bool:
        ids = self._ids()
        if ids is None or self.stopped:
            return False
        lo, hi = ids.spec.ctrl_range
        self.rt.d.ctrl[ids.act_act] = float(np.clip(value, lo, hi))
        return True

    def set_roll(self, rad: float) -> bool:
        ids = self._ids()
        if ids is None or self.stopped:
            return False
        lo, hi = ids.spec.roll_range
        self.rt.d.ctrl[ids.roll_act] = float(np.clip(rad, lo, hi))
        return True

    def state(self) -> Dict[str, Any]:
        ids = self._ids()
        if ids is None:
            return {"mounted": None, "value": None, "roll": None, "stopped": self.stopped}
        d = self.rt.d
        return {"mounted": ids.spec.tool_id, "value": float(d.ctrl[ids.act_act]), "roll": float(d.ctrl[ids.roll_act]),
                "q_roll": float(d.qpos[ids.roll_qadr]), "q_act": float(d.qpos[ids.act_qadr]), "stopped": self.stopped}


class ToolRuntime:
    """Everything WP2 does per control tick on one (model, data)."""

    def __init__(self, m, d, arm_sink: Optional[Callable] = None, cut_mode: Optional[str] = None,
                 publish: Optional[Callable[[str, Any], None]] = None):
        from atharv_nandi.sim.world.modules import tools as tools_mod
        self.tools_mod = tools_mod
        self.m, self.d = m, d
        self.h = tools_mod.handles(m)
        if self.h is None:
            raise ValueError("world has no tools")
        self.plant = MjToolPlant(m, d)
        self.changer = ToolChanger(mounted=self.plant.mounted())
        self.act = MountedToolActuation(self)
        self.arm_sink = arm_sink
        self.publish_fn = publish
        self.cut_mode = cut_mode or cut_mode_from_env()
        self.catch = tools_mod.catch_ids(m)
        self.dt = 1.0 / RATE_HZ
        self.pool = None
        self.prune_targets: List[dict] = []
        try:
            from atharv_nandi.core.fruit_pool import FruitPool
            self.pool = FruitPool(m)
        except Exception:  # noqa: BLE001 - a world without a pool or with unknown plans: no fruit features
            self.pool = None
        try:
            from atharv_nandi.sim.world.modules import trees as trees_mod
            self.trees_mod = trees_mod
            for plan in trees_mod.plans_for_model(m):
                off = np.asarray(plan["pos"], float)
                for p in plan["truth"].get("prune", []):
                    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, p["geom"])
                    if gid >= 0:
                        self.prune_targets.append({**p, "gid": gid, "a_w": np.asarray(p["a"], float) + off,
                                                   "b_w": np.asarray(p["b"], float) + off,
                                                   "cut_w": np.asarray(p["cut_world"], float)})
        except Exception:  # noqa: BLE001
            self.trees_mod = None
        self.events: List[Dict[str, Any]] = []
        self.reset_runtime()

    # ------------------------------------------------------------------ reset
    def reset_runtime(self):
        self.cut_rule = cut_logic.CutRule(mode="physics" if self.cut_mode in ("physics", "resist") else "demo")
        self.cut_target: Optional[dict] = None
        self.closed_since: Optional[float] = None
        self.miss_sent = False
        self.twist: Dict[int, twist_logic.TwistRelease] = {}
        self.caught: Dict[str, Any] = {}           # {"slot", "eq", "mass"} of the fruit held by the mounted tool
        self.debris: Dict[int, cut_logic.Debris] = {}
        self.next_debris = 0
        self.cut_count = 0
        self._last_state_t = -math.inf
        self._last_state_key = None
        self._change_seen = 0
        self._actuate_seen = 0
        self.arm_cmd: Optional[dict] = None
        self.changer = ToolChanger(mounted=self.plant.mounted())
        self.act.stopped = False
        d = self.d
        for ids in self.h.tools.values():
            d.ctrl[ids.roll_act] = 0.0
            d.ctrl[ids.act_act] = ids.spec.ctrl_range[1] if ids.spec.kind == "twist" else 0.0
            d.qfrc_applied[ids.act_dadr] = 0.0

    def reset_model(self):
        """World reset: every tool racked with parked contacts and compiled welds, catches off, debris parked."""
        m, d = self.m, self.d
        for k, ids in self.h.tools.items():
            self.tools_mod.set_contact(m, k, False)
            if ids.rigid:                       # rigid build: grip weld on at X_nom, no rack weld
                d.eq_active[ids.grip_eq] = 1
                m.eq_data[ids.grip_eq][3:10] = list(reg.X_NOM_POS) + list(reg.X_NOM_QUAT)
                continue
            d.eq_active[ids.rack_eq] = 1
            d.eq_active[ids.grip_eq] = 0
            m.eq_data[ids.grip_eq][3:10] = list(reg.X_NOM_POS) + list(reg.X_NOM_QUAT)
        for e in self.catch.values():
            d.eq_active[e] = 0
        src = self.tools_mod.debris_sources(m)
        for i in self.tools_mod.debris_ids(m):
            m.geom_rgba[i["geom"]][3] = 0.0
            d.mocap_pos[i["mocap"]] = [self.tools_mod.DEBRIS_PARK[0] + 0.5 * i["k"], self.tools_mod.DEBRIS_PARK[1],
                                       self.tools_mod.DEBRIS_PARK[2]]
            src[i["k"]] = None
        self.reset_runtime()

    # ------------------------------------------------------------------ helpers
    def emit(self, event: Dict[str, Any]):
        ev = {"t": round(float(self.d.time), 4), **event}
        self.events.append(ev)
        if self.publish_fn is not None:
            self.publish_fn(EVENT_TOPIC, ev)

    def interlocks(self, hold: bool, values: Optional[dict] = None) -> Interlocks:
        values = values or {}
        mast = values.get("mast") or {}
        ext = mast.get("ext_m") if isinstance(mast, dict) else None
        stab = values.get("stability") or {}
        margin = stab.get("margin_m") if isinstance(stab, dict) else None
        work = values.get("work_field") or {}
        return Interlocks(estop=bool(hold), holding=self.holding(),
                          crown_ext_m=float(ext) if ext is not None else self.plant.crown_ext_m(),
                          margin_m=float(margin) if margin is not None else self.plant.margin_m(),
                          person_in_field=bool(work.get("person")) if isinstance(work, dict) else False)

    def holding(self) -> bool:
        ja = jb = 0.0
        ids = self.h.tools.get("T4")
        if self.changer.mounted == "T4" and ids is not None and ids.touch_adr:
            ja, jb = (grasp_sense.touch(self.d, a) for a in ids.touch_adr)
        picker = self.changer.mounted is None and self.picker_holding()
        return grasp_sense.holding(bool(self.caught), ja, jb, picker)

    def picker_holding(self) -> bool:
        """T1 picker: both finger bodies in contact with the same free body that is not the robot or a tool (a
        crop in the fingers). The model has no finger-pad touch sensors, so contacts stand in for them."""
        m, d = self.m, self.d
        if not hasattr(self, "_finger_ids"):
            self._finger_ids = tuple(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                                     for n in ("finger_left", "finger_right"))
            self._robot_root = int(m.body_rootid[self._finger_ids[0]]) if self._finger_ids[0] >= 0 else -1
            self._tool_bodies = {int(i.body) for i in self.h.tools.values()}
        if min(self._finger_ids) < 0:
            return False
        touched = {b: set() for b in self._finger_ids}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for f, o in ((b1, b2), (b2, b1)):
                if f in touched:
                    root = int(m.body_rootid[o])
                    if o != 0 and root != self._robot_root and root not in self._tool_bodies \
                            and int(m.body_dofnum[root]) > 0:
                        touched[f].add(root)
        a, b = (touched[f] for f in self._finger_ids)
        return bool(a & b)

    def payload_kg(self) -> float:
        m = 0.0 if self.changer.mounted is None else reg.get(self.changer.mounted).mass_kg
        return m + float(self.caught.get("mass", 0.0))

    def change_message(self, msg: Any, hold: bool = False, values: Optional[dict] = None):
        """One ``/nandi/tool/change`` message (already JSON-decoded where it was JSON). Never raises."""
        def bad(why):
            self.emit({"event": "tool_change_refused", "tool": None, "reason": f"bad_request: {why}"})
            return None
        if isinstance(msg, str):
            target = msg.strip()
            if target.lower() in ("", "null", "none"):
                return bad("bare null (send {\"tool\": null} to dock)")
            return self.request_change(target, hold, values)
        if not isinstance(msg, dict):
            return bad(f"expected a JSON object, got {type(msg).__name__}")
        if msg.get("ack_fault") is True:
            return self.ack_fault(hold)
        if msg.get("release") is True:
            return self.release_request(hold)
        if "tool" not in msg:
            return bad("no 'tool' key")
        target = msg["tool"]
        if target is not None and not isinstance(target, str):
            return bad(f"'tool' must be a string or null, got {type(target).__name__}")
        return self.request_change(target, hold, values)

    def ack_fault(self, hold: bool = False) -> bool:
        """Operator assist acknowledged (``{"ack_fault": true}``): a changer in ``fault`` goes back to idle."""
        if hold:
            self.emit({"event": "ack_fault_refused", "reason": "estop"})
            return False
        if self.changer.phase != "fault":
            self.emit({"event": "ack_fault_refused", "reason": "safe_park_running" if self.changer.faulted
                       else "no_fault"})
            return False
        reason = self.changer.reason
        self.changer.clear_fault()
        self.arm_cmd = None
        self.emit({"event": "tool_fault_cleared", "reason": reason, "mounted": self.changer.mounted})
        return True

    def release_request(self, hold: bool = False) -> bool:
        """``{"release": true}``: let go of the caught fruit (the live path to ``release_catch``)."""
        if hold:
            self.emit({"event": "release_refused", "reason": "estop"})
            return False
        slot = self.release_catch()
        if slot is None:
            self.emit({"event": "release_refused", "reason": "nothing_caught"})
            return False
        self.emit({"event": "catch_released", "slot": slot})
        return True

    def request_change(self, target: Optional[str], hold: bool = False, values: Optional[dict] = None):
        try:
            plans = self.plant.plans_for_change(target)
        except RigidMount:
            self.emit({"event": "tool_change_refused", "tool": target, "reason": "rigid_mount",
                       "mounted": self.h.rigid})
            return None
        except (KeyError, ValueError, TypeError) as exc:
            self.emit({"event": "tool_change_refused", "tool": target, "reason": f"bad_request: {exc}"})
            return None
        if not plans:
            self.emit({"event": "tool_change_done", "ok": True, "tool": target, "reason": "already_mounted",
                       "direction": None})
            return None
        q_now = self.arm_cmd["q5"] if self.arm_cmd else tuple(float(self.d.qpos[a]) for a in self.plant.ik.qadr)
        gate = self.changer.request(plans, self.interlocks(hold, values), q_now)
        if not gate:
            self.emit({"event": "tool_change_refused", "tool": target, "reason": gate.reason, "values": gate.values})
        return gate

    # ------------------------------------------------------------------ tick
    def tick(self, t: float, hold: bool = False, values: Optional[dict] = None, inbox: Optional[dict] = None):
        d, m = self.d, self.m
        moved = self.tools_mod.seat_tools(m, d, tol_m=RESEAT_M)     # the robot was teleported (root qpos write)
        if moved:
            self.emit({"event": "tools_reseated", "bodies": moved})
        if hold and not self.act.stopped:
            self.act.stop()
        elif not hold and self.act.stopped:
            self.act.resume()
        self.values = values or {}
        if inbox:
            if "change" in inbox:
                self.change_message(inbox["change"], hold, values)
            if "actuate" in inbox:
                self.actuate(inbox["actuate"], hold, values)
        # tool change
        job = self.changer.job or (self.changer.queue[0] if self.changer.queue else None)
        plan = job.plan if job is not None and hasattr(job, "plan") else job
        obs = self.plant.observe(plan.tool_id if plan else None, plan.direction if plan else None, estop=hold)
        if (plan is not None or self.changer.phase == "safe_park") and not hold:
            obs.interlock = self.running_interlock(values)
        was_busy = self.changer.busy
        cmd = self.changer.step(obs)
        if cmd.ops:
            self.plant.apply_ops(cmd.ops)
            for op, k in cmd.ops:
                ids = self.h[k]
                d.ctrl[ids.roll_act] = 0.0
                d.ctrl[ids.act_act] = ids.spec.ctrl_range[1] if ids.spec.kind == "twist" else 0.0
        for ev in cmd.events:
            self.emit(ev)
        if was_busy or self.changer.busy:
            self.arm_cmd = {"q5": [float(v) for v in cmd.q5] if cmd.q5 is not None else None,
                            "fingers": [float(v) for v in cmd.fingers], "payload_kg": self.payload_kg(),
                            "phase": cmd.phase}
            if cmd.q5 is not None:
                if self.arm_sink is not None:
                    self.arm_sink(cmd.q5, cmd.fingers, self.payload_kg())
                elif self.publish_fn is not None:
                    self.publish_fn(ARM_TOPIC, self.arm_cmd)
        mounted = self.changer.mounted
        if hold:
            self._hold_monitors()
        elif mounted in ("T2", "T5") and not self.changer.busy:
            self._cut_tick(t, mounted)
        elif mounted == "T4" and not self.changer.busy:
            self._twist_tick(t)
        self._debris_tick(t)
        self._state_tick(t)

    def running_interlock(self, values: Optional[dict] = None) -> str:
        """The swap_gate reason (person / crown up / low margin) that holds now, '' if none (holding excluded)."""
        il = self.interlocks(False, values)
        il.holding = False
        g = tc.swap_gate(il)
        return "" if g else g.reason

    def _hold_monitors(self):
        """E-stop: no cut, no twist release. The blade needs a fresh closing (ctrl < 0.5 first) after the hold, the
        twist roll count restarts, and no opposing blade torque is applied."""
        self.cut_rule.fired = True
        self.miss_sent = True
        self.cut_target = None
        for ids in self.h.tools.values():
            self.d.qfrc_applied[ids.act_dadr] = 0.0
        for stat in self.twist.values():
            stat.roll0 = None

    # ------------------------------------------------------------------ actuation
    def _close_gate(self, req: dict, values: Optional[dict]) -> Optional[str]:
        """MVP-SPEC 2.3 blade-close interlock; the refusal reason or None. Only for a closing command."""
        mounted = self.changer.mounted
        values = values if values is not None else getattr(self, "values", {}) or {}
        return tc.close_gate(self.h[mounted].spec.kind, req.get("value"), values,
                          lambda: self._nearest_target_m(mounted))

    def _nearest_target_m(self, mounted: str) -> float:
        """TCP -> nearest target the wrist camera would track (assigned attached pool fruit: cut point for T2, body
        for T4; prune targets for T5)."""
        d = self.d
        tcp = np.array(d.site_xpos[self.h[mounted].tcp])
        best = math.inf
        if mounted in ("T2", "T4") and self.pool is not None:
            for k in self.pool.assignments:
                if not self.pool.is_attached(d, k):
                    continue
                s = self.pool.slots[k]
                p = d.site_xpos[s.cut_k] if mounted == "T2" else d.xpos[s.body]
                best = min(best, float(np.linalg.norm(np.array(p) - tcp)))
        elif mounted == "T5":
            hidden = set(self.trees_mod.hidden(self.m)) if self.trees_mod is not None else set()
            for p in self.prune_targets:
                if p["gid"] not in hidden:
                    best = min(best, float(np.linalg.norm(p["cut_w"] - tcp)))
        return best

    def actuate(self, req: Any, hold: bool = False, values: Optional[dict] = None) -> bool:
        if not isinstance(req, dict):
            self.emit({"event": "actuate_refused", "reason": "bad_request"})
            return False
        if hold:
            self.emit({"event": "actuate_refused", "reason": "estop"})
            return False
        if self.changer.busy:
            self.emit({"event": "actuate_refused", "reason": "tool_change_busy"})
            return False
        if self.changer.mounted is None:
            self.emit({"event": "actuate_refused", "reason": "no_tool_mounted"})
            return False
        want = req.get("tool")
        if want not in (None, self.changer.mounted, reg.get(self.changer.mounted).name):
            self.emit({"event": "actuate_refused", "reason": "tool_not_mounted", "tool": want})
            return False
        why = self._close_gate(req, values)
        if why is not None:
            self.emit({"event": "actuate_refused", "reason": why, "tool": self.changer.mounted})
            return False
        ok = True
        if req.get("stop"):
            self.act.stop()
            self.act.resume()
            return True
        if req.get("open"):
            self.act.open_blade()
        for key, fn in (("roll", self.act.set_roll), ("value", self.act.set)):
            if key in req:
                v = req[key]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    self.emit({"event": "actuate_refused", "reason": f"bad_{key}"})
                    return False
                ok = fn(float(v)) and ok
        return ok

    # ------------------------------------------------------------------ cut
    def _blade_state(self, ids):
        m, d = self.m, self.d
        t = ids.spec
        a, b = np.array(d.site_xpos[ids.blade_a]), np.array(d.site_xpos[ids.blade_b])
        jid = int(m.dof_jntid[ids.act_dadr])
        hinge = np.array(d.xaxis[jid])
        q = float(d.sensordata[ids.pos_adr]) if ids.pos_adr >= 0 else float(d.qpos[ids.act_qadr])   # blade jointpos
        frac = q / t.close_rad
        cap = float(m.actuator_forcerange[ids.act_act, 1]) * float(m.actuator_gear[ids.act_act, 0]) / t.lever_m
        return a, b, hinge, frac, cap

    def _cut_candidates(self, mounted: str, a, b):
        d = self.d
        out = []
        if mounted == "T2" and self.pool is not None:
            for k, asg in self.pool.assignments.items():
                if not self.pool.is_attached(d, k):
                    continue
                s = self.pool.slots[k]
                cut = np.array(d.site_xpos[s.cut_k])
                dist, _c, _s = cut_logic.point_segment(cut, a, b)
                if dist > CUT_TARGET_RANGE_M:
                    continue
                stem = np.array(d.site_xpos[s.stem_end]) - np.array(d.site_xpos[s.anchor])
                f = self.pool.fruit[asg["fruit_id"]]
                try:
                    f_cut = reg.f_cut_n(f["crop"])
                except KeyError:
                    f_cut = None              # not a CLIPPER crop (papaya, banana): refused, never cut
                out.append({"kind": "fruit", "slot": k, "fruit_id": asg["fruit_id"], "crop": f["crop"], "dist": dist,
                            "line": stem, "f_cut": f_cut, "leave_m": f.get("pedicel_leave_m", 0.0)})
        elif mounted == "T5":
            hidden = set(self.trees_mod.hidden(self.m)) if self.trees_mod is not None else set()
            for p in self.prune_targets:
                if p["gid"] in hidden:
                    continue
                dist, _c, _s = cut_logic.point_segment(p["cut_w"], a, b)
                out.append({"kind": "prune", "target": p, "dist": dist, "line": p["b_w"] - p["a_w"],
                            "f_cut": reg.prune_force_n(p["diameter_mm"])})
        return out

    def _cut_tick(self, t: float, mounted: str):
        d = self.d
        ids = self.h[mounted]
        a, b, hinge, frac, cap = self._blade_state(ids)
        ctrl = float(d.ctrl[ids.act_act])
        if ctrl < 0.5:
            self.cut_rule.reset()
            self.closed_since, self.miss_sent, self.cut_target = None, False, None
            d.qfrc_applied[ids.act_dadr] = 0.0
        cands = [c for c in self._cut_candidates(mounted, a, b) if c["dist"] <= CUT_TARGET_RANGE_M]
        tgt = min(cands, key=lambda c: c["dist"]) if cands else None
        if tgt is not None and tgt["f_cut"] is None:
            # a crop the CLIPPER does not cut (no cut force): refused, one event per closing
            if ctrl >= cut_logic.CTRL_CLOSED and 1000.0 * tgt["dist"] < self.cut_rule.window_mm and not self.miss_sent:
                self.emit({"event": "cut_refused", "tool": mounted, "target": tgt["fruit_id"], "crop": tgt["crop"],
                           "reason": "no_cut_force"})
                self.miss_sent = True
            d.qfrc_applied[ids.act_dadr] = 0.0
            return
        if self.cut_mode == "resist":
            in_window = tgt is not None and 1000.0 * tgt["dist"] < self.cut_rule.window_mm and not self.cut_rule.fired
            d.qfrc_applied[ids.act_dadr] = -tgt["f_cut"] * ids.spec.lever_m if in_window else 0.0
        if tgt is None:
            if ctrl >= cut_logic.CTRL_CLOSED:
                self._maybe_miss(t, mounted, None)
            self.cut_rule.update(cut_logic.CutInputs(t, ctrl, frac, cap, math.inf, math.inf, 90.0))
            return
        x = cut_logic.CutInputs(t, ctrl, frac, cap, tgt["f_cut"], 1000.0 * tgt["dist"],
                                cut_logic.line_angle_deg(tgt["line"], hinge))
        if tgt["kind"] == "prune":
            p = tgt["target"]
            gate = cut_logic.prune_gate(p["diameter_mm"], keep=bool(p.get("keep")), in_band=bool(p.get("in_band", True)))
            if not gate:
                # PRUNER refuses (OS 5.3: open, MARK over_capacity + paste_required): no cut, one event per closing
                if ctrl >= cut_logic.CTRL_CLOSED and x.dist_mm < self.cut_rule.window_mm and not self.miss_sent:
                    self.emit({"event": "prune_refused", "target": p["id"], "reason": gate.reason,
                               "diameter_mm": p["diameter_mm"], "paste_required": bool(gate.values["paste_required"]),
                               "f_cut_n": round(tgt["f_cut"], 1), "capacity_n": round(cap, 1)})
                    self.miss_sent = True
                if self.cut_mode == "resist":
                    d.qfrc_applied[ids.act_dadr] = 0.0
                return
        dec = self.cut_rule.update(x)
        if dec.fire:
            self._fire_cut(t, mounted, ids, tgt, a, b, x)
        elif ctrl >= cut_logic.CTRL_CLOSED:
            self._maybe_miss(t, mounted, dec.conditions, tgt, x)

    def _maybe_miss(self, t, mounted, conditions, tgt=None, x=None):
        if self.closed_since is None:
            self.closed_since = t
        if not self.miss_sent and not self.cut_rule.fired and t - self.closed_since >= MISS_AFTER_S:
            self.miss_sent = True
            ev = {"event": "cut_miss", "tool": mounted, "conditions": conditions}
            if tgt is not None:
                ev.update({"dist_mm": round(x.dist_mm, 2), "angle_deg": round(x.angle_deg, 1),
                           "target": tgt.get("fruit_id") or tgt["target"]["id"]})
            self.emit(ev)

    def _fire_cut(self, t, mounted, ids, tgt, a, b, x):
        d, m = self.d, self.m
        d.qfrc_applied[ids.act_dadr] = 0.0
        self.cut_count += 1
        if tgt["kind"] == "fruit":
            k = tgt["slot"]
            s = self.pool.slots[k]
            stem, anchor = np.array(d.site_xpos[s.stem_end]), np.array(d.site_xpos[s.anchor])
            cut_p = cut_logic.pedicel_cut_point(stem, anchor, a, b)
            err = cut_logic.pedicel_error_mm(cut_p, stem, tgt["leave_m"])
            self.pool.release(d, k)
            caught = self._catch(ids.spec.name, k)
            self.emit({"event": "cut", "tool": mounted, "target": tgt["fruit_id"], "slot": k, "crop": tgt["crop"],
                       "dist_mm": round(x.dist_mm, 2), "angle_deg": round(x.angle_deg, 1),
                       "pedicel_mm": round(1000.0 * float(np.linalg.norm(cut_p - stem)), 2),
                       "pedicel_error_mm": round(err, 2), "caught": caught, "assisted": ["logical_cut", "weld_grasp"]})
        else:
            p = tgt["target"]
            self.trees_mod.hide_geom(m, p["gid"])
            k = self._spawn_debris(t, p)
            self.emit({"event": "cut", "tool": mounted, "target": p["id"], "job": p["job"],
                       "diameter_mm": p["diameter_mm"], "dist_mm": round(x.dist_mm, 2),
                       "angle_deg": round(x.angle_deg, 1), "debris": k, "assisted": ["logical_cut", "kinematic_debris"]})

    def _catch(self, tool_name: str, slot: int) -> bool:
        e = self.catch.get((tool_name, slot))
        if e is None:
            return False
        m, d = self.m, self.d
        head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{tool_name}_head")
        s = self.pool.slots[slot]
        R1 = np.array(d.xmat[head]).reshape(3, 3)
        rel_p = R1.T @ (np.array(d.xpos[s.body]) - np.array(d.xpos[head]))
        q1c, rel_q = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(q1c, d.xquat[head])
        mujoco.mju_mulQuat(rel_q, q1c, d.xquat[s.body])
        m.eq_data[e] = [0.0, 0.0, 0.0, *rel_p.tolist(), *rel_q.tolist(), 1.0]
        d.eq_active[e] = 1
        self.caught = {"slot": slot, "eq": e, "mass": float(m.body_mass[s.body])}
        return True

    def release_catch(self) -> Optional[int]:
        """Let go of the caught fruit (placement: WP3 takes over); returns its slot."""
        if not self.caught:
            return None
        self.d.eq_active[self.caught["eq"]] = 0
        slot = self.caught["slot"]
        self.caught = {}
        return slot

    # ------------------------------------------------------------------ debris
    def _ground_z(self, p) -> float:
        m, d = self.m, self.d
        start = np.array(p, float)
        geomid = np.zeros(1, dtype=np.int32)
        grp = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        for _ in range(8):
            dist = mujoco.mj_ray(m, d, start, np.array([0.0, 0.0, -1.0]), grp, 1, -1, geomid)
            if dist < 0:
                return 0.0
            hit = start + np.array([0, 0, -dist])
            g = int(geomid[0])
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
            if bname.startswith("tree_") or bname.startswith("debris_") or m.body_mocapid[m.geom_bodyid[g]] >= 0:
                start = hit + np.array([0, 0, -1e-4])
                continue
            return float(hit[2])
        return float(start[2])

    def _spawn_debris(self, t, p) -> int:
        m, d = self.m, self.d
        ids = self.tools_mod.debris_ids(m)
        if not ids:
            return -1
        i = ids[self.next_debris % len(ids)]
        self.next_debris += 1
        a, b = p["a_w"], p["b_w"]
        L = float(np.linalg.norm(b - a))
        r = max(0.002, float(p["diameter_mm"]) / 2000.0)
        half = max(0.005, min(0.15, L / 2.0))
        mid = (a + b) / 2.0
        axis = (b - a) / max(L, 1e-9)
        q = np.zeros(4)
        mujoco.mju_quatZ2Vec(q, axis)
        m.geom_size[i["geom"]] = [r, half, 0.0]
        rg = m.geom_rgba[i["geom"]]
        src_rgba = m.geom_rgba[p["gid"]]
        m.geom_rgba[i["geom"]] = [src_rgba[0], src_rgba[1], src_rgba[2], 1.0] if rg is not None else rg
        m.geom_rbound[i["geom"]] = half + r
        # lying on the ground, along the horizontal projection of the shoot
        flat = np.array([axis[0], axis[1], 0.0])
        flat = flat / np.linalg.norm(flat) if np.linalg.norm(flat) > 1e-6 else np.array([1.0, 0.0, 0.0])
        q_rest = np.zeros(4)
        mujoco.mju_quatZ2Vec(q_rest, flat)
        deb = cut_logic.Debris(tuple(mid), tuple(q_rest), float(t), self._ground_z(mid) + r, source=p["geom"])
        self.debris[i["k"]] = deb
        d.mocap_pos[i["mocap"]] = mid
        d.mocap_quat[i["mocap"]] = q
        self.tools_mod.debris_sources(m)[i["k"]] = p["geom"]
        return i["k"]

    def _debris_tick(self, t):
        if not self.debris:
            return
        d = self.d
        ids = {i["k"]: i for i in self.tools_mod.debris_ids(self.m)}
        for k, deb in list(self.debris.items()):
            mid = ids[k]["mocap"]
            d.mocap_pos[mid] = deb.pos(t)
            if deb.landed:
                d.mocap_quat[mid] = deb.quat
                del self.debris[k]

    # ------------------------------------------------------------------ twist
    def _tau_detach(self, crop: str) -> float:
        """tau_detach of the crop profile (WP23 config), cached; 0.15 N.m (OS 3.4 placeholder) without one."""
        cache = self.__dict__.setdefault("_tau_cache", {})
        if crop not in cache:
            try:
                from atharv_nandi.sim.world import crops as crops_mod
                v = crops_mod.load_profile(crop).get("tau_detach_nm")
            except Exception:  # noqa: BLE001
                v = None
            cache[crop] = float(v) if v else 0.15
        return cache[crop]

    def _twist_tick(self, t):
        if self.pool is None:
            return
        d = self.d
        ids = self.h["T4"]
        ta, tb = (grasp_sense.touch(d, a) for a in ids.touch_adr)
        grip = grasp_sense.jaw_grip(ta, tb, TWIST_TOUCH_N)
        jaw_open = float(d.ctrl[ids.act_act]) > JAW_OPEN_CTRL_M
        tcp = np.array(d.site_xpos[ids.tcp])
        roll = float(d.sensordata[ids.pos_adr]) if ids.pos_adr >= 0 else float(d.qpos[ids.roll_qadr])   # roll jointpos
        for k, asg in self.pool.assignments.items():
            s = self.pool.slots[k]
            if s.kind != "weld" or not self.pool.is_attached(d, k):
                self.twist.pop(k, None)
                continue
            f = self.pool.fruit[asg["fruit_id"]]
            tau = self._tau_detach(f["crop"])
            axis = twist_logic.pedicel_axis(d.site_xpos[s.stem_end], d.site_xpos[s.anchor])
            _F, tau_w = self.pool.reaction(d, k)
            st_tq = twist_logic.axis_torque(tau_w, axis)
            stat = self.twist.get(k)
            if stat is None:
                stat = self.twist[k] = twist_logic.TwistRelease(tau, self.dt)
                stat.capture_baseline(st_tq)
                continue
            near = float(np.linalg.norm(np.array(d.xpos[s.body]) - tcp)) < 0.10
            if grip and near and stat.roll0 is None and not jaw_open:
                stat.arm_roll(roll)
            elif stat.roll0 is not None and (jaw_open or not near):
                stat.roll0 = None          # jaws commanded open or the fruit left them: the roll restarts
            if stat.update(st_tq, roll if stat.roll0 is not None else None):
                self.pool.release(d, k)
                caught = self._catch("twist", k)
                self.emit({"event": "twist_release", "tool": "T4", "target": asg["fruit_id"], "slot": k,
                           "crop": f["crop"], "tau_detach_nm": tau, "tau_peak_nm": round(stat.peak, 4),
                           "roll_deg": round(stat.roll_deg, 1), "caught": caught,
                           "assisted": ["logical_twist_release", "weld_grasp"]})
                self.twist.pop(k, None)

    # ------------------------------------------------------------------ state
    def state(self) -> Dict[str, Any]:
        mounted = self.changer.mounted
        cs = self.changer.state()
        a = self.act.state()
        return {"mounted": mounted, "parked": [k for k in reg.TOOL_IDS if k in self.h.tools and k != mounted],
                "mount": "rigid" if self.h.rigid else "rack", "phase": cs["phase"],
                "reason": cs["reason"], "busy": cs["busy"],
                "tcp_frame": reg.get(mounted).tcp_site if mounted else "gripper_tip",
                "assisted": list(ASSISTED), "roll": a.get("roll"), "value": a.get("value"), "stopped": a["stopped"],
                "caught": self.caught.get("slot"), "cut_mode": self.cut_mode, "cuts": self.cut_count,
                "sensors": self.sensors(mounted), "last": cs["last"]}

    def tcp_range_m(self, ids) -> float:
        """T2 TCP rangefinder: mj_ray from site ``clipper_range`` along its z (= the tool axis), every geom group,
        the tool head excluded, like an MJCF rangefinder; -1 = no hit. Cast here at 50 Hz while mounted instead of
        as a sensor on every physics step (6.4 us per step in orchard_demo, measured)."""
        m, d = self.m, self.d
        s = ids.range_site
        geomid = np.zeros(1, dtype=np.int32)
        vec = np.array(d.site_xmat[s]).reshape(3, 3)[:, 2]
        return float(mujoco.mj_ray(m, d, np.array(d.site_xpos[s]), vec, None, 1, int(m.site_bodyid[s]), geomid))

    def sensors(self, mounted: Optional[str] = None) -> Dict[str, Any]:
        """OS 2.4 tool sensors of the mounted tool: blade / roll jointpos (rad), T2 TCP rangefinder (m along the
        tool axis from the TCP, -1 = nothing within the ray), T4 pad touches (N)."""
        mounted = self.changer.mounted if mounted is None else mounted
        if mounted is None:
            return {}
        ids, sd = self.h[mounted], self.d.sensordata
        out: Dict[str, Any] = {}
        if ids.pos_adr >= 0:
            out[ids.spec.pos_sensor] = round(float(sd[ids.pos_adr]), 4)
        if ids.range_site >= 0:
            out[ids.spec.range_name] = round(self.tcp_range_m(ids), 4)
        for name, adr in zip(ids.spec.touch_sensors, ids.touch_adr):
            out[name] = round(float(sd[adr]), 3)
        return out

    def _state_tick(self, t):
        st = self.state()
        key = json.dumps({k: st[k] for k in ("mounted", "phase", "reason", "busy", "caught", "stopped")},
                         sort_keys=True, default=str)
        if self.publish_fn is not None and (key != self._last_state_key or t - self._last_state_t >= STATE_PERIOD_S):
            self._last_state_key, self._last_state_t = key, t
            self.publish_fn(STATE_TOPIC, {"t": round(t, 3), **st})
        self.last_state = st


# ---------------------------------------------------------------------------------------------- hook glue
_RUNTIMES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def runtime_for(m, d, **kw) -> Optional[ToolRuntime]:
    """The runtime of (m, d), created on first use; None for a world without tools."""
    from atharv_nandi.sim.world.modules import tools as tools_mod
    if tools_mod.handles(m) is None:
        return None
    rt = _RUNTIMES.get(m)
    if rt is None or rt.d is not d:
        rt = ToolRuntime(m, d, **kw)
        _RUNTIMES[m] = rt
    return rt


def _hook(m, d, t, bus):
    rt = runtime_for(m, d, publish=bus.publish)
    if rt is None:
        return
    if rt.publish_fn is None:
        rt.publish_fn = bus.publish
    inbox = {}
    for key, topic, attr in (("change", CHANGE_TOPIC, "_change_seen"), ("actuate", ACTUATE_TOPIC, "_actuate_seen")):
        n = bus.arrival_count(topic)
        if n != getattr(rt, attr):
            setattr(rt, attr, n)
            msg = bus.latest(topic)
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except ValueError:
                    msg = msg if key == "change" else None     # a bare tool id ('T2') is a change request
            inbox[key] = msg
    rt.tick(t, hold=bus.holding(), values=bus.values, inbox=inbox)
    bus.values["tools"] = rt.last_state


def _reset(m, d):
    if m is None:
        return
    rt = _RUNTIMES.get(m)
    if rt is not None and d is not None:
        if rt.d is not d:
            rt = runtime_for(m, d)
        rt.reset_model()


def _mast_gate(ext_from, ext_to, ctx):
    """WP24 raise gate: no mast raise while a tool swap runs (OS 2.3 rule 5, crown stowed for a change)."""
    from atharv_nandi.core.interfaces import GateDecision
    rt = _RUNTIMES.get(ctx.get("m")) if isinstance(ctx, dict) and ctx.get("m") is not None else None
    if rt is not None and rt.changer.busy:
        return GateDecision.refuse("tool_change", "tool_change_busy", phase=rt.changer.phase)
    return GateDecision.ok("tool_change")


def _register_mast_gate():
    from atharv_nandi.sim.hooks import mast as mast_hook
    mast_hook.register_gate(_mast_gate)


def register(hooks):
    _register_mast_gate()
    hooks.register("tools", _hook, rate_hz=RATE_HZ, mutates_model=True, topics_in=(CHANGE_TOPIC, ACTUATE_TOPIC),
                   reset=_reset)


def install(driver, cut_mode: Optional[str] = None) -> ToolRuntime:
    """Run WP2 in a ``SimDriver`` built with ``load_hooks=False``: a 50 Hz task that ticks the runtime, applies
    the swap arm targets to the driver (with gravity feed-forward for the tool and caught fruit) and keeps the
    events in ``runtime.events``."""
    ik = driver.ik

    def sink(q5, fingers, payload):
        q = np.asarray(q5, float)
        off = ik.gravity_offsets(driver.d.qpos, q, payload)
        driver.arm = list(q + off) + [float(fingers[0]), float(fingers[1])]

    rt = ToolRuntime(driver.m, driver.d, arm_sink=sink, cut_mode=cut_mode)
    rt.plant.ik = ik
    _RUNTIMES[driver.m] = rt
    _register_mast_gate()
    driver.add_task("tools", lambda drv: rt.tick(drv.t, hold=bool(drv.holding), values=drv.bus.values), RATE_HZ)
    return rt


# ---------------------------------------------------------------------------------------------- driver helpers
class DriverArm:
    """Joint-space arm moves for a ``SimDriver`` with WP2 installed (tests, probes; WP8 owns the real planner):
    smoothstep moves with gravity feed-forward for the tool and caught fruit, and a closed-loop TCP alignment
    standing in for the wrist-camera servo (re-aim at target + measured error)."""

    def __init__(self, driver, rt: ToolRuntime):
        self.drv, self.rt = driver, rt
        self.q = np.array(rt.plant.q_home, float)

    def move(self, q_goal, seconds: float, settle_s: float = 0.3):
        drv, rt = self.drv, self.rt
        start, goal, t0 = self.q.copy(), np.asarray(q_goal, float), drv.t

        def task(dr):
            s = min(1.0, (dr.t - t0) / seconds)
            s = s * s * (3.0 - 2.0 * s)
            q = start + s * (goal - start)
            off = dr.ik.gravity_offsets(dr.d.qpos, q, rt.payload_kg())
            dr.arm = list(q + off) + list(dr.arm[5:7])

        drv.add_task("tool_arm_move", task, 100.0)
        try:
            drv.wait(seconds + settle_s)
        finally:
            drv.remove_task("tool_arm_move")
        self.q = goal

    def tcp_to(self, point, seconds: float = 1.0, pitch: float = 0.0, tool_id: Optional[str] = None):
        from atharv_nandi.core.tools.tool_changer import tcp_closed_form
        t = reg.get(tool_id or self.rt.changer.mounted)
        q, ax = tcp_closed_form(self.drv.ik, self.drv.d.qpos, point, t.tcp_from_wrist_m, pitch)
        if q is None:
            raise ValueError(f"TCP target {np.round(point, 3)} out of reach")
        self.move(q, seconds)
        return ax

    def align(self, point_fn: Callable[[], Any], iters: int = 3, seconds: float = 0.4, pitch: float = 0.0) -> float:
        """Closed-loop TCP alignment on ``point_fn()``; returns the final TCP error (m)."""
        ids = self.rt.h[self.rt.changer.mounted]
        aim = np.array(point_fn(), float)
        for _ in range(iters):
            self.tcp_to(aim, seconds, pitch)
            aim = aim + (np.array(point_fn(), float) - np.array(self.drv.d.site_xpos[ids.tcp]))
        return float(np.linalg.norm(np.array(point_fn(), float) - np.array(self.drv.d.site_xpos[ids.tcp])))
