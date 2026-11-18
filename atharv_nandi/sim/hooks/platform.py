"""WP7 platform hooks: battery and route energy, GNSS, IMU, pose estimate, link heartbeats, wind, weather.

Registered by ``hooks.load_all()`` in sim_runner and the WP0b driver (MVP-SPEC 2.5, 5.6, 5.7; ORCHARD-MVP-SPEC 1.3,
3.6, 4.5, 5.3, 5.4). Three hooks:

    platform_energy  every physics step (substep point): integrates P = sum |actuator_force * actuator_velocity| / eta
                     + P_idle over every actuator (robot + appended), i.e. the exact integrated electrical energy.
    platform         50 Hz: battery (Battery.consume), crown-motion energy (bus.values["mast"]["ext_m"], WP24),
                     route-energy coefficient tracker, reserve / projection, GNSS + RTK supervisor, IMU noise layer,
                     pose estimate, local and cloud link heartbeats, weather stub; publishes the topics below.
    wind             50 Hz: scripted wind and gusts -> anemometer + gust flag -> drag in d.xfrc_applied on the trunk
                     and mast sections. A calm script writes nothing (legacy physics byte-identical).

Publishes (std_msgs/String JSON, all [PROPOSED] in the WP7 contract notes):
    /nandi/imu 50 Hz · /nandi/gnss/fix 10 Hz · /nandi/pose_estimate 10 Hz · /nandi/platform/battery 1 Hz
    (BatteryState superset + projected_end_soc + reserve) · /nandi/energy 1 Hz · /nandi/link/local and
    /nandi/link/cloud 2 Hz + on change · /nandi/wind 10 Hz + on gust change · /nandi/weather 1 Hz ·
    /nandi/platform/alert on events (rtk_lost, rtk_recovered, low_battery_return, local_link_lost_hold,
    cloud_link_lost_hold, link_restored, gust, docked_charged, cmd_rejected).
Subscribes (topics_in): /nandi/link/local/heartbeat, /nandi/link/cloud/heartbeat (any JSON incl. null; each delivery
is a beat), /nandi/link/cmd, /nandi/wind/cmd, /nandi/weather/cmd, /nandi/gnss/cmd, /nandi/platform/plan.
Shares on the bus: values["platform"], values["wind"], values["weather"], values["link"] (read by WP6/WP27/WP28).

Env (all optional; JSON text or a path to a JSON file): NANDI_WIND, NANDI_WEATHER, NANDI_GNSS, NANDI_LINK,
NANDI_ENERGY_CARD (last run card: seeds the coefficients); NANDI_IMU_NOISE=0 for ideal IMU values; NANDI_SEED.
NANDI_LINK {"clock": "sim_and_wall"} also requires the wall-clock heartbeat age past the timeout (faster-than-real
runs fed by a wall-paced heartbeat source); default "sim". A malformed command is rejected (alert cmd_rejected) and
never disables the hook. bus.values["platform"]["t"] and each values["link"][k]["t"] stamp the sim time of the
last update: a reader compares them with d.time to see stale values (hook faulted).
"""
import json
import math
import os
import time
from typing import Dict, Optional

from atharv_nandi.core import energy_model as em
from atharv_nandi.core import geo_frame as gf
from atharv_nandi.core import gnss_model as gm
from atharv_nandi.core import imu_model as im
from atharv_nandi.core import route_energy as re_
from atharv_nandi.core import weather_stub as ws
from atharv_nandi.core import wind_model as wm

LOCAL_TIMEOUT_S = 0.5     # ORCHARD-MVP-SPEC 5.3 / MP 8: local safety link heartbeat
CLOUD_TIMEOUT_S = 5.0     # cloud console heartbeat
LINKS = {"local": (LOCAL_TIMEOUT_S, "stop_hold", "local_link_lost", "local_link_lost_hold"),
         "cloud": (CLOUD_TIMEOUT_S, "safe_park_hold", "cloud_link_lost", "cloud_link_lost_hold")}
TOPIC_HB = {"local": "/nandi/link/local/heartbeat", "cloud": "/nandi/link/cloud/heartbeat"}
TOPIC_LINK = {"local": "/nandi/link/local", "cloud": "/nandi/link/cloud"}
TOPICS_IN = (TOPIC_HB["local"], TOPIC_HB["cloud"], "/nandi/link/cmd", "/nandi/wind/cmd", "/nandi/weather/cmd",
             "/nandi/gnss/cmd", "/nandi/platform/plan")
ROUTE_SAMPLE_S = 0.5      # trunk xy sampled at 2 Hz for route metres (filters the gait sway)
DOCK_RADIUS_M = 0.15      # = mission_executor.DOCK_RADIUS_M: idle at home = on the (ASSUMED) charger
DOCK_DWELL_S = 5.0        # ASSUMED: parked (no plan, < DOCK_SPEED_MPS) this long at home before the pack is full
DOCK_SPEED_MPS = 0.05
LINK_CLOCKS = ("sim", "sim_and_wall")


def _env_json(name: str) -> Optional[dict]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    if raw.startswith("{"):
        return json.loads(raw)
    with open(raw, encoding="utf-8") as fh:
        return json.load(fh)


class LinkMonitor:
    """Heartbeat watchdog for one link. States: unknown (no beat yet) | up | lost. A loss latches
    ``needs_confirm`` until ``ack()`` (operator "Resume / Halt"). ``drop(t, dur)`` ignores beats (injection)."""

    def __init__(self, name: str, timeout_s: float, action: str, reason: str, alert: str, clock: str = "sim"):
        if clock not in LINK_CLOCKS:
            raise ValueError(f"link clock must be one of {LINK_CLOCKS}")
        self.name, self.timeout_s, self.action_on_loss, self.reason, self.alert = name, timeout_s, action, reason, alert
        # "sim": age in sim time (the robot moves in sim time, so at RTF <= 1 this is the physical detection).
        # "sim_and_wall": a loss also needs the wall-clock age past the timeout, for runs faster than real time fed
        # by a wall-paced heartbeat source (else normal beats look like a loss).
        self.clock = clock
        self.reset()

    def reset(self):
        self.state = "unknown"
        self.last_beat: Optional[float] = None
        self.last_beat_wall: Optional[float] = None
        self.lost_since: Optional[float] = None
        self.needs_confirm = False
        self.drop_until = -math.inf
        self.losses = 0

    def drop(self, t: float, dur: float):
        self.drop_until = t + max(0.0, dur)

    def beat(self, t: float, wall: Optional[float] = None):
        if t < self.drop_until:
            return
        self.last_beat = t
        self.last_beat_wall = wall

    def ack(self):
        self.needs_confirm = False

    def update(self, t: float, wall: Optional[float] = None) -> Optional[str]:
        """Returns 'lost' / 'restored' on a transition, else None."""
        if self.last_beat is None:
            return None
        late = t - self.last_beat > self.timeout_s
        if (late and self.clock == "sim_and_wall" and wall is not None and self.last_beat_wall is not None
                and t >= self.drop_until):
            late = wall - self.last_beat_wall > self.timeout_s
        if self.state != "lost" and late:
            self.state, self.lost_since, self.needs_confirm = "lost", t, True
            self.losses += 1
            return "lost"
        if self.state == "lost" and not late:
            self.state, self.lost_since = "up", None
            return "restored"
        if self.state == "unknown":
            self.state = "up"
        return None

    def status(self, t: float) -> dict:
        """``action``: the hold action while lost; ``await_confirm`` after a restore until ``ack()`` (operator
        Resume / Halt, ORCHARD-MVP-SPEC 5.3: never resume on the restore alone); else ``none``. ``t`` stamps the
        status so a reader can tell a stale value."""
        lost = self.state == "lost"
        age = None if self.last_beat is None else round(t - self.last_beat, 3)
        action = self.action_on_loss if lost else ("await_confirm" if self.needs_confirm else "none")
        return {"t": round(t, 4), "link": self.name, "state": self.state, "age_s": age,
                "timeout_s": self.timeout_s, "lost_since": self.lost_since, "losses": self.losses,
                "action": action, "reason": self.reason if (lost or self.needs_confirm) else "",
                "needs_confirm": self.needs_confirm, "injected_drop": t < self.drop_until, "clock": self.clock}


class _Every:
    """Sim-time decimator for a publish rate."""

    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self.next = 0.0

    def due(self, t: float) -> bool:
        if t + 1e-9 >= self.next:
            self.next = (math.floor((t + 1e-9) / self.period) + 1) * self.period
            return True
        return False

    def reset(self):
        self.next = 0.0


class Platform:
    """All WP7 state for one MjModel. Created lazily by the first hook call."""

    def __init__(self, m):
        import mujoco
        self.mj = mujoco
        seed = int(os.environ.get("NANDI_SEED", "7") or 7)
        self.origin, cfg = gf.load_site()
        home = cfg.get("home", {}) or {}
        self.home = (float(home.get("x", 0.0)), float(home.get("y", 0.0)))
        self.params = em.EnergyParams.from_dict(cfg.get("energy"))
        self.battery = em.Battery(self.params)
        seed_coef = re_.Coefficients.seed(self.params, re_.SCAN_SPEED_MPS)
        card = os.environ.get("NANDI_ENERGY_CARD", "")
        self.coef_seed = re_.Coefficients.from_run_card(card, seed_coef) if card else seed_coef
        self.tracker = re_.CoefficientTracker(self.coef_seed)
        self.reserve = re_.ReserveMonitor(self.params.capacity_wh, self.params.reserve_pack)
        self.gnss = gm.GnssModel(self.origin, gm.GnssScript.from_dict(_env_json("NANDI_GNSS")), seed=seed)
        self.rtk = gm.RtkSupervisor()
        self.pose = gm.PoseEstimator(seed=seed)
        self.imu = im.ImuModel(seed=seed, enabled=os.environ.get("NANDI_IMU_NOISE", "1") != "0")
        self.weather = ws.WeatherStub(ws.WeatherScript.from_dict(_env_json("NANDI_WEATHER")))
        link_cfg = _env_json("NANDI_LINK") or {}
        clock = str(link_cfg.get("clock", "sim"))
        self.links = {k: LinkMonitor(k, float(link_cfg.get(f"{k}_timeout_s", v[0])), v[1], v[2], v[3], clock)
                      for k, v in LINKS.items()}
        self._wind_env = _env_json("NANDI_WIND")
        self.wind_script = wm.WindScript.from_dict(self._wind_env)
        self.anemo = wm.Anemometer()
        self.drag = wm.WindDrag(m)
        # model handles
        sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)  # noqa: E731
        self.antenna = next((s for s in (sid(n) for n in gm.ANTENNA_SITES) if s >= 0), -1)
        self.anemo_site = sid("anemometer")
        self.trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.sensor = {}
        for n in ("imu_quat", "imu_gyro", "imu_acc"):
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)
            if i >= 0:
                self.sensor[n] = (int(m.sensor_adr[i]), int(m.sensor_dim[i]))
        self.rates = {k: _Every(hz) for k, hz in (("imu", 50.0), ("gnss", 10.0), ("pose", 10.0), ("battery", 1.0),
                                                  ("energy", 1.0), ("link", 2.0), ("wind", 10.0), ("weather", 1.0),
                                                  ("route", 1.0 / ROUTE_SAMPLE_S))}
        self.reset_state()

    # ------------------------------------------------------------------ lifecycle
    def reset_state(self):
        self.battery.reset()
        self.battery.set_projection(None, None)
        self.tracker.reset()
        self.reserve.clear()
        self.gnss.reset()
        self.rtk.reset()
        self.pose.reset(self.home)
        self.imu.reset()
        self.weather.reset()
        for lk in self.links.values():
            lk.reset()
        # runtime injections (wind/cmd, weather/cmd temp + rain, gnss/cmd) are dropped: back to the configured scripts
        self.wind_script = wm.WindScript.from_dict(self._wind_env)
        self.anemo.reset()
        self.drag.clear(None)
        for r in self.rates.values():
            r.reset()
        self.e_step_wh = 0.0          # energy integrated by platform_energy since the last platform tick
        self.e_total_wh = 0.0
        self.p_last_w = self.params.p_idle_w
        self.route_m = 0.0
        self.items = 0
        self.crown_m = 0.0
        self._last_xy = None
        self._route_acc = 0.0
        self._last_tick_t = None
        self._last_ext = None
        self._items_seen = 0
        self.plan = None              # (t_received, RoutePlan, route_m_at, items_at)
        self.rtk_state = gm.NONE
        self.last_fix: Optional[dict] = None
        self._inbox_seen: Dict[str, int] = {}
        self._gust_prev = False
        self._item_active = False
        self._wind_last_t: Optional[float] = None
        self._dock_since: Optional[float] = None
        self.docked = False
        self.charges = 0
        self.rejected: Dict[str, str] = {}   # topic -> last rejected command error

    # ------------------------------------------------------------------ energy (every step)
    def energy_step(self, m, d):
        p_mech = float(abs(d.actuator_force * d.actuator_velocity).sum()) if m.nu else 0.0
        p = (p_mech if math.isfinite(p_mech) else 0.0) / self.params.eta + self.params.p_idle_w
        self.e_step_wh += p * m.opt.timestep / 3600.0
        self.p_last_w = p

    # ------------------------------------------------------------------ helpers
    def _arrived(self, bus, topic) -> bool:
        """True when a message arrived on ``topic`` since the last check (the bus delivery count, so a repeated
        value, ``1``, ``true``, ``null`` or ``{}`` is a new message too)."""
        n = bus.arrival_count(topic)
        if n == self._inbox_seen.get(topic, 0):
            return False
        self._inbox_seen[topic] = n
        return True

    def _alert(self, bus, t, code, **kw):
        bus.publish("/nandi/platform/alert", {"t": round(t, 4), "id": code, **kw})

    def remaining_plan(self) -> Optional[re_.RoutePlan]:
        if self.plan is None:
            return None
        t0, plan, r0, i0, c0, tnow = self.plan
        return re_.RoutePlan(route_m=max(0.0, plan.route_m - (self.route_m - r0)),
                             items=max(0, plan.items - (self.items - i0)),
                             t_work_s=max(0.0, plan.t_work_s - (tnow - t0)),
                             crown_m=max(0.0, plan.crown_m - (self.crown_m - c0)), return_m=plan.return_m)

    # ------------------------------------------------------------------ 50 Hz tick
    def tick(self, m, d, t, bus):
        self._inputs(t, bus)
        dt = 0.0 if self._last_tick_t is None else max(0.0, t - self._last_tick_t)
        self._last_tick_t = t
        # crown-mast travel (WP24 shares bus.values["mast"] = {"ext_m": ...}); kinematic mast: model energy
        mast = bus.values.get("mast") or {}
        ext = mast.get("ext_m") if isinstance(mast, dict) else None
        d_crown = 0.0
        if isinstance(ext, (int, float)) and math.isfinite(ext):
            if self._last_ext is not None:
                d_crown = abs(float(ext) - self._last_ext)
            self._last_ext = float(ext)
        e_crown = d_crown * re_.ASSUMED_CROWN_WH_PER_M
        de = self.e_step_wh + e_crown
        self.e_step_wh = 0.0
        self.e_total_wh += de
        self.crown_m += d_crown
        self.battery.consume(t, de, (de * 3600.0 / dt) if dt > 0 else self.p_last_w)
        # route metres (2 Hz trunk xy) and items (bus counter from task layers)
        xy = (float(d.qpos[0]), float(d.qpos[1]))
        d_route = 0.0
        if self.rates["route"].due(t):
            if self._last_xy is not None:
                d_route = math.hypot(xy[0] - self._last_xy[0], xy[1] - self._last_xy[1])
                self.route_m += d_route
            self._last_xy = xy
        items_now = bus.values.get("platform_items")
        d_items = 0
        if isinstance(items_now, int) and items_now > self._items_seen:
            d_items = items_now - self._items_seen
            self._items_seen = items_now
            self.items += d_items
        self.tracker.update(dt, d_route, d_items, d_crown, de)
        coef = self.tracker.coefficients()
        # reserve and projection
        est = (self.pose.x, self.pose.y)
        rem = self.remaining_plan()
        if rem is not None:
            self.plan = self.plan[:5] + (t,)
            rem = self.remaining_plan()
            prev = self.reserve.action
            act = self.reserve.update(t, self.battery.soc, rem, coef)
            chk = self.reserve.last
            self.battery.set_projection(chk.projected_end_soc, chk.reserve)
            if act != prev and act == re_.ACTION_RETURN_AND_WAIT:
                self._alert(bus, t, "low_battery_return", reason=re_.REASON_ENERGY_RESERVE,
                            projected_end_soc=round(chk.projected_end_soc, 4), reserve=round(chk.reserve, 4))
        else:
            # no plan (no mission layer sends one yet): the remaining route is empty, so the projected end SoC is
            # the SoC now, checked against the walk home from the pose estimate + 15 % (MVP-SPEC 5.6, OS 5.3)
            ret = math.hypot(est[0] - self.home[0], est[1] - self.home[1])
            prev = self.reserve.action
            act = self.reserve.update(t, self.battery.soc, re_.RoutePlan(return_m=ret), coef)
            chk = self.reserve.last
            self.battery.set_projection(None, chk.reserve)
            if act != prev and act == re_.ACTION_RETURN_AND_WAIT:
                self._alert(bus, t, "low_battery_return", reason=re_.REASON_ENERGY_RESERVE,
                            projected_end_soc=round(chk.projected_end_soc, 4), reserve=round(chk.reserve, 4),
                            plan=False)
        self._dock(d, t, bus, xy)
        self._gnss_pose(m, d, t, bus, xy)
        self._imu(m, d, t, bus)
        self._links(t, bus)
        self._weather(t, bus)
        bat = {**self.battery.status(), "docked": self.docked, "charges": self.charges}
        bus.values["platform"] = {"t": t, "soc": bat["soc"], "docked": self.docked,
                                  "projected_end_soc": bat["projected_end_soc"],
                                  "reserve": bat["reserve"], "energy_action": self.reserve.action,
                                  "rtk": self.rtk_state, "rtk_action": self.rtk.action,
                                  "pose": [round(self.pose.x, 4), round(self.pose.y, 4)]}
        if self.rates["battery"].due(t):
            bus.publish("/nandi/platform/battery", {**bat, "t": round(t, 3)})
        if self.rates["energy"].due(t):
            bus.publish("/nandi/energy", {"t": round(t, 3), "integrated_wh": round(self.e_total_wh, 5),
                                          "route_m": round(self.route_m, 3), "items": self.items,
                                          "crown_m": round(self.crown_m, 3), "coefficients": coef.to_dict(),
                                          "reserve_monitor": self.reserve.status(),
                                          "plan_remaining": None if rem is None else rem.__dict__,
                                          "assumed": True})

    def _dock(self, d, t, bus, xy):
        """Same rule as mission_executor (one battery story for the console): parked within DOCK_RADIUS_M of home
        with no plan for DOCK_DWELL_S -> on the (ASSUMED) charger: the pack is full and the reserve latch clears."""
        speed = math.hypot(float(d.qvel[0]), float(d.qvel[1]))
        at_home = math.hypot(xy[0] - self.home[0], xy[1] - self.home[1]) <= DOCK_RADIUS_M
        if self.plan is None and at_home and speed < DOCK_SPEED_MPS:
            if self._dock_since is None:
                self._dock_since = t
            if t - self._dock_since >= DOCK_DWELL_S - 1e-9:
                if not self.docked:
                    self.docked = True
                    if self.battery.soc < 0.9995:
                        self.charges += 1
                        self._alert(bus, t, "docked_charged", soc_before=round(self.battery.soc, 4))
                    self.reserve.clear()
                self.battery.charge_full(t)          # held full while docked (idle drain from the charger)
        else:
            self._dock_since = None
            self.docked = False

    def _reject(self, bus, t, topic, exc):
        err = f"{type(exc).__name__}: {exc}"
        self.rejected[topic] = err
        self._alert(bus, t, "cmd_rejected", topic=topic, error=err[:200])

    def _inputs(self, t, bus):
        """Heartbeats, then each command topic in its own guard: a malformed command is rejected (alert
        ``cmd_rejected``, state unchanged) and never disables the hook (the link watchdogs keep running)."""
        wall = time.monotonic()
        for k in LINKS:
            if self._arrived(bus, TOPIC_HB[k]):
                self.links[k].beat(t, wall)
        for topic, fn in (("/nandi/link/cmd", self._cmd_link), ("/nandi/wind/cmd", self._cmd_wind),
                          ("/nandi/weather/cmd", self._cmd_weather), ("/nandi/gnss/cmd", self._cmd_gnss),
                          ("/nandi/platform/plan", self._cmd_plan)):
            if not self._arrived(bus, topic):
                continue
            cmd = bus.latest(topic)
            try:
                if not isinstance(cmd, dict):
                    raise TypeError("command must be a JSON object")
                fn(t, cmd)
            except Exception as exc:  # noqa: BLE001 - reject the command, keep the hook
                self._reject(bus, t, topic, exc)

    def _cmd_link(self, t, cmd):
        drop, ack = cmd.get("drop"), cmd.get("ack")
        dur = 0.0
        if drop is not None:
            if not isinstance(drop, dict) or drop.get("link") not in self.links:
                raise ValueError("drop needs an object with link local or cloud")
            dur = wm._num(drop.get("dur", 1.0))
            if dur < 0:
                raise ValueError("drop dur must be >= 0")
        if ack is not None and ack not in self.links:
            raise ValueError("ack must be local or cloud")
        if drop is not None:
            self.links[drop["link"]].drop(t, dur)
        if ack is not None:
            self.links[ack].ack()

    def _cmd_wind(self, t, cmd):
        cmd = dict(cmd)
        gusts = cmd.get("gusts")
        if gusts is not None:
            if not isinstance(gusts, list) or not all(isinstance(g, dict) for g in gusts):
                raise TypeError("gusts must be a list of objects")
            # gust t0 is relative to arrival; new dicts, so the delivered payload on the bus is not modified
            cmd["gusts"] = [dict(g, t0=wm._num(g.get("t0", 0.0)) + t) for g in gusts]
        self.wind_script = wm.WindScript.from_dict(cmd)     # strict: raises before anything changes

    def _cmd_weather(self, t, cmd):
        rain = cmd.get("rain")
        ev = None
        if rain is not None:
            if not isinstance(rain, dict):
                raise TypeError("rain must be an object")
            ev = ws.RainEvent(t + wm._num(rain.get("t0", 0.0)), wm._num(rain.get("dur", 600.0)),
                              wm._num(rain.get("mm_h", 5.0)))          # validates
        if "temp_c" in cmd:
            temp = cmd["temp_c"]
            self.weather.set_temp(None if temp is None else wm._num(temp))
        if ev is not None:
            self.weather.set_rain(ev.t0, ev.dur, ev.mm_h)

    def _cmd_gnss(self, t, cmd):
        dr = cmd.get("dropout")
        if dr is None:
            return
        if not isinstance(dr, dict):
            raise TypeError("dropout must be an object")
        t0, dur = wm._num(dr.get("t0", 0.0)), wm._num(dr.get("dur", 10.0))
        state = dr.get("state", gm.NONE)
        if t0 < 0 or dur <= 0:
            raise ValueError("dropout needs t0 >= 0 and dur > 0")
        if state not in gm.STATES:
            raise ValueError(f"dropout state must be one of {gm.STATES}")
        self.gnss.inject(t + t0, dur, state)

    def _cmd_plan(self, t, plan):
        wp = plan.get("waypoint")
        if wp is not None and not (isinstance(wp, (list, tuple)) and len(wp) >= 2
                                   and all(wm._finite(v) for v in wp[:2])):
            raise ValueError("waypoint must be [x, y]")
        new = None
        if not plan.get("clear"):
            for k in ("route_m", "t_work_s", "crown_m", "return_m"):
                if k in plan:
                    wm._num(plan[k])
            items = plan.get("items", 0)
            if not isinstance(items, int) or isinstance(items, bool):
                raise ValueError("items must be an integer")
            new = (t, re_.RoutePlan.from_dict(plan), self.route_m, self.items, self.crown_m, t)
        if new is None:
            self.plan = None
            self.reserve.clear()
        else:
            self.plan = new
        if wp is not None:
            self.rtk.waypoint(list(wp), self.rtk_state)
        self._item_active = bool(plan.get("item_active", False))

    def _gnss_pose(self, m, d, t, bus, xy):
        if self.antenna >= 0:
            p = d.site_xpos[self.antenna]
            ax, ay, az = float(p[0]), float(p[1]), float(p[2])
        else:
            ax, ay, az = xy[0], xy[1], float(d.qpos[2])
        self.pose.predict(xy)
        st = self.gnss.state_at(t)
        prev = self.rtk_state
        self.rtk_state = st
        if self.rates["gnss"].due(t):
            fix = self.gnss.sample(t, ax, ay, az)
            self.last_fix = fix
            bus.publish("/nandi/gnss/fix", fix)
            loc = fix["local"]
            # the antenna sits above the trunk: the fix is corrected back to the trunk xy offset (rigid, known)
            fix_xy = None if loc is None else (loc[0] - (ax - xy[0]), loc[1] - (ay - xy[1]))
            self.pose.correct(fix_xy, fix["sigma_h_m"], st)
        prev_action = self.rtk.action
        self.rtk.observe((self.pose.x, self.pose.y), st)
        act = self.rtk.update(t, st, self._item_active)
        if st != prev and prev == gm.FIXED:
            self._alert(bus, t, "rtk_lost", rtk=st, reason=gm.REASON_RTK)
        if st == gm.FIXED and prev != gm.FIXED and prev_action != gm.A_CONTINUE:
            self._alert(bus, t, "rtk_recovered")
        if act != prev_action and act == gm.A_RETURN_ODOM:
            self._alert(bus, t, "rtk_return_odometry", target=self.rtk.return_target, reason=gm.REASON_RTK)
        if self.rates["pose"].due(t):
            yaw = _yaw(d.qpos[3:7])
            bus.publish("/nandi/pose_estimate", {**self.pose.estimate(t, yaw), "rtk": st,
                                                 "rtk_supervisor": self.rtk.status(t)})

    def _imu(self, m, d, t, bus):
        if not self.sensor or not self.rates["imu"].due(t):
            return

        def read(n):
            a, k = self.sensor[n]
            return [float(v) for v in d.sensordata[a:a + k]]
        if all(n in self.sensor for n in ("imu_quat", "imu_gyro", "imu_acc")):
            bus.publish("/nandi/imu", self.imu.sample(t, read("imu_quat"), read("imu_gyro"), read("imu_acc")))

    def _links(self, t, bus):
        changed = False
        for k, lk in self.links.items():
            ev = lk.update(t, time.monotonic())
            if ev == "lost":
                self._alert(bus, t, lk.alert, link=k, reason=lk.reason, action=lk.action_on_loss)
                changed = True
            elif ev == "restored":
                self._alert(bus, t, "link_restored", link=k, needs_confirm=lk.needs_confirm)
                changed = True
        bus.values["link"] = {k: lk.status(t) for k, lk in self.links.items()}
        if changed or self.rates["link"].due(t):
            for k, lk in self.links.items():
                bus.publish(TOPIC_LINK[k], {"t": round(t, 3), **lk.status(t)})

    def _weather(self, t, bus):
        st = self.weather.state(t)
        bus.values["weather"] = st
        if self.rates["weather"].due(t):
            bus.publish("/nandi/weather", st)

    # ------------------------------------------------------------------ wind (50 Hz)
    def wind_tick(self, m, d, t, bus):
        """Any exception removes this hook's drag from xfrc_applied before it propagates (the registry then disables
        the hook, so no stale force is left in physics)."""
        try:
            self._wind_tick(m, d, t, bus)
        except BaseException:
            self.drag.clear(d)
            raise

    def _wind_tick(self, m, d, t, bus):
        if self._wind_last_t is not None and t < self._wind_last_t - 1e-9:
            # sim time went back without hooks.reset (MuJoCo's own auto-reset on a bad QACC zeroes xfrc_applied):
            # the bookkeeping is forgotten, not subtracted
            self.drag.clear(None)
            self.anemo.reset()
        self._wind_last_t = t
        z = None
        if self.anemo_site >= 0:
            z = float(d.site_xpos[self.anemo_site][2])
        w = self.wind_script.velocity(t, z)
        st = self.anemo.update(t, w)
        if not self.wind_script.calm or self.drag._applied:
            st["force_n"] = round(self.drag.apply(m, d, self.wind_script, t), 4)
        else:
            st["force_n"] = 0.0
        st["drag_bodies"] = self.drag.body_names
        bus.values["wind"] = st
        if st["gust"] and not self._gust_prev:
            self._alert(bus, t, "gust", speed_mps=st["speed_mps"], mean_mps=st["mean_mps"], reason="wind")
        changed = st["gust"] != self._gust_prev
        self._gust_prev = st["gust"]
        if changed or self.rates["wind"].due(t):
            bus.publish("/nandi/wind", st)


def _yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


_STATE: Dict[int, Platform] = {}


def _get(m) -> Platform:
    key = id(m)
    p = _STATE.get(key)
    if p is None:
        p = _STATE[key] = Platform(m)
    return p


def _energy(m, d, t, bus):
    _get(m).energy_step(m, d)


def _tick(m, d, t, bus):
    _get(m).tick(m, d, t, bus)


def _wind(m, d, t, bus):
    _get(m).wind_tick(m, d, t, bus)


def _reset(m, d):
    if m is None:
        return
    p = _STATE.get(id(m))
    if p is not None:
        p.reset_state()   # mj_resetData* zeroed xfrc_applied: the drag bookkeeping is forgotten, not subtracted


def register(hooks):
    hooks.register("platform_energy", _energy, every_step=True, reset=_reset)
    hooks.register("platform", _tick, rate_hz=50.0, topics_in=TOPICS_IN)
    hooks.register("wind", _wind, rate_hz=50.0)
