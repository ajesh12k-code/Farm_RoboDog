#!/usr/bin/env python3
"""Atharv Nandi run recorder: one run directory of evidence per session, filled live.

    runs/<run_id>/              run_id = <utc yyyymmddThhmmssZ>-<short git sha>-<NANDI_SEED>
        meta.json               run_id, device_id, git_sha, dirty, sw, config sha256s, NANDI_*, start_utc
        kis_v1.jsonl            kis/v1 messages (atharv_nandi/nodes/telemetry_bridge.py, running in this process)
        harvest_attempts.jsonl  one line per auto_harvest attempt; success = basket truth
        events.jsonl            falls, world resets, interventions, e-stops
        joint_effort_summary.json  peak/RMS |effort| and |velocity| vs MJCF forcerange (needs /nandi/joint_states)
        manifest.json           SHA-256 per file + Merkle root (refreshed every summary_period_s)
    runs/CURRENT                the run_id being written

Publishes /nandi/run_status (std_msgs/String JSON, 1 Hz) with the run_id and live KPI counters for
the consoles. Start it next to the stack:  python3 -m atharv_nandi.nodes run_recorder
"""
import datetime as _dt
import json
import os
import time
from typing import Any, Dict, List, Optional

import atharv_nandi.nodes.telemetry_bridge as tb  # noqa: E402
from atharv_nandi.core import manifest as mf  # noqa: E402
from atharv_nandi.core import model_layout as ml_  # noqa: E402
from atharv_nandi.core import run_log as rl  # noqa: E402

REPO = tb.REPO
MISSION_RUNNING_STATES = ("running", "active", "executing", "harvesting")


def crop_positions(qpos, layout=None) -> Dict[str, List[float]]:
    """{crop_item_k: [x, y, z]} of the legacy crops. With a ``model_layout.Layout`` (or the current one of this
    process, when its nq matches) appended module joints are never counted as crops (WP0)."""
    return ml_.legacy_crop_positions(qpos, layout)


def estop_engaged(d: Dict[str, Any]) -> bool:
    for key in ("estop", "e_stop", "estop_engaged"):
        if isinstance(d.get(key), bool):
            return d[key]
    return str(d.get("state", "")).lower() in ("estop", "e_stop", "emergency_stop")


class Recorder:
    """Owns the run directory. Feed it through hooks(); call tick(now) often and close() once."""

    def __init__(self, run_dir: str, meta: Dict[str, Any], bridge: Optional[tb.TelemetryBridge] = None,
                 force_limits: Optional[Dict[str, float]] = None, summary_period_s: float = 5.0,
                 publish_status=None, wall_ms=lambda: int(time.time() * 1000), layout=None):
        self.dir, self.meta, self.bridge = run_dir, meta, bridge
        self.layout = layout
        self.force_limits = force_limits or {}
        self.summary_period_s = summary_period_s
        self.publish_status = publish_status
        self.wall_ms = wall_ms
        self.tracker = rl.AttemptTracker(sw=meta.get("sw", ""), run_id=meta["run_id"],
                                         device_id=meta.get("device_id", rl.DEVICE_ID), wall_ms=wall_ms)
        self.interventions = rl.InterventionDetector()
        self.estops = rl.EdgeCounter()
        self.effort = rl.EffortStats()
        self.attempts_log = rl.JsonlWriter(os.path.join(run_dir, "harvest_attempts.jsonl"))
        self.events_log = rl.JsonlWriter(os.path.join(run_dir, "events.jsonl"))
        self.successes = 0
        self.attempts = 0
        self.falls = 0
        self._falls_seen = 0
        self._last_elapsed = None
        self._harvest_active = False
        self._mission_active = False
        self._last_summary = -1e9
        self._last_status = -1e9
        self._t0 = None
        self.joint_states_seen = False

    # ------------------------------------------------------------------ inputs
    def hooks(self) -> Dict[str, Any]:
        return {"qpos": self.on_qpos, "harvest": self.on_harvest, "game": self.on_game,
                "mission": self.on_mission, "safety": self.on_safety, "joint_states": self.on_joint_states,
                "cmd_vel": self.on_manual("cmd_vel"), "manual_arm": self.on_manual("manual_arm"),
                "tick": self.tick}

    def event(self, now: float, kind: str, **detail):
        self.events_log.write({"t": round(now - (self._t0 or now), 3), "utc_ms": self.wall_ms(), "kind": kind,
                               "run_id": self.meta["run_id"], "device_id": self.meta.get("device_id"), **detail})

    def on_qpos(self, now: float, q):
        self.tracker.crop_positions(crop_positions(q, self.layout))

    def _update_mission_flag(self):
        self.interventions.mission_active = self._harvest_active or self._mission_active

    def on_harvest(self, now: float, h: Dict[str, Any]):
        self._harvest_active = h.get("mode") == rl.ARM_HARVEST_MODE
        self._update_mission_flag()
        self.tracker.harvest_status(now, h)

    def on_mission(self, now: float, d: Dict[str, Any]):
        self._mission_active = str(d.get("state", "")).lower() in MISSION_RUNNING_STATES
        self._update_mission_flag()

    def on_game(self, now: float, g: Dict[str, Any]):
        if self._last_elapsed is not None and g["elapsed_s"] + 0.5 < self._last_elapsed:
            self.event(now, "world_reset", elapsed_before_s=self._last_elapsed)
            self._falls_seen = 0
        self._last_elapsed = g["elapsed_s"]
        if g["falls"] > self._falls_seen:
            for _ in range(g["falls"] - self._falls_seen):
                self.falls += 1
                self.event(now, "fall", sim_elapsed_s=g["elapsed_s"])
        self._falls_seen = g["falls"]
        self.tracker.game_status(now, g)
        self._drain(now)

    def on_safety(self, now: float, d: Dict[str, Any]):
        if self.estops.update(estop_engaged(d)):
            self.event(now, "estop", detail={k: d[k] for k in list(d)[:8]})

    def on_joint_states(self, now: float, names, effort, velocity):
        self.joint_states_seen = True
        self.effort.add(names, effort, velocity)

    def on_manual(self, source: str):
        def fn(now: float, nonzero: bool = True):
            if self.interventions.manual(now, source, nonzero):
                self.event(now, "intervention", source=source)
        return fn

    # ------------------------------------------------------------------ outputs
    def _drain(self, now: float, attempts: Optional[List[Dict[str, Any]]] = None):
        for a in (attempts if attempts is not None else self.tracker.poll(now)):
            self.attempts += 1
            self.successes += int(a["success"])
            self.attempts_log.write(a)
            if self.bridge is not None:
                self.bridge.on_attempt(a)

    def live_status(self, now: float) -> Dict[str, Any]:
        hours = max(1e-9, (now - (self._t0 or now)) / 3600.0)
        return {
            "run_id": self.meta["run_id"], "device_id": self.meta.get("device_id"), "sw": self.meta.get("sw"),
            "run_dir": os.path.relpath(self.dir, REPO),
            "elapsed_s": round(now - (self._t0 or now), 1),
            "attempts": self.attempts, "picks": self.successes,
            "pick_success_rate": round(self.successes / self.attempts, 3) if self.attempts else None,
            "picks_per_hour": round(self.successes / hours, 2) if self._t0 and now > self._t0 else 0.0,
            "falls": self.falls, "estops": self.estops.count, "interventions": self.interventions.count,
            "kis_messages": sum(getattr(s, "count", 0) for s in (self.bridge.sinks if self.bridge else [])),
            "data_provenance": "simulated",
        }

    def write_summaries(self, final: bool = False):
        if self.joint_states_seen:
            summary = self.effort.summary(self.force_limits, source="/nandi/joint_states")
        else:
            summary = {"source": None, "samples": 0, "actuators": {},
                       "note": "no /nandi/joint_states received (safety package not running); "
                               "use tools/headless_eval.py for a sim torque report"}
        summary.update({"run_id": self.meta["run_id"], "device_id": self.meta.get("device_id"),
                        "date": _dt.datetime.now(_dt.timezone.utc).date().isoformat()})
        rl.write_json(os.path.join(self.dir, "joint_effort_summary.json"), summary)
        mf.write_manifest(self.dir, self.meta, final=final, closed_ms=self.wall_ms() if final else None)

    def tick(self, now: float):
        if self._t0 is None:
            self._t0 = now
        self._drain(now)
        if now - self._last_status >= 1.0:
            self._last_status = now
            if self.publish_status:
                self.publish_status(self.live_status(now))
        if now - self._last_summary >= self.summary_period_s:
            self._last_summary = now
            self.write_summaries()

    def close(self, now: float):
        self._drain(now, self.tracker.flush(now))
        self.event(now, "run_end", attempts=self.attempts, picks=self.successes)
        self.meta["end_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")
        rl.write_json(os.path.join(self.dir, "meta.json"), self.meta)
        self.attempts_log.close()
        self.events_log.close()
        if self.bridge is not None:
            self.bridge.stop()
        self.write_summaries(final=True)


def open_run(cfg: Dict[str, Any], config_path: str, repo: str = REPO, env=None) -> Dict[str, Any]:
    env = os.environ if env is None else env
    sha, dirty = rl.git_sha(repo), rl.git_dirty(repo)
    run_id = env.get("NANDI_RUN_ID") or rl.make_run_id(rl.utc_stamp(), (sha or "nogit")[:7],
                                                       env.get("NANDI_SEED", "7"))
    runs_dir = os.path.join(repo, (cfg.get("run", {}) or {}).get("runs_dir", "runs"))
    run_dir = os.path.join(runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    ident = cfg.get("identity", {}) or {}
    meta = rl.build_meta(run_id, repo, [config_path], env=env, device_id=ident.get("device", rl.DEVICE_ID),
                         extra={"tenant": ident.get("tenant"), "site": ident.get("site"),
                                "start_utc_ms": int(time.time() * 1000)})
    rl.write_json(os.path.join(run_dir, "meta.json"), meta)
    with open(os.path.join(runs_dir, "CURRENT"), "w", encoding="utf-8") as fh:
        fh.write(run_id + "\n")
    return {"run_id": run_id, "run_dir": run_dir, "meta": meta}


def force_limits_from_model() -> Dict[str, float]:
    return model_and_limits()[1]


def model_and_limits():
    """(Layout | None, force limits) of the live world: one model build serves both."""
    try:
        from atharv_nandi.core.farm_scene import load_farm_model
        model = load_farm_model()
        return ml_.current(), rl.mjcf_force_limits(model)
    except Exception:  # MuJoCo missing: no layout (pre-WP0 crop rule), and the summary has no % of limit
        return None, {}


def main():
    import rclpy
    from rclpy.executors import ExternalShutdownException
    tb.install_sigterm()
    from std_msgs.msg import String
    config_path = os.environ.get("NANDI_TELEMETRY_CONFIG", tb.DEFAULT_CONFIG)
    cfg = tb.load_config(config_path)
    run = open_run(cfg, config_path)
    meta, run_dir = run["meta"], run["run_dir"]
    mapper = tb.make_mapper(cfg, meta["sw"])
    meta["boot"] = mapper.boot
    meta["kis_will"] = mapper.will()
    rl.write_json(os.path.join(run_dir, "meta.json"), meta)

    rclpy.init()
    bridge = tb.TelemetryBridge(mapper, tb.make_sinks(cfg, run_dir, mapper), cfg, run["run_id"])
    holder: Dict[str, Any] = {}
    layout, limits = model_and_limits()
    recorder = Recorder(run_dir, meta, bridge, limits,
                        float((cfg.get("run", {}) or {}).get("summary_period_s", 5.0)),
                        publish_status=lambda d: holder["pub"].publish(String(data=json.dumps(d))), layout=layout)
    node = tb.make_bridge_node(bridge, recorder.hooks(), name="nandi_run_recorder")
    holder["pub"] = node.create_publisher(String, "/nandi/run_status", 10)
    bridge.log = node.get_logger().warn
    bridge.start()
    node.get_logger().info(f"Recording run {run['run_id']} -> {run_dir}")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        recorder.close(time.monotonic())
        node.get_logger().info(f"Run {run['run_id']} closed: {recorder.attempts} attempts, {recorder.successes} picks")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
