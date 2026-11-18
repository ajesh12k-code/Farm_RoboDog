#!/usr/bin/env python3
"""Atharv Nandi -> Kisaanu kis/v1 telemetry bridge (simulated device; file sink by default).

Subscribes (all std_msgs/String JSON unless noted):
    /nandi/sim_qpos (Float64MultiArray)   -> tele/pose at rates_hz.pose
    /nandi/game_status, /nandi/harvest_status, /nandi/locomotion_status
                                          -> tele/mission at rates_hz.mission and on change; state
    /nandi/mission_status   (when present) -> tele/mission state
    /nandi/battery          (when present) -> tele/batt  (String JSON or sensor_msgs/BatteryState)
    /nandi/crop_detections  (when present) -> tele/detect
    /nandi/safety_status    (when present) -> state.safety
Every message goes to runs/<run_id>/kis_v1.jsonl as {"topic", "retain", "payload"}; an MQTT sink is
available behind config (sinks.mqtt.enabled, default false). Mapping rules: atharv_nandi/core/kis_mapper.py.

The pure part (TelemetryBridge) has no ROS in it; BridgeNode wires it to rclpy. Usually started via
atharv_nandi/nodes/run_recorder.py, which adds the per-attempt harvest log, KPIs and the manifest to the same run.
Standalone: python3 -m atharv_nandi.nodes telemetry_bridge  (creates its own run directory).
"""
import json
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional

from atharv_nandi import node_params  # noqa: E402
from atharv_nandi import paths
from atharv_nandi.core import kis_mapper as km  # noqa: E402
from atharv_nandi.core import telemetry as tm  # noqa: E402

REPO = paths.REPO
DEFAULT_CONFIG = os.path.join(REPO, "config", "telemetry.yaml")


# ---------------------------------------------------------------------- config + sinks
def load_config(path: str = DEFAULT_CONFIG) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml
    except ImportError:  # the ROS-free CI deps have no PyYAML; the config is a plain nested map
        return mini_yaml(text)
    return yaml.safe_load(text) or {}


def _scalar(v: str) -> Any:
    v = v.strip()
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    if v in ("null", "~", ""):
        return None
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def mini_yaml(text: str) -> Dict[str, Any]:
    """Nested `key: value` maps with # comments (all config/telemetry.yaml uses)."""
    root: Dict[str, Any] = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if val.strip():
            parent[key.strip()] = _scalar(val)
        else:
            parent[key.strip()] = child = {}
            stack.append((indent, child))
    return root


class FileSink:
    """JSONL file sink: one {"topic", "retain", "payload"} object per line, flushed per line."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")
        self.count = 0

    def send(self, topic: str, payload: Any, retain: bool):
        self._fh.write(json.dumps({"topic": topic, "retain": retain, "payload": payload},
                                  separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n")
        self._fh.flush()
        self.count += 1

    def close(self):
        self._fh.close()


class MqttSink:
    """MQTT 5 sink (optional). paho-mqtt is imported here, only when the sink is enabled.

    Broker URL and credentials come from the environment variables named in config; nothing about
    a broker is stored in the repository.
    """

    def __init__(self, cfg: Dict[str, Any], mapper: km.KisMapper):
        import paho.mqtt.client as mqtt  # optional dependency
        from urllib.parse import urlparse
        url = os.environ.get(cfg.get("url_env", "NANDI_MQTT_URL"), "")
        if not url:
            raise RuntimeError(f"MQTT sink enabled but ${cfg.get('url_env', 'NANDI_MQTT_URL')} is not set")
        u = urlparse(url)
        transport = "websockets" if u.scheme in ("ws", "wss") else "tcp"
        self.client = mqtt.Client(client_id=mapper.device, protocol=mqtt.MQTTv5, transport=transport)
        user = os.environ.get(cfg.get("username_env", "NANDI_MQTT_USERNAME"))
        if user:
            self.client.username_pw_set(user, os.environ.get(cfg.get("password_env", "NANDI_MQTT_PASSWORD")))
        if u.scheme in ("wss", "mqtts"):
            self.client.tls_set()
        if transport == "websockets":
            self.client.ws_set_options(path=u.path or "/mqtt")
        will = mapper.will()
        props = None
        try:
            from paho.mqtt.packettypes import PacketTypes
            from paho.mqtt.properties import Properties
            props = Properties(PacketTypes.WILLMESSAGE)
            props.WillDelayInterval = will["will_delay_s"]
        except ImportError:
            pass
        self.client.will_set(will["topic"], will["payload"], qos=will["qos"], retain=will["retain"],
                             properties=props)
        port = u.port or (443 if u.scheme == "wss" else 80 if u.scheme == "ws" else 8883 if u.scheme == "mqtts" else 1883)
        self.client.connect(u.hostname, port, keepalive=int(cfg.get("keepalive_s", 60)))
        self.client.loop_start()
        self.count = 0

    def send(self, topic: str, payload: Any, retain: bool):
        self.client.publish(topic, km.encode(payload), qos=1, retain=retain)
        self.count += 1

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()


def make_sinks(cfg: Dict[str, Any], run_dir: str, mapper: km.KisMapper) -> List[Any]:
    sinks_cfg = cfg.get("sinks", {}) or {}
    sinks: List[Any] = []
    if (sinks_cfg.get("file") or {}).get("enabled", True):
        sinks.append(FileSink(os.path.join(run_dir, "kis_v1.jsonl")))
    if (sinks_cfg.get("mqtt") or {}).get("enabled", False):
        sinks.append(MqttSink(sinks_cfg["mqtt"], mapper))
    return sinks


def make_mapper(cfg: Dict[str, Any], sw: str, boot: Optional[int] = None) -> km.KisMapper:
    ident = cfg.get("identity", {}) or {}
    return km.KisMapper(ident.get("tenant", "sim-nandi"), ident.get("site", "sim-field-01"),
                        ident.get("device", "nandi-sim-01"),
                        boot=boot if boot is not None else random.getrandbits(32),
                        origin=cfg.get("site_origin"), sw=sw)


# ---------------------------------------------------------------------- pure bridge
class TelemetryBridge:
    """Rate-limits and maps /nandi/* state to kis/v1 messages. Feed it, then call tick(now)."""

    def __init__(self, mapper: km.KisMapper, sinks: List[Any], cfg: Dict[str, Any], run_id: str = "",
                 wall_ms: Callable[[], int] = lambda: int(time.time() * 1000), log: Callable[[str], None] = print):
        self.m, self.sinks, self.run_id, self.wall_ms, self.log = mapper, sinks, run_id, wall_ms, log
        rates = cfg.get("rates_hz", {}) or {}
        self.pose_period = 1.0 / max(0.1, float(rates.get("pose", 2.0)))
        self.mission_period = 1.0 / max(0.1, float(rates.get("mission", 1.0)))
        self.qpos: Optional[List[float]] = None
        # simulated instant in epoch ms for tele/pose sim.t (broker contract section 7): anchor + game elapsed_s
        self.sim_t_ms: Optional[int] = None
        self._sim_anchor_ms: Optional[float] = None
        self._sim_elapsed: Optional[float] = None
        self.state_min_interval_s = float((cfg.get("state", {}) or {}).get("min_interval_s", 1.0))
        self._last_state_s = -1e9
        self._state_pending = False
        self.harvest = tm.parse_harvest_status(None)
        self.game = tm.parse_game_status(None)
        self.loco = tm.parse_locomotion_status(None)
        self.mission_status: Optional[Dict[str, Any]] = None
        self.safety: Optional[Dict[str, Any]] = None
        self._last_pose = self._last_mission = -1e9
        self._mission_key = None
        self._state_key = None
        self.dropped = 0  # payloads refused by the mapper (size, shape)

    # -- emit
    def _send(self, topic: str, payload: Any):
        retain = km.retain_for(topic)
        for s in self.sinks:
            try:
                s.send(topic, payload, retain)
            except Exception as exc:  # a sink failure must not stop the others or the run
                self.log(f"sink {type(s).__name__} failed: {exc}")

    def _emit(self, build: Callable[[], Any]):
        try:
            topic, payload = build()
        except (km.PayloadTooLarge, km.BadSlug, ValueError) as exc:
            self.dropped += 1
            self.log(f"kis/v1 payload refused: {exc}")
            return None
        self._send(topic, payload)
        return topic, payload

    def start(self):
        self._send(*self.m.status(True))
        self.emit_state(force=True)

    def stop(self):
        self.emit_state(force=True)
        self._send(*self.m.status(False))
        for s in self.sinks:
            try:
                s.close()
            except Exception:
                pass

    # -- inputs
    def on_qpos(self, qpos):
        self.qpos = list(qpos)  # sim.t comes from on_game / set_sim_elapsed, never reset here

    def set_sim_elapsed(self, elapsed_s: float):
        """Sim seconds since the world (re)started -> sim.t epoch ms. The anchor is fixed on the first
        value and re-taken when elapsed goes backwards (reset_world), so sim.t never runs backwards."""
        try:
            elapsed = float(elapsed_s)
        except (TypeError, ValueError):
            return
        if elapsed != elapsed or elapsed in (float("inf"), float("-inf")):
            return
        if self._sim_anchor_ms is None or (self._sim_elapsed is not None and elapsed + 0.5 < self._sim_elapsed):
            self._sim_anchor_ms = self.wall_ms() - elapsed * 1000.0
        self._sim_elapsed = elapsed
        self.sim_t_ms = int(round(self._sim_anchor_ms + elapsed * 1000.0))

    def on_harvest(self, h: Dict[str, Any], now: float):
        self.harvest = h
        self._maybe_mission(now)
        self.emit_state()

    def on_game(self, g: Dict[str, Any], now: float):
        self.game = g
        if "elapsed_s" in g:
            self.set_sim_elapsed(g["elapsed_s"])
        self._maybe_mission(now)

    def on_loco(self, loco: Dict[str, Any]):
        self.loco = loco
        self.emit_state()

    def on_mission_status(self, d: Dict[str, Any], now: float):
        self.mission_status = d
        self._maybe_mission(now)

    def on_safety(self, d: Dict[str, Any]):
        self.safety = d
        self.emit_state()

    def on_battery(self, d: Dict[str, Any]):
        self._emit(lambda: self.m.batt(self.wall_ms(), d))

    def on_detections(self, d: Any):
        dets = d.get("detections", []) if isinstance(d, dict) else d
        model = d.get("model") if isinstance(d, dict) else None
        if isinstance(dets, list):
            self._emit(lambda: self.m.detect(self.wall_ms(), [x for x in dets if isinstance(x, dict)], model))

    def on_attempt(self, attempt: Dict[str, Any]):
        """PROPOSED tele/harvest: one finalised attempt from the recorder."""
        slim = {k: v for k, v in attempt.items() if k not in ("phases",)}
        self._emit(lambda: self.m.harvest(self.wall_ms(), slim))

    # -- periodic
    def tick(self, now: float):
        if self.qpos is not None and now - self._last_pose >= self.pose_period:
            # fixed cadence: advance by one period (no drift from tick jitter), resync if far behind
            self._last_pose = now if now - self._last_pose > 2 * self.pose_period else self._last_pose + self.pose_period
            q = self.qpos
            self._emit(lambda: self.m.pose(self.wall_ms(), q, self.sim_t_ms))
        if now - self._last_mission >= self.mission_period:
            self._mission(now)
        if self._state_pending:
            self.emit_state()

    def _mission_body(self) -> Dict[str, Any]:
        return km.mission_summary(self.harvest, self.game, self.mission_status, self.run_id)

    def _maybe_mission(self, now: float):
        body = self._mission_body()
        key = (body["state"], body["phase"], body["target"], body["picked"], body["falls"], body["fallen"])
        if key != self._mission_key:
            self._mission(now, body)

    def _mission(self, now: float, body: Optional[Dict[str, Any]] = None):
        body = body or self._mission_body()
        self._mission_key = (body["state"], body["phase"], body["target"], body["picked"], body["falls"],
                             body["fallen"])
        self._last_mission = now
        self._emit(lambda: self.m.mission(self.wall_ms(), body))

    def emit_state(self, force: bool = False):
        """Retained state on change (broker contract section 2). Only discrete safety fields (bool / short
        string) go into state, so drifting numbers or timestamps in /nandi/safety_status cannot turn
        state into a stream; change-driven sends are also spaced by state.min_interval_s (default 1 s),
        a change inside that window is sent from tick() once it has passed."""
        extra = {"run_id": self.run_id, "arm_mode": self.harvest.get("mode"),
                 "gait": self.loco.get("mode"), "policy_loaded": bool(self.loco.get("policy_loaded"))}
        if self.safety is not None:
            extra["safety"] = safety_for_state(self.safety)
        key = json.dumps(extra, sort_keys=True, default=str)
        if not force and key == self._state_key:
            self._state_pending = False
            return
        now_s = self.wall_ms() / 1000.0
        if not force and now_s - self._last_state_s < self.state_min_interval_s:
            self._state_pending = True
            return
        self._state_key = key
        self._state_pending = False
        self._last_state_s = now_s
        mode = "auto" if self.harvest.get("mode") == "ik_harvest" else (
            "manual" if self.harvest.get("mode") == "manual" else "idle")
        self._emit(lambda: self.m.state(self.wall_ms(), mode, extra))


_VOLATILE_KEY_PARTS = ("time", "stamp", "ts", "age", "seq", "count")


def safety_for_state(safety: Dict[str, Any], limit: int = 16) -> Dict[str, Any]:
    """Discrete /nandi/safety_status fields for retained state: bools, None and short strings, minus
    keys that look like clocks or counters. Numbers (tilt, margins) belong in telemetry, not state."""
    out: Dict[str, Any] = {}
    for k, v in safety.items():
        if len(out) >= limit:
            break
        lk = str(k).lower()
        if any(part in lk.split("_") for part in _VOLATILE_KEY_PARTS):
            continue
        if v is None or isinstance(v, bool) or (isinstance(v, str) and len(v) <= 64):
            out[str(k)] = v
    return out


def install_sigterm():
    """SIGTERM (launch script `kill 0`, docker stop) ends the spin like Ctrl+C, so the run closes cleanly
    (status offline, final manifest)."""
    import signal

    def _raise(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _raise)


def load_json(text) -> Optional[Any]:
    if not isinstance(text, str) or not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def battery_state_to_dict(msg) -> Dict[str, Any]:
    """sensor_msgs/BatteryState -> tele/batt fields (NaN fields are dropped by the mapper)."""
    out = {"voltage_v": float(msg.voltage), "current_a": float(msg.current)}
    if msg.percentage == msg.percentage:  # not NaN
        out["soc"] = float(msg.percentage)
    if msg.voltage == msg.voltage and msg.current == msg.current:
        out["power_w"] = float(msg.voltage) * float(msg.current)
    if msg.temperature == msg.temperature:
        out["temperature_c"] = float(msg.temperature)
    return out


# ---------------------------------------------------------------------- ROS wiring
OPTIONAL_TOPICS = ("/nandi/mission_status", "/nandi/battery", "/nandi/crop_detections",
                   "/nandi/safety_status", "/nandi/joint_states")


def make_bridge_node(bridge: TelemetryBridge, hooks: Optional[Dict[str, Callable]] = None, name: str = "nandi_telemetry_bridge"):
    """rclpy node feeding `bridge`; hooks (recorder) get the same parsed inputs: qpos, harvest, game,
    loco, mission, safety, battery, joint_states, cmd_vel, manual_arm, tick."""
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray, String

    hooks = hooks or {}

    def hook(name_, *args):
        fn = hooks.get(name_)
        if fn:
            fn(*args)

    class BridgeNode(Node):
        def __init__(self):
            super().__init__(name)
            node_params.declare(self, name)  # WPR: config/nandi.yaml + NANDI_* env (env wins)
            self._optional: Dict[str, Any] = {}
            self.create_subscription(Float64MultiArray, "/nandi/sim_qpos", self.on_qpos, 10)
            self.create_subscription(String, "/nandi/harvest_status", self.on_harvest, 10)
            self.create_subscription(String, "/nandi/game_status", self.on_game, 10)
            self.create_subscription(String, "/nandi/locomotion_status", self.on_loco, 10)
            try:
                from geometry_msgs.msg import Twist
                self.create_subscription(Twist, "/nandi/cmd_vel", self.on_cmd_vel, 10)
            except ImportError:
                pass
            self.create_subscription(Float64MultiArray, "/nandi/manual_arm", self.on_manual_arm, 10)
            self.create_timer(0.1, self.on_tick)
            self.create_timer(2.0, self.discover_optional)
            self.discover_optional()

        @staticmethod
        def now() -> float:
            return time.monotonic()

        # optional topics: subscribe once they appear, with the type they are advertised with
        def discover_optional(self):
            for topic, types in self.get_topic_names_and_types():
                if topic not in OPTIONAL_TOPICS or topic in self._optional:
                    continue
                sub = self._subscribe_optional(topic, types)
                if sub is not None:
                    self._optional[topic] = sub
                    self.get_logger().info(f"telemetry: subscribed to {topic} ({types[0]})")

        def _subscribe_optional(self, topic: str, types):
            from std_msgs.msg import String as S
            if "std_msgs/msg/String" in types:
                return self.create_subscription(S, topic, lambda m, t=topic: self.on_json(t, m.data), 10)
            if topic == "/nandi/battery" and "sensor_msgs/msg/BatteryState" in types:
                from sensor_msgs.msg import BatteryState
                return self.create_subscription(
                    BatteryState, topic, lambda m: self.on_battery(battery_state_to_dict(m)), 10)
            if topic == "/nandi/joint_states" and "sensor_msgs/msg/JointState" in types:
                from sensor_msgs.msg import JointState
                return self.create_subscription(JointState, topic, self.on_joint_states, 50)
            return None

        def on_json(self, topic: str, text: str):
            d = load_json(text)
            if d is None:
                return
            now = self.now()
            if topic == "/nandi/mission_status" and isinstance(d, dict):
                bridge.on_mission_status(d, now)
                hook("mission", now, d)
            elif topic == "/nandi/battery" and isinstance(d, dict):
                self.on_battery(d)
            elif topic == "/nandi/crop_detections":
                bridge.on_detections(d)
            elif topic == "/nandi/safety_status" and isinstance(d, dict):
                bridge.on_safety(d)
                hook("safety", now, d)

        def on_battery(self, d: Dict[str, Any]):
            bridge.on_battery(d)
            hook("battery", self.now(), d)

        def on_joint_states(self, msg):
            hook("joint_states", self.now(), list(msg.name), list(msg.effort), list(msg.velocity))

        def on_qpos(self, msg):
            q = list(msg.data)
            bridge.on_qpos(q)
            hook("qpos", self.now(), q)

        def on_harvest(self, msg):
            h = tm.parse_harvest_status(msg.data)
            if h["ok"]:
                now = self.now()
                bridge.on_harvest(h, now)
                hook("harvest", now, h)

        def on_game(self, msg):
            g = tm.parse_game_status(msg.data)
            if g["ok"]:
                now = self.now()
                bridge.on_game(g, now)
                hook("game", now, g)

        def on_loco(self, msg):
            loco = tm.parse_locomotion_status(msg.data)
            if loco["ok"]:
                bridge.on_loco(loco)
                hook("loco", self.now(), loco)

        def on_cmd_vel(self, msg):
            nonzero = any(abs(v) > 1e-6 for v in (msg.linear.x, msg.linear.y, msg.angular.z))
            hook("cmd_vel", self.now(), nonzero)

        def on_manual_arm(self, msg):
            hook("manual_arm", self.now(), True)

        def on_tick(self):
            now = self.now()
            bridge.tick(now)
            hook("tick", now)

    return BridgeNode()


def main():
    """Standalone bridge: own run directory with meta.json + kis_v1.jsonl (no attempt log / KPIs)."""
    import rclpy
    from rclpy.executors import ExternalShutdownException
    install_sigterm()
    from atharv_nandi.core import run_log as rl
    cfg = load_config(os.environ.get("NANDI_TELEMETRY_CONFIG", DEFAULT_CONFIG))
    sha, dirty = rl.git_sha(REPO), rl.git_dirty(REPO)
    run_id = rl.make_run_id(rl.utc_stamp(), (sha or "nogit")[:7], os.environ.get("NANDI_SEED", "7"))
    run_dir = os.path.join(REPO, (cfg.get("run", {}) or {}).get("runs_dir", "runs"), run_id)
    os.makedirs(run_dir, exist_ok=True)
    mapper = make_mapper(cfg, rl.sw_version(sha, dirty))
    rl.write_json(os.path.join(run_dir, "meta.json"), rl.build_meta(run_id, REPO, [DEFAULT_CONFIG]))
    rclpy.init()
    bridge = TelemetryBridge(mapper, make_sinks(cfg, run_dir, mapper), cfg, run_id)
    node = make_bridge_node(bridge)
    bridge.log = node.get_logger().warn
    bridge.start()
    node.get_logger().info(f"kis/v1 bridge -> {run_dir}/kis_v1.jsonl")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        bridge.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
