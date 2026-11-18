#!/usr/bin/env python3
"""Atharv Nandi safety supervisor: latched e-stop, geofence, fall response, liveness, WP6 field latches.

Thin ROS 2 shell around atharv_nandi/core/safety_logic.py (SafetyCore). The latch lives here;
sim_runner enforces it (safe hold of every actuator, arm_targets/leg_targets ignored) and
locomotion refuses cmd_vel while /nandi/safety_status says estopped. The latch is saved to
estop.state_file and restored on start, so restarting this node does not re-arm the robot.
WP6: /nandi/safety_fields (atharv_nandi/sim/hooks/stability.py, 10 Hz) carries the hook's protective-field latch
(protective / animal / bumper / stability). The supervisor adopts it as an e-stop latch with that reason, so
sim_runner enforces it as the safe hold; /nandi/estop_reset is refused until the hook reports the field clear
(protective_clear) in a payload younger than 1 s.
WP6: /nandi/handheld_estop/state (WP1 hook handheld_estop) latches the e-stop with reason 'handheld' on
estop_request; /nandi/estop_reset is refused while the paired handheld still reports its mushroom pressed (or the
state is older than 1 s after a handheld latch).
See README, "Safety".
"""
import json
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from atharv_nandi import node_params  # noqa: E402
from atharv_nandi.core.game_status import is_fallen, tilt_deg  # noqa: E402
from atharv_nandi.core.safety_logic import (  # noqa: E402
    DEFAULT_STATE_FILE, HOLD_MODE_NONE, HOLD_MODE_SAFE, REASON_SUPERVISOR_LOST, LivenessTracker, SafetyCore,
    StaleMonitor, build_safety_status, heartbeat_json, load_latch_state, load_safety_config, parse_heartbeat,
    save_latch_state,
)

STATUS_PERIOD_S = 0.1      # /nandi/safety_status at 10 Hz (contract: >= 5 Hz), plus on every change
HEARTBEAT_PERIOD_S = 0.5
HARVEST_NODE = "harvest"   # liveness inferred from /nandi/harvest_status freshness
SIM_NODE = "sim_runner"


class SafetySupervisorNode(Node):
    def __init__(self):
        super().__init__("nandi_safety_supervisor")
        node_params.declare(self, "nandi_safety_supervisor")  # WPR: config/nandi.yaml + NANDI_* env (env wins)
        self.cfg = load_safety_config()
        self.core = SafetyCore(self.cfg)
        self.stale = StaleMonitor(self.cfg["stale"]["max_age_s"])
        self.stale_topics = list(self.cfg["stale"]["topics"])
        self.live = LivenessTracker(self.cfg["liveness"]["max_age_s"])
        self.t0 = time.monotonic()
        now = self._now()
        for topic in self.stale_topics:
            self.stale.expect(topic, now)
        self.hold_mode = HOLD_MODE_NONE
        self.last_stop = None
        self._n_events = 0
        # The latch survives a supervisor restart (README, "Safety"): restore it from the state file.
        self.state_file = (os.environ.get("NANDI_ESTOP_STATE")
                           or self.cfg["estop"].get("state_file") or DEFAULT_STATE_FILE)
        if self.core.restore(load_latch_state(self.state_file), now):
            self.get_logger().error(
                f"E-STOP restored from {self.state_file}: reason={self.core.latch.reason} "
                "(still latched from before this supervisor started; /nandi/estop_reset re-arms)")
            self._n_events = len(self.core.events)
        self._save_state()

        self.create_subscription(Float64MultiArray, "/nandi/sim_qpos", self.on_qpos, 10)
        self.create_subscription(Float64MultiArray, "/nandi/leg_targets", self.on_legs, 10)
        self.create_subscription(String, "/nandi/heartbeat", self.on_heartbeat, 50)
        self.create_subscription(String, "/nandi/harvest_status", self.on_harvest_status, 10)
        self.create_subscription(String, "/nandi/safety_fields", self.on_safety_fields, 10)
        self.create_subscription(String, "/nandi/handheld_estop/state", self.on_handheld, 10)
        self.status_pub = self.create_publisher(String, "/nandi/safety_status", 10)
        self.hb_pub = self.create_publisher(String, "/nandi/heartbeat", 10)
        self.create_service(Trigger, "/nandi/estop", self.on_estop)
        self.create_service(Trigger, "/nandi/estop_reset", self.on_estop_reset)
        self.create_timer(STATUS_PERIOD_S, self.publish_status)
        self.create_timer(HEARTBEAT_PERIOD_S, self.publish_heartbeat)
        gf = self.cfg["geofence"]
        self.get_logger().info(
            f"Safety: speed_cap={self.core.speed_cap.max_speed_scale}, geofence "
            f"{'on ' + json.dumps(gf['polygon']) if gf.get('enabled', True) else 'off'}, "
            f"fall tilt>{self.core.fall.tilt_deg} deg for {self.core.fall.sustain_s} s"
        )

    def _now(self) -> float:
        return time.monotonic() - self.t0

    # --- inputs -----------------------------------------------------------------------------
    def on_qpos(self, msg: Float64MultiArray):
        now = self._now()
        self.stale.touch("/nandi/sim_qpos", now)
        q = msg.data
        if len(q) < 7:
            return
        pos, quat = q[0:3], q[3:7]
        reason = self.core.update_pose(now, (q[0], q[1]), tilt_deg(quat), is_fallen(pos, quat))
        if reason:
            self._log_new_events()
            self._save_state()
            self.publish_status()

    def on_legs(self, msg: Float64MultiArray):
        self.stale.touch("/nandi/leg_targets", self._now())

    def on_heartbeat(self, msg: String):
        hb = parse_heartbeat(msg.data)
        if hb is None:
            return
        self.live.beat(hb["node"], self._now(), hb["ok"])
        if hb["node"] == SIM_NODE:
            info = hb["info"]
            self.hold_mode = str(info.get("hold_mode", HOLD_MODE_NONE))
            # sim_runner safe-held on its own (it lost our status, e.g. while we restarted): adopt
            # that as a latch so the status tells the truth and estop_reset is the way out. A hold
            # that started before the last re-arm is a heartbeat still in flight: ignore it.
            since = info.get("hold_since")
            if (self.hold_mode == HOLD_MODE_SAFE and info.get("hold_reason") == REASON_SUPERVISOR_LOST
                    and not self.core.latch.estopped and isinstance(since, (int, float))
                    and (self.core.rearmed_at is None or since > self.core.rearmed_at)):
                if self.core.engage_supervisor_lost(self._now()):
                    self._log_new_events()
                    self._save_state()
                    self.publish_status()
            if isinstance(info.get("last_stop"), dict):
                if info["last_stop"] != self.last_stop:
                    self.get_logger().info(f"E-stop stop metrics: {info['last_stop']}")
                self.last_stop = info["last_stop"]

    def on_harvest_status(self, msg: String):
        self.live.beat(HARVEST_NODE, self._now(), True)

    def on_safety_fields(self, msg: String):
        try:
            fields = json.loads(msg.data)
        except ValueError:
            return
        if isinstance(fields, dict) and self.core.update_fields(self._now(), fields):
            self._log_new_events()
            self._save_state()
            self.publish_status()

    def on_handheld(self, msg: String):
        try:
            state = json.loads(msg.data)
        except ValueError:
            return
        if isinstance(state, dict) and self.core.update_handheld(self._now(), state):
            self._log_new_events()
            self._save_state()
            self.publish_status()

    # --- services ---------------------------------------------------------------------------
    def on_estop(self, request, response):
        if self.core.engage_operator(self._now()):
            self._log_new_events()
            self._save_state()
            response.message = "E-STOP engaged (operator)"
        else:
            response.message = f"Already e-stopped ({self.core.latch.reason})"
        response.success = True
        self.publish_status()
        return response

    def on_estop_reset(self, request, response):
        response.success, response.message = self.core.reset(self._now())
        if response.success:
            self._save_state()
            self.get_logger().info(response.message)
        else:
            self.get_logger().warning(response.message)
        self.publish_status()
        return response

    # --- outputs ----------------------------------------------------------------------------
    def _save_state(self):
        try:
            save_latch_state(self.state_file, self.core.state())
        except OSError as exc:
            self.get_logger().warning(f"Could not save the e-stop latch to {self.state_file}: {exc}")

    def _log_new_events(self):
        for ev in self.core.events[self._n_events:]:
            self.get_logger().error(f"E-STOP latched: reason={ev['reason']} ({ev['detail']}) at t={ev['t']} s")
        self._n_events = len(self.core.events)

    def publish_status(self):
        now = self._now()
        nodes = self.live.nodes(now)
        status = build_safety_status(
            self.core.latch, self.core.speed_cap.max_speed_scale, self.core.geofence_ok, self.core.fallen,
            self.stale.stale(now, self.stale_topics), self.last_stop, nodes, self.hold_mode,
            self.core.margin, self.core.tilt, self.core.engaged_at, self.core.rearmed_at,
        )
        self.status_pub.publish(String(data=json.dumps(status)))

    def publish_heartbeat(self):
        info = {"estopped": self.core.latch.estopped, "reason": self.core.latch.reason}
        self.hb_pub.publish(String(data=heartbeat_json("safety_supervisor", self._now(), True, info)))


def main():
    rclpy.init()
    node = SafetySupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
