#!/usr/bin/env python3
"""Atharv Nandi stability monitor: publishes /nandi/stability (std_msgs/String JSON, 10 Hz). ADVISORY.

Light ROS 2 node beside sim_runner (separate process; sim_runner is untouched and never waits on it).
It builds the same world as the sim through controllers.farm_scene.load_farm_model() (same NANDI_*
environment, so the payload / mast mocks match), copies every /nandi/sim_qpos into its own MjData and
runs mj_forward only (atharv_nandi/core/stability_probe.py; one probe < 2 ms, measured in
tests/test_stability_probe.py). It never steps the physics.

    in   /nandi/sim_qpos      Float64MultiArray (nq)  robot + crop poses
         /nandi/game_status   String JSON             in_basket_names -> basket mass
         /nandi/joint_states  sensor_msgs/JointState  optional: leg effort RMS vs 48 Nm continuous
    out  /nandi/stability     String JSON             schema: atharv_nandi/core/stability.make_status, docs/hld.md

Every flag is advisory ("advisory": true, "enforced": false): nothing consumes the inhibit list yet.
Flags and the UNSUPPORTED mode are debounced over stability.INHIBIT_DEBOUNCE_S (0.30 s) so tripod-gait
transients do not block walking; the per-sample verdict is published as mode_raw / inhibit_raw.
"""
import json
import os
import time

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from atharv_nandi import node_params  # noqa: E402
from atharv_nandi.core.farm_scene import load_farm_model  # noqa: E402
from atharv_nandi.core.stability import build_stability_status  # noqa: E402
from atharv_nandi.core.stability_probe import MonitorCore, env_float  # noqa: E402

STATUS_PERIOD_S = 0.1  # 10 Hz
HEAD_KG_DEFAULT = 1.5  # NANDI_MAST_HEAD_KG default of the mast mock (spec v1 section 4)


class StabilityMonitorNode(Node):
    def __init__(self):
        super().__init__("nandi_stability_monitor")
        node_params.declare(self, "nandi_stability_monitor")  # WPR: config/nandi.yaml + NANDI_* env (env wins)
        model = load_farm_model()
        self.core = MonitorCore(model, payload_kg=env_float(os.environ, "NANDI_PAYLOAD_KG", 0.0),
                                head_kg=env_float(os.environ, "NANDI_MAST_HEAD_KG", HEAD_KG_DEFAULT))
        self.t0 = time.monotonic()
        self._warned_nq = False
        self.create_subscription(Float64MultiArray, "/nandi/sim_qpos", self.on_qpos, 10)
        self.create_subscription(String, "/nandi/game_status", self.on_game, 10)
        self.create_subscription(JointState, "/nandi/joint_states", self.on_joint_states, 10)
        self.pub = self.create_publisher(String, "/nandi/stability", 10)
        self.create_timer(STATUS_PERIOD_S, self.tick)
        self.get_logger().info(
            f"Stability monitor (advisory): nq={model.nq}, robot {self.core.probe.robot_mass_kg:.2f} kg, "
            f"payload mock {self.core.payload_kg:g} kg, mast {'yes' if self.core.probe.head_site >= 0 else 'no'}")

    def _now(self) -> float:
        return time.monotonic() - self.t0

    def on_qpos(self, msg: Float64MultiArray):
        if not self.core.on_qpos(msg.data) and not self._warned_nq:
            self._warned_nq = True
            self.get_logger().warning(
                f"sim_qpos has {len(msg.data)} values, model nq={self.core.model.nq}: ignoring "
                "(stability_monitor and sim_runner must see the same NANDI_* environment)")

    def on_game(self, msg: String):
        try:
            self.core.on_game(json.loads(msg.data))
        except ValueError:
            pass

    def on_joint_states(self, msg: JointState):
        if msg.effort:
            self.core.on_joint_states(self._now(), list(msg.name), list(msg.effort))

    def tick(self):
        try:
            status = self.core.status(self._now())
        except Exception as exc:  # a bad snapshot must not kill the advisory monitor
            self.get_logger().error(f"stability probe failed: {exc}", throttle_duration_sec=5.0)
            return
        if status is not None:
            self.pub.publish(String(data=build_stability_status(status)))


def main():
    rclpy.init()
    node = StabilityMonitorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass  # SIGINT / SIGTERM (launch_nandi.sh kill 0): exit cleanly, status 0, no restart
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
