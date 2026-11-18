#!/usr/bin/env python3
"""ROS 2 service /compute_work_envelope: reach wireframe of the dorsal arm as compact JSON.

The envelope only depends on the robot geometry, so it is computed once at start-up and the cached
JSON (~30 KB, mm precision) is returned instantly. The previous version recomputed per call and
returned ~210 KB of full-precision floats in a Trigger response, which did not come back reliably
(`ros2 service call` printed nothing).
"""
import json
import time

import mujoco
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from atharv_nandi import node_params  # noqa: E402
from atharv_nandi.core.arm_ik_solver import compute_work_envelope
from atharv_nandi.core.farm_scene import load_farm_model


class WorkEnvelopeServiceNode(Node):
    def __init__(self):
        super().__init__("work_envelope_service")
        node_params.declare(self, "work_envelope_service")  # WPR: config/nandi.yaml + NANDI_* env (env wins)
        self.declare_parameter("xml_path", "")  # empty -> shared farm world (atharv_nandi/core/farm_scene.py)
        self.cached_json, self.summary = None, ""
        self._compute()
        self.srv = self.create_service(Trigger, "/compute_work_envelope", self.handle_compute_request)
        self.get_logger().info("ROS 2 Service '/compute_work_envelope' ready (model-agnostic kinematic solver).")

    def _compute(self):
        xml_path = self.get_parameter("xml_path").get_parameter_value().string_value
        t0 = time.time()
        try:
            model = mujoco.MjModel.from_xml_path(xml_path) if xml_path else load_farm_model()
            payload = compute_work_envelope(model)
        except Exception as exc:
            self.get_logger().error(f"Work envelope calculation failed: {exc}")
            return str(exc)
        self.cached_json = json.dumps(payload, separators=(",", ":"))
        self.summary = (
            f"joints {payload['joints_discovered']}: {len(payload['canopy_segments'])} canopy + "
            f"{len(payload['floor_segments'])} floor segments, {len(self.cached_json) / 1024:.0f} KB, "
            f"{time.time() - t0:.2f} s"
        )
        self.get_logger().info(f"Computed work envelope for {self.summary}")
        return None

    def handle_compute_request(self, request, response):
        error = self._compute() if self.cached_json is None else None
        response.success = self.cached_json is not None
        response.message = self.cached_json if response.success else (error or "Work envelope unavailable")
        return response


def main():
    rclpy.init()
    node = WorkEnvelopeServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
