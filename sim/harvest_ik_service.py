#!/usr/bin/env python3
import os
import sys
import math
import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

sys.path.insert(0, os.path.dirname(__file__))
from controllers.arm_ik_solver import DorsalArmIKSolver


class HarvestIKServiceNode(Node):
    """Thin ROS 2 node switching between RL arm stabilization and IK pick-and-place."""

    def __init__(self):
        super().__init__("farmdog6_harvest_ik_service")
        xml_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "models/KisaanuNexusFarmDog6/kisaanu_nexus_farm_dog6.xml")
        )
        self.ik = DorsalArmIKSolver(xml_path)
        self.latest_qpos = self.ik.data.qpos.copy()
        self.arm_cmd = [0.0, 0.0, 0.1, -1.8, 1.4, 0.0, 0.0]
        self.mode = "rl_balance"  # 'rl_balance' | 'ik_harvest' | 'manual'

        self.create_subscription(Float64MultiArray, "/farmdog6/sim_qpos", self.on_qpos, 10)
        self.create_subscription(Float64MultiArray, "/farmdog6/rl_arm_targets", self.on_rl_arm, 10)
        self.create_subscription(Float64MultiArray, "/farmdog6/manual_arm", self.on_manual, 10)
        self.arm_pub = self.create_publisher(Float64MultiArray, "/farmdog6/arm_targets", 10)
        self.create_timer(0.02, lambda: self.arm_pub.publish(Float64MultiArray(data=self.arm_cmd)))
        self.create_service(Trigger, "/farmdog6/auto_harvest", self.on_auto_harvest)

    def on_qpos(self, msg: Float64MultiArray):
        if len(msg.data) == self.ik.model.nq:
            self.latest_qpos[:] = msg.data

    def on_rl_arm(self, msg: Float64MultiArray):
        if self.mode == "rl_balance" and len(msg.data) == 5:
            self.arm_cmd = [*msg.data, 0.0, 0.0]

    def on_manual(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            self.mode, self.arm_cmd = "manual", list(msg.data)

    def _move(self, target_5: np.ndarray, grip: float, duration: float = 0.8):
        start_5, start_g = np.array(self.arm_cmd[:5]), float(self.arm_cmd[5])
        steps = max(1, int(duration / 0.02))
        for s in range(1, steps + 1):
            if self.mode != "ik_harvest":
                return
            a = 0.5 * (1.0 - math.cos(math.pi * s / steps))
            q = (1.0 - a) * start_5 + a * target_5
            g = (1.0 - a) * start_g + a * grip
            self.arm_cmd = [*q.tolist(), g, g]
            time.sleep(0.02)

    def on_auto_harvest(self, request, response):
        if self.mode == "ik_harvest":
            response.success, response.message = False, "Harvest already running."
            return response

        trunk_pos, trunk_rot, crop_pos = self.ik.get_scene_poses(self.latest_qpos)
        dist = float(np.linalg.norm(crop_pos[:2] - trunk_pos[:2]))
        if dist > 1.08:
            response.success, response.message = False, f"Crop too far ({dist:.2f}m)."
            return response

        self.mode = "ik_harvest"

        def _sequence():
            try:
                q1 = self.ik.solve(self.latest_qpos, crop_pos + [0.0, 0.0, 0.09], np.array([0.24, 0.0, 1.35, -0.55, -0.80]))
                self._move(q1, 0.0, 0.9)
                q2 = self.ik.solve(self.latest_qpos, crop_pos + [0.0, 0.0, 0.004], q1)
                self._move(q2, 0.0, 0.6)
                self._move(q2, 0.019, 0.4)
                q3 = self.ik.solve(self.latest_qpos, crop_pos + [0.0, 0.0, 0.34], q2)
                self._move(q3, 0.019, 0.7)
                basket_pt = trunk_pos + (trunk_rot @ np.array([-0.14, 0.0, 0.24]))
                q4 = self.ik.solve(self.latest_qpos, basket_pt, np.array([-0.16, 0.0, 0.70, -1.60, 0.60]))
                self._move(q4, 0.019, 0.9)
                self._move(q4, 0.0, 0.4)
                self._move(np.array([0.0, 0.0, 0.1, -1.8, 1.4]), 0.0, 0.7)
            finally:
                self.mode = "rl_balance"

        threading.Thread(target=_sequence, daemon=True).start()
        response.success, response.message = True, "Switched to IK: harvesting crop."
        return response


def main():
    rclpy.init()
    node = HarvestIKServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
