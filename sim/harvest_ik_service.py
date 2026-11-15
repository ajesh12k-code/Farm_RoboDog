#!/usr/bin/env python3
import os
import math
import time
import threading
import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger


class HarvestIKServiceNode(Node):
    """Independent Jacobian IK & 6-stage dorsal arm harvesting service."""

    def __init__(self):
        super().__init__("farmdog6_harvest_ik_service")
        xml_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "models/KisaanuNexusFarmDog6/kisaanu_nexus_farm_dog6.xml",
            )
        )
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.ik_data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.ik_data, 0)

        self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
        self.arm_limits = np.array([
            [-0.26, 0.26], [-3.14, 3.14], [-1.60, 2.40], [-2.60, 2.60], [-2.00, 2.00]
        ], dtype=np.float64)

        self.arm_cmd = [0.0, 0.0, 0.1, -1.8, 1.4, 0.0, 0.0]
        self.latest_qpos = self.ik_data.qpos.copy()
        self.is_harvesting = False

        # Subscribe to full simulation qpos & manual arm overrides
        self.create_subscription(Float64MultiArray, "/farmdog6/sim_qpos", self.on_sim_qpos, 10)
        self.create_subscription(Float64MultiArray, "/farmdog6/manual_arm", self.on_manual_arm, 10)
        # Publish 7-DOF arm commands at 50 Hz
        self.arm_pub = self.create_publisher(Float64MultiArray, "/farmdog6/arm_targets", 10)
        self.create_timer(0.02, self.publish_arm)

        # Service to trigger autonomous IK pick-and-place
        self.create_service(Trigger, "/farmdog6/auto_harvest", self.handle_auto_harvest)
        self.get_logger().info("Harvest IK Service ready (/farmdog6/auto_harvest).")

    def on_sim_qpos(self, msg: Float64MultiArray):
        if len(msg.data) == self.model.nq:
            self.latest_qpos = np.array(msg.data, dtype=np.float64)

    def on_manual_arm(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            self.is_harvesting = False
            self.arm_cmd = list(msg.data)

    def publish_arm(self):
        msg = Float64MultiArray()
        msg.data = self.arm_cmd
        self.arm_pub.publish(msg)

    def solve_ik(self, target_pos: np.ndarray, seed_q: np.ndarray, max_iters: int = 120):
        self.ik_data.qpos[:] = self.latest_qpos[:]
        self.ik_data.qpos[7:12] = seed_q[:5]
        mujoco.mj_forward(self.model, self.ik_data)

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        for _ in range(max_iters):
            mujoco.mj_kinematics(self.model, self.ik_data)
            err = target_pos - self.ik_data.site_xpos[self.tip_site_id]
            if np.linalg.norm(err) < 0.004:
                break
            mujoco.mj_jacSite(self.model, self.ik_data, jacp, None, self.tip_site_id)
            J = jacp[:, 6:11]
            dq = J.T @ np.linalg.solve(J @ J.T + (0.04 ** 2) * np.eye(3), err)
            self.ik_data.qpos[7:12] = np.clip(
                self.ik_data.qpos[7:12] + 0.65 * dq, self.arm_limits[:, 0], self.arm_limits[:, 1]
            )
        return self.ik_data.qpos[7:12].copy()

    def _move_smooth(self, target_5dof: np.ndarray, grip: float, duration: float = 0.8):
        start_5 = np.array(self.arm_cmd[:5], dtype=np.float64)
        start_g = float(self.arm_cmd[5])
        steps = max(1, int(duration / 0.02))
        for s in range(1, steps + 1):
            if not self.is_harvesting:
                return
            a = 0.5 * (1.0 - math.cos(math.pi * s / steps))
            q = (1.0 - a) * start_5 + a * target_5dof
            g = (1.0 - a) * start_g + a * grip
            self.arm_cmd = [float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]), g, g]
            time.sleep(0.02)

    def handle_auto_harvest(self, request, response):
        if self.is_harvesting:
            response.success = False
            response.message = "Harvest sequence already in progress."
            return response

        # Read live trunk and crop positions from latest_qpos (trunk=0..6, crop_item_1=32..38)
        self.ik_data.qpos[:] = self.latest_qpos[:]
        mujoco.mj_forward(self.model, self.ik_data)
        trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        crop_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "crop_item_1")

        trunk_pos = self.ik_data.xpos[trunk_id].copy()
        trunk_rot = self.ik_data.xmat[trunk_id].reshape(3, 3).copy()
        crop_pos = self.ik_data.xpos[crop_id].copy()

        dist_xy = float(np.linalg.norm(crop_pos[:2] - trunk_pos[:2]))
        if dist_xy > 1.08:
            response.success = False
            response.message = f"Crop out of reach ({dist_xy:.2f}m > 1.08m). Walk closer first."
            return response

        self.is_harvesting = True

        def _run_sequence():
            try:
                q_hover = self.solve_ik(crop_pos + [0.0, 0.0, 0.09], np.array([0.24, 0.0, 1.35, -0.55, -0.80]))
                self._move_smooth(q_hover, 0.0, 0.9)

                q_grasp = self.solve_ik(crop_pos + [0.0, 0.0, 0.004], q_hover)
                self._move_smooth(q_grasp, 0.0, 0.6)
                self._move_smooth(q_grasp, 0.019, 0.4)

                q_lift = self.solve_ik(crop_pos + [0.0, 0.0, 0.34], q_grasp)
                self._move_smooth(q_lift, 0.019, 0.7)

                basket_pos = trunk_pos + (trunk_rot @ np.array([-0.14, 0.0, 0.24]))
                q_basket = self.solve_ik(basket_pos, np.array([-0.16, 0.0, 0.70, -1.60, 0.60]))
                self._move_smooth(q_basket, 0.019, 0.9)
                self._move_smooth(q_basket, 0.0, 0.4)

                self._move_smooth(np.array([0.0, 0.0, 0.1, -1.8, 1.4]), 0.0, 0.7)
            finally:
                self.is_harvesting = False

        threading.Thread(target=_run_sequence, daemon=True).start()
        response.success = True
        response.message = "Auto-harvest IK sequence started."
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
