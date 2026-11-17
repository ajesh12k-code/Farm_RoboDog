#!/usr/bin/env python3
"""Atharv Nandi locomotion node: tripod gait (default) or the 23-DOF RL balance policy.

Thin ROS 2 shell; all logic lives in controllers/tripod_gait.py, controllers/locomotion_logic.py
and controllers/rl_balance_policy.py. See README / ROS contract for topic semantics.
"""
import os
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

sys.path.insert(0, os.path.dirname(__file__))
from controllers.locomotion_logic import (  # noqa: E402
    GaitParams, HoldDeadman, LocomotionModeSelector, locomotion_status_json, parse_cmd_vel, quat_to_roll_pitch,
)
from controllers.rl_balance_policy import RLBalanceController  # noqa: E402
from controllers.tripod_gait import TripodGaitGenerator  # noqa: E402


class LocomotionServiceNode(Node):
    def __init__(self):
        super().__init__("nandi_locomotion_service")
        self.dt = 0.02
        self.gait = TripodGaitGenerator(self.dt)
        self.deadman = HoldDeadman()
        self.rl = RLBalanceController(self.dt)  # builds the shared world via load_farm_model()
        self.selector = LocomotionModeSelector(self.rl.policy is not None, self.rl.policy_path)
        self.prev_qpos = self.rl.data.qpos.copy()
        self.curr_qpos = self.rl.data.qpos.copy()
        self._warned_rl_steer = False

        if self.selector.policy_loaded:
            self.get_logger().info(f"RL policy loaded: {self.rl.policy_path} (start in Tripod; toggle to use RL)")
        else:
            self.get_logger().warning(f"RL mode unavailable ({self.rl.load_error}); using Tripod gait.")

        self.create_subscription(Twist, "/nandi/cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(Float64MultiArray, "/nandi/gait_params", self.on_gait_params, 10)
        self.create_subscription(Float64MultiArray, "/nandi/sim_qpos", self.on_qpos, 10)
        self.create_service(Trigger, "/nandi/stop_motion", self.on_stop)
        self.create_service(Trigger, "/nandi/toggle_locomotion_mode", self.on_toggle_mode)

        self.leg_pub = self.create_publisher(Float64MultiArray, "/nandi/leg_targets", 10)
        self.arm_pub = self.create_publisher(Float64MultiArray, "/nandi/rl_arm_targets", 10)
        self.status_pub = self.create_publisher(String, "/nandi/locomotion_status", 10)
        self.create_timer(self.dt, self.step)
        self.create_timer(0.5, self.publish_status)

    # --- inputs -----------------------------------------------------------------------
    def on_qpos(self, msg: Float64MultiArray):
        if len(msg.data) == self.rl.model.nq:
            self.prev_qpos = self.curr_qpos.copy()
            self.curr_qpos[:] = msg.data

    def on_cmd_vel(self, msg: Twist):
        cmd = parse_cmd_vel(msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z)
        self.deadman.feed(cmd, time.monotonic())
        self.gait.set_command(cmd.vx, cmd.vy, cmd.yaw, cmd.hold)  # all-zero => release (finish cycle)

    def on_gait_params(self, msg: Float64MultiArray):
        params = GaitParams.from_sequence(list(msg.data), self.gait.params)
        self.gait.set_params(params)
        self.publish_status()

    def on_stop(self, request, response):
        self.gait.stop()
        self.deadman.disarm()
        self.rl.reset_arm_to_rest()
        response.success, response.message = True, "Locomotion stopped"
        return response

    def on_toggle_mode(self, request, response):
        response.success, response.message = self.selector.toggle()
        self.get_logger().info(response.message)
        self.publish_status()
        return response

    # --- outputs ----------------------------------------------------------------------
    def publish_status(self):
        data = locomotion_status_json(self.selector, self.gait.params, self.gait.is_walking)
        self.status_pub.publish(String(data=data))

    def step(self):
        if self.deadman.expired(time.monotonic()):
            self.gait.release()

        roll, pitch = quat_to_roll_pitch(*self.curr_qpos[3:7])
        leg_targets = self.gait.step(roll, pitch)  # always advances gait timing / single-step budget
        arm_targets = None
        if self.selector.mode == "rl" and self.gait.is_active:
            vx, vy, yaw = self.gait.cmd
            if (abs(vy) > 0.05 or abs(yaw) > 0.05 or vx < 0.0) and not self._warned_rl_steer:
                self.get_logger().warning("RL balance policy only walks forward; strafe/turn/reverse are ignored.")
                self._warned_rl_steer = True
            arm_targets, leg_targets = self.rl.predict(self.prev_qpos, self.curr_qpos)
        if arm_targets is None:
            arm_targets = self.rl.reset_arm_to_rest()

        self.leg_pub.publish(Float64MultiArray(data=[float(v) for v in leg_targets]))
        self.arm_pub.publish(Float64MultiArray(data=[float(v) for v in arm_targets]))


def main():
    rclpy.init()
    node = LocomotionServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
