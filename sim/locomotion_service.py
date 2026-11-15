#!/usr/bin/env python3
import os
import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

sys.path.insert(0, os.path.dirname(__file__))
from controllers.tripod_gait import TripodGaitGenerator
from controllers.rl_balance_policy import RLBalanceController


class LocomotionServiceNode(Node):
    """Thin ROS 2 node switching between 23-DOF RL balance policy and tripod gait."""

    def __init__(self):
        super().__init__("farmdog6_locomotion_service")
        xml_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "models/KisaanuNexusFarmDog6/kisaanu_nexus_farm_dog6.xml")
        )
        self.dt = 0.02
        self.gait = TripodGaitGenerator(self.dt)
        self.rl = RLBalanceController(xml_path, self.dt)
        self.prev_qpos = self.rl.data.qpos.copy()
        self.curr_qpos = self.rl.data.qpos.copy()

        if self.rl.policy_path:
            self.get_logger().info(f"RL policy loaded: {self.rl.policy_path}")
        else:
            self.get_logger().warn("No RL checkpoint yet; using tripod gait.")

        self.create_subscription(Twist, "/farmdog6/cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(Float64MultiArray, "/farmdog6/sim_qpos", self.on_qpos, 10)
        self.create_service(Trigger, "/farmdog6/stop_motion", self.on_stop)

        self.leg_pub = self.create_publisher(Float64MultiArray, "/farmdog6/leg_targets", 10)
        self.arm_pub = self.create_publisher(Float64MultiArray, "/farmdog6/rl_arm_targets", 10)
        self.create_timer(self.dt, self.step)

    def on_qpos(self, msg: Float64MultiArray):
        if len(msg.data) == self.rl.model.nq:
            self.prev_qpos = self.curr_qpos.copy()
            self.curr_qpos[:] = msg.data

    def on_cmd_vel(self, msg: Twist):
        self.gait.set_command(float(msg.linear.x), float(msg.linear.y), msg.angular.z > 0.5)

    def on_stop(self, request, response):
        self.gait.stop()
        self.rl.reset_arm_to_rest()
        response.success, response.message = True, "Stopped; arm returning to rest."
        return response

    def step(self):
        use_rl = (self.rl.policy is not None) and self.gait.is_active and (self.gait.vx > 0.1) and (abs(self.gait.vy) < 0.1)
        if use_rl:
            if not self.gait.is_held:
                self.gait.remaining_phase = max(0.0, self.gait.remaining_phase - 0.226)
                if self.gait.remaining_phase == 0.0:
                    self.gait.vx = 0.0
            arm_targets, leg_targets = self.rl.predict(self.prev_qpos, self.curr_qpos)
        else:
            leg_targets = self.gait.step()
            arm_targets = self.rl.reset_arm_to_rest()

        self.leg_pub.publish(Float64MultiArray(data=leg_targets))
        self.arm_pub.publish(Float64MultiArray(data=arm_targets))


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
