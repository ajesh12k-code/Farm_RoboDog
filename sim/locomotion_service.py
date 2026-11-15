#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger


class LocomotionServiceNode(Node):
    """Independent 18-DOF hexapod tripod gait service & command publisher."""

    def __init__(self):
        super().__init__("farmdog6_locomotion_service")
        self.vx = 0.0
        self.vy = 0.0
        self.is_held = False
        self.remaining_phase = 0.0
        self.gait_phase = 0.0

        self.leg_signs = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        self.tripod_phase = [0.0, math.pi, math.pi, 0.0, 0.0, math.pi]

        # Subscribe to velocity commands (linear.x = fwd/back, linear.y = left/right, angular.z > 0 = hold)
        self.create_subscription(Twist, "/farmdog6/cmd_vel", self.on_cmd_vel, 10)
        # Service to trigger an immediate emergency stop
        self.create_service(Trigger, "/farmdog6/stop_motion", self.handle_stop)
        # Publish 18 leg joint targets at 100 Hz
        self.leg_pub = self.create_publisher(Float64MultiArray, "/farmdog6/leg_targets", 10)
        self.dt = 0.01
        self.create_timer(self.dt, self.update_gait)
        self.get_logger().info("Locomotion Service ready (/farmdog6/cmd_vel, /farmdog6/stop_motion).")

    def on_cmd_vel(self, msg: Twist):
        if abs(msg.linear.x) < 1e-3 and abs(msg.linear.y) < 1e-3:
            self.is_held = False
            return
        self.vx = float(msg.linear.x)
        self.vy = float(msg.linear.y)
        self.is_held = bool(msg.angular.z > 0.5)
        if self.remaining_phase <= 0.05:
            self.remaining_phase = 2.0 * math.pi

    def handle_stop(self, request, response):
        self.is_held = False
        self.remaining_phase = 0.0
        self.vx = 0.0
        self.vy = 0.0
        response.success = True
        response.message = "Locomotion halted."
        return response

    def update_gait(self):
        active = self.is_held or (self.remaining_phase > 0.0)
        if active and math.hypot(self.vx, self.vy) > 1e-3:
            d_phase = 2.0 * math.pi * 1.8 * self.dt
            self.gait_phase = (self.gait_phase + d_phase) % (2.0 * math.pi)
            if not self.is_held:
                self.remaining_phase = max(0.0, self.remaining_phase - d_phase)
                if self.remaining_phase == 0.0:
                    self.vx = 0.0
                    self.vy = 0.0
        else:
            self.gait_phase = 0.0

        targets = []
        for i in range(6):
            side = self.leg_signs[i]
            base_hip, base_thigh, base_calf = side * 0.12, 0.80, -1.60
            if not active or math.hypot(self.vx, self.vy) < 1e-3:
                targets.extend([base_hip, base_thigh, base_calf])
                continue

            phase = self.gait_phase + self.tripod_phase[i]
            swing = math.sin(phase)
            lift = max(0.0, math.cos(phase))
            targets.extend([
                base_hip + 0.22 * self.vy * swing,
                base_thigh - 0.28 * self.vx * swing - 0.15 * lift,
                base_calf - 0.42 * lift,
            ])

        msg = Float64MultiArray()
        msg.data = targets
        self.leg_pub.publish(msg)


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
