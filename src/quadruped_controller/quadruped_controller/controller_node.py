import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class QuadrupedController(Node):
    def __init__(self):
        super().__init__('quadruped_controller')

        self.home_pose = [
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
        ]
        self.current_positions = None
        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        self.cmd_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.state_sub = self.create_subscription(
            JointState, '/joint_states', self.state_callback, 10
        )

        self.create_timer(0.01, self.control_loop)
        self.get_logger().info('Quadruped Controller ready. Waiting for /joint_states...')

    def state_callback(self, msg: JointState):
        if self.current_positions is None:
            self.get_logger().info(f'Connected to simulator ({len(msg.position)} joints). Starting motion...')
        self.current_positions = list(msg.position)

    def control_loop(self):
        if self.current_positions is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - self.start_time

        offset = 0.2 * math.sin(2.0 * math.pi * 0.5 * elapsed)
        cmd = list(self.home_pose)
        for leg in range(4):
            cmd[leg * 3 + 1] += offset
            cmd[leg * 3 + 2] -= 2.0 * offset

        msg = Float64MultiArray()
        msg.data = cmd
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = QuadrupedController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
