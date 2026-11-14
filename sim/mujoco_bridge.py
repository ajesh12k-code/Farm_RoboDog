import os
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class MuJoCoBridge(Node):
    def __init__(self):
        super().__init__('mujoco_bridge')

        model_path = os.path.join(os.path.dirname(__file__), 'models/unitree_a1/scene.xml')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Reset to upright 'home' standing keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.target_ctrl = self.data.qpos[7:].copy()
        self.data.ctrl[:] = self.target_ctrl

        # Extract the 12 actuated joint names in actuator order
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.actuator_trnid[i, 0])
            for i in range(self.model.nu)
        ]

        # Publishers & Subscribers
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.cmd_sub = self.create_subscription(
            Float64MultiArray, '/joint_commands', self.cmd_callback, 10
        )

        # Publish joint telemetry at 100 Hz
        self.create_timer(0.01, self.publish_joint_states)
        self.get_logger().info('MuJoCo Bridge active. Publishing /joint_states, listening on /joint_commands.')

    def cmd_callback(self, msg: Float64MultiArray):
        if len(msg.data) == self.model.nu:
            self.target_ctrl = np.array(msg.data, dtype=np.float64)

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.data.qpos[7:].tolist()
        msg.velocity = self.data.qvel[6:].tolist()
        self.joint_pub.publish(msg)


def main():
    rclpy.init()
    bridge = MuJoCoBridge()

    # Spin ROS 2 callbacks in a background thread so the GUI viewer stays smooth
    ros_thread = threading.Thread(target=rclpy.spin, args=(bridge,), daemon=True)
    ros_thread.start()

    with mujoco.viewer.launch_passive(bridge.model, bridge.data) as viewer:
        while viewer.is_running() and rclpy.ok():
            step_start = time.time()
            bridge.data.ctrl[:] = bridge.target_ctrl
            mujoco.mj_step(bridge.model, bridge.data)
            viewer.sync()

            dt = bridge.model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

    bridge.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
