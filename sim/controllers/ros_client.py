import threading  # Import threading for non-blocking ROS 2 spin and service calls
import rclpy  # Import ROS 2 Python client library
from rclpy.node import Node  # Import base ROS 2 Node class
from geometry_msgs.msg import Twist  # Import velocity command message type
from std_msgs.msg import Float64MultiArray, String  # Import array and JSON string messages
from std_srvs.srv import Trigger  # Import standard Trigger service definition

from controllers.telemetry import parse_qpos_telemetry  # Import pure telemetry parser


class FarmDog6ROSClient(Node):
    """Thin ROS 2 topic publisher, telemetry subscriber, and async service caller."""

    def __init__(self):
        super().__init__("farmdog6_control_panel")
        # Publishers for locomotion velocity, manual arm overrides, and 3D wireframe
        self.cmd_pub = self.create_publisher(Twist, "/farmdog6/cmd_vel", 10)
        self.arm_pub = self.create_publisher(Float64MultiArray, "/farmdog6/manual_arm", 10)
        self.wf_pub = self.create_publisher(String, "/farmdog6/workspace_wireframe", 10)

        # Service clients for stop, RL/Tripod gait toggle, Auto-Harvest, and workspace solver
        self.service_clients = {
            "stop": self.create_client(Trigger, "/farmdog6/stop_motion"),
            "mode": self.create_client(Trigger, "/farmdog6/toggle_locomotion_mode"),
            "harvest": self.create_client(Trigger, "/farmdog6/auto_harvest"),
            "workspace": self.create_client(Trigger, "/compute_work_envelope"),
        }

        # Latest parsed telemetry tuple: (forward_x_m, height_z_m, tilt_deg)
        self.telemetry = (0.0, 0.0, 0.0)
        # Subscribe to full MuJoCo qpos stream published at 50 Hz by sim_runner.py
        self.create_subscription(Float64MultiArray, "/farmdog6/sim_qpos", self._on_qpos, 10)
        # Spin ROS 2 callbacks in a background daemon thread so Tkinter never freezes
        threading.Thread(target=self._spin_bg, daemon=True).start()

    def _spin_bg(self):
        """Continuously process incoming ROS 2 subscription and service responses."""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def _on_qpos(self, msg: Float64MultiArray):
        """Update cached telemetry whenever a new simulation state arrives."""
        self.telemetry = parse_qpos_telemetry(msg.data)

    def send_vel(self, vx: float, vy: float, hold: bool):
        """Publish linear velocity (vx, vy) and continuous-hold flag (angular.z)."""
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = vx, vy, (1.0 if hold else 0.0)
        self.cmd_pub.publish(msg)

    def send_arm(self, joints_7dof: list):
        """Publish 7-DOF manual dorsal arm + gripper targets."""
        self.arm_pub.publish(Float64MultiArray(data=joints_7dof))

    def send_wireframe(self, json_str: str):
        """Publish 3D workspace wireframe JSON (or empty string to hide)."""
        self.wf_pub.publish(String(data=json_str))

    def call_service(self, key: str, on_done=None):
        """Invoke a Trigger service asynchronously and pass (success, message) to on_done."""
        client = self.service_clients[key]

        def _worker():
            # Verify service availability before sending request
            if not client.wait_for_service(timeout_sec=1.0):
                return on_done and on_done(False, "Service offline")
            fut = client.call_async(Trigger.Request())
            # Poll future completion while _spin_bg drives ROS 2 executors
            for _ in range(80):
                if fut.done():
                    break
                threading.Event().wait(0.05)
            res = fut.result() if fut.done() else None
            if on_done:
                on_done(bool(res and res.success), res.message if res else "Timeout")

        threading.Thread(target=_worker, daemon=True).start()
