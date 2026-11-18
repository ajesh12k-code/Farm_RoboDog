import threading  # Import threading for non-blocking ROS 2 spin and service calls
import rclpy  # Import ROS 2 Python client library
from rclpy.node import Node  # Import base ROS 2 Node class
from geometry_msgs.msg import Twist  # Import velocity command message type
from std_msgs.msg import Float64MultiArray, String  # Import array and JSON string messages
from std_srvs.srv import Trigger  # Import standard Trigger service definition

from atharv_nandi import node_params  # noqa: E402
from atharv_nandi.core.key_drive import DriveCommand, clamp_gait_params  # Import pure operator-input types
from atharv_nandi.core.telemetry import (  # Import pure telemetry / status parsers
    parse_game_status, parse_harvest_status, parse_locomotion_status, parse_qpos_telemetry,
)

SERVICE_POLL_S = 0.05   # Trigger reply poll interval; SERVICE_POLL_N polls (4 s) before "Timeout"
SERVICE_POLL_N = 80


class NandiROSClient(Node):
    """Thin ROS 2 topic publisher, telemetry/status subscriber, and async service caller."""

    def __init__(self):
        super().__init__("nandi_control_panel")
        node_params.declare(self, "nandi_control_panel")  # WPR: config/nandi.yaml + NANDI_* env (env wins)
        # Publishers for locomotion velocity, gait tuning, manual arm overrides, and 3D wireframe
        self.cmd_pub = self.create_publisher(Twist, "/nandi/cmd_vel", 10)
        self.gait_pub = self.create_publisher(Float64MultiArray, "/nandi/gait_params", 10)
        self.arm_pub = self.create_publisher(Float64MultiArray, "/nandi/manual_arm", 10)
        self.wf_pub = self.create_publisher(String, "/nandi/workspace_wireframe", 10)

        # Trigger service clients keyed by the short names the GUI uses
        self.service_clients = {
            "stop": self.create_client(Trigger, "/nandi/stop_motion"),
            "mode": self.create_client(Trigger, "/nandi/toggle_locomotion_mode"),
            "harvest": self.create_client(Trigger, "/nandi/auto_harvest"),
            "release": self.create_client(Trigger, "/nandi/release_arm"),
            "reset": self.create_client(Trigger, "/nandi/reset_world"),
            "workspace": self.create_client(Trigger, "/compute_work_envelope"),
        }

        # Latest parsed telemetry tuple: (forward_x_m, height_z_m, tilt_deg)
        self.telemetry = (0.0, 0.0, 0.0)
        # Latest parsed JSON status dicts ("ok" is False until a valid message arrives)
        self.loco_status = parse_locomotion_status(None)
        self.harvest_status = parse_harvest_status(None)
        self.game_status = parse_game_status(None)
        # Shared envelope state: the web console and reset_world also publish workspace_wireframe
        self.wireframe_shown = False
        self.wireframe_json = ""  # last non-empty wireframe JSON seen on the topic

        # Subscribe to full MuJoCo qpos stream (50 Hz, sim_runner.py) and the JSON status topics
        self.create_subscription(Float64MultiArray, "/nandi/sim_qpos", self._on_qpos, 10)
        self.create_subscription(String, "/nandi/locomotion_status", self._on_loco_status, 10)
        self.create_subscription(String, "/nandi/harvest_status", self._on_harvest_status, 10)
        self.create_subscription(String, "/nandi/game_status", self._on_game_status, 10)
        self.create_subscription(String, "/nandi/workspace_wireframe", self._on_wireframe, 10)
        # Spin ROS 2 callbacks in a background daemon thread so Tkinter never freezes
        threading.Thread(target=self._spin_bg, daemon=True).start()

    def _spin_bg(self):
        """Continuously process incoming ROS 2 subscription and service responses."""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def _on_qpos(self, msg: Float64MultiArray):
        """Update cached telemetry whenever a new simulation state arrives."""
        self.telemetry = parse_qpos_telemetry(msg.data)

    # Status callbacks replace the whole dict (atomic reference swap; the GUI only reads)
    def _on_loco_status(self, msg: String):
        self.loco_status = parse_locomotion_status(msg.data)

    def _on_harvest_status(self, msg: String):
        self.harvest_status = parse_harvest_status(msg.data)

    def _on_game_status(self, msg: String):
        self.game_status = parse_game_status(msg.data)

    def _on_wireframe(self, msg: String):
        if msg.data:
            self.wireframe_json = msg.data
        self.wireframe_shown = bool(msg.data)

    def send_drive(self, cmd: DriveCommand):
        """Publish cmd_vel: linear.x=vx, linear.y=vy, angular.z=yaw, linear.z=1.0 hold / 0.0 step-or-release."""
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(cmd.vx), float(cmd.vy), float(cmd.yaw)
        msg.linear.z = 1.0 if cmd.hold else 0.0
        self.cmd_pub.publish(msg)

    def send_gait_params(self, params: list):
        """Publish [speed_scale, step_height_scale, freq_hz, balance_gain] (clamped to contract ranges)."""
        self.gait_pub.publish(Float64MultiArray(data=clamp_gait_params(params)))

    def send_arm(self, joints_7dof: list):
        """Publish 7-DOF manual dorsal arm + gripper targets."""
        self.arm_pub.publish(Float64MultiArray(data=[float(v) for v in joints_7dof]))

    def send_wireframe(self, json_str: str):
        """Publish 3D workspace wireframe JSON (or empty string to hide)."""
        self.wf_pub.publish(String(data=json_str))

    def call_service(self, key: str, on_done=None):
        """Invoke a Trigger service asynchronously and pass (success, message) to on_done.

        on_done runs on a worker thread: GUI callers must marshal widget updates to the Tk thread.
        """
        client = self.service_clients[key]

        def _worker():
            # Verify service availability before sending request
            if not client.wait_for_service(timeout_sec=1.0):
                return on_done and on_done(False, "Service offline")
            fut = client.call_async(Trigger.Request())
            # Poll future completion while _spin_bg drives ROS 2 executors
            waiter = threading.Event()
            for _ in range(SERVICE_POLL_N):
                if fut.done():
                    break
                waiter.wait(SERVICE_POLL_S)
            if fut.done():
                res = fut.result()
            else:
                # timed out: drop the request so a late reply is not kept (or delivered) forever (WPCOV review)
                res = None
                fut.cancel()
                remove = getattr(client, "remove_pending_request", None)
                if remove is not None:
                    remove(fut)
            if on_done:
                on_done(bool(res and res.success), res.message if res else "Timeout")

        threading.Thread(target=_worker, daemon=True).start()
