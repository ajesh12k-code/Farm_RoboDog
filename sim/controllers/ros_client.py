import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger


class FarmDog6ROSClient(Node):
    """Decoupled ROS 2 topic publisher and async service caller for the GUI."""

    def __init__(self):
        super().__init__("farmdog6_control_panel")
        self.cmd_pub = self.create_publisher(Twist, "/farmdog6/cmd_vel", 10)
        self.arm_pub = self.create_publisher(Float64MultiArray, "/farmdog6/manual_arm", 10)
        self.wf_pub = self.create_publisher(String, "/farmdog6/workspace_wireframe", 10)

        self.service_clients = {
            "stop": self.create_client(Trigger, "/farmdog6/stop_motion"),
            "mode": self.create_client(Trigger, "/farmdog6/toggle_locomotion_mode"),
            "harvest": self.create_client(Trigger, "/farmdog6/auto_harvest"),
            "workspace": self.create_client(Trigger, "/compute_work_envelope"),
        }

    def send_vel(self, vx: float, vy: float, hold: bool):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = vx, vy, (1.0 if hold else 0.0)
        self.cmd_pub.publish(msg)

    def send_arm(self, joints_7dof: list):
        self.arm_pub.publish(Float64MultiArray(data=joints_7dof))

    def send_wireframe(self, json_str: str):
        self.wf_pub.publish(String(data=json_str))

    def call_service(self, key: str, on_done=None):
        client = self.service_clients[key]

        def _worker():
            if not client.wait_for_service(timeout_sec=1.0):
                if on_done:
                    on_done(False, "Service offline")
                return
            fut = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, fut, timeout_sec=4.0)
            res = fut.result()
            if on_done:
                on_done(bool(res and res.success), res.message if res else "Timeout")

        threading.Thread(target=_worker, daemon=True).start()
