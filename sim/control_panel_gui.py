#!/usr/bin/env python3
import threading
import tkinter as tk
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger


class ControlPanelClient(Node):
    """Thin Tkinter UI client invoking modular ROS 2 services."""

    def __init__(self):
        super().__init__("farmdog6_control_panel")
        self.cmd_pub = self.create_publisher(Twist, "/farmdog6/cmd_vel", 10)
        self.arm_pub = self.create_publisher(Float64MultiArray, "/farmdog6/manual_arm", 10)
        self.wf_pub = self.create_publisher(String, "/farmdog6/workspace_wireframe", 10)

        self.stop_cli = self.create_client(Trigger, "/farmdog6/stop_motion")
        self.harvest_cli = self.create_client(Trigger, "/farmdog6/auto_harvest")
        self.ws_cli = self.create_client(Trigger, "/compute_work_envelope")

        self.arm_cmd = [0.0, 0.0, 0.1, -1.8, 1.4, 0.0, 0.0]
        self.cached_wf = ""
        self.wf_visible = False

    def send_move(self, vx: float, vy: float, hold: bool):
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = 1.0 if hold else 0.0
        self.cmd_pub.publish(msg)

    def call_trigger(self, client, status_lbl=None, on_success=None):
        def _call():
            if not client.wait_for_service(timeout_sec=1.0):
                if status_lbl:
                    status_lbl.config(text="Service offline", fg="#f87171")
                return
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=4.0)
            res = future.result()
            if res and status_lbl:
                status_lbl.config(
                    text=res.message[:48],
                    fg="#34d399" if res.success else "#f87171",
                )
            if res and res.success and on_success:
                on_success(res.message)

        threading.Thread(target=_call, daemon=True).start()

    def build_ui(self):
        root = tk.Tk()
        root.title("KisaanuNexusFarmDog6 Service Console")
        root.attributes("-topmost", True)
        root.configure(bg="#1e242b")

        lbl = {"bg": "#1e242b", "fg": "#e6edf3", "font": ("Sans", 10, "bold")}
        btn = {"bg": "#2d6a4f", "fg": "white", "font": ("Sans", 10, "bold"), "width": 11, "pady": 6, "bd": 0}

        tk.Label(root, text="LOCOMOTION SERVICE (TAP = 1 STEP | HOLD = WALK)", **lbl).pack(pady=(10, 4))
        pad = tk.Frame(root, bg="#1e242b")
        pad.pack(pady=4)

        for text, r, c, vx, vy in [("▲ Forward", 0, 1, 1.0, 0.0), ("◄ Left", 1, 0, 0.0, 1.0),
                                   ("Right ►", 1, 2, 0.0, -1.0), ("▼ Backward", 2, 1, -1.0, 0.0)]:
            b = tk.Button(pad, text=text, **btn)
            b.grid(row=r, column=c, padx=4, pady=4)
            b.bind("<ButtonPress-1>", lambda e, x=vx, y=vy: self.send_move(x, y, True))
            b.bind("<ButtonRelease-1>", lambda e: self.send_move(0.0, 0.0, False))

        tk.Button(
            pad, text="■ STOP", command=lambda: self.call_trigger(self.stop_cli),
            bg="#b5173c", fg="white", font=("Sans", 10, "bold"), width=11, pady=6, bd=0
        ).grid(row=1, column=1, padx=4, pady=4)

        tk.Label(root, text="IK HARVEST SERVICE", **lbl).pack(pady=(10, 2))
        harv_lbl = tk.Label(root, text="Service: /farmdog6/auto_harvest", bg="#1e242b", fg="#94a3b8", font=("Sans", 8))
        tk.Button(
            root, text="🌾 Auto-Harvest (IK Pick & Basket)",
            command=lambda: (self.call_trigger(self.stop_cli), self.call_trigger(self.harvest_cli, harv_lbl)),
            bg="#7c3aed", fg="white", font=("Sans", 10, "bold"), width=32, pady=7, bd=0
        ).pack(pady=4)
        harv_lbl.pack()

        tk.Label(root, text="WORK ENVELOPE SERVICE", **lbl).pack(pady=(10, 2))
        ws_lbl = tk.Label(root, text="Service: /compute_work_envelope", bg="#1e242b", fg="#94a3b8", font=("Sans", 8))
        ws_row = tk.Frame(root, bg="#1e242b")
        ws_row.pack(pady=4)

        def _on_ws_data(payload_json: str):
            self.cached_wf = payload_json
            self.wf_visible = True
            m = String()
            m.data = payload_json
            self.wf_pub.publish(m)
            ws_lbl.config(text="Wireframe active in simulation!", fg="#34d399")

        def _toggle_wf():
            if not self.cached_wf:
                return
            self.wf_visible = not self.wf_visible
            m = String()
            m.data = self.cached_wf if self.wf_visible else ""
            self.wf_pub.publish(m)

        tk.Button(
            ws_row, text="🔄 Compute Envelope",
            command=lambda: self.call_trigger(self.ws_cli, ws_lbl, _on_ws_data),
            bg="#1d4ed8", fg="white", font=("Sans", 9, "bold"), width=19, pady=6, bd=0
        ).grid(row=0, column=0, padx=3)
        tk.Button(
            ws_row, text="👁 Show/Hide", command=_toggle_wf,
            bg="#334155", fg="white", font=("Sans", 9, "bold"), width=12, pady=6, bd=0
        ).grid(row=0, column=1, padx=3)
        ws_lbl.pack()

        tk.Label(root, text="MANUAL ARM OVERRIDES", **lbl).pack(pady=(10, 2))
        specs = [("Gantry (m)", -0.26, 0.26, 0.01, 0), ("Arm Yaw", -3.14, 3.14, 0.05, 1),
                 ("Shoulder", -1.60, 2.40, 0.05, 2), ("Elbow", -2.60, 2.60, 0.05, 3),
                 ("Wrist", -2.00, 2.00, 0.05, 4), ("Gripper", 0.00, 0.022, 0.001, 5)]
        for text, lo, hi, res, idx in specs:
            f = tk.Frame(root, bg="#1e242b")
            f.pack(fill="x", padx=12, pady=1)
            tk.Label(f, text=text, bg="#1e242b", fg="#cbd5e1", width=13, anchor="w", font=("Sans", 9)).pack(side="left")
            s = tk.Scale(
                f, from_=lo, to=hi, resolution=res, orient="horizontal", length=175,
                bg="#1e242b", fg="white", highlightthickness=0,
                command=lambda v, j=idx: self._on_slider(j, float(v))
            )
            s.set(self.arm_cmd[idx])
            s.pack(side="right")

        root.mainloop()

    def _on_slider(self, idx: int, val: float):
        self.arm_cmd[idx] = val
        if idx == 5:
            self.arm_cmd[6] = val
        msg = Float64MultiArray()
        msg.data = self.arm_cmd
        self.arm_pub.publish(msg)


def main():
    rclpy.init()
    node = ControlPanelClient()
    try:
        node.build_ui()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
