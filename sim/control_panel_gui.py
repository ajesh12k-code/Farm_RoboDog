#!/usr/bin/env python3
import os  # Import os to resolve local package directory
import sys  # Import sys to add sim/ to Python import path
import tkinter as tk  # Import Tkinter for window and event loop management
import rclpy  # Import rclpy for ROS 2 initialization and shutdown

sys.path.insert(0, os.path.dirname(__file__))
from controllers.ros_client import FarmDog6ROSClient  # Import decoupled ROS 2 bridge
from controllers.telemetry import format_telemetry  # Import telemetry formatter
from controllers.gui_widgets import BG, make_label, make_button, make_slider_row  # Import UI helpers


class ControlPanelApp:
    """Compact declarative Tkinter console for KisaanuNexusFarmDog6."""

    def __init__(self, client: FarmDog6ROSClient):
        self.cli = client
        # Default stowed rest targets for the 5 dorsal arm joints + 2 gripper fingers
        self.arm_cmd = [0.0, 0.0, 0.1, -1.8, 1.4, 0.0, 0.0]
        # Cached workspace wireframe JSON and visibility toggle state
        self.cached_wf, self.wf_visible = "", False

        # Configure always-on-top dark operator window
        self.root = tk.Tk()
        self.root.title("KisaanuNexusFarmDog6 Console")
        self.root.attributes("-topmost", True)
        self.root.configure(bg=BG)

        self._build_ui()
        # Start 10 Hz periodic UI refresh for live telemetry
        self._tick()

    def _build_ui(self):
        """Assemble telemetry bar, D-pad, gait toggle, IK harvest, and manual sliders."""
        # Top telemetry status bar
        bar = tk.Frame(self.root, bg="#0f172a", pady=5)
        bar.pack(fill="x", padx=8, pady=(8, 4))
        self.telem_lbl = make_label(bar, "Dist: +0.00 m  |  Height: 0.00 m  |  Tilt: 0.0°", "#38bdf8", 9)
        self.telem_lbl.pack()

        # Locomotion D-pad (press-and-hold walks continuously; quick tap takes 1 step)
        make_label(self.root, "LOCOMOTION (TAP = 1 STEP | HOLD = WALK)").pack(pady=(6, 2))
        pad = tk.Frame(self.root, bg=BG)
        pad.pack()
        for text, r, c, vx, vy in [("▲ Forward", 0, 1, 1.0, 0.0), ("◄ Left", 1, 0, 0.0, 1.0),
                                   ("Right ►", 1, 2, 0.0, -1.0), ("▼ Backward", 2, 1, -1.0, 0.0)]:
            b = make_button(pad, text, "#2d6a4f", width=10)
            b.grid(row=r, column=c, padx=3, pady=3)
            b.bind("<ButtonPress-1>", lambda e, x=vx, y=vy: self.cli.send_vel(x, y, True))
            b.bind("<ButtonRelease-1>", lambda e: self.cli.send_vel(0.0, 0.0, False))
        make_button(pad, "■ STOP", "#b5173c", lambda: self.cli.call_service("stop"), width=10).grid(row=1, column=1)

        # Live toggle button between 23-DOF RL Balance Policy and 18-DOF Tripod Gait
        mode_lbl = make_label(self.root, "Mode: RL Balance Policy", "#34d399", 8, False)
        mode_btn = make_button(self.root, "🤖 Switch Gait (RL ↔ Tripod)", "#0f766e")
        mode_btn.config(command=lambda: self.cli.call_service(
            "mode", lambda ok, m: (mode_lbl.config(text=m[:44], fg="#34d399" if ok else "#f87171"),
                                   mode_btn.config(bg="#0f766e" if "RL" in m else "#b45309"))))
        mode_btn.pack(pady=(4, 1))
        mode_lbl.pack()

        # Autonomous IK Pick-and-Place and 3D Work Envelope buttons
        make_label(self.root, "IK HARVEST & WORK ENVELOPE").pack(pady=(8, 2))
        status_lbl = make_label(self.root, "Services Ready", "#94a3b8", 8, False)
        make_button(self.root, "🌾 Auto-Harvest (IK Pick & Basket)", "#7c3aed",
                    lambda: (self.cli.call_service("stop"),
                             self.cli.call_service("harvest", lambda ok, m: status_lbl.config(
                                 text=m[:44], fg="#34d399" if ok else "#f87171"))), pady=6).pack(pady=2)

        ws_row = tk.Frame(self.root, bg=BG)
        ws_row.pack(pady=2)
        make_button(ws_row, "🔄 Compute Envelope", "#1d4ed8",
                    lambda: self.cli.call_service("workspace", self._on_ws(status_lbl)), width=18).grid(row=0, column=0, padx=2)
        make_button(ws_row, "👁 Show/Hide", "#334155", self._toggle_ws, width=11).grid(row=0, column=1, padx=2)
        status_lbl.pack()

        # Manual joint sliders for the 5 dorsal gantry/arm joints + parallel gripper
        make_label(self.root, "MANUAL ARM OVERRIDES").pack(pady=(8, 2))
        for name, lo, hi, res, idx in [("Gantry", -0.26, 0.26, 0.01, 0), ("Yaw", -3.14, 3.14, 0.05, 1),
                                       ("Shoulder", -1.6, 2.4, 0.05, 2), ("Elbow", -2.6, 2.6, 0.05, 3),
                                       ("Wrist", -2.0, 2.0, 0.05, 4), ("Gripper", 0.0, 0.022, 0.001, 5)]:
            make_slider_row(self.root, name, lo, hi, res, self.arm_cmd[idx], lambda v, j=idx: self._slide(j, v))

    def _tick(self):
        """Refresh the top telemetry status bar every 100 ms (10 Hz)."""
        text, color = format_telemetry(self.cli.telemetry)
        self.telem_lbl.config(text=text, fg=color)
        self.root.after(100, self._tick)

    def _on_ws(self, lbl):
        """Cache computed workspace wireframe JSON and render it in MuJoCo."""
        def _cb(ok, msg):
            if ok:
                self.cached_wf, self.wf_visible = msg, True
                self.cli.send_wireframe(msg)
                lbl.config(text="Wireframe rendered in MuJoCo", fg="#34d399")
            else:
                lbl.config(text=msg[:44], fg="#f87171")
        return _cb

    def _toggle_ws(self):
        """Show or hide the cached 3D workspace wireframe in MuJoCo."""
        if self.cached_wf:
            self.wf_visible = not self.wf_visible
            self.cli.send_wireframe(self.cached_wf if self.wf_visible else "")

    def _slide(self, idx: int, val: float):
        """Publish updated manual joint target whenever a slider moves."""
        self.arm_cmd[idx] = val
        # Mirror gripper finger opening across both left (5) and right (6) fingers
        if idx == 5:
            self.arm_cmd[6] = val
        self.cli.send_arm(self.arm_cmd)


def main():
    """Initialize ROS 2 client node and run the Tkinter UI event loop."""
    rclpy.init()
    client = FarmDog6ROSClient()
    try:
        ControlPanelApp(client).root.mainloop()
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
