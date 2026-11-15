#!/usr/bin/env python3
import os
import sys
import tkinter as tk
import rclpy

sys.path.insert(0, os.path.dirname(__file__))
from controllers.ros_client import FarmDog6ROSClient


class ControlPanelApp:
    """Compact Tkinter operator UI backed by FarmDog6ROSClient."""

    def __init__(self, client: FarmDog6ROSClient):
        self.cli = client
        self.arm_cmd = [0.0, 0.0, 0.1, -1.8, 1.4, 0.0, 0.0]
        self.cached_wf = ""
        self.wf_visible = False

        self.root = tk.Tk()
        self.root.title("KisaanuNexusFarmDog6 Console")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#1e242b")
        self._build_layout()

    def _lbl(self, text, fg="#e6edf3", size=10, bold=True):
        font = ("Sans", size, "bold") if bold else ("Sans", size)
        return tk.Label(self.root, text=text, bg="#1e242b", fg=fg, font=font)

    def _build_layout(self):
        self._lbl("LOCOMOTION (TAP = 1 STEP | HOLD = WALK)").pack(pady=(8, 2))
        pad = tk.Frame(self.root, bg="#1e242b")
        pad.pack()

        dirs = [("▲ Forward", 0, 1, 1.0, 0.0), ("◄ Left", 1, 0, 0.0, 1.0),
                ("Right ►", 1, 2, 0.0, -1.0), ("▼ Backward", 2, 1, -1.0, 0.0)]
        for text, r, c, vx, vy in dirs:
            b = tk.Button(pad, text=text, bg="#2d6a4f", fg="white", font=("Sans", 9, "bold"), width=10, pady=5, bd=0)
            b.grid(row=r, column=c, padx=3, pady=3)
            b.bind("<ButtonPress-1>", lambda e, x=vx, y=vy: self.cli.send_vel(x, y, True))
            b.bind("<ButtonRelease-1>", lambda e: self.cli.send_vel(0.0, 0.0, False))

        tk.Button(pad, text="■ STOP", command=lambda: self.cli.call_service("stop"),
                  bg="#b5173c", fg="white", font=("Sans", 9, "bold"), width=10, pady=5, bd=0).grid(row=1, column=1)

        mode_lbl = self._lbl("Mode: RL Balance Policy", "#34d399", 8, False)
        mode_btn = tk.Button(self.root, text="🤖 Switch Gait (RL ↔ Tripod)", bg="#0f766e", fg="white",
                             font=("Sans", 9, "bold"), width=30, pady=5, bd=0)
        mode_btn.config(command=lambda: self.cli.call_service(
            "mode", lambda ok, msg: (mode_lbl.config(text=msg[:44], fg="#34d399" if ok else "#f87171"),
                                     mode_btn.config(bg="#0f766e" if "RL" in msg else "#b45309"))))
        mode_btn.pack(pady=(4, 1))
        mode_lbl.pack()

        self._lbl("IK HARVEST & WORK ENVELOPE").pack(pady=(8, 2))
        status_lbl = self._lbl("Services Ready", "#94a3b8", 8, False)
        tk.Button(self.root, text="🌾 Auto-Harvest (IK Pick & Basket)", bg="#7c3aed", fg="white",
                  font=("Sans", 9, "bold"), width=30, pady=6, bd=0,
                  command=lambda: (self.cli.call_service("stop"),
                                   self.cli.call_service("harvest", lambda ok, m: status_lbl.config(
                                       text=m[:44], fg="#34d399" if ok else "#f87171")))).pack(pady=2)

        ws_row = tk.Frame(self.root, bg="#1e242b")
        ws_row.pack(pady=2)
        tk.Button(ws_row, text="🔄 Compute Envelope", bg="#1d4ed8", fg="white", font=("Sans", 9, "bold"), width=18, pady=5, bd=0,
                  command=lambda: self.cli.call_service("workspace", self._on_ws_ready(status_lbl))).grid(row=0, column=0, padx=2)
        tk.Button(ws_row, text="👁 Show/Hide", bg="#334155", fg="white", font=("Sans", 9, "bold"), width=11, pady=5, bd=0,
                  command=self._toggle_ws).grid(row=0, column=1, padx=2)
        status_lbl.pack()

        self._lbl("MANUAL ARM OVERRIDES").pack(pady=(8, 2))
        specs = [("Gantry", -0.26, 0.26, 0.01, 0), ("Yaw", -3.14, 3.14, 0.05, 1), ("Shoulder", -1.6, 2.4, 0.05, 2),
                 ("Elbow", -2.6, 2.6, 0.05, 3), ("Wrist", -2.0, 2.0, 0.05, 4), ("Gripper", 0.0, 0.022, 0.001, 5)]
        for name, lo, hi, res, idx in specs:
            row = tk.Frame(self.root, bg="#1e242b")
            row.pack(fill="x", padx=10)
            tk.Label(row, text=name, bg="#1e242b", fg="#cbd5e1", width=9, anchor="w", font=("Sans", 8)).pack(side="left")
            s = tk.Scale(row, from_=lo, to=hi, resolution=res, orient="horizontal", length=185, bg="#1e242b",
                         fg="white", highlightthickness=0, command=lambda v, j=idx: self._on_slide(j, float(v)))
            s.set(self.arm_cmd[idx])
            s.pack(side="right")

    def _on_ws_ready(self, lbl):
        def _cb(ok, msg):
            if ok:
                self.cached_wf, self.wf_visible = msg, True
                self.cli.send_wireframe(msg)
                lbl.config(text="Wireframe rendered in MuJoCo", fg="#34d399")
            else:
                lbl.config(text=msg[:44], fg="#f87171")
        return _cb

    def _toggle_ws(self):
        if self.cached_wf:
            self.wf_visible = not self.wf_visible
            self.cli.send_wireframe(self.cached_wf if self.wf_visible else "")

    def _on_slide(self, idx: int, val: float):
        self.arm_cmd[idx] = val
        if idx == 5:
            self.arm_cmd[6] = val
        self.cli.send_arm(self.arm_cmd)


def main():
    rclpy.init()
    client = FarmDog6ROSClient()
    try:
        ControlPanelApp(client).root.mainloop()
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
