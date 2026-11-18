#!/usr/bin/env python3
import queue  # Import queue to hand ROS worker-thread results to the Tk thread
import tkinter as tk  # Import Tkinter for window and event loop management
from tkinter import messagebox  # Confirm dialog for Reset E-STOP
import rclpy  # Import rclpy for ROS 2 initialization and shutdown
from std_srvs.srv import Trigger  # E-STOP / E-STOP reset service clients

from atharv_nandi.core.ros_client import NandiROSClient  # Import decoupled ROS 2 bridge
from atharv_nandi.core.key_drive import (  # Import pure keyboard / D-pad / gait-slider logic
    GAIT_DEFAULTS, GAIT_PARAMS, KEY_HELP, PAD_PREFIX, KeyDrive, action_for_key, clamp_gait_params,
    drive_axis, normalize_key,
)
from atharv_nandi.core.telemetry import (  # Import pure telemetry / status formatters
    BAD_RED, OK_GREEN, format_game, format_harvest, format_mode, format_telemetry, shorten,
)
from atharv_nandi.core.gui_widgets import (  # Import UI helpers
    BG, make_button, make_label, make_scrollable, make_slider_row, set_button_dimmed,
)

PANEL_W = 370  # content width; fits beside the MuJoCo viewer on a 1600x900 desktop
ARM_REST = [0.0, 0.0, 0.1, -1.8, 1.4, 0.0, 0.0]  # stowed dorsal arm (5) + gripper fingers (2)
MODE_BTN_BG = "#0f766e"
ESTOP_BG = "#dc2626"
CONTROL_MASK = 0x0004  # Tk event.state bit for Ctrl: Ctrl+<key> is never a robot command


class ControlPanelApp:
    """Compact declarative Tkinter console for Atharv Nandi."""

    def __init__(self, client: NandiROSClient):
        self.cli = client
        # E-STOP / reset go to the safety supervisor (optional package); the GUI adds the clients itself
        self.cli.service_clients.setdefault("estop", self.cli.create_client(Trigger, "/nandi/estop"))
        self.cli.service_clients.setdefault("estop_reset", self.cli.create_client(Trigger, "/nandi/estop_reset"))
        # Manual arm targets mirrored by the sliders (start at the stowed rest pose)
        self.arm_cmd = list(ARM_REST)
        self.arm_scales = []
        # Gait parameters mirrored by the sliders; synced once from the first locomotion_status
        self.gait_cmd = list(GAIT_DEFAULTS)
        self.gait_scales = []
        self.gait_synced = False
        # Cached workspace wireframe JSON and visibility toggle state
        self.cached_wf, self.wf_visible = "", False
        # Debounced keyboard + D-pad state -> cmd_vel publisher
        self.keys = KeyDrive()
        # Service callbacks arrive on worker threads; Tk widgets are only touched from _tick
        self._ui_queue = queue.SimpleQueue()
        self._mode_view = None
        self._actions = {
            "stop": self._stop, "harvest": self._harvest, "reset": self._reset_world,
            "gait": self._toggle_gait, "release": self._release_arm,
        }

        # Configure always-on-top dark operator window docked at the right edge of the screen
        self.root = tk.Tk()
        self.root.title("Atharv Nandi Console")
        self.root.attributes("-topmost", True)
        self.root.configure(bg=BG)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        win_w, win_h = PANEL_W + 20, min(880, sh - 40)
        self.root.geometry(f"{win_w}x{win_h}+{max(0, sw - win_w - 4)}+0")

        # Big E-STOP pinned above the scrolling panel so it is always visible
        estop_row = tk.Frame(self.root, bg=BG)
        estop_row.pack(side="top", fill="x", padx=8, pady=(6, 0))
        make_button(estop_row, "E-STOP", ESTOP_BG, self._estop, width=20, pady=8, size=12).pack(side="left", fill="x", expand=True)
        make_button(estop_row, "Reset E-STOP", "#475569", self._estop_reset, width=12).pack(side="left", padx=(6, 0))
        self._build_ui(make_scrollable(self.root, PANEL_W))
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.focus_set()
        # Start 10 Hz UI refresh and 20 Hz drive-command loop
        self._tick()
        self._drive_tick()

    # ---- layout -------------------------------------------------------------------------------
    def _build_ui(self, ui: tk.Frame):
        """Assemble telemetry/score panel, D-pad, gait controls, IK harvest, and manual sliders."""
        # Top telemetry + score status panel
        bar = tk.Frame(ui, bg="#0f172a", pady=4)
        bar.pack(fill="x", padx=8, pady=(8, 4))
        self.telem_lbl = make_label(bar, "Dist: +0.00 m  |  Height: 0.00 m  |  Tilt: 0.0°", "#38bdf8", 9)
        self.telem_lbl.pack()
        self.game_lbl = make_label(bar, "Score: waiting for sim...", "#94a3b8", 9)
        self.game_lbl.pack()
        self.arm_lbl = make_label(bar, "Arm: waiting for harvest node...", "#94a3b8", 8, False)
        self.arm_lbl.pack()
        self.last_lbl = make_label(bar, "", "#94a3b8", 8, False)
        self.last_lbl.pack()

        # Locomotion D-pad (press-and-hold walks continuously; quick tap takes 1 step)
        make_label(ui, "LOCOMOTION (TAP = 1 STEP | HOLD = WALK)").pack(pady=(6, 2))
        pad = tk.Frame(ui, bg=BG)
        pad.pack()
        for text, r, c, key in [("↺ Turn [Q]", 0, 0, "q"), ("▲ Fwd [W]", 0, 1, "w"), ("Turn [E] ↻", 0, 2, "e"),
                                ("◄ Left [A]", 1, 0, "a"), ("Right [D] ►", 1, 2, "d"), ("▼ Back [S]", 2, 1, "s")]:
            b = make_button(pad, text, "#2d6a4f", width=11)
            b.grid(row=r, column=c, padx=3, pady=2)
            b.bind("<ButtonPress-1>", lambda e, k=key: self._pad(k, True))
            b.bind("<ButtonRelease-1>", lambda e, k=key: self._pad(k, False))
        make_button(pad, "■ STOP", "#b5173c", self._stop, width=11).grid(row=1, column=1, padx=3, pady=2)
        act = tk.Frame(ui, bg=BG)
        act.pack(pady=(2, 0))
        make_button(act, "Release Arm [Esc]", "#475569", self._release_arm, width=17).grid(row=0, column=0, padx=2)
        make_button(act, "Reset World [R]", "#9a3412", self._reset_world, width=17).grid(row=0, column=1, padx=2)
        help_lbl = make_label(ui, KEY_HELP + "  (click this panel to focus)", "#64748b", 7, False)
        help_lbl.config(wraplength=PANEL_W - 20, justify="center")
        help_lbl.pack(pady=(3, 0))

        # Gait mode (truthful, from /nandi/locomotion_status) and gait tuning sliders
        self.mode_btn = make_button(ui, "Switch Gait (RL ↔ Tripod) [G]", MODE_BTN_BG, self._toggle_gait)
        self.mode_btn.pack(pady=(6, 1))
        self.mode_lbl = make_label(ui, "Locomotion: waiting for status...", "#94a3b8", 8, False)
        self.mode_lbl.pack()
        make_label(ui, "GAIT PARAMETERS", size=9).pack(pady=(4, 0))
        for i, spec in enumerate(GAIT_PARAMS):
            self.gait_scales.append(make_slider_row(ui, spec.label, spec.lo, spec.hi, spec.resolution,
                                                    self.gait_cmd[i], lambda v, j=i: self._gait_slide(j, v)))

        # Autonomous IK Pick-and-Place and 3D Work Envelope buttons
        make_label(ui, "IK HARVEST & WORK ENVELOPE").pack(pady=(8, 2))
        make_button(ui, "Auto-Harvest Nearest Crop [H]", "#7c3aed", self._harvest, pady=5).pack(pady=2)
        ws_row = tk.Frame(ui, bg=BG)
        ws_row.pack(pady=2)
        make_button(ws_row, "Compute Envelope", "#1d4ed8", self._compute_ws, width=18).grid(row=0, column=0, padx=2)
        make_button(ws_row, "Show/Hide", "#334155", self._toggle_ws, width=11).grid(row=0, column=1, padx=2)
        self.status_lbl = make_label(ui, "Services Ready", "#94a3b8", 8, False)
        self.status_lbl.pack()

        # Manual joint sliders for the 5 dorsal gantry/arm joints + parallel gripper
        make_label(ui, "MANUAL ARM OVERRIDES").pack(pady=(8, 2))
        for name, lo, hi, res, idx in [("Gantry", -0.26, 0.26, 0.01, 0), ("Yaw", -3.14, 3.14, 0.05, 1),
                                       ("Shoulder", -1.6, 2.4, 0.05, 2), ("Elbow", -2.6, 2.6, 0.05, 3),
                                       ("Wrist", -2.0, 2.0, 0.05, 4), ("Gripper", 0.0, 0.04, 0.001, 5)]:
            self.arm_scales.append(make_slider_row(ui, name, lo, hi, res, self.arm_cmd[idx],
                                                   lambda v, j=idx, r=res: self._slide(j, v, r)))
        tk.Frame(ui, bg=BG, height=8).pack()

    def _bind_keys(self):
        """Route every key event in the window (root bindtag) through the KeyDrive state."""
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", lambda e: self.keys.release(e.keysym))
        # Losing focus means KeyRelease events stop arriving: drop held keys so the robot stops
        self.root.bind("<FocusOut>", lambda e: self.keys.clear())

    def _on_close(self):
        """Release any held walk before the window goes away (dead-man would stop it in 0.5 s anyway)."""
        self.keys.clear()
        self._send_drive()
        self.root.destroy()

    # ---- periodic loops ------------------------------------------------------------------------
    def _tick(self):
        """Every 100 ms: run queued service callbacks and refresh telemetry/status widgets."""
        while True:
            try:
                fn = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            fn()

        text, color = format_telemetry(self.cli.telemetry)
        self.telem_lbl.config(text=text, fg=color)
        text, color = format_game(self.cli.game_status)
        self.game_lbl.config(text=text, fg=color)
        line, last, color = format_harvest(self.cli.harvest_status)
        self.arm_lbl.config(text=shorten(line), fg=color)
        self.last_lbl.config(text=shorten(last))

        loco = self.cli.loco_status
        view = format_mode(loco)
        if view != self._mode_view:
            self._mode_view = view
            self.mode_lbl.config(text=view.label, fg=view.color)
            set_button_dimmed(self.mode_btn, not view.toggle_enabled, MODE_BTN_BG)
        if loco["ok"] and not self.gait_synced:
            self._set_gait_sliders(clamp_gait_params(loco["gait_params"]))
            self.gait_synced = True
        self.root.after(100, self._tick)

    def _drive_tick(self):
        """Every 50 ms: publish cmd_vel on change and re-publish held commands at 5 Hz."""
        self._send_drive()
        self.root.after(50, self._drive_tick)

    def _send_drive(self):
        cmd = self.keys.poll()
        if cmd is not None:
            self.cli.send_drive(cmd)

    # ---- input handlers ------------------------------------------------------------------------
    def _on_key_press(self, event):
        key = normalize_key(event.keysym)
        if event.state & CONTROL_MASK or not self.keys.press(key):
            return  # Ctrl shortcut, or X11 autorepeat of a key already held
        action = action_for_key(key)
        if action:
            self._actions[action]()
        elif drive_axis(key):
            self._send_drive()  # react immediately instead of waiting for the next tick

    def _pad(self, key: str, pressed: bool):
        """D-pad button press/release: same semantics as the keyboard (mouse needs no debounce)."""
        if pressed:
            self.keys.press(PAD_PREFIX + key)
        else:
            self.keys.release(PAD_PREFIX + key, immediate=True)
        self._send_drive()

    def _gait_slide(self, idx: int, val: float):
        """Publish /nandi/gait_params when the operator moves a gait slider."""
        if abs(val - self.gait_cmd[idx]) < GAIT_PARAMS[idx].resolution / 2:
            return  # programmatic set() echo, not an operator change
        self.gait_cmd[idx] = val
        self.cli.send_gait_params(self.gait_cmd)

    def _set_gait_sliders(self, values):
        self.gait_cmd = list(values)
        for scale, v in zip(self.gait_scales, values):
            scale.set(v)

    def _slide(self, idx: int, val: float, res: float):
        """Publish updated manual joint target whenever the operator moves an arm slider."""
        if abs(val - self.arm_cmd[idx]) < res / 2:
            return  # programmatic set() echo: do not grab the arm into manual mode
        self.arm_cmd[idx] = val
        # Mirror gripper finger opening across both left (5) and right (6) fingers
        if idx == 5:
            self.arm_cmd[6] = val
        self.cli.send_arm(self.arm_cmd)

    def _reset_arm_sliders(self):
        """Show the rest pose on the arm sliders after the arm is released/reset."""
        self.arm_cmd = list(ARM_REST)
        for scale, v in zip(self.arm_scales, ARM_REST):
            scale.set(v)

    # ---- actions (buttons and keys) ------------------------------------------------------------
    def _post(self, fn):
        self._ui_queue.put(fn)

    def _call(self, key: str, title: str, then=None):
        """Call a Trigger service; show '<title>: <message>' and run then(ok) on the Tk thread."""
        def _done(ok, msg):
            def _apply():
                self.status_lbl.config(text=shorten(f"{title}: {msg}"), fg=OK_GREEN if ok else BAD_RED)
                if then:
                    then(ok)
            self._post(_apply)
        self.cli.call_service(key, _done)

    def _stop(self):
        """STOP: halt the gait now and hand the arm back to RL balance (aborts any harvest)."""
        self.keys.stop_driving()
        self._send_drive()
        self._call("stop", "Stop")
        self._call("release", "Release arm", lambda ok: ok and self._reset_arm_sliders())

    def _estop(self):
        """E-STOP: stop the gait now and latch the safety supervisor; without one, also abort the arm."""
        self.keys.stop_driving()
        self._send_drive()
        self._call("stop", "Stop")

        def _fallback(ok):
            if not ok:  # /nandi/estop offline: at least abort a running harvest
                self._call("release", "E-STOP fallback: release arm", lambda ok2: ok2 and self._reset_arm_sliders())
        self._call("estop", "E-STOP", _fallback)

    def _estop_reset(self):
        if messagebox.askyesno("Reset E-STOP", "Clear the E-STOP latch? The robot may move again.", parent=self.root):
            self._call("estop_reset", "Reset E-STOP")

    def _release_arm(self):
        self._call("release", "Release arm", lambda ok: ok and self._reset_arm_sliders())

    def _harvest(self):
        """Stop walking first, then start the async auto-harvest sequence."""
        self.keys.stop_driving()
        self._send_drive()
        self._call("stop", "Stop", lambda ok: self._call("harvest", "Harvest"))

    def _reset_world(self):
        """Reset robot + crops to the keyframe; sim_runner also clears the wireframe."""
        self.keys.stop_driving()
        self._send_drive()

        def _after(ok):
            if ok:
                self.wf_visible = False
                self._reset_arm_sliders()
        self._call("reset", "Reset", _after)

    def _toggle_gait(self):
        """Ask locomotion to toggle RL/Tripod; the label itself follows locomotion_status."""
        self._call("mode", "Gait")

    def _compute_ws(self):
        """Compute the work envelope, cache the wireframe JSON and render it in MuJoCo."""
        def _done(ok, msg):
            def _apply():
                if ok:
                    self.cached_wf, self.wf_visible = msg, True
                    self.cli.send_wireframe(msg)
                    self.status_lbl.config(text="Wireframe rendered in MuJoCo", fg=OK_GREEN)
                else:
                    self.status_lbl.config(text=shorten(f"Envelope: {msg}"), fg=BAD_RED)
            self._post(_apply)
        self.cli.call_service("workspace", _done)

    def _toggle_ws(self):
        """Show or hide the cached 3D workspace wireframe in MuJoCo (and in every browser)."""
        self.cached_wf = self.cached_wf or self.cli.wireframe_json  # may have been computed from the web UI
        if self.cached_wf:
            # Follow the topic, not a local flag: the web UI and reset_world also show/clear it
            self.wf_visible = not self.cli.wireframe_shown
            self.cli.send_wireframe(self.cached_wf if self.wf_visible else "")


def main():
    """Initialize ROS 2 client node and run the Tkinter UI event loop."""
    rclpy.init()
    client = NandiROSClient()
    try:
        ControlPanelApp(client).root.mainloop()
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
