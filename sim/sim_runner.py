#!/usr/bin/env python3
"""Atharv Nandi physics + MuJoCo passive viewer, paced to wall-clock.

Threads:
    rclpy spin thread  - subscriptions + services; only touches the small shared state below,
                         always under self._lock.
    sim loop (main)    - owns MjModel/MjData exclusively: steps, publishes, renders, resets.
"""
import json
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from controllers.farm_scene import crop_body_names, crop_qpos_slice, load_farm_model
from controllers.game_status import FallCounter, RtfMeter, build_game_status, is_fallen, plan_catch_up

QPOS_PUB_PERIOD_S = 0.02     # /nandi/sim_qpos at 50 Hz of sim time
VIEWER_PERIOD_S = 1.0 / 30   # viewer.sync at most ~30 Hz wall
STATUS_PERIOD_S = 0.5        # /nandi/game_status at 2 Hz wall
MAX_CATCHUP_S = 0.05         # at most 50 ms of sim per loop pass; older debt is dropped
RESET_TIMEOUT_S = 2.0
# After a reset, ignore /nandi/arm_targets this long: the harvest node keeps publishing its
# pre-reset (slewed) arm command until it sees the reset in /nandi/sim_qpos and snaps to rest.
ARM_RESET_HOLD_S = 0.3


class NandiSimRunner(Node):
    """MuJoCo physics & 3D renderer subscribing to modular service outputs."""

    def __init__(self):
        super().__init__("nandi_sim_runner")
        self.model = load_farm_model()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.crop_names = crop_body_names(self.model)
        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.home_ctrl = np.array(self.model.key_ctrl[0], dtype=np.float64)

        # Shared with the spin thread (guarded by _lock).
        self._lock = threading.Lock()
        self._arm_targets = self.home_ctrl[:7].copy()
        self._arm_hold_until = 0.0        # time.monotonic() before which arm_targets are ignored
        self._leg_targets = self.home_ctrl[7:].copy()
        self._wireframe = None            # (canopy_segs, floor_segs) or None
        self._reset_requests = []         # threading.Event per pending /reset_world call

        # Owned by the sim loop.
        self.falls = FallCounter()
        self.rtf = RtfMeter()

        self.create_subscription(Float64MultiArray, "/nandi/arm_targets", self.on_arm, 10)
        self.create_subscription(Float64MultiArray, "/nandi/leg_targets", self.on_legs, 10)
        self.create_subscription(String, "/nandi/workspace_wireframe", self.on_wireframe, 10)
        self.qpos_pub = self.create_publisher(Float64MultiArray, "/nandi/sim_qpos", 10)
        self.status_pub = self.create_publisher(String, "/nandi/game_status", 10)
        # Reset clears the envelope for every viewer (Tk panel, browsers), not just this one.
        self.wf_pub = self.create_publisher(String, "/nandi/workspace_wireframe", 10)
        self.create_service(Trigger, "/nandi/reset_world", self.on_reset_world)
        # After a reset, ask the controllers to stand down so the robot stays at home.
        self.stop_motion_cli = self.create_client(Trigger, "/nandi/stop_motion")
        self.release_arm_cli = self.create_client(Trigger, "/nandi/release_arm")
        self.get_logger().info(
            f"World: nq={self.model.nq}, {len(self.crop_names)} crops, dt={self.model.opt.timestep}"
        )

    # ---------------- spin-thread callbacks ----------------
    def on_arm(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            with self._lock:
                if time.monotonic() < self._arm_hold_until:
                    return  # stale pre-reset command
                self._arm_targets = np.array(msg.data, dtype=np.float64)

    def on_legs(self, msg: Float64MultiArray):
        if len(msg.data) == 18:
            with self._lock:
                self._leg_targets = np.array(msg.data, dtype=np.float64)

    def on_wireframe(self, msg: String):
        wf = None
        if msg.data:
            try:
                data = json.loads(msg.data)
                wf = (
                    np.array(data["canopy_segments"], dtype=np.float64).reshape(-1, 2, 3),
                    np.array(data["floor_segments"], dtype=np.float64).reshape(-1, 2, 3),
                )
            except (ValueError, KeyError, TypeError) as exc:
                self.get_logger().warn(f"Ignoring malformed workspace_wireframe: {exc}")
                return
        with self._lock:
            self._wireframe = wf

    def on_reset_world(self, request, response):
        done = threading.Event()
        with self._lock:
            self._reset_requests.append(done)
        if not done.wait(RESET_TIMEOUT_S):
            with self._lock:
                if done in self._reset_requests:
                    self._reset_requests.remove(done)
            response.success = False
            response.message = "Sim loop not responding - world not reset"
            return response
        self.wf_pub.publish(String(data=""))
        for cli in (self.stop_motion_cli, self.release_arm_cli):
            if cli.service_is_ready():
                cli.call_async(Trigger.Request())  # fire-and-forget; never block the executor
        response.success = True
        response.message = f"World reset: robot + {len(self.crop_names)} crops at home, falls/timer cleared"
        return response

    # ---------------- sim-loop helpers ----------------
    def _apply_reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        with self._lock:
            self._arm_targets = self.home_ctrl[:7].copy()
            self._arm_hold_until = time.monotonic() + ARM_RESET_HOLD_S
            self._leg_targets = self.home_ctrl[7:].copy()
            self._wireframe = None
        self.falls.reset()
        self.rtf.rebase()

    def _game_status(self) -> dict:
        qpos = self.data.qpos
        crops = {name: qpos[crop_qpos_slice(k)][:3] for k, name in enumerate(self.crop_names, start=1)}
        return build_game_status(
            crops, qpos[0:3], qpos[3:7], self.falls.falls, self.falls.fallen,
            self.data.time, self.rtf.rtf,
        )

    def _publish_status(self):
        self.status_pub.publish(String(data=json.dumps(self._game_status())))

    def _publish_qpos(self):
        msg = Float64MultiArray()
        msg.data = self.data.qpos.tolist()
        self.qpos_pub.publish(msg)

    def _render_wireframe(self, viewer, wireframe):
        scn = viewer.user_scn
        scn.ngeom = 0
        if wireframe is None:
            return
        canopy, floor = wireframe

        t_pos = self.data.xpos[self.trunk_id]
        t_rot = self.data.xmat[self.trunk_id].reshape(3, 3)
        c1 = (canopy[:, 0, :] @ t_rot.T) + t_pos
        c2 = (canopy[:, 1, :] @ t_rot.T) + t_pos
        f1 = (floor[:, 0, :] @ t_rot.T) + t_pos
        f2 = (floor[:, 1, :] @ t_rot.T) + t_pos

        cyan = np.array([0.15, 0.85, 0.95, 0.26], dtype=np.float32)
        red = np.array([1.0, 0.22, 0.22, 0.55], dtype=np.float32)
        z3, eye = np.zeros(3), np.eye(3).flatten()
        idx = 0
        for p1, p2 in zip(c1, c2):
            if idx >= scn.maxgeom - 10:
                break
            mujoco.mjv_initGeom(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_CAPSULE, z3, z3, eye, cyan)
            mujoco.mjv_connector(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0025, p1, p2)
            idx += 1
        for p1, p2 in zip(f1, f2):
            if idx >= scn.maxgeom - 10:
                break
            mujoco.mjv_initGeom(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_CAPSULE, z3, z3, eye, red)
            mujoco.mjv_connector(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0035, p1, p2)
            idx += 1
        scn.ngeom = idx

    # ---------------- sim loop ----------------
    def run_sim(self):
        dt = self.model.opt.timestep
        qpos_every = max(1, int(round(QPOS_PUB_PERIOD_S / dt)))
        max_steps = max(1, int(round(MAX_CATCHUP_S / dt)))

        steps = 0                     # physics steps since the last reset (sim time = steps * dt)
        anchor = time.perf_counter()  # wall time at which `steps` was 0
        last_view = last_status = -1e9

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and rclpy.ok():
                with self._lock:
                    pending, self._reset_requests = self._reset_requests, []
                if pending:
                    self._apply_reset()
                    steps, anchor = 0, time.perf_counter()
                    for ev in pending:
                        ev.set()
                    last_status = -1e9  # publish the fresh status right away

                now = time.perf_counter()
                n, dropped = plan_catch_up((now - anchor) - steps * dt, dt, max_steps)
                anchor += dropped  # forgive debt we cannot pay back

                if n:
                    with self._lock:
                        arm, legs = self._arm_targets, self._leg_targets
                    self.data.ctrl[:7] = arm
                    self.data.ctrl[7:] = legs
                    for _ in range(n):
                        mujoco.mj_step(self.model, self.data)
                        steps += 1
                        if steps % qpos_every == 0:  # steps restarts at 0 on reset, like sim time
                            self._publish_qpos()
                    qpos = self.data.qpos
                    self.falls.update(is_fallen(qpos[0:3], qpos[3:7]))

                now = time.perf_counter()
                self.rtf.update(self.data.time, now)
                if now - last_view >= VIEWER_PERIOD_S:
                    with self._lock:
                        wireframe = self._wireframe
                    self._render_wireframe(viewer, wireframe)
                    viewer.sync()
                    last_view = now
                if now - last_status >= STATUS_PERIOD_S:
                    self._publish_status()
                    last_status = now

                if n == 0:
                    # Not yet due: sleep until the next physics step (or viewer frame) is.
                    behind = (time.perf_counter() - anchor) - steps * dt
                    time.sleep(max(0.0, min(dt - behind, VIEWER_PERIOD_S)))


def main():
    rclpy.init()
    node = NandiSimRunner()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        node.run_sim()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
