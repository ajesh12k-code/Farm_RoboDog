#!/usr/bin/env python3
"""Atharv Nandi arm arbitration node: RL balance targets, manual override, or IK auto-harvest.

Modes (/nandi/harvest_status "mode"):
    rl_balance  arm follows /nandi/rl_arm_targets (locomotion); default
    manual      arm follows /nandi/manual_arm until /nandi/release_arm
    ik_harvest  HarvestSequence (atharv_nandi/core/harvest_logic.py) owns the arm in a worker thread
    tool_change WP2 rack swap running: /nandi/tool/arm_targets (no feed-forward) + gravity offsets are forwarded
                to /nandi/arm_targets (WP8, harvest_logic.tool_arm_command); back to rl_balance when the swap ends

Threading: rclpy callbacks run on the executor thread, the harvest sequence on its own thread.
All shared state is guarded by self.lock; everything is published from the executor thread (timer).
"""
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from atharv_nandi import node_params  # noqa: E402
from atharv_nandi.core import harvest_logic as hl
from atharv_nandi.core.arm_ik_solver import DorsalArmIKSolver
from atharv_nandi.core.farm_scene import CROP_RADIUS, crop_body_names, load_farm_model

PUBLISH_DT = hl.PUBLISH_DT
SLEW_PER_TICK = hl.SLEW_PER_TICK  # max change of the published arm command per tick (see harvest_logic)


class _NodeIO:
    """HarvestSequence I/O bound to the node: latest qpos in, arm goal out, abortable waits."""

    def __init__(self, node: "HarvestIKServiceNode", abort: threading.Event):
        self.node, self.abort = node, abort

    def qpos(self) -> np.ndarray:
        with self.node.lock:
            return self.node.latest_qpos.copy()

    def command(self, arm7):
        with self.node.lock:
            if self.abort.is_set():
                raise hl.HarvestAborted()
            self.node.arm_goal = list(arm7)

    def wait(self, seconds: float):
        if self.abort.wait(seconds):
            raise hl.HarvestAborted()

    def phase(self, name: str):
        with self.node.lock:
            if not self.abort.is_set():
                self.node.phase = name
                self.node.status_dirty = True


class HarvestIKServiceNode(Node):
    def __init__(self):
        super().__init__("nandi_harvest_ik_service")
        node_params.declare(self, "nandi_harvest_ik_service")  # WPR: config/nandi.yaml + NANDI_* env (env wins)
        model = load_farm_model()
        self.ik = DorsalArmIKSolver(model)  # used only by the (single) harvest thread
        # the swap forwarding (executor thread) computes gravity offsets on its own solver: DorsalArmIKSolver writes
        # its MjData in _fk / mj_rne, and an aborted harvest worker may still be inside self.ik.solve()
        self.tool_ik = DorsalArmIKSolver(model)
        self.crops = crop_body_names(model)
        self.nq = model.nq

        self.lock = threading.Lock()
        self.latest_qpos = self.ik.data.qpos.copy()  # home keyframe until sim_runner publishes
        self.have_qpos = False
        self.arm_goal = list(hl.REST_ARM)
        self.arm_out = list(hl.REST_ARM)
        self.mode, self.phase, self.target, self.last_result = "rl_balance", "idle", None, ""
        self.abort = threading.Event()
        self.worker = None  # last harvest thread; a new run waits until it has fully exited
        self.status_dirty = True

        opening = self.ik.gripper_opening()
        if opening < 2 * CROP_RADIUS:
            self.get_logger().warn(
                f"Gripper opens {opening * 100:.1f} cm but crops are {2 * CROP_RADIUS * 100:.1f} cm wide: "
                "grasps will fail (and be reported as failed); check the finger geometry in the robot MJCF."
            )

        self.create_subscription(Float64MultiArray, "/nandi/sim_qpos", self.on_qpos, 10)
        self.create_subscription(Float64MultiArray, "/nandi/rl_arm_targets", self.on_rl_arm, 10)
        self.create_subscription(Float64MultiArray, "/nandi/manual_arm", self.on_manual, 10)
        self.create_subscription(String, "/nandi/tool/arm_targets", self.on_tool_arm, 10)
        self.tool_seen = 0.0
        self.arm_pub = self.create_publisher(Float64MultiArray, "/nandi/arm_targets", 10)
        self.status_pub = self.create_publisher(String, "/nandi/harvest_status", 10)
        self.create_service(Trigger, "/nandi/auto_harvest", self.on_auto_harvest)
        self.create_service(Trigger, "/nandi/release_arm", self.on_release_arm)
        self.create_timer(PUBLISH_DT, self.on_tick)
        self.create_timer(0.5, self.on_status_timer)
        self.get_logger().info(f"Harvest IK ready: {len(self.crops)} crops, nq={self.nq}.")

    # ------------------------------------------------------------------ subscriptions
    def on_qpos(self, msg: Float64MultiArray):
        if len(msg.data) != self.nq:
            return
        q = np.array(msg.data, dtype=np.float64)
        with self.lock:
            reset = self.have_qpos and hl.world_reset_detected(self.latest_qpos, q)
            self.latest_qpos, self.have_qpos = q, True
            if reset:
                self._world_reset_locked()

    def _world_reset_locked(self):
        """The world jumped back to the keyframe (/nandi/reset_world): stand the arm down at rest.

        Snaps the slewed command too: slewing from the pre-reset pose would drive the freshly reset arm
        back out over the reset crops. sim_runner's release_arm call arrives later and is then a no-op.
        """
        if self.mode == "ik_harvest":
            self.abort.set()
            self.last_result = f"World reset: harvest of {self.target} aborted"
        self.mode, self.phase, self.target = "rl_balance", "idle", None
        self.arm_goal = list(hl.REST_ARM)
        self.arm_out = list(hl.REST_ARM)  # == home keyframe arm ctrl
        self.status_dirty = True

    def on_rl_arm(self, msg: Float64MultiArray):
        if len(msg.data) != 5:
            return
        with self.lock:
            if self.mode == "tool_change" and self.get_clock().now().nanoseconds * 1e-9 - self.tool_seen > 0.5:
                self.mode, self.phase = "rl_balance", "idle"      # the swap topic went quiet: swap over
                self.status_dirty = True
            if self.mode == "rl_balance":
                self.arm_goal = [*msg.data, self.ik.grip_open, self.ik.grip_open]

    def on_manual(self, msg: Float64MultiArray):
        if len(msg.data) != 7:
            return
        with self.lock:
            if self.mode == "ik_harvest":  # operator override wins over a running harvest
                self.abort.set()
                self.last_result = f"Manual override: harvest of {self.target} aborted"
            if self.mode != "manual":
                self.mode, self.phase, self.target = "manual", "idle", None
                self.status_dirty = True
            self.arm_goal = list(msg.data)

    def on_tool_arm(self, msg: String):
        """WP2 swap arm targets (50 Hz while a swap runs). A running harvest or the manual arm wins (the swap
        interlocks refuse a change while holding; the operator owns manual). While the swap owns the arm
        (``tool_change``) /nandi/auto_harvest is refused. Gravity offsets come from ``self.tool_ik`` (never the
        harvest worker's solver)."""
        with self.lock:
            if self.mode in ("ik_harvest", "manual"):
                return
            arm7 = hl.tool_arm_command(msg.data, self.tool_ik, self.latest_qpos,
                                       (self.tool_ik.grip_open, self.tool_ik.grip_open))
            if arm7 is None:
                if self.mode == "tool_change":
                    self.mode, self.phase = "rl_balance", "idle"
                    self.arm_goal = list(hl.REST_ARM)
                    self.status_dirty = True
                return
            if self.mode != "tool_change":
                self.mode, self.phase, self.target = "tool_change", "swap", None
                self.status_dirty = True
            self.arm_goal = arm7
            self.arm_out = list(arm7)      # the swap already rate-limits its moves (WP2 ToolChanger)
            self.tool_seen = self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ services
    def on_release_arm(self, request, response):
        with self.lock:
            prev = self.mode
            if prev == "ik_harvest":
                self.abort.set()
                self.last_result = f"Released: harvest of {self.target} aborted"
            self.mode, self.phase, self.target = "rl_balance", "idle", None
            self.arm_goal = list(hl.REST_ARM)  # locomotion's rl_arm_targets take over from here
            self.status_dirty = True
        response.success = True
        response.message = {
            "rl_balance": "Arm already in RL balance mode",
            "manual": "Manual arm released to RL balance",
            "ik_harvest": "Harvest aborted; arm released to RL balance",
            "tool_change": "Tool swap arm targets released to RL balance",
        }[prev]
        return response

    def on_auto_harvest(self, request, response):
        with self.lock:
            if self.mode == "ik_harvest":
                response.success, response.message = False, f"Harvest already running ({self.target})"
                return response
            if self.mode == "tool_change":
                # a WP2 rack swap owns the arm mid-dock; taking it now would pull the tool out of the rack
                response.success, response.message = False, "Tool change running - try again when the swap ends"
                return response
            if self.worker is not None and self.worker.is_alive():
                # An aborted run can still be inside ik.solve(); the solver's MjData is not thread-safe.
                response.success, response.message = False, "Previous harvest still stopping - try again"
                return response
            if not self.have_qpos:
                response.success, response.message = False, "No sim state yet (is sim_runner running?)"
                return response
            target, message = hl.select_target(self.latest_qpos, self.crops)
            if target is None:
                self.last_result, self.status_dirty = message, True
                response.success, response.message = False, message
                return response
            self.abort = abort = threading.Event()
            self.mode, self.phase, self.target = "ik_harvest", "approach", target
            self.status_dirty = True
            start_cmd = list(self.arm_out)
            self.worker = threading.Thread(target=self._run_harvest, args=(target, abort, start_cmd), daemon=True)
            self.worker.start()
        response.success, response.message = True, message
        return response

    def _run_harvest(self, target: str, abort: threading.Event, start_cmd):
        try:
            ok, message = hl.HarvestSequence(self.ik, _NodeIO(self, abort)).run(target, start_cmd)
        except Exception as exc:  # never leave the arm stuck in ik_harvest
            ok, message = False, f"Harvest error: {exc}"
        with self.lock:
            if abort.is_set():  # aborted: release_arm / manual already own the arm and the status
                return
            abort.set()
            self.mode, self.phase, self.target, self.last_result = "rl_balance", "idle", None, message
            self.arm_goal = list(hl.REST_ARM)
            self.status_dirty = True
        # rclpy fixes the severity per call site, so info and warn need separate call sites
        if ok:
            self.get_logger().info(message)
        else:
            self.get_logger().warn(message)

    # ------------------------------------------------------------------ timers (publish here only)
    def on_tick(self):
        with self.lock:
            self.arm_out = hl.slew(self.arm_out, self.arm_goal, SLEW_PER_TICK)
            out = self.arm_out
            status = None
            if self.status_dirty:
                status = hl.status_json(self.mode, self.phase, self.target, self.last_result)
                self.status_dirty = False
        self.arm_pub.publish(Float64MultiArray(data=out))
        if status is not None:
            self.status_pub.publish(String(data=status))

    def on_status_timer(self):
        with self.lock:
            self.status_dirty = True


def main():
    rclpy.init()
    node = HarvestIKServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
