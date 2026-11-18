#!/usr/bin/env python3
"""Atharv Nandi locomotion node: tripod gait (default) or the 23-DOF RL balance policy.

Thin ROS 2 shell; all logic lives in atharv_nandi/core/tripod_gait.py, atharv_nandi/core/locomotion_logic.py
and atharv_nandi/core/rl_balance_policy.py. See README / ROS contract for topic semantics.

Safety (README, "Safety"): cmd_vel is speed-capped (config/safety.yaml) and refused while
/nandi/safety_status says estopped (the gait stops on the rising edge); if /nandi/sim_qpos goes
stale for more than 0.5 s the gait stops and the last leg targets are held.
WP6 /nandi/motion_inhibit (atharv_nandi/sim/hooks/stability.py): a stance / walk / stop inhibit stops the gait and refuses
walking cmd_vel while set; the warning-field speed_scale (0.3) scales cmd_vel. A silent topic inhibits nothing.
"""
import json
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from atharv_nandi import node_params  # noqa: E402
from atharv_nandi.core.locomotion_logic import (  # noqa: E402
    GaitParams, HoldDeadman, LocomotionModeSelector, LocomotionSafetyGate, MotionInhibitGate, locomotion_status_json,
    parse_cmd_vel, quat_to_roll_pitch,
)
from atharv_nandi.core.safety_logic import heartbeat_json, parse_safety_status  # noqa: E402
from atharv_nandi.core.rl_balance_policy import RLBalanceController  # noqa: E402
from atharv_nandi.core.tripod_gait import MAX_JOINT_SPEED_RPS, TripodGaitGenerator, rate_limit  # noqa: E402


class LocomotionServiceNode(Node):
    def __init__(self):
        super().__init__("nandi_locomotion_service")
        node_params.declare(self, "nandi_locomotion_service")  # WPR: config/nandi.yaml + NANDI_* env (env wins)
        self.dt = 0.02
        self.gait = TripodGaitGenerator(self.dt)
        self.deadman = HoldDeadman()
        self.rl = RLBalanceController(self.dt)  # builds the shared world via load_farm_model()
        self.selector = LocomotionModeSelector(self.rl.policy is not None, self.rl.policy_path)
        self.prev_qpos = self.rl.data.qpos.copy()
        self.curr_qpos = self.rl.data.qpos.copy()
        self._warned_rl_steer = False
        self.stop_seq = 0  # /nandi/stop_motion calls, published in locomotion_status
        self.safety = LocomotionSafetyGate()
        self.inhibit = MotionInhibitGate()
        self.gait.set_params(self.safety.clamp_params(self.gait.params))
        self._holding = False
        self._last_legs = None
        self._t0 = time.monotonic()

        if self.selector.policy_loaded:
            self.get_logger().info(f"RL policy loaded: {self.rl.policy_path} (start in Tripod; toggle to use RL)")
        else:
            self.get_logger().warning(f"RL mode unavailable ({self.rl.load_error}); using Tripod gait.")

        self.create_subscription(Twist, "/nandi/cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(Float64MultiArray, "/nandi/gait_params", self.on_gait_params, 10)
        self.create_subscription(Float64MultiArray, "/nandi/sim_qpos", self.on_qpos, 10)
        self.create_subscription(String, "/nandi/safety_status", self.on_safety_status, 10)
        self.create_subscription(String, "/nandi/motion_inhibit", self.on_motion_inhibit, 10)
        self.create_service(Trigger, "/nandi/stop_motion", self.on_stop)
        self.create_service(Trigger, "/nandi/toggle_locomotion_mode", self.on_toggle_mode)

        self.leg_pub = self.create_publisher(Float64MultiArray, "/nandi/leg_targets", 10)
        self.arm_pub = self.create_publisher(Float64MultiArray, "/nandi/rl_arm_targets", 10)
        self.status_pub = self.create_publisher(String, "/nandi/locomotion_status", 10)
        self.hb_pub = self.create_publisher(String, "/nandi/heartbeat", 10)
        self.create_timer(self.dt, self.step)
        self.create_timer(0.5, self.publish_status)
        self.create_timer(0.5, self.publish_heartbeat)

    # --- inputs -----------------------------------------------------------------------
    def on_qpos(self, msg: Float64MultiArray):
        if len(msg.data) == self.rl.model.nq:
            self.prev_qpos = self.curr_qpos.copy()
            self.curr_qpos[:] = msg.data
            self.safety.qpos_seen(time.monotonic())

    def on_safety_status(self, msg: String):
        st = parse_safety_status(msg.data)
        if st is not None and self.safety.set_estop(st["estopped"], st["latched_reason"]):
            self.gait.stop()
            self.deadman.disarm()
            self.get_logger().warning(f"E-STOP ({st['latched_reason']}): gait stopped, cmd_vel refused")

    def on_motion_inhibit(self, msg: String):
        try:
            mi = json.loads(msg.data)
        except ValueError:
            return
        if self.inhibit.update(mi, time.monotonic()):
            self.gait.stop()
            self.deadman.disarm()
            self.get_logger().warning(f"Motion inhibit ({', '.join(self.inhibit.reasons)}): gait stopped")

    def on_cmd_vel(self, msg: Twist):
        cmd = self.safety.filter_cmd(parse_cmd_vel(msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z))
        if cmd is None:
            if self.safety.refused % 25 == 1:
                self.get_logger().warning(f"cmd_vel refused: e-stopped ({self.safety.reason})")
            return
        self.inhibit.expire(time.monotonic())
        cmd = self.inhibit.filter_cmd(cmd)
        if cmd is None:
            return
        self.deadman.feed(cmd, time.monotonic())
        self.gait.set_command(cmd.vx, cmd.vy, cmd.yaw, cmd.hold)  # all-zero => release (finish cycle)

    def on_gait_params(self, msg: Float64MultiArray):
        params = self.safety.clamp_params(GaitParams.from_sequence(list(msg.data), self.gait.params))
        self.gait.set_params(params)
        self.publish_status()

    def on_stop(self, request, response):
        self.gait.stop()
        self.deadman.disarm()
        self.rl.reset_arm_to_rest()
        self.stop_seq += 1
        self.publish_status()  # now, so a mission holding cmd_vel sees the stop before its next tick
        response.success, response.message = True, "Locomotion stopped"
        return response

    def on_toggle_mode(self, request, response):
        response.success, response.message = self.selector.toggle()
        self.get_logger().info(response.message)
        self.publish_status()
        return response

    # --- outputs ----------------------------------------------------------------------
    def publish_status(self):
        data = locomotion_status_json(self.selector, self.gait.params, self.gait.is_walking,
                                      self.safety.status(time.monotonic()), self.stop_seq)
        self.status_pub.publish(String(data=data))

    def publish_heartbeat(self):
        now = time.monotonic()
        info = {"walking": self.gait.is_walking, "holding_stale_qpos": self.safety.hold_position(now),
                "estopped": self.safety.estopped}
        self.hb_pub.publish(String(data=heartbeat_json("locomotion", now - self._t0, True, info)))

    def step(self):
        now = time.monotonic()
        if self.deadman.expired(now):
            self.gait.release()

        if self.safety.hold_position(now):
            # sim_qpos is stale: stop the gait and keep commanding the last leg pose.
            if not self._holding:
                self._holding = True
                self.gait.stop()
                self.deadman.disarm()
                self.get_logger().warning("/nandi/sim_qpos stale: holding position")
            if self._last_legs is not None:
                self.leg_pub.publish(Float64MultiArray(data=self._last_legs))
            return
        if self._holding:
            self._holding = False
            self.get_logger().info("/nandi/sim_qpos fresh again: gait resumes on the next command")

        roll, pitch = quat_to_roll_pitch(*self.curr_qpos[3:7])
        leg_targets = self.gait.step(roll, pitch)  # always advances gait timing / single-step budget
        arm_targets = None
        if self.selector.mode == "rl" and self.gait.is_active:
            vx, vy, yaw = self.gait.cmd
            if (abs(vy) > 0.05 or abs(yaw) > 0.05 or vx < 0.0) and not self._warned_rl_steer:
                self.get_logger().warning("RL balance policy only walks forward; strafe/turn/reverse are ignored.")
                self._warned_rl_steer = True
            arm_targets, leg_targets = self.rl.predict(self.prev_qpos, self.curr_qpos)
        if arm_targets is None:
            arm_targets = self.rl.reset_arm_to_rest()

        # Leg speed cap on the final output (RL targets included; review F5). The gait limits its own output, but
        # rl.predict() targets are not, and a tripod <-> RL switch would otherwise jump. The gait then resumes
        # from what was actually sent.
        if self._last_legs is not None:
            leg_targets = rate_limit(self._last_legs, leg_targets, MAX_JOINT_SPEED_RPS * self.gait.dt)
        self.gait.last_targets = [float(v) for v in leg_targets]
        self._last_legs = [float(v) for v in leg_targets]
        self.leg_pub.publish(Float64MultiArray(data=self._last_legs))
        self.arm_pub.publish(Float64MultiArray(data=[float(v) for v in arm_targets]))


def main():
    rclpy.init()
    node = LocomotionServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
