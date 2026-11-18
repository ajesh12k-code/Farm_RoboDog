#!/usr/bin/env python3
"""Atharv Nandi mission executor: walk a zone's crop rows, pick every crop in it, return home.

Thin ROS 2 shell around atharv_nandi/core/mission_logic.py (state machine, planner), nav_logic.py (pose to
goal, cmd_vel hold), energy_model.py (battery) and geo_frame.py (site frame, zones). See README
"Missions" for the contract.

    in   /nandi/mission/command (String JSON)   /nandi/sim_qpos   /nandi/game_status   /nandi/harvest_status
         /nandi/locomotion_status   /nandi/safety_status (optional)   /nandi/crop_detections (optional)
         /nandi/joint_states (optional, sensor_msgs/JointState: effort x velocity -> power)
    out  /nandi/cmd_vel (10 Hz while walking)   /nandi/mission_status (2 Hz)   /nandi/battery (2 Hz)
         /nandi/mission/status, /nandi/battery_status (the same, under the web console's names)
    srv  /nandi/mission/abort (Trigger)
    uses /nandi/auto_harvest, /nandi/stop_motion, /nandi/release_arm

Single-threaded executor: every callback, timer and service future runs on one thread, so the
controller needs no lock. Service calls are asynchronous; nothing here blocks the executor.

Outside stops: a /nandi/stop_motion from anyone else (web console STOP, Tk panel, reset_world) shows
up as a rise of 'stop_seq' in /nandi/locomotion_status, and a world reset as a trunk jump in
/nandi/sim_qpos. Either ends the running mission as 'aborted' at once, so its 10 Hz hold does not
restart the walk. The battery is reset to full on a world reset and while docked at home with no
mission (ASSUMED charger at home).
"""
import json
import math
import os
import time
from datetime import datetime, timezone

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from atharv_nandi import node_params  # noqa: E402
from atharv_nandi import paths
from atharv_nandi.core import geo_frame as gf  # noqa: E402
from atharv_nandi.core import mission_logic as ml  # noqa: E402
from atharv_nandi.core.energy_model import Battery, electrical_power_w  # noqa: E402
from atharv_nandi.core import model_layout as mlay  # noqa: E402
from atharv_nandi.core.farm_scene import RIDGES, load_farm_model, scene_config  # noqa: E402
from atharv_nandi.core.locomotion_logic import quat_to_yaw  # noqa: E402
from atharv_nandi.core.nav_logic import CMD_RATE_HZ, CmdVelHold  # noqa: E402

SAFETY_FILE = paths.config_path("safety.yaml")
QPOS_SIM_DT = 0.02          # sim_runner publishes /nandi/sim_qpos every 0.02 s of sim time
JOINT_STATES_FRESH_S = 1.0  # older joint_states -> idle-only drain
SPEED_WINDOW = 5            # qpos samples for the trunk speed estimate (0.1 s of sim time)
RESET_JUMP_M = 0.1          # trunk moved further than this in one qpos sample -> world reset
DOCK_RADIUS_M = 0.15        # idle within this of home = on the (ASSUMED) charger: battery held full


class MissionExecutorNode(Node):
    def __init__(self):
        super().__init__("nandi_mission_executor")
        node_params.declare(self, "nandi_mission_executor")  # WPR: config/nandi.yaml + NANDI_* env (env wins)
        self.origin, self.site_cfg = gf.load_site()
        self.cfg = ml.config_from_site(self.origin, self.site_cfg, geofence=None)
        self.cfg.keep_out_x = ml.ridge_keep_out(RIDGES, scene_config()[1])
        self._load_geofence()
        self.ctl = ml.MissionController(self.cfg)
        self.battery = Battery(self.cfg.energy)
        self.hold = CmdVelHold()
        # Name-based layout of the live world (WP0): legacy crop truth never reads appended module joints.
        try:
            load_farm_model()
            self.layout = mlay.current()
        except Exception as exc:  # no MuJoCo: fall back to the legacy shape rule, and say so
            self.layout = None
            self.get_logger().warn(f"no world layout ({exc}); crop truth assumes the legacy qpos shape")

        # latest observations
        self.qpos = None
        self.sim_t = 0.0
        self.trunk_hist = []  # (sim_t, x, y)
        self.game = {}
        self.harvest = {}
        self.walking = False
        self.loco_mode = ""        # '' until /nandi/locomotion_status is seen
        self.stop_seq = None       # last /nandi/locomotion_status stop_seq (None = not reported)
        self.own_stops = 0         # stop_motion calls of ours whose stop_seq rise is still to come
        self.estopped = False
        self.detections, self.detections_t = {}, None
        self.power_w, self.power_t = None, None
        self.last_state = self.ctl.state

        self.create_subscription(String, "/nandi/mission/command", self.on_command, 10)
        self.create_subscription(Float64MultiArray, "/nandi/sim_qpos", self.on_qpos, 10)
        self.create_subscription(String, "/nandi/game_status", self.on_game, 10)
        self.create_subscription(String, "/nandi/harvest_status", self.on_harvest, 10)
        self.create_subscription(String, "/nandi/locomotion_status", self.on_loco, 10)
        self.create_subscription(String, "/nandi/safety_status", self.on_safety, 10)
        self.create_subscription(String, "/nandi/crop_detections", self.on_detections, 10)
        self.create_subscription(JointState, "/nandi/joint_states", self.on_joint_states, 10)
        self.cmd_pub = self.create_publisher(Twist, "/nandi/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, "/nandi/mission_status", 10)
        self.battery_pub = self.create_publisher(String, "/nandi/battery", 10)
        # the web console's panels read these names (console_protocol.EXT_TOPICS): same payloads
        self.console_status_pub = self.create_publisher(String, "/nandi/mission/status", 10)
        self.console_battery_pub = self.create_publisher(String, "/nandi/battery_status", 10)
        self.create_service(Trigger, "/nandi/mission/abort", self.on_abort)
        self.cli = {name: self.create_client(Trigger, f"/nandi/{name}")
                    for name in ("auto_harvest", "stop_motion", "release_arm")}
        self.create_timer(1.0 / CMD_RATE_HZ, self.on_tick)
        self.create_timer(0.5, self.publish_status)
        self.get_logger().info(
            f"Mission executor ready: site {self.origin.name} ({self.origin.lat0}, {self.origin.lon0}), "
            f"basket {self.cfg.capacity.limit} crops, battery {self.cfg.energy.capacity_wh} Wh (ASSUMED), "
            f"geofence from {self.geofence_src}")

    # ------------------------------------------------------------------ config
    def _load_geofence(self):
        safety = None
        if os.path.exists(SAFETY_FILE):
            try:
                safety = gf.load_flat_yaml(SAFETY_FILE)
            except Exception as exc:  # malformed file: fall back, but say so
                self.get_logger().warn(f"config/safety.yaml unreadable ({exc}); using the site fallback geofence")
        fence, self.geofence_src = ml.geofence_from_config(safety, self.site_cfg)
        self.cfg.geofence = fence

    # ------------------------------------------------------------------ inputs
    def on_qpos(self, msg: Float64MultiArray):
        if len(msg.data) < mlay.ROBOT_NQ:
            return
        q = list(msg.data)
        reset = self.qpos is not None and math.hypot(q[0] - self.qpos[0], q[1] - self.qpos[1]) > RESET_JUMP_M
        self.qpos = q
        self.sim_t += QPOS_SIM_DT
        if reset:
            self.trunk_hist = []  # no speed spike
            self.battery.reset()
            self._outside_stop("world reset (/nandi/reset_world)")
        self.trunk_hist = (self.trunk_hist + [(self.sim_t, q[0], q[1])])[-SPEED_WINDOW:]
        fresh = self.power_t is not None and time.monotonic() - self.power_t < JOINT_STATES_FRESH_S
        self.battery.update(self.sim_t, self.power_w if fresh else None)
        if (self.ctl.state not in ml.ACTIVE_STATES
                and math.hypot(q[0] - self.cfg.home[0], q[1] - self.cfg.home[1]) <= DOCK_RADIUS_M):
            self.battery.reset()  # docked at home with no mission: on the charger (ASSUMED)

    def on_game(self, msg: String):
        try:
            self.game = json.loads(msg.data)
        except ValueError:
            pass

    def on_harvest(self, msg: String):
        try:
            self.harvest = json.loads(msg.data)
        except ValueError:
            pass

    def on_loco(self, msg: String):
        try:
            d = json.loads(msg.data)
            self.walking = bool(d.get("walking", False))
            self.loco_mode = str(d.get("mode", ""))
            seq = d.get("stop_seq")
        except (ValueError, AttributeError, TypeError):
            return
        if not isinstance(seq, int) or isinstance(seq, bool):
            return  # older locomotion_service: outside stops cannot be seen
        if self.stop_seq is not None and seq != self.stop_seq:
            rises = seq - self.stop_seq if seq > self.stop_seq else 1  # lower = locomotion restarted
            outside = rises - min(rises, self.own_stops)
            self.own_stops = max(0, self.own_stops - rises)
            self.stop_seq = seq
            if outside > 0:
                self._outside_stop("operator stop (/nandi/stop_motion)")
        self.stop_seq = seq
        if self.loco_mode != ml.TRIPOD_MODE and self.ctl.state in ml.ACTIVE_STATES:
            self.on_tick()  # fail the mission now, not at the next 10 Hz tick

    def _outside_stop(self, detail: str):
        if self.ctl.external_stop(detail):
            self.get_logger().warn(f"mission {self.ctl.mission_id} aborted: {detail}")
            self._publish_cmd((0.0, 0.0, 0.0, 0.0))  # stand still now; the tick repeats the release
            self.on_tick()

    def on_safety(self, msg: String):
        was = self.estopped
        self.estopped = ml.parse_estop(msg.data)
        if self.estopped and not was:
            self.on_tick()  # react now, not at the next 10 Hz tick

    def on_detections(self, msg: String):
        self.detections, self.detections_t = ml.parse_detections(msg.data), time.monotonic()

    def on_joint_states(self, msg: JointState):
        if msg.effort and msg.velocity:
            self.power_w = electrical_power_w(msg.effort, msg.velocity, self.cfg.energy)
            self.power_t = time.monotonic()

    def observe(self) -> ml.Observation:
        now = time.monotonic()
        o = ml.Observation(t=now, utc=datetime.now(timezone.utc), soc=self.battery.soc, estopped=self.estopped,
                           walking=self.walking, loco_mode=self.loco_mode)
        if self.qpos is not None:
            q = self.qpos
            o.pose = (q[0], q[1], quat_to_yaw(*q[3:7]))
            if len(self.trunk_hist) >= 2:
                (t0, x0, y0), (t1, x1, y1) = self.trunk_hist[0], self.trunk_hist[-1]
                o.trunk_speed = math.hypot(x1 - x0, y1 - y0) / max(t1 - t0, 1e-6)
            truth = {n: tuple(v) for n, v in mlay.legacy_crop_positions(q, self.layout).items()}
            age = None if self.detections_t is None else now - self.detections_t
            o.crops, o.crop_source = ml.choose_crops(truth, self.detections, age)
        if self.game:
            o.in_basket = int(self.game.get("in_basket", 0))
            o.in_basket_names = tuple(self.game.get("in_basket_names", ()))
            o.fallen = bool(self.game.get("fallen", False))
        h = self.harvest
        o.harvest_mode = h.get("mode", "rl_balance")
        o.harvest_phase = h.get("phase", "idle")
        o.harvest_target = h.get("target")
        o.harvest_result = h.get("last_result", "")
        return o

    # ------------------------------------------------------------------ commands
    def on_command(self, msg: String):
        self._load_geofence()  # safety.yaml may have appeared or changed
        try:
            ok, reason, detail = self.ctl.submit(msg.data, self.observe())
        except Exception as exc:  # belt and braces: one bad command must not kill the node
            self.get_logger().error(f"mission command crashed the parser: {exc!r}")
            return
        self.last_state = self.ctl.state
        if ok:  # (rclpy forbids one call site logging at two severities)
            self.get_logger().info(f"mission accepted: {detail}")
        else:
            self.get_logger().warn(f"mission rejected: {reason} - {detail}")
        self.publish_status()  # answer within one message, not at the next 2 Hz tick

    def on_abort(self, request, response):
        response.success, response.message = self.ctl.abort()
        if response.success:
            self._publish_cmd((0.0, 0.0, 0.0, 0.0))  # stop now; the tick repeats the release
            self.on_tick()
        return response

    def _call(self, name: str):
        cli = self.cli[name]
        if not cli.service_is_ready():
            if name == "auto_harvest":
                self.ctl.on_harvest_reply(False, "auto_harvest service not available")
            return
        fut = cli.call_async(Trigger.Request())
        if name == "stop_motion" and self.stop_seq is not None:
            self.own_stops += 1  # its stop_seq rise is ours, not an operator STOP
        if name == "auto_harvest":
            def done(f):
                try:
                    res = f.result()
                    self.ctl.on_harvest_reply(res.success, res.message)
                except Exception as exc:  # service died mid-call
                    self.ctl.on_harvest_reply(False, f"auto_harvest call failed: {exc}")
            fut.add_done_callback(done)

    def _publish_cmd(self, out):
        vx, vy, yaw, hold = out
        t = Twist()
        t.linear.x, t.linear.y, t.linear.z, t.angular.z = float(vx), float(vy), float(hold), float(yaw)
        self.cmd_pub.publish(t)

    # ------------------------------------------------------------------ timers
    def on_tick(self):
        act = self.ctl.step(self.observe())
        if act.force_release:
            self.hold.force_release()
        out = self.hold.tick(act.cmd)
        if out is not None:
            self._publish_cmd(out)
        for name in act.calls:
            self._call(name)
        if self.ctl.state != self.last_state:
            self.get_logger().info(f"mission {self.ctl.mission_id}: {self.last_state} -> {self.ctl.state} "
                                   f"({self.ctl.reason or self.ctl.return_reason})")
            self.last_state = self.ctl.state
            self.publish_status()
        for ev in self.ctl.events:
            self.get_logger().info(ev)
        self.ctl.events.clear()

    def publish_status(self):
        for reply in self.ctl.replies:  # busy rejections: answer the sender, then the active mission
            self.status_pub.publish(String(data=json.dumps(reply)))
        self.ctl.replies.clear()
        st = String(data=self.ctl.status_json())
        self.status_pub.publish(st)
        self.console_status_pub.publish(st)
        bat = self.battery.status()
        self.battery_pub.publish(String(data=json.dumps(bat)))
        self.console_battery_pub.publish(String(data=json.dumps({**bat, "percentage": bat.get("soc")})))


def main():
    rclpy.init()
    node = MissionExecutorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
