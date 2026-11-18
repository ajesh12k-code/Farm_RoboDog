#!/usr/bin/env python3
"""ROS wiring smoke test for atharv_nandi/nodes/mission_executor.py (no physics): run the executor in the background,
then this script plays sim_runner / harvest / locomotion with a frozen home world and checks:

  1. an expired command is answered 'rejected' / 'expired' within 1 s
  2. a valid zone_a command goes accepted -> running and /nandi/cmd_vel arrives at ~10 Hz with hold=1
  3. /nandi/mission/abort -> 'aborted' within 0.5 s and the next cmd_vel is the all-zero release
  4. /nandi/battery is published with source 'idle_only' (no joint_states); it drains during a mission
     and is held full while idle at home
  5. an outside /nandi/stop_motion (console STOP) during a mission -> 'aborted', no held cmd_vel after it
  6. a second command while a mission runs gets its own 'rejected' / 'busy' status
  7. max_picks NaN -> 'rejected' / 'command_invalid', and the executor still answers afterwards

    source /opt/ros/humble/setup.bash
    python3 -m atharv_nandi.nodes mission_executor & python3 tools/mission_ros_smoke.py
    # against the real locomotion node instead of the fake one here:
    python3 -m atharv_nandi.nodes locomotion_service & python3 -m atharv_nandi.nodes mission_executor & python3 tools/mission_ros_smoke.py --real-loco
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from atharv_nandi.core.farm_scene import load_farm_model  # noqa: E402


class Smoke(Node):
    def __init__(self, real_loco=False):
        super().__init__("nandi_mission_smoke")
        self.real_loco, self.stop_seq = real_loco, 0
        self.model = load_farm_model()
        self.qpos = [float(v) for v in self.model.key_qpos[0]]
        self.status, self.battery, self.cmds = [], [], []
        self.create_subscription(String, "/nandi/mission_status", lambda m: self.status.append((time.monotonic(), json.loads(m.data))), 10)
        self.create_subscription(String, "/nandi/battery", lambda m: self.battery.append(json.loads(m.data)), 10)
        self.create_subscription(Twist, "/nandi/cmd_vel", lambda m: self.cmds.append((time.monotonic(), m.linear.x, m.angular.z, m.linear.z)), 10)
        self.qpos_pub = self.create_publisher(Float64MultiArray, "/nandi/sim_qpos", 10)
        self.game_pub = self.create_publisher(String, "/nandi/game_status", 10)
        self.harv_pub = self.create_publisher(String, "/nandi/harvest_status", 10)
        self.cmd_pub = self.create_publisher(String, "/nandi/mission/command", 10)
        self.abort_cli = self.create_client(Trigger, "/nandi/mission/abort")
        self.loco_pub = self.create_publisher(String, "/nandi/locomotion_status", 10)
        self.walking = False
        self.loco_seen = []
        self.create_subscription(Twist, "/nandi/cmd_vel", self.on_cmd, 10)
        self.create_subscription(String, "/nandi/locomotion_status",
                                 lambda m: self.loco_seen.append((time.monotonic(), json.loads(m.data))), 10)
        if real_loco:
            self.stop_cli = self.create_client(Trigger, "/nandi/stop_motion")
        else:
            self.create_service(Trigger, "/nandi/stop_motion", self.stop_srv)
        self.create_service(Trigger, "/nandi/release_arm", self.ok_srv)
        self.create_timer(0.02, lambda: self.qpos_pub.publish(Float64MultiArray(data=self.qpos)))
        self.create_timer(0.5, self.pub_status)

    def ok_srv(self, req, res):
        res.success, res.message = True, "ok"
        return res

    def on_cmd(self, m):  # the fake locomotion: walking follows the hold flag, like the gait
        self.walking = m.linear.z >= 0.5

    def stop_srv(self, req, res):  # mirrors locomotion_service.on_stop: count + publish at once
        self.stop_seq += 1
        self.walking = False
        self.pub_loco()
        res.success, res.message = True, "Locomotion stopped"
        return res

    def pub_loco(self):
        if not self.real_loco:
            self.loco_pub.publish(String(data=json.dumps({"mode": "tripod", "walking": self.walking,
                                                          "stop_seq": self.stop_seq})))

    def outside_stop(self):
        """What the console STOP does: call /nandi/stop_motion (not the mission abort)."""
        if self.real_loco:
            fut = self.stop_cli.call_async(Trigger.Request())
            t0 = time.monotonic()
            while not fut.done() and time.monotonic() - t0 < 2.0:
                rclpy.spin_once(self, timeout_sec=0.01)
        else:
            self.stop_srv(None, Trigger.Response())
        return time.monotonic()

    def pub_status(self):
        self.pub_loco()
        self.game_pub.publish(String(data=json.dumps({"crops_total": 6, "in_basket": 0, "in_basket_names": [], "falls": 0, "fallen": False})))
        self.harv_pub.publish(String(data=json.dumps({"mode": "rl_balance", "phase": "idle", "target": None, "last_result": ""})))

    def spin_for(self, s):
        end = time.monotonic() + s
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def send(self, not_after):
        with open(os.path.join(ROOT, "config", "missions", "zone_a.geojson")) as fh:
            zone = json.load(fh)
        cmd = {"mission_id": f"smoke-{int(time.time() * 1000) % 100000}", "mode": "harvest_zone", "zone": zone,
               "not_after": not_after.isoformat().replace("+00:00", "Z")}
        self.cmd_pub.publish(String(data=json.dumps(cmd)))
        return cmd["mission_id"], time.monotonic()

    def wait_state(self, mid, states, timeout):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.01)
            for t, st in self.status:
                if st.get("mission_id") == mid and st.get("state") in states and t >= t0 - 0.05:
                    return st, t
        return None, None


def main():
    rclpy.init()
    n = Smoke(real_loco="--real-loco" in sys.argv)
    n.spin_for(3.0)  # executor discovers us, gets qpos + game_status
    out = {}
    now = datetime.now(timezone.utc)
    mid, t0 = n.send(now - timedelta(seconds=5))
    st, t = n.wait_state(mid, ("rejected",), 2.0)
    out["expired"] = None if st is None else {"state": st["state"], "reason": st["reason_code"], "latency_s": round(t - t0, 3)}

    mid, t0 = n.send(now + timedelta(minutes=30))
    st, t = n.wait_state(mid, ("accepted", "running", "rejected"), 2.0)
    out["accept"] = None if st is None else {"state": st["state"], "reason": st["reason_code"],
                                             "detail": st["detail"], "latency_s": round(t - t0, 3)}
    n.cmds.clear()
    n.spin_for(2.0)
    held = [c for c in n.cmds if c[3] >= 0.5]
    gaps = [b[0] - a[0] for a, b in zip(held, held[1:])]
    out["cmd_vel"] = {"n_in_2s": len(n.cmds), "held": len(held), "max_gap_s": round(max(gaps), 3) if gaps else None,
                      "sample": [round(v, 3) for v in held[-1][1:]] if held else None}
    st, _ = n.wait_state(mid, ("running",), 1.0)
    out["running"] = st is not None
    out["battery_during_mission"] = n.battery[-1] if n.battery else None
    bid, t0 = n.send(now + timedelta(minutes=30))
    st, t = n.wait_state(bid, ("rejected",), 1.0)
    out["busy"] = None if st is None else {"reason": st["reason_code"], "active": st.get("active_mission_id"),
                                           "latency_s": round(t - t0, 3)}
    st, _ = n.wait_state(mid, ("running",), 1.0)
    out["busy"] = {**(out["busy"] or {}), "running_mission_still_running": st is not None}
    fut = n.abort_cli.call_async(Trigger.Request())
    t0 = time.monotonic()
    n.cmds.clear()
    while not fut.done() and time.monotonic() - t0 < 2.0:
        rclpy.spin_once(n, timeout_sec=0.01)
    st, t = n.wait_state(mid, ("aborted",), 1.0)
    n.spin_for(0.5)
    first = n.cmds[0] if n.cmds else None
    out["abort"] = {"reply": fut.result().message if fut.done() else None,
                    "state": None if st is None else st["state"],
                    "latency_s": None if t is None else round(t - t0, 3),
                    "first_cmd_after": None if first is None else [round(v, 3) for v in first[1:]],
                    "first_cmd_latency_s": None if first is None else round(first[0] - t0, 3),
                    "cmds_after_abort": [[round(c[0] - t0, 3)] + [round(v, 3) for v in c[1:]] for c in n.cmds[:6]],
                    "held_after_abort_ack": sum(1 for c in n.cmds if c[3] >= 0.5 and c[0] > t0 + 0.05)}
    out["battery"] = n.battery[-1] if n.battery else None

    # 5. outside STOP during a mission
    mid, t0 = n.send(datetime.now(timezone.utc) + timedelta(minutes=30))
    st, _ = n.wait_state(mid, ("running",), 2.0)
    n.spin_for(1.0)
    n.cmds.clear()
    t0 = time.monotonic()
    t_ack = n.outside_stop()
    st, t = n.wait_state(mid, ("aborted",), 1.0)
    n.spin_for(3.0)
    late = [c for c in n.cmds if c[0] > t_ack + 0.02]
    walk_after = [d.get("walking") for tt, d in n.loco_seen if tt > t_ack + 0.5]
    out["outside_stop"] = {"running_before": st is not None or None,
                           "state": None if st is None else st["state"], "detail": None if st is None else st["detail"],
                           "latency_s": None if t is None else round(t - t0, 3),
                           "cmds_in_3s": len(n.cmds), "held_in_3s": sum(1 for c in n.cmds if c[3] >= 0.5),
                           "held_after_stop_ack": sum(1 for c in late if c[3] >= 0.5),
                           "ack_after_s": round(t_ack - t0, 3), "loco_walking_after": walk_after,
                           "first_cmd_after": [round(v, 3) for v in n.cmds[0][1:]] if n.cmds else None}

    # 7. NaN max_picks, then the node must still answer
    text = json.dumps({"mission_id": "nan-picks", "mode": "harvest_zone", "zone": {"local_polygon": [[0.4, 0.15], [3.9, 0.15], [3.9, 1.2], [0.4, 1.2]]},
                       "not_after": "2030-01-01T00:00:00Z"})[:-1] + ', "max_picks": NaN}'
    n.cmd_pub.publish(String(data=text))
    st, _ = n.wait_state("nan-picks", ("rejected",), 2.0)
    mid, _ = n.send(datetime.now(timezone.utc) - timedelta(seconds=5))
    st2, _ = n.wait_state(mid, ("rejected",), 2.0)
    out["nan_max_picks"] = {"reason": None if st is None else st["reason_code"],
                            "alive_after": st2 is not None and st2["reason_code"] == "expired"}
    n.battery.clear()
    n.spin_for(1.2)
    out["battery_idle_at_home"] = n.battery[-1]["soc"] if n.battery else None  # docked: held full (ASSUMED)
    print(json.dumps(out, indent=1))
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
