import json, math, sys, threading, time
import numpy as np, rclpy, mujoco
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger
from atharv_nandi.core.farm_scene import load_farm_model

SC = sys.argv[1]

class C(Node):
    def __init__(s):
        super().__init__("safety_check")
        s.js = []; s.qpos = None; s.status = None; s.st_times = []
        s.create_subscription(JointState, "/nandi/joint_states", lambda m: s.js.append((time.monotonic(), np.array(m.velocity), list(m.name))), 50)
        s.create_subscription(Float64MultiArray, "/nandi/sim_qpos", s.on_q, 50)
        s.create_subscription(String, "/nandi/safety_status", s.on_st, 50)
        s.cmd = s.create_publisher(Twist, "/nandi/cmd_vel", 10)
        s.arm = s.create_publisher(Float64MultiArray, "/nandi/arm_targets", 10)
        s.fakeq = s.create_publisher(Float64MultiArray, "/nandi/sim_qpos", 10)
        s.estop = s.create_client(Trigger, "/nandi/estop"); s.reset = s.create_client(Trigger, "/nandi/estop_reset")
        s.rw = s.create_client(Trigger, "/nandi/reset_world")
    def on_q(s, m): s.qpos = np.array(m.data)
    def on_st(s, m): s.status = json.loads(m.data); s.st_times.append(time.monotonic())

rclpy.init(); n = C()
threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()
model = load_farm_model(); data = mujoco.MjData(model)
tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
def tip_pos():
    data.qpos[:] = n.qpos; mujoco.mj_kinematics(model, data); return data.site_xpos[tip].copy()
def call(cli, name):
    assert cli.wait_for_service(5.0), name
    r = cli.call_async(Trigger.Request()); t = time.monotonic()
    while not r.done() and time.monotonic() - t < 3: time.sleep(0.005)
    res = r.result(); print(f"CALL {name}: success={res.success} msg={res.message!r}"); return res
def walk(dur, vx=1.0):
    t = time.monotonic()
    while time.monotonic() - t < dur:
        tw = Twist(); tw.linear.x = vx; tw.linear.z = 1.0; n.cmd.publish(tw); time.sleep(0.1)
def wait(cond, to=5.0):
    t = time.monotonic()
    while time.monotonic() - t < to:
        if cond(): return True
        time.sleep(0.01)
    return False

wait(lambda: n.qpos is not None and n.status is not None, 15)
if SC != "fall":
    wait(lambda: n.status and "sim_runner" in n.status["nodes"], 10)
print("initial status:", json.dumps(n.status))
ok = True
if SC == "main":
    st = n.st_times[-1]; c0 = len(n.st_times); time.sleep(2.0)
    print(f"safety_status rate: {(len(n.st_times)-c0)/2.0:.1f} Hz")
    j0 = len(n.js); time.sleep(1.0); print(f"joint_states rate: {len(n.js)-j0:.0f} Hz (wall), names[0:3]={n.js[-1][2][:3]} n={len(n.js[-1][2])}")
    q0 = n.qpos[:2].copy(); t0 = time.monotonic()
    walk(4.0)
    q1 = n.qpos[:2].copy(); speed = np.linalg.norm(q1 - q0) / (time.monotonic() - t0)
    print(f"(a) walked {np.linalg.norm(q1-q0):.3f} m in {time.monotonic()-t0:.2f} s wall, mean {speed:.3f} m/s")
    tw = Twist(); tw.linear.x = 1.0; tw.linear.z = 1.0; n.cmd.publish(tw)
    te = time.monotonic(); call(n.estop, "/nandi/estop")
    # keep sending walk commands (they must be refused)
    def still():
        recent = [v for (t, v, _) in n.js if t > time.monotonic() - 0.05]
        return recent and max(np.max(np.abs(v)) for v in recent) < 0.05
    pub_stop = threading.Event()
    def spam():
        while not pub_stop.is_set(): n.cmd.publish(tw); time.sleep(0.1)
    threading.Thread(target=spam, daemon=True).start()
    hit = wait(still, 3.0); tq = time.monotonic() - te
    vmax_after = [ (round(t-te,3), float(np.max(np.abs(v)))) for (t,v,_) in n.js if t > te][:: 5][:20]
    print(f"(a) joint |v|<0.05 reached={hit} after {tq:.3f} s wall (criterion <= 0.5 s)")
    print("    max|v| samples (t, rad/s):", vmax_after[:12])
    ok &= hit and tq <= 0.5
    wait(lambda: n.status["last_stop"] is not None, 3.0)
    s = n.status; print(f"(a) safety_status: estopped={s['estopped']} reason={s['latched_reason']} hold_mode={s['hold_mode']} last_stop={s['last_stop']}")
    ok &= bool(s["last_stop"]) and s["last_stop"]["dist_m"] < 0.15
    pos_a = n.qpos[:2].copy(); time.sleep(1.5); pub_stop.set()
    print(f"(a) drift while e-stopped with cmd_vel spam: {np.linalg.norm(n.qpos[:2]-pos_a)*100:.2f} cm")
    # (b) arm
    p0 = tip_pos()
    t = time.monotonic()
    while time.monotonic() - t < 2.0:
        n.arm.publish(Float64MultiArray(data=[0.2, 1.0, 1.2, -1.5, 0.8, 0.04, 0.04])); time.sleep(0.02)
    d = np.linalg.norm(tip_pos() - p0)
    print(f"(b) gripper_tip moved {d*100:.2f} cm after 2 s of arm_targets while e-stopped (criterion < 1 cm)")
    ok &= d < 0.01
    call(n.reset, "/nandi/estop_reset")
    wait(lambda: not n.status["estopped"], 2.0)
    t = time.monotonic()
    while time.monotonic() - t < 1.5:
        n.arm.publish(Float64MultiArray(data=[0.2, 1.0, 1.2, -1.5, 0.8, 0.04, 0.04])); time.sleep(0.02)
    print(f"(b-control) after reset the same arm_targets move the tip {np.linalg.norm(tip_pos()-p0)*100:.1f} cm; hold_mode={n.status['hold_mode']}")
    print("nodes:", n.status["nodes"], "stale:", n.status["stale"])
elif SC == "fence":
    walk(12.0)
    wait(lambda: n.status["estopped"], 2.0)
    s = n.status
    print(f"(c) after walking: x={n.qpos[0]:.3f} estopped={s['estopped']} reason={s['latched_reason']} geofence_ok={s['geofence_ok']} margin={s['geofence_margin_m']} last_stop={s['last_stop']}")
    ok &= s["latched_reason"] == "geofence"
    x1 = n.qpos[0]; walk(2.0); print(f"(c) moved {abs(n.qpos[0]-x1)*100:.2f} cm under 2 s more cmd_vel")
    r = call(n.reset, "/nandi/estop_reset"); ok &= not r.success
    call(n.rw, "/nandi/reset_world"); time.sleep(1.0)
    print(f"(c) after reset_world x={n.qpos[0]:.3f} geofence_ok={n.status['geofence_ok']} estopped={n.status['estopped']}")
    r = call(n.reset, "/nandi/estop_reset"); ok &= r.success
    time.sleep(0.3); print(f"(c) estopped={n.status['estopped']}")
elif SC == "fall":
    # supervisor only: feed synthetic sim_qpos (supervisor-side ROS path); physics push is in push.py
    def q(tilt):
        a = math.radians(tilt) / 2; return [0, 0, 0.3, math.cos(a), math.sin(a), 0, 0] + [0.0] * 25
    for k in range(10): n.fakeq.publish(Float64MultiArray(data=q(2.0))); time.sleep(0.02)
    t0 = time.monotonic()
    while not n.status["estopped"] and time.monotonic() - t0 < 2:
        n.fakeq.publish(Float64MultiArray(data=q(45.0))); time.sleep(0.02)
    print(f"(d) tilt 45 deg: latched after {time.monotonic()-t0:.3f} s, reason={n.status['latched_reason']} fallen={n.status['fallen']} tilt={n.status['tilt_deg']}")
    ok &= n.status["latched_reason"] == "fall"
    r = call(n.reset, "/nandi/estop_reset"); ok &= not r.success
    for k in range(10): n.fakeq.publish(Float64MultiArray(data=q(2.0))); time.sleep(0.02)
    time.sleep(0.2); r = call(n.reset, "/nandi/estop_reset"); ok &= r.success
    time.sleep(1.0); print(f"(stale) after fake sim_qpos stops: stale={n.status['stale']}")
    ok &= "/nandi/sim_qpos" in n.status["stale"]
if SC == "stale":
    walk(1.5)
    import subprocess; subprocess.run(["pkill", "-f", r"atharv_nandi[.]nodes sim_runner|lib/atharv_nandi/sim_runner"]); tk = time.monotonic()
    time.sleep(1.2)
    print(f"(stale) {time.monotonic()-tk:.1f} s after killing sim_runner: stale={n.status['stale']} nodes={n.status['nodes']}")
    ok &= "/nandi/sim_qpos" in n.status["stale"]
    print(open("/tmp/loco.log").read().strip().splitlines()[-1])
    ok &= "holding position" in open("/tmp/loco.log").read()
print("RESULT", "PASS" if ok else "FAIL")
import os; os._exit(0 if ok else 1)
