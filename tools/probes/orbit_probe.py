"""Feasibility probe (scratch, not for merge): can the tripod gait orbit a point at a steady radius?

Mode 'side': vx + yaw (tree on the robot's left side, ring camera 90 deg looks at it).
Mode 'face': vy + yaw (crab sideways, trunk nose faces the tree centre).
Prints one JSON row per config: fitted radius, radius drift, tangential speed, bearing error of the
'tree-facing' axis to the fitted centre, trunk roll/pitch/yaw-rate jitter, fall flag, wall time.
Usage: python3 orbit_probe.py side|face <cmd_lin> <cmd_yaw> <speed_scale> <sim_s> [terrain]
"""
import json, math, os, sys, time
mode, c_lin, c_yaw, sscale, sim_s = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
terrain = sys.argv[6] if len(sys.argv) > 6 else "flat"
os.environ["NANDI_TERRAIN"] = terrain
os.environ["NANDI_CROPS"] = "1"
os.environ["NANDI_SEED"] = "7"
import mujoco, numpy as np
from atharv_nandi.core.farm_scene import load_farm_model
from atharv_nandi.core.tripod_gait import TripodGaitGenerator, GaitParams
from atharv_nandi.core.locomotion_logic import quat_to_roll_pitch

m = load_farm_model(); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
g = TripodGaitGenerator(0.02)
p = g.params
g.set_params(GaitParams(speed_scale=sscale, step_height_scale=p.step_height_scale, freq_hz=p.freq_hz, balance_gain=p.balance_gain))
if mode == "side":
    g.set_command(c_lin, 0.0, c_yaw, hold=True)
else:
    g.set_command(0.0, c_lin, c_yaw, hold=True)
arm = list(m.key_ctrl[0][:7]); legs = list(m.key_ctrl[0][7:])
every = int(round(0.02 / m.opt.timestep))
xs, ys, yaws, rolls, pitches, wz, ts = [], [], [], [], [], [], []
t0 = time.perf_counter(); step = 0; fell = False
while d.time < sim_s:
    if step % every == 0:
        r, pt = quat_to_roll_pitch(*d.qpos[3:7]); legs = g.step(r, pt)
    d.ctrl[:7] = arm; d.ctrl[7:] = legs
    mujoco.mj_step(m, d); step += 1
    if step % 25 == 0:  # 20 Hz samples
        q = d.qpos; w, x, y, z = q[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        r, pt = quat_to_roll_pitch(w, x, y, z)
        xs.append(q[0]); ys.append(q[1]); yaws.append(yaw); rolls.append(r); pitches.append(pt)
        wz.append(d.qvel[5]); ts.append(d.time)
        if abs(r) > 0.6 or abs(pt) > 0.6 or q[2] < 0.15:
            fell = True; break
wall = time.perf_counter() - t0
X, Y = np.array(xs), np.array(ys); T = np.array(ts)
k = T > 3.0  # drop the start ramp
X, Y, Yw = X[k], Y[k], np.unwrap(np.array(yaws)[k])
A = np.c_[2 * X, 2 * Y, np.ones_like(X)]
sol, *_ = np.linalg.lstsq(A, X ** 2 + Y ** 2, rcond=None)
cx, cy = sol[0], sol[1]; R = math.sqrt(sol[2] + cx ** 2 + cy ** 2)
rad = np.hypot(X - cx, Y - cy)
ang = np.unwrap(np.arctan2(Y - cy, X - cx))
dur = T[k][-1] - T[k][0]
v_tan = abs(ang[-1] - ang[0]) * R / dur
# bearing from robot to centre vs the tree-facing axis (left = yaw+90 deg for side, nose = yaw for face)
bear = np.arctan2(cy - Y, cx - X)
axis = Yw + (math.pi / 2 if (mode == "side" and c_yaw > 0) else (-math.pi / 2 if mode == "side" else 0.0))
err = np.degrees((bear - axis + math.pi) % (2 * math.pi) - math.pi)
WZ = np.array(wz)[k]
print(json.dumps({
    "mode": mode, "cmd_lin": c_lin, "cmd_yaw": c_yaw, "speed_scale": sscale, "terrain": terrain, "sim_s": round(float(T[-1]), 1),
    "fell": fell, "R_fit_m": round(R, 3), "R_p05_p95": [round(float(np.percentile(rad, 5)), 3), round(float(np.percentile(rad, 95)), 3)],
    "v_tan_mps": round(v_tan, 3), "orbit_deg_done": round(float(np.degrees(abs(ang[-1] - ang[0]))), 1),
    "s_per_orbit": round(2 * math.pi * R / v_tan, 1) if v_tan > 1e-4 else None,
    "bearing_err_deg_mean": round(float(np.mean(err)), 1), "bearing_err_deg_p95abs": round(float(np.percentile(np.abs(err), 95)), 1),
    "roll_pitch_p95abs_deg": [round(float(np.degrees(np.percentile(np.abs(np.array(rolls)[k]), 95))), 2),
                              round(float(np.degrees(np.percentile(np.abs(np.array(pitches)[k]), 95))), 2)],
    "yaw_rate_mean_dps": round(float(np.degrees(WZ.mean())), 2), "yaw_rate_std_dps": round(float(np.degrees(WZ.std())), 2),
    "wall_s": round(wall, 1), "rtf": round(float(T[-1]) / wall, 2)}))
