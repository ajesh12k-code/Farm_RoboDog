"""Physics push into a fall: lateral force on the trunk, SafetyCore fed from the sim (no ROS)."""
import numpy as np, mujoco
from atharv_nandi.core.farm_scene import load_farm_model
from atharv_nandi.core.game_status import is_fallen, tilt_deg
from atharv_nandi.core.safety_logic import SafetyCore, load_safety_config
m = load_farm_model(); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
core = SafetyCore(load_safety_config())
d.ctrl[:] = m.key_ctrl[0]
for F in (150.0, 300.0, 600.0):
    mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d); core = SafetyCore(load_safety_config())
    latched = None; maxtilt = 0
    while d.time < 3.0:
        # push sideways (+y) at a point above the trunk for 0.4 s => roll moment
        d.xfrc_applied[trunk] = [0, F, 0, F * 0.25, 0, 0] if 0.5 < d.time < 0.9 else 0
        mujoco.mj_step(m, d)
        if int(round(d.time / 0.002)) % 10 == 0:   # 50 Hz like /nandi/sim_qpos
            tl = tilt_deg(d.qpos[3:7]); maxtilt = max(maxtilt, tl)
            r = core.update_pose(d.time, d.qpos[0:2], tl, is_fallen(d.qpos[0:3], d.qpos[3:7]))
            if r and latched is None: latched = (round(d.time, 3), r, round(tl, 1))
    print(f"push {F:.0f} N: max tilt {maxtilt:.1f} deg, latched={latched}, reset allowed now={core.cause_clear()}, events={core.events}")
