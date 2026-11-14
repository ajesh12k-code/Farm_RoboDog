import mujoco
import mujoco.viewer
import time
import os

model_path = os.path.join(os.path.dirname(__file__), "models/unitree_a1/scene.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Reset to the 'home' standing keyframe defined in a1.xml (keyframe 0)
mujoco.mj_resetDataKeyframe(model, data, 0)
target_qpos = data.qpos[7:].copy()  # Save the 12 target joint angles

# PD gains for position-controlled actuators (a1.xml uses position actuators)
data.ctrl[:] = target_qpos

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        data.ctrl[:] = target_qpos
        mujoco.mj_step(model, data)
        viewer.sync()
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
