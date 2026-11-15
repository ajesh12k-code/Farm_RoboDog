#!/usr/bin/env python3
import os
import json
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


class FarmDog6SimRunner(Node):
    """Lightweight MuJoCo physics & 3D renderer subscribing to modular service outputs."""

    def __init__(self):
        super().__init__("farmdog6_sim_runner")
        xml_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "models/KisaanuNexusFarmDog6/kisaanu_nexus_farm_dog6.xml",
            )
        )
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.arm_targets = list(self.data.ctrl[:7])
        self.leg_targets = list(self.data.ctrl[7:])
        self.canopy_segs = None
        self.floor_segs = None

        self.create_subscription(Float64MultiArray, "/farmdog6/arm_targets", self.on_arm, 10)
        self.create_subscription(Float64MultiArray, "/farmdog6/leg_targets", self.on_legs, 10)
        self.create_subscription(String, "/farmdog6/workspace_wireframe", self.on_wireframe, 10)
        self.qpos_pub = self.create_publisher(Float64MultiArray, "/farmdog6/sim_qpos", 10)

    def on_arm(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            self.arm_targets = list(msg.data)

    def on_legs(self, msg: Float64MultiArray):
        if len(msg.data) == 18:
            self.leg_targets = list(msg.data)

    def on_wireframe(self, msg: String):
        if not msg.data:
            self.canopy_segs = None
            return
        data = json.loads(msg.data)
        self.canopy_segs = np.array(data["canopy_segments"], dtype=np.float64)
        self.floor_segs = np.array(data["floor_segments"], dtype=np.float64)

    def _render_wireframe(self, viewer):
        scn = viewer.user_scn
        scn.ngeom = 0
        if self.canopy_segs is None:
            return

        t_pos = self.data.xpos[self.trunk_id]
        t_rot = self.data.xmat[self.trunk_id].reshape(3, 3)
        c1 = (self.canopy_segs[:, 0, :] @ t_rot.T) + t_pos
        c2 = (self.canopy_segs[:, 1, :] @ t_rot.T) + t_pos
        f1 = (self.floor_segs[:, 0, :] @ t_rot.T) + t_pos
        f2 = (self.floor_segs[:, 1, :] @ t_rot.T) + t_pos

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

    def run_sim(self):
        dt = self.model.opt.timestep
        step_counter = 0
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and rclpy.ok():
                t0 = time.time()
                self.data.ctrl[:7] = self.arm_targets
                self.data.ctrl[7:] = self.leg_targets
                mujoco.mj_step(self.model, self.data)

                step_counter += 1
                if step_counter % 10 == 0:  # Publish state at 50 Hz
                    msg = Float64MultiArray()
                    msg.data = self.data.qpos.tolist()
                    self.qpos_pub.publish(msg)

                self._render_wireframe(viewer)
                viewer.sync()
                sleep_t = dt - (time.time() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)


def main():
    rclpy.init()
    node = FarmDog6SimRunner()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        node.run_sim()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
