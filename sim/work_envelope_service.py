#!/usr/bin/env python3
import os
import json
import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


def discover_manipulator_joints(model: mujoco.MjModel, root_arm_body: str = "arm_carriage"):
    """Automatically find all joints in the manipulator subtree and read their XML limits."""
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_arm_body)
    if root_id == -1:
        raise ValueError(f"Manipulator root body '{root_arm_body}' not found in MJCF model.")

    # Collect all body IDs that are descendants of root_arm_body
    arm_body_ids = {root_id}
    for body_id in range(model.nbody):
        parent = body_id
        while parent > 0:
            if parent in arm_body_ids:
                arm_body_ids.add(body_id)
                break
            parent = model.body_parentid[parent]

    joints_info = []
    for jnt_id in range(model.njnt):
        if model.jnt_bodyid[jnt_id] in arm_body_ids:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or f"joint_{jnt_id}"
            jtype = model.jnt_type[jnt_id]
            qpos_adr = int(model.jnt_qposadr[jnt_id])
            low, high = float(model.jnt_range[jnt_id, 0]), float(model.jnt_range[jnt_id, 1])
            # Fallback if joint is unlimited
            if not model.jnt_limited[jnt_id]:
                low, high = (-np.pi, np.pi) if jtype == mujoco.mjtJoint.mjJNT_HINGE else (-0.25, 0.25)
            joints_info.append({
                "id": jnt_id,
                "name": name,
                "type": int(jtype),
                "qpos_adr": qpos_adr,
                "low": low,
                "high": high,
            })
    return joints_info


def compute_dynamic_wireframe(
    xml_path: str,
    base_body_name: str = "trunk",
    tip_site_name: str = "gripper_tip",
    arm_root_name: str = "arm_carriage",
):
    """
    Model-agnostic 3D work envelope generator.
    Reads joint types, ranges, and link geometry directly from the MJCF file.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_site_name)
    joints = discover_manipulator_joints(model, arm_root_name)

    # Identify prismatic (slide) vs revolute (hinge) joints dynamically
    slide_joints = [j for j in joints if j["type"] == mujoco.mjtJoint.mjJNT_SLIDE and "finger" not in j["name"]]
    hinge_joints = [j for j in joints if j["type"] == mujoco.mjtJoint.mjJNT_HINGE]

    if len(hinge_joints) < 2:
        raise RuntimeError("Need at least 2 hinge joints in the arm to sweep a 3D envelope.")

    yaw_jnt = hinge_joints[0]       # Base turret rotation
    shoulder_jnt = hinge_joints[1]  # Primary elevation pitch
    elbow_jnt = hinge_joints[2] if len(hinge_joints) > 2 else None
    gantry_jnt = slide_joints[0] if slide_joints else None

    base_pos = data.xpos[base_id].copy()
    base_rot = data.xmat[base_id].reshape(3, 3).copy()
    floor_z_world = 0.015

    def eval_local_tip(gantry_val, yaw_val, sh_val, el_val=0.0):
        if gantry_jnt is not None:
            data.qpos[gantry_jnt["qpos_adr"]] = gantry_val
        data.qpos[yaw_jnt["qpos_adr"]] = yaw_val
        data.qpos[shoulder_jnt["qpos_adr"]] = sh_val
        if elbow_jnt is not None:
            data.qpos[elbow_jnt["qpos_adr"]] = el_val
        mujoco.mj_kinematics(model, data)

        w_pt = data.site_xpos[tip_id].copy()
        w_pt[2] = max(floor_z_world, w_pt[2])
        # Express in base_body local coordinates so envelope moves/rotates with any chassis
        return ((w_pt - base_pos) @ base_rot).tolist()

    canopy_segs = []
    floor_segs = []

    gantry_extremes = [gantry_jnt["low"], gantry_jnt["high"]] if gantry_jnt else [0.0]
    shoulder_arc = np.linspace(max(-0.3, shoulder_jnt["low"]), shoulder_jnt["high"], 22)
    yaw_fan = np.linspace(yaw_jnt["low"], yaw_jnt["high"], 25)
    floor_local_threshold = (floor_z_world - base_pos[2]) + 0.035

    # 1. Sweep meridian ribs and horizontal latitude rings at each gantry extreme
    for gx in gantry_extremes:
        grid = []
        for yaw in yaw_fan[:-1]:
            rib = [eval_local_tip(gx, yaw, sh, 0.0) for sh in shoulder_arc]
            grid.append(rib)
            for k in range(len(rib) - 1):
                if rib[k][2] <= floor_local_threshold and rib[k + 1][2] <= floor_local_threshold:
                    floor_segs.append([rib[k], rib[k + 1]])
                else:
                    canopy_segs.append([rib[k], rib[k + 1]])

        n_yaws = len(grid)
        for k in range(0, len(shoulder_arc), 3):
            is_floor_ring = (k >= len(shoulder_arc) - 2)
            for y_idx in range(n_yaws):
                p1 = grid[y_idx][k]
                p2 = grid[(y_idx + 1) % n_yaws][k]
                if is_floor_ring:
                    floor_segs.append([p1, p2])
                else:
                    canopy_segs.append([p1, p2])

    # 2. Sweep longitudinal gantry ridges connecting rear to front along the sliding axis
    if gantry_jnt is not None:
        gantry_steps = np.linspace(gantry_jnt["low"], gantry_jnt["high"], 10)
        yaw_ridges = np.linspace(yaw_jnt["low"] * 0.5, yaw_jnt["high"] * 0.5, 5)
        sh_ridges = np.linspace(max(0.0, shoulder_jnt["low"]), shoulder_jnt["high"], 4)
        for yaw in yaw_ridges:
            for sh in sh_ridges:
                ridge = [eval_local_tip(gx, yaw, sh, 0.0) for gx in gantry_steps]
                is_floor = (sh >= sh_ridges[-1] - 0.05)
                for k in range(len(ridge) - 1):
                    if is_floor:
                        floor_segs.append([ridge[k], ridge[k + 1]])
                    else:
                        canopy_segs.append([ridge[k], ridge[k + 1]])

    return {
        "joints_discovered": [j["name"] for j in joints],
        "canopy_segments": canopy_segs,
        "floor_segments": floor_segs,
    }


class WorkEnvelopeServiceNode(Node):
    def __init__(self):
        super().__init__("work_envelope_service")
        default_xml = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "models/KisaanuNexusFarmDog6/kisaanu_nexus_farm_dog6.xml",
            )
        )
        self.declare_parameter("xml_path", default_xml)
        self.srv = self.create_service(
            Trigger, "/compute_work_envelope", self.handle_compute_request
        )
        self.get_logger().info(
            "ROS 2 Service '/compute_work_envelope' ready (model-agnostic kinematic solver)."
        )

    def handle_compute_request(self, request, response):
        xml_path = self.get_parameter("xml_path").get_parameter_value().string_value
        try:
            payload = compute_dynamic_wireframe(xml_path)
            response.success = True
            response.message = json.dumps(payload)
            self.get_logger().info(
                f"Computed work envelope for joints {payload['joints_discovered']}: "
                f"{len(payload['canopy_segments'])} canopy + {len(payload['floor_segments'])} floor segments."
            )
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f"Work envelope calculation failed: {exc}")
        return response


def main():
    rclpy.init()
    node = WorkEnvelopeServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
