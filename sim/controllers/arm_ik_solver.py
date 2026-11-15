import numpy as np
import mujoco


class DorsalArmIKSolver:
    """Damped Least-Squares Jacobian IK solver for KisaanuNexusFarmDog6's 5-DOF arm."""

    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.crop_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "crop_item_1")
        self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
        self.limits = np.array([
            [-0.26, 0.26], [-3.14, 3.14], [-1.60, 2.40], [-2.60, 2.60], [-2.00, 2.00]
        ], dtype=np.float64)

    def get_scene_poses(self, qpos: np.ndarray):
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        return (
            self.data.xpos[self.trunk_id].copy(),
            self.data.xmat[self.trunk_id].reshape(3, 3).copy(),
            self.data.xpos[self.crop_id].copy(),
        )

    def solve(self, qpos: np.ndarray, target_pos: np.ndarray, seed_q: np.ndarray, max_iters: int = 120):
        self.data.qpos[:] = qpos
        self.data.qpos[7:12] = seed_q[:5]
        mujoco.mj_forward(self.model, self.data)

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        for _ in range(max_iters):
            mujoco.mj_kinematics(self.model, self.data)
            err = target_pos - self.data.site_xpos[self.tip_site_id]
            if np.linalg.norm(err) < 0.004:
                break
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.tip_site_id)
            J = jacp[:, 6:11]
            dq = J.T @ np.linalg.solve(J @ J.T + (0.04 ** 2) * np.eye(3), err)
            self.data.qpos[7:12] = np.clip(self.data.qpos[7:12] + 0.65 * dq, self.limits[:, 0], self.limits[:, 1])
        return self.data.qpos[7:12].copy()
