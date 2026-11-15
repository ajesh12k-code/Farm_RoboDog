import os
import glob
import numpy as np
import mujoco

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


class RLBalanceController:
    """Loads the 23-DOF PPO balance policy and computes whole-body arm/leg targets."""

    def __init__(self, xml_path: str, dt: float = 0.02):
        self.dt = dt
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
        self.home_arm = self.data.qpos[7:12].copy()
        self.home_legs = self.data.qpos[14:32].copy()

        self.arm_scale = np.array([0.24, 1.50, 1.10, 1.10, 0.80], dtype=np.float32)
        self.leg_scale = 0.35
        self.last_action = np.zeros(23, dtype=np.float32)
        self.filtered_arm = self.home_arm.copy()
        self.policy, self.policy_path = self._load_latest_policy()

    def _load_latest_policy(self):
        if not HAS_SB3:
            return None, None
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rl_training/models"))
        final_zip = os.path.join(models_dir, "farmdog6_balance_final.zip")
        if os.path.exists(final_zip):
            return PPO.load(final_zip, device="cpu"), final_zip
        ckpts = glob.glob(os.path.join(models_dir, "farmdog6_balance_*_steps.zip"))
        if not ckpts:
            return None, None
        best = max(ckpts, key=lambda p: int(p.split("_")[-2]))
        return PPO.load(best, device="cpu"), best

    def reset_arm_to_rest(self):
        self.last_action[:] = 0.0
        self.filtered_arm = 0.75 * self.filtered_arm + 0.25 * self.home_arm
        return self.filtered_arm.tolist()

    def predict(self, prev_qpos: np.ndarray, curr_qpos: np.ndarray):
        self.data.qpos[:] = curr_qpos
        mujoco.mj_kinematics(self.model, self.data)

        lin_vel = ((curr_qpos[0:3] - prev_qpos[0:3]) / self.dt).astype(np.float32)
        ang_vel = ((curr_qpos[4:7] - prev_qpos[4:7]) * (2.0 / self.dt)).astype(np.float32)
        tip_rel = (self.data.site_xpos[self.tip_site_id] - self.data.xpos[self.trunk_id]).astype(np.float32)

        obs = np.concatenate([
            np.array([curr_qpos[2]], dtype=np.float32),
            curr_qpos[3:7].astype(np.float32),
            lin_vel,
            ang_vel,
            tip_rel,
            np.array([0.0], dtype=np.float32),
            (curr_qpos[7:12] - self.home_arm).astype(np.float32),
            ((curr_qpos[7:12] - prev_qpos[7:12]) / self.dt).astype(np.float32),
            (curr_qpos[14:32] - self.home_legs).astype(np.float32),
            ((curr_qpos[14:32] - prev_qpos[14:32]) / self.dt).astype(np.float32),
            self.last_action,
        ])

        action, _ = self.policy.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.last_action = action

        raw_arm = self.home_arm + self.arm_scale * action[0:5]
        self.filtered_arm = 0.65 * self.filtered_arm + 0.35 * raw_arm
        legs = (self.home_legs + self.leg_scale * action[5:23]).tolist()
        return self.filtered_arm.tolist(), legs
