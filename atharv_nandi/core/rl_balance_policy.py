"""23-DOF PPO balance policy inference for the live world.

The model comes from ``load_farm_model()`` (robot + crops + terrain) so ``nq`` matches
/nandi/sim_qpos. The observation uses ONLY the robot prefix of qpos/qvel ([0:32] / [0:31]),
built exactly like rl_training/nandi_balance_env.py::_get_obs, so a policy trained on the raw
robot MJCF stays compatible.

torch / stable-baselines3 and the trained zip are optional: without them ``policy`` is None,
``policy_path`` is None and the locomotion service stays in tripod mode (and says so).
"""
import glob
import os
from typing import Optional, Tuple

import mujoco
import numpy as np

from atharv_nandi import paths
from atharv_nandi.core.farm_scene import load_farm_model

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:  # runtime image ships without torch/SB3
    PPO = None
    HAS_SB3 = False

OBS_DIM = 84
ACTION_DIM = 23
MODELS_DIR = os.path.join(paths.REPO, "rl_training", "models")
# The policy observes (tip_rel) the arm end point at its original 0.065 m from the wrist pivot. The
# longer harvest gripper moved gripper_tip to 0.105 m; the MJCF keeps the old point as site policy_tip
# so existing PPO policies get the observation they were trained on (rl_training/nandi_balance_env.py
# uses the same site). Models without it (older MJCFs) fall back to gripper_tip.
POLICY_TIP_SITE = "policy_tip"


def policy_tip_site_id(model: mujoco.MjModel) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, POLICY_TIP_SITE)
    return sid if sid >= 0 else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")


# Checkpoint name prefixes, newest first. "farmdog6_" is the pre-rename name: policies trained by the
# original FarmDog6 code (e.g. upstream's farmdog6_balance_final.zip) still load.
POLICY_PREFIXES = ("nandi_balance", "farmdog6_balance")


def find_policy_zip(models_dir: str = MODELS_DIR) -> Optional[str]:
    """<prefix>_final.zip if present (nandi_ before farmdog6_), else the checkpoint with the most steps, else None."""
    for prefix in POLICY_PREFIXES:
        final_zip = os.path.join(models_dir, f"{prefix}_final.zip")
        if os.path.exists(final_zip):
            return final_zip
    best, best_steps = None, -1
    paths = [p for prefix in POLICY_PREFIXES for p in glob.glob(os.path.join(models_dir, f"{prefix}_*_steps.zip"))]
    for path in paths:
        try:
            steps = int(os.path.basename(path).split("_")[-2])
        except ValueError:
            continue
        if steps > best_steps:
            best, best_steps = path, steps
    return best


class RLBalanceController:
    """Loads the 23-DOF PPO balance policy (if available) and computes whole-body arm/leg targets."""

    def __init__(self, dt: float = 0.02, model: Optional[mujoco.MjModel] = None, load_policy: bool = True):
        self.dt = dt
        self.model = model if model is not None else load_farm_model()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.tip_site_id = policy_tip_site_id(self.model)
        self.home_arm = self.data.qpos[7:12].copy()
        self.home_legs = self.data.qpos[14:32].copy()

        self.arm_scale = np.array([0.24, 1.50, 1.10, 1.10, 0.80], dtype=np.float32)
        self.leg_scale = 0.35
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.filtered_arm = self.home_arm.copy()
        self._qvel = np.zeros(self.model.nv)
        self.load_error: Optional[str] = None
        self.policy, self.policy_path = self._load_policy() if load_policy else (None, None)

    def _load_policy(self) -> Tuple[object, Optional[str]]:
        if not HAS_SB3:
            self.load_error = "stable-baselines3 not installed"
            return None, None
        path = find_policy_zip()
        if path is None:
            self.load_error = f"no policy zip in {MODELS_DIR}"
            return None, None
        try:
            policy = PPO.load(path, device="cpu")
        except Exception as exc:  # corrupt/incompatible zip: degrade to tripod, never crash the node
            self.load_error = f"failed to load {path}: {exc}"
            return None, None
        shape = tuple(getattr(policy.observation_space, "shape", ()) or ())
        if shape != (OBS_DIM,):
            self.load_error = f"{path}: observation shape {shape} != ({OBS_DIM},)"
            return None, None
        return policy, path

    def reset_arm_to_rest(self):
        self.last_action[:] = 0.0
        self.filtered_arm = 0.75 * self.filtered_arm + 0.25 * self.home_arm
        return self.filtered_arm.tolist()

    def build_obs(self, prev_qpos: np.ndarray, curr_qpos: np.ndarray) -> np.ndarray:
        """84-D observation from two consecutive full-world qpos samples dt apart (robot prefix only)."""
        self.data.qpos[:] = curr_qpos
        mujoco.mj_kinematics(self.model, self.data)
        # Same tangent-space velocity MuJoCo integrates: free joint linear (world) + angular (body).
        mujoco.mj_differentiatePos(self.model, self._qvel, self.dt,
                                   np.asarray(prev_qpos, dtype=np.float64), np.asarray(curr_qpos, dtype=np.float64))
        qpos, qvel = self.data.qpos, self._qvel
        tip_rel = self.data.site_xpos[self.tip_site_id] - self.data.xpos[self.trunk_id]
        return np.concatenate([
            qpos[2:3], qpos[3:7], qvel[0:3], qvel[3:6], tip_rel, [0.0],
            qpos[7:12] - self.home_arm, qvel[6:11],
            qpos[14:32] - self.home_legs, qvel[13:31],
            self.last_action,
        ]).astype(np.float32)

    def predict(self, prev_qpos: np.ndarray, curr_qpos: np.ndarray):
        if self.policy is None:
            raise RuntimeError("RL policy not loaded")
        obs = self.build_obs(prev_qpos, curr_qpos)
        action, _ = self.policy.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        self.last_action = action

        raw_arm = self.home_arm + self.arm_scale * action[0:5]
        self.filtered_arm = 0.65 * self.filtered_arm + 0.35 * raw_arm
        legs = (self.home_legs + self.leg_scale * action[5:23]).tolist()
        return self.filtered_arm.tolist(), legs
