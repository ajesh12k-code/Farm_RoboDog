import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class UnitreeA1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../sim/models/unitree_a1/scene.xml")
        )
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Control frequency: 50 Hz (10 steps * 0.002s timestep = 0.02s)
        self.frame_skip = 10
        self.dt = self.model.opt.timestep * self.frame_skip

        # Save default standing joint angles from keyframe 0 ('home')
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.home_qpos = self.data.qpos[7:].copy()

        # Action: 12 residual joint offsets scaled by 0.4 rad
        self.action_scale = 0.4
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # Observation: [z(1), quat(4), lin_vel(3), ang_vel(3), qpos_err(12), qvel(12), last_action(12)] -> 47 dims
        obs_dim = 47
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.last_action = np.zeros(12, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 1000  # 20 seconds per episode

    def _get_obs(self):
        z = np.array([self.data.qpos[2]], dtype=np.float32)
        quat = self.data.qpos[3:7].astype(np.float32)
        lin_vel = self.data.qvel[0:3].astype(np.float32)
        ang_vel = self.data.qvel[3:6].astype(np.float32)
        qpos_err = (self.data.qpos[7:] - self.home_qpos).astype(np.float32)
        qvel = self.data.qvel[6:].astype(np.float32)
        return np.concatenate([z, quat, lin_vel, ang_vel, qpos_err, qvel, self.last_action])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Small random noise on initial joint angles for robustness
        noise = self.np_random.uniform(low=-0.03, high=0.03, size=12)
        self.data.qpos[7:] += noise
        self.data.ctrl[:] = self.data.qpos[7:].copy()
        mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(12, dtype=np.float32)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_qpos = self.home_qpos + self.action_scale * action

        x_before = self.data.qpos[0]
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = target_qpos
            mujoco.mj_step(self.model, self.data)
        x_after = self.data.qpos[0]

        self.step_count += 1
        obs = self._get_obs()

        # Reward components
        forward_vel = (x_after - x_before) / self.dt
        alive_bonus = 0.5
        ctrl_cost = 0.02 * np.sum(np.square(action - self.last_action))
        lateral_penalty = 0.3 * abs(self.data.qvel[1])
        tilt_penalty = 1.5 * (self.data.qpos[4] ** 2 + self.data.qpos[5] ** 2)

        reward = float(
            2.0 * forward_vel + alive_bonus - ctrl_cost - lateral_penalty - tilt_penalty
        )

        fallen = bool(self.data.qpos[2] < 0.18 or self.data.qpos[3] < 0.5)
        truncated = bool(self.step_count >= self.max_steps)

        self.last_action = action
        return obs, reward, fallen, truncated, {"forward_vel": forward_vel}
