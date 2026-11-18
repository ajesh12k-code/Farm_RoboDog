import os  # Import os for resolving relative paths to the MJCF model
import numpy as np  # Import NumPy for state and action vector math
import gymnasium as gym  # Import Gymnasium base environment class
from gymnasium import spaces  # Import Box space for continuous control bounds
import mujoco  # Import MuJoCo physics engine

from disturbance_manager import BalanceDisturbanceManager  # Import terrain & push disturbances
from balance_reward import compute_arm_balance_reward  # Import rest-aware balance reward


def policy_tip_site_id(model):
    """Arm end point the policy observes: site policy_tip (gripper_tip's original 0.065 m position, kept
    when the harvest gripper was lengthened) so new and old policies share one observation; same rule
    as atharv_nandi/core/rl_balance_policy.policy_tip_site_id."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "policy_tip")
    return sid if sid >= 0 else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")


class NandiArmBalanceEnv(gym.Env):
    """
    23-DOF Gymnasium environment where AtharvNandi keeps its dorsal arm
    tucked cleanly in rest position during normal walking, deploying it smoothly
    only to stabilize against slopes, slips, or pushes when the gripper is empty.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, holding_item: bool = False):
        super().__init__()
        self.render_mode = render_mode
        # Flag indicating whether the gripper is carrying a crop (locks arm at rest)
        self.holding_item = holding_item

        # Load the AtharvNandi MJCF model
        model_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../atharv_nandi/sim/models/AtharvNandi/atharv_nandi.xml",
            )
        )
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        # Initialize disturbance manager for uneven ridges and lateral pushes
        self.disturbances = BalanceDisturbanceManager(self.model)

        # Look up IDs for trunk body and the arm end point the policy observes (see policy_tip_site_id)
        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.tip_site_id = policy_tip_site_id(self.model)

        # 50 Hz policy frequency (10 sub-steps * 0.002 s = 0.02 s)
        self.frame_skip = 10
        self.dt = self.model.opt.timestep * self.frame_skip

        # Extract stowed rest pose ('home' keyframe 0) for arm and legs
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.home_arm_qpos = self.data.qpos[7:12].copy()
        self.home_leg_qpos = self.data.qpos[14:32].copy()

        # Action scaling: full reach range for emergency arm bracing, ±0.35 rad for legs
        self.arm_action_scale = np.array([0.24, 1.50, 1.10, 1.10, 0.80], dtype=np.float32)
        self.leg_action_scale = 0.35

        # 23-D action space: [5 dorsal gantry/arm joints + 18 leg joints]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(23,), dtype=np.float32)
        # 84-D observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(84,), dtype=np.float32
        )

        # Low-pass filtered arm command buffer for smooth extension and retraction
        self.filtered_arm_target = self.home_arm_qpos.copy()
        self.last_action = np.zeros(23, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 1000

    def _get_obs(self) -> np.ndarray:
        """Assemble the 84-element whole-body proprioceptive & end-effector state."""
        z = np.array([self.data.qpos[2]], dtype=np.float32)
        quat = self.data.qpos[3:7].astype(np.float32)
        lin_vel = self.data.qvel[0:3].astype(np.float32)
        ang_vel = self.data.qvel[3:6].astype(np.float32)

        tip_rel = (self.data.site_xpos[self.tip_site_id] - self.data.xpos[self.trunk_id]).astype(np.float32)
        hold_flag = np.array([1.0 if self.holding_item else 0.0], dtype=np.float32)

        arm_err = (self.data.qpos[7:12] - self.home_arm_qpos).astype(np.float32)
        arm_vel = self.data.qvel[6:11].astype(np.float32)

        leg_err = (self.data.qpos[14:32] - self.home_leg_qpos).astype(np.float32)
        leg_vel = self.data.qvel[13:31].astype(np.float32)

        return np.concatenate([
            z, quat, lin_vel, ang_vel, tip_rel, hold_flag,
            arm_err, arm_vel, leg_err, leg_vel, self.last_action,
        ])

    def reset(self, seed=None, options=None):
        """Reset pose to stowed arm rest position, randomize terrain, and reset timers."""
        super().reset(seed=seed)
        self.disturbances.reset_terrain(self.np_random)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.data.qpos[14:32] += self.np_random.uniform(-0.03, 0.03, size=18)
        self.filtered_arm_target = self.home_arm_qpos.copy()
        self.data.ctrl[0:5] = self.home_arm_qpos.copy()
        self.data.ctrl[5:7] = 0.0
        self.data.ctrl[7:25] = self.data.qpos[14:32].copy()
        mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(23, dtype=np.float32)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Step physics with smooth arm rest/deploy filtering and 18-DOF leg control."""
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        arm_action = action[0:5]
        leg_action = action[5:23]

        # Compute raw desired arm target and smooth it via exponential low-pass filter
        raw_arm_target = self.home_arm_qpos + self.arm_action_scale * arm_action
        self.filtered_arm_target = 0.65 * self.filtered_arm_target + 0.35 * raw_arm_target
        target_legs = self.home_leg_qpos + self.leg_action_scale * leg_action

        # Apply any active external push disturbance to the torso
        self.disturbances.step_disturbance(self.data, self.np_random)

        x_before = float(self.data.qpos[0])
        for _ in range(self.frame_skip):
            self.data.ctrl[0:5] = self.filtered_arm_target
            self.data.ctrl[5:7] = 0.0
            self.data.ctrl[7:25] = target_legs
            mujoco.mj_step(self.model, self.data)
        x_after = float(self.data.qpos[0])

        self.step_count += 1
        obs = self._get_obs()

        # Pass live arm posture error and joint velocity to enforce rest posture when calm
        arm_qpos_err = self.data.qpos[7:12] - self.home_arm_qpos
        arm_qvel = self.data.qvel[6:11]

        reward, fallen, info = compute_arm_balance_reward(
            x_before=x_before,
            x_after=x_after,
            dt=self.dt,
            qpos=self.data.qpos,
            qvel=self.data.qvel,
            arm_qpos_err=arm_qpos_err,
            arm_qvel=arm_qvel,
            leg_action=leg_action,
            last_action=self.last_action,
            full_action=action,
            tip_world_pos=self.data.site_xpos[self.tip_site_id],
            trunk_world_pos=self.data.xpos[self.trunk_id],
            holding_item=self.holding_item,
        )
        truncated = bool(self.step_count >= self.max_steps)
        self.last_action = action
        return obs, reward, fallen, truncated, info
