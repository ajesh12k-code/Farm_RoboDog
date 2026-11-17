import os  # Import os to build reliable relative paths to the XML model
import numpy as np  # Import NumPy for array operations and observation vectors
import gymnasium as gym  # Import Gymnasium base class for RL environments
from gymnasium import spaces  # Import Box spaces to define action and observation bounds
import mujoco  # Import MuJoCo physics engine

from terrain_manager import FarmTerrainRandomizer  # Import modular farm ridge randomizer
from reward_functions import compute_locomotion_reward  # Import modular reward function


class NandiLocomotionEnv(gym.Env):
    """Gymnasium environment for 18-DOF hexapod locomotion on uneven farm terrain."""

    # Declare supported render modes and 50 Hz control frame rate
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        # Initialize parent gym.Env class
        super().__init__()
        # Store requested render mode (None for fast headless training)
        self.render_mode = render_mode

        # Resolve absolute path to the AtharvNandi MJCF XML file
        model_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../sim/models/AtharvNandi/atharv_nandi.xml",
            )
        )
        # Load the static physics model from XML
        self.model = mujoco.MjModel.from_xml_path(model_path)
        # Allocate dynamic simulation state data buffer
        self.data = mujoco.MjData(self.model)
        # Initialize terrain randomizer helper for uneven farm ridges
        self.terrain = FarmTerrainRandomizer(self.model)

        # Execute 10 physics sub-steps (10 * 0.002 s = 0.02 s -> 50 Hz policy frequency)
        self.frame_skip = 10
        # Compute effective control timestep in seconds
        self.dt = self.model.opt.timestep * self.frame_skip

        # Reset simulation to keyframe 0 ('home') to extract reference standing poses
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # Save the 7 stowed dorsal gantry/arm actuator targets (indices 0..6)
        self.stowed_arm_ctrl = self.data.ctrl[0:7].copy()
        # Save the 18 standing leg joint angles (qpos indices 14..31)
        self.home_leg_qpos = self.data.qpos[14:32].copy()

        # Scale normalized [-1, 1] policy outputs to ±0.35 rad residual joint offsets
        self.action_scale = 0.35
        # Define 18-dimensional continuous action space for the 6 legs (3 DOFs per leg)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)

        # Define 65-dimensional continuous observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(65,), dtype=np.float32
        )

        # Buffer storing the previous step's action for smoothness penalties and observation
        self.last_action = np.zeros(18, dtype=np.float32)
        # Track elapsed steps in the current episode
        self.step_count = 0
        # Cap each training episode at 1000 steps (20 seconds of simulated walking)
        self.max_steps = 1000

    def _get_obs(self) -> np.ndarray:
        """Assemble the 65-element proprioceptive state vector."""
        # Torso height above floor (1)
        z = np.array([self.data.qpos[2]], dtype=np.float32)
        # Torso orientation quaternion [w, x, y, z] (4)
        quat = self.data.qpos[3:7].astype(np.float32)
        # Torso linear velocity [vx, vy, vz] (3)
        lin_vel = self.data.qvel[0:3].astype(np.float32)
        # Torso angular velocity [wx, wy, wz] (3)
        ang_vel = self.data.qvel[3:6].astype(np.float32)
        # Current 18 leg joint angles minus standing home angles (18)
        leg_qpos_err = (self.data.qpos[14:32] - self.home_leg_qpos).astype(np.float32)
        # Current 18 leg joint angular velocities from qvel[13..30] (18)
        leg_qvel = self.data.qvel[13:31].astype(np.float32)
        # Concatenate all terms plus last_action (18) into a flat 65-D float32 array
        return np.concatenate([z, quat, lin_vel, ang_vel, leg_qpos_err, leg_qvel, self.last_action])

    def reset(self, seed=None, options=None):
        """Reset episode state, randomize uneven terrain, and return initial observation."""
        # Seed Gymnasium's internal random number generator (self.np_random)
        super().reset(seed=seed)
        # Randomize farm ridge heights and positions for this episode
        self.terrain.randomize(self.np_random)
        # Reset robot pose and velocities back to keyframe 0 ('home')
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Add small uniform noise (±0.03 rad) to initial leg angles for robustness
        noise = self.np_random.uniform(low=-0.03, high=0.03, size=18)
        self.data.qpos[14:32] += noise
        # Hold the 7 dorsal gantry/arm actuators in their stowed pose
        self.data.ctrl[0:7] = self.stowed_arm_ctrl
        # Match initial leg actuator targets to the perturbed starting angles
        self.data.ctrl[7:25] = self.data.qpos[14:32].copy()
        # Recompute forward kinematics and derived quantities
        mujoco.mj_forward(self.model, self.data)

        # Clear previous action memory and reset episode step counter
        self.last_action = np.zeros(18, dtype=np.float32)
        self.step_count = 0
        # Return initial observation and empty info dict per Gymnasium API
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Advance physics by 20 ms using residual leg targets and compute reward."""
        # Clip incoming policy actions strictly within [-1.0, 1.0]
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        # Compute absolute joint position targets around the standing posture
        target_leg_qpos = self.home_leg_qpos + self.action_scale * action

        # Record torso X-position before stepping physics
        x_before = float(self.data.qpos[0])
        # Step MuJoCo `frame_skip` times at 500 Hz physics resolution
        for _ in range(self.frame_skip):
            # Keep dorsal gantry and arm locked in stowed position
            self.data.ctrl[0:7] = self.stowed_arm_ctrl
            # Apply target joint positions to the 18 leg position actuators
            self.data.ctrl[7:25] = target_leg_qpos
            # Advance physics simulation by one 2 ms timestep
            mujoco.mj_step(self.model, self.data)
        # Record torso X-position after stepping physics
        x_after = float(self.data.qpos[0])

        # Increment step counter and build next observation
        self.step_count += 1
        obs = self._get_obs()

        # Evaluate modular reward function and fall termination check
        reward, fallen, info = compute_locomotion_reward(
            x_before, x_after, self.dt, self.data.qpos, self.data.qvel, action, self.last_action
        )
        # Check if episode reached the 1000-step time limit
        truncated = bool(self.step_count >= self.max_steps)

        # Store current action for next step's smoothness penalty
        self.last_action = action
        # Return standard Gymnasium 5-tuple
        return obs, reward, fallen, truncated, info
