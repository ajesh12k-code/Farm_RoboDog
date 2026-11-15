import numpy as np  # Import NumPy for sampling random push vectors
import mujoco  # Import MuJoCo bindings to inspect geometry and body IDs


class BalanceDisturbanceManager:
    """Randomizes farm ridges and applies periodic lateral/pitch pushes to test arm recovery."""

    def __init__(self, model: mujoco.MjModel):
        # Store reference to the compiled MuJoCo physics model
        self.model = model
        # Look up body ID of the main chassis ('trunk') for applying external pushes
        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        # Look up geom IDs of the two uneven farm ridge obstacles
        self.ridge1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ridge_1")
        self.ridge2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ridge_2")

        # Countdown timer (in control steps) until the next external push disturbance
        self.steps_until_push = 80
        # Remaining control steps for which the active push force is sustained
        self.push_duration_left = 0
        # Current 3D world force vector [fx, fy, fz] in Newtons applied to the trunk
        self.current_force = np.zeros(3, dtype=np.float64)

    def reset_terrain(self, rng):
        """Randomize farm ridge heights and positions at the start of each episode."""
        # Vary ridge_1 height between 2.0 cm and 5.5 cm
        if self.ridge1_id != -1:
            h1 = rng.uniform(0.020, 0.055)
            self.model.geom_size[self.ridge1_id, 2] = h1
            self.model.geom_pos[self.ridge1_id, 2] = h1
            self.model.geom_pos[self.ridge1_id, 0] = rng.uniform(0.8, 1.4)

        # Vary ridge_2 height between 2.5 cm and 6.5 cm
        if self.ridge2_id != -1:
            h2 = rng.uniform(0.025, 0.065)
            self.model.geom_size[self.ridge2_id, 2] = h2
            self.model.geom_pos[self.ridge2_id, 2] = h2
            self.model.geom_pos[self.ridge2_id, 0] = rng.uniform(1.6, 2.3)

        # Reset external push timers and clear residual forces
        self.steps_until_push = int(rng.integers(50, 110))
        self.push_duration_left = 0
        self.current_force[:] = 0.0

    def step_disturbance(self, data: mujoco.MjData, rng):
        """Apply a brief sideways/forward impulse every ~1.5–2.5 seconds to induce tilt."""
        # Clear any previously applied external body forces
        data.xfrc_applied[self.trunk_id, :] = 0.0

        # If a push impulse is currently active, apply it and decrement its duration
        if self.push_duration_left > 0:
            data.xfrc_applied[self.trunk_id, 0:3] = self.current_force
            self.push_duration_left -= 1
            return

        # Count down until the next random push event
        self.steps_until_push -= 1
        if self.steps_until_push <= 0:
            # Sample a strong lateral force (±45 N to ±85 N) and smaller fore/aft force (±30 N)
            fy_sign = -1.0 if rng.random() < 0.5 else 1.0
            fy = fy_sign * rng.uniform(45.0, 85.0)
            fx = rng.uniform(-30.0, 30.0)
            # Store force vector and sustain it for 12 control steps (0.24 seconds)
            self.current_force = np.array([fx, fy, 0.0], dtype=np.float64)
            self.push_duration_left = 12
            # Schedule the next push 75 to 140 control steps later
            self.steps_until_push = int(rng.integers(75, 140))
