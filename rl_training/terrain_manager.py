import mujoco  # Import MuJoCo bindings to inspect and modify simulation geometry IDs


class FarmTerrainRandomizer:
    """Randomizes uneven farm ridges in MuJoCo before each training episode."""

    def __init__(self, model: mujoco.MjModel):
        # Store reference to the compiled MuJoCo physics model
        self.model = model
        # Look up the integer geom ID for the first farm ridge obstacle
        self.ridge1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ridge_1")
        # Look up the integer geom ID for the second farm ridge obstacle
        self.ridge2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ridge_2")

    def randomize(self, rng):
        """Apply random heights (1.5 cm to 5.5 cm) and X-offsets using the env RNG."""
        # Check that ridge_1 exists in the loaded XML model
        if self.ridge1_id != -1:
            # Sample half-height between 1.5 cm and 4.5 cm
            h1 = rng.uniform(0.015, 0.045)
            # Update the Z half-extent (size[2]) of ridge_1
            self.model.geom_size[self.ridge1_id, 2] = h1
            # Shift ridge_1's center Z-position so its bottom stays flush with the floor
            self.model.geom_pos[self.ridge1_id, 2] = h1
            # Randomize ridge_1's forward distance between 0.9 m and 1.5 m
            self.model.geom_pos[self.ridge1_id, 0] = rng.uniform(0.9, 1.5)

        # Check that ridge_2 exists in the loaded XML model
        if self.ridge2_id != -1:
            # Sample half-height between 2.0 cm and 5.5 cm
            h2 = rng.uniform(0.020, 0.055)
            # Update the Z half-extent (size[2]) of ridge_2
            self.model.geom_size[self.ridge2_id, 2] = h2
            # Shift ridge_2's center Z-position so its bottom stays flush with the floor
            self.model.geom_pos[self.ridge2_id, 2] = h2
            # Randomize ridge_2's forward distance between 1.8 m and 2.5 m
            self.model.geom_pos[self.ridge2_id, 0] = rng.uniform(1.8, 2.5)
