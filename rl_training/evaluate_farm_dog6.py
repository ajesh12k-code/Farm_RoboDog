import os  # Import os to build file paths and check for saved policy files
import glob  # Import glob to find numbered PPO checkpoint archives
import time  # Import time to pace rendering at real-time 50 Hz speed
import mujoco  # Import MuJoCo physics engine bindings
import mujoco.viewer  # Import MuJoCo interactive 3D viewer
from stable_baselines3 import PPO  # Import PPO loader for trained .zip policies
from farm_dog6_env import FarmDog6LocomotionEnv  # Import our custom 18-DOF hexapod environment


def find_best_policy(models_dir: str) -> str:
    """Locate farmdog6_ppo_final.zip or the highest-step checkpoint in models/."""
    # Construct path to the completed final training archive
    final_zip = os.path.join(models_dir, "farmdog6_ppo_final.zip")
    # Use the final policy immediately if training completed
    if os.path.exists(final_zip):
        return final_zip

    # Otherwise search for all periodic step checkpoints
    checkpoints = glob.glob(os.path.join(models_dir, "farmdog6_ppo_*_steps.zip"))
    # Raise a clear error if no trained checkpoint exists yet
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {models_dir}. Run train_farm_dog6.py first.")

    # Sort checkpoints numerically by the step count embedded in the filename
    return max(checkpoints, key=lambda path: int(path.split("_")[-2]))


def main():
    # Resolve absolute path to the rl_training/models directory
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    # Pick the latest trained PPO policy archive
    policy_path = find_best_policy(models_dir)
    # Print which checkpoint is being evaluated
    print(f"Loading trained policy: {policy_path}")

    # Instantiate our custom KisaanuNexusFarmDog6 Gymnasium environment
    env = FarmDog6LocomotionEnv()
    # Load the trained neural network weights onto CPU
    model = PPO.load(policy_path, device="cpu")

    # Look up the MuJoCo body ID of the robot's main chassis ('trunk')
    trunk_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

    # Reset the environment (randomizes uneven farm ridges and resets robot pose)
    obs, _ = env.reset()

    # Open the passive MuJoCo 3D viewer attached to env.model and env.data
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Set camera mode to automatically track a moving body
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        # Lock camera tracking onto KisaanuNexusFarmDog6's 'trunk' body
        viewer.cam.trackbodyid = trunk_id
        # Set comfortable viewing distance (1.8 m), side-quarter azimuth, and elevation angle
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0

        # Continue stepping until the user closes the MuJoCo window
        while viewer.is_running():
            # Record wall-clock start time for real-time 50 Hz pacing
            step_start = time.time()

            # Query the trained policy deterministically (no exploration noise)
            action, _ = model.predict(obs, deterministic=True)
            # Advance the robot by one 20 ms control step and collect telemetry
            obs, reward, terminated, truncated, info = env.step(action)

            # Sync the 3D viewer graphics with the updated physics state
            viewer.sync()

            # If the robot fell or finished the 20-second episode, print stats and reset
            if terminated or truncated:
                print(
                    f"Episode finished | Distance X: {info['x_pos']:.2f} m | "
                    f"Last Vel: {info['forward_vel']:.2f} m/s"
                )
                obs, _ = env.reset()

            # Sleep for any remaining time in the 0.02 s control step so playback is 1x real-time
            elapsed = time.time() - step_start
            if env.dt - elapsed > 0:
                time.sleep(env.dt - elapsed)


# Standard Python entrypoint guard
if __name__ == "__main__":
    main()
