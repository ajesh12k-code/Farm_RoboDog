import os  # Import os to build file paths and check whether policy archives exist
import glob  # Import glob to search for numbered PPO checkpoint files
import time  # Import time to pace simulation rendering at real-time 50 Hz speed
import numpy as np  # Import NumPy to inspect applied push force magnitudes
import mujoco  # Import MuJoCo physics engine bindings
import mujoco.viewer  # Import MuJoCo interactive 3D viewer
from stable_baselines3 import PPO  # Import PPO loader for trained .zip policies
from farm_dog6_balance_env import FarmDog6ArmBalanceEnv  # Import our 23-DOF arm-balance environment


def find_best_balance_policy(models_dir: str) -> str:
    """Locate farmdog6_balance_final.zip or the highest-step checkpoint in models/."""
    # Construct the path to the completed final training archive
    final_zip = os.path.join(models_dir, "farmdog6_balance_final.zip")
    # Return the final policy immediately if training has finished
    if os.path.exists(final_zip):
        return final_zip

    # Otherwise search for all periodic step checkpoints saved during training
    checkpoints = glob.glob(os.path.join(models_dir, "farmdog6_balance_*_steps.zip"))
    # Raise a clear error if no trained checkpoint has been saved yet
    if not checkpoints:
        raise FileNotFoundError(
            f"No balance checkpoints found in {models_dir}. Wait for the first 100k checkpoint or run train_farm_dog6_balance.py."
        )

    # Sort checkpoints numerically by the step count embedded in the filename
    return max(checkpoints, key=lambda path: int(path.split("_")[-2]))


def main():
    # Resolve the absolute path to the rl_training/models directory
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    # Select the latest trained 23-DOF balance policy
    policy_path = find_best_balance_policy(models_dir)
    # Display which checkpoint file is being loaded
    print(f"Loading 23-DOF arm-assisted balance policy: {policy_path}")

    # Instantiate our 23-DOF environment with holding_item=False so the arm assists balance
    env = FarmDog6ArmBalanceEnv(holding_item=False)
    # Load the trained neural network weights onto CPU
    model = PPO.load(policy_path, device="cpu")

    # Look up the integer body ID of the main chassis ('trunk') for camera tracking
    trunk_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

    # Reset the environment (randomizes uneven farm ridges and initializes push timers)
    obs, _ = env.reset()
    # Track whether an external push was active on the previous step to avoid log spam
    was_pushed = False

    # Open the passive MuJoCo 3D viewer attached to env.model and env.data
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Configure the camera to automatically follow a moving body
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        # Lock camera tracking onto KisaanuNexusFarmDog6's 'trunk' body
        viewer.cam.trackbodyid = trunk_id
        # Set comfortable viewing distance (1.9 m), three-quarter azimuth, and elevation angle
        viewer.cam.distance = 1.9
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -22.0

        # Continue running until the user closes the MuJoCo window
        while viewer.is_running():
            # Record wall-clock start time for 1x real-time 50 Hz playback
            step_start = time.time()

            # Compute deterministic 23-DOF actions (5 dorsal arm joints + 18 leg joints)
            action, _ = model.predict(obs, deterministic=True)
            # Step the simulation forward by 20 ms and collect balance telemetry
            obs, reward, terminated, truncated, info = env.step(action)

            # Read the 3D push force currently applied to the trunk by BalanceDisturbanceManager
            push_vec = env.data.xfrc_applied[trunk_id, 0:3]
            # Compute total force magnitude in Newtons
            push_mag = float(np.linalg.norm(push_vec))
            # Print a live alert when a new lateral/pitch push hits the robot
            if push_mag > 1.0 and not was_pushed:
                print(
                    f"[PUSH EVENT] Force: ({push_vec[0]:+.1f}, {push_vec[1]:+.1f}) N | "
                    f"Tilt: {info['tilt_mag']:.3f} rad | Arm Tip Z: {info['tip_z']:.2f} m"
                )
            was_pushed = push_mag > 1.0

            # Sync the 3D viewer graphics with the updated physics state
            viewer.sync()

            # Reset and print summary statistics when the robot falls or completes 20 seconds
            if terminated or truncated:
                status = "FALLEN" if terminated else "SURVIVED 20s"
                print(
                    f"[{status}] Distance X: {info['x_pos']:.2f} m | "
                    f"Forward Vel: {info['forward_vel']:.2f} m/s | "
                    f"Arm Support Bonus: {info['arm_support_bonus']:.2f}"
                )
                obs, _ = env.reset()
                was_pushed = False

            # Sleep for any remaining time in the 0.02 s step so motion runs at real-time speed
            elapsed = time.time() - step_start
            if env.dt - elapsed > 0:
                time.sleep(env.dt - elapsed)


# Standard Python entrypoint guard
if __name__ == "__main__":
    main()
