import os  # Import os to set thread limits and manage file paths
import glob  # Import glob to locate existing checkpoints for resume

# Cap OpenMP/MKL threads to 1 so parallel workers never deadlock on WSL 2
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch  # Import PyTorch to enforce single-threaded CPU execution per worker
torch.set_num_threads(1)

from stable_baselines3 import PPO  # Import PPO reinforcement learning algorithm
from stable_baselines3.common.env_util import make_vec_env  # Vectorized env builder
from stable_baselines3.common.vec_env import SubprocVecEnv  # Multi-process env wrapper
from stable_baselines3.common.callbacks import CheckpointCallback  # Periodic model saver
from farm_dog6_balance_env import FarmDog6ArmBalanceEnv  # Import 23-DOF balance environment


def get_latest_checkpoint(save_dir: str):
    """Return path to the highest-step balance checkpoint if present, else None."""
    checkpoints = glob.glob(os.path.join(save_dir, "farmdog6_balance_*_steps.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.split("_")[-2]))


def main():
    # Directory for saving periodic and final arm-assisted balance policies
    save_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(save_dir, exist_ok=True)

    # Spawn 4 parallel MuJoCo environments using 'forkserver' for WSL 2 stability
    n_envs = 4
    vec_env = make_vec_env(
        FarmDog6ArmBalanceEnv,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "forkserver"},
    )

    # Save a checkpoint every 25,000 steps per worker (100,000 total steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=save_dir,
        name_prefix="farmdog6_balance",
    )

    # Resume automatically if an earlier checkpoint exists
    latest_ckpt = get_latest_checkpoint(save_dir)
    if latest_ckpt:
        print(f"Resuming 23-DOF arm-balance PPO training from: {latest_ckpt}")
        model = PPO.load(latest_ckpt, env=vec_env, device="cpu")
        reset_timesteps = False
    else:
        print("Starting fresh 23-DOF arm-balance PPO training (1,200,000 timesteps)...")
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        model = PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=512,        # Log rollout stats every 2,048 total environment steps
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.008,
            device="cpu",
            verbose=1,
        )
        reset_timesteps = True

    # Train the whole-body 23-DOF policy across uneven ridges and lateral pushes
    model.learn(
        total_timesteps=1_200_000,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_timesteps,
    )

    # Save final trained policy archive
    final_path = os.path.join(save_dir, "farmdog6_balance_final")
    model.save(final_path)
    print(f"Training complete! Policy saved to {final_path}.zip")
    vec_env.close()


if __name__ == "__main__":
    main()
