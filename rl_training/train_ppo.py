import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from a1_env import UnitreeA1Env


def main():
    save_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(save_dir, exist_ok=True)

    # 4 parallel MuJoCo instances optimized for 4-core/8-thread Ryzen CPU
    n_envs = 4
    vec_env = make_vec_env(UnitreeA1Env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,  # Saves every 100,000 total steps across 4 envs
        save_path=save_dir,
        name_prefix="a1_ppo",
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cpu",
        verbose=1,
    )

    print("Starting PPO training on CPU (500,000 timesteps)...")
    model.learn(total_timesteps=500_000, callback=checkpoint_callback)

    final_path = os.path.join(save_dir, "a1_ppo_final")
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}.zip")
    vec_env.close()


if __name__ == "__main__":
    main()
