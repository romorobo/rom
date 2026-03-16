import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from sim_env import G1LabEnv

def make_env(xml_path, rank, seed=0):
    """
    Utility function for multiprocess env.
    """
    def _init():
        env = G1LabEnv(xml_path, render_mode=None) # Headless mode for cluster
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Train G1 Reach and Grasp Policy on Nebius Compute")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments to run")
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="Total timesteps for training")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for Tensorboard logs")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    xml_path = os.path.join("models", "unitree_mujoco-main", "unitree_robots", "g1", "g1_lab_scene.xml")

    print(f"Initializing {args.num_envs} headless environments...")
    
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(xml_path, i) for i in range(args.num_envs)])
    env = VecMonitor(env)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // args.num_envs, 1), # Save periodically
        save_path=args.checkpoints_dir,
        name_prefix="g1_reach_grasp_ppo"
    )

    print("Initializing PPO model...")
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
    )

    print(f"Starting training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    print("Training complete! Saving final model...")
    model.save(os.path.join(args.checkpoints_dir, "g1_reach_grasp_ppo_final"))
    env.close()

if __name__ == "__main__":
    main()
