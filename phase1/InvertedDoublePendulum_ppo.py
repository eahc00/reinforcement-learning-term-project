import os, time
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from callback import SaveOnBestTrainingRewardCallback

import numpy as np
# np.bool8 = np.bool

def train(
        env_id,
        log_base_dir="logs",
        model_base_dir="models",
        model_name=None,
        total_timesteps=2e5
):
    # Environment
    env = make_vec_env(env_id, n_envs=1)
    env = VecMonitor(env, log_base_dir)

    # Agent Model
    if model_name is None:    
        model_name = env_id + "_PPO"
    log_dir = os.path.join(log_base_dir, env_id)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        # -------------------------------------------------
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=20,
        gamma=0.98,
        gae_lambda=0.8,
        clip_range=0.2,
        ent_coef=1e-6,
        vf_coef=0.4,
        max_grad_norm=0.5,
        # -------------------------------------------------
        tensorboard_log=log_dir,
        verbose=1,
        seed=None,
        device='auto',
    )            

    # Train
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_base_dir)   
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=4,
        tb_log_name="PPO",
        reset_num_timesteps=True,
        progress_bar=True,
    )

    # Save the trained model
    save_path = os.path.join(os.getcwd(), model_base_dir, model_name)
    model.save(save_path)

    # close the environment
    env.close()


def run(env_id, model_base_dir="models", model_name=None, n_episodes=5):
    # Environment
    env = gym.make(env_id, render_mode='human')
    
    # Model
    if model_name is None:
        model_name = env_id + "_PPO"
    model_path = os.path.join(model_base_dir, model_name)
    model = PPO.load(model_path, env)

    # Run    
    for episode in range(n_episodes):
        obs, info = env.reset()

        episode_reward = 0
        n_steps = 0
        while True:
            action, next_state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.01)
            episode_reward += reward
            n_steps += 1
            if terminated or truncated:
                time.sleep(1.0)
                print("n_steps: {}".format(n_steps))
                print("episode_reward: {}".format(episode_reward))
                episode_reward = 0
                n_steps = 0
                break
            
    # close the environment
    env.close()


if __name__ == "__main__":
    # env_id = "CartPole-v1"
    # env_id = "Pendulum-v1"
    env_id = "InvertedDoublePendulum-v5"
    
    train(env_id)
    run(env_id)