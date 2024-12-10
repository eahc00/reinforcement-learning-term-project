import os, time
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecMonitor

from callback import SaveOnBestTrainingRewardCallback

# import numpy as np
# np.bool8 = np.bool

def train(env_id, log_base_dir="logs", model_base_dir="models", model_name=None, total_timesteps=100000):   
    # Environment
    env = make_vec_env(env_id, n_envs=8)
    env = VecMonitor(env, log_base_dir)

    # Agent Model
    if model_name is None:    
        model_name = env_id + "_PPO"
    log_dir = os.path.join(log_base_dir, env_id)
    model = SAC(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=None,
        gamma=0.99,
        # -------------------------------------------------
        learning_rate=0.001,
        batch_size=256,
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
        tb_log_name="SAC",
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
    model = SAC.load(model_path, env)

    # Run    
    for episode in range(n_episodes):
        obs, info = env.reset()

        episode_reward = 0
        while True:
            action, next_state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.01)
            episode_reward += reward
            if terminated or truncated:
                time.sleep(1.0)
                print("n_steps: {}".format(episode_reward))
                episode_reward = 0
                break
            
    # close the environment
    env.close()


if __name__ == "__main__":
    # env_id = "InvertedPendulum-v5"
    # env_id = "CartPole-v1"
    env_id = "InvertedDoublePendulum-v5"
    # env_id = "Pendulum-v1"
    
    train(env_id)
    run(env_id)