import os, time
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecMonitor

from callback import SaveOnBestTrainingRewardCallback

# import numpy as np
# np.bool8 = np.bool

def train(env_id, log_base_dir="logs", model_base_dir="models", model_name=None, total_timesteps=5e5):
    # Environment
    env = make_vec_env(env_id, n_envs=8)
    env = VecMonitor(env, log_base_dir)

    # Agent Model
    if model_name is None:    
        model_name = env_id + "_A2C"
    log_dir = os.path.join(log_base_dir, env_id)
    model = A2C(
        policy="MlpPolicy",
        env=env,
        # -------------------------------------------------
        learning_rate=7e-4,
        n_steps = 5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        rms_prop_eps=1e-5,
        use_rms_prop=True,
        use_sde=False,
        sde_sample_freq=-1,
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        normalize_advantage=False,
        stats_window_size=100,
        policy_kwargs=None,
        # -------------------------------------------------
        tensorboard_log = log_dir,
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
        tb_log_name="A2C",
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
        model_name = env_id + "_A2C"
    model_path = os.path.join(model_base_dir, model_name)
    model = A2C.load(model_path, env)

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
    env_id = "CartPole-v1"
    
    train(env_id)
    run(env_id)