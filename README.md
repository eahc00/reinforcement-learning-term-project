# reinforcement-learning-term-project

### phase1
InvertedPendulum-v5 환경에 대하여 stable baselines3 zoo hyper parameter 적용 실험.

- Problem : stable baselines3에 v5 버전 파라미터가 명시 되어 있지 않음. 
- v2 파라미터로 진행했지만 제대로 학습되지 않음.
- hyperparameter 튜닝을 진행하고 비교함.
    - VecNormalize 적용 차이 : v2에 있는 VecNormalize를 빼니까 성능이 훨씬 개선됨.
    - learning rate : 학습률을 낮추니 훨씬 안정적으로 학습. Normalize 유무의 영향이 있다고 추측됨.
    - batch size : 마찬가지로 Normalize를 빼고 batch size를 낮추니 안정적으로 학습됨.

- InvertedDoublePendulum-v5에 대해서도 같은 방식으로 가능.


### phase2
이산적 알고리즘 별 cartpole 비교
- A2C
- PPO

- PPO가 수렴이 더 빠르고 더 높은 reward를 가진다.


### Phase3
연속적 알고리즘 별 inverted double pendulum
- InvertedPendulum과 InvertedDoublePendulum에 대한 하이퍼 파라미터 정의가 없음.
- SAC
	- InvertedPendulum-v5 : batch_size=128
- TD3
	- n_envs = 4, total_timesteps를 각 2e+5, 3e+5 학습 -> 안정적으로 학습 가능.
