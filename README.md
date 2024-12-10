# reinforcement-learning-term-project

### phase1
InvertedPendulum-v5 환경에 대하여 stable baselines3 zoo hyper parameter 적용 실험.

- Problem : stable baselines3에 v5 버전 파라미터가 명시 되어 있지 않음. 
- v2 파라미터로 진행했지만 제대로 학습되지 않음.
- hyperparameter 튜닝을 진행하고 비교함.
    - VecNormalize 적용 차이 : v2에 있는 VecNormalize를 빼니까 성능이 훨씬 개선됨.
    - learning rate : 학습률을 낮추니 훨씬 안정적으로 학습. Normalize 유무의 영향이 있다고 추측됨.
    - batch size : 마찬가지로 Normalize를 빼고 batch size를 낮추니 안정적으로 학습됨.