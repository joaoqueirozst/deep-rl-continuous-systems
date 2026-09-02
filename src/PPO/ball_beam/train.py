import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from env import BallBeamSystem

# Create and wrap the environment with Monitor
env = BallBeamSystem()

check_env(env, warn=True)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
)

model.learn(total_timesteps=500000)
model.save("model_ball_beam_PPO")
print("Modelo salvo em model_ball_beam_PPO.zip")
