from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from env import InvertedPendulum

# Create and wrap the environment with Monitor
env = InvertedPendulum()

check_env(env, warn=True)

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=1,
    target_update_interval=1000,
    exploration_fraction=0.5,
    exploration_final_eps=0.1,
)

model.learn(total_timesteps=500000)
model.save("model_pendulum_DQN")
print("Modelo salvo em model_pendulum_DQN.zip")
