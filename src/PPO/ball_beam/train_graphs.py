import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from env import BallBeamSystem

# Callback de captura média e erro mínimo por episódio
class PositionErrorCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.mean_errors = []
        self.min_errors = []
        self._current = []

    def _on_step(self) -> bool:
        obs = self.locals.get("new_obs")
        if obs is not None:

            pos = float(obs[0][0])
            setpoint = float(obs[0][2])
            self._current.append(abs(pos - setpoint))

        dones = self.locals.get("dones")
        if dones is not None and dones[0]:
            if self._current:
                self.mean_errors.append(float(np.mean(self._current)))
                self.min_errors.append(float(np.min(self._current)))
            self._current = []

        return True

# Env
env = BallBeamSystem()
env = Monitor(env)

check_env(env, warn=True)

error_callback = PositionErrorCallback()

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

model.learn(total_timesteps=500000, callback=error_callback)
model.save("model_ball_beam_PPO")
print("Modelo salvo em model_ball_beam_PPO.zip")

rewards = np.array(env.get_episode_rewards())
ep_lengths = np.array(env.get_episode_lengths())
timesteps_cum = np.cumsum(ep_lengths)
n_episodes = len(rewards)
episodes = np.arange(1, n_episodes + 1)

mean_errors = np.array(error_callback.mean_errors)
min_errors = np.array(error_callback.min_errors)
ep_err = np.arange(1, len(mean_errors) + 1)

window = 20

def moving_avg(data, w):
    return np.convolve(data, np.ones(w) / w, mode='valid')

# Img 1: Recompensa vs. Timesteps
ma = moving_avg(rewards, window)

plt.figure(figsize=(12, 4))
plt.plot(timesteps_cum, rewards, color='grey', linewidth=0.8, alpha=0.4, label='Episódios')
plt.plot(timesteps_cum[window - 1:], ma, color='black', linewidth=0.8, label='Média')
plt.xlabel('Timesteps')
plt.ylabel('Recompensa por episódio')
plt.title('PPO Ball and Beam: Recompensa vs. Timesteps')
plt.legend()
plt.tight_layout()
plt.savefig('1.png', dpi=300)
plt.show()
print("Gráfico 1 salvo: 1.png")

# Img 2: Recompensa vs. Episódio
ma_ep = moving_avg(rewards, window)

plt.figure(figsize=(12, 4))
plt.plot(episodes, rewards, color='grey', linewidth=0.8, alpha=0.4, label='Episódios')
plt.plot(episodes[window - 1:], ma_ep, color='black', linewidth=0.8, label='Média')
plt.xlabel('Episódio')
plt.ylabel('Recompensa por episódio')
plt.title('PPO Ball and Beam: Recompensa vs. Episódio')
plt.legend()
plt.tight_layout()
plt.savefig('2.png', dpi=300)
plt.show()
print("Gráfico 2 salvo: 2.png")
