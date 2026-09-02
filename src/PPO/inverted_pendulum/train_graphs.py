import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from env import InvertedPendulum

last_n_steps = 50

# Callback de captura média e erro mínimo por episódio
class PositionErrorCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_indices = [] # episódio global em que houve truncated
        self.eq_errors = [] # erro médio nos últimos N steps
        self._current = [] # acumula erro normalizado do episódio
        self._ep_count = 0 # contador global de episódios
 
    def _on_step(self) -> bool:
        obs = self.locals.get("new_obs")
        if obs is not None:

            pos = float(obs[0][0])
            setpoint = float(obs[0][2])
            self._current.append(abs(pos - setpoint))
 
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if dones is not None and dones[0]:
            self._ep_count += 1
            truncated = infos[0].get("TimeLimit.truncated", False) if infos else False

            if truncated and len(self._current) >= last_n_steps:
                last_errors = self._current[-last_n_steps:]
                self.eq_errors.append(float(np.mean(last_errors)))
                self.ep_indices.append(self._ep_count)
 
            self._current = []
 
        return True

# Env
env = InvertedPendulum()
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
model.save("model_pendulum_PPO")
print("Modelo salvo em model_pendulum_PPO.zip")

rewards = np.array(env.get_episode_rewards())
ep_lengths = np.array(env.get_episode_lengths())
timesteps_cum = np.cumsum(ep_lengths)
n_episodes = len(rewards)
episodes = np.arange(1, n_episodes + 1)

eq_errors  = np.array(error_callback.eq_errors)
ep_indices = np.array(error_callback.ep_indices)

window = 20

def moving_avg(data, w):
    return np.convolve(data, np.ones(w)/w, mode='valid')

# Img 1: Recompensa vs. Timesteps
ma = moving_avg(rewards, window)

plt.figure(figsize=(12, 4))
plt.plot(timesteps_cum, rewards, color='grey', linewidth=0.8, alpha=0.4, label='Episódios')
plt.plot(timesteps_cum[window - 1:], ma, color='black', linewidth=0.8, label='Média')
plt.xlabel('Timesteps')
plt.ylabel('Recompensa por episódio')
plt.title('PPO Inverted Pendulum: Recompensa vs. Timesteps')
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
plt.title('PPO Inverted Pendulum: Recompensa vs. Episódio')
plt.legend()
plt.tight_layout()
plt.savefig('2.png', dpi=300)
plt.show()
print("Gráfico 2 salvo: 2.png")

# # Img 3: Erro de posição vs. Episódio (média + mínimo)
# if len(eq_errors) >= window:
#     ma_eq = moving_avg(eq_errors, window)
#     plt.plot(ep_indices, eq_errors, color='black', linewidth=0.8, alpha=0.4, label='Erro de equilíbrio')
#     plt.plot(ep_indices[window - 1:], ma_eq, color='black', linewidth=2, label='Média')

# else:
#     # Plota sem média móvel
#     plt.plot(ep_indices, eq_errors, color='black', linewidth=1.0, label='Erro de equilíbrio')
 
# plt.axhline(y=0.05, color='grey', linewidth=1.0, linestyle='--', label='Tolerância (0.05 m)')
# plt.yscale('log')
# plt.xlabel('Episódio')
# plt.ylabel('Erro médio nos últimos 50 steps (normalizado)') # Indica estabilização real do sistema
# plt.title('PPO Inverted Pendulum: Erro de equilíbrio vs. Episódio')
# plt.legend()
# plt.tight_layout()
# plt.savefig('3.png', dpi=300)
# plt.show()
# print("Gráfico 3 salvo: 3.png")
