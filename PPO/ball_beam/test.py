from stable_baselines3 import PPO
from env import BallBeamSystem

model = PPO.load("model_ball_beam_PPO")

env = BallBeamSystem(render_mode="human")
obs, info = env.reset()

while not env.finish:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated:
        obs, info = env.reset()

env.close()
