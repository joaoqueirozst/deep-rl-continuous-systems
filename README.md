# DRL for Continuous System Control

> Project developed for the implementation of machine learning techniques, utilizing *Deep Reinforcement Learning* (DRL) algorithms for the control of continuous systems.

The repository includes the development of computational implementations for analyzing the use of machine learning techniques in the control of continuous systems. In this work, agents were trained using the *Deep Q-Network* (DQN) and *Proximal Policy Optimization* (PPO) algorithms, with the aim of learning control policies for the *Ball and Beam* and *Cart-Pendulum* systems.

## Installation

Use Python 3.8 to match the reference training environment (Python 3.8.10). From the repository root:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. See [dependency versions and execution instructions](src/README.md#dependencies-and-installation) for details, including CPU and CUDA installation options.

# About Environments

These systems were implemented using the [Pygame](https://www.pygame.org/docs/) library. This choice enables greater control over the modeling, including the modification of physical parameters for each system, specific simulation conditions, and control strategies, thereby allowing for specific analyses and comparisons between the algorithms under investigation and the results already available in the literature for existing environments.

Below, both systems are implemented.

<p align="center">
  <img src="env.png" width="600"/>
</p>
