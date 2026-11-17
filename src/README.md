# System Control Based on DQN and PPO

Overall, the study conducted in this project was based on the implementation of machine learning techniques, specifically *Deep Reinforcement Learning* (DRL) algorithms, in continuous systems.

## Technologies Used
- `Stable-baselines3` ([SB3](https://stable-baselines3.readthedocs.io/en/master/)), which provides standardized implementations of reinforcement learning algorithms;
- DRL algorithms based on DQN and PPO;
- [Pygame](https://www.pygame.org/docs/), for modeling the systems.

## Development

The main objective of the project was to implement the *Ball and Beam* and *Cart-Pendulum* systems and to use the DRL algorithms DQN and PPO to learn the dynamics of each.

The choice of these two distinct methods allows for comparing the behavior of an algorithm based on discretized actions (DQN) with another capable of operating directly in a continuous space (PPO).

Furthermore, properly structuring the **Reward Function** is essential, as it is a key element of reinforcement learning that establishes a quantitative criterion to guide the agent toward the desired behavior. In this project, the functions were structured according to the dynamic characteristics of each system, taking into account variables such as the *error relative to the reference*, *velocities*, *angular variables*, and *control effort*.

### Dependencies and installation

The reference training environment used **Python 3.8.10**. The versions below are pinned in the root [`requirements.txt`](../requirements.txt), shared by all DQN and PPO environments:

| Library | Version | Purpose |
| --- | --- | --- |
| Stable-Baselines3 | 2.4.1 | DQN and PPO implementations |
| PyTorch | 2.4.1 | Neural networks and CPU/GPU execution |
| NumPy | 1.24.4 | Numerical calculations and environment state |
| Gymnasium | 1.0.0 | Environment API and action/observation spaces |
| Cloudpickle | 3.1.2 | Serialization used by Stable-Baselines3 |
| Pygame | 2.6.1 | Simulation rendering |
| Matplotlib | 3.7.5 | Training reward plots |

Stable-Baselines3, PyTorch, NumPy, Gymnasium, and Cloudpickle versions were recovered from `system_info.txt` inside all four pre-trained model ZIP files. The original PyTorch build was `2.4.1+cu121` (CUDA 12.1). Pygame and Matplotlib versions were not recorded there; their pins are compatible selections for this dependency configuration, rather than recovered training versions. Other transitive dependencies are resolved by pip and are not fully locked by this file.

Use Python 3.8 to match the reference environment. These pins are not intended for Python 3.12 or newer. From the **repository root**, create and activate a virtual environment, then install the dependencies:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `py -3.8 -m venv .venv` and activate it with `.venv\Scripts\activate`.

For CPU-only execution on Linux or Windows, install the CPU build of PyTorch **before** installing `requirements.txt`:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

To use the original CUDA 12.1 build on Linux or Windows with a compatible NVIDIA GPU and driver, use the same order:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

These build options follow the [official PyTorch 2.4.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v241). The `torch==2.4.1` requirement accepts both builds.

### Training and evaluation

Run commands from the selected algorithm/environment directory so that imports and model paths resolve correctly. For example:

```bash
cd src/PPO/inverted_pendulum
```

You can also select `src/PPO/ball_beam`, `src/DQN/inverted_pendulum`, or `src/DQN/ball_beam`. To train the agents, run `train.py` (or `train_graphs.py` to analyze training using reward graphs):

```bash
python3 train.py
```

After training, the code exports a `.ZIP` file containing the model's learned policy, however, if you prefer a pre-trained model, this repository provides the trained agent file for each **environment**.
> It is worth noting that all training sessions were conducted with GPU acceleration, using the infrastructure made available during the research at [LabSEA](https://github.com/Lab-SEA).

Finally, to test the trained model and validate the control system, run:
```bash
python3 test.py
```

## Final Remarks
The work demonstrated the ability of DRL algorithms to learn control policies for the *Ball and Beam* and *Cart-Pendulum* systems. Both systems exhibited satisfactory performance in simulations, and the results highlighted the importance of a properly structured reward function in guiding the agents' learning.

Future work could involve extending this application to other dynamic systems and investigating new ways to structure reward functions within the environment.

## Supplementary Work
Another step taken during the research was applying the adopted methodologies to a different simulation environment: **Gazebo**.

For more information on what was done and how, visit: [Simulation in Gazebo](https://github.com/joaoqueirozst/ball-and-beam-system-gz).
