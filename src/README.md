# System Control Based on DQN and PPO

Overall, the study conducted in this project was based on the implementation of machine learning techniques — specifically *Deep Reinforcement Learning* (DRL) algorithms — in continuous systems.

## Technologies Used
- `Stable-baselines3` ([SB3](https://stable-baselines3.readthedocs.io/en/master/)), which provides standardized implementations of reinforcement learning algorithms;
- DRL algorithms based on DQN and PPO;
- [Pygame](https://www.pygame.org/docs/), for modeling the systems.

## Development

The main objective of the project was to implement the *Ball and Beam* and *Cart-Pendulum* systems and to use the DRL algorithms DQN and PPO to learn the dynamics of each.

The choice of these two distinct methods allows for comparing the behavior of an algorithm based on discretized actions (DQN) with another capable of operating directly in a continuous space (PPO).

Furthermore, properly structuring the **Reward Function** is essential, as it is a key element of reinforcement learning that establishes a quantitative criterion to guide the agent toward the desired behavior. In this project, the functions were structured according to the dynamic characteristics of each system, taking into account variables such as the *error relative to the reference*, *velocities*, *angular variables*, and *control effort*.

In light of the observations made above, it is possible to proceed to importing the libraries.

```bash
pip install pygame stable-baselines3 gymnasium
```

To train the agents, simply run the `train.py` code (or, if you want to analyze the training using reward graphs, `train_graphs.py`).

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
