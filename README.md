# RL_models

This repository contains implementations of three reinforcement learning (RL) algorithms: Monte Carlo, SARSA, and Q-learning. These algorithms were tested on a grid world environment with two different approaches:

1. **Neutral Position Without Penalty**: In this approach, the neutral positions in the grid world do not penalize the agent. The agent is free to explore the grid without any negative reinforcement from neutral positions.

2. **Neutral Position With Penalty**: In this approach, the neutral positions impose a penalty of -1. This penalty encourages the agent to find the shortest path to the goal, as lingering in neutral positions results in a cumulative negative reward.

The goal of these experiments is to compare the performance and behavior of the RL algorithms under different conditions and to observe how the penalty influences the agent's pathfinding strategy.

