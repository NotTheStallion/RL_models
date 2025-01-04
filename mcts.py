import numpy as np
import random
from math import sqrt, log
from simple.grid_world_env import GridWorldEnvSlow


class TreeNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.action = None
        self.children = {}
        self.visit_count = 0
        self.total_reward = 0.0

    def is_fully_expanded(self, action_space):
        return len(self.children) == len(action_space)

    def best_child(self, exploration_param=1.41):
        """
        Select the child with the highest UCB score.
        """
        return max(
            self.children.values(),
            key=lambda child: (child.total_reward / (child.visit_count + 1e-6)) +
                              exploration_param * sqrt(log(self.visit_count + 1) / (child.visit_count + 1e-6))
        )

    def add_child(self, action, child_state):
        """
        Add a new child node for a given action.
        """
        child_node = TreeNode(state=child_state, parent=self)
        self.children[action] = child_node
        self.action = action
        return child_node


class MCTSAgent:
    def __init__(self, env, num_simulations, exploration_param=1.41):
        self.env = env
        self.num_simulations = num_simulations
        self.exploration_param = exploration_param

    def search(self, initial_state):
        """
        Perform MCTS starting from the initial state.
        """
        root = TreeNode(state=initial_state)

        for _ in range(self.num_simulations):
            node = root

            # Selection
            while not node.is_fully_expanded(self.env.actions) and node.children:
                node = node.best_child(self.exploration_param)

            # Expansion
            if not node.is_fully_expanded(self.env.actions):
                untried_actions = [action for action in self.env.actions if action not in node.children]
                action = random.choice(untried_actions)
                next_state, reward, done, _ = self.env.step(action)
                node = node.add_child(action, next_state)

                if done:
                    node.total_reward += reward
                    continue

            # Simulation
            total_reward = self.simulate(node.state)

            # Backpropagation
            self.backpropagate(node, total_reward)

        self.env.reset_to_state(root.state)
        return root.best_child(exploration_param=0).action

    def simulate(self, state):
        """
        Perform a random rollout to a terminal state.
        """
        total_reward = 0
        current_state = state
        done = False

        while not done:
            action = random.choice(self.env.actions)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            current_state = next_state

        return total_reward

    def backpropagate(self, node, reward):
        """
        Update the tree nodes with the simulation result.
        """
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent


if __name__ == "__main__":
    random_seed = 2023
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Define environment
    height = 4
    width = 4
    number_of_holes = 4

    env = GridWorldEnvSlow(height, width, number_of_holes)
    initial_state = env.reset()
    env.render()

    # MCTS Agent
    mcts_agent = MCTSAgent(env=env, num_simulations=100)

    done = False
    current_state = initial_state

    while not done:
        # Perform MCTS
        optimal_action = mcts_agent.search(current_state)
        print(f"Optimal Action from MCTS: {optimal_action}")

        next_state, reward, done, _ = env.step(optimal_action)
        print(f"Reward: {reward}, done: {done}")
        env.render()

        current_state = next_state

