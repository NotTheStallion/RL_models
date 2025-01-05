import numpy as np
import math
import random
from simple.grid_world_env import GridWorldEnv

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self, action_space):
        return len(self.children) == len(action_space)

    def best_child(self, exploration_weight=1.0)->'Node':
        weights = [
            (child.value / (child.visits + 1e-6)) + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            for child in self.children
        ]
        # print(weights)
        print([child.value for child in self.children])
        return self.children[np.argmax(weights)]

class MCTSAgent:
    def __init__(self, env:GridWorldEnv, iterations=5, exploration_weight=1):
        self.env:GridWorldEnv = env
        self.iterations = iterations
        self.exploration_weight = exploration_weight

    def select(self, node:Node, action_space):
        while node.state != self.env.terminal_state:
            if not node.is_fully_expanded(action_space):
                return self.expand(node, action_space)
            else:
                node = node.best_child(self.exploration_weight)
        return node

    def expand(self, node:Node, action_space):
        tried_actions = [child.action for child in node.children]
        for action in action_space:
            if action not in tried_actions:
                self.env.reset_to_state(node.state)
                next_state, _, _, _ = self.env.step(action)
                child_node = Node(state=next_state, parent=node, action=action)
                node.children.append(child_node)
                return child_node

    def simulate(self, node:Node):
        current_state = node.state
        total_reward = 0
        depth = 0

        while current_state != self.env.terminal_state and depth < 50:
            action = random.choice(self.env.actions)
            # self.env.reset_to_state(current_state)
            current_state, reward, _, _ = self.env.step(action)
            total_reward = reward
            depth += 1

        self.env.reset_to_state(node.state)
        print(f"Simulated reward: {total_reward}")
        return total_reward

    def backpropagate(self, node:Node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def search(self, root_state):
        root = Node(state=root_state)

        for _ in range(self.iterations):
            node = self.select(root, self.env.actions)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
            self.env.reset_to_state(root_state)

        self.env.reset_to_state(root_state)
        return root.best_child(exploration_weight=0).action

if __name__ == "__main__":
    from simple.grid_world_env import GridWorldEnvSlow

    random_seed = 2020
    np.random.seed(random_seed)
    random.seed(random_seed)

    height = 4
    width = 4
    number_of_holes = 4

    env = GridWorldEnvSlow(height, width, number_of_holes)

    agent = MCTSAgent(env, iterations=100, exploration_weight=1.4)

    num_episodes = 10

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # env.render()
        while not done:
            action = agent.search(state)
            state, reward, done, _ = env.step(action)
            # env.render()
            total_reward += reward
            steps += 1

        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}, Steps: {steps}")
