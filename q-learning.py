import numpy as np
import random
from grid_world_env import GridWorldEnv

class QLearningAgent:
    def __init__(self, env: GridWorldEnv, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        super().__init__()
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros((env.grid_size[0], env.grid_size[1], env.action_space))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.env.action_space - 1)  # Explore action space
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploit learned values

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        td_target = reward + self.discount_factor * self.q_table[next_state[0], next_state[1], best_next_action]
        td_error = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.learning_rate * td_error

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def play(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = np.argmax(self.q_table[state[0], state[1]])
                state, reward, done = self.env.step(action)
                self.env.render()

if __name__ == "__main__":
    env = GridWorldEnv()
    agent = QLearningAgent(env)

    # Train the agent
    agent.train(episodes=1000)

    # Test the agent
    agent.play(episodes=10)