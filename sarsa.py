import numpy as np
from grid_world_env import GridWorldEnv

class GridWorldEnv(GridWorldMDP):
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4)):
        super().__init__(*grid_size, 1, start, goal)
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.state = None
        self.observation_space = np.prod(self.grid_size)
        self.action_space = 4  # up, down, left, right
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.grid_size[0] - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.grid_size[1] - 1, y + 1)
        self.state = (x, y)
        reward = -1
        done = self.state == self.goal
        if done:
            reward = 0
        return self.state, reward, done

    def render(self):
        grid = np.zeros(self.grid_size)
        x, y = self.state
        grid[x, y] = 1
        gx, gy = self.goal
        grid[gx, gy] = 2
        print(grid)

class SARSAAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((state_space, action_space))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action):
        predict = self.q_table[state, action]
        target = reward + self.gamma * self.q_table[next_state, next_action]
        self.q_table[state, action] += self.alpha * (target - predict)

def train_sarsa(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action)
            state, action = next_state, next_action

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')
    agent = SARSAAgent(state_space=env.observation_space.n, action_space=env.action_space.n)
    train_sarsa(env, agent)