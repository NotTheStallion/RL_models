import numpy as np
from grid_world import GridWorldMDP

class GridWorldEnv():
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4)):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.state = None
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(self.grid_size[0] - 1, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
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

    def play(self, policy, render=False):
        self.reset()
        done = False
        total_reward = 0
        while not done:
            action = policy(self.state)
            state, reward, done = self.step(action)
            total_reward += reward
            if render:
                self.render()
        return total_reward

# Example usage
if __name__ == "__main__":
    env = GridWorldEnv()
    
    def random_policy(state):
        return np.random.choice(['up', 'down', 'left', 'right'])
    
    total_reward = env.play(random_policy)
    print(f"Total reward: {total_reward}")