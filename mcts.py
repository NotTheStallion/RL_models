import numpy as np
import random
from collections import defaultdict
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

class MCTS:
    def __init__(self, env, n_iterations=1000, exploration_weight=1.4):
        self.env = env
        self.n_iterations = n_iterations
        self.exploration_weight = exploration_weight
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.children = dict()

    def choose(self, state):
        if state not in self.children:
            return random.choice(self.env.get_possible_actions(state))
        
        def score(action):
            if (state, action) not in self.Q:
                return float('-inf')
            return self.Q[(state, action)] / self.N[(state, action)]
        
        return max(self.children[state], key=score)

    def do_rollout(self, state):
        path = self._select(state)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, state):
        path = []
        while True:
            path.append(state)
            if state not in self.children or not self.children[state]:
                return path
            unexplored = self.children[state] - self.N.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            state = self._uct_select(state)

    def _expand(self, state):
        if state in self.children:
            return
        self.children[state] = self.env.get_possible_actions(state)

    def _simulate(self, state):
        env_copy = self.env.clone()
        env_copy.set_state(state)
        while not env_copy.is_terminal():
            action = random.choice(env_copy.get_possible_actions(state))
            state, reward, done, _ = env_copy.step(action)
            if done:
                return reward
        return 0

    def _backpropagate(self, path, reward):
        for state in reversed(path):
            if state in self.N:
                self.N[state] += 1
                self.Q[state] += reward
            else:
                self.N[state] = 1
                self.Q[state] = reward

    def _uct_select(self, state):
        log_N_vertex = np.log(self.N[state])

        def uct(action):
            return self.Q[(state, action)] / self.N[(state, action)] + self.exploration_weight * np.sqrt(
                log_N_vertex / self.N[(state, action)])

        return max(self.children[state], key=uct)





if __name__ == "__main__":
    env = GridWorldEnv()
    mcts = MCTS(env)

    initial_state = env.reset()
    for _ in range(mcts.n_iterations):
        mcts.do_rollout(initial_state)

    best_action = mcts.choose(initial_state)
    print(f"Best action from initial state: {best_action}")