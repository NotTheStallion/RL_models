import random
from grid_world import GridWorldMDP

class GridWorldEnv(GridWorldMDP):
    def __init__(self, height: int, width: int, number_of_holes: int):
        super().__init__(height, width, number_of_holes)
        self.current_state = self.initial_state

    def reset(self):
        """
        Resets the environment to the initial state.
        Returns:
            The initial state.
        """
        self.current_state = self.initial_state
        return self.current_state
    
    def reset_to_state(self, state):
        """
        Resets the environment to a specific state.
        Args:
            state: The state to reset to.
        """
        if state in self.states:
            self.current_state = state
        else:
            raise ValueError(f"Invalid state: {state}")

    def step(self, action):
        """
        Executes the given action and updates the environment.
        Args:
            action: A tuple representing the action (dx, dy).
        Returns:
            A tuple (new_state, reward, done, info), where:
            - new_state: The new state after taking the action.
            - reward: The reward obtained after taking the action.
            - done: A boolean indicating if the episode has ended.
            - info: Additional information (empty in this case).
        """
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")

        new_state = (self.current_state[0] + action[0], self.current_state[1] + action[1])
        if new_state not in self.states:
            new_state = self.current_state  # Invalid action, stay in the current state
        
        if new_state in self.bad_states:
            reward = -1
            done = True
        elif new_state == self.terminal_state:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        self.current_state = new_state
        return new_state, reward, done, {}

    def render(self):
        """
        Renders the current state of the environment.
        """
        cell_width = 3
        horizontal_border = "+" + ("-" * cell_width + "+") * self.width

        print(horizontal_border)
        for i in range(self.height):
            row = "|"
            for j in range(self.width):
                if (i, j) == self.current_state:
                    cell = "A".center(cell_width)  # Agent's position
                elif (i, j) == self.terminal_state:
                    cell = "T".center(cell_width)
                elif (i, j) in self.bad_states:
                    cell = "X".center(cell_width)
                else:
                    cell = ".".center(cell_width)
                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()


if __name__ == "__main__":
    env = GridWorldEnv(4, 4, 4)
    env.render()
    
    done = False
    total_reward = 0
    env.reset()
    
    while not done:
        action = random.choice(env.actions)  # Take a random action
        new_state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    
    print(f"Episode finished with total reward: {total_reward}")
