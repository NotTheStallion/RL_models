import random
from advanced.advanced_grid import AdvGridWorldMDP
from abc import ABC, abstractmethod

class AdvGridWorldEnv(AdvGridWorldMDP):
    def __init__(self, height: int, width: int, goal_type: str):
        super().__init__(height, width, goal_type)
        self.current_state = self.initial_state

    def reset(self):
        """
        Resets the environment to the initial state.
        Returns:
            The initial state.
        """
        if self.goal_type == 'material':
            self.current_state = random.choice(list(self.states - set(self.bad_states) - set(self.terminal_states)))
        elif self.goal_type == 'clean':
            self.current_state = (0, 0)
        elif self.goal_type == 'recharge':
            self.current_state = (self.height-1, 0)
            
        # self.current_state = random.choice(list(self.states - set(self.bad_states) - set(self.terminal_states)))
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

    @abstractmethod
    def step(self, action):
        """
        Takes a step in the environment.
        Args:
            action: The action to take.
        Returns:
            A tuple of (next_state, reward, done, info).
        """
        pass

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
                    cell = "A".center(cell_width)
                elif (i, j) in self.terminal_states:
                    if self.goal_type == 'material':
                        cell = "H".center(cell_width)
                    elif self.goal_type == 'clean':
                        cell = "E".center(cell_width)
                    elif self.goal_type == 'recharge':
                        cell = "G".center(cell_width)
                elif (i, j) in self.bad_states:
                    cell = "X".center(cell_width)
                elif (i, j) in self.start_cells:
                    cell = "o".center(cell_width)
                elif (i, j) in self.materials:
                    cell = "⚙".center(cell_width)
                elif (i, j) in self.clean:
                    cell = "🗑".center(cell_width)
                elif (i, j) in self.recharge:
                    cell = "⌁".center(cell_width)
                elif (i, j) in self.bad_states:
                    cell = "X".center(cell_width)
                else:
                    cell = ".".center(cell_width)
                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()


class AdvGridWorldEnvSlow(AdvGridWorldEnv):
    def step(self, action):
        """
        Executes the given action and updates the environment.
        # @note : this environment ensures that the agent finds the the terminal state but no penality if it stays in a neutral box
        """
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")

        new_state = (self.current_state[0] + action[0], self.current_state[1] + action[1])
        if new_state not in self.states:
            new_state = self.current_state
        
        if new_state in self.bad_states:
            reward = -100
            done = True
        elif new_state in self.terminal_states:
            reward = 5
            done = True
        elif new_state in self.tmp_materials:
            reward = 10
            self.tmp_materials.remove(new_state)
            done = False
        elif new_state in self.tmp_clean:
            reward = 30
            self.tmp_clean.remove(new_state)
            done = False
        elif new_state in self.recharge:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        self.current_state = new_state
        return new_state, reward, done, {}


class AdvGridWorldEnvFast(AdvGridWorldEnv):
    def step(self, action):
        """
        Executes the given action and updates the environment.
        # @note : this environment ensures that the agent finds the quickest route to the terminal state
        """
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")

        new_state = (self.current_state[0] + action[0], self.current_state[1] + action[1])
        if new_state not in self.states:
            new_state = self.current_state  # Invalid action, stay in the current state
        
        if new_state in self.bad_states:
            reward = -10
            done = True
        elif new_state == self.terminal_state:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        self.current_state = new_state
        return new_state, reward, done, {}


if __name__ == "__main__":
    # Define environments H, E, G
    hangar = AdvGridWorldEnvSlow(5, 5, 'material')
    warehouse = AdvGridWorldEnvSlow(6, 6, 'clean')
    garage = AdvGridWorldEnvSlow(4, 4, 'recharge')
    hangar.render()
    
    done = False
    total_reward = 0
    hangar.reset()
    
    while not done:
        action = random.choice(hangar.actions)  # Take a random action
        new_state, reward, done, info = hangar.step(action)
        total_reward += reward
        hangar.render()
    
    print(f"Episode finished with total reward: {total_reward}")
