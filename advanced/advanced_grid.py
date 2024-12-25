import random
import numpy as np

class GridWorldMDP:
    def __init__(self, height: int, width: int, goal_type: str):
        self.height = height
        self.width = width
        self.goal_type = goal_type  # 'material', 'clean', or 'recharge'

        self.states = set((i, j) for i in range(height) for j in range(width))

        self.UP = (-1, 0)
        self.DOWN = (1, 0)
        self.LEFT = (0, -1)
        self.RIGHT = (0, 1)

        self.actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT]

        if self.goal_type == 'material':
            self.goal_cells = [(height-1, width-1)]
        elif self.goal_type == 'clean':
            self.goal_cells = [(0, width-1)]
        elif self.goal_type == 'recharge':
            self.goal_cells = [(height-1, 0)]
        
        self.obstacles = random.sample(list(self.states - set(self.goal_cells)), k=(height * width) // 4)

        self.transition_probabilities = {(state, action, new_state): 0 for state in self.states for action in self.actions for new_state in self.states}
        self.rewards = {(state, action, new_state): -0.1 for state in self.states for action in self.actions for new_state in self.states}  # Default penalty for steps

        for state in self.states:
            if state in self.obstacles or state in self.goal_cells:
                continue
            for action in self.actions:
                new_state = (state[0] + action[0], state[1] + action[1])
                if new_state not in self.states or new_state in self.obstacles:
                    new_state = state
                self.transition_probabilities[(state, action, new_state)] = 1
                if new_state in self.goal_cells:
                    self.rewards[(state, action, new_state)] = 10  # Reward for achieving the goal

        self.initial_state = random.choice(list(self.states - set(self.obstacles) - set(self.goal_cells)))

    def print_board(self):
        cell_width = 3
        horizontal_border = "+" + ("-" * cell_width + "+") * self.width

        print(horizontal_border)
        for i in range(self.height):
            row = "|"
            for j in range(self.width):
                if (i, j) in self.goal_cells:
                    if self.goal_type == 'material':
                        cell = "H".center(cell_width)
                    elif self.goal_type == 'clean':
                        cell = "E".center(cell_width)
                    elif self.goal_type == 'recharge':
                        cell = "G".center(cell_width)
                elif (i, j) in self.obstacles:
                    cell = "X".center(cell_width)
                else:
                    cell = ".".center(cell_width)
                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()

    def print_policy(self, policy):
        # Visualize the policy on the board
        pass  # Same as in the provided script

    def print_value_function(self, V):
        # Visualize value function
        pass  # Same as in the provided script
    

### Define and Test
if __name__ == "__main__":
    # Define environments H, E, G
    hangar = GridWorldMDP(5, 5, 'material')
    warehouse = GridWorldMDP(6, 6, 'clean')
    garage = GridWorldMDP(4, 4, 'recharge')

    # Visualize the environments and policies
    print("Hangar:")
    hangar.print_board()

    print("Warehouse:")
    warehouse.print_board()

    print("Garage:")
    garage.print_board()
