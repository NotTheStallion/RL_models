import random
import numpy as np

class AdvGridWorldMDP:
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
            self.start_cells = []
        elif self.goal_type == 'clean':
            self.start_cells = [(0, 0)]
        elif self.goal_type == 'recharge':
            self.start_cells = [(height-1, 0)]

        if self.goal_type == 'material':
            self.terminal_states = [(height-1, width-1)]
        elif self.goal_type == 'clean':
            self.terminal_states = [(0, width-1)]
        elif self.goal_type == 'recharge':
            self.terminal_states = []
        
        self.bad_states = random.sample(list(self.states - set(self.terminal_states) - set(self.start_cells)), k=(height * width) // 4)
        
        self.materials = []
        self.tmp_materials = []
        self.clean = []
        self.tmp_clean = []
        self.recharge = []
        self.tmp_recharge = []
        
        if self.goal_type == 'material':
            self.materials = random.sample(list(self.states - set(self.bad_states) - set(self.terminal_states) - set(self.start_cells)), k=2)
            self.tmp_materials = self.materials.copy()
        elif self.goal_type == 'clean':
            self.clean = list(self.states - set(self.bad_states) - set(self.terminal_states) - set(self.start_cells))
            self.tmp_clean = self.clean.copy()
        elif self.goal_type == 'recharge':
            self.recharge = random.sample(list(self.states - set(self.bad_states) - set(self.terminal_states) - set(self.start_cells)), k=2)
            self.tmp_recharge = self.recharge.copy()
        
        self.transition_probabilities = {(state, action, new_state): 0 for state in self.states for action in self.actions for new_state in self.states}
        self.rewards = {(state, action, new_state): -0.1 for state in self.states for action in self.actions for new_state in self.states}  # Default penalty for steps

        for state in self.states:
            if state in self.bad_states or state in self.terminal_states:
                continue
            for action in self.actions:
                new_state = (state[0] + action[0], state[1] + action[1])
                if new_state not in self.states or new_state in self.bad_states:
                    new_state = state
                self.transition_probabilities[(state, action, new_state)] = 1
                if new_state in self.terminal_states:
                    self.rewards[(state, action, new_state)] = 10  # Reward for achieving the goal

        self.initial_state = random.choice(list(self.states - set(self.bad_states) - set(self.terminal_states)))

    def print_board(self):
        cell_width = 3
        horizontal_border = "+" + ("-" * cell_width + "+") * self.width

        print(horizontal_border)
        for i in range(self.height):
            row = "|"
            for j in range(self.width):
                if (i, j) in self.terminal_states:
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
                else:
                    cell = ".".center(cell_width)
                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()

    def print_policy(self, policy: dict):
        cell_width = 3

        horizontal_border = "+" + ("-" * cell_width + "+") * self.width

        print(horizontal_border)
        for i in range(self.height):
            row = "|"
            for j in range(self.width):
                if (i, j) in self.terminal_states:
                    cell = "T".center(cell_width)
                elif (i, j) in self.bad_states:
                    cell = "X".center(cell_width)
                else:
                    action = policy[(i, j)]
                    # Use arrows to represent actions
                    if action == (1, 0):
                        cell = "↓".center(cell_width)
                    elif action == (-1, 0):
                        cell = "↑".center(cell_width)
                    elif action == (0, 1):
                        cell = "→".center(cell_width)
                    elif action == (0, -1):
                        cell = "←".center(cell_width)
                    else:
                        cell = " ".center(cell_width)  # Fallback for undefined actions
                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()


    def print_value_function(self, V):
        max_length = max(len(f"{V.get((i, j), 0):.2f}") for i in range(self.height) for j in range(self.width))

        cell_width = max_length + 2
        horizontal_border = "+" + ("-" * cell_width + "+") * self.width

        print(horizontal_border)
        for i in range(self.height):
            row = "|"
            for j in range(self.width):
                if (i, j) in self.terminal_states:
                    cell = "T".center(cell_width)
                elif (i, j) in self.bad_states:
                    cell = "X".center(cell_width)
                else:
                    cell = f"{V.get((i, j), 0):.2f}".center(cell_width)
                row += cell + "|"
            print(row)
            print(horizontal_border)
        print()
    

### Define and Test
if __name__ == "__main__":
    # Define environments H, E, G
    hangar = AdvGridWorldMDP(5, 5, 'material')
    warehouse = AdvGridWorldMDP(6, 6, 'clean')
    garage = AdvGridWorldMDP(4, 4, 'recharge')

    # Visualize the environments and policies
    print("Hangar:")
    hangar.print_board()

    print("Warehouse:")
    warehouse.print_board()

    print("Garage:")
    garage.print_board()
