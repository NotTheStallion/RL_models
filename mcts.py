import math
import random
from grid_world_env import GridWorldEnv

class MCTSNode:
    def __init__(self, state, parent=None):
        """
        Represents a node in the Monte Carlo Tree.
        Args:
            state: The environment state this node represents.
            parent: The parent node of this node.
        """
        self.state = state  # The state represented by this node
        self.parent = parent  # Parent node
        self.children = {}  # Dictionary mapping actions to child nodes
        self.visits = 0  # Number of times this node was visited
        self.total_reward = 0  # Total reward accumulated

    def is_fully_expanded(self, action_space):
        """
        Checks if all possible actions have been expanded.
        Args:
            action_space: List of all possible actions.
        Returns:
            True if all actions have been expanded, False otherwise.
        """
        return len(self.children) == len(action_space)

    def best_child(self, exploration_weight=1.0):
        """
        Selects the best child node using the Upper Confidence Bound (UCB) formula.
        Args:
            exploration_weight: Weight for exploration in the UCB formula.
        Returns:
            The child node with the highest UCB value.
        """
        return max(self.children.values(), key=lambda child: child.ucb_score(exploration_weight))

    def ucb_score(self, exploration_weight):
        """
        Calculates the Upper Confidence Bound (UCB) score.
        Args:
            exploration_weight: Weight for exploration.
        Returns:
            The UCB score.
        """
        if self.visits == 0:
            return float('inf')  # Infinite score for unvisited nodes
        exploitation_score = self.total_reward / self.visits
        exploration_score = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation_score + exploration_score

    def expand(self, action, next_state):
        """
        Expands this node by adding a child node for a given action and state.
        Args:
            action: The action taken to reach the child state.
            next_state: The child state reached by the action.
        Returns:
            The new child node.
        """
        child_node = MCTSNode(state=next_state, parent=self)
        self.children[action] = child_node
        return child_node

    def backpropagate(self, reward):
        """
        Backpropagates the reward to update this node and all its ancestors.
        Args:
            reward: The reward to propagate.
        """
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)

def monte_carlo_tree_search(env, root_state, num_simulations, exploration_weight=1.0):
    """
    Performs Monte Carlo Tree Search to find the best action from the root state.
    Args:
        env: The environment instance.
        root_state: The starting state.
        num_simulations: Number of simulations to perform.
        exploration_weight: Exploration weight in UCB formula.
    Returns:
        The best action to take from the root state.
    """
    root = MCTSNode(root_state)

    for _ in range(num_simulations):
        # Selection
        node = root
        done = False
        env.reset_to_state(root.state)
        while node.is_fully_expanded(env.actions) and node.children:
            action, node = max(node.children.items(), key=lambda item: item[1].ucb_score(exploration_weight))
            _, _, done, _ = env.step(action)
            if done:
                break

        # Expansion
        if not done:
            for action in env.actions:
                if action not in node.children:
                    next_state, _, _, _ = env.step(action)
                    node.expand(action, next_state)

        # Simulation
        current_state = env.current_state
        total_reward = 0
        for _ in range(100):  # Simulate up to 100 steps
            action = random.choice(env.actions)
            _, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break

        # Backpropagation
        node.backpropagate(total_reward)

    # Choose the best action based on visit count
    best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
    return best_action

if __name__ == "__main__":
    # Initialize the environment
    env = GridWorldEnv(4, 4, 4)

    # Start state
    initial_state = env.reset()

    # Perform MCTS until reaching the final state
    num_simulations = 1000
    best_actions = {}

    current_state = initial_state

    while True:
        best_action = monte_carlo_tree_search(env, current_state, num_simulations)
        best_actions[current_state] = best_action
        next_state, _, done, _ = env.step(best_action)
        print(f"Best action from state {current_state}: {best_action} / Next state: {next_state}")
        current_state = next_state
        if current_state == env.terminal_state:
            print("Reached the final state.")
            break

    print("Best actions for each state:")
    for state, action in best_actions.items():
        print(f"State {state}: Best action {action}")
