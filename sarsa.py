import numpy as np
import random
from grid_world_env import GridWorldEnv
import matplotlib.pyplot as plt



class SARSAAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.6, exploration_decay=0.995, min_exploration_rate=0.01):
        """
        Initializes the SARSA agent.
        Args:
            state_size: The number of possible states.
            action_size: The number of possible actions.
            learning_rate: Step size for updating Q-values.
            discount_factor: Discount factor for future rewards.
            exploration_rate: Initial exploration rate for epsilon-greedy policy.
            exploration_decay: Decay rate for exploration.
            min_exploration_rate: Minimum exploration rate.
        """
        self.q_table = np.zeros((state_size, action_size))  # Initialize Q-table
        self.alpha = learning_rate  # Learning rate
        self.gamma = discount_factor  # Discount factor
        self.epsilon = exploration_rate  # Exploration rate
        self.epsilon_decay = exploration_decay  # Exploration decay rate
        self.epsilon_min = min_exploration_rate  # Minimum exploration rate
        self.action_size = action_size  # Number of actions available

    def state_order(self, state):
        """
        Converts a 2D grid state into a unique index for the Q-table.
        """
        return state[0] * self.action_size + state[1]

    def choose_action(self, state):
        """
        Chooses an action using the epsilon-greedy policy.
        Args:
            state: Current state of the environment.
        Returns:
            The selected action.
        """
        state_ord = self.state_order(state)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))  # Random action for exploration
        else:
            return np.argmax(self.q_table[state_ord])  # Best action based on Q-table

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        Updates the Q-value using the SARSA update rule.
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state observed.
            next_action: Next action taken.
            done: Boolean indicating if the episode has ended.
        """
        state_ord = self.state_order(state)
        next_state_ord = self.state_order(next_state)

        # Q-value for current state-action pair
        current_q = self.q_table[state_ord, action]

        # Target value for SARSA
        if done:
            target = reward  # No future rewards if episode is done
        else:
            target = reward + self.gamma * self.q_table[next_state_ord, next_action]  # SARSA update rule

        # Update Q-value
        self.q_table[state_ord, action] += self.alpha * (target - current_q)

        # Decay epsilon after each step
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def policy(self):
        """
        Extracts the policy from the learned Q-table.
        Returns:
            A dictionary mapping states to optimal actions.
        """
        return {state: env.actions[np.argmax(self.q_table[self.state_order(state)])] for state in env.states}

if __name__ == "__main__":
    # Set random seed for reproducibility
    random_seed = 2020
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Environment parameters
    height = 4
    width = 4
    number_of_holes = 4

    # Initialize the environment
    env = GridWorldEnv(height, width, number_of_holes)
    state_size = height * width
    action_size = len(env.actions)

    # Initialize the SARSA agent
    agent = SARSAAgent(state_size=state_size, action_size=action_size)

    # Training parameters
    num_episodes = 1000
    max_steps_per_episode = 100

    # Tracking performance
    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()  # Reset environment
        action = agent.choose_action(state)  # Initial action
        done = False
        total_reward = 0

        for _ in range(max_steps_per_episode):
            next_state, reward, done, _ = env.step(env.actions[action])  # Step in environment
            next_action = agent.choose_action(next_state)  # Choose next action
            
            # Update Q-table using SARSA
            agent.learn(state, action, reward, next_state, next_action, done)

            state = next_state  # Transition to the next state
            action = next_action  # Transition to the next action
            total_reward += reward

            if done:
                break

        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: Reward = {total_reward}, Epsilon = {agent.epsilon:.4f}")

    # Plot training progress
    plt.plot(rewards_per_episode)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

    # Display learned policy
    learned_policy = agent.policy()
    print("\nLearned Policy:")
    env.print_policy(learned_policy)
