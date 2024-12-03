import numpy as np
import random
from grid_world_env import GridWorldEnv
import matplotlib.pyplot as plt


class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.5, exploration_decay=0.995, min_exploration_rate=0.01):
        """
        Initializes the Q-Learning agent with necessary parameters.
        
        Args:
            state_size: The number of possible states.
            action_size: The number of possible actions.
            learning_rate: The rate at which the agent learns.
            discount_factor: The discount factor for future rewards.
            exploration_rate: The initial exploration rate (epsilon).
            exploration_decay: The rate at which exploration decreases.
            min_exploration_rate: The minimum exploration rate.
        """
        self.q_table = np.zeros((state_size, action_size))  # Initialize Q-table as a matrix
        self.action_size = action_size  # Number of possible actions
        self.alpha = learning_rate  # Learning rate
        self.gamma = discount_factor  # Discount factor
        self.epsilon = exploration_rate  # Exploration rate
        self.epsilon_decay = exploration_decay  # Exploration decay rate
        self.epsilon_min = min_exploration_rate  # Minimum exploration rate

    def choose_action(self, state):
        """
        Chooses an action based on the epsilon-greedy policy.
        
        Args:
            state: The current state.
        
        Returns:
            The action chosen.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: choose a random action
            action = random.choice(range(self.action_size))
        else:
            # Exploitation: choose the best action based on Q-table
            max_value = np.max(self.q_table[state])
            # In case multiple actions have the same max value, randomly choose among them
            actions_with_max_value = np.where(self.q_table[state] == max_value)[0]
            action = random.choice(actions_with_max_value)
        return action

    def learn(self, state, action, reward, next_state, done):
        """
        Updates the Q-table based on the action taken and the reward received.
        
        Args:
            state: The previous state.
            action: The action taken.
            reward: The reward received.
            next_state: The state transitioned to.
            done: Boolean indicating if the episode has ended.
        """
        current_q = self.q_table[state, action]  # Current Q-value
        if done:
            target = reward  # If done, the target is the immediate reward
        else:
            # Estimate of optimal future value
            next_max = np.max(self.q_table[next_state])
            target = reward + self.gamma * next_max

        # Q-Learning update rule
        self.q_table[state, action] += self.alpha * (target - current_q)

        # if done:
        #     # Decay exploration rate after each episode
        #     self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


if __name__ == "__main__":
    # Environment parameters
    height = 4
    width = 4
    number_of_holes = 4

    # Initialize the environment
    env = GridWorldEnv(height, width, number_of_holes)
    state_size = height * width
    action_size = len(env.actions)

    # Initialize the Q-Learning agent
    agent = QLearningAgent(state_size=state_size, action_size=action_size)

    # Training parameters
    num_episodes = 1000
    max_steps_per_episode = 100  # To prevent infinite loops

    # Tracking performance
    rewards_per_episode = []
    steps_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()  # Reset environment at the start of each episode
        done = False
        total_reward = 0
        steps = 0

        for step in range(max_steps_per_episode):
            action = agent.choose_action(sum(state))  # Choose an action based on current policy
            print(f"action {action}")
            next_state, reward, done, _ = env.step(env.actions[action])  # Take the action in the environment

            agent.learn(state, action, reward, sum(next_state), done)  # Update Q-table

            state = next_state  # Move to the next state
            total_reward += reward  # Accumulate reward
            steps += 1

            if done:
                break  # Episode ends if done is True

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        # Optional: Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.4f}")

    # Plotting the rewards and steps over episodes
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_episode)
    plt.title('Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 2, 2)
    plt.plot(steps_per_episode)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.tight_layout()
    plt.show()

    # Testing the learned policy
    test_episodes = 5
    for test in range(test_episodes):
        state = env.reset()
        done = False
        print(f"Test Episode {test + 1}:")
        env.render()

        while not done:
            # Choose the best action based on the learned Q-table
            max_value = np.max(agent.q_table[sum(state)])
            print(f" q_table {agent.q_table}")
            actions_with_max_value = np.where(agent.q_table[sum(state)] == max_value)[0]
            action = random.choice(actions_with_max_value)

            next_state, reward, done, _ = env.step(action)
            env.render()

            state = next_state

        print(f"Episode {test + 1} finished with reward {reward}\n")