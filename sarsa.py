import numpy as np
import random
from grid_world_env import GridWorldEnv
import matplotlib.pyplot as plt



class SARSAAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.99, epsilon=0.6, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initializes the SARSA agent.
        """
        self.q_table = np.zeros((state_size, action_size))  # Initialize Q-table
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_size = action_size

    def state_order(self, state):
        """
        Converts a 2D grid state into a unique index for the Q-table.
        """
        return state[0] * self.action_size + state[1]

    def choose_action(self, state):
        """
        Chooses an action using the epsilon-greedy policy.
        """
        state_ord = self.state_order(state)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            return np.argmax(self.q_table[state_ord])

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        Updates the Q-value using the SARSA update rule.
        """
        state_ord = self.state_order(state)
        next_state_ord = self.state_order(next_state)

        current_q = self.q_table[state_ord, action]

        # ! Update rule for SARSA
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state_ord, next_action]

        self.q_table[state_ord, action] += self.alpha * (target - current_q)
        # ! End update rule for SARSA

        # @note : the more we train the less we explore
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
    random_seed = 2020
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    height = 4
    width = 4
    number_of_holes = 4

    env = GridWorldEnv(height, width, number_of_holes)
    state_size = height * width
    action_size = len(env.actions)

    agent = SARSAAgent(state_size=state_size, action_size=action_size)

    num_episodes = 1000
    max_steps_per_episode = 100


    rewards_per_episode = []
    steps_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()  # Reset environment
        action = agent.choose_action(state)
        done = False
        total_reward = 0
        steps = 0

        for _ in range(max_steps_per_episode):
            next_state, reward, done, _ = env.step(env.actions[action])
            next_action = agent.choose_action(next_state)
            
            agent.learn(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

            if done:
                break

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: Reward = {total_reward}, Epsilon = {agent.epsilon:.4f}")



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

    # Display learned policy
    learned_policy = agent.policy()
    print("\nLearned SARSA Policy:")
    env.print_policy(learned_policy)
