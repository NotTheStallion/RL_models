import numpy as np
import random
from simple.grid_world_env import GridWorldEnvSlow
import matplotlib.pyplot as plt

class MonteCarloAgent:
    def __init__(self, env, state_size, action_size, gamma=0.99, epsilon=0.7, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initializes the Monte Carlo agent.
        """
        self.env = env
        self.q_table = np.zeros((state_size, action_size))
        self.returns = { (s, a): [] for s in range(state_size) for a in range(action_size) }
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_size = action_size
        self.visits = np.zeros(state_size)

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

    def learn(self, trajectory):
        """
        Updates the Q-value using Monte Carlo policy learning.
        """
        G = 0  # Initialize return
        visited = set()

        # Traverse trajectory in reverse
        for state, action, reward in reversed(trajectory):
            state_ord = self.state_order(state)
            G = reward + self.gamma * G  # Calculate return

            if (state_ord, action) not in visited:
                visited.add((state_ord, action))
                self.returns[(state_ord, action)].append(G)
                self.q_table[state_ord, action] = np.mean(self.returns[(state_ord, action)])
                self.visits[state_ord] += 1

        # @note : the more we train the less we explore
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def policy(self):
        """
        Extracts the policy from the learned Q-table.
        Returns:
            A dictionary mapping states to optimal actions.
        """
        return {state: self.env.actions[np.argmax(self.q_table[self.state_order(state)])] for state in self.env.states}

if __name__ == "__main__":
    random_seed = 2020
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    height = 4
    width = 4
    number_of_holes = 4

    env = GridWorldEnvSlow(height, width, number_of_holes)
    state_size = height * width
    action_size = len(env.actions)

    agent = MonteCarloAgent(env, state_size=state_size, action_size=action_size)

    num_episodes = 1000
    max_steps_per_episode = 100

    rewards_per_episode = []
    steps_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()  # Reset environment
        done = False
        total_reward = 0
        steps = 0
        trajectory = []

        for _ in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(env.actions[action])
            
            trajectory.append((state, action, reward))
            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        agent.learn(trajectory)
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
    print("\nLearned Monte Carlo Policy:")
    env.print_policy(learned_policy)
