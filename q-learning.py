import numpy as np
import random
from grid_world_env import GridWorldEnv
import matplotlib.pyplot as plt


class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=0.6, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initializes the Q-Learning agent with necessary parameters.
        """
        self.q_table = np.zeros((state_size, action_size))
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    
    def state_order(self, state):
        return state[0] * self.action_size + state[1]

    def choose_action(self, state):
        """
        Chooses an action based on the epsilon-greedy policy.
        """
        state_ord = self.state_order(state)
        if random.uniform(0, 1) < self.epsilon:
            # Exploration
            action = random.choice(range(self.action_size))
        else:
            # Exploitation
            max_value = np.max(self.q_table[state_ord])

            actions_with_max_value = np.where(self.q_table[state_ord] == max_value)[0]
            action = random.choice(actions_with_max_value)
        return action

    def learn(self, state, action, reward, next_state, done):
        """
        Updates the Q-table based on the action taken and the reward received.
        """
        current_q = self.q_table[self.state_order(state), action]
        
        # ! Update rule for Q-learning
        if done:
            target = reward
        else:
            next_max = np.max(self.q_table[self.state_order(next_state)])
            target = reward + self.gamma * next_max

        self.q_table[self.state_order(state), action] += self.alpha * (target - current_q)
        # ! End update rule for Q-learning

        # @note : the more we train the less we explore
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
    def policy(self):
        """
        Obtains the learned policy from the Q-table.
        
        Returns:
            A dictionary mapping states to actions.
        """
        policy = {state: random.choice(env.actions) for state in env.states}

        for state in env.states:
            best_action = np.argmax(agent.q_table[agent.state_order(state)])
            policy[state] = env.actions[best_action]
        return policy


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

    agent = QLearningAgent(state_size=state_size, action_size=action_size)


    num_episodes = 1000
    max_steps_per_episode = 100  # To prevent infinite loops


    rewards_per_episode = []
    steps_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()  # Reset environment at the start of each episode
        done = False
        total_reward = 0
        steps = 0

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)  # Choose an action based on current policy
            next_state, reward, done, _ = env.step(env.actions[action])

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

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

    

    print("Learned Q-learning Policy:")
    env.print_policy(agent.policy())
    