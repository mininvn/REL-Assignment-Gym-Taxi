import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from time import time


# Define the plotting function early in the script
def plot_returns(returns):
    plt.plot(np.arange(len(returns)), returns)
    plt.title('Episode returns')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.show()

class SARSAAgent:
    def __init__(self, env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor=0.95):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, obs) -> int:
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_values[obs])

    def update(self, obs, action, reward, terminated, next_obs, next_action):
        if not terminated:
            td_target = reward + self.discount_factor * self.q_values[next_obs][next_action]
            td_error = td_target - self.q_values[obs][action]
            self.q_values[obs][action] += self.learning_rate * td_error

    def decay_epsilon(self):
        """Decrease the exploration rate epsilon until it reaches its final value"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_q_values(self, filename="q_values.pkl"):
        """Save the Q-table to a file."""
        with open(filename, "wb") as f:
            pickle.dump(dict(self.q_values), f)
        print(f"Q-table saved to {filename}")

    def load_q_values(self, filename="q_values.pkl"):
        """Load the Q-table from a file."""
        with open(filename, "rb") as f:
            self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n), pickle.load(f))
        print(f"Q-table loaded from {filename}")

def train_agent(agent, env, episodes, eval_interval=100):
    rewards = []
    best_reward = -np.inf
    for i in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        total_reward = 0  # Initialize total_reward correctly here

        while not terminated and not truncated:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.get_action(next_obs)  # Get the next action using current policy
            agent.update(obs, action, reward, terminated, next_obs, next_action)
            obs = next_obs
            action = next_action
            total_reward += reward  # Use the correct variable name here

        agent.decay_epsilon()
        rewards.append(total_reward)

        if i % eval_interval == 0 and i > 0:
            avg_return = np.mean(rewards[max(0, i - eval_interval):i])
            best_reward = max(avg_return, best_reward)
            print(f"Episode {i} -> best_reward={best_reward}")

    return rewards


# Initialize the environment and agent
env = gym.make('Taxi-v3', render_mode='ansi')
episodes = 15000
learning_rate = 0.1
discount_factor = 0.9
initial_epsilon = 1
final_epsilon = 0.1
# epsilon_decay = ((final_epsilon - initial_epsilon) / (episodes/2))
epsilon_decay = 0.99  # Exponential decay

agent = SARSAAgent(env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon)

# Train the agent and plot the returns
s = time()
returns = train_agent(agent, env, episodes)
print("--- SARSA %s seconds ---" % (time() - s))
# plot_returns(returns)

agent.save_q_values("taxi_q_values.pkl")

def plot_returns(returns):
    plt.plot(np.arange(len(returns)), returns)
    plt.title('Episode returns')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.show()

def run_agent(agent, env):
    agent.epsilon = 0  # No need to keep exploring
    obs, _ = env.reset()
    env.render()
    terminated = truncated = False

    while not terminated and not truncated:
        action = agent.get_action(obs)
        next_obs, _, terminated, truncated, _ = env.step(action)
        print(env.render())
        obs = next_obs

env = gym.make('Taxi-v3')
agent = SARSAAgent(env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon)

# Load Q-values from a file
agent.load_q_values("taxi_q_values.pkl")

# Run the agent without training
env = gym.make('Taxi-v3', render_mode='ansi')

run_agent(agent, env)
