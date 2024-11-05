import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import load_q_table, clear_console
from time import sleep, time

Qs = ['Q1.csv', 'Q2.csv']

def train(n_episodes, hyperparams, env, Q_paths):
  lr = hyperparams["lr"] or 0.1
  gamma = hyperparams["gamma"] or 0.6
  print(f"Start training double q learning: lr: {lr}, gamma: {gamma}")

  epsilon = 1
  epsilon_decay = 0.0001

  Q1 = np.zeros([env.observation_space.n, env.action_space.n])
  Q2 = np.zeros([env.observation_space.n, env.action_space.n])

  for i in range(n_episodes):
    state = env.reset()[0]
    done = False
    truncated = False
    penalties, total_rewards = 0, 0
      
    while not done and not truncated:
      if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
      else:
        action_values = Q1[state, :] + Q2[state, :]
        action = np.argmax(action_values)

      next_state, reward, done, truncated, _ = env.step(action)

      if random.uniform(0, 1) < 0.5:
        best_action = np.argmax(Q1[next_state, :])
        old_value = Q1[state, action]
        new_value = old_value + lr * (reward + gamma * Q2[next_state, best_action] - old_value)
        Q1[state, action] = new_value
      else:
        best_action = np.argmax(Q2[next_state, :])
        old_value = Q2[state, action]
        new_value = old_value + lr * (reward + gamma * Q1[next_state, best_action] - old_value)
        Q2[state, action] = new_value

      if reward == -10:
        penalties += 1
      
      state = next_state
      total_rewards += reward
    
    epsilon = max(epsilon - epsilon_decay, 0)
    if epsilon == 0:
      lr = 0.0001

    if i % 100 == 0:
      # print(f"Episode: {i}")
      np.savetxt(Q_paths[0], Q1, delimiter=",")
      np.savetxt(Q_paths[1], Q2, delimiter=",")

  print("Training finished.\n")

# visualization
def run(Qs, env):
  state = env.reset()[0]
  done = False
  truncated = False
  Q1, Q2 = Qs

  while not done and not truncated:
    # clear_console()
    text = env.render()
    print(text)

    action_values = Q1[state, :] + Q2[state, :]
    action = np.argmax(action_values)

    next_state, reward, done, truncated, _ = env.step(action)
    state = next_state
    print(state)
        
  print("Run finished.\n")

# train
env = gym.make('Taxi-v3')
n_episodes = 15_000

hyperparams = {
  "lr": 0.9,
  "gamma": 0.9, # discount
  "epsilon": 1,
  "epsilon_decay": 0.0001
}

s = time()
train(n_episodes, hyperparams, env, Qs)
print("--- Double Q-learning %s seconds ---" % (time() - s))

# run
env = gym.make('Taxi-v3', render_mode="ansi")

Q1 = load_q_table(Qs[0])
Q2 = load_q_table(Qs[1])

run([Q1, Q2], env)