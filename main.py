import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import load_q_table, clear_console
from time import sleep, time

q_table_path = "q_table.csv"

def set_seed(seed):
  random.seed(seed)

seed = 3
set_seed(seed)

def train(n_episodes, hyperparams, env, out='q_table.csv'):
  lr = hyperparams["lr"] or 0.1
  gamma = hyperparams["gamma"] or 0.6
  epsilon = hyperparams["epsilon"] or 0.1
  print(f"Start training: lr: {lr}, gamma: {gamma}, eps: {epsilon}")

  q_table = np.zeros([env.observation_space.n, env.action_space.n])

  for i in range(n_episodes):
    state = env.reset()[0]
    done = False
    penalties, reward, = 0, 0
      
    while not done:
      if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
      else:
        action = np.argmax(q_table[state])

      next_state, reward, done, info, _ = env.step(action)
      
      old_value = q_table[state, action]
      next_max = np.max(q_table[next_state])

      new_value = (1 - lr) * old_value + lr * (reward + gamma * next_max)
      q_table[state, action] = new_value

      if reward == -10:
        penalties += 1

      state = next_state
          
    if i % 100 == 0:
      print(f"Episode: {i}")

  print("Training finished.\n")
  np.savetxt(out, q_table, delimiter=",")

# visualization
def run(q_table, env):
  state = env.reset()[0]
  epsilon = 0.1
  done = False
    
  while not done:
    text = env.render()
    print(text)
    action = np.argmax(q_table[state])
    next_state, reward, done, info, _ = env.step(action)
    state = next_state
        
  print("Run finished.\n")

# train
env = gym.make('Taxi-v3', render_mode="rgb_array").env
n_episodes = 15_000
hyperparams = {
  "lr": 0.05,
  "gamma": 0.7, # discount
  "epsilon": 0.2
}

s = time()
train(n_episodes, hyperparams, env, q_table_path)
print("--- Q-learning %s seconds ---" % (time() - s))

# run
env = gym.make('Taxi-v3', render_mode="human").env
q_table = load_q_table(q_table_path)

run(q_table, env)