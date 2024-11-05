from os import system
import numpy as np

def clear_console():
  system('cls')

def load_q_table(path):
  return np.loadtxt(open(path, "rb"), delimiter=",")
