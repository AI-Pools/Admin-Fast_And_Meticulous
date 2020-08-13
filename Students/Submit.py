import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from matplotlib import style
from BlobEnv import BlobEnv

style.use("ggplot")

# SIZE = 10
EPISODES = 0   # TODO
SHOW_EVERY = 0 # TODO

LEARNING_RATE = 0 # TODO
DISCOUNT = 0      # TODO

epsilon = 0   # TODO
EPS_DECAY = 0 # TODO

start_q_table = None # It can aslo be a path to a Qtable

stepAllowed = 200
BlobEnv = BlobEnv(stepAllowed=stepAllowed)

if start_q_table is None:
    pass
    # TODO: Init the Qtable
else:
    pass
    # TODO: Load the Qtable

for episode in range(EPISODES):
    # TODO: after X episode show some stat about the agent
    for i in range(stepAllowed):
        pass
        # TODO: Do an action depending of the obs, epsilon and the Qtable (be shure the episode is not already done)

        # TODO: Update the Qtable

        # TODO: Show every X episode a full one

    # TODO: Don't forget to reset the env after each episode

# TODO: Show a graph showing the evolution of the rewards

# TODO: Save the current qtable