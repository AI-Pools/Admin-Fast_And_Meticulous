import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from matplotlib import style
from BlobEnv import BlobEnv

style.use("ggplot")

EPISODES = 25_000
SHOW_EVERY = 5000

LEARNING_RATE = 0.1
DISCOUNT = 0.95

epsilon = 0.9
EPS_DECAY = 0.9998

start_q_table = None

stepAllowed = 200
BlobEnv = BlobEnv(stepAllowed=stepAllowed)

if start_q_table is None:
    q_table = {}
    for x1 in range(-BlobEnv.SIZE + 1, BlobEnv.SIZE):
        for y1 in range(-BlobEnv.SIZE + 1, BlobEnv.SIZE):
            for x2 in range(-BlobEnv.SIZE + 1, BlobEnv.SIZE):
                for y2 in range(-BlobEnv.SIZE + 1, BlobEnv.SIZE):
                    q_table[((x1,y1), (x2,y2))] = [np.random.uniform(-5,0) for i in range (4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []

for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(f"On episode {episode}, epsilon is {epsilon}")
        print(f"Mean on {SHOW_EVERY} ep: {np.mean(episode_rewards[-SHOW_EVERY:])}\n")
        show = True
    else:
        show = False
    episode_reward = 0
    for i in range(stepAllowed):
        obs = BlobEnv.getObs()
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        obs, new_obs, reward = BlobEnv.doAction(action)

        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]
        if reward == BlobEnv.FOOD_REWARD:
            new_q = BlobEnv.FOOD_REWARD
        elif reward == -BlobEnv.ENEMY_PENALTY:
            new_q = -BlobEnv.ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q
        if show:
            BlobEnv.show()
        episode_reward += reward
        if reward == BlobEnv.FOOD_REWARD or reward == -BlobEnv.ENEMY_PENALTY:
            break
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    BlobEnv.resetEpisode()

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)