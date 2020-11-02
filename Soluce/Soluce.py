import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from BlobEnv import BlobEnv
from collections import namedtuple

EPISODES = 50_001
SHOW_EVERY = 5000

LEARNING_RATE = 0.1
DISCOUNT = 0.95

epsilon = 0.9
EPS_DECAY = 0.99998

MEM_SIZE = 400

start_q_table = None

stepAllowed = 200
BlobEnv = BlobEnv(stepAllowed=stepAllowed)

def init_Qtable():
    q_table = {}
    for x1 in range(-BlobEnv.SIZE + 1, BlobEnv.SIZE):
        for y1 in range(-BlobEnv.SIZE + 1, BlobEnv.SIZE):
            for x2 in range(-BlobEnv.SIZE + 1, BlobEnv.SIZE):
                for y2 in range(-BlobEnv.SIZE + 1, BlobEnv.SIZE):
                    q_table[((x1,y1), (x2,y2))] = [np.random.uniform(-5,0) for i in range (4)]
    return q_table

def select_action(epsilon, q_table, state):
    if np.random.random() > epsilon:
        action = np.argmax(q_table[obs])
    else:
        action = np.random.randint(0, 4)
    return action

if start_q_table is None:
    q_table = init_Qtable()
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

class Memory():
    def __init__(self, mem_size):
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.memory = []
        self.mem_size = mem_size

    def add(self, state, action, reward, next_state):
        if len(self.memory) >= self.mem_size:
            self.memory = self.memory[:1]
        transition = self.transition(state, action, reward, next_state)
        self.memory.append(transition)

    def train(self, q_table):
        for transition in memory.memory:
            state, action, reward, next_state = transition
            q_table[state][action] = (1 - LEARNING_RATE) * q_table[state][action] + LEARNING_RATE * (reward + DISCOUNT * np.max(q_table[next_state]))

        return q_table

episode_rewards = []
memory = Memory(MEM_SIZE)

for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0 and episode != 0:
        print(f"On episode {episode}, epsilon is {epsilon}")
        print(f"Mean on {SHOW_EVERY} ep: {np.mean(episode_rewards[-SHOW_EVERY:])}\n")
        show = True
    else:
        show = False

    episode_reward = 0
    obs = BlobEnv.getObs()

    for i in range(stepAllowed):
        action = select_action(epsilon, q_table, obs)
        obs, new_obs, reward, done = BlobEnv.doAction(action)
        memory.add(obs, action, reward, new_obs)
        if show:
            BlobEnv.show()
        episode_reward += reward
        if done:
            break

    q_table = memory.train(q_table)
    epsilon *= EPS_DECAY
    BlobEnv.resetEpisode()
    episode_rewards.append(episode_reward)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"q_table-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)