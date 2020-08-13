### Soluce code: Soluce.py

### Details:

Hyperparameters:
```py
EPISODES = 25_000
SHOW_EVERY = 5000

LEARNING_RATE = 0.1
DISCOUNT = 0.95

epsilon = 0.9
EPS_DECAY = 0.9998
```

Init the Qtable:
```py
q_table = {}
    for x1 in range(-BlobEnv.SIZE + 1, BlobEnv.SIZE):
        for y1 in range(-BlobEnv.SIZE + 1, BlobEnv.SIZE):
            for x2 in range(-BlobEnv.SIZE + 1, BlobEnv.SIZE):
                for y2 in range(-BlobEnv.SIZE + 1, BlobEnv.SIZE):
                    q_table[((x1,y1), (x2,y2))] = [np.random.uniform(-5,0) for i in range (4)]
```
Load the Qtable:
```py
with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)
```

After X episode show some stat about the agent:
```py
if episode % SHOW_EVERY == 0:
    print(f"On episode {episode}, epsilon is {epsilon}")
    print(f"Mean on {SHOW_EVERY} ep: {np.mean(episode_rewards[-SHOW_EVERY:])}\n")
    show = True
```

Do an action depending of the obs, epsilon and the Qtable:
```py
 obs = BlobEnv.getObs()
if np.random.random() > epsilon:
    action = np.argmax(q_table[obs])
else:
    action = np.random.randint(0, 4)
obs, new_obs, reward = BlobEnv.doAction(action)
```

Update the Qtable:
```py
max_future_q = np.max(q_table[new_obs])
current_q = q_table[obs][action]
if reward == BlobEnv.FOOD_REWARD:
    new_q = BlobEnv.FOOD_REWARD
elif reward == -BlobEnv.ENEMY_PENALTY:
    new_q = -BlobEnv.ENEMY_PENALTY
else:
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward+DISCOUNT * max_future_q)
q_table[obs][action] = new_q
```

Show every X episode a full one:
```py
if show:
    BlobEnv.show()
```

Don't forget to reset the env after each episode: ``BlobEnv.resetEpisode()``

Show a graph showing the evolution of the rewards:
```py
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()
```

Save the current qtable:
```py
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
```
### <u>For Any other question please refer to Junior/Senior responsable</u>