from learning.collections_env.collections_env import CollectionsEnv

import numpy as np
import matplotlib.pyplot as plt

env = CollectionsEnv()

done = False
obs = env.reset()
rew_path = []
ls = []
ws = []
while not done:
    state, reward, done, _ = env.step(0.0)
    rew_path.append(reward)
    ls.append(state[0])
    ws.append(state[1])

fig, ax = plt.subplots()
ax.plot(rew_path)
ax.set_title('Reward path')
ax.set_xlabel('Step')
fig.show()

fig, ax = plt.subplots()
ax.plot(ls)
ax.set_title('Lambda')
ax.set_xlabel('Step')
fig.show()

fig, ax = plt.subplots()
ax.plot(ws)
ax.set_title('Balance')
ax.set_xlabel('Step')
fig.show()

print(f'Accumulated reward: {np.sum(rew_path)}')


