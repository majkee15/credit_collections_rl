from learning.collections_env.collections_env import CollectionsEnv

import numpy as np
import matplotlib.pyplot as plt

env = CollectionsEnv()

done = False
obs = env.reset()
rew_path = []
ls = []
while not done:
    state, reward, done, _ = env.step(0)
    rew_path.append(reward)
    ls.append(state[0])

plt.plot(rew_path)
plt.show()

plt.plot(ls)
plt.show()
print(f'Accumulated reward: {np.sum(rew_path)}')
