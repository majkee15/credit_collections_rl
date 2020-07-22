from learning.collections_env.collections_env import CollectionsEnv

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=1)
env = CollectionsEnv(reward_shaping='continuous')

done = False
state = env.reset()
rew_path = []
discrete_rew = []
ls = []
ws = []

ls.append(state[0])
ws.append(state[1])
while not done:
    state, reward, done, _ = env.step(0.0)
    rew_path.append(reward)

    if state[1] != ws[-1]:
        # jump has occured
        discrete_rew.append((ws[-1] - state[1]) * np.exp(-env.params.rho * env.current_time))
    else:
        discrete_rew.append(0)

    ls.append(state[0])
    ws.append(state[1])



print(discrete_rew)


fig, ax = plt.subplots()
ax.plot(rew_path)
ax.set_title('Continuous Reward')
ax.set_xlabel('Step')
fig.show()

fig, ax = plt.subplots()
ax.plot(np.cumsum(rew_path))
ax.plot(np.cumsum(discrete_rew))
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

print(f'Accumulated reward continuous: {np.sum(rew_path)}')
print(f'Accumulated reward sparse: {np.sum(discrete_rew)}')


