from collections import defaultdict
from learning.collections_env import DiscretizedObservationWrapper, DiscretizedActionWrapper
from learning.collections_env import CollectionsEnv
import numpy as np

Q = defaultdict(float)
gamma = 0.99  # Discounting factor
alpha = 0.5  # soft update param

env = CollectionsEnv()

print(f'Action space: {env.action_space}')
print(f'State space: {env.observation_space}')

env = DiscretizedActionWrapper(env)
env = DiscretizedObservationWrapper(env)
actions = range(env.action_space.n)

print(f'Discretized action space: {env.action_space}')
print(f'Discretized state space: {env.observation_space}')


def update_Q(s, r, a, s_next, done):
    max_q_next = max([Q[s_next, a] for a in actions])
    # Do not include the next state's value if currently at the terminal state.
    Q[s, a] += alpha * (r + gamma * max_q_next * (1.0 - done) - Q[s, a])


def act(ob):
    if np.random.random() < epsilon:
        # action_space.sample() is a convenient function to get a random action
        # that is compatible with this given action space.
        return env.action_space.sample()

    # Pick the action with highest q value.
    qvals = {a: Q[ob, a] for a in actions}
    max_q = max(qvals.values())
    # In case multiple actions have the same maximum q value.
    actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
    return np.random.choice(actions_with_max_q)


def qlearn(env, episodes, Q=None):
    rewards = []
    time_erect = []
    avg_reward = []
    for i_episode in range(episodes):
        ob = env.reset()
        print(ob)
        print(env.obs_val_bins)
        print(env._ind_to_cell(ob))
        reward = 0.0
        for t in range(n_steps):
            # env.render()
            # take action based on an observation
            a = act(ob)
            ob_next, r, done, _ = env.step(a)
            update_Q(ob, r, a, ob_next, done)
            reward = reward + r
            if done:
                avg_reward.append(reward)
                time_erect.append(t+1)
                print("Episode finished after {} timesteps".format(t+1))

                break
            else:
                ob = ob_next
    return avg_reward, time_erect





# RUN
if __name__ == '__main__':

    # n_steps = 200
    # episodes = 1
    # epsilon = 0.1  # 10% chances to apply a random action
    #
    #
    # avg_reward, time_erect = qlearn(env, episodes)
    # plt.plot(np.arange(len(avg_reward)), avg_reward)
    # plt.xlabel('Episodes')
    # plt.ylabel('Average Reward')
    # plt.title('Average Reward vs Episodes')
    # plt.savefig('rewards.jpg')
    # plt.close()
    #
    # plt.plot(time_erect)
    # plt.xlabel('Episodes')
    # plt.ylabel('Time Erect')
    # plt.title('Time Erect')
    # plt.savefig('erect.jpg')
    # plt.close()

    ob = env.reset()
    print(env.observation_space.n)
    print(ob)
    print(env.obs_val_bins)
