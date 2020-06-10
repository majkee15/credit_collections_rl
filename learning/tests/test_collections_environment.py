from learning.collections_env.collections_env import CollectionsEnv
from dcc import Parameters, AAV
from learning.collections_env import utils

import matplotlib.pyplot as plt
import numpy as np

import unittest

SEED = 1


class TestContinuousEnvironment(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestContinuousEnvironment, self).__init__(*args, **kwargs)
        self.seed = np.random.seed(SEED)
        self.env = CollectionsEnv()

    def test_distribution_properties(self):
        action = 0
        self.env.dt = 0.05
        self.env.reset()
        # lambdas = []
        while True:
            ob, rew, done, _ = self.env.step(action)
            # lambdas.append(ob[0])
            # ws.append(ob[1])
            if len(self.env.arrivals) >= 3000:
                break
        arrivals = np.array(self.env.arrivals)
        repayments = np.array(self.env.repayments)
        p_value = utils.modelcheck(arrivals, repayments, self.env.params)
        msg = f'P_value: {p_value} on {len(self.env.arrivals)} arrivals.'
        # plt.plot(lambdas)
        # plt.show()
        self.assertGreater(p_value, 0.01, msg)






def test1(dt):
    """
    Plots the realization of the process according to the environment
    Returns:
    """

    np.random.seed(SEED)
    env = CollectionsEnv()
    env.dt = dt
    lambdas = []
    ws = []
    xdone = None
    step = 0
    done = False
    while not done:
        step += 1
        ob, rew, done, _ = env.step(0)
        lambdas.append(ob[0])
        ws.append(ob[1])
        if done and xdone is None:
            xdone = step * env.dt
    x = np.linspace(0, step * env.dt, step)
    fig, ax = plt.subplots()
    ax.plot(x, lambdas)
    ax.axvline(xdone, linestyle='--', color='black')
    ax.axhline(env.params.lambdainf, linestyle ='--', color='red', linewidth=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Intensity')
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(ws, lambdas)
    ax.axhline(env.params.lambdainf, linestyle='--', color='red', linewidth=0.5)
    ax.set_xlabel('Balance')
    ax.set_ylabel('Intensity')
    fig.show()


def test2(dt):

    np.random.seed(SEED)
    """
    Test the discretization wrapper
    Returns:

    """
    env = CollectionsEnv()
    env.dt = dt
    # envdiscr = DiscretizedObservationWrapper(env, n_bins=[10000, 100])
    envdiscr = DiscretizedObservationWrapper(env, log=True, n_bins=[50, 25])
    ob = envdiscr.reset()
    a = 0
    state_min = env.observation_space.low
    state_max = env.observation_space.high
    ob_min = envdiscr.observation(state_min)
    ob_max = envdiscr.observation(state_max)
    print(state_min)
    print(ob_min)
    print(ob_max)
    print(envdiscr.observation_space)

    step = 0
    lambdas = []
    ws = []
    xdone = 0
    wdone = 100
    done = False
    while not done:
        step += 1
        ob, rew, done, _ = envdiscr.step(0)

        cell = envdiscr._ind_to_cont(ob)
        lambdas.append(cell[0])
        ws.append(cell[1])
        if done and xdone == 0:
            xdone = step * env.dt

    x = np.linspace(0, step * env.dt, step)
    fig, ax = plt.subplots()
    ax.plot(x, lambdas)
    ax.axvline(xdone, linestyle='--', color='black')
    ax.axhline(envdiscr.params.lambdainf, linestyle ='--', color='red', linewidth=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Intensity')
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(ws, lambdas)
    ax.axhline(envdiscr.params.lambdainf, linestyle='--', color='red', linewidth=0.5)
    ax.set_xlabel('Balance')
    ax.set_ylabel('Intensity')
    fig.show()

    print(f'Done at {xdone}')

def test3():
    """
    Test the account value
    Returns:

    """
    lambda0 = 1
    w0 = 100
    dt = 0.01
    s0 = np.array([lambda0, w0], dtype=np.float32)
    p = Parameters()
    env = CollectionsEnv()
    reward_per_episode = []
    n_epochs = 1000
    for epoch in range(n_epochs):
        # print(f'start in state: {env.current_state}')
        env.reset()
        # print(f'reset in state: {env.current_state}')
        tot_reward_per_episode = 0
        lambdas = []
        ws = []
        # print(f'Start state: {env.starting_state}')
        while True:
            action = 0
            state_next, reward, terminal, _ = env.step(action)
            # print(f'Start state: {env.starting_state}')
            ws.append(state_next[1])
            tot_reward_per_episode += reward
            lambdas.append(state_next[0])
            if terminal:
                reward_per_episode.append(tot_reward_per_episode)
                break
    aav = AAV(p)
    exact_v = -aav.u(lambda0, w0)
    plt.show()
    plt.plot(reward_per_episode, marker='x')
    plt.axhline(exact_v, color='red', linestyle='--')
    plt.show()
    print(exact_v)
    print(np.mean(reward_per_episode))
