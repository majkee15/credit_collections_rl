import os

import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete
from gym.utils import colorize
from copy import deepcopy

class TrainConfig:
    lr = 0.001
    n_steps = 20000
    warmup_steps = 10000
    batch_size = 64
    log_every_step = 1000

    # give an extra bonus if done; only needed for certain tasks.
    done_reward = None


class Policy:
    def __init__(self, env, name, training=True, gamma=0.99, deterministic=False):
        self.env = env
        self.gamma = gamma
        self.training = training
        self.name = name

        if deterministic:
            np.random.seed(1)

    @property
    def act_size(self):
        # number of options of an action; this only makes sense for discrete actions.
        if isinstance(self.env.action_space, Discrete):
            return self.env.action_space.n
        else:
            return None

    @property
    def act_dim(self):
        # dimension of an action; this only makes sense for continuous actions.
        if isinstance(self.env.action_space, Box):
            return list(self.env.action_space.shape)
        else:
            return []

    @property
    def state_dim(self):
        # dimension of a state.
        return list(self.env.observation_space.shape)

    def obs_to_inputs(self, ob):
        return ob.flatten()

    def act(self, state, **kwargs):
        pass

    def build(self):
        pass

    def train(self, *args, **kwargs):
        pass

    def evaluate(self, n_episodes):
        reward_history = []
        reward = 0.

        for i in range(n_episodes):
            ob = self.env.reset()
            done = False
            while not done:
                a = self.act(ob)
                new_ob, r, done, _ = self.env.step(a)
                self.env.render()
                reward += r
                ob = new_ob

            reward_history.append(reward)
            reward = 0.

        print("Avg. reward over {} episodes: {:.4f}".format(n_episodes, np.mean(reward_history)))