import os

import logging
import datetime


import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete
from gym.utils import colorize

from learning.utils.misc import Config
from learning.utils.misc import REPO_ROOT, RESOURCE_ROOT

from abc import ABC, abstractmethod



class TrainConfigBase(Config):
    lr = 0.001
    n_steps = 10000
    warmup_steps = 5000
    batch_size = 64
    log_every_step = 1000

    # give an extra bonus if done; only needed for certain tasks.
    done_reward = None


class Policy(ABC):

    def __init__(self, env, name, training=True, deterministic=False):
        self.env = env

        self.training = training
        self.name = self.__class__.__name__ + '--' + name

        if deterministic:
            np.random.seed(1)

        # Logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
        # self.logger.info('Instantiated class ' + self.__class__.__name__)

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

    @staticmethod
    def obs_to_inputs(self, ob):
        return ob.flatten()

    @abstractmethod
    def get_action(self, state, **kwargs):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    def evaluate(self, n_episodes):
        # TODO: evaluate uses default setting of the environment, i.g., random start
        # this should be done in parallel
        # and it should be depending on a starting state!
        reward_history = []

        for i in range(n_episodes):
            ob = self.env.reset()
            done = False
            reward = 0.
            while not done:
                a, q = self.get_action(ob, epsilon=0.0)
                new_ob, r, done, _ = self.env.step(a)
                # self.env.render()
                reward += r
                ob = new_ob

            reward_history.append(reward)

        #print("Avg. reward over {} episodes: {:.4f}".format(n_episodes, np.mean(reward_history)))
        self.logger.info("Avg. reward over {} episodes: {:.4f}".format(n_episodes, np.mean(reward_history)))
        return reward_history


class BaseModelMixin(ABC):

    def __init__(self, model_name, experiment_name=None):
        self._saver = None
        self._writer = None
        self._experiment_name = experiment_name
        self.model_name = model_name
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def _get_dir(self, dir_name):
        if self._experiment_name is not None:
            path = os.path.join(RESOURCE_ROOT, dir_name, self._experiment_name, self.model_name, self.current_time)
        else:
            path = os.path.join(RESOURCE_ROOT, dir_name, self.model_name, self.current_time)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def log_dir(self):
        return self._get_dir('training_logs')

    @property
    def checkpoint_dir(self):
        return self._get_dir('checkpoints')

    @property
    def model_dir(self):
        return self._get_dir('models')

    @property
    def tb_dir(self):
        # tensorboard
        return self._get_dir('tb_logs')

    @property
    def writer(self):
        if self._writer is None:
            self._writer = tf.summary.create_file_writer(self.tb_dir)
        return self._writer
