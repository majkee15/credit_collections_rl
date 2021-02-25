import numpy as np
from gym import Env


class CollectionsEnvironment(Env):
    def __init__(self, params, config):

        super(CollectionsEnvironment, self).__init__()
        self.params = params

        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass