import gym
import numpy as np

class DiscretizedActionWrapper(gym.ActionWrapper):
    """
    This wrapper converts a Box observation into a single integer.
    """

    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        # Check whether the env is already discrete
        assert isinstance(env.action_space, gym.spaces.Box)

        low = self.action_space.low if low is None else low
        high = self.action_space.high if high is None else high

        self.n_action_bins = n_bins
        self.action_bins = np.linspace(low, high, n_bins + 1)
        self.action_space = gym.spaces.Discrete(self.action_bins[0].shape[0])
        print("New ob space:", gym.spaces.Discrete((self.n_action_bins + 1) ** len(low)))

    def _aind_to_cell(self, ind):
        return self.action_bins[ind]

    def action(self, action):
        return self.action_bins[action][0]

if __name__ == '__main__':
    from learning.collections_env.gymcollectionsenv import CollectionsEnv
    env = CollectionsEnv()
    enva = DiscretizedActionWrapper(env)
    print(enva.action(10))
    print(enva.observation_space.sample())
    print(env.observation_space.sample())
    print(enva.observation_space.sample())