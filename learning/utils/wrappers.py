import gym
import numpy as np
from gym.spaces import Box, Discrete


class DiscretizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, Box)

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        low = np.array(low)
        high = np.array(high)

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in
                         zip(low.flatten(), high.flatten())]
        self.ob_shape = self.observation_space.shape

        print("New ob space:", Discrete((n_bins + 1) ** len(low)))
        self.observation_space = Discrete(n_bins ** len(low))

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.val_bins)]
        return self._convert_to_one_number(digits)



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
        print("New action space:", gym.spaces.Discrete((self.n_action_bins + 1) ** len(low)))
        self.action_space.n = self.n_action_bins + 1

    def _aind_to_cell(self, ind):
        return self.action_bins[ind]

    def action(self, action):
        return self.action_bins[action][0]


if __name__ == '__main__':
    from learning.collections_env.gymcollectionsenv import CollectionsEnv
    env = CollectionsEnv()
    bins = 1
    enva = DiscretizedActionWrapper(env, n_bins=bins)
    print(enva.action(0))
    print(enva.action(bins))
    print(f' N actions: {enva.action_space.n}')

    sample = enva.action_space.sample()
    print(f'Sampled: {sample}')
    print(f'Action: {enva.action(sample)}')

    sample = enva.action_space.sample()
    print(f'Sampled: {sample}')
    print(f'Action: {enva.action(sample)}')


    sample = enva.action_space.sample()
    print(f'Sampled: {sample}')
    print(f'Action: {enva.action(sample)}')