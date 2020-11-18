import gym
import numpy as np
from gym.spaces import Box, Discrete
import pickle
from sklearn.preprocessing import PolynomialFeatures

# This is Legacy
# class DiscretizedObservationWrapper(gym.ObservationWrapper):
#     def __init__(self, env, n_bins=10, low=None, high=None):
#         super().__init__(env)
#         assert isinstance(env.observation_space, Box)
#
#         low = self.observation_space.low if low is None else low
#         high = self.observation_space.high if high is None else high
#
#         low = np.array(low)
#         high = np.array(high)
#
#         self.n_bins = n_bins
#         self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in
#                          zip(low.flatten(), high.flatten())]
#         self.ob_shape = self.observation_space.shape
#
#         print("New ob space:", Discrete((n_bins + 1) ** len(low)))
#         self.observation_space = Discrete(n_bins ** len(low))
#
#     def _convert_to_one_number(self, digits):
#         return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])
#
#     def observation(self, observation):
#         digits = [np.digitize([x], bins)[0]
#                   for x, bins in zip(observation.flatten(), self.val_bins)]
#         return self._convert_to_one_number(digits)


class DiscretizedActionWrapper(gym.ActionWrapper):
    """
    This wrapper converts a Box observation into a single integer.
    """

    def __init__(self, env, bins, low=None, high=None):
        super().__init__(env)
        # Check whether the env is already discrete
        assert isinstance(env.action_space, gym.spaces.Box)

        low = self.action_space.low if low is None else low
        high = self.action_space.high if high is None else high

        if isinstance(bins, np.ndarray):
            self.action_bins = np.array(bins)
            self.n_action_bins = bins.shape[0]
        elif np.isscalar(bins):
            self.n_action_bins = bins
            self.action_bins = np.linspace(low, high, bins)

        self.action_space = gym.spaces.Discrete(self.action_bins.shape[0])
        print("New action space:", gym.spaces.Discrete(self.n_action_bins))
        self.action_space.n = self.n_action_bins

    def action(self, action):
        return self.action_bins[action]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def reverse_action(self, action):
        raise NotImplemented('Reverse action not implemented.')
        pass

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class StateNormalization(gym.ObservationWrapper):
    # squishes the statespace in [0, 1]
    def __init__(self, environment):
        super().__init__(environment)
        self.low = self.observation_space.low
        self.high = self.observation_space.high

    def observation(self, observation):
        return (observation - self.low) / self.high

    def convert_back(self, normalized_observation):
        return (normalized_observation * self.high) + self.low

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class PolynomialObservationWrapper(gym.ObservationWrapper):
    # converts statespace n = 3 to polynomial features
    def __init__(self, environment, poly_order):
        super().__init__(environment)
        self.poly_order = poly_order
        self.poly = PolynomialFeatures(3)

    def observation(self, observation):
        if observation.ndim == 1:
            return self.poly.fit_transform(observation[None, :]).flatten()
        else:
            return self.poly.fit_transform(observation)

    def convert_back(self, transformed_observation):
        # untransformed observations are at position 1 and 2
        if transformed_observation.ndim == 1:
            return transformed_observation[:, 1:3]
        else:
            return transformed_observation[:, 1:3]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)




if __name__ == '__main__':
    from learning.collections_env.collections_env import CollectionsEnv
    env = CollectionsEnv()

    print("3 actions initialized from integer")
    bins = 2

    enva = DiscretizedActionWrapper(env, bins=bins)

    print(f'Action bins: {enva.action_bins}')
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

    print("3 actions initialized from array")

    bins = np.linspace(0, enva.MAX_ACTION, 4)
    enva = DiscretizedActionWrapper(env, bins=bins)

    print(f'Action bins: {enva.action_bins}')
    print(f' N actions: {enva.action_space.n}')

    sample = enva.action_space.sample()
    print(f'Sampled: {sample}')
    print(f'Action: {enva.action(sample)}')

    sample = enva.action_space.sample()
    print(f'Sampled: {sample}')
    print(f'Action: {enva.action(sample)}')

    #######
    enva = StateNormalization(enva)
    print(f'State [0.11, 0] is squished into:{enva.observation(np.array([0.11, 0]))}')
    print(f'State [5, 100] is squished into:{enva.observation(np.array([5, 100]))}')
    print(f'State [2, 50] is squished into:{enva.observation(np.array([2, 50]))}')

