import gym
import numpy as np
from gym.spaces import Box, Discrete
import pickle
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import BSpline
from itertools import product


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
    # TODO: normalization probably does not work
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


class SplineObservationWrapper(gym.ObservationWrapper):
    # converts state space to cubic spline basis features
    # n_knots + order - 1 basis functions
    # TODO: Normalization does not work
    def __init__(self, environment, n_l_knots, n_w_knots, normalized):
        super().__init__(environment)
        self.n_l_knots = n_l_knots
        self.n_w_knots = n_w_knots
        if normalized:
            high = [1, 1]
            low = [0, 0]
        else:
            high = self.observation_space.high
            low = self.observation_space.low
        self.w_knots = np.linspace(0, high[1], n_w_knots)
        self.l_knots = np.linspace(low[0] * 0.95, high[0] * 1.01, n_l_knots)
        self.total_features = (n_w_knots + 2) * (n_l_knots + 2)
        self.w_np_knots = np.concatenate((3 * [0], self.w_knots, 3 * [high[1]]))
        self.l_np_knots = np.concatenate((3 * [low[0] * 0.95], self.l_knots, 3 * [high[0] * 1.01]))

    def transform_1d_w(self, eval_point):
        # the first element of the transform is the original eval_point
        # hence +1 on the next line
        y_py = np.zeros((eval_point.shape[0], self.n_w_knots + 2 + 1))
        y_py[:, 0] = eval_point
        for i in range(self.n_w_knots + 2):
            y_py[:, i + 1] = BSpline.construct_fast(self.w_np_knots,
                                                    (np.arange(len(self.w_knots) + 2) == i).astype(float),
                                                    3,
                                                    extrapolate=False)(eval_point)
        return y_py

    def transform_1d_l(self, eval_point):
        # the first element of the transform is the original eval_point
        # hence +1 on the next line
        y_py = np.zeros((eval_point.shape[0], self.n_l_knots + 2 + 1))
        y_py[:, 0] = eval_point
        for i in range(self.n_l_knots + 2):
            y_py[:, i + 1] = BSpline.construct_fast(self.l_np_knots,
                                                    (np.arange(len(self.l_knots) + 2) == i).astype(float),
                                                    3,
                                                    extrapolate=False)(eval_point)
        return y_py

    def transform_2d(self, xy_inp):
        w_features = self.transform_1d_w(xy_inp[:, 1])
        l_features = self.transform_1d_l(xy_inp[:, 0])
        final = np.zeros((len(xy_inp), self.total_features + 2))
        # TODO: here I think it should be w_features[:,0]
        #    it seems other way around, balance should be first
        final[:, 0] = l_features[:, 0]
        final[:, 1] = w_features[:, 0]

        for r, row in enumerate(w_features):
            final[r, 2:] = [i * j for i, j in product(row[1:], l_features[r, 1:])]
        if np.isnan(final).any():
            print("NAN")
        return final

    def transform_1d_l_der(self, eval_point):
        y_first = np.zeros((eval_point.shape[0], self.n_l_knots + 2))
        for i in range(self.n_l_knots + 2):
            y_first[:, i] = BSpline.construct_fast(self.l_np_knots,
                                                   (np.arange(len(self.l_knots) + 2) == i).astype(float),
                                                   3,
                                                   extrapolate=False).derivative()(eval_point)
        return y_first

    def transform_1d_w_der(self, eval_point):
        # the first element of the transform is the original eval_point
        # hence +1 on the next line
        y_first = np.zeros((eval_point.shape[0], self.n_w_knots + 2))
        for i in range(self.n_w_knots + 2):
            y_first[:, i] = BSpline.construct_fast(self.w_np_knots,
                                                   (np.arange(len(self.w_knots) + 2) == i).astype(float),
                                                   3,
                                                   extrapolate=False).derivative()(eval_point)
        return y_first

    def penalization(self, lam, w):
        w_features = self.transform_1d_w(w)
        l_features = self.transform_1d_l(lam)
        w_features_der = self.transform_1d_w_der(w)
        l_features_der = self.transform_1d_l_der(lam)
        first_der_features = self.total_features
        first_w = np.zeros((len(w), first_der_features), dtype='float32')
        first_l = np.zeros_like(first_w)
        #     final[:, 0] = l_features[:, 0]
        #     final[:, 1] = w_features[:, 1]
        for r, row in enumerate(w_features):
            first_w[r, :] = [i * j for i, j in product(w_features_der[r, :], l_features[r, 1:])]
            first_l[r, :] = [i * j for i, j in product(w_features[r, 1:], l_features_der[r, :])]

        return first_w, first_l

    def observation(self, observation):
        if observation.ndim == 1:
            return self.transform_2d(observation[None, :]).flatten()
        else:
            return self.transform_2d(observation)

    def convert_back(self, transformed_observation):
        # untransformed observations are at position 1 and 2 -- position 0 is intercept
        if transformed_observation.ndim == 1:
            # was [:, 0:2]
            return transformed_observation[0:2]
        else:
            return transformed_observation[:, 0:2]

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
