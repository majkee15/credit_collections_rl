import gym
import numpy as np


# Discretized observation wrapper
class DiscretizedObservationWrapper(gym.ObservationWrapper):
    """
    This wrapper converts a Box observation into a single integer.
    """
    def __init__(self, env, n_bins=(40, 25), low_obs=None, high_obs=None, log=False):
        super().__init__(env)
        # Check whether the env is already discrete
        assert isinstance(env.observation_space, gym.spaces.Box)

        low_obs = self.observation_space.low if low_obs is None else low_obs
        high_obs = self.observation_space.high if high_obs is None else high_obs

        self.n_obs_bins = n_bins
        if log:
            self.obs_val_bins = [np.linspace(low_obs[0], high_obs[0], n_bins[0]),
                            np.logspace(0, 20, base=(1-env.params.r_), num=n_bins[1]) * high_obs[1]]
        else:
            self.obs_val_bins = [np.linspace(h[0], h[1], n_bins[i]) for i, h in
                                 enumerate(zip(low_obs.flatten(), high_obs.flatten()))]

        self.observation_space = gym.spaces.Discrete(np.product([bin.shape[0] for bin in self.obs_val_bins]))

    def _cell_to_ind(self, digits):
        return digits[0] * self.n_obs_bins[1] + digits[1]

    def _ind_to_cell(self, number):
        i = int(number / (self.n_obs_bins[1]))
        j = int(number % (self.n_obs_bins[1]))
        return i, j

    def observation(self, observation):
        # converts continuous observation to comply with discretized bin
        digits = [np.digitize([x], bins, right=True)[0]
                  for x, bins in zip(observation.flatten(), self.obs_val_bins)]
        digits = [np.minimum(digit, self.n_obs_bins[i]-1) for i, digit in enumerate(digits)]
        return self._cell_to_ind(digits)

    def _ind_to_cont(self, ind):
        i, j = self._ind_to_cell(ind)
        return self.obs_val_bins[0][i], self.obs_val_bins[1][j]


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
        self.action_val_bins = [np.linspace(l, h, n_bins) for l, h in
                                zip(low.flatten(), high.flatten())]
        self.action_space = gym.spaces.Discrete(self.action_val_bins[0].shape[0])
        ## Not needed


    def _convert_to_one_number(self, digits):
        pass

    def _aind_to_cell(self, ind):
        return self.action_val_bins[0][ind]

    def action(self, action):
        # digits = [np.digitize([x], bins)[0]
        #           for x, bins in zip(action.flatten(), self.val_bins)]
        # digits = np.digitize(action, self.action_val_bins[0], right=True)
        return self.action_val_bins[0][action]
