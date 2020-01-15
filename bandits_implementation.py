import time
import numpy as np
from itertools import product
from functools import partial
from chp import CHP


class Bandit(object):

    def generate_reward(self, i):
        raise NotImplementedError


class CollectionBandit(Bandit):

    def __init__(self, chp, constraint, delta):
        self.chp = chp
        self.delta_b = delta[0]
        self.delta_m = delta[1]
        self.bmax = constraint[0]
        self.mmax = constraint[1]
        b = np.arange(0, self.bmax, self.delta_b)
        m = np.arange(0, self.mmax, self.delta_m)
        self.actions = list(product(b,m))
        self.nactions = len(self.actions)

    def generate_reward(self, i):
        # The player selected the i-th machine.
        control = partial(CollectionBandit.prep_linear_control, self.actions[i][0], self.actions[i][1])
        self.chp.control_function = control
        return -self.chp.calculate_value_single()

    @staticmethod
    def prep_linear_control(b, m, w):
        if isinstance(w, np.ndarray):
            return (m * (w - b)).clip(min=0)
        elif np.isscalar(w):
            return np.maximum(m * (w - b), 0)
        else:
            raise ValueError('Stupid error in w.')


class Solver(object):
    def __init__(self, bandit):
        """
        bandit (Bandit): the target bandit to solve.
        """

        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.counts = np.array([0] * self.bandit.nactions)
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        # Regret probably not necessery for this implementation as it remains unknown
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.
        self.bestestimate = []

    def update_regret(self, i):
        # i (int): index of the selected machine.
        # self.regret += self.bandit.best_proba - self.bandit.probas[i]
        # self.regrets.append(self.regret)
        raise NotImplementedError

    @property
    def estimated_probas(self):
        raise NotImplementedError

    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError

    def run(self, num_steps):
        assert self.bandit is not None
        for _ in range(num_steps):
            i = self.run_one_step()

            self.counts[i] += 1
            self.actions.append(i)
            # self.update_regret(i)


class EpsilonGreedy(Solver):
    def __init__(self, bandit, eps, init_value=0.0):
        """
        eps (float): the probability to explore at each time step.
        init_proba (float): default to be 1.0; optimistic initialization
        """
        super(EpsilonGreedy, self).__init__(bandit)

        assert 0. <= eps <= 1.0
        self.eps = eps

        self.estimates = [init_value] * self.bandit.nactions  # Optimistic initialization

    @property
    def estimated_values(self):
        return self.estimates

    def run_one_step(self):
        if np.random.random() < self.eps:
            # Let's do random exploration!
            i = np.random.randint(0, self.bandit.nactions)
        else:
            # Pick the best one.
            # i = max(range(self.bandit.n), key=lambda x: self.estimates[x])
            i = np.argmax(self.estimates)

        r = self.bandit.generate_reward(i)
        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])
        self.bestestimate.append(np.max(self.estimates))
        return i


class UCB1(Solver):
    def __init__(self, bandit, init_value=0.0, c=1):
        super(UCB1, self).__init__(bandit)
        self.t = 0
        self.estimates = np.array([init_value] * self.bandit.nactions, dtype=np.double)
        self.c = c

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        self.t += 1

        # Pick the best one with consideration of upper confidence bounds.
        # i = max(range(self.bandit.n), key=lambda x: self.estimates[x] + np.sqrt(
        #     2 * np.log(self.t) / (1 + self.counts[x])))
        i = np.argmax(self.estimates + self.c * np.sqrt(np.log(self.t)/(1 + self.counts)))
        r = self.bandit.generate_reward(i)

        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])
        self.bestestimate.append(np.max(self.estimates))

        return i

if __name__ == '__main__':
    chp = CHP(starting_balance=75, starting_intensity=1, marginal_cost=6,
              collection_horizon=10000, lambda_infty=0.1, kappa=0.7,
              delta10=0.02, delta11=0.5, control_function=None, rho=0.06, value_precision_thershold=0.01)
    bandit = CollectionBandit(chp,[75, 1], delta=[1, 0.005])
    print(bandit.actions)
    print(bandit.generate_reward(4))
    eps_solver = EpsilonGreedy(bandit, 0.03)
    ucb_solver = UCB1(bandit, init_value=50)
    ucb_solver.run(1000)
    print(ucb_solver.estimates)