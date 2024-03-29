import gym
from gym import spaces
import pickle
import numpy as np
from dcc import Parameters
import copy
from learning.collections_env import utils
from learning.collections_env.repayment_distribution import UniformRepayment, BetaRepayment

MAX_ACCOUNT_BALANCE = 200.0
MIN_ACCOUNT_BALANCE = 1.0
MAX_ACTION = 5.0


class CollectionsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, w0=MAX_ACCOUNT_BALANCE, params=Parameters(), repayment_dist=None, reward_shaping='discrete',
                 randomize_start=False, starting_state=None, max_lambda=None, xi=1.0):
        """
        :param w0:
        :param params:
        :param repayment_dist:
        :param reward_shaping:
        :param randomize_start:
        :param starting_state:
        :param max_lambda:
        :param xi: shaping parameter

        """
        super(CollectionsEnv, self).__init__()

        # Environment specific
        self.params = params
        self.dt = 0.05
        self.w0 = w0
        self.lambda0 = self.params.lambda0
        if starting_state is None:
            self.starting_state = np.array([self.lambda0 + 0.01, self.w0], dtype=np.float32)
        else:
            self.starting_state = starting_state
            assert isinstance(self.starting_state[0],  (np.floating, float, np.float64)), \
                "starting_state has to be FLOAT.'"

        # GYM specific attributes
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([MAX_ACTION]), dtype=np.float32)
        self.MIN_ACCOUNT_BALANCE = MIN_ACCOUNT_BALANCE
        self.MAX_ACTION = MAX_ACTION
        if max_lambda is None:
            self.MAX_LAMBDA = utils.lambda_bound(self.w0, MIN_ACCOUNT_BALANCE, self.params)
        else:
            self.MAX_LAMBDA = max_lambda
        self.observation_space = spaces.Box(low=np.array([self.params.lambdainf, self.MIN_ACCOUNT_BALANCE]),
                                            high=np.array([self.MAX_LAMBDA, self.w0]),
                                            dtype=np.float32)
        self.reward_range = (0, self.w0)

        # step dependent
        self.current_step = 0
        self.current_time = 0
        self.current_state = copy.deepcopy(self.starting_state)
        self.done = False
        self._draw = np.random.exponential(1)
        self.accumulated_under_intensity = 0
        self.arrivals = []
        self.repayments = []

        # Continuous reward
        self.reward_shaping = reward_shaping

        if reward_shaping == 'continuous':
            self.xi = 1.0
        elif reward_shaping == 'discrete':
            self.xi = 0.0
        elif reward_shaping == 'mixed':
            self.xi = xi
        else:
            raise NotImplementedError("Not implemented shaping method.")

        # randomize starts
        self.randomize_start = randomize_start

        if repayment_dist is None:
            self.repayment_dist = UniformRepayment(self.params)
        else:
            self.repayment_dist = repayment_dist

    @property
    def starting_state(self):
        return self.__starting_state

    @starting_state.setter
    def starting_state(self, starting_state):
        if isinstance(starting_state, np.ndarray):
            if starting_state.shape[0] == 2:
                self.__starting_state = starting_state
        else:
            raise TypeError(f"Cannot assign {starting_state} int starting state.")

    def reset(self, tostate=None):
        # TODO: this if false statement does not work as expected
        # when tostate supplied it does not work if randomize_start is true
        if tostate is not None and type(tostate) is np.ndarray:
            tostate = tostate.astype('float64')
            self.current_state = tostate.copy()

        elif tostate is None and self.randomize_start:
            draw_lambda = np.random.uniform(self.params.lambda0, self.MAX_LAMBDA)
            draw_w = np.random.uniform(MIN_ACCOUNT_BALANCE, self.w0)
            self.current_state = np.array([draw_lambda, draw_w])

        else:
            self.current_state = self.starting_state.copy()
        self.done = False
        self.current_step = 0
        self.current_time = 0
        self.accumulated_under_intensity = 0
        self.arrivals = []
        self.repayments = []
        self._draw = np.random.exponential(1)
        # self.logger.info(f'State reset {self.starting_state}')
        return self.current_state

    def observation(self, state):
        # only for inheritance purposes and compatibility with wrappers
        return state

    def convert_back(self, state):
        # only for inheritance purposes and compatibility with wrappers
        return state

    def intensity_integral(self, t, lambda0):
        # Computes the integral of the intensity function
        return self.params.lambdainf * t + (lambda0 - self.params.lambdainf) * \
               (1 - np.exp(-self.params.kappa * t)) / self.params.kappa

    def step(self, action: float):
        self.current_step += 1
        self.current_time += self.dt
        self.current_state[0] += action * self.params.delta2
        self.accumulated_under_intensity += self.intensity_integral(self.dt, self.current_state[0])
        if self.accumulated_under_intensity >= self._draw:
            r = self.repayment_dist.draw(self.current_state[1])
            self.current_state[0] += self.params.delta10 + self.params.delta11 * r
            self._draw = np.random.exponential(1)
            self.accumulated_under_intensity = 0
            self.arrivals.append(self.current_time)
            self.repayments.append(r)

        else:
            drifted = self.drift(self.dt, self.current_state[0])
            self.current_state[0] = drifted
            r = 0

        # reward formulation
        if self.reward_shaping == 'continuous':
            reward = self.current_state[1] * self.repayment_dist.mean(self.current_state[1]) * self.current_state[
                0] * self.dt \
                     - action * self.params.c
            self.current_state[1] = self.current_state[1] - self.current_state[1] * \
                                    self.repayment_dist.mean(self.current_state[1]) * self.current_state[0] * self.dt
        elif self.reward_shaping == 'discrete':
            # sparse reward formulation
            reward = (r * self.current_state[1] - action * self.params.c)
            self.current_state[1] = self.current_state[1] * (1 - r)

        elif self.reward_shaping == 'mixed':
            reward = self.xi * self.current_state[1] * self.repayment_dist.mean(self.current_state[1]) * \
                     self.current_state[0] * self.dt + (1 - self.xi) * r * self.current_state[
                         1] - action * self.params.c

        else:
            raise NotImplementedError('Not implemented rewards.')

        if self.current_state[1] < self.MIN_ACCOUNT_BALANCE:
            self.done = True

        # snap back to the highest allowed point
        if self.current_state[0] > self.MAX_LAMBDA:
            self.current_state[0] = self.MAX_LAMBDA

        return self.current_state, reward, self.done, None

    def drift(self, s, lambda_start):
        # deterministic draft between jumps
        return self.params.lambdainf + (lambda_start - self.params.lambdainf) * np.exp(-self.params.kappa * s)

    def dphi(self, lambda_start, dt):
        # derivative of drift at 0
        return (self.params.lambdainf - lambda_start) / self.params.kappa * dt

    def render(self, mode='human', close=False):
        pass

    def reward(self, r, action, dc):
        (r * self.current_state[1] - action * self.params.c) * dc
        raise NotImplementedError("Not implemented method `reward`.")

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    env = CollectionsEnv(randomize_start=True)
    env.dt = 0.1
    nsteps = 30000
    x = np.linspace(0, nsteps * env.dt, nsteps)
    lambdas = []
    ws = []
    counter = 0
    cum_rew = 0
    for step in range(nsteps):
        ob, rew, done, _ = env.step(0)
        lambdas.append(ob[0])
        ws.append(ob[1])
        if rew > 0:
            # print(rew)
            counter += 1
            cum_rew += rew

        # if counter > 50:
        #     break
    plt.plot(x, lambdas)
    # plt.plot(lambdas)
    plt.show()

    plt.plot(x, ws, marker='x')
    # plt.plot(ws)
    plt.show()

    print(cum_rew)

    print(env.reset())
    print(env.reset())
    print(env.reset())
