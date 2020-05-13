import gym
from gym import spaces
import numpy as np
from dcc import Parameters
import copy
from collections_env import utils

MAX_ACCOUNT_BALANCE = 100.0
MIN_ACCONT_BALANCE = 1
MAX_ACTION = 0.3


class CollectionsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CollectionsEnv, self).__init__()

        # Environment specific
        self.params = Parameters()
        self.dt = 0.01
        self.w0 = MAX_ACCOUNT_BALANCE
        self.lambda0 = self.params.lambda0
        self.starting_state = np.array([self.lambda0, self.w0], dtype=np.float32)

        # GYM specific attributes
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([MAX_ACTION]), dtype=np.float16)
        self.MIN_ACCONT_BALANCE = MIN_ACCONT_BALANCE
        MAX_LAMBDA = utils.lambda_bound(MAX_ACCOUNT_BALANCE, MIN_ACCONT_BALANCE, self.params)
        self.observation_space = spaces.Box(low=np.array([self.params.lambdainf, self.MIN_ACCONT_BALANCE]),
                                            high=np.array([MAX_LAMBDA, MAX_ACCOUNT_BALANCE]),
                                            dtype=np.float16)
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # step dependent
        self.current_step = 0
        self.current_time = 0
        self.current_state = copy.deepcopy(self.starting_state)
        self.done = False
        self._draw = np.random.exponential(1)
        self.accumulated_under_intensity = 0
        self.arrivals = []
        self.repayments =[]

    def reset(self):
        self.current_state[:] = self.starting_state[:]
        self.done = False
        self.current_step = 0
        self.current_time = 0
        self.accumulated_under_intensity = 0
        self.arrivals = []
        self.repayments = []
        self._draw = np.random.exponential(1)
        # self.logger.info(f'State reset {self.starting_state}')
        return self.current_state

    def intensity_integral(self, t, lambda0):
        # Computes the integral of the intensity function
        return self.params.lambdainf * t + (lambda0 - self.params.lambdainf) * \
               (1 - np.exp(-self.params.kappa * t)) / self.params.kappa

    def step(self, action: float):
        self.current_step += 1
        self.current_time += self.dt
        self.current_state[0] += action * self.params.delta2
        discount_factor = np.exp(-self.params.rho * self.current_time)
        self.accumulated_under_intensity += self.intensity_integral(self.dt, self.current_state[0])
        if self.accumulated_under_intensity >= self._draw:
            r = self.params.sample_repayment()[0]
            self.current_state[0] += self.params.delta10 + self.params.delta11 * r
            self._draw = np.random.exponential(1)
            self.accumulated_under_intensity = 0
            self.arrivals.append(self.current_time)
            self.repayments.append(r)
        else:
            drifted = self.drift(self.dt, self.current_state[0])
            self.current_state[0] = drifted
            r = 0
        reward = (r * self.current_state[1] - action * self.params.c) * discount_factor
        self.current_state[1] = self.current_state[1] * (1 - r)

        if self.current_state[1] < self.MIN_ACCONT_BALANCE:
            self.done = True

        return self.current_state, reward, self.done, None

    def drift(self, s, lambda_start):
        # deterministic draft between jumps
        return self.params.lambdainf + (lambda_start - self.params.lambdainf) * np.exp(-self.params.kappa * s)

    def dphi(self, lambda_start, dt):
        # derivative of drift at 0
        return (self.params.lambdainf - lambda_start) / self.params.kappa * dt

    def render(self, mode='human', close=False):
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    env = CollectionsEnv()
    env.dt = 0.1
    nsteps = 30000
    x = np.linspace(0, nsteps * env.dt, nsteps)
    lambdas = []
    ws = []
    counter = 0
    for step in range(nsteps):
        ob, rew, done, _ = env.step(0)
        lambdas.append(ob[0])
        ws.append(ob[1])
        if rew > 0:
            print(rew)
            counter += 1

        # if counter > 50:
        #     break
    plt.plot(x, lambdas)
    # plt.plot(lambdas)
    plt.show()

    plt.plot(x, ws, marker='x')
    # plt.plot(ws)
    plt.show()
