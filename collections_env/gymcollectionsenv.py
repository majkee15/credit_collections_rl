import gym
from gym import spaces
import numpy as np
from dcc import Parameters
import copy

MAX_ACCOUNT_BALANCE = 100.0
MIN_ACCONT_BALANCE = 0.5
MAX_ACTION = 0.3
MAX_LAMBDA = 5

class CollectionsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CollectionsEnv, self).__init__()
        self.params = Parameters()
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.action_space = spaces.Box(
            low=np.array([0]), high=np.array([MAX_ACTION]), dtype=np.float16)
        self.MIN_ACCONT_BALANCE = MIN_ACCONT_BALANCE
        self.observation_space = spaces.Box(low=np.array([self.params.lambdainf, self.MIN_ACCONT_BALANCE]), high=np.array([MAX_LAMBDA, MAX_ACCOUNT_BALANCE]),
                                            dtype=np.float16)

        self.current_step = 0
        self.current_time = 0
        self.t_from_jump = 0
        self.lambda0 = 1.0
        self.last_lam0 = self.lambda0
        self.w0 = MAX_ACCOUNT_BALANCE
        self.starting_state = np.array([self.lambda0, MAX_ACCOUNT_BALANCE], dtype=np.float32)
        self.current_state = copy.deepcopy(self.starting_state)
        self.dt = 0.01
        self.done = False

    def intensity_integral(self, t, lambda0):
        return self.params.lambdainf * t + (lambda0 - self.params.lambdainf) * \
            (1 - np.exp(-self.params.kappa * t)) / self.params.kappa
    #
    # def step(self, action):
    #     self.current_step += 1
    #     self.current_time += self.dt
    #     self.t_from_jump += self.dt
    #     self.current_state[0] += action * self.params.delta2
    #     discount_factor = np.exp(-self.params.rho * self.current_time)
    #     self._draw = np.random.exponential(1)
    #     threshold = self.intensity_integral(self.t_from_jump, self.last_lam0)
    #     if threshold >= self._draw:
    #         r = self.params.sample_repayment()
    #         self.last_lam0 = self.params.delta10 + self.params.delta11 * r
    #         self.current_state[0] = self.last_lam0
    #         self._draw = np.random.exponential()
    #         self.t_from_jump = 0
    #     else:
    #         drifted = self.drift(self.t_from_jump, self.last_lam0)
    #         self.current_state[0] = drifted
    #         r = 0
    #     rew = (r * self.current_state[1] - action * self.params.c) * discount_factor
    #     self.current_state[1] = self.current_state[1] * (1 - r)
    #     if self.current_state[1] < self.MIN_ACCONT_BALANCE:
    #         self.done = True
    #         self.current_state[1] = 0
    #
    #     return self.current_state, rew, self.done, None

    # OLD STEP
    def step(self, action):
        # TODO: DEFINE NEW STEP
        # Arrival happens
        self.current_step += 1
        self.current_time += self.dt
        self.current_state[0] += action * self.params.delta2
        discount_factor = np.exp(-self.params.rho * self.current_time)
        threshold = self.current_state[0] * self.dt
        draw = np.random.random()
        if threshold >= draw:
            # self.logger.info('JUMP')
            r = self.params.sample_repayment()[0]
            self.current_state[0] = self.drift(self.dt, self.current_state[0]) + self.params.delta10 + self.params.delta11 * r
        else:
            drifted = self.drift(self.dt, self.current_state[0])
            self.current_state[0] = drifted
            r = 0

        rew = (r * self.current_state[1] - action * self.params.c) * discount_factor
        self.current_state[1] = self.current_state[1] * (1 - r)
        # self.logger.info(f'State: {self.current_state}, Threshold: {threshold}, Draw: {draw}, R: {r}, '
        #                  f'DC: {discount_factor}')

        if self.current_state[1] < self.MIN_ACCONT_BALANCE:
            self.done = True
            self.current_state[1] = 0

        return self.current_state, rew, self.done, None

    def drift(self, s, lambda_start):
        # deterministic draft between jumps
        return self.params.lambdainf + (lambda_start - self.params.lambdainf) * np.exp(-self.params.kappa * s)

    def dphi(self, lambda_start, dt):
        return (self.params.lambdainf - lambda_start) / self.params.kappa * dt

    def reset(self):
        self.current_state[:] = self.starting_state[:]
        self.done = False
        self.current_step = 0
        self.current_time = 0
        # self.logger.info(f'State reset {self.starting_state}')
        return self.current_state

    def render(self, mode='human', close=False):
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    env = CollectionsEnv()
    nsteps = 10000
    x = np.linspace(0,nsteps * env.dt, nsteps)
    lambdas = []
    for step in range(nsteps):
        ob, rew, done, _ = env.step(0)
        lambdas.append(ob[0])
    plt.plot(x, lambdas)
    plt.show()
