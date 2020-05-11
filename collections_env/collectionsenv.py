from dcc import Parameters, AAV
from base import Base
import numpy as np
import matplotlib.pyplot as plt
import copy


class CollectionsEnv(Base):

    def __init__(self, starting_state, params, dt=0.1):
        super().__init__(__class__.__name__)
        self.params = params
        self.current_step = 0
        self.current_time = 0
        self.action_space = None
        self.starting_state = np.array([starting_state[0], starting_state[1]])
        self.current_state = copy.deepcopy(starting_state)
        self.dt = dt
        self.done = False
        self.logger.info(f'Instantiated @ {self.__class__.__name__}')

    def reset(self):
        self.current_state = np.array([1, 100], dtype=np.float32)
        self.done = False
        self.current_step = 0
        self.current_time = 0
        # self.logger.info(f'State reset {self.starting_state}')
        return self.current_state

    def step(self, action):
        # Arrival happens
        self.current_step += 1
        self.current_time += self.dt
        self.current_state[0] += action * self.params.delta2
        # self.logger.info(f'{self.current_state}')
        discount_factor = np.exp(-self.params.rho * self.current_time)
        threshold = self.current_state[0] * self.dt
        draw = np.random.random()
        if threshold >= draw:
            # self.logger.info('JUMP')
            r = self.params.sample_repayment()
            self.current_state[0] = self.current_state[0] + self.params.delta10 + self.params.delta11 * r
        else:
            drifted = self.drift(self.dt, self.current_state[0])
            self.current_state[0] = drifted
            r = 0

        rew = (r * self.current_state[1] - action * self.params.c) * discount_factor
        self.current_state[1] = self.current_state[1] * (1 - r)
        # self.logger.info(f'State: {self.current_state}, Threshold: {threshold}, Draw: {draw}, R: {r}, '
        #                  f'DC: {discount_factor}')

        if self.current_state[1] < 0.01:
            self.done = True

        return self.current_state, rew, self.done

    def drift(self, s, lambda_start):
        # deterministic draft between jumps
        return self.params.lambdainf + (lambda_start - self.params.lambdainf) * np.exp(-self.params.kappa * s)

    def dphi(self, lambda_start, dt):
        return (self.params.lambdainf - lambda_start) / self.params.kappa * dt



if __name__ == '__main__':

    lambda0 = 1
    w0 = 100
    dt = 0.01
    s0 = np.array([lambda0, w0], dtype=np.float32)
    p = Parameters()
    env = CollectionsEnv(s0, p, dt)
    reward_per_episode = []
    n_epochs = 500
    for epoch in range(n_epochs):
        # print(f'start in state: {env.current_state}')
        env.reset()
        # print(f'reset in state: {env.current_state}')
        tot_reward_per_episode = 0
        lambdas = []
        ws = []
        # print(f'Start state: {env.starting_state}')
        while True:
            action = 0
            state_next, reward, terminal = env.step(action)
            # print(f'Start state: {env.starting_state}')
            ws.append(state_next[1])
            tot_reward_per_episode += reward
            lambdas.append(state_next[0])
            if terminal:
                reward_per_episode.append(tot_reward_per_episode)
                break


    # plt.plot(ws, lambdas, marker='x')
    plt.show()
    plt.plot(reward_per_episode, marker='x')
    plt.show()
    aav = AAV(p)
    print(aav.u(lambda0, w0))
    print(np.mean(reward_per_episode))
