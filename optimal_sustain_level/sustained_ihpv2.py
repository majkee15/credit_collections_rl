from base import Base
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import multiprocessing as mp
import copy


class SustainedIHP(Base):

    def __init__(self, starting_balance, starting_intensity, marginal_cost=1.0, collection_horizon=None,
                 lambda_infty=None, kappa=None, delta10=None, delta11=None, sustain_level=0.0,
                 value_precision_threshold=0.001, rho=0.05):
        super().__init__(__class__.__name__)
        self.starting_intensity = starting_intensity
        self.starting_balance = starting_balance
        self.marginal_cost = marginal_cost
        self.collection_horizon = collection_horizon
        self.lambda_infty = lambda_infty
        self.kappa = kappa
        self.delta10 = delta10
        self.delta11 = delta11
        self.rho = rho
        self.value_precision_threshold = value_precision_threshold
        self.sustain_level = sustain_level
        self.repayment_draw = partial(np.random.uniform, low=0.2, high=1.0, )
        self.logger.info(f'Instantiated sustain process @ ')
        # reset attributes
        # self.arrivals = np.array([])
        self.balances = np.array([self.starting_balance])
        # self.continuous_effort_duration = np.array([])
        # self.continuous_effort_start_sustain = np.array([])
        # self.costs = 0
        # self.reset_state()

    def reset_state(self):
        # self.arrivals = np.array([])
        self.balances = np.array([self.starting_balance])
        # self.continuous_effort_duration = np.array([])
        # self.continuous_effort_start_sustain = np.array([])
        # self.costs = 0
        # self.logger.info('State reset performed.')
        pass

    def drift(self, s, lambda_start):
        # deterministic drift
        return self.lambda_infty + (lambda_start - self.lambda_infty) * np.exp(-self.kappa * s)

    def controlled_intensity(self, s, lhat):
        # intensity of the ihp sustained at lhat
        lambda_start = self.starting_intensity
        control_level = lhat
        drifted = self.drift(s, lambda_start)
        return np.maximum(drifted, control_level)

    def theta(self, l, lhat):
        # equation for duration to reach a lambda^hat
        # lambda > lambda^hat
        try:
            numerator = lhat - self.lambda_infty
            denominator = l - self.lambda_infty

            if l < lhat:
                # junmp occurs first
                return 0.0
            elif denominator < 0:
                print('Denominator of the log in theta < 0.')
                return np.NaN()
            else:
                t = -1 / self.kappa * \
                    np.log(numerator / denominator)
                return t
        except Exception as e:
            msg = f'Numerator {(lhat - self.lambda_infty)} \n' \
                f'Denominator {l - self.lambda_infty}'
            self.logger.warning(msg)

    def next_arrival(self, lstart, lhat):
        # generates interarrival times
        flag_finished = False
        s = 0
        t_to_sustain = self.theta(lstart, lhat)
        jump_size = np.maximum(0, lhat - lstart)
        while not flag_finished:
            lstar = max(lhat, lstart)
            w = -np.log(np.random.rand()) / lstar
            s = s + w
            d = np.random.rand()
            potential_repayment = self.repayment_draw()
            intensity_at_jump_minus = self.controlled_intensity(s, lhat)
            if d * lstar <= intensity_at_jump_minus:
                # if t_to_sustain <= s:
                #     collected = balance * potential_repayment * np.exp(-self.rho * s)
                #     cost = jump_size * self.marginal_cost * np.exp(-self.rho * s)
                # else:
                #     collected = balance * potential_repayment * np.exp(-self.rho * s)
                #     cost = (np.exp(-self.rho * t_to_sustain) - np.exp(-self.rho * s))/self.rho * \
                #            (intensity_at_jump_minus - self.lambda_infty) * self.kappa
                sustain_drift_time = np.nanmax([0.0, s - t_to_sustain])
                collected = self.starting_balance * potential_repayment * np.exp(-self.rho * s)
                # sust_costs = np.divide(self.marginal_cost, self.rho) * (1-np.exp(-self.rho * sustain_drift_time))
                if jump_size > 0.0 or s > t_to_sustain:
                    sust_costs = self.kappa * (lhat - self.lambda_infty) * \
                        (np.exp(-self.rho * t_to_sustain) - np.exp(-self.rho * s))/self.rho
                else:
                    sust_costs = 0.0
                jump_cost = np.maximum(lhat - lstart, 0.0) * self.marginal_cost
                flag_finished = True
                return s, collected, sust_costs, jump_cost

    def calculate_value_single(self):

        s, collected, sust_costs, jump_cost = self.next_arrival(self.starting_intensity, self.sustain_level)
        revenue = -collected + sust_costs + jump_cost
        return revenue

    def calculate_value_mc(self, add_discount_time=0, mc_iteration=10000):
        running_sum_sust_costs = 0
        running_sum_jump_costs = 0
        running_sum_collected = 0
        running_sum_arrival = 0
        for i in range(mc_iteration):
            s, collected, sust_costs, jump_cost = self.next_arrival(self.starting_intensity,
                                                                    self.sustain_level)
            running_sum_sust_costs += sust_costs
            running_sum_jump_costs += jump_cost
            running_sum_collected += collected
            running_sum_arrival += s
        return running_sum_collected / mc_iteration, running_sum_sust_costs / mc_iteration, \
               running_sum_jump_costs / mc_iteration, running_sum_arrival / mc_iteration

    def plot_profile(self, lhatmin, lhatmax, points, it, add_discount_time=0, plotflag=(True, True, True)):
        lambda_hats = np.linspace(lhatmin, lhatmax, points)
        lambda_hats = np.insert(lambda_hats, 0, 0)
        sust_costsmc = np.zeros_like(lambda_hats)
        jump_costsmc = np.zeros_like(lambda_hats)
        collectedmc = np.zeros_like(lambda_hats)
        arrivalsmc = np.zeros_like(lambda_hats)
        accvalsmc = np.zeros_like(lambda_hats)
        for i, hat in enumerate(lambda_hats):
            sihp.sustain_level = hat
            collected, sust_cost, jump_cost, ar = sihp.calculate_value_mc(add_discount_time, mc_iteration=it)
            accvalsmc[i] = sust_cost + jump_cost - collected
            sust_costsmc[i] = sust_cost
            jump_costsmc[i] = jump_cost
            collectedmc[i] = -collected
            arrivalsmc[i] = ar

        lhatstar_index = np.nanargmin(accvalsmc)
        lhatstar = lambda_hats[lhatstar_index]
        arstar = arrivalsmc[lhatstar_index]

        if plotflag[0]:
            plt.figure()
            plt.plot(lambda_hats, accvalsmc, marker='x')
            plt.plot(lambda_hats, collectedmc, marker='x')
            plt.axvline(x=lhatstar, color='r')
            plt.xlabel('\hat{\lambda}')
            plt.ylabel('v')
            plt.legend(['Account value', 'Amount collected'])
            plt.title(f'Balance {self.starting_balance}.')
            plt.show()
        if plotflag[1]:
            plt.figure()
            plt.plot(lambda_hats, sust_costsmc, marker='x')
            plt.plot(lambda_hats, jump_costsmc, marker='x')
            plt.plot(lambda_hats, sust_costsmc + jump_costsmc, marker='x')
            plt.xlabel('\hat{\lambda}')
            plt.ylabel('costs')
            plt.legend(['Sustain costs', 'Jump costs', 'Total'])
            plt.show()
        if plotflag[2]:
            plt.figure()
            plt.plot(lambda_hats, arrivalsmc, marker='x')
            plt.xlabel('\hat{\lambda}')
            plt.ylabel('first arrival')
            plt.show()
            print(accvalsmc)
        return lhatstar, arstar


if __name__ == '__main__':
    lambda_hat = 0.11
    it = 100000
    w0 = 60
    sihp = SustainedIHP(starting_balance=w0, starting_intensity=1, marginal_cost=1.5,
                        collection_horizon=100, lambda_infty=0.1, kappa=0.7, value_precision_threshold=0.0,
                        delta10=0.02, delta11=0.5, sustain_level=lambda_hat, rho=0.20)
    # sihp.plot_profile(lambda_hat, 5.0, 50, it)
    ws = np.flip(np.linspace(0, w0, 50))
    lhatstars = np.zeros_like(ws)
    added_time = 0
    lmax  = 7
    for i, w in enumerate(ws):
        sihp.starting_balance = w
        print(w)
        print(added_time)

        lhatstars[i], arstar = sihp.plot_profile(lambda_hat, lmax, 30, it, add_discount_time=added_time,
                                                 plotflag=(True, False, False))
        lmax = lhatstars[i]
        added_time += arstar

    plt.figure()
    plt.plot(ws, lhatstars, marker='x')
    plt.xlabel('w')
    plt.ylabel('lhatstar')
    plt.ylim([0.0, 5.0])
    plt.show()
