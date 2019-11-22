import pandas as pd
import numpy as np


class CollectionAccount:

    def __init__(self, starting_balance, starting_intensity, horizon):
        self.balance = starting_balance
        self.starting_intensity = starting_intensity
        self.horizon = horizon

    @staticmethod
    def rel_repayment(n_repayments):
        # relative repayment
        return np.random.rand(n_repayments)

    @staticmethod
    def poisson_pp(T, par=1):
        # poisson pp for repayments
        t = 0
        s = 0
        arrivals = np.array([])

        while s <= T:
            increment = np.random.exponential(1/par)
            s = s + increment
            arrivals = np.append(arrivals, s)

        if arrivals[-1]>T:
            arrivals = arrivals[:-1]

        return arrivals

    def get_history(self):
        arrivals = self.poisson_pp(self.horizon, self.starting_intensity)
        repayments = self.rel_repayment(len(arrivals))
        balances = self.get_balance(repayments)
        return arrivals, balances

    def get_balance(self, repayments):
        return self.balance * repayments.cumprod()

if __name__ == '__main__':
    intensity = 0.1
    T = 100
    w = 1000
    acc = CollectionAccount(w, intensity, T)
    print(acc.get_history())