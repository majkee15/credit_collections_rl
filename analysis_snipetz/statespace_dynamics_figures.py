import os
import numpy as np
import matplotlib.pyplot as plt
from dcc import Parameters


class SimpleDeterministicPath:
    def __init__(self, params, arrivals, repayments, action_times, actions_sizes):
        self.w0 = 1000
        self.params = params
        self.arrivals = arrivals
        self.repayments = repayments
        self.action_times = action_times
        self.action_sizes = actions_sizes
        self.fig_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                     '..', 'figs', 'sample_path'))
        os.makedirs(self.fig_path, exist_ok=True)

    def intensity(self, t):
        drift = self.params.lambdainf + np.exp(- self.params.kappa * t) * (self.params.lambda0 - self.params.lambdainf)
        jumps = np.sum((self.params.delta10 + self.repayments[self.arrivals < t] * self.params.delta2) *
                       np.exp(- self.params.kappa * (t - self.arrivals[self.arrivals < t])))
        actions = np.sum(np.exp(- self.params.kappa * (t - self.action_times[self.action_times < t])) *
                         self.action_sizes[action_times < t])
        return drift + jumps + actions

    def balance(self, t):
        balances = self.w0 * np.product((1 - self.repayments[self.arrivals < t]))
        return balances

    def plot_intensity(self, t_grid, save=False):
        ls = np.zeros_like(t_grid)
        for i, t in enumerate(t_grid):
            try:
                ls[i] = self.intensity(t)
            except:
                print('hulipero')
                self.intensity(t)

        fig, ax = plt.subplots()
        ax.plot(t_grid, ls)
        ax.axhline(self.params.lambda0, linestyle='--', color='red')
        ax.set_xlabel('Time')
        ax.set_ylabel('Intensity')
        fig.show()
        if save:
            name = os.path.join(self.fig_path, 'time-intensity.eps')
            fig.savefig(name, format='eps')

    def plot_balance(self, t_grid, save=False):
        ws = np.zeros_like(t_grid)
        for i, t in enumerate(t_grid):
            ws[i] = self.balance(t)

        fig, ax = plt.subplots()
        ax.plot(t_grid, ws)
        ax.set_ylim([0, self.w0 + self.w0 * 0.1])
        ax.set_xlabel('Time')
        ax.set_ylabel('Balance')
        fig.show()
        if save:
            name = os.path.join(self.fig_path, 'time-balance.eps')
            fig.savefig(name, format='eps')

    def plot_statespace_movement(self, t_grid, save=False):
        ws = np.zeros_like(t_grid)
        ls = np.zeros_like(t_grid)
        for i, t in enumerate(t_grid):
            ws[i] = self.balance(t)
            ls[i] = self.intensity(t)
        fig, ax = plt.subplots()
        ax.plot(ws, ls)
        ax.axhline(self.params.lambda0, linestyle='--', color='red')
        ax.set_xlabel('Balance')
        ax.set_ylabel('Intensity')
        ax.set_xlim([0, self.w0])
        fig.show()
        if save:
            name = os.path.join(self.fig_path, 'balance-intensity.eps')
            fig.savefig(name, format='eps')

    def plot_actions(self, t_grid, save=False):
        ls = np.zeros_like(t_grid)
        at = np.zeros_like(t_grid)
        for i, t in enumerate(t_grid):
            ls[i] = np.sum(np.exp(- self.params.kappa * (t - self.action_times[self.action_times < t])) *
                           self.action_sizes[action_times < t])
            at[i] = np.sum(self.action_sizes[action_times < t])
        fig, ax = plt.subplots()
        ax.plot(t_grid, ls)
        ax.plot(t_grid, at)
        ax.set_ylim([0, np.max(np.max(at) * (1.05))])
        ax.set_xlabel('Time')
        ax.set_ylabel('Intensity')
        fig.show()
        if save:
            name = os.path.join(self.fig_path, 'time-action.eps')
            fig.savefig(name, format='eps')


if __name__ == '__main__':
    arrivals = np.array([2, 3, 5])
    repayments = np.array([0.3, 0.7, 0.2])
    action_times = np.array([0.1, 7.8])
    action_sizes = np.array([2.0, 1.0])
    p = Parameters()

    sdp = SimpleDeterministicPath(p, arrivals, repayments, action_times, action_sizes)

    t_mesh = np.linspace(0, 10, 500)

    print(sdp.fig_path)

    sdp.plot_intensity(t_mesh, True)
    sdp.plot_balance(t_mesh, True)
    sdp.plot_statespace_movement(t_mesh, True)
    sdp.plot_actions(t_mesh, True)
