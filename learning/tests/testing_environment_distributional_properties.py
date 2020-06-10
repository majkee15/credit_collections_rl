import numpy as np
import os
from scipy.integrate import quad
from dcc import Parameters, AAV
import matplotlib.pyplot as plt
from scipy.stats import kstest
import seaborn as sns
import logging

from joblib import Parallel, delayed
from multiprocessing import cpu_count


from learning.collections_env.collections_env import CollectionsEnv


class MotherSimulator():
    def __init__(self, dt, params):
        self.dt = dt
        self.params = params
        self.name = 'Abstract'
        self.logger = logging.getLogger('MotherSimulator logger')
        self.logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))

        self.env = CollectionsEnv()

    def kernel(self, t, r):
        # Return Kernel evaluation at time t
        return np.sum(np.multiply(self.params.delta10 + self.params.delta11 * r,
                                  np.exp(np.multiply(-self.params.kappa, t))))

    def getlambda(self, s, arrivals, reps):
        # Returns lambdas(either vector or a scalar) based on a realized process history arrivals
        # arrivals[arrivals <= s] cannot have a strict inequality because of the thinning algo
        # for straight inequalities the next candidate lstar is smaller than the actual intensity at the point!
        if np.isscalar(s):
            lamb = self.params.lambdainf + (self.params.lambda0 - self.params.lambdainf) * \
                   np.exp(-self.params.kappa * s) + \
                   np.sum(self.kernel(s - arrivals[arrivals <= s], reps[arrivals <= s]))
            return lamb
        else:
            n = len(s)
            lambdas = np.zeros(n)
            for i in range(0, n):
                lambdas[i] = self.params.lambdainf + (self.params.lambda0 - self.params.lambdainf) * \
                             np.exp(-self.params.kappa * s[i]) + \
                             np.sum(self.kernel(s[i] - arrivals[arrivals <= s[i]], reps[arrivals <= s[i]]))
            return lambdas

    def _timetransform(self, arrivals, repayments):
        thetas = np.zeros(len(arrivals) - 1)
        for i in range(1, len(arrivals)):
            thetas[i - 1] = quad(self.getlambda, arrivals[i - 1], arrivals[i], (arrivals, repayments))[0]
        return thetas

    def modelcheck(self, arrivals, repayments, verbose=False):
        # fig, ax = plt.subplots()
        residuals = self._timetransform(arrivals, repayments)
        # res = stats.probplot(thetas, dist=stats.expon, sparams=1, fit=False, plot=sns.mpl.pyplot)
        _, p_value = kstest(residuals, 'expon', args=(0, 1))
        if verbose:
            print(f'P-value: {p_value}')
        return p_value

    def drift(self, s, lambda_start):
        # deterministic draft between jumps
        return self.params.lambdainf + (lambda_start - self.params.lambdainf) * np.exp(-self.params.kappa * s)


    def getpath(self, horizon):
        state = self.env.reset()
        rew_path = []
        state_path = []
        t = 0
        done = False
        while not done:
            t += self.env.dt
            state, reward, done, _ = self.env.step(0)
            rew_path.append(reward)
            state_path.append(state[0])

        return np.sum(np.array(rew_path))

    def value_acc(self, w0, n=5000):
        vals = []
        horizon = -np.log(10e-16) / self.params.kappa
        for i in range(n):
            val = self.getpath(horizon)
            # res = np.cumprod(1 - repayments) * w0
            # vals.append(np.sum(- np.diff(np.insert(res, 0, w0)) * np.exp(-self.params.rho * arrivals)))
            vals.append(val)
        return vals

    def plot_value_dist(self, w0, n):
        aav = AAV(self.params)
        u = -aav.u(self.params.lambda0, w0)
        mc = self.value_acc(w0, n)
        fig, ax = plt.subplots()
        sns.distplot(mc, ax=ax)
        ax.axvline(np.mean(mc), color='red', linestyle='--')
        ax.axvline(u, color='green', linestyle='--')
        label = self.name + ' dt:' + str(self.dt)
        ax.set_title(label)
        fig.show()


class StepSizeEffect:

    def __init__(self, simulator, dts):
        self.dts = dts
        self.simulator = simulator

    def sim_value(self, w, n=5000):
        means = np.zeros_like(self.dts)
        stds = np.zeros_like(self.dts)

        for i, dt in enumerate(self.dts):
            vals = self.simulator.value_acc(w, n)
            means[i] = np.mean(vals)
            stds[i] = np.std(vals)

        aav = AAV(p)
        u = -aav.u(self.simulator.params.lambda0, w)
        fig, ax = plt.subplots()
        ax.plot(self.dts, means, marker='x')
        ax.fill_between(self.dts, means - stds, means + stds, alpha=0.5, color='salmon')
        ax.plot(self.dts, np.ones_like(self.dts) * u)
        fig.show()

    def sim_distributions(self, w, n):
        for i, dt in enumerate(self.dts):
            self.simulator.dt = dt
            self.simulator.plot_value_dist(w, n)

    def sim_pvalue(self, parallel_flag=False):
        pvals = np.zeros_like(self.dts)
        if parallel_flag:
            cores = cpu_count() - 1
            # Parallel(n_jobs=cores)(delayed(self.simulator)(i ** 2) for i in range(10))
            raise NotImplementedError('Parallel method is not supported for this method.')
        else:
            for i, dt in enumerate(self.dts):
                self.simulator.logger.info(f'Simulating paths for dt: {dt},')
                self.simulator.dt = dt
                ars, reps = self.simulator.getpath(1000)
                pvals[i] = self.simulator.modelcheck(ars, reps)



        fig, ax = plt.subplots()
        ax.plot(self.dts, pvals, marker='x')
        label = self.simulator.name
        ax.set_title(label)
        ax.set_ylim([0,1])
        fig.show()



if __name__ == '__main__':
    w0 = 100
    dt = 0.05
    params = Parameters()
    ms = MotherSimulator(dt, params)
    print(ms.plot_value_dist(w0, n=500))


