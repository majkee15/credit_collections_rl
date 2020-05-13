import numpy as np
import sys
from scipy.integrate import quad
from dcc import Parameters, AAV
import matplotlib.pyplot as plt
from scipy.stats import kstest
import seaborn as sns

from abc import ABC, abstractmethod

# Tests the performance of step algorithms
# Step algorithm performance on value of the account


class MotherSimulator(ABC):
    def __init__(self, params, dt):
        self.params = params
        self.dt = dt
        self.name = 'Abstract'

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


    @abstractmethod
    def getpath(self, horizon):
        pass

    def value_acc(self, w0, n=5000):
        vals = []
        horizon = -np.log(10e-16) / self.params.kappa
        for i in range(n):
            arrivals, repayments = self.getpath(horizon)
            res = np.cumprod(1 - repayments) * w0
            vals.append(np.sum(- np.diff(np.insert(res, 0, w0)) * np.exp(-self.params.rho * arrivals)))
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

class Thinning(MotherSimulator):
    # Defines account valuation through the Ogata's thinning method

    def __init__(self, params):
        dt = 1
        super().__init__(params, dt)
        self.name = 'Thinning'

    def getpath(self, horizon):
        # Simulation method based on Ogata's thinning algo
        arrivals = np.array([])
        repayments = np.array([])
        n = 0
        s = 0
        while s < horizon:
            lstar = self.getlambda(s, arrivals, repayments)
            w = -np.log(np.random.rand()) / lstar
            s = s + w
            d = np.random.rand()

            if d * lstar <= self.getlambda(s, arrivals, repayments):
                n += 1
                rep = self.params.sample_repayment()
                arrivals = np.append(arrivals, s)
                repayments = np.append(repayments, rep)


        return arrivals, repayments


class IntegralStep(MotherSimulator):
    def __init__(self, params, dt):
        super().__init__(params, dt)
        self.name = 'Integral'

    def intensity_integral(self, t, lambda0):
        # Computes the integral of the intensity function
        return self.params.lambdainf * t + (lambda0 - self.params.lambdainf) * \
               (1 - np.exp(-self.params.kappa * t)) / self.params.kappa

    def getpath(self, horizon):
        arrivals = []
        lambdas = [self.params.lambda0]
        repayments = []
        l0 = self.params.lambda0
        t = 0
        t_from_jump = 0
        draw = np.random.exponential(1)
        while t <= horizon:
            t += self.dt
            t_from_jump += self.dt
            if self.intensity_integral(t_from_jump, l0) >= draw:
                arrivals.append(t)
                r = self.params.sample_repayment()[0]
                repayments.append(r)
                l0 = lambdas[-1] + self.params.delta10 + self.params.delta11 * r
                lambdas.append(l0)
                draw = np.random.exponential(1)
                t_from_jump = 0
            else:
                lambdas.append(self.drift(t_from_jump, l0))

        arrivals = np.array(arrivals)
        repayments = np.array(repayments)
        return arrivals, repayments#, np.array(lambdas)


class NaiveStep(MotherSimulator):
    def __init__(self, params, dt):
        super().__init__(params, dt)
        self.name = 'Naive'

    def getpath(self, horizon):
        arrivals = []
        # lambdas = [self.params.lambda0]
        repayments = []
        l0 = self.params.lambda0
        t = 0
        t_from_jump = 0
        while t <= horizon:
            t += self.dt
            t_from_jump += self.dt
            draw = np.random.random()
            threshold = l0 * self.dt
            if draw <= threshold:
                arrivals.append(t)
                r = self.params.sample_repayment()[0]
                repayments.append(r)
                l0 = l0 + self.params.delta10 + self.params.delta11 * r
                t_from_jump = 0
            else:
                l0 = self.drift(self.dt, lambda_start=l0)
            #l ambdas.append(l0)
        arrivals = np.array(arrivals)
        repayments = np.array(repayments)
        # plt.plot(lambdas)
        # plt.show()
        return arrivals, repayments  # , np.array(lambdas)


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

    def sim_pvalue(self):
        pvals = np.zeros_like(self.dts)
        for i, dt in enumerate(self.dts):
            self.simulator.dt = dt
            ars, reps = self.simulator.getpath(10000)
            pvals[i] = self.simulator.modelcheck(ars, reps)

        fig, ax = plt.subplots()
        ax.plot(self.dts, pvals, marker='x')
        fig.show()



if __name__ == '__main__':
    p = Parameters()
    w = 100
    # # Check the full implementations
    #
    chp = Thinning(p)
    arrivals, repayments = chp.getpath(10000)
    chp.modelcheck(arrivals, repayments, verbose=True)
    chp.plot_value_dist(100, n=10000)
    # #
    # # Check the integral implementations
    # #
    # chpi = IntegralStep(p, dt=0.01)
    # arrivals, repayments = chpi.getpath(10000)
    # chpi.modelcheck(arrivals, repayments, verbose=True)
    # chpi.plot_value_dist(100, n=20000)

    # # Check the naive step
    # chpn = NaiveStep(p, dt=0.01)
    # arrivals, repayments = chpn.getpath(1000)
    # chpn.modelcheck(arrivals, repayments, verbose=True)
    # chpn.plot_value_dist(100, n=1000)


    # Check the effect of stepsize
    #
    # dts = np.logspace(-4, 0, 4)
    # dts = np.linspace(0.005, 0.4, 10)
    # sim = StepSizeEffect(chpn, dts)
    # sim.sim_value(w)
    # sim.sim_pvalue()

   #  sim.sim_distributions(w, n=5000)



