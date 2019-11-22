from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt


# TODO Marked version of the process
# TODO move as much of the fit method to the parametric learner
# \lambda_t^i = \mu^i + \sum_j=1^n \sum_{\tau_i<t} g^ij (t-\tau_i}
class AbstractCHP(ABC):
    # Base class for all parametric learners
    def __init__(self, mu, control, *args, **kwargs):
        # Constructor Parameters
        # arrivals - timestamps of the arrivals
        self.mu = float(mu)
        self.arrivals = np.array([])
        self.horizon = 0
        self.control = control

    def getpath(self, horizon):
        # Simulation method based on Ogata's thinning algo
        self.horizon = horizon
        arrivals = np.array([])
        n = 0
        s = 0
        while s < horizon:
            lstar = self.getlambda(s, arrivals)
            w = -np.log(np.random.rand()) / lstar
            s = s + w
            d = np.random.rand()

            if d * lstar <= self.getlambda(s, arrivals):
                n += 1
                arrivals = np.append(arrivals, s)
        self.arrivals = arrivals
        return arrivals

    def getlambda(self, s, arrivals):
        # Returns lambdas(either vector or a scalar) based on a realized process history arrivals
        # arrivals[arrivals <= s] cannot have a strict inequality because of the thinning algo
        # for straight inequalities the next candidate lstar is smaller than the actual intensity at the point!
        if np.isscalar(s):
            lamb = self.mu + sum(self.kernel(s - arrivals[arrivals <= s]))
            return lamb
        else:
            n = len(s)
            lambdas = np.zeros(n)
            for i in range(0, n):
                lambdas[i] = self.mu + sum(self.kernel(s[i] - arrivals[arrivals <= s[i]]))
            return lambdas

    def ismonotonic(self, arrivals):
        # Checks is arrivals is monotonically increasing
        return np.all(np.diff(arrivals) > 0)

    def modelcheck(self, show=False):
        # returns 2-sided ks test of Null hypothesis testsuit and thetas are drawn from the same distribution exp(1)
        # return[0] - statistic
        # return[1] - p-value
        try:
            thetas = self._timetransform()
            # alternatively use this
            # testsuite = np.random.exponential(1, len(thetas))
            # stats.ks_2samp(testsuite, thetas)
            if show == True:
                k = stats.probplot(thetas, dist=stats.expon, fit=True, plot=plt, rvalue=False)
                plt.plot(k[0][0], k[0][0], 'k--')
                plt.show()
            return stats.kstest(thetas, 'expon')
        except:
            return [None, None]

    #@abstractmethod
    def integrateintensity(self, lb, ub):
        return integrate.quad(self.getlambda, lb, ub, self.arrivals)[0]

    def getkernelnorms(self):
        pass

    @abstractmethod
    def kernel(self, t):
        pass

    @abstractmethod
    def mle(self, params, arrivals):
        pass

    @abstractmethod
    def fit(self, arrivals):
        pass

    # SEXPERIMENTAL
    # For numerical Fit - Parameter Dependent
    @classmethod
    def _getlambda(cls, s, tau, params):
        # Returns lambdas(either vector or a scalar) based on a realized history tau
        if np.isscalar(s):
            lamb = params[0] + sum(cls._kernel(s - tau[tau <= s], params))
            return lamb
        else:
            n = len(s)
            lambdas = np.zeros(n)
            for i in range(0, n):
                lambdas[i] = params[0] + sum(cls._kernel(s[i] - tau[tau <= s[i]], params))
            return lambdas

    def _kernel(self, t, params):
        pass

    def _qqplot(self, theoretical, empirical):
        # fig = plt.figure()
        percs = np.linspace(0, 100, 201)
        qn_a = np.percentile(theoretical, percs)
        qn_b = np.percentile(empirical, percs)

        plt.plot(qn_a, qn_b, ls="", marker="o")

        x = np.linspace(np.min((qn_a.min(), qn_b.min())), np.max((qn_a.max(), qn_b.max())))
        plt.plot(x, x, color="k", ls="--")

        # fig.suptitle('Q-Q plot', fontsize=20)
        plt.xlabel('Theoretical Quantiles', fontsize=18)
        plt.ylabel('Empirical Quantiles', fontsize=16)
        return plt

    def _timetransform(self):
        thetas = np.zeros(len(self.arrivals) - 1)
        for i in range(1, len(self.arrivals)):
            thetas[i - 1] = self.integrateintensity(self.arrivals[i - 1], self.arrivals[i])
        return thetas
