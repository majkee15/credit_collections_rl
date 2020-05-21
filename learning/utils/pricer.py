import numpy as np
from dcc import AAV, Parameters
from scipy.integrate import quad
from scipy.stats import kstest
import matplotlib.pyplot as plt
import seaborn as sns


class MCPricer:
    def __init__(self, params, threshold):
        self.params = params
        self.threshold = threshold

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

    def getpath(self, w):
        remaining = w
        arrivals = np.array([])
        repayments = np.array([])
        n = 0
        s = 0
        while True:
            lstar = self.getlambda(s, arrivals, repayments)
            w = -np.log(np.random.rand()) / lstar
            s = s + w
            d = np.random.rand()

            if d * lstar <= self.getlambda(s, arrivals, repayments):
                n += 1
                rep = self.params.sample_repayment()
                arrivals = np.append(arrivals, s)
                repayments = np.append(repayments, rep)
                remaining = remaining * (1-rep)

            if remaining <= self.threshold:
                break

        return arrivals, repayments

    def value_acc(self, w0, n=5000):
        vals = []
        horizon = -np.log(10e-16) / self.params.kappa
        for i in range(n):
            arrivals, repayments = self.getpath(w0)
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
        fig.show()

    def price(self, l0, w0,  n=5000):
        # servers only for single calls
        self.params.lambda0 = l0
        return np.mean(self.value_acc(w0, n))

if __name__ == '__main__':
    w0 = 100
    cutoff = 10
    n = 1000
    p = Parameters()
    pricer = MCPricer(p, cutoff)
    pricer.getpath(100)
    pricer.plot_value_dist(w0, n)
    print(pricer.price(w0))
