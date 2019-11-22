import numpy as np
import scipy.optimize as opti

from abstract_chp import AbstractCHP


class ExpCHP(AbstractCHP):
    # Exponential Kernel
    # lambda_t = mu + sum_{t<tau_i} [alpha * exp(-beta*(t-tau_i))]

    def __init__(self, mu=None, alpha=None, beta=None, control=None):
        AbstractCHP.__init__(self, mu, control)
        self.alpha = alpha
        self.beta = beta
        self.params = [mu, alpha, beta]
        self.loglikelihood = None

    def integrateintensity(self, lb, ub):
        # return super().integrateintensity(lb, ub)
        # This method serves for calculation of the process residuals and subsequent Q-Q check
        # Integrate the intensity between two arrivals ONLY!!! does not serve as general integration method
        # The second part is calculated as integral over kernel on interval ub-lb times hte value at the
        # starting point
        # --- this one works god knows why ---
        mu_part = (ub - lb) * self.mu
        second_part = self.alpha / self.beta * (1 - np.exp(-self.beta * (ub - lb))) \
                      * np.sum(np.exp(-self.beta * (lb - self.arrivals[self.arrivals <= lb])))
        return mu_part + second_part
        # attempt for a rigorous function
        # mu_part = (ub - lb) * self.mu
        # second_part = self.alpha / self.beta * (np.sum(np.exp(-self.beta * lb) * self.arrivals[self.arrivals < lb])
        #                                         - np.sum(np.exp(-self.beta * ub) * self.arrivals[self.arrivals < ub]))
        # return mu_part + second_part


    def getkernelnorms(self):
        # Return Kernel Norm
        return self.alpha / self.beta

    def kernel(self, t):
        # Return Kernel evaluation at time t
        return np.multiply(self.alpha, np.exp(np.multiply(-self.beta, t)))

    def fit(self, taus):
        if self.ismonotonic(taus):
            self.arrivals = taus
        else:
            raise Exception('Arrival series is not monotonically increasing.')

        objfun = lambda x: self.mle(x, taus)
        guess = np.ones(len(self.params))
        bounds = opti.Bounds(lb=0 + 10e-6, ub=np.inf, keep_feasible=True)

        res = opti.minimize(objfun, guess, bounds=bounds)
        params = res.x
        self.mu = params[0]
        self.alpha = params[1]
        self.beta = params[2]
        self.params = params
        self.loglikelihood = res.fun
        return params

    def mle(self, params, arrivals):
        # params = [lambdainf,kappa,delta10] len = 4
        n = len(arrivals)
        runningsum = 0
        R_i = np.zeros(n)

        for i in range(0, n):
            if i != 0:
                R_i[i] = np.exp(-params[2] * (arrivals[i] - arrivals[i - 1])) * (params[1] + R_i[i - 1])
                pass
            runningsum = runningsum + np.log(
                params[0] + R_i[i])

        loglike = -(params[0] * arrivals[-1]) - np.sum(
            (params[1] / params[2]) * (1 - np.exp(-params[2] * (arrivals[-1] - arrivals)))) + runningsum
        return -  loglike

    def __repr__(self):
        return "HP with exponential kernel : mu = {0} , alpha = {1}, beta = {2},".format(self.mu, self.alpha,
                                                                                          self.beta)


if __name__ == '__main__':
    from plot_account import PlotAccount

    p1 = ExpCHP(0.3, 1, 2)
    p1.getpath(100)
    plt = PlotAccount(p1)
    plt.plotIntensity()
    plt.modelcheck()
