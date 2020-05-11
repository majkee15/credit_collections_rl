import numpy as np
from dcc import Parameters
from scipy.integrate import quad
from matplotlib import gridspec
from scipy import stats
import seaborn as sns


class StepSim():

    def __init__(self, params, dt=0.01):
        self.params = params
        self.dt = dt
        self.lambda0 = 1

    def naive_arrival(self, lambda0):
        threshold = lambda0 * self.dt
        draw = np.random.random()
        arrival = False
        if draw <= threshold:
            arrival = True
        return arrival

    def integral_arrival(self, lambda0):
        # arrival whenever running area under intensity reaches 1
        # TODO: IMPLEMENT :-P
        w = -np.log(np.random.rand()) / lambda0
        threshold = lambda0 * self.dt
        draw = np.random.random()
        arrival = False
        if draw <= threshold:
            arrival = True
        return arrival

    def intensity_integral(self, t, lambda0):
        return self.params.lambdainf * t + (lambda0 - self.params.lambdainf) * \
               (1 - np.exp(-self.params.kappa * t)) / self.params.kappa

    def simulate_from_integral(self, T=10):
        arrivals = []
        lambdas = [self.lambda0]
        repayments = []
        l0 = self.lambda0
        t = 0
        t_from_jump = 0
        draw = np.random.exponential(1)
        while t <= T:
            t += dt
            t_from_jump += dt
            if self.intensity_integral(t_from_jump, l0) >= draw:
                arrivals.append(t)
                r = 1
                repayments.append(r)
                l0 = self.params.delta10 + self.params.delta11 * r
                lambdas.append(l0)
                draw = np.random.exponential()
                t_from_jump = 0
            else:
                lambdas.append(self.drift(t_from_jump, l0))

        self.arrivals = np.array(arrivals)
        return self.arrivals, np.array(lambdas)

    def simulate(self, T=10):
        t = 0
        arrivals = []
        lambdas = [self.lambda0]
        repayments = []
        l0 = self.lambda0
        while t <= T:
            if self.naive_arrival(l0):
                arrivals.append(t)
                r = 1
                repayments.append(r)
                l0 = l0 + self.params.delta10 + self.params.delta11 * r
            else:
                l0 = self.drift(self.dt, l0)
            lambdas.append(l0)
            t += dt

        self.arrivals = np.array(arrivals)
        return self.arrivals, np.array(lambdas)

    def expected_lambda(self, t):
        rmean = 1
        n = self.params.delta10 + self.params.delta11 * rmean
        if np.isscalar(t):
            return self.drift(t, self.lambda0)/(1-n)
        else:
            return np.array([self.drift(ti, self.lambda0)/(1-n) for ti in t])

    def drift(self, s, lambda_start):
        # deterministic draft between jumps
        return self.params.lambdainf + (lambda_start - self.params.lambdainf) * np.exp(-self.params.kappa * s)

    def kernel(self, t):
        # Return Kernel evaluation at time t
        r = 1
        return np.multiply(self.params.delta10 + self.params.delta11 * r, np.exp(np.multiply(-self.params.kappa, t)))

    def getlambda(self, s, arrivals):
        # Returns lambdas(either vector or a scalar) based on a realized process history arrivals
        # arrivals[arrivals <= s] cannot have a strict inequality because of the thinning algo
        # for straight inequalities the next candidate lstar is smaller than the actual intensity at the point!
        if np.isscalar(s):
            lamb = self.params.lambdainf + (self.lambda0 - self.params.lambdainf) * np.exp(-self.params.kappa * s) + sum(self.kernel(s - arrivals[arrivals <= s]))
            return lamb
        else:
            n = len(s)
            lambdas = np.zeros(n)
            for i in range(0, n):
                lambdas[i] = self.params.lambdainf + (self.lambda0 - self.params.lambdainf) * np.exp(-self.params.kappa * s) \
                             + sum(self.kernel(s[i] - arrivals[arrivals <= s[i]]))
            return lambdas

    def integrateintensity(self, lb, ub):
        return quad(self.getlambda, lb, ub, self.arrivals)[0]

    def _timetransform(self):
        thetas = np.zeros(len(self.arrivals) - 1)
        for i in range(1, len(self.arrivals)):
            thetas[i - 1] = self.integrateintensity(self.arrivals[i - 1], self.arrivals[i])
        return thetas

    def modelcheck(self):
        # thetas = np.empty(len(self.processes))
        counter = 0
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 1, wspace=0.5, hspace=0.5)
        thetas = self._timetransform()
        ax = fig.add_subplot(gs[counter])
        # res = stats.probplot(thetas, dist=stats.expon, sparams=1, fit=False, plot=sns.mpl.pyplot)
        testsuit = np.random.exponential(1, len(thetas) * 2)
        self._qqplot(testsuit, thetas)
        plt.show()
        return stats.ks_2samp(testsuit, thetas)

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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    T = 100
    p = Parameters()
    dt = 0.1
    sim = StepSim(p, dt)
    arrivals, lambdas = sim.simulate_from_integral(T)
    print(arrivals)
    # plt.plot(lambdas)
    # plt.show()

    x = np.linspace(0, T, len(lambdas))
    plt.plot(x, lambdas)
    theoretical_means = sim.expected_lambda(x)
    plt.plot(x, theoretical_means)
    plt.show()
    print(sim.modelcheck())

    spacing = np.linspace(0, T, 10)
    sample_means = np.zeros_like(spacing)
    for i, ti in enumerate(spacing):
        sample_means[i] = np.mean(lambdas[x<ti])
    fig, ax = plt.subplots()
    ax.plot(spacing, sample_means, marker='x')
    ax.plot(x, theoretical_means)
    fig.show()
