import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy import stats
import seaborn as sns


#


class PlotAccount:

    def __init__(self, *args):
        # self.__dict__.update(kwargs)
        self.processes = args

    def modelcheck(self):
        # thetas = np.empty(len(self.processes))
        counter = 0
        fig = plt.figure()
        gs = gridspec.GridSpec(1, len(self.processes), wspace=0.5, hspace=0.5)
        for arg in self.processes:
            thetas = np.zeros(len(arg.arrivals))
            ()
            for i in range(1, len(arg.arrivals)):
                thetas[i - 1] = arg.integrateintensity(arg.arrivals[i - 1], arg.arrivals[i])

            ax = fig.add_subplot(gs[counter])
            # res = stats.probplot(thetas, dist=stats.expon, sparams=1, fit=False, plot=sns.mpl.pyplot)
            testsuit = np.random.exponential(1, len(thetas) * 2)
            self._qqplot(testsuit, thetas)
            print(arg)
            print(stats.ks_2samp(testsuit, thetas))
            counter = counter + 1
        plt.show

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

    def plotIntensity(self):
        for arg in self.processes:
            grid = np.arange(0, arg.arrivals[-1] + 1, 0.1)
            lambdas = arg.getlambda(grid, arg.arrivals)
            print(arg.arrivals)
            plt.plot(grid, lambdas)
            plt.plot(arg.arrivals, np.multiply(arg.mu, np.ones_like(arg.arrivals)), 'xr')
            plt.show()

    def plotKernel(self, tmax, dt=0.1):
        for arg in self.processes:
            grid = np.arange(0, tmax, dt)
            lambdas = arg.kernel(grid)
            plt.plot(grid, lambdas)
            plt.show()


if __name__ == "__main__":
    from exp_chp import ExpCHP


    np.random.seed(3)
    alpha = 0.6
    mu = 0.3
    beta = 0.8
    np.random.seed(1)
    T = 2000
    proc1 = ExpCHP(T, mu, alpha, beta)
    arrivals = proc1.getpath(T)
    print(len(arrivals))

    # proc2 = MHP([alpha], [mu], [beta])
    # proc2.generate_seq(T)

    plotproc = PlotAccount(proc1)
    plotproc.modelcheck()
