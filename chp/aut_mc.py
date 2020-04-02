import numpy as np
from joblib import Parallel, delayed
from base import Base


class CHP(Base):

    def __init__(self, params, control=None, precision=1e-3):

        super().__init__(__class__.__name__)
        self.params = params
        self.precision = precision
        self.logger.info(f'Instantiated @ {self.__class__.__name__}')
        self.history = [(0, 0)]

    def simulate(self, l, w):
        s = 0
        lstart = l
        revenue = 0
        dc_profit = 1
        while dc_profit > self.precision:
            lstar = lstart
            wincr = -np.log(np.random.rand()) / lstar
            s = s + wincr
            d = np.random.rand()
            intensity_at_jump = self.drift(s, lstart)
            if d * lstar <= intensity_at_jump:
                # jump occurs
                r = self.params.sample_repayment()[0]
                self.history.append((s, r))
                intensity_increment = self.params.delta10 + r * self.params.delta11
                dc_profit = r * w * np.exp(-self.params.rho * s)
                revenue = revenue - dc_profit
                lstart = intensity_at_jump + intensity_increment
                w = (1 - r) * w
        return revenue

    def drift(self, s, lambda_start):
        # deterministic draft between jumps
        return self.params.lambdainf + (lambda_start - self.params.lambdainf) * np.exp(-self.params.kappa * s)

    def v(self, l, w, niter=5000):
        """
        Value approximation
        Args:
            l:
            w:

        Returns:

        """
        mc = Parallel(n_jobs=10)(delayed(self.simulate)(l, w) for i in range(niter))
        return np.mean(mc)

if __name__ == '__main__':
    from dcc import Parameters, AAV, OAV
    import matplotlib.pyplot as plt
    params = Parameters()
    chp = CHP(params)
    av = AAV(params)
    lam = 2
    w = 1000
    niter = 10000
    chp.simulate(lam, w)
    print(chp.history)
    fig, ax = plt.subplots()
    for line in chp.history:
        ax.axvline(line[0])
    fig.show()
    # mc = Parallel(n_jobs=10)(delayed(chp.simulate)(lam, w) for i in range(niter))
    # print(np.mean(mc))
    # cummean = np.cumsum(mc) / np.arange(1, niter + 1)
    # plt.plot(cummean)
    # plt.show()


    print(chp.v(lam, w))
    print(av.u(lam, w))

