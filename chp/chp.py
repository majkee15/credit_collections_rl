import numpy as np
from joblib import Parallel, delayed
from base import Base


class CHP(Base):

    def __init__(self, params, control=None, precision=1e-3):

        super().__init__(__class__.__name__)
        self.params = params
        self.precision = precision
        if control is None:
            def control(w): return 0
            self.control = control
        else:
            self.control = control
        self.logger.info(f'Instantiated @ {self.__class__.__name__}')
        self.history = [(0, 0)]

    def simulate(self, l, w):
        s = 0
        lstart = l
        lhat = self.control(w)
        intensity_jump = max(lhat - lstart, 0)
        revenue = 0
        costs = intensity_jump * self.params.chat
        dc_profit = 1
        while dc_profit > self.precision:
            lhat = self.control(w)
            lstar = max(lstart, lhat)
            wincr = -np.log(np.random.rand()) / lstar
            s = s + wincr
            d = np.random.rand()
            intensity_at_jump = self.controlled_intensity(s, lstart, lhat)
            t_to_sustain = self.theta(lstart, lhat)
            if d * lstar <= intensity_at_jump:
                # jump occurs
                r = self.params.sample_repayment()[0]
                self.history.append((s, r))
                intensity_increment = self.params.delta10 + r * self.params.delta11
                dc_profit = r * w * np.exp(-self.params.rho * s)
                revenue = revenue - dc_profit
                if t_to_sustain + self.history[-2][0] < self.history[-1][0]:
                    costs = costs + self.params.chat * (self.params.kappa*(lhat - self.params.lambdainf)) * \
                            (np.exp(-self.params.rho * (t_to_sustain + self.history[-2][0])) -
                             np.exp(-self.params.rho * self.history[-1][0])) / \
                            self.params.rho
                lstart = intensity_at_jump + intensity_increment
                w = (1 - r) * w
                if costs != 0:
                    self.logger.info('There are costs.')
        return revenue + costs

    def drift(self, s, lambda_start):
        # deterministic draft between jumps
        return self.params.lambdainf + (lambda_start - self.params.lambdainf) * np.exp(-self.params.kappa * s)

    def controlled_intensity(self, s, lstart, lhat):
        drifted = self.drift(s, lstart)
        return np.maximum(drifted, lhat)

    def theta(self, lstart, lhat):
        # equation for duration to reach a holding region
        try:
            numerator = lstart - self.params.lambdainf
            denominator = lhat - self.params.lambdainf

            if lstart < lhat:
                return 0
            elif denominator < 0:
                #should be nan
                return np.inf
            else:
                t = 1 / self.params.kappa * \
                    np.log(numerator / denominator)
                return t
        except:
            print('warning')
            print(f'Numerator {lstart - self.params.lambdainf}')
            print(f'Denominator {lhat - self.params.lambdainf}')

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

