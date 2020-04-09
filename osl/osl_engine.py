# optimal sustained level engine
from base import Base
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class OSL(Base):

    def __init__(self, w0, p, spacing='linear', ws_points=30):
        super().__init__(__class__.__name__)
        self.w0 = w0
        self.p = p
        self.n_ws_points = ws_points
        if spacing == 'linear':
            self.ws = np.linspace(0, w0, self.n_ws_points)
        elif spacing == 'log':
            rmin = self.p.r_
            self.ws = np.insert(np.cumprod(np.ones(self.n_ws_points) * (1 - rmin)) * self.w0, 0, self.w0)
            self.ws = np.insert(self.ws, self.n_ws_points + 1, 0)

    def q(self, l, lhat):
        """
        Q function from the text.
        Args:
            l: double
            lhat: double

        Returns: double
            q-value
        """
        with np.errstate(all='raise'):
            # there was a problem with numpy exp that does not handle well negative base with real exponents
            try:
                base = np.divide(lhat - self.p.lambdainf, l - self.p.lambdainf)
                exponent = (self.p.rho + self.p.lambdainf) / self.p.kappa
                res = lhat / (self.p.rho + lhat) * \
                      (np.sign(base) * np.abs(base) ** exponent) * np.exp((lhat - l) / self.p.kappa)
            except Exception:
                print('CAUGHT')
                base = np.divide(lhat - self.p.lambdainf, l - self.p.lambdainf)
                exponent = (self.p.rho +self. p.lambdainf) / self.p.kappa
                self.logger.exception(f'Problem in q. q={0}, '
                                      f'base ={base}, '
                                      f'exponent={exponent}.')
                res = 0
        return res

    def sustain_costs(self, l, lhat):
        return self.p.chat * self.p.kappa * np.divide(lhat - self.p.lambdainf, lhat) * self.q(l, lhat)

    def costs(self, l, lhat):
        if l >= lhat:
            sust_cost = self.sustain_costs(l, lhat)
            jump_cost = 0
        else:
            sust_cost = + self.sustain_costs(lhat, lhat)
            jump_cost = + self.p.chat * (lhat - l)
        return sust_cost, jump_cost

    def integrand_disc_rep(self, l, lam):
        return self.q(lam, l) * np.divide(self.p.rho + l, self.p.kappa * (l - self.p.lambdainf))

    def discounted_repayment(self, w, lam, lhat):
        return w * self.p.rmean * (quad(self.integrand_disc_rep, lhat, lam, args=(lam), limit=200)[0] + self.q(lam, lhat))

    def greedy_v(self, lhat, w, l):
        if l >= lhat:
            v = - self.discounted_repayment(w, l, lhat) + self.sustain_costs(l, lhat)
        else:
            v = + self.p.chat * (lhat - l) + self.sustain_costs(lhat, lhat) - self.discounted_repayment(w, lhat, lhat)
        return v

    def calculate_greedy_frontier(self):
        lambda_hats = np.zeros_like(self.ws)
        v_greedy = np.zeros_like(self.ws)
        x0 = np.array([1.0])
        l = 2.0
        for i, w in enumerate(self.ws):
            res = minimize(self.greedy_v, x0, args=(w, l), bounds=[(self.p.lambdainf, None)])
            v_greedy[i] = res.fun
            lambda_hats[i] = res.x
        return lambda_hats, v_greedy

if __name__ == '__main__':
    from dcc import Parameters
    p = Parameters()
    w0 = 100
    osl = OSL(w0, p)
    lhats, vgreedy = osl.calculate_greedy_frontier()
    print(lhats)
    plt.plot(osl.ws, lhats, marker='x')
    plt.show()




