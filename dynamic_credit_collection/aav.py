from scipy import integrate
from scipy.optimize import fsolve
import numpy as np
from base import Base
import matplotlib.pyplot as plt

# integrand = @(r) (1-(1-r).*exp(-(p.delta10+p.delta11.*r).*y(1))).*p.rdist(r);
# integrate.quad(lambda x: special.jv(2.5,x), 0, 4.5)


class AAV(Base):
    """
    Instance of the autonomous (i.e., not controleld) collection process as per Chehrazi, Weber & Glyn 2019
    “Dynamic Credit-Collections Optimization”.
    """
    def __init__(self, parameters):
        """
        Args:
            parameters: class Parameters
        """
        super().__init__(__class__.__name__)
        self.parameters = parameters
        self.alpha = None
        self.beta = None
        self.t = None
        self.w_ = None
        self.w0star = None
        self.av = None
        self.logger.info(f'Instantiated @ {self.__class__.__name__}')
        # Since the AAV is dependent on the solution of ODE1 that is independent of the balance
        # it is feasible to precompute it here
        

    def reset_state(self):
        self.alpha = None
        self.beta = None
        self.t = None
        self.w_ = None
        self.w0star = None
        self.u = None
        self.logger.info(f'@ {self.__class__.__name__}, state reset')

    def aav_ode(self, y, t):
        def integrand(r): return (1 - (1 - r) * np.exp(-(self.parameters.delta10 + self.parameters.delta11 * r) * y[0])
                                  ) * self.parameters.rdist(r)

        q = integrate.quad(integrand, 0.1, 1)
        dy1 = -self.parameters.kappa * y[0] + q[0]
        dy2 = self.parameters.lambdainf * y[0]
        dydt = [dy1, dy2]
        return dydt

    def u(self, l, w):
        """
        Calculates the autonomous account value of the account.
        Args:
            l: double
                intensity
            w: double
                balance
        Returns: double
            autonomous acc value
        """
        t = np.linspace(0, 100, 201)
        self.t = t
        y0 = [0, 0]
        y = np.array(integrate.odeint(self.aav_ode, y0, t))
        self.alpha = y[:, 0]
        self.beta = y[:, 1]
        I = np.exp(-self.parameters.rho * t - l * y[:, 0] - self.parameters.kappa * y[:, 1])
        q = integrate.trapz(I, t)
        w_ = self.compute_w_()
        self.w_ = w_
        self.w0star = self.compute_w0star()
        self.av = -(1 - self.parameters.rho * q) * w
        return self.av

    def compute_w_(self):
        """
        Calculares minimum actionable balance for a given acc.
        Returns: double
            Minimum actionable balance \underscore{w}
        """
        I = self.alpha * np.exp(-self.parameters.rho * self.t - self.parameters.kappa * self.beta)
        w = (1 / integrate.trapz(I, self.t)) * (self.parameters.c/self.parameters.delta2)* (1/self.parameters.rho)
        return w

    def compute_w0star(self):
        """
        Calculates w0 star, i.e., balance for the intersection of \lambda_\infty and w0star.
        Calculaates
        Returns: double
            w0star
        """
        I = self.alpha * np.exp(-self.parameters.rho * self.t - self.parameters.lambdainf * self.alpha -
                                self.parameters.kappa * self.beta)
        def w0stareq(w0star): return -integrate.trapz(I, self.t) * w0star * self.parameters.rho + \
                                     self.parameters.c/self.parameters.delta2
        return fsolve(w0stareq, np.array([50]))[0]

    def plot_aav(self, l, w_array, plot_flag = False):
        """
        Creates the aav plot w.r.t. different balance values
        Args:
            l:
            w_array:
            plot_flag:

        Returns:

        """
        u_vals = np.zeros_like(w_array)
        for i, w in enumerate(w_array):
            u_vals[i] = self.u(l, w)

        if plot_flag:
            plt.plot(w_array, u_vals, marker='o')
            plt.show()
        return u_vals


class Parameters:
    """
    Defines parameters for the controlled Hawkes process.
    """
    def __init__(self):
        self.lamdbda0 = 0.11
        self.lambdainf = 0.1
        self.kappa = 0.7
        self.delta10 = 0.02
        self.delta11 = 0.5
        self.delta2 = 1
        self.rho = 0.06
        self.c = 6
        self.r_ = 0.1
        self.chat = self.c / self.delta2

    def rdist(self, r):
        """
        PDF of the relative distribution
        Args:
            r: double
        Returns: double
        """
        return 1 / (1-self.r_)


if __name__ == '__main__':
    y0 = [0, 0]
    balance = 75
    # t = np.linspace(0, 60, 101)
    params = Parameters()
    aavcl = AAV(params)
    aavcl.u(1, balance)
    print(aavcl.av)
    print(aavcl.w_)
    print(aavcl.compute_w0star())
    w_array = np.arange(0, 100 ,10)
    print(w_array)
    aavcl.plot_aav(1, w_array, True)
