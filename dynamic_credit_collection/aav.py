from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


# integrand = @(r) (1-(1-r).*exp(-(p.delta10+p.delta11.*r).*y(1))).*p.rdist(r);
# integrate.quad(lambda x: special.jv(2.5,x), 0, 4.5)


def aav_ODE(y, t, parameters):
    """
    Defines the ODE from Chehrazi&Weber 2015 for autonomous
    account value.
    :param y:
    :param t: time
    :param parameters: class defining hp parameters
    :return: dy/dt
    """

    def integrand(r): return (1 - (1 - r) * np.exp(-(parameters.delta10 + parameters.delta11 * r) * y[0])
                              ) * parameters.rdist(r)

    q = integrate.quad(integrand, 0.1, 1)
    dy1 = -parameters.kappa * y[0] + q[0]
    dy2 = parameters.lambdainf * y[0]
    dydt = [dy1, dy2]
    return dydt


def aav(l, w, parameters):
    """
    Calculates the autonomous account value of the account.
    :param l: lambda_0 starting intensity
    :param w: starting balannce
    :param parameters: class defining hp parameters
    :return: autonomous account value
    """
    t = np.linspace(0, 60, 101)
    y0 = [0, 0]
    y = np.array(integrate.odeint(aav_ODE, y0, t, args=(parameters,)))
    I = np.exp(-parameters.rho * t - l * y[:, 0] - parameters.kappa * y[:, 1])
    q = integrate.trapz(I, t)
    return -(1 - parameters.rho * q) * w


class Parameters:
    def __init__(self):
        self.lamdbda0 = 0.11
        self.lambdainf = 0.1
        self.kappa = 0.7
        self.delta10 = 0.02
        self.delta11 = 0.5
        self.rho = 0.06

    def rdist(self, r):
        return 1 / 0.9


if __name__ == '__main__':
    y0 = [0, 0]
    t = np.linspace(0, 60, 101)
    params = Parameters()
    sol = integrate.odeint(aav_ODE, y0, t, args=(params,))
    print(aav(0.11, 100, params))
