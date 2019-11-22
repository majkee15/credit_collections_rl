import numpy as np
import matplotlib.pyplot as plt
# class of a Controlled Hawkes Process from Chehrazi, Weber & Glynn 2017
# \lambda(t) = \lambda_/infty + (\lambda(t)-\lambda_\infty)Exp(-\kappa * t) +
# \delta_1^\top dJ^\top + actions
# Actions are given as an "arbitrary curve in (w, \lambda) space


class CHPMarked:
    def __init__(self, starting_balance, mu=None, alpha=None, beta=None, control_function=None):
        self.arrivals = np.array([])
        self.repayments = np.array([])
        self.starting_balance = starting_balance
        self.balances = np.array([self.starting_balance])
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.control_function = control_function
        self.params = [mu, alpha, beta]
        self.repayment_draw = np.random.rand
        self.flag_simulated = False

    def generate_path(self, horizon):
        arrivals = np.array([])
        repayments = np.array([])
        balances = np.array([self.starting_balance])
        s = 0
        n = 0
        is_collected = False

        while (s < horizon) & (is_collected == False):
            lstar = 100 #self.getlambda(s, arrivals) # maximum value of lambda
            w = -np.log(np.random.rand())/lstar
            s = s + w
            d = np.random.rand()
            potential_repayment = self.repayment_draw()
            potential_balance = balances[-1] * (1 - potential_repayment)
            if d*lstar <= self.getlambda(s, arrivals, potential_balance):
                n += 1
                arrivals = np.append(arrivals, s)
                repayments = np.append(repayments, potential_repayment)
                balances = np.append(balances, potential_balance)

                if repayments[-1] == 1:
                    is_collected = True

        if s > horizon:
            arrivals = arrivals[:-1]
            repayments = repayments[:-1]
            balances = balances[:-1]
            self.arrivals = arrivals
            self.repayments = repayments
            self.balances = balances
            self.flag_simulated = True
        return arrivals, balances

    def getlambda(self, s, arrivals, potential_balance):
        # Returns lambdas(either vector or a scalar) based on a realized process history arrivals
        # arrivals[arrivals <= s] cannot have a strict inequality because of the thinning algo
        # for straight inequalities the next candidate lstar is smaller than the actual intensity at the point!
        # calculate intensity
        #
        if self.control_function is not None:
            controlled_intensity = self.control_function(potential_balance)
            if np.isscalar(s):
                lamb = self.mu + sum(self.kernel(s - arrivals[arrivals <= s]))
                if lamb < controlled_intensity:
                    lambd = controlled_intensity
                return lamb
            else:
                controlled_intensities = self.control_function(potential_balance)
                n = len(s)
                lambdas = np.zeros(n)
                for i in range(0, n):
                    lambdas[i] = self.mu + sum(self.kernel(s[i] - arrivals[arrivals <= s[i]]))
                    if lambdas[i]< controlled_intensity[i]:
                        lambdas[i] = controlled_intensity[i]
                return lambdas
        else:
            if np.isscalar(s):
                lamb = self.mu + sum(self.kernel(s - arrivals[arrivals <= s]))
                return lamb
            else:
                n = len(s)
                lambdas = np.zeros(n)
                for i in range(0, n):
                    lambdas[i] = self.mu + sum(self.kernel(s[i] - arrivals[arrivals <= s[i]]))
                return lambdas

    def kernel(self, t):
        # Return Kernel evaluation at time t
        return np.multiply(self.alpha, np.exp(np.multiply(-self.beta, t)))

    @staticmethod
    def ismonotonic(self, arrivals):
        # Checks is arrivals is monotonically increasing
        return np.all(np.diff(arrivals) > 0)

    def plot_intensity(self, tmin=0, tmax=None, dt=0.1):
        if self.flag_simulated:
            if tmax is None:
                tmax = self.arrivals[-1] + (self.arrivals[-1] -self.arrivals[0])*0.05
            fig, ax = plt.subplots(1, 1)
            grid = np.arange(tmin, tmax, dt)
            lambdas = self.getlambda(grid, self.arrivals)
            ax.plot(grid, lambdas)
            ax.plot(self.arrivals, np.multiply(self.mu, np.ones_like(self.arrivals)), 'xr')
            ax.set_xlabel('Time')
            ax.set_ylabel('Intensity')
            fig.show()
        else:
            raise Exception('Not simulated.')

    def plot_balance(self):
        if self.flag_simulated:
            fig, ax = plt.subplots(1, 1)
            ax.step(self.balances, np.insert(self.arrivals, 0, 0))
            ax.set_xlabel('Balance')
            ax.set_ylabel('Time')
            fig.show()
            return fig
        else:
            raise Exception('Not simulated.')

    def get_balance_at_t(self,t):
        return_balances = np.empty_like(t)
        if self.flag_simulated:
            if np.isscalar(t):
                pass
            else:
                last_balance = self.starting_balance
                for i, point in enumerate(t):
                    index = len(self.arrivals[self.arrivals<point])
                    return_balances[i] = self.balances[index]
        return return_balances

    def plot_states(self, dt=0.1):
        if self.flag_simulated:
            tmax = self.arrivals[-1] + (self.arrivals[-1] - self.arrivals[0]) * 0.05
            tgrid = np.arange(0, tmax, dt)
            arrivals = np.insert(self.arrivals, 0, 0)
            watet = self.get_balance_at_t(tgrid)
            intensity = self.getlambda(tgrid, arrivals, watet)
            fig1, ax1 = plt.subplots(1, 1)
            ax1.plot(watet, intensity)
            ax1.plot(watet, intensity, 'xr')
            if self.control_function is not None:
                wgrid = np.arange(0, self.starting_balance, 1)
                ax1.plot(wgrid, self.control_function(wgrid), color='black', linewidth=0.5)
            ax1.set_xlabel('Balance')
            ax1.set_ylabel('Intensity')
            fig1.show()
            fig2, ax2 = plt.subplots(1, 1)
            ax2.plot(tgrid,intensity)
            ax2.set_xlabel('Balance')
            ax2.set_ylabel('Time')
            fig2.show()
            fig3, ax3 = plt.subplots(1, 1)
            ax3.plot(watet, tgrid)
            ax3.set_xlabel('Balance')
            ax3.set_ylabel('Time')
            fig3.show()

        else:
            raise Exception('Not simulated.')


if __name__ == '__main__':
    def control(w):
        return (0.005 * (w - 200)).clip(min=0)

    # np.random.seed()
    chp = CHPMarked(starting_balance=1000, mu=0.1, alpha=0.6, beta=0.9)
    chp_controlled = CHPMarked(starting_balance=1000, mu=0.1, alpha=0.6, beta=0.9, control_function=control)
    print(chp.generate_path(100))
    print(chp_controlled.generate_path(100))
    # chp.plot_intensity()
    # chp_controlled.plot_intensity()
    chp.plot_states()
    chp_controlled.plot_states()
