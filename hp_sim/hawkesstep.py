import numpy as np
from dcc import Parameters
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from dcc import AAV


class StepSim():
    def __init__(self, w0, params, dt=0.01):
        self.params = params
        self.dt = dt
        self.lambda0 = 1
        self.w0 = w0
        self.value_threshold = 0.1

    def naive_arrival(self, lambda0):
        threshold = lambda0 * self.dt
        draw = np.random.random()
        arrival = False
        if draw <= threshold:
            arrival = True
        return arrival

    def simulate_coarse(self):
        w = self.w0
        t = 0
        l0 = self.lambda0
        reward = 0
        while True:
            discount_factor = np.exp(-self.params.rho * t)
            nt = np.random.poisson(l0 * self.dt)
            l0 =  self.drift(self.dt, l0)
            if nt == 0:
                r = [0]
            else:
                r = [self.params.sample_repayment() for _ in range(nt)]
                l0 = l0 + np.sum([self.params.delta10 + self.params.delta11 * ri for ri in r])

            reward += np.sum(np.cumprod(r) * w * discount_factor)
            w = w - np.sum(np.cumprod(r) * w)
            t += dt
            if w < self.value_threshold:
                break
        return reward

    def intensity_integral(self, t, lambda0):
        return self.params.lambdainf * t + (lambda0 - self.params.lambdainf) * \
               (1 - np.exp(-self.params.kappa * t)) / self.params.kappa

    def simulate_from_integral(self, T=10):
        w = self.w0
        reward = 0
        arrivals = []
        lambdas = [self.lambda0]
        repayments = []
        l0 = self.lambda0
        t = 0
        t_from_jump = 0
        draw = np.random.exponential(1)
        while t <= T:
            discount_factor = np.exp(-self.params.rho * t)
            t += dt
            t_from_jump += dt
            if self.intensity_integral(t_from_jump, l0) >= draw:
                arrivals.append(t)
                r = self.params.sample_repayment()
                repayments.append(r)
                l0 = self.params.delta10 + self.params.delta11 * r
                lambdas.append(l0)
                draw = np.random.exponential()
                t_from_jump = 0
            else:
                r = 0
                lambdas.append(self.drift(t_from_jump, l0))
            reward += w * r * discount_factor
            w = w * (1 - r)
            if w < self.value_threshold:
                break
        self.arrivals = np.array(arrivals)
        return reward

    def simulate(self, type):
        w = self.w0
        if type == 'naive':
            draw_arr = self.naive_arrival
        else:
            draw_arr = self.naive_arrival
        t = 0
        arrivals = []
        lambdas = [self.lambda0]
        repayments = []
        l0 = self.lambda0
        reward = 0
        while True:
            discount_factor = np.exp(-self.params.rho * t)
            if draw_arr(l0):
                arrivals.append(t)
                r = self.params.sample_repayment()
                repayments.append(r)
                l0 = self.drift(self.dt, l0) + self.params.delta10 + self.params.delta11 * r
            else:
                r = 0
                l0 = self.drift(self.dt, l0)
            reward += w * r * discount_factor
            w = w * (1 - r)
            lambdas.append(l0)
            t += dt
            if w < self.value_threshold:
                break
        return reward

    def calc_value(self, n):
        simdata = [self.simulate('naive') for _ in range(n)]
        return np.mean(simdata), np.std(simdata)

    def calc_value_parallel(self, n, simtype ='naive'):
        cores = cpu_count() - 1
        if simtype == 'naive':
            def worker(): return self.simulate('naive')
        elif simtype == 'coarse':
            def worker(): return self.simulate_coarse()
        elif simtype == 'integral':
            def worker(): return self.simulate_from_integral()
        else:
            raise NotImplementedError('Use one of the supported methods.')
        simdata = Parallel(n_jobs=cores)(delayed(worker)() for _ in range(n))
        return np.mean(simdata), np.std(simdata)

    def drift(self, s, lambda_start):
        # deterministic draft between jumps
        return self.params.lambdainf + (lambda_start - self.params.lambdainf) * np.exp(-self.params.kappa * s)

    def test_accuracy(self, n, simtype):
        # dt_grid = np.linspace(0.0005, 0.1, 10)
        dt_grid = np.logspace(-2, 0.5, 5)
        means = np.zeros_like(dt_grid)
        exact = np.zeros_like(dt_grid)
        stds = np.zeros_like(dt_grid)

        aav = AAV(self.params)
        exact[:] = -aav.u(self.lambda0, self.w0)
        for i, dt in enumerate(dt_grid):
            self.dt = dt
            means[i], stds[i] = self.calc_value_parallel(n, simtype)
        fig, ax = plt.subplots()
        ax.plot(dt_grid, means, marker='x')
        ax.fill_between(dt_grid, means-stds, means+stds, alpha=0.5, color='salmon')
        ax.plot(dt_grid, exact)
        ax.set_xlabel(r'$\Delta t$')
        ax.set_ylabel('Account Value')
        ax.legend(['Simulated', 'Exact'])
        return fig

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    w0 = 100
    p = Parameters()
    dt = 0.01
    sim = StepSim(w0, p, dt)
    # plot lambdas
    # arrivals, lambdas = sim.simulate('naive', T)
    # x = np.linspace(0, T, len(lambdas))
    # plt.plot(x, lambdas)
    # plt.show()

    #print(sim.simulate_coarse())
    sim.test_accuracy(1000, simtype='integral').show()
    aav = AAV(p)
    print(aav.u(1, w0))
