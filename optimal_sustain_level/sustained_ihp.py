from base import Base
import numpy as np
import matplotlib.pyplot as plt
from optimal_sustain_level import sustained_ihp_c
import time
import multiprocessing as mp
import functools


class SustainedIHP(Base):
    '''
    Wrapper for the cython code to calculate optimal policies based on Weber 2015
    '''
    def __init__(self, w0, parameters, *args, **kwargs):
        '''
        :param parameters: np.ndarray[dim=1]: Class defining the parameteres of the sustained ihp
        '''
        super().__init__(__class__.__name__)
        self.parameters = parameters
        self.params = np.array([parameters.lamdbda0, parameters.lambdainf, parameters.kappa,
                                parameters.delta10, parameters.delta11, parameters.delta2,
                                parameters.rho, parameters.c], dtype=np.float)
        self.w0 = w0
        rmin = 0.1
        ws_points = 20
        self.ws = np.insert(np.cumprod(np.ones(ws_points) * (1 - rmin)) * self.w0, 0, self.w0)
        self.ws = np.insert(self.ws, ws_points + 1, 0)
        self.logger.info(f'Instantiated sustain process @ {__class__.__name__}')

    # def create_wgrid(self, style, **kwargs):
    #     if style == 'linear':
    #         np.arange

    def calculate_lhat_profile(self, balance, lhatmax, npoints, niter):
        start_time = time.time()
        lhatmin = self.params[1] + self.params[1] * 0.1
        lambda_hats = np.linspace(lhatmin, lhatmax, npoints)
        lambda_hats = np.insert(lambda_hats, 0, 0)
        sust_costsmc = np.zeros_like(lambda_hats)
        jump_costsmc = np.zeros_like(lambda_hats)
        collectedmc = np.zeros_like(lambda_hats)
        arrivalsmc = np.zeros_like(lambda_hats)
        accvalsmc = np.zeros_like(lambda_hats)
        accvalsmc_std = np.zeros_like(lambda_hats)
        for i, lhat in enumerate(lambda_hats):
            intermediate_res = np.asarray(sustained_ihp_c.calculate_value_mc(lhat, niter, balance, self.params))
            ar = intermediate_res[:, 0]
            collected = intermediate_res[:, 1]
            sust_cost = intermediate_res[:, 2]
            jump_cost = intermediate_res[:, 3]
            accvalsmc[i] = np.mean(sust_cost + jump_cost - collected)
            accvalsmc_std[i] = np.std(sust_cost + jump_cost - collected)
            sust_costsmc[i] = np.mean(sust_cost)
            jump_costsmc[i] = np.mean(jump_cost)
            collectedmc[i] = np.mean(-collected)
            arrivalsmc[i] = np.mean(ar)
        lhatstar_index = np.nanargmin(accvalsmc)
        lhatstar = lambda_hats[lhatstar_index]
        execution_time = time.time() - start_time
        m, s = divmod(execution_time, 60)
        self.logger.info(f'{self.calculate_lhat_profile.__name__} finished for balance ${balance:.2f}. '
                         f'Execution time: {m:.2f} min and {s:.2f} seconds.')
        return lhatstar, accvalsmc[lhatstar_index]

    def calc_profile_parallel(self, w0):
        niter = 100000
        lmax = 10
        npoints = 50
        n_cpu = mp.cpu_count() - 1
        ws = np.flip(np.linspace(1, w0, 2))
        worker = functools.partial(self.calculate_lhat_profile, lmax, npoints, niter)
        with mp.Pool(n_cpu) as pool:
            res = pool.map(worker, ws)
            return ws, res

    def calculate_frontier(self):
        # ws = np.flip(np.linspace(0, w0, 30))
        lhatstars = np.zeros_like(self.ws)
        accvalsmc = np.zeros_like(self.ws)
        lmax = 1
        npoints = 15
        niter = 100000
        for i, w in enumerate(self.ws):
            lhatstars[i], accvalsmc[i] = self.calculate_lhat_profile(w, lmax, npoints, niter)
            lmax = lhatstars[i] + lhatstars[i] * 0.01
        print(accvalsmc)
        print(self.ws)
        plt.figure()
        plt.plot(self.ws, lhatstars, marker='x')
        plt.xlabel('w')
        plt.ylabel('lhatstar')
        plt.ylim([0.0, 2.0])
        plt.show()
        plt.figure()

        plt.plot(self.ws, accvalsmc, marker='x')
        plt.xlabel('w')
        plt.ylabel('v')
        plt.show()

    def plot_value(self, lambda_hats, accvalsmc, collectedmc, balance=0, std=None):
        lhatstar_index = np.nanargmin(accvalsmc)
        lhatstar = lambda_hats[lhatstar_index]
        fig, ax = plt.subplots()
        ax.plot(lambda_hats, accvalsmc, marker='x')
        ax.plot(lambda_hats, collectedmc, marker='x')
        ax.axvline(x=lhatstar, color='r')
        ax.set_xlabel('\hat{\lambda}')
        ax.set_ylabel('v')
        ax.legend(['Account value', 'Amount collected'])
        ax.set_title(f'Balance {balance}.')
        if std is not None:
            ax.fill_between(lambda_hats, accvalsmc, accvalsmc + std, color='salmon', alpha=0.5)
            ax.fill_between(lambda_hats, accvalsmc, accvalsmc - std, color='salmon', alpha=0.5)
        return fig

    def plot_costs(self, lambda_hats, sust_costsmc, jump_costsmc):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        ax[0].plot(lambda_hats, sust_costsmc, marker='x')
        ax[0].plot(lambda_hats, jump_costsmc, marker='x')
        ax[0].plot(lambda_hats, sust_costsmc + jump_costsmc, marker='x')
        ax[0].set_xlabel('\hat{\lambda}')
        ax[0].set_ylabel('costs')
        ax[1].plot(lambda_hats, sust_costsmc, marker='x')
        ax[0].legend(['Sustain costs', 'Jump costs', 'Total'])
        ax[1].legend(['Sustain costs'])
        ax[1].set_xlabel('\hat{\lambda}')
        ax[1].set_ylabel('Sustain Costs')
        return fig

class Parameters:
    '''
    Defines the parameters of the sustained HP.
    '''
    def __init__(self):
        self.lamdbda0 = 1
        self.lambdainf = 0.1
        self.kappa = 0.7
        self.delta10 = 0.02
        self.delta11 = 0.5
        self.delta2 = 1
        self.rho = 0.06
        self.c = 6
        self.r_ = 0.1



if __name__ == '__main__':
    lambda_hat = 0.11
    starting_balance = 100
    params = Parameters()
    sihp = SustainedIHP(starting_balance, params)
    sihp.calculate_frontier()
