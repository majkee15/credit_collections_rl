from base import Base
import numpy as np
import matplotlib.pyplot as plt
from legacy.osl.mc import sustained_ihp_c
import time
import multiprocessing as mp
import functools

from dcc import Parameters


class SustainedIHP(Base):
    '''
    Wrapper for the cython code to calculate optimal policies based on Weber 2015
    '''
    def __init__(self, w0, parameters, spacing='linear', ws_points=30, lmax_profile=1.5, niter=1e6, npoints_profile=30,
                 *args, **kwargs):
        """

        Args:
            w0: double
                starting balance
            parameters: object
                class defining parameters of the CHP
            spacing: str
                spacing used for policies construction
            *args:
            **kwargs:
        """
        super().__init__(__class__.__name__)
        self.parameters = parameters
        self.params = np.array([parameters.lamdbda0, parameters.lambdainf, parameters.kappa,
                                parameters.delta10, parameters.delta11, parameters.delta2,
                                parameters.rho, parameters.c], dtype=np.float)
        self.w0 = w0
        # Various setup specs here:
        self.ws_points = ws_points
        self.n_iter_profile = niter
        self.lmax_profile = lmax_profile
        self.npoints_profile = npoints_profile
        # smart grid differentiation
        if spacing == 'linear':
            self.ws = np.linspace(0, w0, self.ws_points)
        elif spacing == 'log':
            rmin = self.parameters.r_
            self.ws = np.insert(np.cumprod(np.ones(self.ws_points) * (1 - rmin)) * self.w0, 0, self.w0)
            self.ws = np.insert(self.ws, self.ws_points + 1, 0)
        else:
            raise ValueError('Spacing is supported only for linear and log specifications.')
        self.logger.info(f'Instantiated sustain process @ {__class__.__name__}')

    def calculate_lhat_profile(self, balance, lhatmax, npoints, niter, dist='uniform'):
        # TODO: this is not ideal, slow, a lot of variation in consecutive lambda hats
        """
        Calculates the full cost/collection profile for a single balance (plot lambda hat / collected)
        Args:
            balance: double
            lhatmax: double
                upper bound on the search space
            npoints: int
                number of grid points
            niter: int
                number of iterations to use for a mc account evaluation
            dist: str
        Returns:

        """
        if dist == 'uniform':
            calculate_value = sustained_ihp_c.calculate_value_mc_nogil
        elif dist == 'generalized':
            def calculate_value(lhat, niter, balance, params):
                low = 0.1
                up = 0.9
                pi = 0.99
                return sustained_ihp_c.calculate_value_mc_gen_nogil(lhat, niter, balance, params, low, up, pi)
        else:
            raise ValueError('Distribution not implemented.')

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
            intermediate_res = np.asarray(calculate_value(lhat, niter, balance, self.params))
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
        # lhatstar = lambda_hats[lhatstar_index]
        execution_time = time.time() - start_time
        m, s = divmod(execution_time, 60)
        self.logger.info(f'{self.calculate_lhat_profile.__name__} finished for balance ${balance:.2f}. '
                         f'Execution time: {m:.2f} min and {s:.2f} seconds.')
        return lambda_hats, accvalsmc, accvalsmc_std

    def objective_function(self):
        pass

    def calc_profile_parallel(self, w0):
        n_cpu = mp.cpu_count() - 1
        ws = np.flip(np.linspace(1, w0, 2))
        worker = functools.partial(self.calculate_lhat_profile,
                                   self.lmax_profile, self.npoints_profile, self.n_iter_profile)
        with mp.Pool(n_cpu) as pool:
            res = pool.map(worker, ws)
            return ws, res

    def calculate_frontier(self,plot_flag=False, dist='uniform'):
        # ws = np.flip(np.linspace(0, w0, 30))
        # lhatstars = np.zeros_like(self.ws)
        # accvalsmc = np.zeros_like(self.ws)
        accvalues = []
        accvalues_std = []
        lhatstars = []
        accvals_star = []
        for i, w in enumerate(self.ws):
            lambda_hats, accvalsmc, accvalsmc_std = self.calculate_lhat_profile(w, self.lmax_profile,
                                                                                self.npoints_profile,
                                                                                self.n_iter_profile, dist)
            lhatstar_index = np.nanargmin(accvalsmc)
            lhatstars.append(lambda_hats[lhatstar_index])
            accvalues.append(accvalsmc)
            accvalues_std.append(accvalsmc_std)
            accvals_star.append(accvalsmc[lhatstar_index])

        if plot_flag:
            plt.figure()
            plt.plot(self.ws, lhatstars, marker='x')
            plt.xlabel('w')
            plt.ylabel('lhatstar')
            plt.ylim([0.0, self.lmax_profile])
            plt.show()
            plt.figure()

            plt.plot(self.ws, accvals_star, marker='x')
            plt.xlabel('w')
            plt.ylabel('v')
        return lhatstars

    def plot_value(self, lambda_hats, accvalsmc, collectedmc=None, balance=0, std=None):
        lhatstar_index = np.nanargmin(accvalsmc)
        lhatstar = lambda_hats[lhatstar_index]
        fig, ax = plt.subplots()
        ax.plot(lambda_hats, accvalsmc, marker='x')
        if collectedmc is not None:
            ax.plot(lambda_hats, collectedmc, marker='x')
        ax.axvline(x=lhatstar, color='r')
        ax.set_xlabel('\hat{\lambda}')
        ax.set_ylabel('v')
        ax.legend(['Account value', 'Amount collected'])
        ax.set_title(f'Balance {balance:.2f}.')
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



if __name__ == '__main__':
    # from dcc import OAV
    # PATH_TO_PICKLE = '/Users/mmark/Documents/credit_collections/credit_collections_rl/dcc/ref_parameters'
    # oc = OAV.load(PATH_TO_PICKLE)
    # oc.plot_statespace()

    starting_balance = 100
    niter = 1000000
    params = Parameters()
    sihp = SustainedIHP(starting_balance, params, lmax_profile=2, ws_points=30, niter=niter)
    approx_lstars = sihp.calculate_frontier(plot_flag=True, dist='uniform')
    print(approx_lstars)

    from legacy.osl.exact import OSL
    osl = OSL(starting_balance, params)
    lhats, vgreedy = osl.calculate_greedy_frontier()
    fig, ax = plt.subplots()
    ax.plot(osl.ws, lhats, marker='x')
    ax.plot(sihp.ws, approx_lstars, marker='x')
    fig.show()