import copy
import os
from functools import partial
from multiprocessing import cpu_count, Pool

import numpy as np

from base import Base
from chp import CHP


class BruteForce(Base):
    # conducts a brute force search for optimal control policy of a form
    # f(w) = max(0, m(w-b)
    # m,b \in \mathbb{R}
    def __init__(self, chp, data_dir, delta, bmax=10, mmax=1, mc_iterations=10000):
        super().__init__(__class__.__name__)
        self.data_dir = data_dir
        self.path = os.getcwd()
        # self.results_dir = os.path.join(self.data_dir, 'mc_curves')
        os.makedirs(self.data_dir, exist_ok=True)
        self.chp = chp
        self.mc_iterations = mc_iterations
        self.delta_w = delta[0]
        self.delta_m = delta[1]
        self.b = np.arange(0, bmax, self.delta_w)
        self.m = np.arange(0, mmax, self.delta_m)
        self.xx, self.yy = np.meshgrid(self.b, self.m)
        self.z = np.zeros_like(self.xx, dtype=np.float64)
        self.logger.info(f'Instantiated Brute Force Search Process @ {self.__class__.__name__}')


    def get_policy_value(self, b, m):
        control = partial(self.prep_linear_control, b, m)
        # self.chp.control_function = control
        # return self.chp.calculate_value(self.mc_iterations)
        return self.calculate_value_parallel(control, self.mc_iterations)

    def run_brute_force_search(self):
        file_path = os.path.join(self.path, self.data_dir)
        z_path = os.path.join(file_path, 'bs_z.npy')
        # saving coordinates
        np.save(os.path.join(file_path, 'bs_xx.npy'), self.xx)
        np.save(os.path.join(file_path, 'bs_yy.npy'), self.yy)
        # Lets dig the routine :*
        self.logger.info(f'The grid is consists of\n'
                         f'b={self.b}\n'
                         f'm={self.m}')
        for i, xval in enumerate(self.b):
            for j, yval in enumerate(self.m):
                self.logger.info(f'evaluating the policy b={xval}, m={yval}')
                self.z[j,i] = self.get_policy_value(xval, yval)
                np.save(z_path, self.z)
        return self.z

    def prep_linear_control(self, b, m, w):
        if isinstance(w, np.ndarray):
            return (m * (w - b)).clip(min=0)
        elif np.isscalar(w):
            return np.maximum(m * (w - b), 0)
        else:
            raise ValueError('Stupid error in w.')

    def instantiate_list(self, control, mc_iterations=10000):
        instance = partial(self.instantiate_single, control)
        return [instance() for i in range(mc_iterations)]

    def instantiate_list_parallel(self, control, mc_iterations=10000):
        worker = partial(self.instantiate_single, control)
        um_cores = cpu_count()
        with Pool(um_cores - 1) as p:
            return p.map(worker, range(mc_iterations))

    def instantiate_single(self, control, _=None):
        # return CHP(starting_balance=100, starting_intensity=1, marginal_cost=1,
        #     collection_horizon=100, lambda_infty=0.1, kappa=0.7,
        #     delta10=0.02, delta11=0.5, control_function=control, rho=0.06)
        self.chp.control_function=control
        obj = copy.deepcopy(self.chp)
        obj.control_function = control
        return obj

    def calculate_value_parallel(self, control, niter, parallel_instantiation=False):

        if parallel_instantiation:
            process_list = self.instantiate_list_parallel(control, niter)
        else:
            process_list = self.instantiate_list(control, niter)

        um_cores = cpu_count()
        with Pool(um_cores - 1) as p:
            return np.mean(p.map(self.price, process_list))

    @staticmethod
    def price(chp):
        return chp.calculate_value_single()

if __name__ == '__main__':

    chp = CHP(starting_balance=75, starting_intensity=1, marginal_cost=6,
              collection_horizon=10000, lambda_infty=0.1, kappa=0.7,
              delta10=0.02, delta11=0.5, control_function=None, rho=0.06, value_precision_thershold=0.01)
    bf = BruteForce(chp, delta=[1, 0.001], data_dir='../results/search_comp_results', bmax=75, mmax=1, mc_iterations=10000)
    print(bf.run_brute_force_search())


