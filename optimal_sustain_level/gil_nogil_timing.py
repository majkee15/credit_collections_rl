# The purpose of this script is to decide whether multi threading is worth it for calculate_value_mc function

from optimal_sustain_level.sustained_ihp import Parameters
from optimal_sustain_level import sustained_ihp_c
import numpy as np


parameters = Parameters()
params = np.array([parameters.lamdbda0, parameters.lambdainf, parameters.kappa,
                                parameters.delta10, parameters.delta11, parameters.delta2,
                                parameters.rho, parameters.c], dtype=np.float)
lambda_hat = 0.11
starting_balance = 100
niter = 5000000

def calc_gil():
 return np.asarray(sustained_ihp_c.calculate_value_mc(lambda_hat, niter, starting_balance, params))

def calc_nogil():
    return np.asarray(sustained_ihp_c.calculate_value_mc_nogil(lambda_hat, niter, starting_balance, params))

if __name__ == '__main__':
    import timeit

    nreps = 100
    print(f' Time of {niter} evaluations with gil: '
          f'{timeit.timeit("calc_gil()", setup="from __main__ import calc_gil", number=nreps)/nreps:.2f}')
    print(f' Time of {niter} evaluations with gil: '
          f'{timeit.timeit("calc_nogil()", setup="from __main__ import calc_nogil", number=nreps)/nreps:.2f}')