from chp import CHP
from multiprocessing import cpu_count, Pool
from functools import partial
import timeit
import numpy as np
import copy

# This investigates ideal design for multiprocess pricing of an account

def no_control(w):
    return np.zeros_like(w)


def price(chp):
    return chp.calculate_value_single()


def instantiate_list(chp, control, mc_iterations=10000):
    instance = partial(instantiate_single, chp, control)
    return [instance() for i in range(mc_iterations)]


def instantiate_list_parallel(chp, control, mc_iterations=10000):
    worker = partial(instantiate_single, chp, control)
    um_cores = cpu_count()
    with Pool(um_cores-1) as p:
        return p.map(worker, range(mc_iterations))


def instantiate_single(chp, control, _=None):
    # return CHP(starting_balance=100, starting_intensity=1, marginal_cost=1,
    #     collection_horizon=100, lambda_infty=0.1, kappa=0.7,
    #     delta10=0.02, delta11=0.5, control_function=control, rho=0.06)
    return copy.deepcopy(chp)


def calculate_value_parallel(chp, control, niter, parallel_instantiation=False):

    if parallel_instantiation:
        process_list = instantiate_list_parallel(chp, control, niter)
    else:
        process_list = instantiate_list(chp, control, niter)

    um_cores = cpu_count()
    with Pool(um_cores - 1) as p:
        return np.mean(p.map(price, process_list))


def calculate_value_single(chp, control, niter):
    instance = instantiate_single(chp, control)
    return instance.calculate_value(niter)


def time_class_creation(chp, control, n_classes=10000):
    instantiate_classes = partial(instantiate_list, chp, control, n_classes)
    instantiate_classes_parallel = partial(instantiate_list_parallel, chp, control, n_classes)
    instantiate_class = partial(instantiate_single, chp, control, n_classes)
    print(f' Instantiating single class: {timeit.timeit(instantiate_class, number=n_classes) / n_classes}')
    print(f'Instantiating array of {n_classes} classes: {timeit.timeit(instantiate_classes, number=1)}')
    print(f'Instantiating array of {n_classes} classes using parallel: {timeit.timeit(instantiate_classes_parallel, number=1)}')


def time_valuations(chp, control, mc_iterations=10000):
    n_of_evaluations = 10
    single = partial(calculate_value_single, chp, control, mc_iterations)
    parallel_single_list = partial(calculate_value_parallel, chp, control, mc_iterations, False)
    parallel_parallel_list = partial(calculate_value_parallel, chp, control, mc_iterations, True)
    print(f'Single_value: {single()}')
    print(f'Parallel-simple value: {parallel_single_list()}')
    print(f'Parallel-parallel value: {parallel_parallel_list()}')
    print(f' Time valuation - simple for {mc_iterations} iterations: '
          f'{timeit.timeit(single, number=n_of_evaluations) / n_of_evaluations}')
    print(f' Time valuation - parallel - simple list for {mc_iterations} iterations: '
          f'{timeit.timeit(parallel_single_list, number=n_of_evaluations) / n_of_evaluations}')
    print(f' Time valuation - parallel - parallel list for {mc_iterations} iterations: '
          f'{timeit.timeit(parallel_parallel_list, number=n_of_evaluations) / n_of_evaluations}')
    pass


if __name__ == '__main__':
    chp = CHP(starting_balance=100, starting_intensity=0.3, marginal_cost=1,
              collection_horizon=10, lambda_infty=0.1, kappa=1,
              delta10=0.05, delta11=0.5, control_function=no_control, rho=0.05)
    iterations = 5000
    # time_class_creation(chp, no_control, n_classes=iterations)
    time_valuations(chp, no_control, iterations)
