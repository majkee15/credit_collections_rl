import numpy as np
import timeit
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import sustained_ihp_c
import matplotlib.pyplot as plt
print(np.get_include())

l0=1
linf = 0.1
kappa = 0.7
delta10 = 0.02
delta11 = 0.5
delta2 = 1
rho = 0.05
c = 1
params = np.array([l0, linf, kappa, delta10, delta11, delta2, rho, c], dtype=np.float)


def time_py():
    return sustained_ihp_c.theta_py(2, 1, params)


def time_c():
    return sustained_ihp_c.theta(2, 1, params)

def time_c_exposed():
    return sustained_ihp_c.theta_cdef_exposed(2, 1, params)


def time_vanilla():
    lhat = 2
    l = 1
    numerator = lhat - params[1]
    denominator = l - params[1]
    if l < lhat:
        t = 0.0
    elif l - params[1] < 0:
        t = np.NaN
    else:
        t = -1 / params[2] * np.log(numerator / denominator)

    return t


def main():
    niter = 1000000
    res = np.asarray(sustained_ihp_c.calculate_value_mc(0.4, niter, 75., params))



if __name__ == '__main__':
    # profiling cpdef and cdef

    # import cProfile
    # cProfile.run('main()', sort='time')
    # print(f"Mean python execution time def: {np.mean(timeit.repeat('time_py()', globals=globals(), repeat = 10))}")
    # print(f"Mean python execution time cpdef: {np.mean(timeit.repeat('time_c()', globals=globals(), repeat = 10))}")
    # print(f"Mean python execution time cdef: {np.mean(timeit.repeat('time_c()', globals=globals(), repeat=10))}")
    # print(f"Mean python execution time vanilla: {np.mean(timeit.repeat('time_vanilla()', globals=globals(), repeat = 10))}")

    # res = []
    # n = 20000
    # for i in range(n):
    #     res.append(sustained_ihp_c.potential_repayment(0.2,1))
    # print(res)
    # plt.hist(res)
    # plt.show()
    import cProfile

    cProfile.run('main()', sort='time')
    # niter = 100000
    # res = np.asarray(sustained_ihp_c.calculate_value_mc(0.4, niter, 75., params))
    # print(res.shape)
    # print(type(res))
    # # print(res)
    # print(np.std(res, 0))
    # # print(np.mean(res, 1))