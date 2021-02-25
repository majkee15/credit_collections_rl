import numpy as np
import timeit
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import sustained_ihp_c
import matplotlib.pyplot as plt


def plot_profile(self, lhatmin, lhatmax, points, it, add_discount_time=0, plotflag=(True, True, True)):
    lambda_hats = np.linspace(lhatmin, lhatmax, points)
    lambda_hats = np.insert(lambda_hats, 0, 0)
    sust_costsmc = np.zeros_like(lambda_hats)
    jump_costsmc = np.zeros_like(lambda_hats)
    collectedmc = np.zeros_like(lambda_hats)
    arrivalsmc = np.zeros_like(lambda_hats)
    accvalsmc = np.zeros_like(lambda_hats)
    for i, hat in enumerate(lambda_hats):
        sihp.sustain_level = hat
        collected, sust_cost, jump_cost, ar = sihp.calculate_value_mc(add_discount_time, mc_iteration=it)
        accvalsmc[i] = sust_cost + jump_cost - collected
        sust_costsmc[i] = sust_cost
        jump_costsmc[i] = jump_cost
        collectedmc[i] = -collected
        arrivalsmc[i] = ar

    lhatstar_index = np.nanargmin(accvalsmc)
    lhatstar = lambda_hats[lhatstar_index]
    arstar = arrivalsmc[lhatstar_index]

    if plotflag[0]:
        plt.figure()
        plt.plot(lambda_hats, accvalsmc, marker='x')
        plt.plot(lambda_hats, collectedmc, marker='x')
        plt.axvline(x=lhatstar, color='r')
        plt.xlabel('\hat{\lambda}')
        plt.ylabel('v')
        plt.legend(['Account value', 'Amount collected'])
        plt.title(f'Balance {self.starting_balance}.')
        plt.show()
    if plotflag[1]:
        plt.figure()
        plt.plot(lambda_hats, sust_costsmc, marker='x')
        plt.plot(lambda_hats, jump_costsmc, marker='x')
        plt.plot(lambda_hats, sust_costsmc + jump_costsmc, marker='x')
        plt.xlabel('\hat{\lambda}')
        plt.ylabel('costs')
        plt.legend(['Sustain costs', 'Jump costs', 'Total'])
        plt.show()
    if plotflag[2]:
        plt.figure()
        plt.plot(lambda_hats, arrivalsmc, marker='x')
        plt.xlabel('\hat{\lambda}')
        plt.ylabel('first arrival')
        plt.show()
        print(accvalsmc)
    return lhatstar, arstar