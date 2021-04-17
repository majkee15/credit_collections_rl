import numpy as np
from learning.policy_pricer import pricer_naive
# from learning.policy_pricer.cython_pricer import cython_pricer_optimized
# from learning.policy_pricer.cython_pricer import cython_pricer_naive

# Imports not working -- analysis done in pricer_profiling.ipynb

if __name__ == '__main__':
    # setup a policy map
    l = np.linspace(0, 7, 500)
    w = np.linspace(0, 200, 500)
    ww, ll = np.meshgrid(w, l)

    autonomous_p = np.zeros_like(ww,    dtype='int64')
    degenerate_p = autonomous_p.copy()
    degenerate_p[:, 300:] = 1

    """
    # if plotting is needed
    plt.pcolormesh(ww, ll, degenerate_p, shading='auto')
    plt.colorbar()
    plt.show()
    """

