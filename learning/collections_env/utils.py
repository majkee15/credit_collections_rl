import numpy as np
from scipy.integrate import quad
from scipy.stats import kstest

# Collection of code snipetz for the collection environment

def lambda_bound(wstart, wtarget, p):
    n = iw(wstart, wtarget, p)
    return n * (p.delta10 + p.delta11) + p.lambda0


def iw(wstart, wtarget, p):
    numerator = np.log(wtarget / wstart)
    denominator = np.log(1 - p.r_)
    return np.ceil(np.divide(numerator, denominator), dtype='float32')


def kernel(t, r, params):
    # Return Kernel evaluation at time t
    return np.sum(np.multiply(params.delta10 + params.delta11 * r, np.exp(np.multiply(-params.kappa, t))))


def getlambda(s, arrivals, reps, params):
    # Returns lambdas(either vector or a scalar) based on a realized process history arrivals
    # arrivals[arrivals <= s] cannot have a strict inequality because of the thinning algo
    # for straight inequalities the next candidate lstar is smaller than the actual intensity at the point!
    if np.isscalar(s):
        lamb = params.lambdainf + (params.lambda0 - params.lambdainf) * np.exp(-params.kappa * s) + \
               np.sum(kernel(s - arrivals[arrivals <= s], reps[arrivals <= s], params))
        return lamb
    else:
        n = len(s)
        lambdas = np.zeros(n)
        for i in range(0, n):
            lambdas[i] = params.lambdainf + (params.lambda0 - params.lambdainf) * np.exp(-params.kappa * s[i]) \
                         + np.sum(kernel(s[i] - arrivals[arrivals <= s[i]], reps[arrivals <= s[i]], params))
        return lambdas


def _timetransform(arrivals, repayments, params):
    thetas = np.zeros(len(arrivals) - 1)
    for i in range(1, len(arrivals)):
        thetas[i - 1] = quad(getlambda, arrivals[i - 1], arrivals[i], (arrivals, repayments, params))[0]
    return thetas


def modelcheck(arrivals, repayments, params, verbose=False):
    # fig, ax = plt.subplots()
    residuals = _timetransform(arrivals, repayments, params)
    # res = stats.probplot(thetas, dist=stats.expon, sparams=1, fit=False, plot=sns.mpl.pyplot)
    _, p_value = kstest(residuals, 'expon', args=(0, 1))
    if verbose:
        print(f'P-value: {p_value}')
    return p_value
