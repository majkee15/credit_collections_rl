import numpy as np
from scipy.integrate import quad
from scipy.stats import kstest
import matplotlib.pyplot as plt
from scipy.stats import probplot, expon
# Collection of code snipetz for the collection environment


def lambda_bound(wstart, wtarget, p):
    """
    Computes the lambda bound of the state space for going from wstart to wtarget, provided no actions.
    Args:
        wstart (float): starting balance
        wtarget (float): ending balance
        p (Parameters):

    Returns:
        double
    """
    n = iw(wstart, wtarget, p)
    return n * (p.delta10 + p.delta11) + p.lambda0


def iw(wstart, wtarget, p):
    """
    Computes maximum number of repayments for going from wstart to wtarget.
    Args:
        wstart (float):
        wtarget (float:
        p:

    Returns:
        int
    """
    numerator = np.log(wtarget / wstart)
    denominator = np.log(1 - p.r_)
    return np.ceil(np.divide(numerator, denominator), dtype='float32')


def kernel(t: float, r, params):
    """
    Kernel evaluation at time t given
    Args:
        t (float):
        r (float):
        params (Parameters):

    Returns:

    """
    return np.sum(np.multiply(params.delta10 + params.delta11 * r, np.exp(np.multiply(-params.kappa, t))))


def getlambda(s, arrivals, reps, params):
    """
    Returns lambdas(either vector or a scalar) based on a realized process history arrivals
    Args:
        s (Union[np.ndarray, float]): time of evaluation of the intensity
        arrivals (np.ndarray):
        reps (np.ndarray):
        params (Parameters):

    Returns:
        np.ndarray

    Note:
        arrivals[arrivals <= s] cannot have a strict inequality because of the thinning algo
        for straight inequalities the next candidate lstar is smaller than the actual intensity at the point!
    """

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
    r"""
    Random time transformation -- computes residuals \xi_i = \int_{\tau_{i-1}}^{\tau_i} for all i<=n where n is
    the number of arrivals.
    If HP, \xi_i are distributed as EXP(1)
    Args:
        arrivals (np.ndarray):
        repayments (np.ndarray):
        params (Parameters):

    Returns:
        np.ndarray: list of residuals
    """

    xi = np.zeros(len(arrivals) - 1)
    for i in range(1, len(arrivals)):
        xi[i - 1] = quad(getlambda, arrivals[i - 1], arrivals[i], (arrivals, repayments, params))[0]
    return xi


def modelcheck(arrivals, repayments, params, verbose=False):
    """
    Tests whether marked HP realization given by arrrivals and repayments come from a HP using Kolgomorov-Smirnov test.
    Args:
        arrivals  (np.ndarray):
        repayments  (np.ndarray):
        params (Parameters):
        verbose (bool): print the p-value

    Returns:
        float: p-value of the KStest
    """
    # fig, ax = plt.subplots()
    residuals = _timetransform(arrivals, repayments, params)
    # res = stats.probplot(thetas, dist=stats.expon, sparams=1, fit=False, plot=sns.mpl.pyplot)
    _, p_value = kstest(residuals, 'expon', args=(0, 1))
    if verbose:
        print(f'P-value: {p_value}')
        probplot(residuals, dist=expon, fit=True, plot=plt, rvalue=False)
    return p_value


def qqplot(theoretical, empirical):
    """
    Q-Q plot of theoretical vs empirical dist.
    Args:
        theoretical:
        empirical:

    Returns:

    """
    fig, ax = plt.subplots()
    percs = np.linspace(0, 100, 201)
    qn_a = np.percentile(theoretical, percs)
    qn_b = np.percentile(empirical, percs)

    ax.plot(qn_a, qn_b, ls="", marker="o")

    x = np.linspace(np.min((qn_a.min(), qn_b.min())), np.max((qn_a.max(), qn_b.max())))
    ax.plot(x, x, color="k", ls="--")

    # fig.suptitle('Q-Q plot', fontsize=20)
    ax.xlabel('Theoretical Quantiles', fontsize=18)
    ax.ylabel('Empirical Quantiles', fontsize=16)
    return fig, ax


if __name__ == '__main__':
    print("penis")