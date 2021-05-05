cimport numpy as cnp
import  numpy as np
from libc.stdlib cimport rand, srand, RAND_MAX

from libc.math cimport log, fmax, exp, ceil

from cython cimport cdivision, boundscheck, wraparound, nonecheck
from cython.parallel cimport prange, parallel

# params order
# [lambda_inf, kappa, delta_10, delta_11, delta_2, c, rho]

cdef:
    double r_ = 0.1

### Random Generators:

## Random Generators
@cdivision(True)
@wraparound(False)
@boundscheck(False)
cdef inline double r2() nogil:
    """
    C-wrapped r.v. generator U[0,1]
    :return: 
    """
    cdef double result = 0.0
    while result == 0.0:
        result = <double>rand() / <double>RAND_MAX
    return result

@cdivision(True)
@wraparound(False)
@boundscheck(False)
cdef inline double potential_repayment(double lower, double upper) nogil:
    cdef double result
    result = r2() * (upper - lower) + lower
    return result

# SUSTAINED IHP

@cdivision(True)
@boundscheck(False)
@wraparound(False)
cpdef inline double t_equation(double lambda_start, double lhat, double[::1] params) nogil:
    """
    Calculates time it takes for the account to drift from lambda_start to lhat
    Args:
        lambda_start: double
        lhat: double
        params: np.ndarray
                [lambda_inf, kappa, delta_10, delta_11, delta_2]

    Returns:
        double
    """

    # equation for duration to reach a holding region
    # try:
    cdef:
        double numerator, denominator, t

    numerator = (lambda_start - params[0])
    denominator = (lhat - params[0])

    if lambda_start <= lhat:
        return 0.0
    else:
        t = 1 / params[1] * \
            log(numerator / denominator)
        return t

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef inline double drift(double s, double lambda_start, double[::1] params) nogil:
    """
    Calculates the intensity at time s drifting from lambda start
    Args:
        s: double
        lambda_start: double 
        params: np.ndarray
            [lambda_inf, kappa, delta_10, delta_11, delta_2]
    Returns:
        double

    """
    return params[0] + (lambda_start - params[0]) * exp(-params[1] * s)


cpdef (double, double) next_arrival(double t_to_sustain, double l, double[::1] params):
    # generates interarrival times
    cdef:
        double s, lstar, relative_repayment, w, d
        bint flag_finished

    flag_finished = False
    s = 0.0
    relative_repayment = 0.0
    while not flag_finished:
        lstar = l
        w = -np.log(r2()) / lstar
        s = s + w
        d = r2()
        if d * lstar <= drift(s, l, params):
            # arrival
            relative_repayment = potential_repayment(r_, 1)
            if s > t_to_sustain:
                relative_repayment = 0.0
                s = t_to_sustain
            flag_finished = True
    return relative_repayment, s



@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef double round_decimals_up(double number, int decimals) nogil:
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    cdef:
        double res
        int factor
    if decimals == 0:
        res =  ceil(number)

    else:
        factor = 10 ** decimals
        res = ceil(number * factor) / factor

    return res


cpdef double single_collection(start_state, ww, ll, policy_map, params, action_bins):
    cdef:
        double relative_repayment, arr_time
    w_vec = ww[0, :]
    l_vec = ll[:, 0]

    l_len = len(l_vec)

    current_w = start_state[1].copy()
    current_l = start_state[0].copy()
    current_time = 0.0

    # This will be evaluated in a loop
    reward = 0.0
    cost = 0.0

    while current_w > 1.0:

        # w_ind = np.digitize(current_w, w_vec, right=True)
        w_ind = np.searchsorted(w_vec, current_w, side='left')
        l_slice = policy_map[:, w_ind]
        ind = np.searchsorted(l_vec, current_l, side='left')
        #ind = np.digitize(current_l, l_vec, right=True)

        # this prevents going out of the policy_map
        if ind >= l_len:
            current_action_type = 0
            current_action = 0.0
        else:
            current_action_type = l_slice[ind]
            current_action = action_bins[current_action_type]

        if current_action != 0.0:
            current_l = current_l + current_action * params[4]
            cost += current_action * params[5] * np.exp(-params[6] * current_time)
        else:

            lhat_slice = l_vec[(l_vec <= current_l) & (l_slice > 0)]

            if lhat_slice.shape[0] > 0:
                lhat = lhat_slice[-1]
                t_to_action = t_equation(current_l, lhat, params)
                time_to_action = round_decimals_up(t_to_action, 2)
            else:
                lhat = params[0] + 0.0001
                time_to_action = 1000

            relative_repayment, arr_time = next_arrival(time_to_action, current_l, params)
            current_time += arr_time
            current_l = drift(arr_time, current_l, params)

            if relative_repayment > 0.0:
                reward += np.exp(-params[6] * current_time) * relative_repayment * current_w
                current_w = current_w * (1 - relative_repayment)
                current_l = current_l + relative_repayment * params[3] + params[2]

    return reward - cost
#
#
def value_account(account, ww, ll, p, params, action_bins, n_iterations=1000):
    vals = np.zeros(n_iterations, dtype='float32')
    for i in range(n_iterations):
        vals[i] = single_collection(account, ww, ll, p, params, action_bins)
    return vals
