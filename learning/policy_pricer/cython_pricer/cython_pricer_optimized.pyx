cimport numpy as cnp
import  numpy as np
from libc.stdlib cimport rand, srand, RAND_MAX

from libc.math cimport log, fmax, exp, ceil

from cython cimport cdivision, boundscheck, wraparound, nonecheck
from cython.parallel cimport prange, parallel

# params order
# [lambda_inf, kappa, delta_10, delta_11, delta_2, c, rho]

cdef double r_ = 0.1

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
    cdef double result
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
cdef inline double t_equation(double lambda_start, double lhat, double[::1] params) nogil:
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
        return 0
    else:
        t = 1 / params[1] * \
            log(numerator / denominator)
        return t

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef inline double drift(double s, double lambda_start, double[::1] params) nogil:
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
    cdef:
        double result
    return params[0] + (lambda_start - params[0]) * exp(-params[1] * s)

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef (double, double) next_arrival(double t_to_sustain, double l, double [::1] params) nogil:
    """
    
    Args:
        t_to_sustain: 
        l: 
        params: 

    Returns: np.ndarray
            [relative_repayment, arrival_time]

    """
    cdef:
        double s = 0.0, relative_repayment = 0.0
        double w, lstar
        bint flag_finished = False
        double [::1] result

    while not flag_finished:
        lstar = l
        w = -log(r2()) / lstar
        s = s + w
        d = r2()
        if d * lstar <= drift(s, l, params):
            # arrival
            relative_repayment = potential_repayment(r_, 1)
        elif s > t_to_sustain:
            relative_repayment = 0.0
            s = t_to_sustain
        flag_finished = True
    return relative_repayment, s

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef double single_collection(double [::1] start_state, double [:, ::1] ww, ll, policy_map, double [::1] params, double [::1] action_bins):
    """
    
    Args:
        start_state: 
        ww: 
        ll: 
        policy_map: 
        params: 
        action_bins: 

    Returns:

    """
    cdef:
        double [::1] w_vec, lhat_slice
        #int [::1] lslice
        double current_time = 0.0, reward = 0.0, cost = 0.0
        double current_w, current_l, current_action, arr_time, relative_repayment, time_to_action, lhat
        int w_ind, current_action_type



    w_vec = ww[0, :]
    l_vec = ll[:, 0]

    current_w = start_state[1]
    current_l = start_state[0]

    while current_w > 1.0:

        w_ind = np.digitize(current_w, w_vec, right=True)
        l_slice = policy_map[:, w_ind]
        current_action_type = l_slice[np.digitize(current_l, l_vec, right=True)]
        current_action = action_bins[current_action_type]

        if current_action != 0.0:
            current_l = current_l + current_action * params[4]
            cost += current_action * params[5]
        else:

            lhat_slice = l_vec[(l_vec <= current_l) & (l_slice > 0)]
            if lhat_slice.shape[0] > 0.0:
                lhat = lhat_slice[-1]
            else:
                lhat = 0.0


            time_to_action = round_decimals_up(t_equation(current_l, lhat, params), 5)

            relative_repayment, arr_time = next_arrival(time_to_action, current_l, params)
            current_time += arr_time
            current_l = drift(arr_time, current_l, params)

            if relative_repayment > 0.0:
                reward += exp(-params[6] * current_time) * relative_repayment * current_w
                current_w = current_w * (1 - relative_repayment)
                current_l = current_l + relative_repayment * params[3] + params[2]

    return reward - cost

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef double single_collection_fast(double [::1] start_state, double [:, ::1] ww, double[:, ::1] ll, cnp.int64_t[:, ::1] policy_map, double [::1] params, double [::1] action_bins) nogil:
    """
    
    Args:
        start_state: 
        ww: 
        ll: 
        policy_map: 
        params: 
        action_bins: 

    Returns:

    """
    cdef:
        double [::1] w_vec
        cnp.int64_t [:] l_slice
        double [:] l_vec, lhat_slice
        double current_time = 0.0, reward = 0.0, cost = 0.0
        double current_w, current_l, current_action, arr_time, relative_repayment, time_to_action, lhat
        cnp.int64_t w_ind, index_lhatslice = 0, current_action_type, lshape, l_ind



    w_vec = ww[0, :]
    l_vec = ll[:, 0]
    lshape = len(l_vec)

    current_w = start_state[1]
    current_l = start_state[0]

    while current_w > 1.0:

        w_ind = mySearchSorted(w_vec, current_w)
        l_slice = policy_map[:, w_ind]
        l_ind= mySearchSorted(l_vec, current_l)

        if l_ind >= lshape:
            current_action_type = 0
            current_action = 0.0
        else:
            current_action_type = l_slice[l_ind]
            current_action = action_bins[current_action_type]

        if current_action != 0.0:
            current_l = current_l + current_action * params[4]
            cost += current_action * params[5]
        else:

            # lhat_slice = l_vec[(l_vec <= current_l) & (l_slice > 0)]
            for i in range(lshape):
                if l_vec[i] <= current_l and l_slice[i]>0:
                    index_lhatslice = index_lhatslice + 1

            lhat_slice = l_vec[:index_lhatslice]




            if  index_lhatslice> 0:
                lhat = lhat_slice[index_lhatslice]
            else:
                lhat = 0

            time_to_action = round_decimals_up(t_equation(current_l, lhat, params), 5)

            relative_repayment, arr_time = next_arrival(time_to_action, current_l, params)
            current_time += arr_time
            current_l = drift(arr_time, current_l, params)

            if relative_repayment > 0.0:
                reward += exp(-params[6] * current_time) * relative_repayment * current_w
                current_w = current_w * (1 - relative_repayment)
                current_l = current_l + relative_repayment * params[3] + params[2]

    return reward - cost


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef double round_decimals_up(double number, int decimals) nogil:
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

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef double [::1] value_account(double [::1] account, double [:, ::1] ww, ll, policy_map, double [::1] params, double [::1] action_bins, int n_iterations=1000):
    cdef:
        double [::1] vals

    vals = np.zeros(n_iterations, dtype=np.float64)
    for i in range(n_iterations):
        vals[i] = single_collection(account, ww, ll, policy_map, params, action_bins)
    return vals


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef double [::1] value_account_fast(double [::1] account, double [:, ::1] ww, double[:, ::1] ll, cnp.int64_t[:, ::1] policy_map, double [::1] params, double [::1] action_bins, int n_iterations=1000):
    cdef:
        double [::1] vals = np.zeros(n_iterations, dtype=np.float64)
        int i
    for i in range(n_iterations):
        vals[i] = single_collection_fast(account, ww, ll, policy_map, params, action_bins)
    return np.asarray(vals)


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef int mySearchSorted(double[:] array, double target) nogil:
    """
    Binary search
    Args:
        array: 
        target: 

    Returns:

    """
    cdef:
        int left = 0, right = len(array), mid, n = len(array)

    while left <= right:

        mid = (left + right) // 2

        if mid == n:
            return n

        if array[mid] == target:
            return mid
        if array[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef int mySearchSorted_callable(double[:] array, double target) nogil:
    """
    Binary search
    Args:
        array: 
        target: 

    Returns:

    """
    cdef:
        int left = 0, right = len(array), mid, n = len(array)

    while left <= right:

        mid = (left + right) // 2

        if mid == n:
            return n

        if array[mid] == target:
            return mid
        if array[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def convert_params_obj(params):
    retarr = np.array([params.lambdainf, params.kappa, params.delta10, params.delta11, params.delta2, params.c, params.rho], dtype=np.float64)
    return retarr
