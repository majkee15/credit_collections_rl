# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

cimport numpy as cnp
import  numpy as np
from libc.math cimport log, fmax, exp
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time,time_t
from cython cimport cdivision, boundscheck, wraparound, nonecheck
from cython.parallel cimport prange, parallel

cdef double[::1] single_result = np.empty(4)
srand(time(NULL))

cdef double r_ = 0.1

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

@cdivision(True)
@wraparound(False)
@boundscheck(False)
cdef inline double potential_repayment_generalized(double lower, double upper, double pi) nogil:
    cdef:
        double result
        double draw

    draw = r2()
    if draw < pi:
        result = lower
    else:
        result = upper

    return result

## Sustained IHP functions

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef double drift(double s, double lambda_start, double[::1] params) nogil:
    cdef:
        double result
    return params[1] + (lambda_start - params[1]) * exp(-params[2] * s)

@cdivision(True)
@wraparound(False)
@boundscheck(False)
cdef inline double[::1] next_arrival(double lstart, double lhat, double balance, double[::1] params) nogil:
    """
    Generates the next arrival of a sustained IHP.
    Args:
        lstart: double
            starting intensity
        lhat: double
            sustain intensity level
        balance: double
            current balance
        params: object
            class defining CHP parameters
    Returns:

    """
    cdef:
        double s = 0.0
        double t_to_sustain = 0.0
        double w, lstar, intensity_at_jump, collected, sust_cost, jump_cost
        bint condition = True
        #double[::1] single_result = np.empty(4)
    t_to_sustain = theta_cdef(lstart, lhat, params)
    jump_size = fmax(0, lhat - lstart)
    while condition:
        lstar = fmax(lhat, lstart)
        w = -log(r2()) / lstar
        s = s + w
        d = r2()
        intensity_at_jump_minus = fmax(drift(s,lstart, params), lhat)
        if d * lstar <= intensity_at_jump_minus:
            sustain_drift_time = 0.0
            relative_repayment = potential_repayment(r_, 1)
            collected = balance * relative_repayment * exp(-params[6] * s)
            if jump_size > 0.0:
                sust_cost = params[7] * (params[2] * (lhat - params[1]) *
                        (1 - exp(-params[6] * s))/params[6])
            elif t_to_sustain < s:
                sust_cost = params[7] * (params[2] * (lhat - params[1]) *
                        (exp(-params[6] * t_to_sustain) - exp(-params[6] * s))/params[6])
            else:
                sust_cost = 0.0
            jump_cost = jump_size * params[7]
            # single_result = np.array([s, collected, sust_cost, jump_cost])
            single_result[0] = s
            single_result[1] = collected
            single_result[2] = sust_cost
            single_result[3] = jump_cost
            condition = False
    return single_result

@cdivision(True)
@wraparound(False)
@boundscheck(False)
cpdef double[:,::1] calculate_value_mc(double sustain_level, int niter, double balance, double[::1] params):
    """
    Monte Carlo calculation of a repayment under sustain intensity at lhat
    Args:
        sustain_level: double
        niter: int
        balance: double
        params: object

    Returns: np.ndarray(dim=2)
        rows - iterations
        columns - arrival time, collected, sustain_cost, jump_cost
    """
    cdef:
        int i, j, data_dim
        double running_sust_costs, running_jump_costs, running_collected, running_s
        double[::1] single_arrival_data# , value_data
        double[:, ::1] value_data = np.zeros((niter, 4), dtype=np.float_)
    for i in range(niter):
        value_data[i,:] = next_arrival(params[0], sustain_level, balance, params)
    return value_data


@cdivision(True)
@wraparound(False)
@boundscheck(False)
cpdef double[:,::1] calculate_value_mc_nogil(double sustain_level, int niter, double balance, double[::1] params):
    """
    Multi-threading version of calculate_value_mc
    Monte Carlo calculation of a repayment under sustain intensity at lhat
    Args:
        sustain_level: double
        niter: int
        balance: double
        params: object

    Returns: np.ndarray(dim=2)
        rows - iterations
        columns - arrival time, collected, sustain_cost, jump_cost
    """
    cdef:
        int i, j, data_dim
        double running_sust_costs, running_jump_costs, running_collected, running_s
        double[::1] single_arrival_data# , value_data
        double[:, ::1] value_data = np.zeros((niter, 4), dtype=np.float_)
    for i in prange(niter, nogil=True, schedule='static'):
        value_data[i,:] = next_arrival(params[0], sustain_level, balance, params)
    return value_data

@cdivision(True)
@boundscheck(False)
@wraparound(False)
cdef inline double theta_cdef(double l, double lhat, double[::1] params) nogil:
    """
    :param l: lambda start
    :param lhat: lambda sustain level
    :param params: vector of parameters
    [l0, linf, kappa, delta10, delta11, delta2, rho, c]
    :return: drift time to sustain level
    """
    cdef:
        double numerator
        double denominator
        double t
    numerator = lhat - params[1]
    denominator = l - params[1]
    if l < lhat:
        # indicates jump happens immediately
        t = 0.0
    elif l - params[1] < 0:
        #starting intensity cannot be bellow lambdainf
        t = -1.0
    else:
        t = -1/params[2] * log(numerator / denominator)

    return t

# GENERALIZED VERSION HERE

@cdivision(True)
@wraparound(False)
@boundscheck(False)
cdef inline double[::1] next_arrival_generalized(double lstart, double lhat, double balance, double[::1] params, double lower, double upper, double pi) nogil:
    """
    Generates the next arrival of a sustained IHP.
    Args:
        lstart: double
            starting intensity
        lhat: double
            sustain intensity level
        balance: double
            current balance
        params: object
            class defining CHP parameters
        lower: double
        
        upper: double
        
        pi: double
    Returns:

    """
    cdef:
        double s = 0.0
        double t_to_sustain = 0.0
        double w, lstar, intensity_at_jump, collected, sust_cost, jump_cost
        bint condition = True
        #double[::1] single_result = np.empty(4)
    t_to_sustain = theta_cdef(lstart, lhat, params)
    jump_size = fmax(0, lhat - lstart)
    while condition:
        lstar = fmax(lhat, lstart)
        w = -log(r2()) / lstar
        s = s + w
        d = r2()
        intensity_at_jump_minus = fmax(drift(s,lstart, params), lhat)
        if d * lstar <= intensity_at_jump_minus:
            sustain_drift_time = 0.0
            relative_repayment = potential_repayment_generalized(lower, upper, pi)
            collected = balance * relative_repayment * exp(-params[6] * s)
            if jump_size > 0.0:
                sust_cost = params[7] * (params[2] * (lhat - params[1]) *
                        (1 - exp(-params[6] * s))/params[6])
            elif t_to_sustain < s:
                sust_cost = params[7] * (params[2] * (lhat - params[1]) *
                        (exp(-params[6] * t_to_sustain) - exp(-params[6] * s))/params[6])
            else:
                sust_cost = 0.0
            jump_cost = jump_size * params[7]
            # single_result = np.array([s, collected, sust_cost, jump_cost])
            single_result[0] = s
            single_result[1] = collected
            single_result[2] = sust_cost
            single_result[3] = jump_cost
            condition = False
    return single_result

@cdivision(True)
@wraparound(False)
@boundscheck(False)
cpdef double[:,::1] calculate_value_mc_gen(double sustain_level, int niter, double balance, double[::1] params, double lower, double upper, double pi):
    """
    Monte Carlo calculation of a repayment under sustain intensity at lhat
    Args:
        sustain_level: double
        niter: int
        balance: double
        params: object
        lower: double
        
        upper: double
        
        pi: double
        

    Returns: np.ndarray(dim=2)
        rows - iterations
        columns - arrival time, collected, sustain_cost, jump_cost
    """
    cdef:
        int i, j, data_dim
        double running_sust_costs, running_jump_costs, running_collected, running_s
        double[::1] single_arrival_data# , value_data
        double[:, ::1] value_data = np.zeros((niter, 4), dtype=np.float_)
    for i in range(niter):
        value_data[i,:] = next_arrival_generalized(params[0], sustain_level, balance, params, lower, upper, pi)
    return value_data


@cdivision(True)
@wraparound(False)
@boundscheck(False)
cpdef double[:,::1] calculate_value_mc_gen_nogil(double sustain_level, int niter, double balance, double[::1] params,
                                             double lower, double upper, double pi):
    """
    Multi-threading version of calculate_value_mc
    Monte Carlo calculation of a repayment under sustain intensity at lhat
    Args:
        sustain_level: double
        niter: int
        balance: double
        params: object
        lower: double
        
        upper: double
        
        pi: double

    Returns: np.ndarray(dim=2)
        rows - iterations
        columns - arrival time, collected, sustain_cost, jump_cost
    """
    cdef:
        int i, j, data_dim
        double running_sust_costs, running_jump_costs, running_collected, running_s
        double[::1] single_arrival_data# , value_data
        double[:, ::1] value_data = np.zeros((niter, 4), dtype=np.float_)
    for i in prange(niter, nogil=True, schedule='static'):
        value_data[i,:] = next_arrival_generalized(params[0], sustain_level, balance, params, lower, upper, pi)
    return value_data

