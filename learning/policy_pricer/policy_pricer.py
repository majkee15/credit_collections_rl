import numpy as np
from joblib import delayed, Parallel
import math
from multiprocessing import cpu_count
from itertools import product

def t_equation(lambda_start, lhat, params):
    # equation for duration to reach a holding region
    try:
        numerator = (lambda_start - params.lambdainf)
        denominator = (lhat - params.lambdainf)

        if lambda_start < lhat:
            return 0
        elif denominator < 0:
            return np.NaN
        else:
            t = 1 / params.kappa * \
                np.log(numerator / denominator)
            return t
    except:
        print('warning')
        print(f'Numerator {(lambda_start - params.lambdainf)}')
        print(f'Denominator {lhat - params.lambdainf}')


def next_arrival(t_to_sustain, l, params):
    # generates interarrival times
    flag_finished = False
    s = 0
    # t_to_sustain = t_equation(l, lhat, params)
    relative_repayment = 0.0
    while not flag_finished:
        lstar = l
        w = -np.log(np.random.rand()) / lstar
        s = s + w
        d = np.random.rand()
        if d * lstar <= drift(s, l, params):
            # arrival
            relative_repayment = params.sample_repayment()[0]
            if np.isnan(t_to_sustain):
                pass
            elif s > t_to_sustain:
                relative_repayment = 0.0
                s = t_to_sustain
            flag_finished = True
    return relative_repayment, s


def drift(s, lambda_start, params):
    # deterministic draft
    return params.lambdainf + (lambda_start - params.lambdainf) * np.exp(-params.kappa * s)


def round_decimals_up(number: float, decimals: int = 2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    if np.isnan(number):
        res = np.NaN
    else:
        factor = 10 ** decimals
        res = math.ceil(number * factor) / factor

    return res


def single_collection(start_state, ww, ll, policy_map, params, action_bins):
    w_vec = ww[0, :]
    l_vec = ll[:, 0]

    current_w = start_state[1].copy()
    current_l = start_state[0].copy()
    current_time = 0

    # This will be evaluated in a loop
    reward = 0.0
    cost = 0.0

    while current_w > 1.0:

        w_ind = np.digitize(current_w, w_vec, right=True)
        l_slice = policy_map[:, w_ind]
        current_action_type = l_slice[np.digitize(current_l, l_vec, right=True)]
        current_action = action_bins[current_action_type]

        if current_action != 0.0:
            current_l = current_l + current_action * params.delta2
            cost += current_action * params.c
        else:

            lhat_slice = l_vec[(l_vec <= current_l) & (l_slice > 0)]
            if lhat_slice.shape[0] > 0:
                lhat = lhat_slice[-1]
            else:
                lhat = 0

            time_to_action = round_decimals_up(t_equation(current_l, lhat, params), 5)

            relative_repayment, arr_time = next_arrival(time_to_action, current_l, params)
            current_time += arr_time
            current_l = drift(arr_time, current_l, params)

            if relative_repayment > 0.0:
                reward += np.exp(-params.rho * current_time) * relative_repayment * current_w
                current_w = current_w * (1 - relative_repayment)
                current_l = current_l + relative_repayment * params.delta11 + params.delta10

    return reward - cost


def value_account_parallel(account, ww, ll, p, params, action_bins, n_iterations=1000):
    #     tasks = []
    #     for _ in range(n_iterations):
    #         tasks.append(single_collection(account, ww, ll, p, params))
    workers = np.minimum(32, cpu_count() - 2)
    rest = Parallel(n_jobs=workers)(
        delayed(single_collection)(account, ww, ll, p, params, action_bins) for i in range(n_iterations))
    return rest


def value_account(account, ww, ll, p, params, action_bins, n_iterations=1000):
    vals = np.zeros(n_iterations, dtype='float32')
    for i in range(n_iterations):
        vals[i] = single_collection(account, ww, ll, p, params, action_bins)
    return vals


def create_map(agent, w_points=80, l_points=80, lam_lim=7, larger_offset=False):
    if larger_offset:
        l = np.linspace(agent.env.observation_space.low[0], lam_lim * 1.2, l_points)
        w = np.linspace(agent.env.observation_space.low[1], agent.env.observation_space.high[1], w_points)

    #     w_normalized = np.linspace(0, 1, w_points)
    #     l_normalized = np.linspace(0, agent.env.observation((lam_lim,100))[0], l_points)
    #     wwn, lln = np.meshgrid(w_normalized, l_normalized)
        ww, ll = np.meshgrid(w, l)
        space_iterator = product(l, w)
        space_product = agent.env.observation(np.array([[i, j] for i, j in space_iterator]))
        predictions = agent.main_net.predict_on_batch(space_product)
        z = np.amax(predictions, axis=1).reshape(l_points, w_points)
        p = np.argmax(predictions, axis=1).reshape(l_points, w_points)
        p[int(p.shape[0]*0.9):, :] = 0.0
    else:

        l = np.linspace(agent.env.observation_space.low[0], lam_lim, l_points)
        w = np.linspace(agent.env.observation_space.low[1], agent.env.observation_space.high[1], w_points)

    #     w_normalized = np.linspace(0, 1, w_points)
    #     l_normalized = np.linspace(0, agent.env.observation((lam_lim,100))[0], l_points)
    #     wwn, lln = np.meshgrid(w_normalized, l_normalized)
        ww, ll = np.meshgrid(w, l)
        space_iterator = product(l, w)
        space_product = agent.env.observation(np.array([[i, j] for i, j in space_iterator]))
        predictions = agent.main_net.predict_on_batch(space_product)
        z = np.amax(predictions, axis=1).reshape(l_points, w_points)
        p = np.argmax(predictions, axis=1).reshape(l_points, w_points)
    return ww, ll, p, z

