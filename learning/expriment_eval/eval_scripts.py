import sys
import os
import pickle
import logging
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from learning.utils.load_experiment import LoadExperiment
from learning.policy_pricer import policy_pricer_python
from learning.policy_pricer.cython_pricer import cython_pricer_optimized

logging.getLogger().setLevel(logging.ERROR)


def calc_checkpoint_value(ldr, accs, checkpoint, niter=5000, w_points=100, l_points=100, lam_lim=10):
    res = np.zeros(len(accs), dtype='float64')
    agent = ldr.load_agent_from_path(checkpoint);
    # model = tf.keras.models.load_model(os.path.join(checkpoint, 'main_net.h5'))
    sys.stdout.flush()
    ww, ll, p, z = policy_pricer_python.create_map(agent, w_points=w_points, l_points=l_points, lam_lim=lam_lim,
                                                   larger_offset=True)
    for i, acc in enumerate(accs):
        vals = np.asarray(cython_pricer_optimized.value_account(acc, ww, ll, p,
                                                                cython_pricer_optimized.convert_params_obj(
                                                                    agent.env.params), agent.env.env.action_bins,
                                                                n_iterations=niter))
        res[i] = np.mean(vals)
    return np.mean(res), np.std(res)


def calc_agent(ldr, name, accs, log_num=0, niter=5000, w_points=100, l_points=100, lam_lim=5, skip_checkpoints=1):
    checkpoint_paths = ldr.list_experiment_checkpoints(name,
                                                       log_number=log_num)  # ldr.list_checkpoint(name, log_num=log_num)
    vals = np.zeros(len(checkpoint_paths))
    stds = np.zeros(len(checkpoint_paths))
    for i, path in enumerate(tqdm(checkpoint_paths[::skip_checkpoints])):
        vals[i], stds[i] = calc_checkpoint_value(ldr, accs, path, niter=niter, w_points=w_points, l_points=l_points,
                                                 lam_lim=lam_lim)
    return vals, stds


def calc_agent_parallel(ldr, name, accs, log_num=0, niter=5000, w_points=100, l_points=100, lam_lim=5, n_jobs=20,
                        skip_checkpoints=1):
    checkpoint_paths = ldr.list_experiment_checkpoints(name,
                                                       log_number=log_num)  # ldr.list_checkpoints(name, log_num=log_num)
    #     vals = np.zeros(len(checkpoint_paths))
    #     stds = np.zeros(len(checkpoint_paths))
    rets = Parallel(n_jobs=n_jobs)(
        delayed(calc_checkpoint_value)(ldr, accs, path, niter=niter, w_points=w_points, l_points=l_points, lam_lim=lam_lim)
        for path in checkpoint_paths[::skip_checkpoints])
    return rets


def value_names(ldr, names, accs, file_name, n_jobs=20, log_num=0, niter=5000, w_points=100, l_points=100, lam_lim=5,
                skip_checkpoints=1):
    if os.path.isfile(os.path.join('results_regularizer', file_name + '.pkl')):
        print('Looks like there are some results already computed. Loading instead.')
        return load_obj(file_name)
    else:
        print('No file found, recomputing for a given portfolio:')
        os.makedirs('results_regularizer', exist_ok=True)
    results = {}
    for i, name in enumerate(tqdm(names)):
        if n_jobs > 1:
            results[name] = (
                calc_agent_parallel(ldr, name, accs, log_num=log_num, niter=niter, w_points=w_points, l_points=l_points,
                                    lam_lim=lam_lim, n_jobs=n_jobs, skip_checkpoints=skip_checkpoints))
        else:
            results[name] = (
                calc_agent(ldr, name, accs, log_num=log_num, niter=niter, w_points=w_points, l_points=l_points,
                           lam_lim=lam_lim, skip_checkpoints=skip_checkpoints))
        save_obj(results, file_name)  # _'mono_MC_5000_evaluated')
    return results


def save_obj(obj, name):
    with open('results_regularizer/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('results_regularizer/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



# Monotonicity computations
def calculate_mono_metric(surface):
    forward_difs_w = np.diff(surface, axis=0)
    forward_difs_l = np.diff(surface, axis=1)
    return np.mean(np.where(forward_difs_w >= 0, 1, 0)), np.mean(np.where(forward_difs_l >= 0, 1, 0))


def calc_checkpoint_mono(ldr, checkpoint, w_points, l_points, lam_lim):
    agent = ldr.load_agent_from_path(checkpoint)
    sys.stdout.flush()
    ww, ll, p, z = policy_pricer_python.create_map(agent, w_points=w_points, l_points=l_points, lam_lim=lam_lim,
                                                   larger_offset=True)
    return calculate_mono_metric(z)


def calc_agent_mono(ldr, name, log_num, w_points, l_points, lam_lim, skip_checkpoints):
    checkpoint_paths = ldr.list_experiment_checkpoints(name, log_number=log_num)
    vals = np.zeros((len(checkpoint_paths), 2))
    for i, path in enumerate(checkpoint_paths[::skip_checkpoints]):
        vals[i] = calc_checkpoint_mono(ldr, heckpoint=path, w_points=w_points, l_points=l_points, lam_lim=lam_lim)
    return vals


def calc_agent_mono_parallel(ldr, name, log_num, w_points, l_points, lam_lim, n_jobs, skip_checkpoints):
    checkpoint_paths = ldr.list_experiment_checkpoints(name, log_number=log_num)
    rets = Parallel(n_jobs=n_jobs)(
        delayed(calc_checkpoint_mono)(ldr, path, w_points=w_points, l_points=l_points, lam_lim=lam_lim) for path in
        checkpoint_paths[::skip_checkpoints])
    return rets


def calculate_mono_names(ldr, names, file_name, n_jobs=20, log_num=0, w_points=100, l_points=100, lam_lim=5,
                         skip_checkpoints=1):
    if os.path.isfile(os.path.join('results_regularizer', file_name + '.pkl')):
        print('Looks like there are some results already computed. Loading instead.')
        return load_obj(file_name)
    else:
        print('No file found, recomputing for a given portfolio:')
        os.makedirs('results_regularizer', exist_ok=True)

    results = {}
    for i, name in enumerate(tqdm(names)):
        if n_jobs > 1:
            results[name] = calc_agent_mono_parallel(ldr, name, log_num=log_num, w_points=w_points, l_points=l_points,
                                                     lam_lim=lam_lim, n_jobs=n_jobs, skip_checkpoints=skip_checkpoints)
        else:
            results[name] = calc_agent_mono(ldr, name, log_num=log_num, w_points=w_points, l_points=l_points,
                                            lam_lim=lam_lim, skip_checkpoints=skip_checkpoints)
        save_obj(results, file_name)
    return results
