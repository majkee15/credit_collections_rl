import os
import numpy as np
from dask import delayed, compute
from dask.distributed import Client

from learning.collections_env import CollectionsEnv, BetaRepayment, UniformRepayment
from learning.utils.wrappers import DiscretizedActionWrapper
from learning.policies.dqn import DQNAgent
from learning.policies.dqn_penalized import DQNAgentPenalized
from learning.policies.dqn_bspline import DQNAgentPoly
from learning.experiments.configs import *
from dcc import Parameters

# deactivate CUDA -- the nature of the code makes it comparable to CPU run appli
# with this disabled we can perform multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

@delayed
def setup_experiment(conf, name, extype='dqn'):
    params = Parameters()
    params.rho = 0.15

    actions_bins = np.array([0., 0.2, 1.0])
    n_actions = len(actions_bins)

    # rep_dist = BetaRepayment(params, 0.9, 0.5, 10, MAX_ACCOUNT_BALANCE)
    rep_dist = UniformRepayment(params)

    c_env = CollectionsEnv(params=params, repayment_dist=rep_dist, reward_shaping='continuous', randomize_start=True,
                           max_lambda=None,
                           starting_state=np.array([3, 200], dtype=np.float32)
                           )
    environment = DiscretizedActionWrapper(c_env, actions_bins)

    if extype == 'dqn':
        dqn = DQNAgent(environment, name, training=True, config=conf, initialize=False)
    elif extype == 'bspline':
        dqn = DQNAgentPoly(environment, name, training=True, config=conf)
    elif extype == 'dqnpenal':
        dqn = DQNAgentPenalized(environment, name, config=conf, training=True)
    else:
        raise ValueError('Unsupported experiment type.')

    return dqn


def runexp(experiment):
    return experiment.run_training()


def run_multiple_delayed(names, experiment_types, configs, n_repeats=1):
    res = []
    for r in range(0, n_repeats):
        for i, n in enumerate(names):
            exp = setup_experiment(configs[i], names[i] + '-' + str(r), experiment_types[i])
            res.append(delayed(runexp(exp)))
    return res


if __name__ == '__main__':
    client = Client(n_workers=40, threads_per_worker=2)

    # names = ['DQN', 'SPLINE', 'MONOSPLINE', 'DQN200p']
    names = ['PenalizedDQN', 'DQN', 'DQNL1', 'DQNL2']
    # names = ['DQN', 'DQNL1']
    experiment_types = ['dqnpenal', 'dqn', 'dqn', 'dqn']
    configs = [DQNPenalized(), DQNBaseConfig(), DQNL1high(), DQNL2low()]
    compute(run_multiple_delayed(names, experiment_types, configs, n_repeats=10), scheduler='distributed')
