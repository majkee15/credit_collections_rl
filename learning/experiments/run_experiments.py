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
from learning.experiments.configs_memory_variation import *
from dcc import Parameters

from learning.utils.portfolio_accounts import load_acc_portfolio, generate_portfolio

# deactivate CUDA -- the nature of the code makes it comparable to CPU run appli
# with this disabled we can perform multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@delayed
def setup_experiment(conf, name, extype='dqn', use_portfolio=False, experiment_name=None):
    if use_portfolio:
        n_acc = 50
        portfolio = generate_portfolio(n_acc)
    else:
        portfolio = None

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
        dqn = DQNAgent(environment, name, training=True, config=conf, initialize=False, portfolio=portfolio,
                       experiment_name=experiment_name)
    elif extype == 'bspline':
        dqn = DQNAgentPoly(environment, name, training=True, config=conf, portfolio=portfolio,
                           experiment_name=experiment_name)
    elif extype == 'dqnpenal':
        dqn = DQNAgentPenalized(environment, name, config=conf, training=True, portfolio=portfolio,
                                experiment_name=experiment_name)
    else:
        raise ValueError('Unsupported experiment type.')

    return dqn


def runexp(experiment):
    return experiment.run_training()


def run_multiple_delayed(names, experiment_types, configs, n_repeats=1, experiment_name=None, description=None,
                         use_portfolio=False):
    res = []
    for r in range(0, n_repeats):
        for i, n in enumerate(names):
            exp = setup_experiment(configs[i], names[i] + '-' + str(r), experiment_types[i],
                                   experiment_name=experiment_name,
                                   use_portfolio=use_portfolio)
            res.append(delayed(runexp(exp)))
    return res


if __name__ == '__main__':
    client = Client(n_workers=50, threads_per_worker=2)
    experiment_description = ""
    experiment_name = 'testing_new_implementation'
    names = ['PDQN', 'DQNV', 'BSpline']
    experiment_types = ['dqnpenal', 'dqn', 'bspline']
    configs = [PDQNFastLrn(), DQNFastLrn(), SplineConstrainedConfig()]
    compute(run_multiple_delayed(names, experiment_types, configs, n_repeats=10, use_portfolio=False,
                                 experiment_name=experiment_name), scheduler='distributed')

