import numpy as np

from learning.utils.misc import Config
from learning.utils.annealing_schedule import LinearSchedule


TOTAL_NE = 15000
LOG_EVERY = 10
PLOT_EVERY = 100
CHECKPOINT_EVERY = 10


class DQNFastLrn(Config):

    n_episodes = TOTAL_NE
    warmup_episodes = int(n_episodes * 0.8)
    checkpoint_every = CHECKPOINT_EVERY

    learning_rate = 0.001
    end_learning_rate = 0.00001
    # decaying learning rate
    learning_rate_schedule = LinearSchedule(learning_rate, end_learning_rate, int(n_episodes * 0.2))
    # gamma (discount factor) is set as exp(-rho * dt) in the body of the learning program
    gamma = np.NaN
    epsilon = 1
    epsilon_final = 0.01
    epsilon_schedule = LinearSchedule(epsilon, epsilon_final, warmup_episodes)
    target_update_every_step = 50
    log_every_episode = LOG_EVERY

    # Net setting
    layers = (128, 128, 128)
    batch_normalization = False
    # Memory setting
    batch_size = 128
    memory_size = 1000000

    # PER setting
    prioritized_memory_replay = True
    replay_alpha = 0.2
    replay_beta = 0.4
    replay_beta_final = 1.0
    beta_schedule = LinearSchedule(replay_beta, replay_beta_final, warmup_episodes, inverse=True)
    prior_eps = 1e-6

    # progression plot
    plot_progression_flag = True
    plot_every_episode = PLOT_EVERY

    # env setting
    normalize_states = True
    regularizer = None
    constrained = False

    compute_portfolio_every = 100


class PDQNFastLrn(DQNFastLrn):
    constrained = True
    penal_coeff = 0.1



