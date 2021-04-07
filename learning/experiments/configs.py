import numpy as np

from learning.utils.misc import Config
from learning.utils.annealing_schedule import AnnealingSchedule


TOTAL_NE = 50000
LOG_EVERY = 20
PLOT_EVERY = 100

# DQN CONFIGS

class DQNBaseConfig(Config):

    n_episodes = TOTAL_NE
    warmup_episodes = int(n_episodes * 0.8)
    checkpoint_every = n_episodes / 100

    learning_rate = 0.001
    end_learning_rate = 0.00001
    # decaying learning rate
    learning_rate_schedule = AnnealingSchedule(learning_rate, end_learning_rate, n_episodes)
    # gamma (discount factor) is set as exp(-rho * dt) in the body of the learning program
    gamma = np.NaN
    epsilon = 1
    epsilon_final = 0.01
    epsilon_schedule = AnnealingSchedule(epsilon, epsilon_final, warmup_episodes)
    target_update_every_step = 50
    log_every_episode = LOG_EVERY

    # Net setting
    layers = (128, 128, 128)
    batch_normalization = False
    # Memory setting
    batch_size = 512
    memory_size = 1000000

    # PER setting
    prioritized_memory_replay = True
    replay_alpha = 0.2
    replay_beta = 0.4
    replay_beta_final = 1.0
    beta_schedule = AnnealingSchedule(replay_beta, replay_beta_final, warmup_episodes, inverse=True)
    prior_eps = 1e-6

    # progression plot
    plot_progression_flag = True
    plot_every_episode = PLOT_EVERY

    # env setting
    normalize_states = True
    regularizer = None
    constrained = False

class DQN200params(DQNBaseConfig):
    layers = (10, 10, 10)


class CDQN200paramsRegularizedL1low(DQNBaseConfig):
    regularizer = 'l1'
    regularizer_parameter = 0.01


class DQN200paramsRegularizedL2low(DQNBaseConfig):
    regularizer = 'l2'
    regularizer_parameter = 0.01


class DQN200paramsRegularizedL1high(DQNBaseConfig):
    regularizer = 'l1'
    regularizer_parameter = 0.1


class DQN200paramsRegularizedL2high(DQNBaseConfig):
    regularizer = 'l2'
    regularizer_parameter = 0.1

# SPLINE CONFIGS

class SplineBaseConfig(Config):
    # Training config specifies the hyper parameters of agent and learning
    n_episodes = TOTAL_NE
    warmup_episodes = int(n_episodes * 0.8)
    checkpoint_every = n_episodes/100
    target_update_every_step = 100

    learning_rate = 0.01
    end_learning_rate = 0.0001
    # decaying learning rate
    learning_rate_schedule = AnnealingSchedule(learning_rate, end_learning_rate, n_episodes)
    # gamma (discount factor) is set as exp(-rho * dt) in the body of the learning program
    gamma = np.NaN
    epsilon = 1
    epsilon_final = 0.01
    epsilon_schedule = AnnealingSchedule(epsilon, epsilon_final, warmup_episodes)
    log_every_episode = LOG_EVERY

    # Memory setting
    batch_size = 512
    memory_size = 1000000

    # PER setting
    prioritized_memory_replay = True
    replay_alpha = 0.2
    replay_beta = 0.4
    replay_beta_final = 1.0
    beta_schedule = AnnealingSchedule(replay_beta, replay_beta_final, warmup_episodes, inverse=True)
    prior_eps = 1e-6

    # progression plot
    plot_progression_flag = True
    plot_every_episode = PLOT_EVERY

    # env setting
    normalize_states = False

    # Approximator setting
    # Poly features dim
    poly_order = 3
    constrained = False


class SplineConstrainedConfig(SplineBaseConfig):
    constrained = True


if __name__ == '__main__':
    test_conf1 = DQN200params()
    test_conf2 = DQNBaseConfig()
    print(DQN200params.layers)
    print(DQNBaseConfig.layers)

    print(DQN200params.regularizer)

    test_conf = DQN200paramsRegularizedL2high()
    test_conf2.save('Prcinka')

