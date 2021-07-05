import os
import numpy as np
import tensorflow as tf
import datetime
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as m

from learning.collections_env import CollectionsEnv, BetaRepayment, UniformRepayment
from learning.utils.wrappers import DiscretizedActionWrapper, StateNormalization, SplineObservationWrapper

from learning.policies.memory import Transition, ReplayMemory, PrioritizedReplayMemory
from learning.policies.base import Policy, BaseModelMixin, TrainConfigBase
from learning.utils.misc import plot_learning_curve, plot_to_image
from learning.utils.construct_poly import construct_spline_approx
from learning.utils.annealing_schedule import LinearSchedule

from sklearn.preprocessing import PolynomialFeatures

from learning.policies.dqn import DQNAgent

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class DefaultConfig(TrainConfigBase):
    # Training config specifies the hyperparameters of agent and learning
    n_episodes = 1000
    warmup_episodes = n_episodes * 0.8
    checkpoint_every = 10
    target_update_every_step = 100

    learning_rate = 0.01
    end_learning_rate = 0.0001
    # decaying learning rate
    learning_rate_schedule = LinearSchedule(learning_rate, end_learning_rate, n_episodes)
    # gamma (discount factor) is set as exp(-rho * dt) in the body of the learning program
    gamma = np.NaN
    epsilon = 1
    epsilon_final = 0.01
    epsilon_schedule = LinearSchedule(epsilon, epsilon_final, warmup_episodes)
    log_every_episode = 10

    # Memory setting
    batch_size = 512
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
    plot_every_episode = target_update_every_step

    # env setting
    normalize_states = False

    # Approximator setting
    # Poly features dim
    poly_order = 3
    constrained = False
    penalize_after = 2000
    penal_coeff = 0.1
    n_l_knots = 6
    n_w_knots = 6

    penal_coeff_schedule = LinearSchedule(0.0, 0.1, 500, inverse=True, delay=0)

    # repayment distribution:


class DQNAgentPoly(DQNAgent):

    def __init__(self, env, name, config=None, training=True, experiment_name=None, portfolio=None):

        DQNAgent.__init__(self, env, name, config, training, experiment_name=experiment_name, portfolio=portfolio)

        self.env = SplineObservationWrapper(self.env, n_l_knots=config.n_l_knots, n_w_knots=config.n_w_knots,
                                            normalized=config.normalize_states)

    def train(self, *args, **kwargs):
        batch = self.memory.sample(self.config.batch_size)
        states = batch['s']
        actions = batch['a']
        rewards = batch['r']
        next_states = batch['s_next']
        dones = batch['done']
        if self.config.prioritized_memory_replay:
            idx = batch['indices']
            weights = batch['weights']

        dqn_variable = self.main_net.trainable_variables

        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)
            # simple dqn
            # target_q = self.target_net.predict_on_batch(next_states)
            # next_action = np.argmax(target_q.numpy(), axis=1)
            # double_dqn
            target_q = self.main_net(next_states)
            next_action = np.argmax(target_q.numpy(), axis=1)
            target_q = self.target_net(next_states)

            target_value = tf.reduce_sum(tf.one_hot(next_action, self.act_size) * target_q, axis=1)
            target_value = (1 - dones) * self.config.gamma * target_value + rewards

            main_q = self.main_net(states)
            main_value = tf.reduce_sum(tf.one_hot(actions, self.act_size) * main_q, axis=1)

            td_error = target_value - main_value
            element_wise_loss = tf.square(td_error) * 0.5

            if self.config.constrained:
                first_l, first_w = self.env.penalization(states[:, 0], states[:, 1])
                # penalization = np.sum(np.maximum(-np.matmul(first_l, self.main_net.weights[0].numpy()), 0 ) +\
                #                np.maximum(-np.matmul(first_w, self.main_net.weights[0].numpy()), 0))

                # power = tf.math.floor(kwargs['epoch'] / 50)
                # coeff = tf.math.minimum(10e4, 2 * tf.math.pow(power, 2))
                coeff = self.config.penal_coeff_schedule.current_p
                penalization = coeff * 0.5 * (
                        tf.reduce_sum(tf.square(tf.math.maximum(-tf.matmul(first_l, dqn_variable), 0))) +
                        tf.reduce_sum(tf.square(tf.math.maximum(-tf.matmul(first_w, dqn_variable), 0))))
            else:
                penalization = tf.constant(0.0)

            if self.config.prioritized_memory_replay:
                error = tf.reduce_mean(element_wise_loss * weights)
            else:
                error = tf.reduce_mean(element_wise_loss)

            error = error + penalization

        if self.config.prioritized_memory_replay:
            self.memory.update_priorities(idx, np.abs(td_error.numpy()) + self.config.prior_eps)

        # loss = self.main_net.train_on_batch(states, target_value)
        dqn_grads = tape.gradient(error, dqn_variable)

        self.optimizer.apply_gradients(zip(dqn_grads, dqn_variable))
        # Logging
        self.global_step += 1
        self._tb_log_holder = {'Gradients': dqn_grads[0], 'Weights': tf.convert_to_tensor(self.main_net.weights[0]),
                               'Prediction': main_value,
                               'Target': target_value, 'TD error': td_error, 'Elementwise Loss': element_wise_loss,
                               'Loss': tf.reduce_mean(element_wise_loss), 'Penalization': penalization}
        return tf.reduce_mean(element_wise_loss)

    def build(self, initialize=False):
        self.target_net = construct_spline_approx(self.env, self.env.total_features)
        self.main_net = tf.keras.models.clone_model(self.target_net)
        self.main_net.set_weights(self.target_net.get_weights())
        self._space_product = self.env.observation(np.array([[i, j] for i, j in self._space_iterator]))


if __name__ == '__main__':
    from dcc import Parameters

    params = Parameters()
    params.rho = 0.15

    actions_bins = np.array([0, 0.2, 0.5, 1.0])
    n_actions = len(actions_bins)

    rep_dist = UniformRepayment(params)

    c_env = CollectionsEnv(params=params, repayment_dist=rep_dist, reward_shaping='continuous', randomize_start=True,
                           max_lambda=None,
                           starting_state=np.array([3, 200], dtype=np.float32)
                           )
    environment = DiscretizedActionWrapper(c_env, actions_bins)

    dqn = DQNAgentPoly(environment, 'BS4ActsConstrTest', training=True, config=DefaultConfig())
    dqn.run_training()
