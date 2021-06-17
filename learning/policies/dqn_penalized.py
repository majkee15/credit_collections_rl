import os
import numpy as np
import tensorflow as tf
import datetime
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as m

from learning.collections_env import CollectionsEnv, BetaRepayment, UniformRepayment
from learning.utils.wrappers import DiscretizedActionWrapper, StateNormalization

from learning.policies.memory import Transition, ReplayMemory, PrioritizedReplayMemory
from learning.policies.base import Policy, BaseModelMixin
from learning.utils.misc import plot_learning_curve, plot_to_image
from learning.utils.construct_poly import construct_poly_approx
from learning.utils.annealing_schedule import LinearSchedule

from learning.policies.dqn import DefaultConfig, DQNAgent

from learning.utils.portfolio_accounts import load_acc_portfolio, generate_portfolio

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class DefaultConfigPenal(DefaultConfig):
    n_episodes = 10000
    warmup_episodes = n_episodes * 0.8
    checkpoint_every = 1000
    log_every_episode = 10
    constrained = True
    penal_coeff = 0.5
    penalize_after = 2000
    penal_cofff_schedule = LinearSchedule(0.0, 0.5, 8000, 1000)

class DQNAgentPenalized(DQNAgent):

    def __init__(self, env, name, config=None, training=True, portfolio=None, experiment_name=None):
        super().__init__(env=env, name=name, config=config, training=training, portfolio=portfolio, experiment_name=experiment_name)

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
        network_input_to_watch = tf.Variable(tf.convert_to_tensor(states, dtype='float32'))

        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)
            with tf.GradientTape() as tape_inner:
                tape_inner.watch(network_input_to_watch)
                # simple dqn
                # target_q = self.target_net.predict_on_batch(next_states)
                # next_action = np.argmax(target_q.numpy(), axis=1)
                # double_dqn -- this is a feature, not a bug
                target_q = self.main_net(next_states)
                next_action = np.argmax(target_q.numpy(), axis=1)
                target_q = self.target_net(next_states)

                target_value = tf.reduce_sum(tf.one_hot(next_action, self.act_size) * target_q, axis=1)
                target_value = (1 - dones) * self.config.gamma * target_value + rewards

                main_q = self.main_net(network_input_to_watch)
                main_value = tf.reduce_sum(tf.one_hot(actions, self.act_size) * main_q, axis=1)
                # main_q_to_penalize = tf.identity(main_q)
            if self.episode > self.config.penalize_after:
                penalization = self.config.penal_coeff * tf.reduce_sum(
                    tf.maximum(-tape_inner.gradient(main_value, network_input_to_watch), 0.0))
            else:
                penalization = tf.constant(0.0)

            td_error = target_value - main_value
            element_wise_loss = tf.square(td_error) * 0.5

            if self.config.prioritized_memory_replay:
                error = tf.reduce_mean(element_wise_loss * weights)

            else:
                error = tf.reduce_mean(element_wise_loss)

            error = error + penalization

        dqn_grads = tape.gradient(error, dqn_variable)

        # Update priorities
        if self.config.prioritized_memory_replay:
            self.memory.update_priorities(idx, np.abs(td_error.numpy()) + self.config.prior_eps)

        self.optimizer.apply_gradients(zip(dqn_grads, dqn_variable))
        # Logging
        self.global_step += 1
        self._tb_log_holder = {'Gradients': dqn_grads[0], 'Weights': tf.convert_to_tensor(self.main_net.weights[0]),
                               'Prediction': main_value,
                               'Target': target_value, 'TD error': td_error, 'Elementwise Loss': element_wise_loss,
                               'Loss': tf.reduce_mean(element_wise_loss), 'Penalization': penalization}
        return tf.reduce_sum(element_wise_loss)


if __name__ == '__main__':
    from dcc import Parameters
    seed = 3
    # np.random.seed(seed)
    # tf.random.set_seed(seed)
    #
    portfolio_acc = generate_portfolio(50)

    MAX_ACCOUNT_BALANCE = 200.0

    params = Parameters()
    params.rho = 0.15

    actions_bins = np.array([0, 0.2, 0.5, 1.0])
    n_actions = len(actions_bins)

    rep_dist = UniformRepayment(params)

    c_env = CollectionsEnv(params=params, repayment_dist=rep_dist, reward_shaping='continuous', randomize_start=True,
                           max_lambda=None,
                           starting_state=np.array([3.0, 200.0], dtype=np.float32)
                           )
    environment = DiscretizedActionWrapper(c_env, actions_bins)

    dqn = DQNAgentPenalized(environment, name='DQNConstrainedTest', training=True, config=DefaultConfigPenal(),
                            portfolio=None)
    dqn.run_training()
