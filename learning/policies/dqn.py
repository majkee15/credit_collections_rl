import os
import numpy as np
import tensorflow as tf
import pickle
from itertools import product
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import matplotlib as m

from learning.collections_env import CollectionsEnv, UniformRepayment
from learning.utils.wrappers import DiscretizedActionWrapper, StateNormalization

from learning.policies.memory import Transition, ReplayMemory, PrioritizedReplayMemory
from learning.policies.base import Policy, BaseModelMixin, TrainConfigBase
from learning.utils.misc import plot_learning_curve, plot_to_image
from learning.utils.construct_nn import construct_nn
from learning.utils.annealing_schedule import LinearSchedule

from learning.policy_pricer.policy_pricer_python import create_map
from policy_pricer_restr.cython_pricer import cython_pricer_optimized

from learning.utils.portfolio_accounts import generate_portfolio

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class DefaultConfig(TrainConfigBase):
    # Training config specifies the hyperparameters of agent and learning
    n_episodes = 10000
    warmup_episodes = n_episodes * 0.8
    checkpoint_every = n_episodes / 100

    learning_rate = 0.001
    end_learning_rate = 0.00001
    # decaying learning rate
    learning_rate_schedule = LinearSchedule(learning_rate, end_learning_rate, warmup_episodes)
    # gamma (discount factor) is set as exp(-rho * dt) in the body of the learning program
    gamma = np.NaN
    epsilon = 1
    epsilon_final = 0.01
    epsilon_schedule = LinearSchedule(epsilon, epsilon_final, warmup_episodes)
    target_update_every_step = 50
    log_every_episode = 100
    compute_portfolio_every = 100

    # Net setting
    layers = (10, 10, 10)
    batch_normalization = False
    # Memory setting
    batch_size = 1024
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
    normalize_states = True
    regularizer = None

    # repayment distribution:


class DQNAgent(BaseModelMixin, Policy):

    def __init__(self, env, name, config=None, training=True, portfolio=None, initialize=False, experiment_name=None):

        if config.normalize_states:
            self.env = StateNormalization(env)
        else:
            self.env = env

        Policy.__init__(self, self.env, name, training=training)
        BaseModelMixin.__init__(self, name, experiment_name=experiment_name)

        self.config = config
        self.config.gamma = np.exp(-env.params.rho * env.dt)

        if config.prioritized_memory_replay:
            self.memory = PrioritizedReplayMemory(alpha=config.replay_alpha, capacity=config.memory_size)
        else:
            self.memory = ReplayMemory(capacity=self.config.memory_size)

        # Optimizer
        self.global_lr = tf.Variable(self.config.learning_rate_schedule.current_p, trainable=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.global_lr)  # , clipnorm=5)

        self.target_net = None
        self.main_net = None
        self._initialize = initialize
        self.global_step = 0
        self.episode = 0

        # Inter-learning plotting parameters
        self._w_points = 60
        self._l_points = 60
        self._l_grid_plot = np.linspace(self.env.observation_space.low[0], self.env.observation_space.high[0],
                                        self._l_points)
        self._w_grid_plot = np.linspace(self.env.observation_space.low[1], self.env.observation_space.high[1],
                                        self._w_points)
        self._n_cpu = cpu_count() - 2
        self._ww, self._ll = np.meshgrid(self._w_grid_plot, self._l_grid_plot)
        self._space_iterator = product(self._l_grid_plot, self._w_grid_plot)
        # We initialize this at bild() to comply with inheritance
        self._space_product = None

        self._tb_log_holder = {}
        self._portfolio = portfolio

    def get_action(self, state, epsilon=0.0):
        q_value = self.main_net.predict_on_batch(state[None, :])
        # q_value = self.main_net(state[None, :], training=False)
        if np.random.rand() <= epsilon:
            # print('Taking random')
            action = np.random.choice(self.act_size)
        else:
            action = np.argmax(q_value)
        return action, q_value

    def update_target(self):
        self.target_net.set_weights(self.main_net.get_weights())

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
            # double_dqn -- this is a feature, not a bug
            target_q = self.main_net(next_states)
            next_action = np.argmax(target_q.numpy(), axis=1)
            target_q = self.target_net(next_states)

            target_value = tf.reduce_sum(tf.one_hot(next_action, self.act_size) * target_q, axis=1)
            target_value = (1 - dones) * self.config.gamma * target_value + rewards

            main_q = self.main_net(states)
            main_value = tf.reduce_sum(tf.one_hot(actions, self.act_size) * main_q, axis=1)
            main_q_to_penalize = tf.identity(main_q)

            td_error = target_value - main_value
            element_wise_loss = tf.square(td_error) * 0.5

            if self.config.prioritized_memory_replay:
                error = tf.reduce_mean(element_wise_loss * weights)
                self.memory.update_priorities(idx, np.abs(td_error.numpy()) + self.config.prior_eps)
            else:
                error = tf.reduce_mean(element_wise_loss)

        # loss = self.main_net.train_on_batch(states, target_value)
        dqn_grads = tape.gradient(error, dqn_variable)

        self.optimizer.apply_gradients(zip(dqn_grads, dqn_variable))
        # Logging
        self.global_step += 1
        self._tb_log_holder = {'Gradients': dqn_grads[0], 'Weights': tf.convert_to_tensor(self.main_net.weights[0]),
                               'Prediction': main_value,
                               'Target': target_value, 'TD error': td_error, 'Elementwise Loss': element_wise_loss,
                               'Loss': tf.reduce_mean(element_wise_loss)}# , 'Penalization': tf.constant([0.0])}
        return tf.reduce_mean(element_wise_loss)

        # def log_tensorboard(self, dqn_grads, main_value, target_value, td_error, element_wise_loss):
        #     with self.writer.as_default():
        #         with tf.name_scope('Network'):
        #             tf.summary.histogram('Weights', self.main_net.weights[0], step=self.global_step)
        #             tf.summary.histogram('Gradients', dqn_grads[0], step=self.global_step)
        #             tf.summary.histogram('Predictions', main_value, step=self.global_step)
        #             tf.summary.histogram('Target', target_value, step=self.global_step)
        #             tf.summary.histogram('TD error', td_error, step=self.global_step)
        #             tf.summary.histogram('Elementwise Loss', element_wise_loss, step=self.global_step)
        #             tf.summary.scalar('Loss', tf.reduce_mean(element_wise_loss), step=self.global_step)

    def log_tensorboard(self, step, **kwargs):
        with self.writer.as_default():
            with tf.name_scope('Network'):
                for kwarg in kwargs:
                    if kwargs[kwarg].ndim > 0:
                        tf.summary.histogram(kwarg, kwargs[kwarg], step=step)
                    else:
                        tf.summary.scalar(kwarg, kwargs[kwarg], step=step)
                    # tf.summary.histogram('Weights', self.main_net.weights[0], step=self.global_step)
                    # tf.summary.histogram('Gradients', dqn_grads[0], step=self.global_step)
                    # tf.summary.histogram('Predictions', main_value, step=self.global_step)
                    # tf.summary.histogram('Target', target_value, step=self.global_step)
                    # tf.summary.histogram('TD error', td_error, step=self.global_step)
                    # tf.summary.histogram('Elementwise Loss', element_wise_loss, step=self.global_step)
                    # tf.summary.scalar('Loss', tf.reduce_mean(element_wise_loss), step=self.global_step)

    def run_training(self):
        self.build()
        n_episodes = self.config.n_episodes
        total_rewards = np.empty(n_episodes)

        for i in range(n_episodes):
            state = self.env.reset()
            if state[0] > self.env.observation_space.high[0]:
                print("Houston, we have a problem.")

            if state[1] > self.env.observation_space.high[1]:
                print("Houston, we have a problem.")

            assert state[0] < self.env.observation_space.high[0]
            assert state[1] < self.env.observation_space.high[1]

            done = False

            score = 0
            while not done:
                action, q_value = self.get_action(state, self.config.epsilon_schedule.current_p)
                next_state, reward, done, info = self.env.step(action)
                self.memory.add(Transition(state, action, reward, next_state, done))
                score += reward

                state = next_state.copy()

                if self.memory.size > self.config.batch_size:
                    self.train(epoch=i)
                    if i % self.config.target_update_every_step == 0:
                        self.update_target()

            total_rewards[i] = score
            avg_rewards = total_rewards[max(0, i - 100):(i + 1)].mean()

            self.config.epsilon_schedule.anneal()
            self.config.beta_schedule.anneal()

            if hasattr(self.config, 'penal_coeff_schedule'):
                self.config.penal_coeff_schedule.anneal()

            self.global_lr.assign(self.config.learning_rate_schedule.anneal())

            if i % self.config.log_every_episode == 0:
                # print("episode:", i, "/", self.config.n_episodes, "episode reward:", score, "avg reward (last 100):",
                #       avg_rewards, "eps:", self.config.epsilon_schedule.current_p, "Learning rate (10e3):",
                #       (self.optimizer._decayed_lr(tf.float32).numpy() * 1000))
                self.logger.info(f"episode:{i}/{self.config.n_episodes} episode reward: {score} avg reward (last 100): "
                                 f"{avg_rewards} eps: {self.config.epsilon_schedule.current_p} Learning rate (10e3): "
                                 f"{self.optimizer._decayed_lr(tf.float32).numpy() * 1000}")

                with self.writer.as_default():
                    with tf.name_scope('Performance'):
                        tf.summary.scalar('episode reward', score, step=i)
                        tf.summary.scalar('running avg reward(100)', avg_rewards, step=i)

                    if self.config.prioritized_memory_replay:
                        with tf.name_scope('Schedules'):
                            tf.summary.scalar('Beta', self.config.beta_schedule.current_p, step=i)
                            tf.summary.scalar('Epsilon', self.config.epsilon_schedule.current_p, step=i)
                            tf.summary.scalar('Learning rate', self.optimizer._decayed_lr(tf.float32).numpy(), step=i)
                            if hasattr(self.config, 'penal_coeff_schedule'):
                                tf.summary.scalar('Penalization coefficient', self.config.penal_coeff_schedule.current_p, step=i)

            if self._portfolio is not None:
                if i % self.config.compute_portfolio_every == 0:
                    with self.writer.as_default():
                        with tf.name_scope('Performance'):
                            price = self.price_portfolio()
                            tf.summary.scalar('Portfolio Performance:', price, step=i)

            # Plot training stats
            self.log_tensorboard(step=i, **self._tb_log_holder)

            # Checkpointing
            if i % self.config.checkpoint_every == 0:
                self.checkpoint(i)

            if i % self.config.plot_every_episode == 0 and self.config.plot_progression_flag:
                self.logger.info(f"episode:{i}/{self.config.n_episodes} Plotting policy")
                self.plot_policy(i)
                self.plot_visit_map(i)

            self.episode += 1

        plot_learning_curve(self.name + '.png', {'rewards': total_rewards})
        self.save()

    def price_portfolio(self):
        ww, ll, p, z = create_map(self, w_points=200, l_points=100, lam_lim=8.0,
                                  larger_offset=True)
        prices = np.zeros(self._portfolio.shape[0])
        for i, acc in enumerate(self._portfolio):
            prices[i] = np.mean(np.asarray(cython_pricer_optimized.value_account(acc, ww, ll, p,
                                                                         cython_pricer_optimized.convert_params_obj(
                                                                             self.env.params),
                                                                         self.env.env.action_bins, n_iterations=1000)))
        return np.mean(prices)

    def checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.model_dir, 'checkpoints', str(epoch))
        os.makedirs(checkpoint_path, exist_ok=True)
        path_model = os.path.join(checkpoint_path, 'main_net.h5')
        env_path = os.path.join(checkpoint_path, 'env.pkl')
        # path_memory_buffer = os.path.join(checkpoint_path, 'buffer.pkl')
        config_path = os.path.join(checkpoint_path, 'train_config.pkl')

        self.config.save(filename=config_path)
        self.main_net.save(path_model)
        # self.memory.save(path_memory_buffer)
        self.env.save(env_path)

    def build(self):
        self.target_net = construct_nn(self.env, self.config, initialize=self._initialize)
        self.main_net = tf.keras.models.clone_model(self.target_net)
        self.main_net.set_weights(self.target_net.get_weights())

        self._space_product = self.env.observation(np.array([[i, j] for i, j in self._space_iterator]))

    def save(self):
        # Saves training procedure
        path_model = os.path.join(self.model_dir, 'main_net.h5')
        path_actions = os.path.join(self.model_dir, 'action_bins.npy')
        env_path = os.path.join(self.model_dir, 'env.pkl')
        path_memory_buffer = os.path.join(self.model_dir, 'buffer.pkl')
        config_path = os.path.join(self.model_dir, 'train_config.pkl')

        self.config.save(config_path)
        self.main_net.save(path_model)
        self.memory.save(path_memory_buffer)
        self.env.save(env_path)
        # np.save(path_actions, self.env.action_bins)

    @classmethod
    def load(cls, model_path, load_buffer=False):
        # loads trained model
        loaded_config = TrainConfigBase.load(os.path.join(model_path, 'train_config.pkl'))
        loaded_env = CollectionsEnv.load(os.path.join(model_path, 'env.pkl'))
        loaded_instance = DQNAgent(loaded_env, model_path, loaded_config, initialize=False, training=False)
        loaded_instance.main_net = tf.keras.models.load_model(os.path.join(model_path, 'main_net.h5'))
        loaded_instance.main_net.compile()
        loaded_instance.target_net = loaded_instance.main_net
        if load_buffer:
            try:
                buffer_path = os.path.join(model_path, 'buffer.pkl')
                with open(buffer_path, 'rb') as f:
                    buffer = pickle.load(f)
                    loaded_instance.memory.buffer = buffer
            except (FileNotFoundError, IOError):
                print('No buffer found.')
        return loaded_instance

    def plot_policy(self, step_i):
        # plots policy in w and lambda space
        predictions = self.main_net.predict_on_batch(self._space_product)
        z = np.amax(predictions, axis=1).reshape(self._l_points, self._w_points)
        p = np.argmax(predictions, axis=1).reshape(self._l_points, self._w_points)
        fig, ax = plt.subplots(nrows=1, ncols=2)
        cmap = m.cm.get_cmap('YlOrBr', self.act_size)
        boundaries = np.linspace(0, self.act_size, self.act_size + 1)
        norm = m.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
        im = ax[0].pcolormesh(self._ww, self._ll, p, cmap=cmap, norm=norm, shading='auto')
        fig.colorbar(im)

        CS = ax[1].contour(self._ww, self._ll, z)
        ax[1].clabel(CS, inline=1, fontsize=10)
        ax[1].set_title('Value function')

        with self.writer.as_default():
            with tf.name_scope('Learning progress'):
                tf.summary.image("Learned policy", plot_to_image(fig), step=step_i)
        return fig

    def plot_visit_map(self, step_i):
        visits = self.env.convert_back(np.array([trans.s for trans in self.memory.buffer]))
        visits_w = visits[:, 1]
        visits_l = visits[:, 0]
        # np.array([vis[1] for vis in visits])
        # visits_l = np.array([vis[0] for vis in visits])

        # Generate some test data
        # mask = visits_w >= 0

        x = visits_w  # [mask]
        y = visits_l  # [mask]

        # This part of code shouldn't be needed given the convert_back method
        # if self.config.normalize_states:
        #     rangespace= [[0, 1], [0, 1]]
        # else:
        #     rangespace = [[0, self.env.w0], [0, self.env.MAX_LAMBDA]]

        rangespace = [[0, self.env.w0], [0, self.env.MAX_LAMBDA]]

        heatmap, xedges, yedges = np.histogram2d(x, y, bins=(30, 20), range=rangespace)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        fig, ax = plt.subplots()
        mappable = ax.imshow(np.where(heatmap.T != 0, np.log(heatmap.T), 0), extent=extent, origin='lower',
                             interpolation='nearest', aspect='auto')
        ax.set_title('Visit heatmap')
        fig.colorbar(mappable)

        with self.writer.as_default():
            with tf.name_scope('Learning progress'):
                tf.summary.image("Visit heatmap", plot_to_image(fig), step=step_i)
        return fig


if __name__ == '__main__':
    from dcc import Parameters

    params = Parameters()
    params.rho = 0.15

    actions_bins = np.array([0, 0.2, 0.5, 1.0])
    n_actions = len(actions_bins)

    # rep_dist = BetaRepayment(params, 0.9, 0.5, 10, MAX_ACCOUNT_BALANCE)
    rep_dist = UniformRepayment(params)

    c_env = CollectionsEnv(params=params, repayment_dist=rep_dist, reward_shaping='continuous', randomize_start=True,
                           max_lambda=None,
                           starting_state=np.array([3, 200], dtype=np.float32)
                           )
    environment = DiscretizedActionWrapper(c_env, actions_bins)

    portfolio_acc = generate_portfolio(50)

    experiment_name = None

    dqn = DQNAgent(environment, 'test_constr_log_every_pricer', training=True, config=DefaultConfig(), initialize=False,
                   portfolio=portfolio_acc, experiment_name=experiment_name)
    dqn.run_training()
