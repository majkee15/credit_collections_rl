# dqn policy
import os
import numpy as np
import tensorflow as tf
import datetime
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as m

from learning.collections_env import CollectionsEnv
from learning.utils.wrappers import DiscretizedActionWrapper, StateNormalization

from learning.policies.memory import Transition, ReplayMemory, PrioritizedReplayMemory
from learning.policies.base import Policy, BaseModelMixin, TrainConfig
from learning.utils.misc import plot_learning_curve, plot_to_image
from learning.utils.construct_nn import construct_nn
from learning.utils.annealing_schedule import AnnealingSchedule


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class DefaultConfig(TrainConfig):
    n_episodes = 20000
    warmup_episodes = 12000

    # fixed learning rate
    learning_rate = 0.001
    end_learning_rate = 0.00001
    # decaying learning rate
    # learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=learning_rate,
    #                                                               decay_steps=warmup_episodes,
    #                                                               end_learning_rate=end_learning_rate, power=1.0)
    learning_rate_schedule = AnnealingSchedule(learning_rate, end_learning_rate, n_episodes)
    gamma = 0.99
    epsilon = 1
    epsilon_final = 0.01
    epsilon_schedule = AnnealingSchedule(epsilon, epsilon_final, warmup_episodes)
    target_update_every_step = 50
    log_every_episode = 10

    # Memory setting
    batch_normalization = True
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
    plot_every_episode = target_update_every_step

    # env setting
    normalize_states = True

class DQNAgent(Policy, BaseModelMixin):

    def __init__(self, env, name, config=None, training=True, layers=(128, 128, 128), initialize=False):

        if config.normalize_states:
            self.env = StateNormalization(env)

        Policy.__init__(self, self.env, name, training=training)
        BaseModelMixin.__init__(self, name)

        self.config = config
        if config.prioritized_memory_replay:
            self.memory = PrioritizedReplayMemory(alpha=config.replay_alpha, capacity=config.memory_size)
        else:
            self.memory = ReplayMemory(capacity=self.config.memory_size)
        self.layers = layers

        # Optimizer
        self.global_lr = tf.Variable(self.config.learning_rate_schedule.current_p, trainable=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.global_lr)#, clipnorm=5)

        self.target_net = construct_nn(self.env, self.layers, self.config, initialize=initialize)
        self.main_net = tf.keras.models.clone_model(self.target_net)
        self.main_net.set_weights(self.target_net.get_weights())
        # number of training epochs = global - step
        self.global_step = 0
    # def append_sample(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, epsilon):
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

    def train(self):
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
            target_q = self.main_net.predict_on_batch(next_states)
            next_action = np.argmax(target_q.numpy(), axis=1)
            target_q = self.target_net.predict_on_batch(next_states)

            target_value = tf.reduce_sum(tf.one_hot(next_action, self.act_size) * target_q, axis=1)
            target_value = (1 - dones) * self.config.gamma * target_value + rewards

            main_q = self.main_net.predict_on_batch(states)
            main_value = tf.reduce_sum(tf.one_hot(actions, self.act_size) * main_q, axis=1)

            td_error = target_value - main_value

            element_wise_loss = tf.square(td_error) * 0.5

            if self.config.prioritized_memory_replay:
                error = tf.reduce_mean(element_wise_loss * weights)
            else:
                error = tf.reduce_mean(element_wise_loss)


        if self.config.prioritized_memory_replay:
            self.memory.update_priorities(idx, np.abs(td_error.numpy()) + self.config.prior_eps)
        # loss = self.main_net.train_on_batch(states, target_value)
        dqn_grads = tape.gradient(error, dqn_variable)
        self.optimizer.apply_gradients(zip(dqn_grads, dqn_variable))
        # Logging
        self.global_step += 1
        with self.writer.as_default():
            with tf.name_scope('Network'):
                tf.summary.histogram('Weights', self.main_net.weights[0], step=self.global_step)
                tf.summary.histogram('Gradients', dqn_grads[0], step=self.global_step)
                tf.summary.histogram('Predictions', main_value, step=self.global_step)
                tf.summary.histogram('Target', target_value, step=self.global_step)
                tf.summary.histogram('TD error', td_error, step=self.global_step)
                tf.summary.histogram('Elementwise Loss', element_wise_loss, step=self.global_step)
                tf.summary.scalar('Loss', tf.reduce_mean(element_wise_loss), step=self.global_step)
        return tf.reduce_mean(element_wise_loss)

    def run(self):

        n_episodes = self.config.n_episodes
        loss = None

        total_rewards = np.empty(n_episodes)

        for i in range(n_episodes):
            state = self.env.reset()
            done = False

            score = 0
            while not done:
                action, q_value = self.get_action(state, self.config.epsilon_schedule.current_p)
                next_state, reward, done, info = self.env.step(action)
                self.memory.add(Transition(state, action, reward, next_state, done))
                score += reward

                state = next_state.copy()

                if self.memory.size > self.config.batch_size:
                    loss = self.train()
                    if i % self.config.target_update_every_step == 0:
                        self.update_target()

            total_rewards[i] = score
            avg_rewards = total_rewards[max(0, i - 100):(i + 1)].mean()

            self.config.epsilon_schedule.anneal()
            self.config.beta_schedule.anneal()
            self.global_lr.assign(self.config.learning_rate_schedule.anneal())

            with self.writer.as_default():
                with tf.name_scope('Performance'):
                    tf.summary.scalar('episode reward', score, step=i)
                    tf.summary.scalar('running avg reward(100)', avg_rewards, step=i)

                if self.config.prioritized_memory_replay:
                    with tf.name_scope('Schedules'):
                        tf.summary.scalar('Beta', self.config.beta_schedule.current_p, step=i)
                        tf.summary.scalar('Epsilon', self.config.epsilon_schedule.current_p, step=i)
                        tf.summary.scalar('Learning rate', self.optimizer._decayed_lr(tf.float32).numpy(), step=i)

            if i % self.config.log_every_episode == 0:
                print("episode:", i, "/", self.config.n_episodes, "episode reward:", score, "avg reward (last 100):",
                      avg_rewards, "eps:", self.config.epsilon_schedule.current_p, "Learning rate (10e3):",
                      (self.optimizer._decayed_lr(tf.float32).numpy() * 1000))

            if i % self.config.plot_every_episode == 0 and self.config.plot_progression_flag:
                now = datetime.now().time()
                print(f'{now}, Plotting policy')
                self.plot_policy(i)
                self.plot_visit_map(i)

        plot_learning_curve(self.name + '.png', {'rewards': total_rewards})
        self.save()

    # Summary writing routines

    def write_summaries(self):
        with tf.name_scope('Layer 1'):
            with tf.name_scope('W'):
                mean = tf.reduce_mean(W)
                tf.summary.scalar('mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(W - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.histogram('histogram', var)

    def save(self):
       #current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
    def load(cls, model_path):
        loaded_config = TrainConfig.load(os.path.join(model_path, 'train_config.pkl'))
        loaded_env = CollectionsEnv.load(os.path.join(model_path, 'env.pkl'))
        loaded_instance = DQNAgent(loaded_env, model_path, loaded_config, initialize=False, training=False)
        loaded_instance.main_net = tf.keras.models.load_model(os.path.join(model_path, 'main_net.h5'))
        loaded_instance.target_net = tf.keras.models.load_model(os.path.join(model_path, 'main_net.h5'))
        try:
            buffer_path = os.path.join(model_path, 'buffer.pkl')
            with open(buffer_path, 'rb') as f:
                buffer = pickle.load(f)
                loaded_instance.memory.buffer = buffer
        except (FileNotFoundError, IOError):
            print('No buffer found.')

        return loaded_instance

    def plot_policy(self, step_i):
        w_points = 60
        l_points = 60
        l = np.linspace(self.env.observation_space.low[0], self.env.observation_space.high[0], l_points)
        w = np.linspace(self.env.observation_space.low[1], self.env.observation_space.high[1], w_points)
        ww, ll = np.meshgrid(w, l)
        z = np.zeros_like(ww)
        p = np.zeros_like(ww)
        for i, xp in enumerate(w):
            for j, yp in enumerate(l):
                fixed_obs = self.env.observation(np.array([yp, xp]))
                z[j, i] = np.amax(self.main_net.predict_on_batch(fixed_obs[None, :]))
                p[j, i] = environment.action(np.argmax(self.main_net.predict_on_batch(fixed_obs[None, :])))

        fig, ax = plt.subplots(nrows=1, ncols=2)
        im = ax[0].pcolor(ww, ll, p)
        cdict = {
            'red': ((0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
            'green': ((0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
            'blue': ((0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
        }

        cm = m.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
        im = ax[0].pcolor(ww, ll, p, cmap=cm)
        fig.colorbar(im)

        CS = ax[1].contour(ww, ll, z)
        ax[1].clabel(CS, inline=1, fontsize=10)
        ax[1].set_title('Value function')

        with self.writer.as_default():
            with tf.name_scope('Learning progress'):
                tf.summary.image("Learned policy", plot_to_image(fig), step=step_i)
        return fig

    def plot_visit_map(self, step_i):
        visits = [trans.s for trans in self.memory.buffer]
        visits_w = np.array([vis[1] for vis in visits])
        visits_l = np.array([vis[0] for vis in visits])

        # Generate some test data
        mask = visits_w >= 0

        x = visits_w[mask]
        y = visits_l[mask]

        # fig,ax = plt.subplots(figsize=(10,10))

        heatmap, xedges, yedges = np.histogram2d(x, y, bins=(30, 20))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        fig, ax = plt.subplots()
        mappable = ax.imshow(heatmap.T, extent=extent, origin='lower', interpolation='nearest', aspect='auto')
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

    actions_bins = np.array([0, 1.0])
    layers_shape = (64, 64, 64)
    n_actions = len(actions_bins)
    c_env = CollectionsEnv(params=params, reward_shaping='continuous', randomize_start=True, max_lambda=None,
                           starting_state=np.array([3, 200], dtype=np.float32)
                           )
    environment = DiscretizedActionWrapper(c_env, actions_bins)

    dqn = DQNAgent(environment, 'DDQN_20K', training=True, config=DefaultConfig(), initialize=False)
    dqn.run()
