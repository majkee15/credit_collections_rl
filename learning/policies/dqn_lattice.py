# dqn policy
import os
import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl
import datetime

from learning.collections_env import CollectionsEnv
from learning.utils.wrappers import DiscretizedActionWrapper, StateNormalization

from learning.policies.memory import Transition, ReplayMemory, PrioritizedReplayMemory
from learning.policies.base import Policy, BaseModelMixin, TrainConfig
from learning.utils.misc import plot_learning_curve
from learning.utils.annealing_schedule import AnnealingSchedule


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class DefaultConfig(TrainConfig):
    n_episodes = 2000
    warmup_episodes = 1400

    # fixed learning rate
    learning_rate = 0.001
    end_learning_rate = 0.0001
    # decaying learning rate
    # learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=learning_rate,
    #                                                               decay_steps=warmup_episodes,
    #                                                               end_learning_rate=end_learning_rate, power=1.0)
    learning_rate_schedule = AnnealingSchedule(learning_rate, end_learning_rate, n_episodes)
    gamma = 1.0
    epsilon = 1.0
    epsilon_final = 0.05
    epsilon_schedule = AnnealingSchedule(epsilon, epsilon_final, warmup_episodes)
    target_update_every_step = 50
    log_every_episode = 10

    # Memory setting
    batch_normalization = True
    batch_size = 512
    memory_size = 100000
    # PER setting

    prioritized_memory_replay = True
    replay_alpha = 0.2
    replay_beta = 0.4
    replay_beta_final = 1.0
    beta_schedule = AnnealingSchedule(replay_beta, replay_beta_final, warmup_episodes, inverse=True)
    prior_eps = 1e-6


class DQNAgentLattice(Policy, BaseModelMixin):

    def __init__(self, env, name, config=None, training=True, lattice_sizes=[10, 10]):

        Policy.__init__(self, env, name, training=training)
        BaseModelMixin.__init__(self, name)

        self.config = config
        self.batch_size = self.config.batch_size
        if config.prioritized_memory_replay:
            self.memory = PrioritizedReplayMemory(alpha=config.replay_alpha, capacity=config.memory_size)
        else:
            self.memory = ReplayMemory(capacity=self.config.memory_size)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)#, clipnorm=5)

        lattice_sizes = [20, 20]
        lattice = tfl.layers.Lattice(
            # Number of vertices along each dimension.
            units=2,
            lattice_sizes=lattice_sizes,
            # You can specify monotonicity constraints.
            monotonicities=['increasing', 'increasing'],
            output_min=0.0,
            output_max=self.env.w0)
        calibrator_w = tfl.layers.PWLCalibration(
            input_keypoints=np.linspace(0, self.env.w0, num=120),
            units=1,

            dtype=tf.float32,
            output_min=0.0,
            output_max=lattice_sizes[0] - 1.0,
            monotonicity='increasing')

        calibrator_l = tfl.layers.PWLCalibration(
            # Every PWLCalibration layer must have keypoints of piecewise linear
            # function specified. Easiest way to specify them is to uniformly cover
            # entire input range by using numpy.linspace().
            input_keypoints=np.linspace(0, self.env.MAX_LAMBDA, num=200),
            units=1,
            # You need to ensure that input keypoints have same dtype as layer input.
            # You can do it by setting dtype here or by providing keypoints in such
            # format which will be converted to deisred tf.dtype by default.
            dtype=tf.float32,
            # Output range must correspond to expected lattice input range.
            output_min=0.0,
            output_max=lattice_sizes[1] - 1.0,
            monotonicity='increasing')

        combined_calibrators = combined_calibrators = tfl.layers.ParallelCombination([calibrator_w, calibrator_l],
                                                                                     single_output=True)

        model = tf.keras.models.Sequential()
        model.add(combined_calibrators)
        model.add(tf.keras.layers.RepeatVector(2))
        model.add(lattice)
        model.compile(loss=tf.keras.losses.mean_squared_error,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

        self.main_net = model

        self.target_net = tf.keras.models.clone_model(model)
        self.global_step = 0

    def get_action(self, state, epsilon):
        q_value = self.main_net.predict_on_batch(state[None, :])
        if np.random.rand() <= epsilon:
            # print('Taking random')
            action = np.random.choice(self.act_size)
        else:
            action = np.argmax(q_value)
        return action, q_value

    def update_target(self):
        self.target_net.set_weights(self.main_net.get_weights())

    def train(self):
        batch = self.memory.sample(self.batch_size)
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

                state = next_state

                if self.memory.size > self.batch_size:
                    loss = self.train()
                    if i % self.config.target_update_every_step == 0:
                        self.update_target()

            total_rewards[i] = score
            avg_rewards = total_rewards[max(0, i - 100):(i + 1)].mean()

            self.config.epsilon_schedule.anneal()
            self.config.beta_schedule.anneal()
            self.global_lr = self.config.learning_rate_schedule.anneal()

            with self.writer.as_default():
                with tf.name_scope('Performance'):
                    tf.summary.scalar('episode reward', score, step=i)
                    tf.summary.scalar('running avg reward(100)', avg_rewards, step=i)

                if self.config.prioritized_memory_replay:
                    with tf.name_scope('Schedules'):
                        tf.summary.scalar('Beta', self.config.beta_schedule.current_p, step=i)
                        tf.summary.scalar('Epsilon', self.config.epsilon_schedule.current_p, step=i)
                        # tf.summary.scalar('Learning rate', self.optimizer._decayed_lr(tf.float32).numpy(), step=i)

            if i % self.config.log_every_episode == 0:
                print("episode:", i, "/", self.config.n_episodes, "episode reward:", score,"avg reward (last 100):",
                      avg_rewards, "eps:", self.config.epsilon_schedule.current_p, "Learning rate (10e3):",)
                      #(self.optimizer._decayed_lr(tf.float32).numpy() * 1000))

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
        path_memory_buffer = os.path.join(self.model_dir, 'buffer.pkl')

        self.main_net.save(path_model)
        self.memory.save(path_memory_buffer)
        np.save(path_actions, self.env.action_bins)

    def load(self, model_path):

        # self.main_net = tf.keras.models.load_model(os.path.join(model_path, 'main_net.h5'))
        # self.target_net = tf.keras.models.load_model(os.path.join(model_path, 'main_net.h5'))
        self.main_net.load_weights(os.path.join(model_path, 'main_net.h5'))
        self.target_net.load_weights(os.path.join(model_path, 'main_net.h5'))
        self.action_bins = np.load(os.path.join(model_path, 'action_bins.npy'))

if __name__ == '__main__':
    actions_bins = np.array([0, 1.0])
    n_actions = len(actions_bins)
    c_env = CollectionsEnv(continuous_reward=True, randomize_start=False)
    environment = DiscretizedActionWrapper(c_env, actions_bins)
   #  environment = StateNormalization(environment)

    dqn = DQNAgentLattice(environment, 'DQNLattice', training=True, config=DefaultConfig())
    dqn.run()
