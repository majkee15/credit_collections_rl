import tensorflow as tf
import numpy as np

from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
from collections import deque
import datetime
import os

import random

import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')

from learning.collections_env.collections_env import CollectionsEnv
from learning.utils.wrappers import DiscretizedActionWrapper, StateNormalization
from learning.policies.memory import ReplayMemory
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



class Config:
    def __init__(self, warmup_episodes):
        self.learning_rate = 0.00001
        self.alpha = 0.8
        self.gamma = 1

        # exploration / exploitation
        self.warmup_episodes = warmup_episodes
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.eps_drop = (self.epsilon_start - self.epsilon_final) / self.warmup_episodes

class DQNAgent:
    def __init__(self, env, config=None, layers=(32, 32, 32), batch_size=100, memory_size=2000, batch_normalizaton=False):

        self.env = env
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.shape[0]
        self.config = config
        self.batch_size = batch_size
        self.memory = ReplayMemory()
        self.layers = layers
        #for soft update
        self.tau = 0.08


        #Optimizer
        self.optimizer = optimizers.Adam(lr=self.config.learning_rate)#, clipnorm=5)
        # Target net
        self.target_net = tf.keras.Sequential()
        self.target_net.add(tf.keras.layers.Input(shape=env.observation_space.shape))
        for i, layer_size in enumerate(self.layers):
            self.target_net.add(tf.keras.layers.Dense(layer_size, activation='relu'))
            if batch_normalizaton:
                self.target_net.add(tf.keras.layers.BatchNormalization())
        self.target_net.add(tf.keras.layers.Dense(env.action_space.n, activation='linear'))
        self.target_net.build()

        self.target_net.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

        # Main net
        self.main_net = tf.keras.Sequential()
        self.main_net.add(tf.keras.layers.Input(shape=env.observation_space.shape))
        for i, layer_size in enumerate(self.layers):
            self.main_net.add(tf.keras.layers.Dense(layer_size, activation='relu'))
            if batch_normalizaton:
                self.main_net.add(tf.keras.layers.BatchNormalization())
        self.main_net.add(tf.keras.layers.Dense(env.action_space.n, activation='linear'))
        self.main_net.build()

        self.main_net.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    # def append_sample(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, epsilon):
        q_value = self.main_net.predict_on_batch(state[None, :])
        if np.random.rand() <= epsilon:
            # print('Taking random')
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(q_value)
        return action, q_value

    def update_target(self):
        # hard target update every C steps
        self.target_net.set_weights(self.main_net.get_weights())

    def train(self):
        batch = self.memory.sample(self.batch_size)
        states = np.array([val[0] for val in batch])
        actions = np.array([val[1] for val in batch])
        rewards = np.array([val[2] for val in batch])
        next_states = np.array([val[3] for val in batch])
        dones = np.array([val[4] for val in batch])

        dqn_variable = self.main_net.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)
            target_q = self.target_net.predict_on_batch(next_states)
            next_action = np.argmax(target_q.numpy(), axis=1)
            # the next saction should be selected using the online net

            target_value = tf.reduce_sum(tf.one_hot(next_action, self.action_size) * target_q, axis=1)
            target_value = (1 - dones) * self.config.gamma * target_value + rewards

            main_q = self.main_net.predict_on_batch(states)
            main_value = tf.reduce_sum(tf.one_hot(actions, self.action_size) * main_q, axis=1)

            error = tf.square(main_value - target_value) * 0.5
            error = tf.reduce_mean(error)

        # loss = self.main_net.train_on_batch(states, target_value)
        dqn_grads = tape.gradient(error, dqn_variable)
        loss = self.optimizer.apply_gradients(zip(dqn_grads, dqn_variable))
        return loss

    def run(self, n_episodes=5000):
        epsilon = self.config.epsilon_start

        total_rewards = np.empty(n_episodes, dtype=np.float64)
        loss_per_episode = np.empty(n_episodes, dtype=np.float64)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        log_dir = 'logs/dqn/' + current_time
        summary_writer = tf.summary.create_file_writer(log_dir)

        for i in range(n_episodes):
            state = self.env.reset()
            done = False
            score = 0
            loss = 0
            while not done:
                action, q_value = self.get_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)
                self.memory.add((state, action, reward, next_state, done))
                score += reward

                state = next_state

                if self.memory.size > self.batch_size:
                    loss += self.train()
                    if i % 50 == 0 and done:
                        self.update_target()

            if epsilon > self.config.epsilon_final:
                epsilon = max(self.config.epsilon_final, epsilon - self.config.eps_drop)

            total_rewards[i] = score
            loss_per_episode[i] = np.mean(loss)
            avg_rewards = total_rewards[max(0, i - 100):(i + 1)].mean()

            with summary_writer.as_default():
                tf.summary.scalar('episode reward', score, step=i)
                tf.summary.scalar('running avg reward(100)', avg_rewards, step=i)
                tf.summary.scalar('Loss', loss_per_episode[i], step=i)
            if i % 10 == 0:
                print("episode:", i, "episode reward:", score, "eps:", epsilon, "avg reward (last 100):",
                      avg_rewards, "episode loss: ", loss_per_episode[i])

        #save_dir = 'saved_models/dqn/'
        file_name = 'dqn' + current_time + '.h5'
        self.save(file_name)

    def save(self, name):
        self.main_net.save(name)

    def load(self, file):
        self.main_net = tf.keras.models.load_model(file)
        self.target_net = self.main_net



if __name__ == '__main__':
    # actions_bins = np.array([0, 0.5, 1, 5])
    actions_bins = np.array([0, 0.5])
    layers_shape = (128, 128, 128)
    n_actions = len(actions_bins)
    c_env = CollectionsEnv()
    environment = DiscretizedActionWrapper(c_env, actions_bins)
    environment = StateNormalization(environment)

    config = Config(150)

    agent = DQNAgent(environment, config, layers=layers_shape, batch_size=200, batch_normalizaton=True)
    agent.run(n_episodes=300)
