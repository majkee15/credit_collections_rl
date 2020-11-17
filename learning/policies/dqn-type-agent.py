import tensorflow as tf
import numpy as np
import os
import pickle

from learning.policies.base import Policy, BaseModelMixin, TrainConfig
from learning.policies.memory import Transition, ReplayMemory, PrioritizedReplayMemory
from learning.utils.wrappers import StateNormalization
from learning.collections_env import CollectionsEnv


class DQNAgent(Policy, BaseModelMixin, ABC):
    """
    Class that specifies DQN type learning
    """
    def __init__(self, env, name, config=None, training=True, initialize=False):

        if config.normalize_states:
            self.env = StateNormalization(env)

        Policy.__init__(self, self.env, name, training=training)
        BaseModelMixin.__init__(self, name)

        self.config = config
        self.config.gamma = np.exp(-env.params.rho * env.dt)
        if config.prioritized_memory_replay:
            self.memory = PrioritizedReplayMemory(alpha=config.replay_alpha, capacity=config.memory_size)
        else:
            self.memory = ReplayMemory(capacity=self.config.memory_size)

        # Optimizer
        self.global_lr = tf.Variable(self.config.learning_rate_schedule.current_p, trainable=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.global_lr)  # , clipnorm=5)

        self.main_net = None
        self.target_net = None

        # number of training epochs = global - step
        self.global_step = 0

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

    @classmethod
    def load(cls, model_path):
        # loads trained model
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

    def build(self):
        pass

