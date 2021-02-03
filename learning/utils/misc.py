import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import io
import tensorflow as tf
import pickle

from gym.wrappers.monitor import load_results
from copy import deepcopy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESOURCE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources'))


class Config:
    def __init__(self, **kwargs):
        # read parameters from parents, and children can override the values.
        parents = []
        queue = [self.__class__]
        while queue:
            parent = queue.pop()
            if issubclass(parent, Config) and parent is not Config:
                parents.append(parent)
                for p in reversed(parent.__bases__):
                    queue.append(p)

        params = {}
        for cfg in reversed(parents):
            params.update(cfg.__dict__)

        # Set all instance variable based on kwargs and default class variables
        for key, value in params.items():
            if key.startswith('__'):
                continue

            if key in kwargs:
                # override default with provided parameter
                value = kwargs[key]
            else:
                # Need to make copies of class variables so that they aren't changed by instances
                value = deepcopy(value)

            self.__dict__[key] = value

    def __setattr__(self, name, value):
        if name not in self.__dict__:
            raise AttributeError(f"{self.__class__.__name__} does not have attribute {name}")
        self.__dict__[name] = value

    def __getattr__(self, name):
        # Raise error on assignment of missing variable
        if name not in self.__dict__:
            raise AttributeError(f"{self.__class__.__name__} does not have attribute {name}")
        return self.__dict__[name]

    def as_dict(self):
        return deepcopy(self.__dict__)

    def copy(self):
        return self.__class__(**self.as_dict())

    def get(self, name, default):
        return self.as_dict().get(name, default)

    def __repr__(self):
        return super().__repr__() + "\n" # + self.dumps()

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


def plot_learning_curve(filename, value_dict, xlabel='step'):
    # Plot step vs the mean(last 50 episodes' rewards)
    fig = plt.figure(figsize=(12, 4 * len(value_dict)))

    for i, (key, values) in enumerate(value_dict.items()):
        ax = fig.add_subplot(len(value_dict), 1, i + 1)
        ax.plot(range(len(values)), values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(key)
        ax.grid('k--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.join(RESOURCE_ROOT, 'figs'), exist_ok=True)
    plt.savefig(os.path.join(RESOURCE_ROOT, 'figs', filename))


def plot_from_monitor_results(monitor_dir, window=10):
    assert os.path.exists(monitor_dir)
    if monitor_dir.endswith('/'):
        monitor_dir = monitor_dir[:-1]

    data = load_results(monitor_dir)
    n_episodes = len(data['episode_lengths'])
    assert n_episodes > 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), tight_layout=True, sharex=True)

    ax1.plot(range(n_episodes), pd.rolling_mean(np.array(data['episode_lengths']), window))
    ax1.set_xlabel('episode')
    ax1.set_ylabel('episode length')
    ax1.grid('k--', alpha=0.6)

    ax2.plot(range(n_episodes), pd.rolling_mean(np.array(data['episode_rewards']), window))
    ax2.set_xlabel('episode')
    ax2.set_ylabel('episode reward')
    ax2.grid('k--', alpha=0.6)

    os.makedirs(os.path.join(REPO_ROOT, 'figs'), exist_ok=True)
    plt.savefig(os.path.join(REPO_ROOT, 'figs', os.path.basename(monitor_dir) + '-monitor'))


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

# if __name__ == '__main__':
#     class TrainConfig(Config):
#         lr = 0.001
#         n_steps = 10000
#         warmup_steps = 5000
#         batch_size = 64
#         log_every_step = 1000
#
#         # give an extra bonus if done; only needed for certain tasks.
#         done_reward = None
#     cf = TrainConfig()
#     # print(cf)
#     print(cf.__dict__)
#     print(cf.log_every_step)
#     print(cf)