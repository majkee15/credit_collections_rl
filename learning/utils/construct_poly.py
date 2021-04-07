import tensorflow as tf
from tensorflow import keras
import numpy as np


def construct_poly_approx(env, poly_order):
    """
    Constructs a linear approximator with polynomial features up to degree 3.
    Args:
        env: instance of CollectionsEnv
        poly_order: int_+ <= 3

    Returns:
            tf2 model instance
    """

    if not isinstance(poly_order, int):
        raise ValueError("poly_order has to be an integer.")
    elif poly_order > 3:
        raise ValueError("Only polynomial up to degree 3 are supported.")

    if poly_order == 1:
        features = 3
    if poly_order == 2:
        features == 6
    if poly_order == 3:
        features = 10

    n_act = env.action_space.n

    inputs = tf.keras.Input(shape=(features,))
    x = tf.keras.layers.Dense(n_act, activation='linear', use_bias=False)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x, name="combined")
    model.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam')
    return model


def construct_spline_approx(env, total_features):
    inputs = tf.keras.Input(shape=(total_features + 2,), dtype='float32')
    first_layer = tf.keras.layers.Dense(env.action_space.n, activation='linear', use_bias=False, dtype='float32') \
        (inputs[:, 2:])
    model = tf.keras.Model(inputs=inputs, outputs=first_layer, name="combined")
    model.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam')
    return model

