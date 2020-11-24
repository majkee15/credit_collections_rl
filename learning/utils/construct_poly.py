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
    inputs = tf.keras.Input(shape=(total_features + 2,))
    first_layer = tf.keras.layers.Dense(env.action_space.n, activation='linear', use_bias=False)(inputs[:, 2:])
    model = tf.keras.Model(inputs=inputs, outputs=first_layer, name="combined")
    model.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam')
    return model


def calculate_penalization_poly(l, w, theta,  degree: int = 3):
    if degree != 3:
        raise NotImplementedError("Implemented only for polnomials of dim=3.")
    # these are hardcoded derivatives corresponding to parameters for degree 3
    # i.e. matrix multiplication with parameters yields the constraint
    first_d_l = np.array([0, 1, 0, 2*l, w, 0, 3*l*l, 2*l*w, w*w, 0])
    second_d_l = np.array([0, 0, 0, 2, 0, 0, 6*l, 2*w, 0, 0])
    first_d_w = np.array([0, 0, 1, 0, l, 2*w, 0, l*l, 2*w*l, 3*w*w])
    second_d_w = np.array([0, 0, 0, 0, 0, 2, 0, 0, 2*l, 6*w])

    # theta is expected as 2 x 10 matrix
    return None


def calculate_penalization_bspline(l, w, theta,  degree: int = 3):
    raise NotImplementedError('Not implemented yet.')