import tensorflow_lattice as tfl
import tensorflow as tf
import numpy as np


def construct_lattice(env):
    '''
    Constructs deep lattice network with 832 parameters
    Returns:

    '''

    min_w, min_l = (env.MIN_ACCOUNT_BALANCE, env.params.lambdainf)
    max_w, max_l = (env.w0, env.MAX_LAMBDA)

    lattice_units_layer = [3, 3, 3, 2]

    n_lattice_points = [3, 3, 3, 3]

    lattice1 = tfl.layers.Lattice(units=lattice_units_layer[0], lattice_sizes=[n_lattice_points[0]] * 2,
                                  monotonicities=2 * ['increasing'], output_min=0, output_max=1)

    lattice2 = tfl.layers.Lattice(
        units=lattice_units_layer[1],
        lattice_sizes=[n_lattice_points[1]] * lattice_units_layer[0],
        monotonicities=['increasing'] * lattice_units_layer[0], output_min=0, output_max=1)

    lattice3 = tfl.layers.Lattice(
        units=lattice_units_layer[2],
        lattice_sizes=[n_lattice_points[2]] * lattice_units_layer[1],
        # You can specify monotonicity constraints.
        monotonicities=['increasing'] * lattice_units_layer[1], output_min=0, output_max=1)

    lattice4 = tfl.layers.Lattice(
        units=lattice_units_layer[3],
        lattice_sizes=[n_lattice_points[3]] * lattice_units_layer[2],
        # You can specify monotonicity constraints.
        monotonicities=['increasing'] * lattice_units_layer[2], output_min=0, output_max=1)

    combined_calibrators = []

    for i in range(lattice_units_layer[0]):
        calibration_layer_l = tfl.layers.PWLCalibration(input_keypoints=np.linspace(0, max_l, num=200),
                                                        dtype=tf.float32, output_min=0.0,
                                                        output_max=n_lattice_points[0] - 1.0,
                                                        monotonicity='increasing', convexity='concave')
        calibration_layer_w = tfl.layers.PWLCalibration(input_keypoints=np.linspace(0, max_w, num=200),
                                                        dtype=tf.float32, output_min=0.0,
                                                        output_max=n_lattice_points[0] - 1.0,
                                                        monotonicity='increasing', convexity='convex')
        combined_calibrators.append(calibration_layer_l)
        combined_calibrators.append(calibration_layer_w)

    input_callibrators = tfl.layers.ParallelCombination(combined_calibrators, single_output=True)

    calibrator2 = tfl.layers.PWLCalibration(
        input_keypoints=np.linspace(0, 1, num=50),
        units=lattice_units_layer[0] * lattice_units_layer[1],
        dtype=tf.float32,
        output_min=0.0,
        output_max=n_lattice_points[1] - 1.0,
        monotonicity='increasing')

    calibrator3 = tfl.layers.PWLCalibration(
        input_keypoints=np.linspace(0, 1, num=50),
        units=lattice_units_layer[1] * lattice_units_layer[2],
        dtype=tf.float32,
        output_min=0.0,
        output_max=n_lattice_points[2] - 1.0,
        monotonicity='increasing')

    calibrator4 = tfl.layers.PWLCalibration(
        input_keypoints=np.linspace(0, 1, num=50),
        units=lattice_units_layer[2] * lattice_units_layer[3],
        dtype=tf.float32,
        output_min=0.0,
        output_max=n_lattice_points[3] - 1.0,
        monotonicity='increasing')

    calibratorf = tfl.layers.PWLCalibration(
        # Every PWLCalibration layer must have keypoints of piecewise linear
        # function specified. Easiest way to specify them is to uniformly cover
        # entire input range by using numpy.linspace().
        input_keypoints=np.linspace(0, 1, num=100),
        units=2,
        # You need to ensure that input keypoints have same dtype as layer input.
        # You can do it by setting dtype here or by providing keypoints in such
        # format which will be converted to deisred tf.dtype by default.
        dtype=tf.float32,
        # Output range must correspond to expected lattice input range.
        output_min=0.0,
        output_max=max_w * 2,
        monotonicity='increasing')

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.RepeatVector(lattice_units_layer[0]))
    model.add(tf.keras.layers.Flatten())
    model.add(input_callibrators)
    model.add(tf.keras.layers.Reshape((lattice_units_layer[0], 2)))
    model.add(lattice1)
    model.add(tf.keras.layers.RepeatVector(lattice_units_layer[1]))
    model.add(tf.keras.layers.Flatten())
    model.add(calibrator2)
    model.add(tf.keras.layers.Reshape((lattice_units_layer[1], lattice_units_layer[0])))
    model.add(lattice2)
    model.add(tf.keras.layers.RepeatVector(lattice_units_layer[2]))
    model.add(tf.keras.layers.Flatten())
    model.add(calibrator3)
    model.add(tf.keras.layers.Reshape((lattice_units_layer[2], lattice_units_layer[1])))
    model.add(lattice3)
    model.add(tf.keras.layers.RepeatVector(lattice_units_layer[3]))
    model.add(tf.keras.layers.Flatten())
    model.add(calibrator4)
    model.add(tf.keras.layers.Reshape((lattice_units_layer[3], lattice_units_layer[2])))
    model.add(lattice4)
    model.add(calibratorf)
    # model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam())

    return model


if __name__ == '__main__':
    from learning.collections_env import CollectionsEnv

    environ = CollectionsEnv()
    model = construct_lattice(environ)
