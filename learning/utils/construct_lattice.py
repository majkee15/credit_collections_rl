import tensorflow_lattice as tfl
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

import matplotlib as m
import matplotlib.pyplot as plt

from dcc import AAV, Parameters


def construct_lattice(env, config, initialize=False):
    """
    Constructs deep lattice network with 832 parameters
    Returns:
            lattice model instance
    """

    min_w, min_l = (env.MIN_ACCOUNT_BALANCE, env.params.lambdainf)
    max_w, max_l = (env.w0, env.MAX_LAMBDA)

    ## Calibrators Block

    combined_calibratorsl = []
    combined_calibratorsr = []

    lattice_units_layer = [3, 3, 1]
    n_lattice_points = [3, 3, 3]

    for i in range(lattice_units_layer[0]):
        calibration_layer_l_l = tfl.layers.PWLCalibration(input_keypoints=np.linspace(0, max_l, num=100),
                                                          dtype=tf.float32, output_min=0.0,
                                                          output_max=n_lattice_points[0] - 1.0,
                                                          monotonicity='increasing', convexity='concave')
        calibration_layer_w_l = tfl.layers.PWLCalibration(input_keypoints=np.linspace(0, max_w, num=100),
                                                          dtype=tf.float32, output_min=0.0,
                                                          output_max=n_lattice_points[0] - 1.0,
                                                          monotonicity='increasing', convexity='convex')
        calibration_layer_l_r = tfl.layers.PWLCalibration(input_keypoints=np.linspace(0, max_l, num=100),
                                                          dtype=tf.float32, output_min=0.0,
                                                          output_max=n_lattice_points[0] - 1.0,
                                                          monotonicity='increasing', convexity='concave')
        calibration_layer_w_r = tfl.layers.PWLCalibration(input_keypoints=np.linspace(0, max_w, num=100),
                                                          dtype=tf.float32, output_min=0.0,
                                                          output_max=n_lattice_points[0] - 1.0,
                                                          monotonicity='increasing', convexity='convex')

        combined_calibratorsl.append(calibration_layer_l_l)
        combined_calibratorsl.append(calibration_layer_w_l)

        combined_calibratorsr.append(calibration_layer_l_r)
        combined_calibratorsr.append(calibration_layer_w_r)

    input_callibratorsl = tfl.layers.ParallelCombination(combined_calibratorsl, single_output=True)
    input_callibratorsr = tfl.layers.ParallelCombination(combined_calibratorsr, single_output=True)

    # API FUNCTIONAL MODEL
    inputs = keras.Input(shape=(2,))
    repeated_input = keras.layers.RepeatVector(lattice_units_layer[0])(inputs)
    repeated_input = keras.layers.Flatten()(repeated_input)
    # left tree
    calibrator1l = input_callibratorsl(repeated_input)
    calibrator1l = keras.layers.Reshape((lattice_units_layer[0], 2))(calibrator1l)
    lattice1l = tfl.layers.Lattice(units=lattice_units_layer[0], lattice_sizes=[n_lattice_points[0]] * 2,
                                   monotonicities=2 * ['increasing'], output_min=0, output_max=1)(calibrator1l)
    lattice1l = keras.layers.RepeatVector(lattice_units_layer[1])(lattice1l)
    lattice1l = keras.layers.Flatten()(lattice1l)
    calibrator2l = tfl.layers.PWLCalibration(input_keypoints=np.linspace(0, 1, num=100),
                                             units=lattice_units_layer[0] * lattice_units_layer[1], dtype=tf.float32,
                                             output_min=0.0, output_max=n_lattice_points[1] - 1.0,
                                             monotonicity='increasing')(lattice1l)
    calibrator2l = keras.layers.Reshape((lattice_units_layer[1], lattice_units_layer[0]))(calibrator2l)

    lattice2l = tfl.layers.Lattice(units=lattice_units_layer[1],
                                   lattice_sizes=[n_lattice_points[1]] * lattice_units_layer[0],
                                   monotonicities=['increasing'] * lattice_units_layer[0], output_min=0, output_max=1)(
        calibrator2l)

    calibrator3l = tfl.layers.PWLCalibration(units=lattice_units_layer[1] * lattice_units_layer[2],
                                             input_keypoints=np.linspace(0, 1, 100),
                                             dtype=tf.float32, monotonicity='increasing', output_min=0, output_max=n_lattice_points[2] - 1.0)(
        lattice2l)

    lattice3l = tfl.layers.Lattice(units=lattice_units_layer[2],
                                   lattice_sizes=[n_lattice_points[2]] * lattice_units_layer[1],
                                   monotonicities=['increasing'] * lattice_units_layer[0], output_min=0, output_max=1)(
        calibrator3l)

    calibratorfl = tfl.layers.PWLCalibration(units=lattice_units_layer[2], input_keypoints=np.linspace(0, 1, 100),
                                             dtype=tf.float32, monotonicity='increasing')(lattice3l)

    # right tree
    calibrator1r = input_callibratorsr(repeated_input)
    calibrator1r = keras.layers.Reshape((lattice_units_layer[0], 2))(calibrator1r)
    lattice1r = tfl.layers.Lattice(units=lattice_units_layer[0], lattice_sizes=[n_lattice_points[0]] * 2,
                                   monotonicities=2 * ['increasing'], output_min=0, output_max=1)(calibrator1r)
    lattice1r = keras.layers.RepeatVector(lattice_units_layer[1])(lattice1r)
    lattice1r = keras.layers.Flatten()(lattice1l)
    calibrator2r = tfl.layers.PWLCalibration(input_keypoints=np.linspace(0, 1, num=100),
                                             units=lattice_units_layer[0] * lattice_units_layer[1], dtype=tf.float32,
                                             output_min=0.0, output_max=n_lattice_points[1] - 1.0,
                                             monotonicity='increasing')(lattice1r)
    calibrator2r = keras.layers.Reshape((lattice_units_layer[1], lattice_units_layer[0]))(calibrator2r)

    lattice2r = tfl.layers.Lattice(units=lattice_units_layer[1],
                                   lattice_sizes=[n_lattice_points[1]] * lattice_units_layer[0],
                                   monotonicities=['increasing'] * lattice_units_layer[0], output_min=0, output_max=1)(
        calibrator2r)

    calibrator3r = tfl.layers.PWLCalibration(units=lattice_units_layer[1] * lattice_units_layer[2],
                                             input_keypoints=np.linspace(0, 1, 100),
                                             dtype=tf.float32, monotonicity='increasing', output_min=0, output_max=n_lattice_points[2] - 1.0)(
        lattice2r)

    lattice3r = tfl.layers.Lattice(units=lattice_units_layer[2],
                                   lattice_sizes=[n_lattice_points[2]] * lattice_units_layer[1],
                                   monotonicities=['increasing'] * lattice_units_layer[0], output_min=0, output_max=1)(
        calibrator3r)

    calibratorfr = tfl.layers.PWLCalibration(units=lattice_units_layer[2], input_keypoints=np.linspace(0, 1, 100),
                                             dtype=tf.float32, monotonicity='increasing')(lattice3r)

    # Combine
    final = keras.layers.Concatenate()([calibratorfl, calibratorfr])

    model = keras.Model(inputs=inputs, outputs=final, name="combined")
    model.compile(loss=tf.keras.losses.mean_squared_error, optimizer='adam', metrics=['mean_absolute_percentage_error'])


    if initialize:
        # works only for one discrete action
        aav = AAV(env.params)
        if config.normalize_states:
            lowbounds = env.observation(np.array([env.params.lambda0, env.MIN_ACCOUNT_BALANCE]))
            highbounds = env.observation(np.array([env.MAX_LAMBDA, env.w0]))
            ws = np.linspace(lowbounds[1], highbounds[1], 100)
            ls = np.linspace(lowbounds[0], highbounds[0], 100)
        else:
            ws = np.linspace(0, env.w0, 100)
            ls = np.linspace(0, env.MAX_LAMBDA, 100)
        wt = np.linspace(0, env.w0, 100)
        lt = np.linspace(0, env.MAX_LAMBDA, 100)
        ww, ll = np.meshgrid(ws, ls)
        z = np.zeros_like(ww)
        zt = np.zeros_like(ww)
        features = []
        for i, wx in enumerate(ws):
            for j, ly in enumerate(ls):
                z[j, i] = -aav.u(lt[j], wt[i])
                zt[j, i] = -aav.u(lt[j] + env.action(1), wt[i]) - env.action(1) * env.params.c
                features.append([ls[j], ws[i], z[j, i], zt[j, i]])

        dataset = pd.DataFrame(features, columns=['l', 'w', 'target', 'target2'])
        train_dataset = dataset.sample(frac=1.0, random_state=0)
        # test_dataset = dataset.drop(train_dataset.index)
        train_labels = train_dataset[['target', 'target2']].copy()
        # test_labels = test_dataset[['target', 'target2']].copy()
        train_dataset = train_dataset.drop(labels=['target', 'target2'], axis=1)
        # test_dataset = test_dataset.drop(labels=['target', 'target2'], axis=1)

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        model.fit(train_dataset.to_numpy(), train_labels.to_numpy(), epochs=50, validation_split=0.1,
                  shuffle=False, verbose=True, callbacks=[callback])

        lattice_pred = np.zeros_like(ww)
        lattice_pred2 = np.zeros_like(ww)
        for i, wx in enumerate(ws):
            for j, ly in enumerate(ls):
                obs = np.array([ls[j], ws[i]])
                pred = model.predict_on_batch(obs[None, :]).numpy()
                lattice_pred[j, i] = pred[0][0]
                lattice_pred2[j, i] = pred[0][1]
        fig, ax = plt.subplots(ncols=2, nrows=1)

        CS = ax[0].contour(ww, ll, lattice_pred)
        ax[0].clabel(CS, inline=1, fontsize=10)
        ax[0].set_title('vf')
        CS = ax[1].contour(ww, ll, lattice_pred2)
        ax[1].clabel(CS, inline=1, fontsize=10)
        ax[1].set_title('vf')
        plt.show()
        ##
        w_points = 60
        l_points = 60
        l = np.linspace(0, 5, l_points)
        w = np.linspace(0, 200, w_points)
        ww, ll = np.meshgrid(w, l)
        z = np.zeros_like(ww)
        p = np.zeros_like(ww)
        for i, xp in enumerate(w):
            for j, yp in enumerate(l):
                fixed_obs = np.array([ls[j], ws[i]])
                z[j, i] = np.argmax(model.predict_on_batch(fixed_obs[None, :]).numpy().flatten())

        fig, ax = plt.subplots(nrows=1, ncols=2)
        im = ax[0].pcolor(ww, ll, p)
        cdict = {
            'red': ((0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
            'green': ((0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
            'blue': ((0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
        }

        cm = m.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
        im = ax[0].pcolor(ww, ll, z, cmap=cm)
        fig.colorbar(im)
        fig.show()

    return model


if __name__ == '__main__':
    from learning.collections_env import CollectionsEnv

    environ = CollectionsEnv()
    model = construct_lattice(environ)
