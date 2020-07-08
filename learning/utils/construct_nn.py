import tensorflow as tf
from dcc import AAV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as m

def construct_nn(env, layers, config, initialize=False):
    target_net = tf.keras.Sequential()
    target_net.add(tf.keras.layers.Input(shape=env.observation_space.shape))
    for i, layer_size in enumerate(layers):
        target_net.add(tf.keras.layers.Dense(layer_size, activation='relu'))
        if config.batch_normalization:
            target_net.add(tf.keras.layers.BatchNormalization())
    target_net.add(tf.keras.layers.Dense(env.action_space.n, activation='linear'))
    target_net.build()
    target_net.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

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
        train_dataset = dataset.sample(frac=0.8, random_state=0)
        test_dataset = dataset.drop(train_dataset.index)
        train_labels = train_dataset[['target', 'target2']].copy()
        test_labels = test_dataset[['target', 'target2']].copy()
        train_dataset = train_dataset.drop(labels=['target', 'target2'], axis=1)
        test_dataset = test_dataset.drop(labels=['target', 'target2'], axis=1)

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        target_net.fit(train_dataset.to_numpy(), train_labels.to_numpy(), epochs=1000, validation_split=0.1,
                       shuffle=False, verbose=True, callbacks=[callback])

        lattice_pred = np.zeros_like(ww)
        lattice_pred2 = np.zeros_like(ww)
        for i, wx in enumerate(ws):
            for j, ly in enumerate(ls):
                obs = np.array([ls[j], ws[i]])
                pred = target_net.predict_on_batch(obs[None, :]).numpy()
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
                z[j, i] = np.argmax(target_net.predict_on_batch(fixed_obs[None, :]).numpy().flatten())

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

    return target_net
