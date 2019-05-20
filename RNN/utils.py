import os
import pickle
import numpy as np
from numpy import cos, sin, arctan2
import pandas as pd
from keras.models import Sequential, model_from_json
from keras.initializers import Identity, RandomNormal
from keras.layers import Dense, Dropout, LSTM, SimpleRNN, Dropout, GRU
from keras.optimizers import Adam

def decode_angles(cos_sin_vals):
    return arctan2(cos_sin_vals[:, 1], cos_sin_vals[:, 0]).reshape(-1, 1)

def trim_to_mult_of(df, size):
    if len(df)%size == 0:
        return df
    return df.iloc[:-(len(df)%size), :]

def plot_target_angles(arr, decode=False):
    # visualize theta (for debugging)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.show()
    for i in range(len(arr)):
        _gnd_truth = decode_angles(arr[i]) if decode else arr[i]
        padding = np.zeros((_gnd_truth.shape[0],2))
        vis_data = np.concatenate((padding, _gnd_truth), axis=1)

        for k in range(len(vis_data)):
            ax.clear()
            visualize_3D(np.expand_dims(vis_data[k].reshape(1, -1), axis=0), ax, plt_arrow=True)
            plt.draw()
            plt.pause(1)

def plot_multiple_trajectories():
    '''
    # for plotting multiple curves
    _X_train = X_train[:4]
    _y_train = y_train[:4]
    _X_test = X_test[:4]
    _y_test = y_test[:4]
    plot_l = 2
    plot_w = 2
    # plot learning curve
    train_fig, train_axes = plt.subplots(plot_l, plot_w)
    test_fig, test_axes = plt.subplots(plot_l, plot_w)
    
    train_fig.title = 'train trials'
    test_fig.title = 'test trials'
    train_fig.show()
    test_fig.show()
    for it in range(iterations):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)

        train_loss_history.append(calc_error(model, X_train, y_train, output_scaler))
        test_loss_history.append(calc_error(model, X_test, y_test, output_scaler)) 

        for idx, (x, y) in enumerate(product(range(plot_l), range(plot_w))):
            train_axes[x, y].clear()
            test_axes[x, y].clear()
       
        for idx, (x, y) in enumerate(product(range(plot_l), range(plot_w))):
            train_predictions = model.predict(twoD2threeD(_X_train[plot_l*x+y])) 
            test_predictions = model.predict(twoD2threeD(_X_test[plot_l*x+y]))
            visualize_3D(twoD2threeD(_y_train[plot_l*x+y]), train_axes[x, y])
            visualize_3D(train_predictions, train_axes[x, y])

            visualize_3D(twoD2threeD(_y_test[plot_l*x+y]), test_axes[x, y])
            visualize_3D(test_predictions, test_axes[x, y])
    '''
def angle_dist(angles):
    ang1 = math.degrees(angles[0])
    ang2 = math.degrees(angles[1])
    
    a = ang1 - ang2
    return np.radians(np.abs((a+180)%360-180))

def save_obj(obj, dirpath, name):
    with open(os.path.join(dirpath, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(dirpath, name):
    with open(os.path.join(dirpath, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

def load_dfs(dirpath):
    X_train = pd.read_pickle(os.path.join(dirpath, 'X_train.pkl'))
    X_test = pd.read_pickle(os.path.join(dirpath, 'X_test.pkl'))
    theta_y_train = pd.read_pickle(os.path.join(dirpath, 'theta_y_train.pkl'))
    pos_y_train = pd.read_pickle(os.path.join(dirpath, 'pos_y_train.pkl'))
    theta_y_test = pd.read_pickle(os.path.join(dirpath, 'theta_y_test.pkl'))
    pos_y_test = pd.read_pickle(os.path.join(dirpath, 'pos_y_test.pkl'))

    data_info = load_obj(dirpath, 'data_info')

    return (X_train, X_test, theta_y_train, pos_y_train, theta_y_test, pos_y_test, data_info)

def make_model(num_batches, time_step, layers_dims):
    model = Sequential()

    model.add(Dense(layers_dims[1], input_shape=(time_step, layers_dims[0]),
        kernel_initializer=RandomNormal(stddev=np.sqrt(2./layers_dims[0])), activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(layers_dims[1], activation='tanh', 
        kernel_initializer=RandomNormal(stddev=np.sqrt(2./layers_dims[1]))))
    model.add(Dropout(0.3))
    model.add(LSTM(layers_dims[2], activation='tanh', return_sequences=True))
    model.add(Dense(layers_dims[3]))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def load_model(dirpath, model_fname):
    # load theta RNN model
    json_file = open(os.path.join(dirpath, '{}.json'.format(model_fname)), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(os.path.join(dirpath, "{}.h5".format(model_fname)))
    print("Loaded {} from disk".format(model_fname))
    return model

def save_model(model, dirpath, model_fname):
    model_json = model.to_json()
    with open(os.path.join(dirpath, model_fname+'.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(dirpath, model_fname+'.h5'))
