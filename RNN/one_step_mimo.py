import argparse
import math
import os
import sys
import pdb
import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle

from itertools import product
from numpy import arctan2
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, LSTM, Dropout, GRU
from keras.initializers import Identity, RandomNormal
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from utils import decode_angles, load_obj, make_model, save_model, load_model, load_dfs

from visualize import visualize_3D

layers_dims = [5, 10, 20, 4]
fname = 'trial_1000_0_to_3.csv'

# network parameter
p = layers_dims[0]
J = layers_dims[-1]

# TODOs
#   Hyperparmaters:
#       - dropout keep_prob
#       - Gradient clipping
#   Evaluation:
#       - stability (gradient visualization, gradient clipping)
#       - learning speed
#       - predict arbitrary length

def convert_to_inference_model(original_model):
    original_model_json = original_model.to_json()
    inference_model_dict = json.loads(original_model_json)

    layers = inference_model_dict['config']['layers']
    for layer in layers:
        if 'stateful' in layer['config']:
            layer['config']['stateful'] = True

        if 'batch_input_shape' in layer['config']:
            layer['config']['batch_input_shape'][0] = 1
            layer['config']['batch_input_shape'][1] = None

    inference_model = model_from_json(json.dumps(inference_model_dict))
    inference_model.set_weights(original_model.get_weights())
    inference_model.reset_states()
    return inference_model

def predict_seq(theta_model, xy_model, X, start_theta):
    # X is a 2D sequence of input features
    current_x = 0
    current_y = 0
    current_theta = start_theta

    trajectory = []

    for i in range(len(X)):
        encoded_theta = np.array([np.cos(current_theta), np.sin(current_theta)])
        pos_X = X[i]
        pos_X = np.append(pos_X, current_theta).reshape(1, -1)

        theta_X = X[i]
        theta_X = np.expand_dims(np.append(theta_X, encoded_theta).reshape(1, -1), axis=0)

        pos_prediction = xy_model.predict(pos_X).ravel()
        theta_prediction = theta_model.predict(theta_X).ravel()

        current_x += pos_prediction[0]
        current_y += pos_prediction[1]

        print(decode_angles(theta_prediction.reshape(1, -1)))
        current_theta = decode_angles(theta_prediction.reshape(1, -1)).ravel()[0]

        trajectory.append(np.array([current_x, current_y, current_theta]))

    return np.array(trajectory)

def calc_error(model, X, y, output_scaler):
    # X, y are unnormalized 3D
    rmse = 0

    for i in range(len(X)):
        predictions = predict_seq(model, X[i], output_scaler)
        unnorm_y = output_scaler.inverse_transform(y[i])

        rmse += np.sqrt(mean_squared_error(y[i], predictions))
    return rmse/y.size

def recover_y(pos_y, theta_y, timestep, num_trials):
    y = pd.concat([pos_y, theta_y], axis=1)
    y.loc[:, 'theta'] = y.loc[:, ['theta_cos', 'theta_sin']].values
    y = y.drop(['theta_cos', 'theta_sin'], axis=1)
    
    for i in range(num_trials):
        print(i)
        time_seg = range((i*timestep), ((i+1)*timestep))
        y.iloc[time_seg].loc[:, ['model_pos_x', 'model_pos_y']] = y.iloc[time_seg].loc[:, ['model_pos_x', 'model_pos_y']].cumsum()

    return y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get path to data directory')
    parser.add_argument('--datadir', required=True)
    args = parser.parse_args(sys.argv[1:])

    datadir = args.datadir
    if not os.path.isdir(datadir):
        print('invalid DATA_DIR (pass in as argument)')

    dirpath = os.path.abspath(os.path.join(datadir, fname.split('.')[0]))

    X_train, X_test, theta_y_train, pos_y_train, \
        theta_y_test, pos_y_test, train_traj, test_traj, data_info = load_dfs(dirpath)
    
    theta_model = load_model(dirpath, 'theta_model')
    xy_model = load_model(dirpath, 'pos_model')

    num_trials = data_info['num_trials']
    num_train_trials = data_info['num_train_trials']
    num_test_trials = data_info['num_test_trials']
    timestep = data_info['timestep']

    # y_train = recover_y(pos_y_train, theta_y_train, timestep, num_train_trials)
    # y_test = recover_y(pos_y_test, theta_y_test, timestep, num_test_trials)
    theta_model = convert_to_inference_model(theta_model)

    X_sel = ['sim_time', 'left_pwm', 'right_pwm']
    y_sel = ['model_pos_x', 'model_pos_y', 'theta']
    X_train = X_train.loc[:, X_sel]
    X_test = X_test.loc[:, X_sel]
    # y_train = y_train.loc[:, y_sel]
    # y_test = y_test.loc[:, y_sel]

    X_train_arr = X_train.values.reshape(
        -1, timestep, len(X_sel))
    X_test_arr = X_test.values.reshape(
            -1, timestep, len(X_sel))
    y_train_arr = train_traj.values.reshape(
            -1, timestep, len(y_sel))
    y_test_arr = test_traj.values.reshape(
                -1, timestep, len(y_sel))

    # plot trajectory for batch
    n = 40
    _X_train = X_train_arr[n]
    _X_test = X_test_arr[n]
    _y_train = y_train_arr[n]
    _y_test = y_test_arr[n]
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # train trajectory
    traj1 = predict_seq(theta_model, xy_model, _X_train, _y_train[0,2])
    # test trajectory
    traj2 = predict_seq(theta_model, xy_model, _X_test, _y_test[0,2])

    ax1.set_title('train example')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    visualize_3D(np.expand_dims(traj1, axis=0), ax1, plt_arrow=True)
    visualize_3D(np.expand_dims(_y_train, axis=0), ax1, plt_arrow=True)
    ax1.legend(['predicted', 'ground truth'])
    ax2.set_title('test example')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    visualize_3D(np.expand_dims(traj2, axis=0), ax2, plt_arrow=True)
    visualize_3D(np.expand_dims(_y_test, axis=0), ax2, plt_arrow=True)
    ax2.legend(['predicted', 'ground truth'])
    plt.draw()
    plt.show()
    