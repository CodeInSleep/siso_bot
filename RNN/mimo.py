import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
import pdb
from numpy import cos, sin, arctan2
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from transform import transform, downsample, truncate, scale
from visualize import visualize_3D
from utils import decode_angles, plot_target_angles, save_model, angle_dist, make_model, load_model, load_obj, convert_to_inference_model

input_fields = ['left_pwm', 'right_pwm']

layers_dims = [4, 10, 20, 3]

fields = ['input', 'sim_time', 'left_pwm', 'right_pwm',
        'theta_cos', 'theta_sin']

data_cached = True
model_cached = False
fname = 'trial_1000_0_to_3.csv'

def encode_angle(df, theta_field):
    df.loc[:, theta_field+'_cos'] = df.loc[:, theta_field].apply(lambda x: cos(x))
    df.loc[:, theta_field+'_sin'] = df.loc[:, theta_field].apply(lambda x: sin(x))

def update_angle(old_angle, angle_diff):
    new_angle = old_angle + angle_diff
    if new_angle > 360:
        return new_angle - 360
    elif new_angle < 0:
        return 360+new_angle
    return new_angle

def predict_seq(model, X, initial_state, start, gnd_truth=None):
    # X is a 2D sequence of input features
    current_x = initial_state[0]
    current_y = initial_state[1]
    current_theta = initial_state[2]

    print('initial_state: ', initial_state)

    trajectory = []

    for i in range(len(X)):
        if gnd_truth is None:
            # encoded_theta = np.array([np.cos(current_theta), np.sin(current_theta)])
            # _X = np.append(X[i], [encoded_theta[0], encoded_theta[1]]).reshape(1, -1)
            _X = np.append(X[i], current_theta)
        else:
            #parallels series
            # encoded_theta = np.array([np.cos(gnd_truth[i, 2]), np.sin(gnd_truth[i, 2])])
            # _X = np.append(X[i], encoded_theta).reshape(1, -1)
            _X = np.append(X[i], gnd_truth[i, 2])
        
        prediction = model.predict(_X.reshape(1, 1, -1)).ravel()

        if i >= start:
            current_x += prediction[0]
            current_y += prediction[1]
            current_theta = update_angle(current_theta, prediction[2])

            # current_theta = decode_angles(prediction[2:].reshape(1, -1)).ravel()[0]

            trajectory.append(np.array([current_x, current_y, np.radians(current_theta)]))

    trajectory = np.array(trajectory)
    return trajectory

def plot_example(gnd_truth, predictions, n):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.title('example')
    plt.xlabel('x')
    plt.ylabel('y')

    gnd = gnd_truth[n]
    gnd = np.concatenate((gnd[:, :2], decode_angles(gnd[:, 2:]).reshape(-1, 1)), axis=1)
    # decode angle
    pred = predictions[n]
    pred = np.concatenate((pred[:, :2], decode_angles(pred[:, 2:]).reshape(-1, 1)), axis=1)

    visualize_3D(np.expand_dims(gnd, axis=0), ax1, plt_arrow=True)
    visualize_3D(np.expand_dims(pred, axis=0), ax1, plt_arrow=True)
    plt.legend(['ground truth', 'prediction'])
    plt.title('test example {}'.format(n))
    plt.show()

def plot_trajectories(pred_traj, gnd_traj, ax1):
    plt.cla()
    ax1.set_title('trajectories')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    visualize_3D(np.expand_dims(pred_traj, axis=0), ax1, plt_arrow=True)
    visualize_3D(np.expand_dims(gnd_traj, axis=0), ax1, plt_arrow=True)
    ax1.legend(['predicted', 'ground truth'])
    
    plt.draw()
    plt.pause(5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get path to data directory')
    parser.add_argument('--datadir', required=True)
    args = parser.parse_args(sys.argv[1:])

    datadir = args.datadir
    if not os.path.isdir(datadir):
        print('invalid DATA_DIR (pass in as argument)')

    dirpath = os.path.abspath(os.path.join(datadir, fname.split('.')[0]))
    print('dirpath: ', dirpath)
    datafile = os.path.join(dirpath, fname)
    df = pd.read_csv(datafile, engine='python')
    # for trial 2000
    # df = df.drop(['blank1', 'blank2'], axis=1)

    p = layers_dims[0]
    J = layers_dims[-1]
    X_train, X_test, y_train, y_test, timestep = transform(df, layers_dims, dirpath, cached=data_cached)
    # X columns: ['sim_time', 'left_pwm', 'right_pwm', 'model_pos_x(t-1)', 'model_pos_y(t-1)', 'theta(t-1)_cos', 'theta(t-1)_sin']
    pdb.set_trace()
    # X_train = X_train.values.reshape(
    #     -1, timestep, p)
    # X_test = X_test.values.reshape(
    #         -1, timestep, p) 
    # y_train = y_train.values.reshape(
    #         -1, timestep, J)
    # y_test = y_test.values.reshape(
    #             -1, timestep, J)

    theta_model, xy_model = make_model(None, layers_dims, lr=1e-4)
   
    iterations = 10
    epochs = 1
    # learning curver
    train_loss_history = []
    test_loss_history = []
    
    # decoded_y_train = decode_angles(y_train)
    # y_test = decode_angles(y_test)
    
    train_trial_names = load_obj(dirpath, 'train_trial_names')
    test_trial_names = load_obj(dirpath, 'test_trial_names')

    testfile = os.path.join(dirpath, fname)
    testrun = pd.read_csv(testfile, engine='python')
    testrun = scale(testrun, ['model_pos_x', 'model_pos_y'], 1000)
    print('train_trial_names: ', [(idx, name) for idx, name in enumerate(train_trial_names)])
    print('test_trial_names: ', [(idx, name) for idx, name in enumerate(test_trial_names)])
       
    input_scaler = joblib.load(os.path.join(dirpath, 'input_scaler.pkl'))

    x_sel = ['sim_time', 'left_pwm', 'right_pwm']
    y_sel = ['model_pos_x', 'model_pos_y', 'theta']
    start = 400
    finish = 420
    _testrun = downsample(testrun, rate='0.2S')
    testrun_X = _testrun.iloc[start: finish].loc[:, x_sel]
    testrun_y = _testrun.iloc[start: finish].loc[:, y_sel]

    # X = X.loc[:, x_sel]
    # y = X.loc[:, y_sel]
        
    testrun_X.loc[:, 'sim_time'] = testrun_X.loc[:, 'sim_time'].diff().fillna(0)
    testrun_X.loc[:, ['left_pwm', 'right_pwm']] = input_scaler.transform(testrun_X.loc[:, ['left_pwm', 'right_pwm']])
    testrun_y.loc[:, 'theta'] = testrun_y.loc[:, 'theta']*np.pi/180
    testrun_X = testrun_X.values
    testrun_y = testrun_y.values

    start = 0
    fig = plt.figure()
    ax1 = fig.add_subplot(121)

    plot_debug = False
    model_to_train = 'theta'
    model = theta_model if model_to_train == 'theta' else xy_model
    selector = range(2, 3) if model_to_train == 'theta' else range(0, 2)
    model_fname = model_to_train+'_model'

    pdb.set_trace()
    for it in range(iterations):
        print("iteration %d" % it)
        if not model_cached:
            for i in range(len(X_train)):
                X = X_train.iloc[i].values.reshape(1, -1)
                y = y_train.iloc[i].values.reshape(1, -1)[:, selector]
                model.fit(X, y, epochs=epochs, shuffle=False)

            # calculate rmse for train data
            train_se = []

            train_diff = 0
            for i in range(len(X_train)):
                X = X_train.iloc[i].values.reshape(1, -1)
                pred = model.predict(X).reshape(1, -1)
                gnd_truth = y_train.iloc[i].values.reshape(1, -1)[:, selector]
                train_diff += np.sum((pred - gnd_truth)**2)

            train_rmse = np.sqrt(train_diff/len(X_train))
            train_loss_history.append(train_rmse)
            
            test_se = []

            test_diff = 0
            for i in range(len(X_test)):
                X = X_test.iloc[i].values.reshape(1, -1)
                pred = model.predict(X).reshape(1, -1)
                gnd_truth = y_test.iloc[i].values.reshape(1, -1)[:, selector]
                test_diff += np.sum((pred - gnd_truth)**2)
            test_rmse = np.sqrt(test_diff/len(X_test))
            test_loss_history.append(test_rmse)

            print('train rmse: %f' % train_rmse)
            print('test rmse: %f' % test_rmse)

            if plot_debug:
                stateful_model = convert_to_inference_model(model)

                predictions = predict_seq(stateful_model, testrun_X, testrun_y[start], start, gnd_truth=testrun_y)
                plot_trajectories(predictions, testrun_y[start:], ax1)

    if not model_cached:
        save_model(model, dirpath, model_fname)

        fig = plt.figure()
        ep_range = range(0, iterations)
        plt.plot(ep_range, train_loss_history)
        plt.plot(ep_range, test_loss_history)
        plt.title('theta miso prediction (RNN)')
        plt.xlabel('iteration (30 epochs/iteration)')
        plt.ylabel('RMSE')
        plt.show()

    