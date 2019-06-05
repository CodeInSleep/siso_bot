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

layers_dims = [5, 10, 20, 4]

fields = ['input', 'sim_time', 'left_pwm', 'right_pwm',
        'theta_cos', 'theta_sin']

data_cached = False
model_cached = False
fname = 'real_robot_data.csv'
model_fname = fname.split('.')[0]+'_model'
dirname = 'real_robot_data'

def encode_angle(df, theta_field):
    df.loc[:, theta_field+'_cos'] = df.loc[:, theta_field].apply(lambda x: cos(x))
    df.loc[:, theta_field+'_sin'] = df.loc[:, theta_field].apply(lambda x: sin(x))

def predict_seq(model, X, initial_state, start, gnd_truth=None):
    # X is a 2D sequence of input features
    current_x = initial_state[0]
    current_y = initial_state[1]
    current_theta = initial_state[2]

    print('initial_state: ', initial_state)

    trajectory = []

    for i in range(len(X)):
        if gnd_truth is None or i % 10 != 0:
            encoded_theta = np.array([np.cos(current_theta), np.sin(current_theta)])
            _X = np.append(X[i], encoded_theta).reshape(1, -1)
        else:
            #parallels series
            encoded_theta = np.array([np.cos(gnd_truth[i, 2]), np.sin(gnd_truth[i, 2])])
            _X = np.append(X[i], encoded_theta).reshape(1, -1)

        prediction = model.predict(np.expand_dims(_X, axis=0)).ravel()

        if i > start:
            current_x += prediction[0]
            current_y += prediction[1]

            current_theta = decode_angles(prediction[2:].reshape(1, -1)).ravel()[0] if gnd_truth is None else gnd_truth[i, 2]

            trajectory.append(np.array([current_x, current_y, current_theta]))

    trajectory = np.array(trajectory)
    model.reset_states()
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

def plot_trajectories(pred_traj, gnd_traj, ax):
    plt.cla()
    ax1.set_title('trajectories')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    visualize_3D(np.expand_dims(pred_traj, axis=0), ax, plt_arrow=True, color='green')
    visualize_3D(np.expand_dims(gnd_traj, axis=0), ax, plt_arrow=True, color='red')
    ax1.legend(['predicted', 'ground truth'])
    
    plt.draw()
    plt.pause(5)

def plot_multiple_trajectories(model, X, y):
    # pdb.set_trace()
    num_plots = 16
    
    total_seq_len = 1600
    seq_len = int(total_seq_len/num_plots)
    X = X[:total_seq_len]
    y = y[:total_seq_len]
    # plot learning curve

    l = int(math.sqrt(num_plots))
    fig, axes = plt.subplots(l, l)
    
    fig.show()

    for i in range(l):
        for j in range(l):
            axes[i, j].clear()
            _X = X[(i*l+j)*seq_len:(i*l+j+1)*seq_len]
            _y = y[(i*l+j)*seq_len:(i*l+j+1)*seq_len]

            pred_traj = predict_seq(model, _X, _y[0])

            plot_trajectories(pred_traj, _y, axes[i, j])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get path to data directory')
    parser.add_argument('--datadir', required=True)
    args = parser.parse_args(sys.argv[1:])

    datadir = args.datadir
    if not os.path.isdir(datadir):
        print('invalid DATA_DIR (pass in as argument)')

    dirpath = os.path.abspath(os.path.join(datadir, dirname))
    print('dirpath: ', dirpath)
    datafile = os.path.join(dirpath, fname)
    df = pd.read_csv(datafile, engine='python')
    # for trial 2000
    # df = df.drop(['blank1', 'blank2'], axis=1)

    p = layers_dims[0]
    J = layers_dims[-1]
    X_train, X_test, y_train, y_test, timestep = transform(df, layers_dims, dirpath, cached=data_cached)
    # X columns: ['sim_time', 'left_pwm', 'right_pwm', 'model_pos_x(t-1)', 'model_pos_y(t-1)', 'theta(t-1)_cos', 'theta(t-1)_sin']
    # pdb.set_trace()
    # X_train = X_train.values.reshape(
    #     -1, timestep, p)
    # X_test = X_test.values.reshape(
    #         -1, timestep, p) 
    # y_train = y_train.values.reshape(
    #         -1, timestep, J)
    # y_test = y_test.values.reshape(
    #             -1, timestep, J)
    if not model_cached:
        model = make_model(None, layers_dims)
    else:
        model = load_model(dirpath, model_fname)
    iterations = 100
    epochs = 1
    # learning curver
    train_loss_history = []
    test_loss_history = []
    
    # decoded_y_train = decode_angles(y_train)
    # y_test = decode_angles(y_test)
    
    pdb.set_trace()

    train_trial_names = load_obj(dirpath, 'train_trial_names')
    test_trial_names = load_obj(dirpath, 'test_trial_names')

    test_fname = 'straight_1.csv' 
    testfile = os.path.join(dirpath, test_fname)
    testrun = pd.read_csv(testfile, engine='python')
    # testrun = scale(testrun, ['model_pos_x', 'model_pos_y'], 1000)
    print('train_trial_names: ', [(idx, name) for idx, name in enumerate(train_trial_names)])
    print('test_trial_names: ', [(idx, name) for idx, name in enumerate(test_trial_names)])
       
    input_scaler = joblib.load(os.path.join(dirpath, 'input_scaler.pkl'))

    x_sel = ['sim_time', 'left_pwm', 'right_pwm']
    y_sel = ['model_pos_x', 'model_pos_y', 'theta']
    # _testrun = downsample(testrun, rate='0.5S')

    testrun_X = testrun.loc[:, x_sel]
    testrun_y = testrun.loc[:, y_sel]
        
    testrun_X.loc[:, 'sim_time'] = testrun_X.loc[:, 'sim_time'].diff().fillna(0)
    testrun_X.loc[:, ['left_pwm', 'right_pwm']] = input_scaler.transform(testrun_X.loc[:, ['left_pwm', 'right_pwm']])
    testrun_y.loc[:, 'theta'] = testrun_y.loc[:, 'theta']*np.pi/180
    testrun_X = testrun_X.values
    testrun_y = testrun_y.values

    pdb.set_trace()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    plot_debug = True

    for it in range(iterations):
        print("iteration %d" % it)
        
        for i in range(len(X_train)):
            X = X_train.iloc[i].values.reshape(1, 1, -1)
            y = y_train.iloc[i].values.reshape(1, 1, -1)
            model.fit(X, y, epochs=epochs, verbose=1, shuffle=False)
        # for j in range(int(train_bl/time_step)-1):
        #     time_seg = range((j*time_step), ((j+1)*time_step))
        #     model.fit(X_train[:, time_seg, :], y_train[:, time_seg, :], 
        #             batch_size=num_batches, verbose=1, shuffle=False)
        # model.reset_states()

        # calculate rmse for train data
        train_se = []

        train_diff = 0
        for i in range(len(X_train)):
            X = X_train.iloc[i].values.reshape(1, 1, -1)
            pred = model.predict(X).reshape(1, -1)
            pred = np.concatenate((pred[:, :2], decode_angles(pred[:, 2:]).reshape(-1, 1)), axis=1)
            # decoded_train_predictions.append(pred)
            gnd_truth = y_train.iloc[i].values.reshape(1, -1)
            gnd_truth = np.concatenate((gnd_truth[:, :2], decode_angles(gnd_truth[:, 2:]).reshape(-1, 1)), axis=1)
            # decoded_train_gnd.append(gnd_truth)

            # diff = np.sum(np.apply_along_axis(angle_dist, 1, \
            #     np.concatenate((pred[:, 2].reshape(-1, 1), gnd_truth[:, 2].reshape(-1, 1)), axis=1))**2)
            # print('Training errors:')
            # print('angle diff: %f' % diff)
            train_diff += np.sum((pred - gnd_truth)**2)
            # print('total diff: %f' % diff)
            # train_se.append(diff)

        train_rmse = np.sqrt(train_diff/len(X_train))
        train_loss_history.append(train_rmse)
        
        test_se = []

        test_diff = 0
        # decoded_test_predictions = []
        # decoded_test_gnd = []
        for i in range(len(X_test)):
            X = X_test.iloc[i].values.reshape(1, 1, -1)
            pred = model.predict(X).reshape(1, -1)
            pred = np.concatenate((pred[:, :2], decode_angles(pred[:, 2:]).reshape(-1, 1)), axis=1)
            # decoded_test_predictions.append(pred)
            gnd_truth = y_test.iloc[i].values.reshape(1, -1)
            gnd_truth = np.concatenate((gnd_truth[:, :2], decode_angles(gnd_truth[:, 2:]).reshape(-1, 1)), axis=1)
            # decoded_test_gnd.append(gnd_truth)

            # print('Test errors:')
            # print('angle diff: %f' % diff)
            # diff = np.sum(np.apply_along_axis(angle_dist, 1, \
            #     np.concatenate((pred[:, 2].reshape(-1, 1), gnd_truth[:, 2].reshape(-1, 1)), axis=1))**2)
            # print('total diff: %f' % diff)
            test_diff += np.sum((pred - gnd_truth)**2)
            # test_se.append(diff)

        test_rmse = np.sqrt(test_diff/len(X_test))
        test_loss_history.append(test_rmse)

        print('train rmse: %f' % train_rmse)
        print('test rmse: %f' % test_rmse)

        if plot_debug:
            start = 0
            stateful_model = convert_to_inference_model(model)

            predictions = predict_seq(stateful_model, testrun_X, testrun_y[start], start, gnd_truth=testrun_y)
            
            # plot_multiple_trajectories(stateful_model, testrun_X, testrun_y)
            plot_trajectories(predictions[start:], testrun_y[start:], ax1)

    save_model(model, dirpath, model_fname)

    fig = plt.figure()
    ep_range = range(0, iterations)
    plt.plot(ep_range, train_loss_history)
    plt.plot(ep_range, test_loss_history)
    plt.title('Learning Curve on real robot data)')
    plt.xlabel('iteration (1 epochs/iteration)')
    plt.ylabel('RMSE')
    plt.show()

    
