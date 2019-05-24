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
from transform import transform
from visualize import visualize_3D
from utils import decode_angles, plot_target_angles, save_model, angle_dist, make_model, load_model, load_obj, convert_to_inference_model

input_fields = ['left_pwm', 'right_pwm']

layers_dims = [5, 10, 20, 4]
fname = 'real_robot_data.csv'
model_fname = 'multi_step_mimo_model'

fields = ['input', 'sim_time', 'left_pwm', 'right_pwm',
        'theta_cos', 'theta_sin']

def encode_angle(df, theta_field):
    df.loc[:, theta_field+'_cos'] = df.loc[:, theta_field].apply(lambda x: cos(x))
    df.loc[:, theta_field+'_sin'] = df.loc[:, theta_field].apply(lambda x: sin(x))

def predict_seq(model, X, initial_state):
    # X is a 2D sequence of input features
    current_x = initial_state[0]
    current_y = initial_state[1]
    current_theta = initial_state[2]

    print('initial_state: ', initial_state)

    trajectory = []

    for i in range(len(X)):
        encoded_theta = np.array([np.cos(current_theta), np.sin(current_theta)])
        _X = np.append(X[i], [encoded_theta[0], encoded_theta[1]]).reshape(1, -1)

        prediction = model.predict(np.expand_dims(_X, axis=0)).ravel()

        current_x += prediction[0]
        current_y += prediction[1]

        current_theta = decode_angles(prediction[2:].reshape(1, -1)).ravel()[0]

        trajectory.append(np.array([current_x, current_y, current_theta]))

    print('trajectory: ', trajectory[:10])
    return np.array(trajectory)

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

def plot_trajectories(pred_traj, gnd_traj):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_title('trajectories')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    visualize_3D(np.expand_dims(pred_traj, axis=0), ax1, plt_arrow=True)
    visualize_3D(np.expand_dims(gnd_traj, axis=0), ax1, plt_arrow=True)
    ax1.legend(['predicted', 'ground truth'])
    
    plt.draw()
    plt.show()

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

    p = layers_dims[0]
    J = layers_dims[-1]
    X_train, X_test, y_train, y_test, timestep = transform(df, layers_dims, dirpath, cached=False)
    # X columns: ['sim_time', 'left_pwm', 'right_pwm', 'model_pos_x(t-1)', 'model_pos_y(t-1)', 'theta(t-1)_cos', 'theta(t-1)_sin']
    pdb.set_trace()
    X_train = X_train.values.reshape(
        -1, timestep, p)
    X_test = X_test.values.reshape(
            -1, timestep, p) 
    y_train = y_train.values.reshape(
            -1, timestep, J)
    y_test = y_test.values.reshape(
                -1, timestep, J)

    num_batches = 16
    model = make_model(num_batches, timestep, layers_dims)
   
    iterations = 10
    epochs = 30
    # learning curver
    train_loss_history = []
    test_loss_history = []
    
    # decoded_y_train = decode_angles(y_train)
    # y_test = decode_angles(y_test)
    
    model_cached = False
    train_trial_names = load_obj(dirpath, 'train_trial_names')
    test_trial_names = load_obj(dirpath, 'test_trial_names')

    if model_cached:
        print('train_trial_names: ', [(idx, name) for idx, name in enumerate(train_trial_names)])
        print('test_trial_names: ', [(idx, name) for idx, name in enumerate(test_trial_names)])
        model = load_model(dirpath, model_fname)
        model = convert_to_inference_model(model)
        input_scaler = joblib.load(os.path.join(dirpath, 'input_scaler.pkl'))

        x_sel = ['sim_time', 'left_pwm', 'right_pwm']
        y_sel = ['model_pos_x', 'model_pos_y', 'theta']
        start = 400
        finish = 410
        X = df.iloc[start: finish].loc[:, x_sel]
        y = df.iloc[start: finish].loc[:, y_sel]

        X.loc[:, 'sim_time'] = X.loc[:, 'sim_time'].diff().fillna(0)
        X.loc[:, ['left_pwm', 'right_pwm']] = input_scaler.transform(X.loc[:, ['left_pwm', 'right_pwm']])
        y.loc[:, 'theta'] = y.loc[:, 'theta']*np.pi/180

        X = X.values
        y = y.values

        predictions = predict_seq(model, X, y[0])
        plot_trajectories(predictions, y)

    for it in range(iterations):
        print("iteration %d" % it)
        if not model_cached:
            model.fit(X_train, y_train, epochs=epochs, batch_size=num_batches, verbose=1, shuffle=False)
            # for j in range(int(train_bl/time_step)-1):
            #     time_seg = range((j*time_step), ((j+1)*time_step))
            #     model.fit(X_train[:, time_seg, :], y_train[:, time_seg, :], 
            #             batch_size=num_batches, verbose=1, shuffle=False)
            # model.reset_states()

            # calculate rmse for train data
            train_se = []

            train_predictions = model.predict(X_train)
            decoded_train_predictions = []
            decoded_train_gnd = []
            for i in range(len(train_predictions)):
                pred = train_predictions[i]
                pred = np.concatenate((pred[:, :2], decode_angles(pred[:, 2:]).reshape(-1, 1)), axis=1)
                decoded_train_predictions.append(pred)
                gnd_truth = y_train[i]
                gnd_truth = np.concatenate((gnd_truth[:, :2], decode_angles(gnd_truth[:, 2:]).reshape(-1, 1)), axis=1)
                decoded_train_gnd.append(gnd_truth)

                diff = np.sum(np.apply_along_axis(angle_dist, 1, \
                    np.concatenate((pred[:, 2].reshape(-1, 1), gnd_truth[:, 2].reshape(-1, 1)), axis=1))**2)
                diff += np.sum((pred[:, :2] - gnd_truth[:, :2])**2)
                train_se.append(diff)

            train_rmse = np.sqrt(np.mean(np.array(train_se))/timestep)
            train_loss_history.append(train_rmse)
            
            test_se = []

            test_predictions = model.predict(X_test)
            decoded_test_predictions = []
            decoded_test_gnd = []
            for i in range(len(X_test)):
                pred = test_predictions[i]
                pred = np.concatenate((pred[:, :2], decode_angles(pred[:, 2:]).reshape(-1, 1)), axis=1)
                decoded_test_predictions.append(pred)
                gnd_truth = y_test[i]
                gnd_truth = np.concatenate((gnd_truth[:, :2], decode_angles(gnd_truth[:, 2:]).reshape(-1, 1)), axis=1)
                decoded_test_gnd.append(gnd_truth)

                diff = np.sum(np.apply_along_axis(angle_dist, 1, \
                    np.concatenate((pred[:, 2].reshape(-1, 1), gnd_truth[:, 2].reshape(-1, 1)), axis=1))**2)
                diff += np.sum((pred[:, :2] - gnd_truth[:, :2])**2)
                test_se.append(diff)

            test_rmse = np.sqrt(np.mean(np.array(test_se))/timestep)
            test_loss_history.append(test_rmse)

            print('train_rmse: ', train_rmse)
            print('test_rmse: ', test_rmse)

    
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

    