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
from transform import transform, upsample, truncate, scale
from visualize import visualize_3D
from utils import decode_angles, plot_target_angles, save_model, \
    angle_dist, make_model, load_model, load_obj, convert_to_inference_model, \
    angle_diff

input_fields = ['left_pwm', 'right_pwm']

layers_dims = [5, 10, 20, 4]

fields = ['input', 'sim_time', 'left_pwm', 'right_pwm',
        'theta_cos', 'theta_sin']

data_cached = False
model_cached = False
fname = 'start_and_final_5.csv'
model_fname = fname.split('.')[0]+'_FNN_model'
dirname = 'real_robot_data'
np.random.seed(6)


def encode_angle(df, theta_field):
    df.loc[:, theta_field+'_cos'] = df.loc[:, theta_field].apply(lambda x: cos(x))
    df.loc[:, theta_field+'_sin'] = df.loc[:, theta_field].apply(lambda x: sin(x))

def update_angle(old_angle, diff):
    new_angle = old_angle + diff
    while new_angle > np.pi or new_angle < -np.pi:
        new_angle = angle_diff(np.array([[new_angle, 0]])).ravel()[0]
    return new_angle

def predict_seq(stateful_model, X, initial_state, output_scaler=None, gnd_truth=None):
    # X is a 2D sequence of input features
    current_x = initial_state[0]
    current_y = initial_state[1]
    current_theta = initial_state[2]

    print('initial_state: ', initial_state)

    trajectory = [initial_state]

    interval = 10
    for i in range(len(X)):
        # if gnd_truth is None or i % interval != 0:
        if gnd_truth is None or i == 0:
            encoded_theta = np.array([np.cos(current_theta), np.sin(current_theta)])
            _X = np.append(X[i, :3], encoded_theta).reshape(1, 1, -1)
        else:
            #parallels series
            encoded_theta = np.array([np.cos(gnd_truth[i-1, 2]), np.sin(gnd_truth[i-1, 2])])
            _X = np.append(X[i, :3], encoded_theta).reshape(1, 1, -1)
        # _X = np.append(X[i], current_theta).reshape(1, 1, -1)

        predictions = model.predict(_X).ravel()
        print('predictions: ', predictions)

        unnorm_xy = output_scaler.inverse_transform(predictions[:2].reshape(1, -1)).ravel()
        current_x += unnorm_xy[0]
        current_y += unnorm_xy[1]

        current_theta = gnd_truth[i, 2]
        # current_theta = arctan2(predictions[3], predictions[2])
        #current_theta = update_angle(current_theta, predictions[2])

        #np.concatenate((pred[:, :2], decode_angles(pred[:, 2:]).reshape(-1, 1)), axis=1)

        trajectory.append(np.array([current_x, current_y, current_theta]))

    stateful_model.reset_states()
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

def plot_trajectories(pred_traj, gnd_traj, ax):
    plt.cla()
    ax1.set_title('Predicted vs. Ground Truth Trajectory')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    visualize_3D(np.expand_dims(pred_traj, axis=0), ax, plt_arrow=True, color='green')
    visualize_3D(np.expand_dims(pred_traj, axis=0), ax, plt_arrow=False, color='green')
    visualize_3D(np.expand_dims(gnd_traj, axis=0), ax, plt_arrow=True, color='red')
    visualize_3D(np.expand_dims(gnd_traj, axis=0), ax, plt_arrow=False, color='red')
    ax1.legend(['predicted', 'ground truth'])
    
    plt.draw()
    plt.pause(5)

def plot_multiple_trajectories(model, X, y, warmup=0):
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

            pred_traj = predict_seq(model, _X, _y[0], warmup)

            plot_trajectories(pred_traj, _y, axes[i, j])

def reverse_map(df, field, mapping=None):
    if mapping is not None:
        right_wheel_inputs = np.sort(df.loc[:, field].unique())
        reversed_right_wheel_inputs = np.flip(right_wheel_inputs)
        mapping = {
            rw_input: reversed_right_wheel_inputs[idx] \
                for idx, rw_input in enumerate(list(right_wheel_inputs))
        }

    df.loc[:, field] = df.loc[:, field].replace(mapping)
    return df

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
    right_pwm_mapping = reverse_map(df, 'right_pwm')
    X_train, X_test, y_train, y_test = transform(df, dirpath, cached=data_cached, split=True)
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

    if model_cached:
        model = load_model(dirpath, model_fname)
    else:
        model = make_model(None, layers_dims)
   
    iterations = 300
    epochs = 30
    # learning curver
    train_loss_history = []
    test_loss_history = []
    
    # decoded_y_train = decode_angles(y_train)
    # y_test = decode_angles(y_test)
    
    # train_trial_names = load_obj(dirpath, 'train_trial_names')
    # test_trial_names = load_obj(dirpath, 'test_trial_names')

    test_fname = 'v_shape_path_2.csv' 
    testfile = os.path.join(dirpath, test_fname)
    testrun = pd.read_csv(testfile, engine='python')
    # testrun = scale(testrun, ['model_pos_x', 'model_pos_y'], 1000)
    # print('train_trial_names: ', [(idx, name) for idx, name in enumerate(train_trial_names)])
    # print('test_trial_names: ', [(idx, name) for idx, name in enumerate(test_trial_names)])
       
    input_scaler = joblib.load(os.path.join(dirpath, 'input_scaler.pkl'))
    output_scaler = joblib.load(os.path.join(dirpath, 'output_scaler.pkl'))
    # theta_data = theta_data.groupby('input').apply(lambda x: upsample(x, rate='0.01S', start_of_batches=start_of_batches))

    # pdb.set_trace()
    reverse_map(testrun, 'right_pwm', mapping=right_pwm_mapping)
    testrun = transform(testrun, dirpath, split=False, input_scaler=input_scaler, output_scaler=output_scaler)

    x_sel = ['time_duration', 'left_pwm', 'right_pwm', 'theta_start_cos', 'theta_start_sin']
    y_sel = ['model_pos_x_final', 'model_pos_y_final', 'theta_final']

    start = 2

    gnd_truth = testrun.iloc[:start].loc[:, ['model_pos_x_start', 'model_pos_y_start', 'theta_start']].values
    initial_state = gnd_truth[0]

    testrun_X = testrun.iloc[:start].loc[:, x_sel].values

    testrun_y = testrun.iloc[:start].loc[:, y_sel].values

    test_traj = np.concatenate((initial_state.reshape(1, -1), testrun_y), axis=0)

    pdb.set_trace()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    plot_debug = True

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    for it in range(iterations):
        print("iteration %d" % it)
        model.fit(X_train.reshape(len(X_train), 1, -1), y_train.reshape(len(y_train), 1, -1), epochs=epochs, verbose=1, shuffle=True)
        # calculate rmse for train data
        train_se = []

        train_diff = 0
        train_predictions = model.predict(X_train.reshape(len(X_train), 1, -1)).reshape(y_train.shape)

        for i in range(len(X_train)):
            # collect xy difference
            train_diff += np.sum((train_predictions - y_train)**2)
            # collect theta difference
            train_diff += np.sum(angle_dist(np.concatenate((train_predictions[:, 2].reshape(-1, 1), y_train[:, 2].reshape(-1, 1)), axis=1))**2)

        train_rmse = np.sqrt(train_diff/len(X_train))
        train_loss_history.append(train_rmse)
        
        test_se = []


        test_diff = 0
        test_predictions = model.predict(X_test.reshape(len(X_test), 1, -1)).reshape(y_test.shape)

        for i in range(len(X_test)):
            # collect xy difference
            test_diff += np.sum((test_predictions - y_test)**2)
            # collect theta difference
            test_diff += np.sum(angle_dist(np.concatenate((test_predictions[:, 2].reshape(-1, 1), y_test[:, 2].reshape(-1, 1)), axis=1))**2)

        test_rmse = np.sqrt(test_diff/len(X_test))
        test_loss_history.append(test_rmse)

        print('train rmse: %f' % train_rmse)
        print('test rmse: %f' % test_rmse)

        if plot_debug:
            stateful_model = convert_to_inference_model(model)
            
            predictions = predict_seq(stateful_model, testrun_X, initial_state, output_scaler=output_scaler, gnd_truth=testrun_y)
            plot_trajectories(predictions, test_traj, ax1)

    print('Saving trained model...')
    save_model(model, dirpath, model_fname)

    fig = plt.figure()
    ep_range = range(0, iterations)
    plt.plot(ep_range, train_loss_history)
    plt.plot(ep_range, test_loss_history)
    plt.title('theta miso prediction (RNN)')
    plt.xlabel('iteration (30 epochs/iteration)')
    plt.ylabel('RMSE')
    plt.show()

    
