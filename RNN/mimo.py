import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
import pdb
from numpy import cos, sin, arctan2
import matplotlib.pyplot as plt

from transform import transform
from visualize import visualize_3D
from utils import decode_angles, plot_target_angles, save_obj, make_model, save_model

input_fields = ['left_pwm', 'right_pwm']

layers_dims = [7, 10, 20, 4]
fname = 'trial_1000_0_to_3.csv'
model_fname = 'model'

####
# Issues:
#   - For batch_size = 32, the size of each 
fields = ['input', 'sim_time', 'left_pwm', 'right_pwm',
        'theta_cos', 'theta_sin']
###

def encode_angle(df, theta_field):
    df.loc[:, theta_field+'_cos'] = df.loc[:, theta_field].apply(lambda x: cos(x))
    df.loc[:, theta_field+'_sin'] = df.loc[:, theta_field].apply(lambda x: sin(x))

def angle_dist(angles):
    ang1 = math.degrees(angles[0])
    ang2 = math.degrees(angles[1])
    
    a = ang1 - ang2
    return np.radians(np.abs((a+180)%360-180))**2

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
    X_train, X_test, y_train, y_test, timestep = transform(df, layers_dims, dirpath, cached=True)
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
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_title('train example')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2.set_title('test example')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    # decoded_y_train = decode_angles(y_train)
    # y_test = decode_angles(y_test)
    model_cached = False
    if model_cached:
        model = load_model(dirpath, model_fname)

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

        n_train = 10
        n_test = 10
        
        ax1.cla()
        ax2.cla()
        _pred_train = decoded_train_predictions[n_train]
        _y_train = decoded_train_gnd[n_train]
        _pred_test = decoded_test_predictions[n_test]
        _y_test = decoded_test_gnd[n_test]

        visualize_3D(np.expand_dims(_pred_train, axis=0), ax1, plt_arrow=True)
        visualize_3D(np.expand_dims(_y_train, axis=0), ax1, plt_arrow=True)
        ax1.legend(['predicted', 'ground truth'])
        ax1.grid(True)

        visualize_3D(np.expand_dims(_pred_test, axis=0), ax2, plt_arrow=True)
        visualize_3D(np.expand_dims(_y_test, axis=0), ax2, plt_arrow=True)
        ax2.legend(['predicted', 'ground truth'])
        ax2.grid(True)
        plt.pause(5)

        print('train rmse on iteration %d: %f' % (it, train_rmse))
        print('test rmse on iteration %d: %f' % (it, test_rmse))

    
    save_model(model, dirpath, model_fname)

    fig = plt.figure()
    ep_range = range(0, iterations)
    plt.plot(ep_range, train_loss_history)
    plt.plot(ep_range, test_loss_history)
    plt.title('theta miso prediction (RNN)')
    plt.xlabel('epoch')
    plt.ylabel('RMSE theta(rad)')
    plt.show()

    
