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
from utils import decode_angles, plot_target_angles, save_obj, make_model, save_model, angle_dist

input_fields = ['left_pwm', 'right_pwm']

layers_dims = [5, 10, 20, 2]
fname = 'trial_1000_0_to_3.csv'

####
# Issues:
#   - For batch_size = 32, the size of each 
fields = ['input', 'sim_time', 'left_pwm', 'right_pwm',
        'theta_cos', 'theta_sin']
###

def encode_angle(df, theta_field):
    df.loc[:, theta_field+'_cos'] = df.loc[:, theta_field].apply(lambda x: cos(x))
    df.loc[:, theta_field+'_sin'] = df.loc[:, theta_field].apply(lambda x: sin(x))
    
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
    X_train, X_test, theta_y_train, _, theta_y_test, _, timestep = transform(df, layers_dims, dirpath, cached=False)
    X_train = X_train.values.reshape(
        -1, timestep, p)
    X_test = X_test.values.reshape(
            -1, timestep, p)
    theta_y_train = theta_y_train.values.reshape(
            -1, timestep, J)
    theta_y_test = theta_y_test.values.reshape(
                -1, timestep, J)

    num_batches = 16
    theta_model = make_model(num_batches, timestep, layers_dims)
   
    iterations = 10
    epochs = 30
    # learning curver
    train_loss_history = []
    test_loss_history = []
    
    # decoded_y_train = decode_angles(y_train)
    # y_test = decode_angles(y_test)
    for it in range(iterations):
        print("iteration %d" % it)
        theta_model.fit(X_train, theta_y_train, epochs=epochs, batch_size=num_batches, verbose=1, shuffle=False)
        # for j in range(int(train_bl/time_step)-1):
        #     time_seg = range((j*time_step), ((j+1)*time_step))
        #     model.fit(X_train[:, time_seg, :], y_train[:, time_seg, :], 
        #             batch_size=num_batches, verbose=1, shuffle=False)
        # model.reset_states()

        # calculate rmse for train data
        train_se = []
        # for j in range(int(train_bl/time_step)-1):
        #     time_seg = range((j*time_step), ((j+1)*time_step))
        #     pred = model.predict(X_train[:, time_seg, :])

        #     train_se_over_timestep = []
        #     for k in range(num_batches):
        #         _pred = decode_angles(pred[k])
        #         _gnd_truth = decode_angles(y_train[k, time_seg, :])
                
        #         diff = np.apply_along_axis(angle_dist, 1,
        #                 np.concatenate((_pred, _gnd_truth), axis=1))
        #         train_se_over_timestep.append(diff**2)
        #     train_se.append(train_se_over_timestep)
        # model.reset_states()
        train_predictions = theta_model.predict(X_train)
        for i in range(len(train_predictions)):
            pred = decode_angles(train_predictions[i])
            gnd_truth = decode_angles(theta_y_train[i])

            diff = np.apply_along_axis(angle_dist, 1,
                np.concatenate((pred, gnd_truth), axis=1))
            train_se.append(diff**2)

        train_rmse = np.sqrt(np.mean(np.array(train_se)))
        train_loss_history.append(train_rmse)
        
        test_se = []
        # for j in range(int(test_bl/time_step)-1):
        #     time_seg = range((j*time_step), ((j+1)*time_step))
        #     pred = model.predict(X_test[:, time_seg, :])
            
        #     test_se_over_timestep = []
        #     for k in range(num_batches):
        #         _pred = decode_angles(pred[k])
        #         _gnd_truth = decode_angles(y_test[k, time_seg, :])
        #         diff = np.apply_along_axis(angle_dist, 1,
        #                 np.concatenate((_pred, _gnd_truth),
        #                     axis=1))
        #         test_se_over_timestep.append(diff**2)
        #     test_se.append(test_se_over_timestep)
        # model.reset_states()
        test_predictions = theta_model.predict(X_test)
        for i in range(len(X_test)):
            pred = decode_angles(test_predictions[i])
            gnd_truth = decode_angles(theta_y_test[i])

            diff = np.apply_along_axis(angle_dist, 1,
                np.concatenate((pred, gnd_truth), axis=1))
            test_se.append(diff**2)

            plt.cla()
            plt.plot(pred)
            plt.plot(gnd_truth)
            plt.legend(['pred', 'ground truth'])
            plt.grid(True)
            plt.pause(0.000001)

        test_rmse = np.sqrt(np.mean(np.array(test_se)))
        test_loss_history.append(test_rmse)

        print('train rmse on iteration %d: %f' % (it, train_rmse))
        print('test rmse on iteration %d: %f' % (it, test_rmse))

    model_fname = 'theta_model'
    save_model(theta_model, dirpath, model_fname)

    error_plot = plt.figure()
    ep_range = range(0, iterations)
    plt.plot(ep_range, train_loss_history)
    plt.plot(ep_range, test_loss_history)
    plt.title('theta miso prediction (RNN)')
    plt.xlabel('epoch')
    plt.ylabel('RMSE theta(rad)')
    error_plot.show()

    
