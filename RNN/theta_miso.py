import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
from numpy import cos, sin, arctan2
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, model_from_json
from keras.initializers import Identity, RandomNormal
from keras.layers import Dense, Dropout, LSTM, SimpleRNN, Dropout, GRU
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from transform import truncate, difference
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from visualize import visualize_3D
import pdb
input_fields = ['left_pwm', 'right_pwm']

layers_dims = [5, 10, 20, 2]
fname = 'trial_1000.csv'

####
# Issues:
#   - For batch_size = 32, the size of each 
fields = ['input', 'sim_time', 'left_pwm', 'right_pwm',
        'theta_cos', 'theta_sin']
###

max_duration = 259

def encode_angle(df, theta_field):
    df.loc[:, theta_field+'_cos'] = df.loc[:, theta_field].apply(lambda x: cos(x))
    df.loc[:, theta_field+'_sin'] = df.loc[:, theta_field].apply(lambda x: sin(x))

def decode_angles(cos_sin_vals):
    return arctan2(cos_sin_vals[:, 1], cos_sin_vals[:, 0]).reshape(-1, 1)

def trim_to_mult_of(df, size):
    if len(df)%size == 0:
        return df
    return df.iloc[:-(len(df)%size), :]

def label_trials(df):
    trial_counts = {}

    transitions = df.loc[:, 'left_pwm'].diff().nonzero()[0]
    trial_intervals = []
    prev_t = transitions[0]
    for t in transitions[1:]:
        trial_intervals.append((prev_t, t))
        prev_t = t
    counta = 0
    for start, end in trial_intervals:
        current_trial_name = df.iloc[start+1].loc['input']
        # no change in trial
        if current_trial_name in trial_counts:
            trial_counts[current_trial_name] += 1
        else:
            trial_counts[current_trial_name] = 1
        trial_idx = trial_counts[current_trial_name]
        df.loc[start:end, 'input'] = current_trial_name + '_' + str(trial_idx)
    
    return df

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

def transform_group(group_df, max_duration, output_fields):
    """
        transform each group so that each trial have length of max_duration
    """
    print(group_df.name)
    #group_df = group_df.reset_index().drop('input', axis=1)
    cols = group_df.columns

    if max_duration - group_df.shape[0] != 0:
        padding = pd.DataFrame(np.zeros((max_duration-group_df.shape[0], group_df.shape[1]), dtype=int))
        padding.columns = cols
        
        padding.loc[:, output_fields] = np.repeat(np.array(group_df.iloc[group_df.shape[0]-1].loc[output_fields]),len(padding)).reshape(len(padding), -1)
    
        # pad the time series with
        padded_group_df = pd.DataFrame(pd.np.row_stack([group_df, padding]))
        padded_group_df.columns = cols
        padded_group_df = padded_group_df.fillna(0)
    else:
        padded_group_df = group_df
    #return padded_group_df
    return padded_group_df

def transform(df, cached=False):
    if not cached:
        df.loc[:, 'theta'] = df.loc[:, 'theta'].apply(lambda x: math.radians(x))

        df.loc[:, 'left_pwm'] = df.loc[:, 'left_pwm'].apply(truncate, args=(3,))
        df.loc[:, 'right_pwm'] = df.loc[:, 'right_pwm'].apply(truncate, args=(3,))

        df.loc[:, 'input'] = 'l_'+df.loc[:, 'left_pwm'].map(str)+'_r_'+df.loc[:, 'right_pwm'].map(str)

        # normalize inputs
        input_scaler = MinMaxScaler(feature_range=(0, 1))
        df.loc[:,input_fields] = input_scaler.fit_transform(df.loc[:,input_fields])

        theta_data = df.loc[:, ['input', 'sim_time']+input_fields+['theta']]
        theta_data.loc[:, 'theta(t-1)'] = theta_data.loc[:, 'theta'].shift(1)
        theta_data.loc[:, 'sim_time'] = theta_data.loc[:, 'sim_time'].diff()
        theta_data = theta_data.iloc[1:]
        #plot_target_angles(np.expand_dims(theta_data.loc[:, 'theta'].values.reshape(-1, 1), axis=0))
        encode_angle(theta_data, 'theta(t-1)')
        encode_angle(theta_data, 'theta')
        
        theta_data = theta_data.drop(['theta(t-1)', 'theta'], axis=1)
        # theta_data = theta_data.reindex(columns=fields)

        # to debug
        # theta_data = theta_data.iloc[:1000]

        theta_data = label_trials(theta_data)

        # group by trial name and turn trials into batches
        grouped = theta_data.groupby('input')
        start_of_batches = grouped.first()
        num_trials = len(grouped)
        # store max duration of a trial
        max_duration = max(grouped['sim_time'].count())

        theta_data = theta_data.groupby('input').apply(lambda x: transform_group(
            x, max_duration, ['theta(t-1)_cos', 'theta(t-1)_sin', 'theta_cos', 'theta_sin']))

        n_train = int(num_trials*0.7)*max_duration
        train_data = theta_data[:n_train]
        test_data = theta_data[n_train:]

        X_sel = ['sim_time', 'left_pwm', 'right_pwm', 'theta(t-1)_cos', 'theta(t-1)_sin']
        y_sel = ['theta_cos', 'theta_sin']
        X_train = train_data.loc[:, X_sel]
        y_train = train_data.loc[:, y_sel]
        X_test = test_data.loc[:, X_sel]
        y_test = test_data.loc[:, y_sel]

        p = layers_dims[0]
        J = layers_dims[-1]

        X_train = trim_to_mult_of(X_train, max_duration)
        X_test = trim_to_mult_of(X_test, max_duration)
        y_train = trim_to_mult_of(y_train, max_duration)
        y_test = trim_to_mult_of(y_test, max_duration)

        X_train.to_pickle('X_train.pkl')
        X_test.to_pickle('X_test.pkl')
        y_train.to_pickle('y_train.pkl')
        y_test.to_pickle('y_test.pkl')
        print('number of trials in train: %d' % (len(X_train)/max_duration))
        print('number of trials in test: %d' % (len(X_test)/max_duration))

    else:
        X_train = pd.read_pickle('X_train.pkl')
        X_test = pd.read_pickle('X_test.pkl')
        y_train = pd.read_pickle('y_train.pkl')
        y_test = pd.read_pickle('y_test.pkl')

    p = layers_dims[0]
    J = layers_dims[-1]
    
    max_duration = 259
    X_train = X_train.values.reshape(
            -1, max_duration, p)
    X_test = X_test.values.reshape(
            -1, max_duration, p)
    y_train = y_train.values.reshape(
            -1, max_duration, J)
    y_test = y_test.values.reshape(
            -1, max_duration, J)

    # for debugging
    # train_bl = 50
    # test_bl = 50
    # X_train = X_train[:, :train_bl, :]
    # X_test = X_test[:, :test_bl, :]
    # y_train = y_train[:, :train_bl, :]
    # y_test = y_test[:, :test_bl, :]

    return (X_train, X_test, y_train, y_test)

def make_model(num_batches, time_step):
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

def angle_dist(angles):
    ang1 = math.degrees(angles[0])
    ang2 = math.degrees(angles[1])
    
    a = ang1 - ang2
    return np.radians(np.abs((a+180)%360-180))
    
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

    X_train, X_test, y_train, y_test = transform(df, cached=True)

    num_batches = 16
    time_step = max_duration
    model = make_model(num_batches, time_step)
   

    iterations = 5
    epochs = 30
    # learning curver
    train_loss_history = []
    test_loss_history = []
    
    # decoded_y_train = decode_angles(y_train)
    # y_test = decode_angles(y_test)
    for it in range(iterations):
        print("iteration %d" % it)
        model.fit(X_train, y_train, epochs=epochs, batch_size=num_batches, verbose=1, shuffle=False)
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
        train_predictions = model.predict(X_train)
        for i in range(len(train_predictions)):
            pred = decode_angles(train_predictions[i])
            gnd_truth = decode_angles(y_train[i])

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
        test_predictions = model.predict(X_test)
        for i in range(len(X_test)):
            pred = decode_angles(test_predictions[i])
            gnd_truth = decode_angles(y_test[i])

            diff = np.apply_along_axis(angle_dist, 1,
                np.concatenate((pred, gnd_truth), axis=1))
            test_se.append(diff**2)

        test_rmse = np.sqrt(np.mean(np.array(test_se)))
        test_loss_history.append(test_rmse)

        print('train rmse on iteration %d: %f' % (it, train_rmse))
        print('test rmse on iteration %d: %f' % (it, test_rmse))

    ep_range = range(0, iterations)
    plt.plot(ep_range, train_loss_history)
    plt.plot(ep_range, test_loss_history)
    plt.title('theta miso prediction (RNN)')
    plt.xlabel('epoch')
    plt.ylabel('RMSE theta(rad)')
    plt.show()

    fname = 'theta_model'
    model_json = model.to_json()
    with open(fname+'.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(fname+'.h5')

