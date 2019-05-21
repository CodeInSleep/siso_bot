import os
import math
import pandas as pd
import numpy as np
from numpy import cos, sin, arctan2
import pdb
import matplotlib.pyplot as plt
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from visualize import visualize_3D
from utils import trim_to_mult_of, save_obj, load_obj

input_fields = ['left_pwm', 'right_pwm']
output_fields = ['model_pos_x', 'model_pos_y']
others = ['sim_time']
network_settings = {
    'p': len(input_fields)+2,
    'J': len(output_fields)+3,
    'batch_size': 32,
    'timestep': 5
}

np.random.seed(7)
def truncate(num, digits):
    stepper = pow(10.0, digits)
    return math.trunc(num*stepper)/stepper

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
    return Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob

def rotate(xy_df, rad):
    # rotational matrix
    rotation = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
    return pd.DataFrame(np.matmul(rotation, xy_df.T).T)

def encode_angle(df, theta_field):
    df.loc[:, theta_field+'_cos'] = df.loc[:, theta_field].apply(lambda x: cos(x))
    df.loc[:, theta_field+'_sin'] = df.loc[:, theta_field].apply(lambda x: sin(x))
 
def remove_bias(df, fields, start_states):
    df.loc[:, fields] =  df.loc[:, fields] - start_states.loc[df.name, fields]
    return df

def remove_bias_in_batches(df, batch_size):
    # assign batch numbers to group by
    df.loc[:, 'batch_no'] = df.index//batch_size

    grouped = df.groupby('batch_no')
    start_of_batches = grouped.first()
    # remove x, y, thetas bias of each batch
    df.loc[:, output_fields] = grouped.apply(lambda x: remove_bias(x, output_fields, start_of_batches))
    # rotate so that each batch starts at theta=0

    # time difference on the output fields
    df.loc[:, output_fields+['sim_time']] = df.groupby('batch_no').apply(
            lambda x: x.loc[:, output_fields+['sim_time']].diff().fillna(0))
    return df, start_of_batches

def train_test_split_to_batches(train_data, test_data, X_sel, y_sel, dims):
    batch_size = dims['batch_size']
    input_dim = dims['p']
    output_dim = dims['J']
    n_train_batch_len = dims['n_train_batch_len']
    n_test_batch_len = dims['n_test_batch_len']
    X_train = train_data.loc[:, X_sel].values.reshape(
            batch_size, n_train_batch_len, input_dim)
    X_test = test_data.loc[:, X_sel].values.reshape(
            batch_size, n_test_batch_len, input_dim)
    y_train = train_data.loc[:, y_sel].values.reshape(
            batch_size, n_train_batch_len, J)
    y_test = test_data.loc[:, y_sel].values.reshape(
            batch_size, n_test_batch_len, J)

    return (X_train, X_test, y_train, y_test)

def trim_to_batch_size(df, batch_size):
    return df.iloc[:-(len(df)%batch_size), :]

def calc_batch_size(df, batch_size):
    n_train = int(0.7*len(df))
    n_test = len(df) - n_train
    n_train_batch_len = int(n_train/batch_size)
    n_test_batch_len = int(n_test/batch_size)
    return (n_train, n_test, n_train_batch_len, n_test_batch_len)

def label_trials(df):
    trial_counts = {}

    # TODO: add right transitions
    left_transitions = df.loc[:, 'left_pwm'].diff().to_numpy().nonzero()[0]
    right_transitions = df.loc[:, 'right_pwm'].diff().to_numpy().nonzero()[0]
    transitions = np.union1d(left_transitions, right_transitions)

    trial_intervals = []
    prev_t = 0
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
    current_trial_name = df.iloc[end+1].loc['input']
    if current_trial_name in trial_counts:
        trial_counts[current_trial_name] += 1
    else:
        trial_counts[current_trial_name] = 1
    trial_idx = trial_counts[current_trial_name]
    df.loc[end:, 'input'] = current_trial_name + '_' + str(trial_idx)
    return df

def transform_group(group_df, max_duration, output_fields):
    """
        transform each group so that each trial have length of max_duration
    """
    print(group_df.name)
    #group_df = group_df.reset_index().drop('input', axis=1)
    cols = group_df.columns
    group_df.reset_index()

    if max_duration - group_df.shape[0] != 0:
        padding = pd.DataFrame(np.zeros((max_duration-group_df.shape[0], group_df.shape[1]), dtype=int))
        padding.columns = cols
        
        padding.loc[:, output_fields] = np.repeat(group_df.iloc[group_df.shape[0]-1].loc[output_fields].values.reshape(-1, 1), len(padding), axis=1).T
    
        # pad the time series with
        padded_group_df = pd.DataFrame(pd.np.row_stack([group_df, padding]))
        padded_group_df.columns = cols
        padded_group_df = padded_group_df.fillna(0)
    else:
        padded_group_df = group_df

    #return padded_group_df
    return padded_group_df

def diff(df, fields):
    df.loc[:, fields] = df.loc[:, fields].diff().fillna(0)
    return df

def transform(df, layers_dims, dirpath, cached=False):
    X_train_fname = os.path.join(dirpath, 'X_train.pkl')
    X_test_fname = os.path.join(dirpath, 'X_test.pkl')
    theta_y_train_fname = os.path.join(dirpath, 'theta_y_train.pkl')
    theta_y_test_fname = os.path.join(dirpath, 'theta_y_test.pkl')
    pos_y_train_fname = os.path.join(dirpath, 'pos_y_train.pkl')
    pos_y_test_fname = os.path.join(dirpath, 'pos_y_test.pkl')
    train_traj_fname = os.path.join(dirpath, 'traj_train.pkl')
    test_traj_fname = os.path.join(dirpath, 'traj_test.pkl')
    if not cached:
        df.loc[:, 'theta'] = df.loc[:, 'theta'].apply(lambda x: math.radians(x))

        df.loc[:, 'left_pwm'] = df.loc[:, 'left_pwm'].apply(truncate, args=(3,))
        df.loc[:, 'right_pwm'] = df.loc[:, 'right_pwm'].apply(truncate, args=(3,))

        df.loc[:, 'input'] = 'l_'+df.loc[:, 'left_pwm'].map(str)+'_r_'+df.loc[:, 'right_pwm'].map(str)

        # normalize inputs
        input_scaler = MinMaxScaler(feature_range=(0, 1))
        df.loc[:,input_fields] = input_scaler.fit_transform(df.loc[:,input_fields])

        theta_data = df.loc[:, ['input', 'sim_time']+input_fields+output_fields+['theta']]
        theta_data.loc[:, 'theta(t-1)'] = theta_data.loc[:, 'theta'].shift(1)

        theta_data = label_trials(theta_data)

        # group by trial name and turn trials into batches
        grouped = theta_data.groupby('input')
        start_of_batches = grouped.first()
        num_trials = len(grouped)
        # store max duration of a trial
        max_duration = max(grouped['sim_time'].count())
        # pad sequences up to max duration length
        theta_data = theta_data.groupby('input').apply(lambda x: transform_group(
            x, max_duration, output_fields+['sim_time', 'theta(t-1)', 'theta']))
        theta_data = theta_data.drop('input', axis=1)
        encode_angle(theta_data, 'theta(t-1)')
        encode_angle(theta_data, 'theta')

        target_angles = theta_data.loc[:, 'theta']
        target_pos = theta_data.loc[:, output_fields]

        theta_data = theta_data.groupby('input').apply(lambda x:
            diff(x, ['sim_time', 'model_pos_x', 'model_pos_y']))

        # save ground truth trajectory
        traj_data = pd.concat((target_pos, target_angles), axis=1).copy()        
        # theta_data = theta_data.iloc[1:]
        #plot_target_angles(np.expand_dims(theta_data.loc[:, 'theta'].values.reshape(-1, 1), axis=0))
        
        # theta_data = theta_data.reindex(columns=fields)

        # to debug
        # theta_data = theta_data.iloc[:1000]
        # pdb.set_trace()

        n_train = int(num_trials*0.7)*max_duration
        train_data = theta_data.iloc[:n_train]
        test_data = theta_data.iloc[n_train:]
        train_traj_data = traj_data.iloc[:n_train]
        test_traj_data = traj_data.iloc[n_train:]

        pdb.set_trace()
        train_traj_data = train_traj_data.groupby('input').apply(lambda x: remove_bias(x, output_fields, start_of_batches))
        test_traj_data = test_traj_data.groupby('input').apply(lambda x: remove_bias(x, output_fields, start_of_batches))
        pdb.set_trace()
        X_sel = ['sim_time', 'left_pwm', 'right_pwm', 'theta(t-1)_cos', 'theta(t-1)_sin']
        y_sel = ['theta_cos', 'theta_sin', 'model_pos_x', 'model_pos_y']
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
        train_traj_data = trim_to_mult_of(train_traj_data, max_duration)
        test_traj_data = trim_to_mult_of(test_traj_data, max_duration)

        X_train.to_pickle(X_train_fname)
        X_test.to_pickle(X_test_fname)
        theta_y_train = y_train.loc[:, y_sel[:2]]
        theta_y_train.to_pickle(theta_y_train_fname)
        pos_y_train = y_train.loc[:, y_sel[2:]]
        pos_y_train.to_pickle(pos_y_train_fname)
        theta_y_test = y_test.loc[:, y_sel[:2]]
        theta_y_test.to_pickle(theta_y_test_fname)
        pos_y_test = y_test.loc[:, y_sel[2:]]
        pos_y_test.to_pickle(pos_y_test_fname)
        train_traj_data.to_pickle(train_traj_fname)
        test_traj_data.to_pickle(test_traj_fname)

        # pdb.set_trace()
        print('number of trials in train: %d' % (len(X_train)/max_duration))
        print('number of trials in test: %d' % (len(X_test)/max_duration))
        
        data_info = {
            'timestep': max_duration,
            'num_trials': num_trials,
            'num_train_trials': int(len(X_train)/max_duration),
            'num_test_trials': int(len(X_test)/max_duration)
        }
        save_obj(data_info, dirpath, 'data_info')
    else:
        X_train = pd.read_pickle(X_train_fname)
        X_test = pd.read_pickle(X_test_fname)
        theta_y_train = pd.read_pickle(theta_y_train_fname)
        pos_y_train = pd.read_pickle(pos_y_train_fname)
        theta_y_test = pd.read_pickle(theta_y_test_fname)
        pos_y_test = pd.read_pickle(pos_y_test_fname)

        data_info = load_obj(dirpath, 'data_info')
    
    return (X_train, X_test, theta_y_train, pos_y_train, theta_y_test, pos_y_test, data_info['timestep'])
