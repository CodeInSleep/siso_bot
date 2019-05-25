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
from sklearn.externals import joblib 
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

def scale(df, fields, factor):
    df.loc[:, fields] = df.loc[:, fields]*factor
    return df
# def remove_bias_in_batches(df, batch_size):
#     # assign batch numbers to group by
#     df.loc[:, 'batch_no'] = df.index//batch_size

#     grouped = df.groupby('batch_no')
#     start_of_batches = grouped.first()
#     # remove x, y, thetas bias of each batch
#     df.loc[:, output_fields] = grouped.apply(lambda x: remove_bias(x, output_fields, start_of_batches))
#     # rotate so that each batch starts at theta=0

#     # time difference on the output fields
#     df.loc[:, output_fields+['sim_time']] = df.groupby('batch_no').apply(
#             lambda x: x.loc[:, output_fields+['sim_time']].diff().fillna(0))
#     return df, start_of_batches

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

def extend_group(group_df, max_duration, extend_fields):
    """
        extend each group so that each trial have length of max_duration
    """
    group_df = group_df.reset_index()
    cols = group_df.columns

    if max_duration - group_df.shape[0] != 0:
        padding = pd.DataFrame(np.zeros((max_duration-group_df.shape[0], group_df.shape[1]), dtype=int))
        padding.columns = cols
        
        padding.loc[:, extend_fields] = np.repeat(group_df.iloc[group_df.shape[0]-1].loc[extend_fields].values.reshape(-1, 1), len(padding), axis=1).T
    
        # pad the time series with
        padded_group_df = pd.DataFrame(pd.np.row_stack([group_df, padding]))
        padded_group_df.columns = cols
        padded_group_df = padded_group_df.fillna(0)
    else:
        padded_group_df = group_df
    padded_group_df = padded_group_df.drop(['input', 'timestep'], axis=1)

    #return padded_group_df
    return padded_group_df

def diff(df, fields):
    df.loc[:, fields] = df.loc[:, fields].diff().fillna(0)
    return df

def downsample(df, rate='0.1S', start_of_batches=None):
    # print(df.name)
    # if df.name == 'l_2.9_r_2.9_2':
    if start_of_batches is not None:
        df = remove_bias(df, 'sim_time', start_of_batches)
    df.loc[:, 'sim_time_del'] = pd.to_timedelta(df.loc[:, 'sim_time'].values, unit='s')
    df = df.set_index('sim_time_del')

    df = df.resample(rate).mean().ffill()
    df = df.reset_index()

    return df

def transform(df, layers_dims, dirpath, cached=False):
    X_train_fname = os.path.join(dirpath, 'X_train.pkl')
    X_test_fname = os.path.join(dirpath, 'X_test.pkl')
    y_train_fname = os.path.join(dirpath, 'y_train.pkl')
    y_test_fname = os.path.join(dirpath, 'y_test.pkl')
    
    if not cached:
        df.loc[:, 'theta'] = df.loc[:, 'theta'].apply(lambda x: math.radians(x))

        df.loc[:, 'left_pwm'] = df.loc[:, 'left_pwm'].apply(truncate, args=(3,))
        df.loc[:, 'right_pwm'] = df.loc[:, 'right_pwm'].apply(truncate, args=(3,))

        # make xy in mm
        df = scale(df, ['model_pos_x', 'model_pos_x'], 1000)
        df.loc[:, 'input'] = 'l_'+df.loc[:, 'left_pwm'].map(str)+'_r_'+df.loc[:, 'right_pwm'].map(str)

        print('Normalizing Inputs...')
        # normalize inputs
        input_scaler = MinMaxScaler(feature_range=(0, 1))
        df.loc[:,input_fields] = input_scaler.fit_transform(df.loc[:,input_fields])

        theta_data = df.loc[:, ['input', 'sim_time']+input_fields+output_fields+['theta']]
        theta_data.loc[:, 'theta(t-1)'] = theta_data.loc[:, 'theta'].shift(1)
        # theta_data.loc[:, 'model_pos_x(t-1)'] = theta_data.loc[:, 'model_pos_x'].shift(1)
        # theta_data.loc[:, 'model_pos_y(t-1)'] = theta_data.loc[:, 'model_pos_y'].shift(1)

        print('Labeling Trials...')
        theta_data = label_trials(theta_data)
   
        # group by trial name, record first entry of every batch
        grouped = theta_data.groupby('input')
        start_of_batches = grouped.first()
        num_trials = len(grouped)

        print('Downsampling and diffing sim_time...')
        # downsample
        theta_data = theta_data.groupby('input').apply(lambda x: downsample(x, rate='0.2S', start_of_batches=start_of_batches))
        theta_data = theta_data.rename_axis(['input', 'timestep'])
        theta_data = theta_data.drop('sim_time_del', axis=1)
        # drop the stationary trial
        # theta_data = theta_data.drop('l_90.0_r_90.0_1', axis=0)
        # store max duration of a trial
        max_duration = max(theta_data.groupby(['input']).size())
        # take difference in time
        theta_data = theta_data.groupby(['input']).apply(lambda x:
            diff(x, ['sim_time']))
        start_states = theta_data.groupby('input').first()
    
        print('Removing Biases and Difference position...')
        # remove bias the output_fields bias of each batch
        theta_data = theta_data.groupby('input').apply(lambda x: remove_bias(x, output_fields, start_states))
        theta_data.loc[:, output_fields] = theta_data.groupby('input').apply(lambda x: x.loc[:, output_fields].diff().fillna(0))

        print('Extending groups to max len and encoding angle...')
        theta_data = theta_data.groupby(['input']).apply(lambda x: extend_group(x, max_duration,
            ['theta(t-1)', 'theta']))
        encode_angle(theta_data, 'theta(t-1)')
        encode_angle(theta_data, 'theta')
        
        n_train = int(num_trials*0.7)
        trial_names = start_of_batches.index.to_list()
        train_samples = np.random.choice(num_trials, n_train, replace=False)
        train_trial_names = [trial_names[i] for i in train_samples]
        test_samples = np.array([i for i in range(num_trials) if i not in train_samples])
        test_trial_names = [trial_names[i] for i in test_samples]
        train_data = theta_data.loc[train_trial_names]
        test_data = theta_data.loc[test_trial_names]
        # train_traj_data = traj_data.iloc[:n_train]
        # test_traj_data = traj_data.iloc[n_train:]
        X_sel = ['sim_time', 'left_pwm', 'right_pwm', 'theta(t-1)_cos', 'theta(t-1)_sin']
        y_sel = ['model_pos_x', 'model_pos_y', 'theta_cos', 'theta_sin']
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

        X_train.to_pickle(X_train_fname)
        X_test.to_pickle(X_test_fname)
        y_train.to_pickle(y_train_fname)
        y_test.to_pickle(y_test_fname)
        joblib.dump(input_scaler, os.path.join(dirpath, 'input_scaler.pkl'))

        print('number of trials in train: %d' % (len(X_train)/max_duration))
        print('number of trials in test: %d' % (len(X_test)/max_duration))
        print('max_duration: %d' % max_duration)
        data_info = {
            'timestep': max_duration,
            'num_trials': num_trials,
            'num_train_trials': int(len(X_train)/max_duration),
            'num_test_trials': int(len(X_test)/max_duration)
        }
        save_obj(data_info, dirpath, 'data_info')
        save_obj(train_trial_names, dirpath, 'train_trial_names')
        save_obj(test_trial_names, dirpath, 'test_trial_names')

    else:
        X_train = pd.read_pickle(X_train_fname)
        X_test = pd.read_pickle(X_test_fname)
        y_train = pd.read_pickle(y_train_fname)
        y_test = pd.read_pickle(y_test_fname)

        data_info = load_obj(dirpath, 'data_info')
    
    return (X_train, X_test, y_train, y_test, data_info['timestep'])
