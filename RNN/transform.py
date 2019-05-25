import math
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
from numpy import cos, sin, arctan2
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from visualize import visualize_3D

input_fields = ['left_pwm', 'right_pwm']
output_fields_decoded = ['model_pos_x', 'model_pos_y', 'theta']
output_fields_encoded = ['model_pos_x', 'model_pos_y', 'theta_cos', 'theta_sin']
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
    return df.loc[:, fields] - start_states.loc[df.name, fields]

def remove_bias_in_batches(df, batch_size):
    # assign batch numbers to group by
    df.loc[:, 'batch_no'] = df.index//batch_size

    pdb.set_trace()
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

def transform(df, train_percentage=0.7, count=-1):
    timestep = 5
    df.loc[:, 'theta'] = df.loc[:, 'theta'].apply(lambda x: math.radians(x))
    df.loc[:, 'left_pwm'] = df.loc[:, 'left_pwm'].apply(truncate, args=(3,))
    df.loc[:, 'right_pwm'] = df.loc[:, 'right_pwm'].apply(truncate, args=(3,))
    df.loc[:, 'input'] = 'l_'+df.loc[:, 'left_pwm'].map(str)+'_r_'+df.loc[:, 'right_pwm'].map(str)

    # normalize inputs
    input_scaler = MinMaxScaler(feature_range=(0, 1))
    df.loc[:,input_fields] = input_scaler.fit_transform(df.loc[:,input_fields])

    # provide a summary of inputs
    # input_summary = df.groupby('input').apply(lambda x: x.describe())

    batch_size = network_settings['batch_size']
    timestep = network_settings['timestep']
    p = network_settings['p']
    J = network_settings['J']
    # trim data set to a multiple of batch size
    # df = df.iloc[:-(len(df)%batch_size), :]
    n_train, n_test, n_train_batch_len, n_test_batch_len = calc_batch_size(df, batch_size)
    network_settings['n_train_batch_len'] = n_train_batch_len
    network_settings['n_test_batch_len'] = n_test_batch_len
    df.loc[:, 'theta(t-1)'] = difference(df.loc[:, 'theta'])
    # get rid of first entry without previous angle
    df = df.iloc[1:]

    train_data = df.iloc[:n_train, :]
    test_data = df.iloc[n_train:, :]

    '''
    debug = False
    if debug:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        visualize_3D(np.expand_dims(train_data[:100].loc[:, output_fields_decoded].values, axis=0), ax1, plt_arrow=True)
        fig.show()
    '''
    # remove bias of starting point in 
    train_data, train_start_states = remove_bias_in_batches(train_data, batch_size)
    test_data, test_start_states = remove_bias_in_batches(test_data, batch_size)

    # make sure train and test data are multiple of batch_size
    train_data = trim_to_batch_size(train_data, batch_size)
    test_data = trim_to_batch_size(test_data, batch_size)

    # normalize output values
    output_scaler = MinMaxScaler(feature_range=(0, 1))
    train_data.loc[:, output_fields] = output_scaler.fit_transform(train_data.loc[:, output_fields])
    test_data.loc[:, output_fields] = output_scaler.transform(test_data.loc[:, output_fields])

    X_sel = ['sim_time'] + input_fields + ['theta(t-1)']
    y_sel = output_fields + ['theta']

    X_train, X_test, y_train, y_test = train_test_split_to_batches(
            train_data, test_data, X_sel, y_sel, network_settings)
    # expand the data (L, p+J+1) to (L-tau+1, tau, p+J+1)
    #tra.reshape((batch_size, n_train_batch_l0en, -1))
    #test_data.reshape((batch_size, n_test_batch_len, -1))
   
    data_info = {
            'n_train': n_train,
            'n_test': n_test,
            'train_batch_len': n_train/batch_size,
            'test_batch_len': n_test/batch_size
    }
    # access trial of specific inputs with df.loc['INPUT_VALUES', :]
    return (X_train, X_test, y_train, y_test, input_scaler, output_scaler, train_start_states, test_start_states, data_info)


