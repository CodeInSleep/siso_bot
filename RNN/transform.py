import math
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
from numpy import cos, sin, arctan2
from sklearn.preprocessing import MinMaxScaler
from visualize import visualize_3D

input_fields = ['left_pwm', 'right_pwm']
output_fields = ['model_pos_x', 'model_pos_y', 'theta_cos', 'theta_sin']
#output_fields = ['model_pos_x', 'model_pos_y']
others = ['sim_time']
network_settings = {
    'p': len(input_fields),
    'J': len(output_fields),
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

def remove_bias(df, start_states):
    return df.loc[:, output_fields] - start_states.loc[df.name, output_fields]

def transform_group(group_df, max_duration):
    """
        transform each group so that each trial have length of max_duration
    """
    group_df = group_df.reset_index().drop('input', axis=1)
    cols = group_df.columns
  
    padding = pd.DataFrame(np.zeros((max_duration-group_df.shape[0], group_df.shape[1]), dtype=int))
    padding.columns = cols
    for i in range(len(padding)):
        padding.loc[i, output_fields] = group_df.loc[group_df.shape[0]-1, output_fields]
    # pad the time series with 
    padded_group_df = pd.DataFrame(pd.np.row_stack([group_df, padding]))
    padded_group_df.columns = cols 
    padded_group_df = padded_group_df.fillna(0)

    # add previous observation as feature for train data at current time step
    # for field in output_fields:
    #     padded_group_df.loc[:, field+'(t-1)'] = padded_group_df.loc[:, field].shift(1).fillna(0)

    return padded_group_df

def transform(df, train_percentage=0.7, count=-1):
    timestep = 5
    df.loc[:, 'theta'] = df.loc[:, 'theta'].apply(lambda x: math.radians(x))
    df.loc[:, 'theta_cos'] = df.loc[:, 'theta'].apply(lambda x: cos(x))
    df.loc[:, 'theta_sin'] = df.loc[:, 'theta'].apply(lambda x: sin(x))
    df.loc[:, 'left_pwm'] = df.loc[:, 'left_pwm'].apply(truncate, args=(3,))
    df.loc[:, 'right_pwm'] = df.loc[:, 'right_pwm'].apply(truncate, args=(3,))
    df.loc[:, 'input'] = 'l_'+df.loc[:, 'left_pwm'].map(str)+'_r_'+df.loc[:, 'right_pwm'].map(str)

     # normalize inputs
    input_scaler = MinMaxScaler(feature_range=(0, 1))
    df.loc[:,input_fields] = input_scaler.fit_transform(
            df.loc[:,input_fields])
    # provide a summary of inputs
    input_summary = df.groupby('input').apply(lambda x: x.describe())

    # time difference on the output fields
    df.loc[:, output_fields] = df.loc[:, output_fields].diff().fillna(0)

    batch_size = network_settings['batch_size']
    timestep = network_settings['timestep']
    p = network_settings['p']
    J = network_settings['J']
    # trim data set to a multiple of batch size
    # df = df.iloc[:-(len(df)%batch_size), :]
    n_train = int(0.7*len(df))
    n_test = len(df) - n_train
    train_data = df.iloc[:n_train, :]
    test_data = df.iloc[n_train:, :] 
    
    train_data = train_data.iloc[:-(len(train_data)%batch_size), :]
    test_data = test_data.iloc[:-(len(test_data)%batch_size), :]

    # normalize output values
    output_scaler = MinMaxScaler(feature_range=(0, 1))
    train_data.loc[:, output_fields] = output_scaler.fit_transform(train_data.loc[:, output_fields])
    test_data.loc[:, output_fields] = output_scaler.transform(test_data.loc[:, output_fields])

    n_train_batch_len = int(n_train/batch_size)
    n_test_batch_len = int(n_test/batch_size)

    X_train = train_data.loc[:, input_fields].values.reshape(
            batch_size, n_train_batch_len, p)
    y_train = train_data.loc[:, output_fields].values.reshape(
            batch_size, n_train_batch_len, J)
    X_test = test_data.loc[:, input_fields].values.reshape(
            batch_size, n_test_batch_len, p)
    y_test = test_data.loc[:, output_fields].values.reshape(
            batch_size, n_test_batch_len, J)

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
    return (X_train, X_test, y_train, y_test, input_scaler, output_scaler, data_info)


