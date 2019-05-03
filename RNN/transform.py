import math
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from visualize import visualize_3D

input_fields = ['left_pwm', 'right_pwm']
output_fields = ['model_pos_x', 'model_pos_y', 'theta']
others = ['sim_time']

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

def transform_group(group_df, max_duration):
    group_df = group_df.reset_index().drop('input', axis=1)
    cols = group_df.columns
  
    padding = pd.DataFrame(np.zeros((max_duration-group_df.shape[0], group_df.shape[1]), dtype=int))
    padding.columns = cols
    for i in range(len(padding)):
        padding.loc[i, output_fields] = group_df.loc[group_df.shape[0]-1, output_fields]
    # pad the time series with 
    padded_group_df = pd.DataFrame(pd.np.row_stack([group_df, padding]))
    padded_group_df.columns = cols 
    # padded_group_df.loc[:, output_fields] = padded_group_df.loc[:, output_fields].diff()
    padded_group_df = padded_group_df.fillna(0)

    return padded_group_df

def transform(df, train_percentage=0.7, count=-1):
    df['theta'] = df['theta'].apply(lambda x: math.radians(x))
    df['left_pwm'] = df['left_pwm'].apply(truncate, args=(3,))
    df['right_pwm'] = df['right_pwm'].apply(truncate, args=(3,))
    df['input'] = 'l_'+df['left_pwm'].map(str)+'_r_'+df['right_pwm'].map(str)
    df = df.set_index(['input'])
    df = df.iloc[:count, :]

    # normalize inputs
    input_scaler = MinMaxScaler(feature_range=(0, 1))
    output_scaler = MinMaxScaler(feature_range=(0, 1))
    df.loc[:,input_fields] = input_scaler.fit_transform(df.loc[:,input_fields])
    grouped = df.groupby(df.index)
    num_trials = len(grouped)

    #for key, item in grouped:
    #    print(grouped.get_group(key), '\n\n')

    # store max duration of a trial
    max_duration = max(grouped['sim_time'].count())
    n_train = int(num_trials*train_percentage)

    # the start time of every trial, used later to recover trajectories
    start_times = grouped.first()

    # df.loc[:,'sim_time'] = grouped.apply(lambda x: x.loc[:, ['sim_time']].diff().cumsum().fillna(0))

    debug = True
    debug_trial = 'l_6.0_r_4.0'

    if debug:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        visualize_3D([grouped.get_group(debug_trial).loc[:, output_fields].values], ax1, plt_arrow=True)
    # remove the bias of starting points in each trial
    df.loc[:, output_fields] = grouped.apply(
            lambda x: x.loc[:, output_fields] - start_times.loc[x.name].loc[output_fields])
    grouped = df.groupby(df.index)
    # remove bias in theta
    df.loc[:, ['model_pos_x', 'model_pos_y']] = grouped.apply(
            lambda x: rotate(x.loc[:, ['model_pos_x', 'model_pos_y']], -start_times.loc[x.name].loc['theta']))
    #pdb.set_trace()
    if debug:
        grouped = df.groupby(df.index)   
        visualize_3D([grouped.get_group(debug_trial).loc[:, output_fields].values], ax1, plt_arrow=True)
        plt.show()

    # create new data frame that is of (# of trials, max_duration dimenstion) 
    df = df.groupby(['input']).apply(lambda x: transform_group(x, max_duration))

    trial_names = df.index.levels[0]
    train_samples = np.random.choice(num_trials, n_train, replace=False)
    train_trial_names = [trial_names[i] for i in train_samples]
    test_samples = np.array([i for i in range(num_trials) if i not in train_samples])
    test_trial_names = [trial_names[i] for i in test_samples]

    train_data = df.loc[train_trial_names, :]
    test_data = df.loc[test_trial_names, :] 

    # normalize the output differences
    # train_data.loc[:, output_fields] = output_scaler.fit_transform(train_data.loc[:, output_fields])
    # test_data.loc[:, output_fields] = output_scaler.transform(test_data.loc[:, output_fields])

    # unstack time series to columns
    train_data = train_data.unstack(level=1)
    test_data = test_data.unstack(level=1)

    p = len(input_fields)
    J = len(output_fields)

    train_trial_names = train_data.index
    test_trial_names = test_data.index

    X_train = train_data[input_fields].values.reshape(n_train, p, max_duration).transpose(0, 2, 1)
    X_test = test_data[input_fields].values.reshape(num_trials-n_train, p, max_duration).transpose(0, 2, 1)
    y_train = train_data[output_fields].values.reshape(n_train, J, max_duration).transpose(0, 2, 1)
    y_test = test_data[output_fields].values.reshape(num_trials-n_train, J, max_duration).transpose(0, 2, 1)

    # access trial of specific inputs with df.loc['INPUT_VALUES', :]
    return (X_train, X_test, y_train, y_test, train_trial_names, test_trial_names, output_scaler, start_times, max_duration)

def inverse_transform(predictions, target, start_times):
    # undo what transform function did, converting numpy array to pandas DF
    df = pd.DataFrame(predictions)

