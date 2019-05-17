import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
from numpy import cos, sin, arctan2
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, LSTM, SimpleRNN, Dropout, GRU
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from transform import truncate, difference
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from transform import trim_to_batch_size 
import pdb
input_fields = ['left_pwm', 'right_pwm']

layers_dims = [3, 10, 10, 2]
batch_size = 32
time_step = 20
fname = 'trial_1000.csv'
def encode_angle(df, theta_field):
    df.loc[:, theta_field+'_cos'] = df.loc[:, theta_field].apply(lambda x: cos(x))
    df.loc[:, theta_field+'_sin'] = df.loc[:, theta_field].apply(lambda x: sin(x))

def decode_angles(cos_sin_vals):
    return arctan2(cos_sin_vals[:, 1], cos_sin_vals[:, 0]).reshape(-1, 1)

def transform(df):
    df.loc[:, 'theta'] = df.loc[:, 'theta'].apply(lambda x: math.radians(x))
    df.loc[:, 'left_pwm'] = df.loc[:, 'left_pwm'].apply(truncate, args=(3,))
    df.loc[:, 'right_pwm'] = df.loc[:, 'right_pwm'].apply(truncate, args=(3,))
    df.loc[:, 'input'] = 'l_'+df.loc[:, 'left_pwm'].map(str)+'_r_'+df.loc[:, 'right_pwm'].map(str)

    # normalize inputs
    input_scaler = MinMaxScaler(feature_range=(0, 1))
    df.loc[:,input_fields] = input_scaler.fit_transform(df.loc[:,input_fields])

    theta_data = df.loc[:, ['sim_time']+input_fields+['theta']]
    theta_data.loc[:, 'sim_time'] = theta_data.loc[:, 'sim_time'].diff().fillna(0)
    theta_data = theta_data.iloc[:-1]
    encode_angle(theta_data, 'theta')
    theta_data = theta_data.drop('theta', axis=1)
    theta_data = theta_data.reindex(columns=['sim_time', 'left_pwm', 'right_pwm',
        'theta_cos', 'theta_sin'])

    X_train, X_test, y_train, y_test = train_test_split(theta_data.loc[:, ['sim_time', 'left_pwm', 'right_pwm']], theta_data.loc[:, ['theta_cos', 'theta_sin']], test_size=0.3, random_state = 42)
    
    p = layers_dims[0]
    J = layers_dims[-1]

    X_train = trim_to_batch_size(X_train, batch_size)
    X_test = trim_to_batch_size(X_test, batch_size)
    y_train = trim_to_batch_size(y_train, batch_size)
    y_test = trim_to_batch_size(y_test, batch_size)

    train_batch_len = int(len(X_train)/batch_size)
    test_batch_len = int(len(X_test)/batch_size)
    
    X_train = X_train.values.reshape(
            batch_size, train_batch_len, p)
    X_test = X_test.values.reshape(
            batch_size, test_batch_len, p)
    y_train = y_train.values.reshape(
            batch_size, train_batch_len, J)
    y_test = y_test.values.reshape(
            batch_size, test_batch_len, J)

    return (X_train, X_test, y_train, y_test, train_batch_len, test_batch_len)

def make_model():
    model = Sequential()

    model.add(Dense(layers_dims[1], batch_input_shape=(batch_size, time_step, layers_dims[0]),
        activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(layers_dims[2], activation='tanh', return_sequences=True, stateful=True))
    model.add(Dense(layers_dims[3]))

    optimizer = Adam(lr=1e-5)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model

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

    X_train, X_test, y_train, y_test, train_batch_len, test_batch_len = transform(df)
   
    model = make_model()
   
    iterations = 20
    # learning curver
    train_loss_history = []
    test_loss_history = []
  
    # decoded_y_train = decode_angles(y_train)
    # y_test = decode_angles(y_test)
    for it in range(iterations):
        print("iteration %d" % it)
        for j in range(int(train_batch_len/time_step)-1):
            time_seg = range((j*time_step), ((j+1)*time_step))
            model.fit(X_train[:, time_seg, :], y_train[:, time_seg, :], 
                    batch_size=batch_size, verbose=1, shuffle=False)

        model.reset_states()

        # calculate rmse for train data
        train_rmse = 0
        for j in range(int(train_batch_len/time_step)-1):
            time_seg = range((j*time_step), ((j+1)*time_step))
            pred = model.predict(X_train[:, time_seg, :])
            for k in range(batch_size):
                train_rmse += mean_squared_error(pred[k], y_train[k, time_seg, :])
        train_rmse = np.sqrt(train_rmse)/(batch_size*train_batch_len)
        train_loss_history.append(train_rmse)
        
        test_rmse = 0
        for j in range(int(test_batch_len/time_step)-1):
            time_seg = range((j*time_step), ((j+1)*time_step))
            pred = model.predict(X_test[:, time_seg, :])
            for k in range(batch_size):
                test_rmse += mean_squared_error(pred[k], y_test[k, time_seg, :])
        test_rmse = np.sqrt(test_rmse)/(batch_size*test_batch_len)
        test_loss_history.append(test_rmse)


    ep_range = range(0, iterations)
    plt.plot(ep_range, train_loss_history)
    plt.plot(ep_range, test_loss_history)
    plt.title('theta miso prediction (FNN)')
    plt.xlabel('epoch')
    plt.ylabel('RMSE theta(rad)')
    plt.show()

    fname = 'theta_model'
    model_json = model.to_json()
    with open(fname+'.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(fname+'.h5')

