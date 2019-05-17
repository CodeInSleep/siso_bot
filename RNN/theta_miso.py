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

from transform import train_test_split_to_batches
import pdb
input_fields = ['left_pwm', 'right_pwm']

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
    theta_data.loc[:, 'theta(t-1)'] = difference(theta_data.loc[:, 'theta'])
    theta_data.loc[:, 'sim_time'] = theta_data.loc[:, 'sim_time'].diff().fillna(0)
    theta_data = theta_data.iloc[:-1]
    encode_angle(theta_data, 'theta')
    encode_angle(theta_data, 'theta(t-1)')
    theta_data = theta_data.drop(['theta', 'theta(t-1)'], axis=1)
    pdb.set_trace()
    theta_data = theta_data.reindex(columns=['sim_time', 'left_pwm', 'right_pwm',
        'theta(t-1)_cos', 'theta(t-1)_sin', 'theta_cos', 'theta_sin'])
    pdb.set_trace()
    return theta_data.values

def fit_model(X_train, y_train):
    model = Sequential()

    model.add(Dense(10, batch_input_shape=(batch_size, time_step, layers_dims[0]),
        activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(10, activation='tanh'))
    model.add(Dense(2))

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

    X_train, X_test, y_train, y_test = transform(df)
   
    model = fit_model()
   
    iterations = 6
    epochs = 30
    train_history = []
    test_history = []
    decoded_y_train = decode_angles(y_train)
    y_test = decode_angles(y_test)
    for ep in range(epochs):
        model.fit(X_train, y_train, epochs=epochs)

        train_predictions = model.predict(X_train)
        train_predictions = decode_angles(train_predictions)
        test_predictions = model.predict(X_test)
        test_predictions = decode_angles(test_predictions)
    
        train_history.append(mean_squared_error(train_predictions, decoded_y_train))
        test_history.append(mean_squared_error(test_predictions, y_test))


    ep_range = range(0, epochs)
    plt.plot(ep_range, train_history)
    plt.plot(ep_range, test_history)
    plt.title('theta miso prediction (FNN)')
    plt.xlabel('epoch')
    plt.ylabel('RMSE theta(rad)')
    plt.show()

    fname = 'theta_model'
    model_json = model.to_json()
    with open(fname+'.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(fname+'.h5')

