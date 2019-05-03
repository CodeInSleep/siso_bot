import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pandas import Series
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape
from keras.layers import Embedding
from keras.layers import LSTM, SimpleRNN
from keras.utils import plot_model
from keras.initializers import Identity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import pdb

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from transform import transform, input_fields, output_fields, others
from keras.utils import plot_model
#TODO:
#  - implement cross validation to plot learning curve
def shape_it(X):
    return np.expand_dims(X.reshape((-1,1)),2)

os.environ["SISO_DATA_DIR"] = '/Users/li-wei/siso_bot/RNN/data/'
fname = 'data2.csv'

def twoD2threeD(np_array):
    return np_array.reshape(1, np_array.shape[0], np_array.shape[1])

def calc_error(model, X, y, output_scaler):
    rmse = 0
    predictions = np.array([np.squeeze(model.predict(twoD2threeD(X[i]), batch_size=batch_size), axis=0) for i in range(len(X))])

    for i in range(len(predictions)):
        rmse += np.sqrt(mean_squared_error(y[i], output_scaler.inverse_transform(predictions[i])))
    return rmse/y.size

if __name__ == '__main__':
    if not os.path.isdir(os.environ['SISO_DATA_DIR']):
        print('invalid DATA_DIR (set in ~/.bashrc')

    p = len(input_fields)
    J = len(output_fields)
    dirpath = os.environ['SISO_DATA_DIR']
    datafile = os.path.join(dirpath, fname)
    df = pd.read_csv(datafile, engine='python')
    df = df[input_fields+output_fields+others]

    X_train, X_test, y_train, y_test, train_trial_names, test_trial_names, output_scaler, start_times, max_duration = transform(df, input_fields, output_fields)

    batch_size = 1
    model = Sequential()
    model.add(Dense(p, batch_input_shape=(batch_size, max_duration, p), name='input_layer'))
    model.add(Dense(J, batch_input_shape=(batch_size, max_duration,), activation='tanh', name='hidden_layer'))
    model.add(LSTM(J, batch_input_shape=(batch_size, max_duration, J), name='dynamic_layer',
    kernel_initializer=Identity(J), stateful=True, return_sequences=True))
    model.compile(loss='mean_squared_error', optimizer='adam')

    epochs = 300
    period = 10
    # learning curver
    train_loss_history = []
    test_loss_history = []
    for i in range(epochs):
        for j in range(len(X_train)):
            model.fit(twoD2threeD(X_train[j]), twoD2threeD(y_train[j]), epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
    
        if i % period == 0:
            # plot learning curve
            train_loss_history.append(calc_error(model, X_train, y_train, output_scaler))
            test_loss_history.append(calc_error(model, X_test, y_test, output_scaler))
   
    # examine results
    train_predictions = np.array([np.squeeze(model.predict(twoD2threeD(X_train[i]), batch_size=batch_size), axis=0) for i in range(len(X_train))])
    
    test_predictions = np.array([np.squeeze(model.predict(twoD2threeD(X_test[i]), batch_size=batch_size), axis=0) for i in range(len(X_test))])
   
    def unnorm(arr_3D, scaler):
        arr_3D_unnorm = np.zeros(arr_3D.shape)
        for i in range(len(arr_3D_unnorm)):
            arr_3D_unnorm[i] = scaler.inverse_transform(arr_3D[i])
        arr_3D_unnorm = arr_3D_unnorm.cumsum(axis=0)
        return arr_3D_unnorm

    np.save('train_predictions.npy', unnorm(train_predictions, output_scaler)) 
    np.save('test_predictions.npy', unnorm(test_predictions, output_scaler))
    
    np.save('train_ground.npy', unnorm(y_train, output_scaler))
    np.save('test_ground.npy', y_test)
    plt.figure()
    plt.title('RMSE of train and test dataset')
    epoch_range = range(0, epochs, period)
    plt.plot(epoch_range, train_loss_history)
    plt.plot(epoch_range, test_loss_history)
    plt.legend(['train', 'test'])
    plt.show()

