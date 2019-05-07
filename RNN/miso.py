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
from keras.layers import LSTM, SimpleRNN, Dropout
from keras.utils import plot_model
from keras.initializers import Identity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from visualize import visualize_3D

import pdb

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from transform import transform, input_fields, output_fields, others
from keras.utils import plot_model

os.environ["SISO_DATA_DIR"] = '/Users/li-wei/siso_bot/RNN/data/'
fname = 'trial_3_to_6.csv'
#fname = 'trial_0_to_3dot5_step_0dot1.csv'

# TODOs
#   Hyperparmaters:
#       - dropout keep_prob
#       - parameter initialization
#       - Gradient clipping

def shape_it(X):
    return np.expand_dims(X.reshape((-1,1)),2)

def twoD2threeD(np_array):
    return np_array.reshape(1, np_array.shape[0], np_array.shape[1])

def calc_error(model, X, y, output_scaler):
    # X, y are 3D
    rmse = 0
    predictions = np.array([np.squeeze(model.predict(twoD2threeD(X[i]),
        batch_size=batch_size), axis=0) for i in range(len(X))])

    for i in range(len(predictions)):
        rmse += np.sqrt(mean_squared_error(y[i], predictions[i]))
    return rmse/y.size

if __name__ == '__main__':
    if not os.path.isdir(os.environ['SISO_DATA_DIR']):
        print('invalid DATA_DIR (set in ~/.bashrc')

    p = len(input_fields)
    J = len(output_fields)
    dirpath = os.environ['SISO_DATA_DIR']
    datafile = os.path.join(dirpath, fname)
    df = pd.read_csv(datafile, engine='python')
    df = df.loc[input_fields+output_fields+others]

    X_train, X_test, y_train, y_test, train_trial_names, test_trial_names, \
        output_scaler, start_states, max_duration = transform(df)

    batch_size = 1
    model = Sequential()
    model.add(Dense(p, batch_input_shape=(batch_size, max_duration, p), name='input_layer'))
    model.add(Dense(10, batch_input_shape=(batch_size, max_duration,), name='second_layer'))
    model.add(Dropout(0.8))
    model.add(Dense(J, batch_input_shape=(batch_size, max_duration,),
        activation='tanh', name='hidden_layer'))
    model.add(LSTM(J, batch_input_shape=(batch_size, max_duration, J), name='dynamic_layer',
        return_sequences=True, activation='tanh'))
    model.add(Dense(J))
    model.compile(loss='mean_squared_error', optimizer='adam')

    iterations = 30
    epochs = 30
    period = 10
    # learning curver
    train_loss_history = []
    test_loss_history = []
   
    # for debug purposes
    _X_train = X_train[0]
    _y_train = y_train[0]
    _X_test = X_test[0]
    _y_test = y_test[0]
    # plot learning curve
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.title = 'test trial'
    ax2 = fig.add_subplot(122)
    ax1.title = 'predictions'
    fig.show()
    for it in range(iterations):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)

        train_loss_history.append(calc_error(model, X_train, y_train, output_scaler))
        test_loss_history.append(calc_error(model, X_test, y_test, output_scaler)) 

        ax1.clear()
        ax2.clear()
        test_predictions = model.predict(twoD2threeD(_X_test), batch_size=batch_size)
        visualize_3D(twoD2threeD(_y_test), ax1)
        visualize_3D(test_predictions, ax2)
        plt.draw()
        plt.pause(5)
>>>>>>> overfit

     # examine results
    train_predictions = model.predict(X_train, batch_size=batch_size)
    test_predictions = model.predict(X_test, batch_size=batch_size)
   
    #def unnorm_and_undiff(arr_3D, scaler, trial_names, init_conditions):
    #    arr_3D_unnorm = np.zeros(arr_3D.shape)
    #    for i in range(len(arr_3D_unnorm)):
    #        arr_3D_unnorm[i] = scaler.inverse_transform(arr_3D[i])
    #    arr_3D_unnorm = arr_3D_unnorm.cumsum(axis=0)
    #    return arr_3D_unnorm

    '''
    train_predictions = unnorm_and_undiff(train_predictions, output_scaler,
            train_trial_names, start_states)
    test_predictions = unnorm_and_undiff(test_predictions, output_scaler,
            test_trial_names, start_states)
    train_gnd = unnorm_and_undiff(y_train, output_scaler, train_trial_names, start_states)
    test_gnd = unnorm_and_undiff(y_test, output_scaler, train_trial_names, start_states)
    '''
    np.save('train_predictions.npy', train_predictions) 
    np.save('test_predictions.npy', test_predictions)
    
    np.save('train_ground.npy', _y_train)
    np.save('test_ground.npy', _y_test)
    plt.figure()
    plt.title('RMSE of train and test dataset')
    it_range = range(0, iterations)
    plt.plot(it_range, train_loss_history)
    plt.plot(it_range, test_loss_history)
    plt.legend(['train', 'test'])
    plt.show()

