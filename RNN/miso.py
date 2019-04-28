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
from transform import transform
from keras.utils import plot_model
#TODO:
#  - implement cross validation to plot learning curve
def shape_it(X):
    return np.expand_dims(X.reshape((-1,1)),2)

os.environ["SISO_DATA_DIR"] = '/Users/li-wei/siso_bot/RNN/data/'
fname = 'data2.csv'

input_fields = ['left_pwm', 'right_pwm']
output_fields = ['model_pos_x', 'model_pos_y', 'theta']
others = ['sim_time']

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

    pdb.set_trace()
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

    plt.figure()
    plt.title('RMSE of train and test dataset')
    epoch_range = range(0, epochs, period)
    plt.plot(epoch_range, train_loss_history)
    plt.plot(epoch_range, test_loss_history)
    plt.legend(['train', 'test'])
    plt.show()
    '''
    rmse_train = 0
    rmse_test = 0

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_title('x')
    ax2.set_title('y')
    ax3.set_title('theta')
    train_predictions = np.array([np.squeeze(model.predict(twoD2threeD(X_train[i]), batch_size=batch_size), axis=0) for i in range(len(X_train))])
    for i in range(len(train_predictions)):
        if i < 3:
            ax1.plot(train_predictions[i, :, 0], color='blue')
            ax1.plot(y_train[i, :, 0], color='green')
            ax2.plot(train_predictions[i, :, 1], color='blue')
            ax2.plot(y_train[i, :, 1], color='green')
            ax3.plot(train_predictions[i, :, 2], color='blue')
            ax3.plot(y_train[i, :, 2], color='green') 
        rmse_train += np.sqrt(mean_squared_error(y_train[i], train_predictions[i]))
    print('Train RMSE: %.3f' % (rmse_train/y_train.size))
    pdb.set_trace()
    # plot train_predictions v.s. y_train
    plt.show()

    test_predictions = np.array([np.squeeze(model.predict(twoD2threeD(X_test[i]), batch_size=batch_size), axis=0) for i in range(len(X_test))])

    # report performance
    for i in range(len(X_test)):
        rmse_test += np.sqrt(mean_squared_error(y_test[i], test_predictions[i]))
    print('Test RMSE: %.3f' % (rmse_test/y_test.size))
    # line plot of observed vs predicted
    #plt.plot(y_test.reshape(len(y_test)))
    #plt.plot(test_predictions)
    #plt.show()
    #plt.show()
    '''    
    
    '''
    padded_train_predictions = np.array([_train_predictions for _ in range(J)])
    padded_train_predictions = padded_train_predictions.reshape((padded_train_predictions.shape[1], J))

    train_predictions = scaler.inverse_transform(padded_train_predictions)
    train_predictions = train_predictions[:, 2]
    train_predictions = inverse_difference(y_train.flatten(), train_predictions)
    '''
    # np.insert(train_predictions, 1, np.arange(n_train), axis=1)
   
    '''
    for i in range(len(X_test)):
        # make one-step forecast
        X = X_test[i, :]
        yhat = model.predict(X, batch_size=batch_size)
        # store forecast
        test_predictions.append(scaler.inverse_transform(yhat))

        expected = y_test[i]
        print('Predicted=%f, Expected=%f' % (yhat, expected))
    '''

    # test_predictions = np.array(test_predictions)
    
    '''
    padded_test_predictions = np.array([_test_predictions for _ in range(3)])
    padded_test_predictions = padded_test_predictions.reshape((padded_test_predictions.shape[1], 3))

    test_predictions = scaler.inverse_transform(padded_test_predictions)
    test_predictions = test_predictions[:, 2]
    test_predictions = inverse_difference(y_test.flatten(), test_predictions)
    # np.insert(test_predictions, 1, np.arange(n_train, n_data))
    '''

