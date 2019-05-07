import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D

from pandas import Series
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape
from keras.layers import Embedding
from keras.layers import LSTM, SimpleRNN, Dropout
from keras.utils import plot_model
from keras.initializers import Identity, RandomNormal
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
    if len(np_array.shape) != 2:
        raise AssertionError('np_array must be 2 dimension')
    return np_array.reshape(1, np_array.shape[0], np_array.shape[1])

def calc_error(model, X, y):
    # X, y are 3D
    rmse = 0
    predictions = model.predict(X, batch_size=batch_size)

    for i in range(len(predictions)):
        rmse += np.sqrt(mean_squared_error(y[i], predictions[i]))
    return rmse/y.size

def save_obj(obj, name):
    with open(os.path.join(dirpath, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(os.path.join(dirpath, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    if not os.path.isdir(os.environ['SISO_DATA_DIR']):
        print('invalid DATA_DIR (set in ~/.bashrc')

    p = len(input_fields)
    J = len(output_fields)
    dirpath = os.environ['SISO_DATA_DIR']+fname.split('.')[0]
    datafile = os.path.join(dirpath, fname)
    df = pd.read_csv(datafile, engine='python')
    df = df[input_fields+output_fields+others]

    X_train_fname = os.path.join(dirpath, 'X_train.npy')
    X_test_fname = os.path.join(dirpath, 'X_test.npy')
    y_train_fname = os.path.join(dirpath, 'y_train.npy')
    y_test_fname = os.path.join(dirpath, 'y_test.npy')
    pdb.set_trace()
    if os.path.isfile(X_train_fname):
        X_train = np.load(X_train_fname) 
        X_test = np.load(X_test_fname)
        y_train = np.load(y_train_fname)
        y_test = np.load(y_test_fname)
        parameters = load_obj('parameters')
        train_trial_names = parameters['train_trial_names']
        test_trial_names = parameters['test_trial_names']
        max_duration= parameters['max_duration']
    else:
        X_train, X_test, y_train, y_test, train_trial_names, test_trial_names, \
        output_scaler, start_states, max_duration = transform(df, count=-1)

        np.save(X_train_fname, X_train)
        np.save(X_test_fname, X_test)
        np.save(y_train_fname, y_train)
        np.save(y_test_fname, y_test)
        parameters = {}
        parameters['train_trial_names'] = train_trial_names.tolist()
        parameters['test_trial_names'] = test_trial_names.tolist()
        parameters['max_duration'] = max_duration
        save_obj(parameters, 'parameters')

    pdb.set_trace()
    layers_dims = [p, 10, J, J]
    batch_size = 16
    model = Sequential()
    model.add(Dense(p, batch_input_shape=(None, max_duration, p), name='input_layer'))
    model.add(Dense(10, batch_input_shape=(None, max_duration,), activation='tanh', kernel_initializer=RandomNormal(stddev=np.sqrt(2./layers_dims[0])), name='second_layer'))
    model.add(Dropout(0.7))
    #model.add(Dense(10, batch_input_shape=(batch_size, max_duration,), activation='tanh', kernel_initializer=RandomNormal(stddev=np.sqrt(2./layers_dims[1])), name='third_layer'))
    #model.add(Dropout(0.7))
    model.add(Dense(J, batch_input_shape=(None, max_duration,),
        activation='tanh', kernel_initializer=RandomNormal(stddev=np.sqrt(2./layers_dims[1])), name='hidden_layer'))
    model.add(LSTM(J, batch_input_shape=(batch_size, max_duration, J), name='dynamic_layer',
        return_sequences=True, activation='tanh'))
    model.add(Dense(J))
    model.compile(loss='mean_squared_error', optimizer='adam')

    iterations = 50
    epochs = 10
    period = 10
    # learning curver
    train_loss_history = []
    test_loss_history = []
  

    # for debug purposes
    _X_train = twoD2threeD(X_train[2])
    _y_train = twoD2threeD(y_train[2])
    _X_test = twoD2threeD(X_test[2])
    _y_test = twoD2threeD(y_test[2])
    # plot learning curve
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.show()
    
    for i in range(len(y_train)):
        ax1.clear()
        visualize_3D(twoD2threeD(y_train[i]), ax1) 
        plt.draw()
        plt.pause(5)
    for it in range(iterations):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)
        #pdb.set_trace()
        train_loss_history.append(calc_error(model, X_train, y_train))
        test_loss_history.append(calc_error(model, X_test, y_test))
        ax1.clear()
        ax2.clear()
        test_predictions = model.predict(_X_test, batch_size=batch_size)
        visualize_3D(_y_test, ax1)
        visualize_3D(test_predictions, ax2)
        plt.draw()

    # examine results
    #train_predictions = model.predict(X_train, batch_size=batch_size)
    #test_predictions = model.predict(X_test, batch_size=batch_size)
   
    '''
    train_predictions = unnorm_and_undiff(train_predictions, output_scaler,
            train_trial_names, start_states)
    test_predictions = unnorm_and_undiff(test_predictions, output_scaler,
            test_trial_names, start_states)
    train_gnd = unnorm_and_undiff(y_train, output_scaler, train_trial_names, start_states)
    test_gnd = unnorm_and_undiff(y_test, output_scaler, train_trial_names, start_states)
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
    '''
