import os
import sys
import math
from itertools import product
import pdb

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, SimpleRNN, Dropout
from keras.initializers import Identity, RandomNormal
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from visualize import visualize_3D
from transform import transform, input_fields, output_fields, others

os.environ["SISO_DATA_DIR"] = '/Users/li-wei/siso_bot/RNN/data/'
fname = 'trial_3_to_6.csv'

# network parameter
p = len(input_fields)
J = len(output_fields)
layers_dims = [p+J, 10, J, J]
#fname = 'trial_0_to_3dot5_step_0dot1.csv'

# TODOs
#   Hyperparmaters:
#       - dropout keep_prob
#       - Gradient clipping
#   Evaluation:
#       - stability (gradient visualization, gradient clipping)
#       - learning speed
#       - predict arbitrary length
def shape_it(X):
    return np.expand_dims(X.reshape((-1,1)),2)

def twoD2threeD(np_array):
    if len(np_array.shape) != 2:
        raise AssertionError('np_array must be 2 dimension')
    return np_array.reshape(1, np_array.shape[0], np_array.shape[1])

def predict_seq(model, X, y):
    # X is the input sequence (without ground truth previous prediction)
    prevState = np.zeros((1, J))
    predictions = []
    for i in range(len(X)):
        state = twoD2threeD(np.concatenate((prevState.reshape(1, -1), X[i].reshape((1,-1))), axis=1))
        prediction = model.predict(state)
        predictions.append(prediction.ravel())
        prevState = prediction
    model.reset_states()
    return np.array(predictions)

def calc_error(model, X, y):
    # X, y are 3D
    rmse = 0

    for i in range(len(X)):
        predictions = predict_seq(model, X[i], y[i])
        rmse += np.sqrt(mean_squared_error(y[i], predictions))
    return rmse/y.size

def save_obj(obj, name):
    with open(os.path.join(dirpath, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(os.path.join(dirpath, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

def make_model(model_params, weights=None):
    '''
        turn the stateless model used for training into a stateful one
        for one step prediction
    '''
    time_step = model_params['time_step']
    batch_size = model_params['batch_size']
    stateful = model_params['stateful']

    model = Sequential()
    model.add(Dense(p+J, batch_input_shape=(batch_size, time_step, p+J), name='input_layer'))
    model.add(Dense(layers_dims[1], activation='tanh', kernel_initializer=RandomNormal(stddev=np.sqrt(2./layers_dims[0])), name='second_layer'))
    model.add(Dropout(0.7))
    #model.add(Dense(10, batch_input_shape=(batch_size, max_duration,), activation='tanh', kernel_initializer=RandomNormal(stddev=np.sqrt(2./layers_dims[1])), name='third_layer'))
    #model.add(Dropout(0.7))
    model.add(Dense(J, activation='tanh', kernel_initializer=RandomNormal(stddev=np.sqrt(2./layers_dims[1])), name='hidden_layer'))
    model.add(LSTM(J, name='dynamic_layer', return_sequences=True, activation='tanh', stateful=stateful))
    model.add(Dense(J))
    if weights:
        # override default weights
        model.set_weights(weights)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
    
if __name__ == '__main__':
    if not os.path.isdir(os.environ['SISO_DATA_DIR']):
        print('invalid DATA_DIR (set in ~/.bashrc')

    dirpath = os.environ['SISO_DATA_DIR']+fname.split('.')[0]
    datafile = os.path.join(dirpath, fname)
    df = pd.read_csv(datafile, engine='python')
    df = df[input_fields+output_fields+others]

    X_train_fname = os.path.join(dirpath, 'X_train.npy')
    X_test_fname = os.path.join(dirpath, 'X_test.npy')
    y_train_fname = os.path.join(dirpath, 'y_train.npy')
    y_test_fname = os.path.join(dirpath, 'y_test.npy')
    scaler_fname = os.path.join(dirpath, 'scaler.pkl')
    if os.path.isfile(X_train_fname):
        X_train = np.load(os.path.join(dirpath, 'X_train.npy')) 
        X_test = np.load(os.path.join(dirpath, 'X_test.npy'))
        y_train = np.load(os.path.join(dirpath, 'y_train.npy'))
        y_test = np.load(os.path.join(dirpath, 'y_test.npy'))
        parameters = load_obj('parameters')
        train_trial_names = parameters['train_trial_names']
        test_trial_names = parameters['test_trial_names']
        max_duration= parameters['max_duration']
        output_scaler = joblib.load(scaler_fname)
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
        joblib.dump(output_scaler, scaler_fname)

    batch_size = None
    
    stateless_model_params = {
            'batch_size': batch_size,
            'time_step': max_duration,
            'stateful': False
        }
    model = make_model(stateless_model_params)
    iterations = 100
    epochs = 10
    # learning curver
    train_loss_history = []
    test_loss_history = []
  
    '''
    # for debug purposes
    _X_train = X_train[:4]
    _y_train = y_train[:4]
    _X_test = X_test[:4]
    _y_test = y_test[:4]
    plot_l = 2
    plot_w = 2
    # plot learning curve
    train_fig, train_axes = plt.subplots(plot_l, plot_w)
    test_fig, test_axes = plt.subplots(plot_l, plot_w)
    
    train_fig.title = 'train trials'
    test_fig.title = 'test trials'
    train_fig.show()
    test_fig.show()
    for it in range(iterations):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)

        train_loss_history.append(calc_error(model, X_train, y_train, output_scaler))
        test_loss_history.append(calc_error(model, X_test, y_test, output_scaler)) 

        for idx, (x, y) in enumerate(product(range(plot_l), range(plot_w))):
            train_axes[x, y].clear()
            test_axes[x, y].clear()
       
        for idx, (x, y) in enumerate(product(range(plot_l), range(plot_w))):
            train_predictions = model.predict(twoD2threeD(_X_train[plot_l*x+y])) 
            test_predictions = model.predict(twoD2threeD(_X_test[plot_l*x+y]))
            visualize_3D(twoD2threeD(_y_train[plot_l*x+y]), train_axes[x, y])
            visualize_3D(train_predictions, train_axes[x, y])

            visualize_3D(twoD2threeD(_y_test[plot_l*x+y]), test_axes[x, y])
            visualize_3D(test_predictions, test_axes[x, y])
    '''
    _X_train = X_train[2,:,3:]
    _y_train = output_scaler.inverse_transform(y_train[2])
    _X_test = X_test[2, :, 3:]
    _y_test = output_scaler.inverse_transform(y_test[2])
    # plot learning curve
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.show()
    
    for it in range(iterations):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)
        # create a stateful model for prediction
        stateful_model_params = {
                'batch_size': 1,
                'time_step': 1,
                'stateful': True
            }
        stateful_model = make_model(stateful_model_params, weights=model.get_weights())
        # predict on one trial at a time
        train_predictions = predict_seq(stateful_model, _X_train, _y_train)
        test_predictions = predict_seq(stateful_model, _X_test, _y_test)

        # unnormalize predictions for visualization and usage
        train_predictions = output_scaler.inverse_transform(train_predictions)
        test_predictions = output_scaler.inverse_transform(test_predictions)
       
        train_loss_history.append(calc_error(stateful_model, X_train[:,:,3:], y_train))
        test_loss_history.append(calc_error(stateful_model, X_test[:,:,3:], y_test))
        ax1.clear()
        ax2.clear()
        visualize_3D(twoD2threeD(_y_train), ax1)
        visualize_3D(twoD2threeD(train_predictions), ax2)
        plt.draw()
        plt.pause(5)

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
