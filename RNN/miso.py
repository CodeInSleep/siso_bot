import os
import sys
import math
from itertools import product
import pdb
import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, LSTM, SimpleRNN, Dropout, GRU
from keras.initializers import Identity, RandomNormal
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from visualize import visualize_3D
from transform import transform, input_fields, output_fields, others

os.environ["SISO_DATA_DIR"] = '/Users/li-wei/siso_bot/RNN/data/'
fname = 'trial_0_to_3dot5_step_0dot1.csv'

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

def convert_to_inference_model(original_model):
    original_model_json = original_model.to_json()
    inference_model_dict = json.loads(original_model_json)

    layers = inference_model_dict['config']['layers']
    for layer in layers:
        if 'stateful' in layer['config']:
            layer['config']['stateful'] = True

        if 'batch_input_shape' in layer['config']:
            layer['config']['batch_input_shape'][0] = 1
            layer['config']['batch_input_shape'][1] = None

    inference_model = model_from_json(json.dumps(inference_model_dict))
    inference_model.set_weights(original_model.get_weights())
    inference_model.reset_states()
    return inference_model

def predict_seq(model, X, y):
    # X is the input sequence (without ground truth previous prediction)
    #prevState = np.zeros((1, J))
    predictions = []
    for i in range(len(X)):
        prevState = y[i]
        state = twoD2threeD(np.concatenate((prevState.reshape(1, -1), X[i].reshape((1,-1))), axis=1))
        prediction = model.predict(state)
        predictions.append(prediction.ravel())
        #prevState = prediction
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
    batch_size = model_params['batch_size']
    stateful = model_params['stateful']
    time_step = model_params['time_step']
    
    model = Sequential()
    model.add(Dense(p+J, batch_input_shape=(batch_size, time_step, p+J), name='input_layer'))
    model.add(Dense(layers_dims[1], activation='tanh', kernel_initializer=RandomNormal(stddev=np.sqrt(2./layers_dims[0])), name='second_layer'))
    model.add(Dropout(0.7))
    #model.add(Dense(10, batch_input_shape=(batch_size, max_duration,), activation='tanh', kernel_initializer=RandomNormal(stddev=np.sqrt(2./layers_dims[1])), name='third_layer'))
    #model.add(Dropout(0.7))
    model.add(Dense(J, activation='tanh', kernel_initializer=RandomNormal(stddev=np.sqrt(2./layers_dims[1])), name='hidden_layer'))
    model.add(GRU(J, name='dynamic_layer', return_sequences=True, activation='tanh', stateful=stateful))
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
            'stateful': False,
            'time_step': max_duration,
        }

    model = make_model(stateless_model_params)
    iterations = 50
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
    n = 40
    _X_train = X_train[n,:,3:]
    _y_train = output_scaler.inverse_transform(y_train[n])
    _X_test = X_test[n, :, 3:]
    _y_test = output_scaler.inverse_transform(y_test[n])
    # plot learning curve
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.show()
    
    for it in range(iterations):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)
        # create a stateful model for prediction
        stateful_model = convert_to_inference_model(model)
        # predict on one trial at a time
        train_predictions = predict_seq(stateful_model, _X_train, _y_train)
        test_predictions = predict_seq(stateful_model, _X_test, _y_test)
    
        # unnormalize predictions for visualization and usage
        train_predictions = output_scaler.inverse_transform(train_predictions)
        test_predictions = output_scaler.inverse_transform(test_predictions) 
        

        ax1.clear()
        ax2.clear()
        visualize_3D(twoD2threeD(_y_train), ax1, plt_arrow=True)
        visualize_3D(twoD2threeD(train_predictions), ax2, plt_arrow=True)
        plt.draw()
        plt.pause(4)

        train_loss_history.append(calc_error(stateful_model, X_train[:,:,3:], y_train))
        test_loss_history.append(calc_error(stateful_model, X_test[:,:,3:], y_test))
        
    
    # examine results
    #train_predictions = model.predict(X_train, batch_size=batch_size)
    #test_predictions = model.predict(X_test, batch_size=batch_size)
   
    
    np.save(os.path.join(dirpath, 'train_predictions.npy'), train_predictions) 
    np.save(os.path.join(dirpath, 'test_predictions.npy'), test_predictions)
    
    np.save(os.path.join(dirpath, 'train_ground.npy'), _y_train)
    np.save(os.path.join(dirpath, 'test_ground.npy'), _y_test)
    plt.figure()
    plt.title('RMSE of train and test dataset')
    it_range = range(0, iterations)
    plt.plot(it_range, train_loss_history)
    plt.plot(it_range, test_loss_history)
    plt.legend(['train', 'test'])
    plt.show()
