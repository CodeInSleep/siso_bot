import argparse
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
from numpy import arctan2
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, LSTM, SimpleRNN, Dropout, GRU
from keras.initializers import Identity, RandomNormal
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from visualize import visualize_3D
from transform import transform, input_fields, output_fields, others, network_settings

fname = 'trial_1000.csv'

# network parameter
p = len(input_fields)
J = len(output_fields)
layers_dims = [p, 10, J, J]
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

def predict_seq(model, X, output_scaler):
    # X is the input sequence (without ground truth previous prediction)
    X = twoD2threeD(X) if len(X.shape) == 2 else X
    prediction = model.predict(X)
    #prevState = prediction

    prediction = np.squeeze(prediction)
    prediction = output_scaler.inverse_transform(prediction)
    prediction = np.concatenate((prediction[:,:2], 
            decode_angles(prediction[:,2:])),axis=1)

    return prediction

def calc_error(model, X, y, output_scaler):
    # X, y are unnormalized 3D
    rmse = 0

    for i in range(len(X)):
        predictions = predict_seq(model, X[i], output_scaler)
        unnorm_y = output_scaler.inverse_transform(y[i])

        rmse += np.sqrt(mean_squared_error(y[i], predictions))
    return rmse/y.size

def save_obj(obj, dirpath, name):
    with open(os.path.join(dirpath, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(dirpath, name):
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
    model.add(Dense(p, batch_input_shape=(batch_size, time_step, p), name='input_layer'))
    model.add(Dense(layers_dims[1], activation='tanh', kernel_initializer=RandomNormal(stddev=np.sqrt(2./layers_dims[0])), name='second_layer'))
    model.add(Dropout(0.2))
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

def decode_angles(cos_sin_vals):
    return arctan2(cos_sin_vals[:, 1], cos_sin_vals[:, 0]).reshape(-1, 1)

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

    batch_size = network_settings['batch_size']
    timestep = network_settings['timestep']

    X_train_fname = os.path.join(dirpath, 'X_train.npy')
    X_test_fname = os.path.join(dirpath, 'X_test.npy')
    y_train_fname = os.path.join(dirpath, 'y_train.npy')
    y_test_fname = os.path.join(dirpath, 'y_test.npy')
    input_scaler_fname = os.path.join(dirpath, 'input_scaler.pkl')
    output_scaler_fname = os.path.join(dirpath, 'output_scaler.pkl')   
    if os.path.isfile(X_train_fname):
        X_train = np.load(os.path.join(dirpath, 'X_train.npy'), allow_pickle=True) 
        X_test = np.load(os.path.join(dirpath, 'X_test.npy'), allow_pickle=True)
        y_train = np.load(os.path.join(dirpath, 'y_train.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(dirpath, 'y_test.npy'), allow_pickle=True)
        input_scaler = joblib.load(input_scaler_fname)
        output_scaler = joblib.load(output_scaler_fname)
        data_info = load_obj(dirpath, 'data_info') 
    else:
        X_train, X_test, y_train, y_test, input_scaler, output_scaler, data_info = transform(df, count=-1)

        np.save(X_train_fname, X_train)
        np.save(X_test_fname, X_test)
        np.save(y_train_fname, y_train)
        np.save(y_test_fname, y_test)
        joblib.dump(input_scaler, input_scaler_fname)
        joblib.dump(output_scaler, output_scaler_fname)
        save_obj(data_info, dirpath, 'data_info')

    train_model_settings = {
            'batch_size': batch_size,
            'time_step': timestep,
            'stateful': True,
        }

    model = make_model(train_model_settings)
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

    # for debug purposes
    n = 0
    _X_train = X_train[n]
    _y_train = output_scaler.inverse_transform(y_train[n])
    _y_train = np.concatenate((_y_train[:,:2], 
        decode_angles(_y_train[:,2:])),axis=1)
    #_y_train = y_train[n]
    _X_test = X_test[n]
    _y_test = output_scaler.inverse_transform(y_test[n])
    #_y_test = y_test[n]
    _y_test = np.concatenate((_y_test[:,:2], 
            decode_angles(_y_test[:,2:])),axis=1)
    # plot learning curve
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.show()
   

    train_batch_len = data_info['train_batch_len']
    test_batch_len = data_info['test_batch_len']
    
    for ep in range(epochs):
        for j in range(int(train_batch_len/timestep)-1):
            time_seg = range((j*timestep), ((j+1)*timestep))
            model.fit(X_train[:, time_seg, :], y_train[:, time_seg, :], 
                    batch_size=batch_size, verbose=0, shuffle=False)

        model.reset_states()

        # calculate rmse for train data
        train_rmse = 0
        for j in range(int(data_info['train_batch_len']/timestep)-1):
            time_seg = range((j*timestep), ((j+1)*timestep))
            pred = model.predict(X_train[:, time_seg, :])
            for k in range(batch_size):
                train_rmse += mean_squared_error(pred[k], y_train[k, time_seg, :])
        train_rmse = np.sqrt(train_rmse)/(batch_size*train_batch_len)
        train_loss_history.append(train_rmse)
        
        test_rmse = 0
        for j in range(int(data_info['test_batch_len']/timestep)-1):
            time_seg = range((j*timestep), ((j+1)*timestep))
            pred = model.predict(X_test[:, time_seg, :])
            for k in range(batch_size):
                test_rmse += mean_squared_error(pred[k], y_test[k, time_seg, :])
        test_rmse = np.sqrt(test_rmse)/(batch_size*test_batch_len)
        test_loss_history.append(test_rmse)

        # make prediction on a test sequence
        pred_model = convert_to_inference_model(model)
        # predict on one trial at a time
        test_predictions = predict_seq(pred_model, _X_test,
                output_scaler)

        ax1.clear()
        ax2.clear()
        visualize_3D(twoD2threeD(np.cumsum(_y_test, axis=0)), 
                ax1, plt_arrow=True) 
        visualize_3D(twoD2threeD(np.cumsum(test_predictions, axis=0)), 
                ax2, plt_arrow=True)
        plt.draw()
        plt.pause(4)
       
        #train_loss_history.append(mean_squared_error(X_train, y_train,
        #    output_scaler))
        #test_loss_history.append(calc_error(stateful_model, X_test, y_test, output_scaler))
        
        #train_predictions = model.predict(X_train)
        #test_predictions = model.predict(X_test)
        #train_cost = np.mean((train_predictions-y_train)**2)
        #test_cost = np.mean((test_predictions-y_test)**2)
        #train_loss_history.append(train_cost)
        #test_loss_history.append(test_cost)
        print('Epoch %d' % ep)
        print('train_cost: %f' % train_rmse)
        print('test_cost: %f' % test_rmse)

    # examine results
    #train_predictions = model.predict(X_train, batch_size=batch_size)
    #test_predictions = model.predict(X_test, batch_size=batch_size)
   
    
    plt.figure()
    plt.title('RMSE of train and test dataset')
    ep_range = range(0, epochs)
    plt.plot(ep_range, train_loss_history)
    plt.plot(ep_range, test_loss_history)
    plt.legend(['train', 'test'])
    plt.show()
    
