import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load_dfs, decode_angles, save_model
from keras.models import Sequential, model_from_json
from keras.initializers import Identity, RandomNormal
from keras.layers import Dense, Dropout, LSTM, SimpleRNN, Dropout, GRU
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

fname = 'trial_1000_0_to_3.csv'
layers_dims = [4, 10, 10, 2]

def convert_angles(df):
	df.loc[:, 'theta(t-1)'] = decode_angles(df.loc[:, ['theta(t-1)_cos', 'theta(t-1)_sin']].values)
	df = df.drop(['theta(t-1)_cos', 'theta(t-1)_sin'], axis=1)

	return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get path to data directory')
    parser.add_argument('--datadir', required=True)
    args = parser.parse_args(sys.argv[1:])

    datadir = args.datadir
    if not os.path.isdir(datadir):
        print('invalid DATA_DIR (pass in as argument)')

    dirpath = os.path.abspath(os.path.join(datadir, fname.split('.')[0]))

    X_train_fname = os.path.join(dirpath, 'X_train.pkl')
    X_test_fname = os.path.join(dirpath, 'X_test.pkl')
    theta_y_train_fname = os.path.join(dirpath, 'theta_y_train.pkl')
    theta_y_test_fname = os.path.join(dirpath, 'theta_y_test.pkl')
    pos_y_train_fname = os.path.join(dirpath, 'pos_y_train.pkl')
    pos_y_test_fname = os.path.join(dirpath, 'pos_y_test.pkl')

    X_train = pd.read_pickle(X_train_fname)
    X_test = pd.read_pickle(X_test_fname)
    pos_y_train = pd.read_pickle(pos_y_train_fname)
    pos_y_test = pd.read_pickle(pos_y_test_fname)

    X_train = convert_angles(X_train)
    X_test = convert_angles(X_test)
    # FNN for dx dy prediction
    # 	inputs: v_l, v_r, theta, dt
    # 	outputs: dx, dy
    fnn_model = Sequential()
    fnn_model.add(Dense(layers_dims[1], activation='relu', input_shape=(layers_dims[0],)))
    fnn_model.add(Dropout(0.2))
    fnn_model.add(Dense(layers_dims[2], activation='relu'))
    fnn_model.add(Dropout(0.2))
    fnn_model.add(Dense(layers_dims[3]))

    fnn_model.compile(loss='mean_squared_error', optimizer='adam')

    iterations = 2
    epochs = 10
    train_loss_history = []
    test_loss_history = []

    X_train = X_train.values
    pos_y_train = pos_y_train.values
    pos_y_test = pos_y_test.values

    for it in range(iterations):
    	fnn_model.fit(X_train, pos_y_train, epochs=epochs, batch_size=32)

    	# make train predictions
    	train_predictions = fnn_model.predict(X_train)
    	train_mse = mean_squared_error(train_predictions, pos_y_train)
    	train_loss_history.append(train_mse)

    	test_predictions = fnn_model.predict(X_test)
    	test_mse = mean_squared_error(test_predictions, pos_y_test)
    	test_loss_history.append(test_mse)
    	print('train_mse: %f' % train_mse)
    	print('test_mse: %f' % test_mse)
    plt.figure()
    plt.title('RMSE of train and test dataset')
    it_range = range(0, iterations)
    plt.plot(it_range, train_loss_history)
    plt.plot(it_range, test_loss_history)
    plt.legend(['train', 'test'])
    plt.show()
    
    model_fname = 'pos_model'
    save_model(fnn_model, dirpath, model_fname)