import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
from numpy import cos, sin, arctan2
import matplotlib.pyplot as plt
import sklearn.base
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from keras.models import Sequential, model_from_json
from keras.initializers import Identity, RandomNormal
from keras.layers import Dense, Dropout, LSTM, SimpleRNN, Dropout, GRU, BatchNormalization, Activation
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

import csv
import pdb

fname_train = 'real_robot_data.csv'
fname_test = 'real_robot_data.csv'

model_fname = 'fnn_model'

model_cached = False
layers_dims = [4, 10, 20, 20, 3]

def load_model(dirpath, model_fname):
    # load theta RNN model
    json_file = open(os.path.join(dirpath, '{}.json'.format(model_fname)), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(os.path.join(dirpath, "{}.h5".format(model_fname)))
    print("Loaded {} from disk".format(model_fname))
    return model

def save_model(model, dirpath, model_fname):
    model_json = model.to_json()
    with open(os.path.join(dirpath, model_fname+'.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(dirpath, model_fname+'.h5'))

def make_model(layers_dims, lr=1e-3):
    model = Sequential()

    model.add(Dense(layers_dims[1], input_shape=(layers_dims[0],),
        kernel_initializer=RandomNormal(stddev=np.sqrt(2./layers_dims[0]))))
    # model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(layers_dims[2], activation='tanh', 
        kernel_initializer=RandomNormal(stddev=np.sqrt(2./layers_dims[1]))))
    model.add(Dropout(0.3))
    model.add(Dense(layers_dims[3], activation='tanh', 
        kernel_initializer=RandomNormal(stddev=np.sqrt(2./layers_dims[1]))))
    model.add(Dropout(0.3))
    # model.add(LSTM(layers_dims[2], activation='tanh', return_sequences=True))
    model.add(Dense(layers_dims[4]))

    optimizer = Adam(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def truncate(num, digits):
    stepper = pow(10.0, digits)
    return math.trunc(num*stepper)/stepper

def diff(df, fields):
    df.loc[:, fields] = df.loc[:, fields].diff().fillna(0)
    return df

def diffTheta(ang1, ang2):
    a = ang1 - ang2
    return (a+180)%360-180

def l2Diff(predictions, targets):
    return np.sqrt(np.square(predictions-targets))

def transformData(df, rangeLim=None):
    df.loc[:, 'left_pwm'] = df.loc[:, 'left_pwm'].apply(truncate, args=(3,))
    df.loc[:, 'right_pwm'] = df.loc[:, 'right_pwm'].apply(truncate, args=(3,))

    df.loc[:, 'theta_prev'] = df.loc[:, 'theta'].shift(1)

    # make xy in mm
    # df.loc[:, 'model_pos_x'] = df.loc[:, 'model_pos_x']*1000
    # df.loc[:, 'model_pos_x'] = df.loc[:, 'model_pos_x'].apply(truncate, args=(3,))
    # df.loc[:, 'model_pos_y'] = df.loc[:, 'model_pos_y']*1000
    # df.loc[:, 'model_pos_y'] = df.loc[:, 'model_pos_y'].apply(truncate, args=(3,))


    df.loc[:, 'delta_theta'] = diffTheta(df.loc[:, 'theta'], df.loc[:, 'theta_prev'])

    start_state = df.loc[:, ['model_pos_x', 'model_pos_y', 'theta']].head(1)
    df.loc[:, 'dt'] = df.loc[:, 'sim_time'].diff()
    df.loc[:, 'delta_x'] = df.loc[:, 'model_pos_x'].diff()
    df.loc[:, 'delta_y'] = df.loc[:, 'model_pos_y'].diff()
    df.drop(df.head(1).index, inplace=True)

    sysOut_actual = df.loc[:, ['model_pos_x', 'model_pos_y', 'theta']].values
    sysOut = df.loc[:, ['delta_x', 'delta_y', 'delta_theta']].values
    sysIn = df.loc[:, ['dt', 'left_pwm', 'right_pwm', 'theta_prev']].values

    if not rangeLim is None:
        sysOut = sysOut[rangeLim,:]
        sysIn = sysIn[rangeLim,:]
    return sysOut,sysIn,sysOut_actual

class VectorRegression(sklearn.base.BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        n, m = y.shape
        # Fit a separate regressor for each column of y
        self.estimators_ = [sklearn.base.clone(self.estimator).fit(X, y[:, i])
                               for i in range(m)]
        return self

    def predict(self, X):
        # Join regressors' predictions
        res = [est.predict(X)[:, np.newaxis] for est in self.estimators_]
        return np.hstack(res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get path to data directory')
    parser.add_argument('--datadir', required=True)
    args = parser.parse_args(sys.argv[1:])

    datadir = args.datadir
    if not os.path.isdir(datadir):
        print('invalid DATA_DIR (pass in as argument)')

    dirpath_train = os.path.abspath(os.path.join(datadir, fname_train.split('.')[0]))
    dirpath_test = os.path.abspath(os.path.join(datadir, fname_test.split('.')[0]))
    datafile_train = os.path.join(dirpath_train, fname_train)
    datafile_test = os.path.join(dirpath_test, fname_test)

    df_train = pd.read_csv(datafile_train, engine='python')

    df_test = pd.read_csv(datafile_test, engine='python')

    sysOut_train, sysIn_train, sysOut_actual_train = transformData(df_train, range(0,1000));
    sysOut_test, sysIn_test, sysOut_actual_test = transformData(df_test, range(1000, 2586));

    lasso = Lasso()
    svr_rbf = VectorRegression(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1))
    svr_lin = VectorRegression(SVR(kernel='linear', C=100, gamma='auto'))
    svr_poly = VectorRegression(SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1))
    
    train_loss_history = []
    test_loss_history = []
    model = 'nn'
    if model == 'dt':
        dtr = DecisionTreeRegressor(max_depth=100)
        dtr.fit(sysIn_train, sysOut_train)
        clf = dtr
        title = "Decision Tree"
    elif model == 'nn':
        if model_cached:
            model = load_model('.', model_fname)
        else:
            model = make_model(layers_dims)

        iterations = 50
        epochs = 10
        for it in range(iterations):
            model.fit(sysIn_train, sysOut_train, epochs=epochs, verbose=1, shuffle=False)

            sysOut_pred = model.predict(sysIn_train)
            train_loss = mean_squared_error(sysOut_pred[:, :2], sysOut_train[:, :2])
            train_loss_history.append(train_loss)

            sysOut_pred = model.predict(sysIn_test)
            test_loss = mean_squared_error(sysOut_pred[:, :2], sysOut_test[:, :2])
            test_loss_history.append(test_loss)

            print('train loss: ', train_loss)
            print('test loss: ', test_loss)

        clf = model
        title = 'Feedforward Neural Network'

    sysOut_pred = clf.predict(sysIn_test)

    traj_act = sysOut_test.cumsum(axis=0)
    traj_pred = sysOut_pred.cumsum(axis=0)

    # Plot the results
    plt.figure()
    plt.plot(traj_act[:,0],traj_act[:,1])

    plt.plot(traj_pred[:,0],traj_pred[:,1])
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("{} Simulation Results".format(title))
    plt.legend(["Actual", "Predicted"])
    plt.show()

    if model == 'nn':
        save_model(model, '.', model_fname)
    
    # plot learning curve
    ep_range = range(0, iterations)
    plt.plot(ep_range, train_loss_history)
    plt.plot(ep_range, test_loss_history)
    plt.title('Learning curve')
    plt.xlabel('iteration (30 epochs/iteration)')
    plt.ylabel('RMSE')
    plt.legend(["training error", "testing error"])
    plt.show()
