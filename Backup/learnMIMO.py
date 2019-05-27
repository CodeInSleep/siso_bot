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

import csv

fname_train = 'trial_1000_0_to_3.csv'
fname_test = 'trial_1000_0_to_3.csv'

def truncate(num, digits):
    stepper = pow(10.0, digits)
    return math.trunc(num*stepper)/stepper

def diff(df, fields):
    print(df)
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
    df.loc[:, 'model_pos_x'] = df.loc[:, 'model_pos_x']*1000
    df.loc[:, 'model_pos_x'] = df.loc[:, 'model_pos_x'].apply(truncate, args=(3,))
    df.loc[:, 'model_pos_y'] = df.loc[:, 'model_pos_y']*1000
    df.loc[:, 'model_pos_y'] = df.loc[:, 'model_pos_y'].apply(truncate, args=(3,))


    df.loc[:, 'delta_theta'] = diffTheta(df.loc[:, 'theta'], df.loc[:, 'theta_prev'])

    start_state = df.loc[:, ['model_pos_x', 'model_pos_y', 'theta']].head(1)
    df.loc[:, 'dt'] = df.loc[:, 'sim_time'].diff()
    df.loc[:, 'delta_x'] = df.loc[:, 'model_pos_x'].diff()
    df.loc[:, 'delta_y'] = df.loc[:, 'model_pos_y'].diff()
    df.drop(df.head(1).index, inplace=True)

    sysOut = df.loc[:, ['delta_x', 'delta_y', 'delta_theta']].values
    sysIn = df.loc[:, ['dt', 'left_pwm', 'right_pwm']].values

    if not rangeLim is None:
        sysOut = sysOut[rangeLim,:]
        sysIn = sysIn[rangeLim,:]
    return sysOut,sysIn

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

    sysOut_train, sysIn_train = transformData(df_train, range(0,16666));
    sysOut_test, sysIn_test = transformData(df_test, range(16667, 25700));

    lasso = Lasso()
    svr_rbf = VectorRegression(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1))
    svr_lin = VectorRegression(SVR(kernel='linear', C=100, gamma='auto'))
    svr_poly = VectorRegression(SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1))
    dtr = DecisionTreeRegressor(max_depth=100)
    clf = dtr
    clf.fit(sysIn_train, sysOut_train)

    sysOut_pred = clf.predict(sysIn_test)

    traj_act = sysOut_test.cumsum(axis=0)

    traj_pred = sysOut_pred.cumsum(axis=0)

    # Plot the results
    plt.figure()
    plt.plot(traj_act[:,0],traj_act[:,1])

    plt.plot(traj_pred[:,0],traj_pred[:,1])
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Decision Tree Regressor - Simulation Results")
    plt.legend(["Actual", "Predicted"])

    plt.show()
