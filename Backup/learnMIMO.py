import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
from numpy import cos, sin, arctan2
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

import csv

fname = 'test_run.csv'

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
    if fname == 'trial_2000_0_to_3':
        df = df.drop(['blank1', 'blank2'], axis=1)

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
    sysIn = df.loc[:, ['dt', 'left_pwm', 'right_pwm', 'theta_prev']].values

    X_train, X_test, y_train, y_test = train_test_split(sysIn, sysOut, test_size=0.33, shuffle=False)
    clf = DecisionTreeRegressor()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(df.loc[:, 'model_pos_x'])


    traj_act = sysOut.cumsum(axis=0)
    # traj_x_test = y_test[:,0].cumsum()
    # traj_y_test = y_test[:,1].cumsum()
    traj_test = y_test.cumsum(axis=0)
    traj_pred = y_pred.cumsum(axis=0)
    traj_train = y_train.cumsum(axis=0)

    # Plot the results
    plt.figure()
    plt.plot(traj_act[:,0],traj_act[:,1])
    plt.plot(traj_train[:,0],traj_train[:,1])
    
    plt.plot(traj_test[:,0],traj_test[:,1])
    plt.plot(traj_pred[:,0],traj_pred[:,1])
    
    plt.legend(["Actual","Train", "Test", "Predicted"])

    plt.show()
