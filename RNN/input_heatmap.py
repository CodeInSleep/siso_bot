import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
import pdb
from numpy import cos, sin, arctan2
import matplotlib.pyplot as plt
from utils import decode_angles, plot_target_angles, save_model, angle_dist, make_model, load_model, load_obj, convert_to_inference_model


fname = 'trial_1000_0_to_3.csv'

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

    # X_train = np.load(os.path.join(dirpath, 'X_train.npy'))
    # X_test = np.load(os.path.join(dirpath, 'X_test.npy'))
    train_trial_names = load_obj(dirpath, 'train_trial_names')
    test_trial_names = load_obj(dirpath, 'test_trial_names')

    l = np.zeros((1,0))
    r = np.zeros((1,0))
    for name in train_trial_names:
        name_parts = name.split('_')
        l_i = float(name_parts[1])
        r_i = float(name_parts[3])
       
        l = np.hstack((l,np.array([[l_i]])))
        r = np.hstack((r,np.array([[r_i]])))
    l = np.asarray(l)[0,:]
    r = np.asarray(r)[0,:]
    fig, ax = plt.subplots()
    h = ax.hist2d(l,r, bins=10)
    plt.colorbar(h[3], ax=ax)
    plt.show()
