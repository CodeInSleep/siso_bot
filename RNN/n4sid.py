from __future__ import division
from past.utils import old_div
import os
import sys
import argparse
import math
import numpy as np
import pandas as pd
from SIPPY import *
from SIPPY import functionset as fset
from SIPPY import functionsetSIM as fsetSIM
import matplotlib.pyplot as plt

fname = 'trial_1000_0_to_3.csv'
modeltype = 'N4SID'

if __name__ == "__main__":
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
    
    u = df[['left_pwm', 'right_pwm']].values
    theta_tmp = df[['theta']].values
    y = df[['model_pos_x','model_pos_y','theta']].values
    #y = np.hstack((y,encoded_theta))
    u_tr = u[0:500,:].T
    y_tr = y[0:500,:].T
    u_ts = u[0:2000,:].T
    y_ts = y[0:2000,:].T
    if modeltype == 'N4SID':
        sys = system_identification(y_tr,u_tr,'N4SID',SS_fixed_order=3)
        xid, yid=fsetSIM.SS_lsim_process_form(sys.A,sys.B,sys.C,sys.D,u_ts,sys.x0)
        RMSE = np.mean(np.sqrt(np.square(y_ts[0]-yid[0] +  np.square(y_ts[1]-yid[1]))))
        print("RMSE", RMSE)
        plt.close("all")
        plt.figure(1)
        plt.plot(y_ts[0],y_ts[1])
        plt.plot(yid[0],yid[1])
        plt.ylabel("y")
        plt.grid()
        plt.xlabel("x")
        plt.title("N4SID Baseline Model for Comparison")
        plt.legend(['Original system','Identified system'])
        plt.show()
# from __future__ import division
# from past.utils import old_div
# #Checking path to access other files
# try:
#     from SIPPY import *
# except ImportError:
#     import sys, os
#     sys.path.append(os.pardir)
#     from SIPPY import *

# import numpy as np
# from SIPPY import functionset as fset
# from SIPPY import functionsetSIM as fsetSIM
# import matplotlib.pyplot as plt

# ts=1.0

# A = np.array([[0.89, 0.],[0., 0.45]])
# B = np.array([[0.3],[2.5]])
# C = np.array([[0.7,1.]])
# D = np.array([[0.0]])


# tfin = 500
# npts = int(old_div(tfin,ts)) + 1
# Time = np.linspace(0, tfin, npts)

# #Input sequence
# U=np.zeros((1,npts))
# U[0]=fset.PRBS_seq(npts,0.05)

# ##Output
# x,yout = fsetSIM.SS_lsim_process_form(A,B,C,D,U)

# #measurement noise
# noise=fset.white_noise_var(npts,[0.15])

# #Output with noise
# y_tot=yout+noise

# ##System identification
# method='N4SID'
# sys_id=system_identification(y_tot,U,method,SS_fixed_order=2)
# xid,yid=fsetSIM.SS_lsim_process_form(sys_id.A,sys_id.B,sys_id.C,sys_id.D,U,sys_id.x0)

# plt.close("all")
# plt.figure(1)
# plt.plot(Time,y_tot[0])
# plt.plot(Time,yid[0])
# plt.ylabel("y_tot")
# plt.grid()
# plt.xlabel("Time")
# plt.title("Ytot")
# plt.legend(['Original system','Identified system, '+method])

# plt.figure(2)
# plt.plot(Time,U[0])
# plt.ylabel("input")
# plt.grid()
# plt.xlabel("Time")
# plt.show()