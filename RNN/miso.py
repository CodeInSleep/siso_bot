import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape
from keras.layers import Embedding
from keras.layers import LSTM, SimpleRNN
from keras.utils import plot_model
from keras.initializers import Identity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pdb

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# number of inputs (left, right velocity values)
p = 2

# number of states (x, y, theta, x_vel, y_vel)
J = 5

batch_size = 1
model = Sequential()
model.add(Dense(J, input_shape=(p,), activation='tanh', name='hidden_layer'))
model.add(Reshape((1, J), name='reshape_layer'))
# TODO: tune regularizer hyperparameter
model.add(SimpleRNN(J, batch_input_shape=(batch_size, 1, J), name='dynamic_layer',
	kernel_initializer=Identity(J)))
model.compile(loss='mean_squared_error', optimizer='adam')


