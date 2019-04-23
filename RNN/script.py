import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pdb

fname = 'data.csv'
# create a new dataset by adding past values as features at each time step
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

if __name__ == '__main__':
   if not os.path.isdir(os.environ['SISO_DATA_DIR']):
        print('invalid DATA_DIR (set in ~/sisobot/export_path.sh')

    dirpath = os.environ['SISO_DATA_DIR']

    datafile = os.path.join(dirpath, fname)
    df = pd.read_csv(datafile, engine='python')
    df = df[['model_pos_x']]
    # def plot_vec(df):
    # 	dim = df.shape[1]
    # 	assert (dim == 3), "not 3 dimensions"

    # 	series = [df[col].values for col in df.columns]
    # 	xs, ys, zs = series
    # 	fig = plt.figure()
    # 	ax = fig.gca(projection='3d')
    # 	ax.plot(xs, ys, zs)

    # def plot_df(df, p_type, fields = []):
    # 	# plotting inputs
    # 	plt.title = p_type

    # 	if len(fields) == 3:
    # 		plot_vec(df[fields])
    # 	else:
    # 		plot = plt.figure()
    # 		for f in fields:
    # 			if f in df:
    # 				df[f].plot()
    # 			else:
    # 				print('field %s does not exist in df' % f)

    # plot_df(df, 'inputs', ['left_wheel_vel_y', 'right_wheel_vel_y'])
    # plot_df(df, 'outputs', ['model_pos_x', 'model_pos_y', 'model_pos_z'])
    # plt.show()


    df = df.values
    df = df.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(df)

    # split into train and test sets
    train_size = int(len(df) * 0.67)
    test_size = len(df) - train_size
    train, test = df[0:train_size,:], df[train_size:len(df),:]
    look_back = 1

    plt.plot(df.flatten(), color='blue')
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(df)
    trainPredictPlot[:, :] = 0
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df)
    testPredictPlot[:, :] = 0
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(df)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(trainPredict.flatten(), color='green')
    plt.plot(testPredict.flatten(), color='red')
    plt.show()


