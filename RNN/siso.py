import numpy as np
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, SimpleRNN, TimeDistributed, LSTM
import matplotlib.pyplot as plt
import pdb

def pwm2Vel(pwm_val):
    return -555*np.arctan(7.2*np.pi*(pwm_val-90)/180)+35

def shape_it(arr, time_step, num_features):
    return arr.reshape(len(arr), time_step, num_features)

n_data = 1000
n_train = int(np.floor(n_data*0.7))

features = np.random.rand(n_data,)*180
features = features.astype(np.float32)
targets = pwm2Vel(features)
targets = targets.astype(np.float32)

X_train = features[:n_train].reshape(-1, 1)
y_train = targets[:n_train].reshape(-1, 1)
X_test = features[n_train:].reshape(-1, 1)
y_test = targets[n_train:].reshape(-1, 1)


dt = 0.001
X_train = shape_it(np.concatenate((np.ones(X_train.shape)*dt, X_train), axis=1), 1, 2)
X_test = shape_it(np.concatenate((np.ones(X_test.shape)*dt, X_test), axis=1), 1, 2)

y_train = shape_it(y_train.cumsum(axis=0)*dt, 1, 1)
y_test = shape_it(y_test.cumsum(axis=0)*dt, 1, 1)

batch_size = 1
max_duration = 1
model = Sequential()
model.add(TimeDistributed(Dense(10, input_shape=(2, ), activation='tanh', )))
model.add(TimeDistributed(Dense(10, activation='tanh',)))
model.add(SimpleRNN(4, return_sequences=True, activation=None))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

iterations = 10
epochs = 10

train_loss_history = []
test_loss_history = []
 
for it in range(iterations):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    predictions = model.predict(X_train)
    train_loss_history.append(np.sqrt(mean_squared_error(np.squeeze(predictions, axis=1), np.squeeze(y_train, axis=1))/len(predictions)))

epoch_range = range(iterations)
plt.plot(epoch_range, train_loss_history)
plt.show()
