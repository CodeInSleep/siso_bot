import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '#663333', '#FFA500', '#4B0082', '0.5']
color_cycler = cycle(colors)

def plot_arrows(trial):
    r = 1
    trial_x = [x for x,y,theta in trial]
    trial_y = [y for x,y,theta in trial]
    trial_u = [r * math.cos(math.radians(theta)) for x,y,theta in trial]
    trial_v = [r * math.sin(math.radians(theta)) for x,y,theta in trial]
    plt.quiver(trial_x, trial_y, trial_u, trial_v, pivot='mid', color=color_cycler.next())

test_pred = np.load('test_predictions.npy')
test_ground = np.load('test_ground.npy')
train_pred = np.load('train_predictions.npy')
train_ground = np.load('train_ground.npy')

for ground, pred in zip(train_pred, train_ground):
    plot_arrows(ground)
    plot_arrows(pred)
    plt.show()
