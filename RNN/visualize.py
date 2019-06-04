import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pdb

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '#663333', '#FFA500', '#4B0082', '0.5']
color_cycler = cycle(colors)

def plot_arrows(trial, ax, tid, plt_arrow=False, color=None):
    #pdb.set_trace()
    r = 1
    trial_x = [x for x,y,theta in trial]
    trial_y = [y for x,y,theta in trial]

    if color is None:
        color=next(color_cycler)

    if plt_arrow:
        trial_u = [r * math.cos(theta) for x,y,theta in trial]
        trial_v = [r * math.sin(theta) for x,y,theta in trial]
        ax.quiver(trial_x, trial_y, trial_u, trial_v, pivot='mid', color=color, gid=tid)
    else:
        ax.plot(trial_x, trial_y, color=color, gid=tid)

def visualize_3D(arr_3d, ax, plt_arrow=False, color=None):
    for tid, trial in enumerate(arr_3d):
        plot_arrows(trial, ax, tid, plt_arrow=plt_arrow, color=color)

if __name__ == '__main__':
    test_pred = np.load('test_predictions.npy')
    test_ground = np.load('test_ground.npy')
    train_pred = np.load('train_predictions.npy')
    train_ground = np.load('train_ground.npy')
    pdb.set_trace()

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    pdb.set_trace()
    #visualize_3D(train_pred, ax1)
    #visualize_3D(train_ground, ax2)
    ax1.plot(train_pred.flatten())
    ax1.set_title('train prediction')
    ax2.plot(train_ground.flatten())
    ax2.set_title('ground truth')
    plt.show()
