import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pdb

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '#663333', '#FFA500', '#4B0082', '0.5']
color_cycler = cycle(colors)

def plot_arrows(trial, ax, tid, plt_arrow=False):
    #pdb.set_trace()
    r = 1
    trial_x = [x for x,y,theta in trial]
    trial_y = [y for x,y,theta in trial]
    if plt_arrow:
        trial_u = [r * math.cos(theta) for x,y,theta in trial]
        trial_v = [r * math.sin(theta) for x,y,theta in trial]
        ax.quiver(trial_x, trial_y, trial_u, trial_v, pivot='mid', color=next(color_cycler), gid=tid)
    else:
        ax.plot(trial_x, trial_y, color=next(color_cycler), gid=tid)

def visualize_3D(arr_3d, ax, plt_arrow=False):
    for tid, trial in enumerate(arr_3d):
        plot_arrows(trial, ax, tid, plt_arrow=plt_arrow)

if __name__ == '__main__':
    test_pred = np.load('test_predictions.npy')
    test_ground = np.load('test_ground.npy')
    train_pred = np.load('train_predictions.npy')
    train_ground = np.load('train_ground.npy')
    pdb.set_trace()

    for ground, pred in zip(train_pred, train_ground):
        plot_arrows(ground)
        plot_arrows(pred)
        plt.show()
