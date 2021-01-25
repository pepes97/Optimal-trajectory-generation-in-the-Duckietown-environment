from matplotlib import pyplot as plt
from matplotlib import cm
from Frenet import Frenet
import numpy as np


def plot_longitudinal_paths(paths: [Frenet]):
    max_cd = max(path.cd for path in paths)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    min_cd = max_cd
    min_path = None
    for path in paths:
        if path.cd < min_cd:
            min_cd = path.cd
            min_path = path
        color = 'k'
        if path.cd > 1000:
            color = 'gray'
        plt.plot(path.t, path.d, color)#c=cm.gnuplot(path.cd / 25))
    plt.plot(min_path.t, min_path.d, '-g', linewidth=2)
    plt.show()

def plot_longitudinal_paths_lst(path_lst: [[Frenet]]):
    max_cd = 1e6
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for plst in path_lst:
        min_cd = max_cd
        min_path = None
        for path in plst:
            if path.cd < min_cd:
                min_cd = path.cd
                min_path = path
            color = 'k'
            if path.cd > 1000:
                color = 'gray'
            plt.plot(path.t, path.d, color)#c=cm.gnuplot(path.cd / 25))
        plt.plot(min_path.t, min_path.d, '-g', linewidth=2)
    plt.show()

def plot_unicycle_path(pos_vect: np.array):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(pos_vect[0, :], pos_vect[1, :])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Unicycle path')
    ax.legend(['Unicycle path'])
    plt.grid()
    plt.show()
            
