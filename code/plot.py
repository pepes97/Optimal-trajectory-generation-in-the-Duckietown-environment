from matplotlib import pyplot as plt
from matplotlib import cm
from Frenet import Frenet
import numpy as np


def plot_lateral_paths(paths: [Frenet]):
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
        plt.plot(path.t, path.d, color, linewidth=0.5)#c=cm.gnuplot(path.cd / 25))
    plt.plot(min_path.t, min_path.d, '-g', linewidth=1)
    plt.show()

def plot_lateral_paths_lst(path_lst: [[Frenet]]):
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
            if path.cd > 30:
                color = 'gray'
            plt.plot(path.t, path.d, color, linewidth= 0.5)#c=cm.gnuplot(path.cd / 25))
            plt.plot(path.t[0], path.d[0], "og")
            plt.xlabel("t/s")
            plt.ylabel("d/m")
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

def plot_trajectory2D(path: np.array):
    assert path.shape[0] == 2
    plot_unicycle_path(path)

def plot_unicycle_evolution(gpose, fpose, path, T_end):
    fig, axs = plt.subplots(2)
    axs[0].plot(gpose[0, :], gpose[1, :], 'r-')
    path_x = np.arange(0, T_end, 0.05)
    axs[0].plot(path_x, np.array([path.compute_pt(i) for i in path_x]))
    axs[0].legend(['Unicycle path', 'Trajectory'])
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

    axs[1].plot(fpose[0,:], fpose[1,:])
    axs[1].set_xlabel('r')
    axs[1].set_ylabel('d')
    
    plt.show()

import matplotlib.animation as animation
def plot_unicycle_evolution_animated(gpose, fpose, est_pose, path, T_end):
    fig, axs = plt.subplots(2)

    line, = axs[0].plot(gpose[0, :], gpose[1, :])
    scat = axs[0].scatter(est_pose[0], path.compute_pt(est_pose[0]), c='r')
    path_x = np.arange(0, T_end, 0.05)
    axs[0].plot(path_x, np.array([path.compute_pt(i) for i in path_x]))

    line2, = axs[1].plot(fpose[0,:], fpose[1,:])
    axs[1].set_xlabel('r')
    axs[1].set_ylabel('d')
    def animate(i):
        line.set_xdata(gpose[0, :i])
        line.set_ydata(gpose[1, :i])
        line2.set_xdata(fpose[0, :i])
        line2.set_ydata(fpose[1, :i])
        try:
            scat.set_offsets([est_pose[i], path.compute_pt(est_pose[i])])
        except:
            ...
        return [line, line2, scat]

    ani = animation.FuncAnimation(
        fig, animate, frames=gpose.shape[1], interval=100)
    axs[0].axis('equal')
    axs[1].axis('equal')
    plt.tight_layout()
    plt.show()
    #ani.save('evol.gif')

def plot_unicycle_evolution_2D_animated(gpose, fpose, est_pose, path, t):
    fig, axs = plt.subplots(2)
    # Plot path
    path_vect = np.zeros((2, t.shape[0]))
    for i in range(t.shape[0]):
        path_i = path.compute_pt(t[i])
        path_vect[:, i] = path_i.T
    axs[0].plot(path_vect[0, :], path_vect[1, :])
    # Plot robot global pose
    line1, = axs[0].plot(gpose[0, :], gpose[1, :])
    line2, = axs[1].plot(fpose[0, :], fpose[1, :])
    # Plot robot projection on path
    proj_pose = path.compute_pt(est_pose[0])
    scat = axs[0].scatter(proj_pose[0], proj_pose[1], c='r')

    # Animation function
    def animate(i):
        line1.set_xdata(gpose[0, :i])
        line1.set_ydata(gpose[1, :i])
        line2.set_xdata(fpose[0, :i])
        line2.set_ydata(fpose[1, :i])
        proj_pose = path.compute_pt(est_pose[i])
        scat.set_offsets([proj_pose[0], proj_pose[1]])
        return [line1, line2, scat]

    ani = animation.FuncAnimation(
        fig, animate, frames=t.shape[0], interval=100, repeat=False)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[1].set_xlabel('r')
    axs[1].set_ylabel('d')
    axs[0].axis('equal')
    axs[1].axis('equal')
    plt.tight_layout()
    plt.show()
    ani.save('evol2D.gif')
    
    
    
    
            
def plot_longitudinal_paths_lst(path_lst: [[Frenet]]):
    max_cv = 1e6
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for plst in path_lst:
        min_cv = max_cv
        min_path = None
        for path in plst:
            if path.cv < min_cv:
                min_cv = path.cv
                min_path = path
            color = 'k'
            if path.cv > 30:
                color = 'gray'
            plt.plot(path.t, path.dot_s, color, linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            plt.plot(path.t[0], path.dot_s[0], "og")
            plt.xlabel("t/s")
            plt.ylabel("$\dot{s}$/m/s")
        plt.plot(min_path.t, min_path.dot_s, '-g', linewidth=2)
    plt.show()


def plot_longitudinal_paths(paths: [Frenet]):
    max_cv = max(path.cv for path in paths)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    min_cv = max_cv
    min_path = None
    for path in paths:
        if path.cv < min_cv:
            min_cv = path.cv
            min_path = path
        color = 'k'
        if path.cv > 1000:
            color = 'gray'
        plt.plot(path.t, path.dot_s, color)#c=cm.gnuplot(path.cd / 25))
    plt.plot(min_path.t, min_path.dot_s, '-g', linewidth=2)
    plt.show()
    
    
    



def plot_lateral_paths_lst_ctot(path_lst: [[Frenet]]):
    max_ctot = 1e6
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for plst in path_lst:
        min_ctot = max_ctot
        min_path = None
        for path in plst:
            if path.ctot < min_ctot:
                min_ctot = path.ctot
                min_path = path
            color = 'k'
            if path.ctot > 30 and path.ctot < 1000:
                color = 'gray'
            if path.ctot < 30:
                color = 'k'
                plt.plot(path.t, path.d, color, linewidth = 0.5)#c=cm.gnuplot(path.cd / 25))
                plt.plot(path.t[0], path.d[0], "og")
                plt.xlabel("t/s")
                plt.ylabel("d/m")
        plt.plot(min_path.t, min_path.d, '-g', linewidth=2)
    plt.show()

            
def plot_longitudinal_paths_lst_ctot(path_lst: [[Frenet]]):
    max_ctot = 1e6
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for plst in path_lst:
        min_ctot = max_ctot
        min_path = None
        for path in plst:
            if path.ctot < min_ctot:
                min_ctot = path.ctot
                min_path = path
            color = 'k'
            if path.ctot > 30:
                color = 'gray'
            plt.plot(path.t, path.dot_s, color, linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            plt.plot(path.t[0], path.dot_s[0], "og")
            plt.xlabel("t/s")
            plt.ylabel("$\dot{s}$/m/s")
        plt.plot(min_path.t, min_path.dot_s, '-g', linewidth=2)
    plt.show()

def plot_following_paths_lst_ctot(path_lst: [[Frenet]], target: Frenet):
    max_ctot = 1e6
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for plst in path_lst:
        min_ctot = max_ctot
        min_path = None
        for path in plst:
            if path.ctot < min_ctot:
                min_ctot = path.ctot
                min_path = path
            color = 'k'
            if path.ctot > 30:
                color = 'gray'
            plt.plot(path.t, path.s, color, linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            plt.plot(path.t[0], path.s[0], "og")
            plt.xlabel("t/s")
            plt.ylabel("s/m")
        plt.plot(min_path.t, min_path.s, '-g', linewidth=2)
        plt.plot(target.t, target.s, '-b', linewidth=2)
    plt.show()