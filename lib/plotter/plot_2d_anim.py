"""plot_2d_anim.py
"""

from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np

from .utils import *
from ..logger import SimulationDataStorage, SimData

def plot_2d_simulation_anim(data: SimulationDataStorage) -> plt.Figure:
    fig, gs = generate_4_2_layout()
    t = data.t
    # Plot on the first row, the target and robot trajectories
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_title('Global view')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    target_traj = data.get(SimData.trajectory_2d)
    robot_traj  = data.get(SimData.robot_pose)
    ax0.plot(target_traj[0, :], target_traj[1, :], label='Path')
    robot_line, = ax0.plot(robot_traj[0, :], robot_traj[1, :], label='Robot path')

    # Plot on the second row, frenet coordinates evolution
    robot_fpose = data.get(SimData.robot_frenet_pose)
    ax1_0 = fig.add_subplot(gs[1, 0])
    ax1_0.set_title('s(t)')
    ax1_0.set_xlabel('t')
    ax1_0.set_ylabel('s')
    s_line, = ax1_0.plot(t, robot_fpose[0, :])
    ax1_1 = fig.add_subplot(gs[1, 1])
    ax1_1.set_title('d(t)')
    ax1_1.set_xlabel('t')
    ax1_1.set_ylabel('d')
    d_line, = ax1_1.plot(t, robot_fpose[1, :])

    # Plot on third row, frenet errors
    error = data.get(SimData.error)
    ax2_0 = fig.add_subplot(gs[2, 0])
    ax2_0.set_title('error_s(t)')
    ax2_0.set_xlabel('t')
    ax2_0.set_ylabel('s')
    es_line, = ax2_0.plot(t, error[0, :])
    ax2_1 = fig.add_subplot(gs[2, 1])
    ax2_1.set_title('error_d(t)')
    ax2_1.set_xlabel('t')
    ax2_1.set_ylabel('d')
    ed_line, = ax2_1.plot(t, error[1, :])

    # Plot on fourth row, control terms
    u = data.get(SimData.control)
    ax3_0 = fig.add_subplot(gs[3, 0])
    ax3_0.set_title('v')
    ax3_0.set_xlabel('t')
    ax3_0.set_ylabel('u')
    u0_line, = ax3_0.plot(t, u[0, :])
    ax3_1 = fig.add_subplot(gs[3, 1])
    ax3_1.set_title('w')
    ax3_1.set_xlabel('t')
    ax3_1.set_ylabel('u')
    u1_line, = ax3_1.plot(t, u[1, :])

    def animate(i):
        robot_line.set_xdata(robot_traj[0, :i])
        robot_line.set_ydata(robot_traj[1, :i])
        s_line.set_xdata(t[:i])
        s_line.set_ydata(robot_fpose[0, :i])
        d_line.set_xdata(t[:i])
        d_line.set_ydata(robot_fpose[1, :i])
        es_line.set_xdata(t[:i])
        es_line.set_ydata(error[0, :i])
        ed_line.set_xdata(t[:i])
        ed_line.set_ydata(error[1, :i])
        u0_line.set_xdata(t[:i])
        u0_line.set_ydata(u[0, :i])
        u1_line.set_xdata(t[:i])
        u1_line.set_ydata(u[1, :i])
        return [robot_line, s_line, d_line, es_line, ed_line, u0_line, u1_line]
    ani = animation.FuncAnimation(fig, animate, frames=t.shape[0], interval=20, blit=True)
    plt.show()
    return fig
    

def plot_2d_simulation_bot_xy_anim(data: SimulationDataStorage) -> plt.Figure:
    fig, ax0 = plt.subplots()
    t = data.t
    # Plot on the first row, the target and robot trajectories
    ax0.set_title('Global view')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    # TODO

