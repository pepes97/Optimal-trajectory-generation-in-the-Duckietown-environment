"""plot_2d_anim.py
"""

from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np

from .utils import *
from ..logger import SimulationDataStorage, SimData
from ..transform import homogeneous_transform
from .plot_2d import *

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

def plot_2d_bot_anim(data: SimulationDataStorage) -> plt.Figure:
    def animate(i):
        pass
    pass
    # TODO

def plot_2d_planner_obstacles_anim(data_storage:SimulationDataStorage, trajectory, robot, sensor, obstacle_lst, t_vect) -> plt.Figure:
    fig, ax = generate_1_1_layout()
    # Plot trajectory
    ax.axis('equal')
    trajectory_line = plot_trajectory(ax, trajectory, t_vect)
    unicycle_poly   = plot_unicycle(ax, robot)
    # Extract data from data_storage
    rpose = data_storage.get(SimData.robot_pose)
    planner_path = data_storage.get(SimData.planner)
    planner_line, = ax.plot(planner_path[0, :], planner_path[1, :], 'g')
    sensor_line   = plot_proximity_sensor(ax, sensor, robot)
    
    def get_obstacle_coordinates(obst_lst: [Obstacle]):
        x_lst = []
        y_lst = []
        for o in obst_lst:
            x_lst.append(o.position[0])
            y_lst.append(o.position[1])
        return np.array([x_lst, y_lst])


    measure_obst = sensor.sense(rpose=rpose[:, 0])
    measure_pts = get_obstacle_coordinates(measure_obst)
    measure_scat = ax.scatter(measure_pts[0, :], measure_pts[1, :], facecolors='none', edgecolors='r')
    plot_obstacles(ax, obstacle_lst, c='k', marker='x')
    # Animation callback
    def animate(i):
        # Center camera to robot
        ax.set_xlim(xmin=rpose[0, i] - 5, xmax=rpose[0, i] + 5)
        ax.set_ylim(ymin=rpose[1, i] - 5, ymax=rpose[1, i] + 5)
        ax.figure.canvas.draw()

        # Sensor
        p1, p2 = compute_sensor_vertices(sensor, rpose[:, i])
        sensor_line.set_xdata([p1[0], rpose[0, i], p2[0]])
        sensor_line.set_ydata([p1[1], rpose[1, i], p2[1]])
        # Get new robot vertices
        unicycle_poly.set_xy(compute_unicycle_vertices(rpose[:, i]))
        # Plot measure 
        measure_obst = sensor.sense(rpose=rpose[:, i])
        logger.debug(f'time_{i}: sensor found {len(measure_obst)} obstacles')
        measure_pts = get_obstacle_coordinates(measure_obst)
        measure_scat.set_offsets(np.c_[measure_pts[0, :], measure_pts[1, :]])
        # Plot next 30 (at most) planner points
        planner_line.set_xdata(planner_path[0, i:i+30])
        planner_line.set_ydata(planner_path[1, i:i+30])
        return [unicycle_poly, planner_line,sensor_line, measure_scat]
    ani = animation.FuncAnimation(fig, animate, frames=t_vect.shape[0], interval=30, blit=False)
    plt.show()
    return fig, ani

def plot_2d_planner_moving_obstacles_anim(data_storage, trajectory,robot, sensor, obstacle_lst, t_vect):
    fig, ax = generate_1_1_layout()
    # Plot trajectory
    ax.axis('equal')
    trajectory_line = plot_trajectory(ax, trajectory, t_vect)
    unicycle_poly   = plot_unicycle(ax, robot)
    # Extract data from data_storage
    rpose = data_storage.get(SimData.robot_pose)
    planner_path = data_storage.get(SimData.planner)
    planner_line, = ax.plot(planner_path[0, :], planner_path[1, :], 'g')
    sensor_line   = plot_proximity_sensor(ax, sensor, robot)
    obstacles_poly = [plot_moving_obstacle(ax, obs) for obs in obstacle_lst]
    
    def get_obstacle_coordinates(obst_lst: [Obstacle]):
        x_lst = []
        y_lst = []
        for o in obst_lst:
            x_lst.append(o.position[0])
            y_lst.append(o.position[1])
        return np.array([x_lst, y_lst])


    measure_obst = sensor.sense(rpose=rpose[:, 0])
    measure_pts = get_obstacle_coordinates(measure_obst)
    measure_scat = ax.scatter(measure_pts[0, :], measure_pts[1, :], s=400, facecolors='none', edgecolors='r')
    # Animation callback
    def animate(i):
        # Center camera to robot
        ax.set_xlim(xmin=rpose[0, i] - 5, xmax=rpose[0, i] + 5)
        ax.set_ylim(ymin=rpose[1, i] - 5, ymax=rpose[1, i] + 5)
        ax.figure.canvas.draw()
        # Sensor
        p1, p2 = compute_sensor_vertices(sensor, rpose[:, i])
        sensor_line.set_xdata([p1[0], rpose[0, i], p2[0]])
        sensor_line.set_ydata([p1[1], rpose[1, i], p2[1]])
        # Get new robot vertices
        unicycle_poly.set_xy(compute_unicycle_vertices(rpose[:, i]))
        # Plot measure 
        sensor.step_obstacle(t_vect[i])
        measure_obst = sensor.sense(rpose=rpose[:, i])
        logger.debug(f'time_{i}: sensor found {len(measure_obst)} obstacles')
        measure_pts = get_obstacle_coordinates(measure_obst)
        for poly,obs in zip(obstacles_poly,obstacle_lst):
            poly.set_xy(compute_unicycle_vertices(obs.pose))
        measure_scat.set_offsets(np.c_[measure_pts[0, :], measure_pts[1, :]])
        # Plot next 30 (at most) planner points
        planner_line.set_xdata(planner_path[0, i:i+30])
        planner_line.set_ydata(planner_path[1, i:i+30])
        return [unicycle_poly, planner_line,sensor_line, measure_scat, obstacles_poly]
    ani = animation.FuncAnimation(fig, animate, frames=t_vect.shape[0], interval=30, blit=False)
    plt.show()
    return fig, ani