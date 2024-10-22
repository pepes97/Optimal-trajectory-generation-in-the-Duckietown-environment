"""test_plot.py
"""

import logging
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from .config import SimulationConfiguration
from .config import DefaultSimulationParameters as dsp
from ..logger import SimulationDataStorage, SimData
from ..trajectory import QuinticTrajectory2D, CircleTrajectory2D, SplineTrajectory2D
from ..transform import FrenetGNTransform
from ..controller import FrenetIOLController
from ..plotter import *
from ..sensor import ProximitySensor

logger = logging.getLogger(__name__)

def test_plot_unicycle(*args, **kwargs):
    plot_flag = False
    store_plot = None
    if 'plot' in kwargs:
        plot_flag = kwargs['plot']
    if 'store_plot' in kwargs:
        store_plot = kwargs['store_plot']
    sim_config = SimulationConfiguration(**kwargs)
    # Extract key objects from configuration object
    t_vect, robot, trajectory, transformer, controller, planner = sim_config.get_elements()
    data_storage = SimulationDataStorage(t_vect)
    data_storage.add_argument(SimData.robot_pose)

    sensor = ProximitySensor(10, np.pi/8)
    sensor.attach(robot)

    data_storage = __simulate_experiment(sim_config, data_storage, trajectory, robot, transformer, controller, planner)

    def __plot_fn(store: str=None):
        fig, ax = generate_1_1_layout()
        ax.axis('equal')
        trajectory_line = plot_trajectory(ax, trajectory, t_vect)
        unicycle_poly = plot_unicycle(ax, robot)
        sensor_line   = plot_proximity_sensor(ax, sensor, robot)

        # Extract data from data_storage
        rpose = data_storage.get(SimData.robot_pose)
        
        def animate(i):
            # Center camera to robot
            ax.set_xlim(xmin=rpose[0, i] - 10, xmax=rpose[0, i] + 10)
            ax.set_ylim(ymin=rpose[1, i] - 10, ymax=rpose[1, i] + 10)
            ax.figure.canvas.draw()
            
            
            # Get new sensor's points 
            p1, p2 = compute_sensor_vertices(sensor, rpose[:, i])
            sensor_line.set_xdata([p1[0], rpose[0, i], p2[0]])
            sensor_line.set_ydata([p1[1], rpose[1, i], p2[1]])
            # Get new robot vertices
            unicycle_poly.set_xy(compute_unicycle_vertices(rpose[:, i]))
            return [unicycle_poly, sensor_line]
        ani = animation.FuncAnimation(fig, animate, frames=t_vect.shape[0], interval=30, blit=False)
        if store is not None:
            ani.save(store)
        plt.show()
    if plot_flag:
        __plot_fn(store_plot)
    return data_storage

def __simulate_experiment(sim_config, data_storage, trajectory, robot, transformer, controller, planner) -> SimulationDataStorage:
    # Simulation loop
    t_vect = sim_config.get_time_vect()
    robot_p = robot.p

    robot_dp = np.zeros(3)
    u = np.zeros(2)
    for i in range(sim_config.get_simulation_length()):
         # Estimate frenet frame
        est_pt = transformer.estimatePosition(trajectory, robot_p)
        # Robot pose in frenet
        robot_fpose = transformer.transform(robot_p)
        # Robot velocity in frenet (need only p_dot and d_dot)
        robot_fdp   = transformer.transform(robot_dp)[0:2]

        # Compute error and derror
        target_pos = trajectory.compute_pt(t_vect[i])
        target_fpos = transformer.transform(target_pos)
        target_dpos = trajectory.compute_first_derivative(t_vect[i])
        target_fdpos = transformer.transform(target_dpos)
        error = target_fpos - robot_fpose[0:2]
        derror = target_fdpos - robot_fdp

        # Get path curvature at estimate
        curvature = trajectory.compute_curvature(est_pt)

        # Compute control
        u = controller.compute(robot_fpose, error, derror, curvature)

        # Step the unicycle
        robot_p, robot_dp = robot.step(u, dsp.dt)

        # log data
        data_storage.set('robot_pose', robot_p, i)
    return data_storage

def test_plot_planner(*args, **kwargs):
    plot_flag = False
    store_plot = None
    if 'plot' in kwargs:
        plot_flag = kwargs['plot']
    if 'store_plot' in kwargs:
        store_plot = kwargs['store_plot']

    sim_config = SimulationConfiguration(**kwargs)
    # Extract key objects from configuration object
    t_vect, robot, trajectory, transformer, controller, planner = sim_config.get_elements()
    data_storage = SimulationDataStorage(t_vect)
    data_storage.add_argument(SimData.robot_pose)
    data_storage.add_argument(SimData.robot_frenet_pose)
    data_storage.add_argument(SimData.control)
    data_storage.add_argument(SimData.trajectory_2d)
    data_storage.add_argument(SimData.target_pos)
    #data_storage.add_argument(SimData.optimal_path)
    data_storage.add_argument('error', 2)
    data_storage.add_argument('derror', 2)

    data_storage = __simulate_experiment_planner(sim_config, data_storage, trajectory, robot, transformer, controller, planner)
 
        
    def __plot_fn(store: str=None):
        # traj_line = plot_trajectory(ax, trajectory, t_vect)
        # robot_poly = plot_unicycle(ax, robot)
        # opt_traj_line, = ax.plot(opt_traj[0, :], opt_traj[1, :], 'g')
        #ax.scatter(transformer.t_vect[0], transformer.t_vect[1])
        fig, ax = generate_1_1_layout()
        ax.axis('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Planner')
        trajectory_line = plot_trajectory(ax, trajectory, t_vect)
        unicycle_poly = plot_unicycle(ax, robot)
        rpose = data_storage.get(SimData.robot_pose)
        #optimal_path = data_storage.get(SimData.optimal_path)
        robot_line, = ax.plot(rpose[0, :], rpose[1, :], label='Robot path')
        #opt_path_line, = ax.plot(optimal_path[0, :], optimal_path[1, :], label='optimal path')
        def animate(i):
            # opt_path_line.set_xdata(opt_path_line[0,:i])
            # opt_path_line.set_ydata(opt_path_line[1,:i])
            robot_line.set_xdata(rpose[0, :i])
            robot_line.set_ydata(rpose[1, :i])
            unicycle_poly.set_xy(compute_unicycle_vertices(rpose[:, i]))
            return [unicycle_poly, robot_line]
    
        ani = animation.FuncAnimation(fig, animate, frames=t_vect.shape[0], interval=50, blit=True)
        if store is not None:
            ani.save(store)
        plt.show()

    if plot_flag:
        __plot_fn(store_plot)
    return data_storage

def __simulate_experiment_planner(sim_config, data_storage, trajectory, robot, transformer, controller, planner) -> SimulationDataStorage:
    t_vect = sim_config.get_time_vect()
    robot.set_initial_pose(np.array([0, -4, 0.0]))
    robot_p = robot.p
    robot_dp = np.zeros(3)
    robot_ddp = np.zeros(3)
    robot_fdp = np.zeros(3)
    robot_fddp = np.zeros(3)
    # Estimate frenet frame
    est_pt = transformer.estimatePosition(trajectory, robot_p)
    # Robot pose in frenet
    robot_fpose = transformer.transform(robot_p)
    robot_fpose = np.array([est_pt, robot_fpose[1], robot_fpose[2]])
    # Robot velocity in frenet (need only p_dot and d_dot)
    robot_fdp = transformer.transform(robot_dp)[:2]
    # Robot acceleration in frenet 
    robot_fddp = transformer.transform(robot_ddp)[:2]
    s0 = (robot_fpose[0],0,0)
    d0 = (robot_fpose[1],0,0)
    planner.initialize(t0 = t_vect[0], p0 = d0, s0 = s0)
    opt_traj = np.zeros((2, len(t_vect)))
    def compute_ortogonal_vect(traj, s):
        t_grad = traj.compute_first_derivative(s)
        t_r = np.arctan2(t_grad[1], t_grad[0])
        return np.array([-np.sin(t_r), np.cos(t_r)])
    
    for i in range(len(t_vect)):
        est_pt = transformer.estimatePosition(trajectory, robot_p)
        robot_fpose = transformer.transform(robot_p)
        robot_fdp = transformer.transform(robot_dp)[:2]
        robot_fddp = transformer.transform(robot_ddp)[:2]
        # Replanning

        dsd = (t_vect[i+1]-t_vect[i])/sim_config.dt if i+1<len(t_vect) else 0.0

        target_s, target_d = planner.replanner(t_vect[i],dsd=dsd) 

        ts, ds = target_s[0], target_d[0]
        dst = dds = target_s[1], target_d[1]
        target_glob = trajectory.compute_pt(ts) + compute_ortogonal_vect(trajectory, ts) * ds
        target_fpos = transformer.transform(target_glob)
        target_dglob = trajectory.compute_first_derivative(ts)
        target_fdpos = transformer.transform(target_dglob)
        # Compute error and derror
        error = target_fpos - robot_fpose[0:2]
        derror = target_fdpos - robot_fdp
        # Get path curvature at estimate
        curvature = trajectory.compute_curvature(est_pt)
        # Compute control
        u = controller.compute(robot_fpose, error, derror, curvature)
         # Step the unicycle
        robot_p, robot_dp = robot.step(u, dsp.dt)
        opt_traj[:, i]= target_glob

        data_storage.set('robot_pose', robot_p, i)
        data_storage.set('robot_frenet_pose', robot_fpose, i)
        data_storage.set('control', u, i)
        data_storage.set('trajectory', target_glob, i)
        data_storage.set('target_pos', target_fpos, i)
        data_storage.set('error', error, i)
        data_storage.set('derror', derror, i)
        #data_storage.set('optimal_path', opt_traj[:, i], i)
    
    return data_storage