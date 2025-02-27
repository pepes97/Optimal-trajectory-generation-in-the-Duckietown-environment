"""test_control.py
Contains tests for controllers
"""

import logging
import numpy as np

from .config import SimulationConfiguration
from .config import DefaultSimulationParameters as dsp

from ..logger import SimulationDataStorage, SimData, timeprofiledecorator
from ..trajectory import QuinticTrajectory2D, CircleTrajectory2D, SplineTrajectory2D
from ..transform import FrenetGNTransformOld
from ..controller import FrenetIOLController
from ..plotter import *

logger=logging.getLogger(__name__)

@timeprofiledecorator
def __simulate_experiment(sim_config, data_storage, trajectory, robot, transformer, controller, planner):
    def compute_ortogonal_vect(traj, s):
        t_grad = traj.compute_first_derivative(s)
        t_r = np.arctan2(t_grad[1], t_grad[0])
        return np.array([-np.sin(t_r), np.cos(t_r)])
    # Simulation loop
    t_vect = sim_config.get_time_vect()
    robot_p = robot.p

    robot_dp = np.zeros(3)
    robot_ddp = np.zeros(3)
    u = np.zeros(2)
    # Initialization
    est_pt = transformer.estimatePosition(trajectory, robot_p)
    # Robot pose in frenet
    robot_fpose = transformer.transform(robot_p)
    s0 = (robot_fpose[0], 0.0, 0.0)
    d0 = (robot_fpose[1], 0.0, 0.0)
    planner.initialize(t0=t_vect[0], p0=d0, s0=s0)
    for i in range(sim_config.get_simulation_length()):
        # Estimate frenet frame
        est_pt = transformer.estimatePosition(trajectory, robot_p)
        # Robot pose in frenet
        robot_fpose = transformer.transform(robot_p)
        # Robot velocity in frenet (need only p_dot and d_dot)
        robot_fdp  = transformer.transform(robot_dp)[0:2]
        # Robot acceleration in frenet 
        robot_fddp  = transformer.transform(robot_ddp)[0:2]
        # Get replanner step
        pos_s, pos_d = planner.replanner(t_vect[i])
        ts, td = pos_s[0], pos_d[0]
        target_pos = trajectory.compute_pt(ts) + compute_ortogonal_vect(trajectory, ts) * td
        #Compute error
        error = np.array([0, pos_d[0]]) - robot_fpose[0:2]
        derror = np.array([pos_s[1], pos_d[1]])
        # Print check
        v_planner = np.array([pos_s[1],pos_d[1]])
        v_xy = transformer.itransform(v_planner)
        logger.info(f'Planner s, d:{ts, td}')
        logger.info(f'Robot s,d :{robot_fpose[:2]}')
        logger.info(f'Error:{error}')
        # Get curvature
        curvature = trajectory.compute_curvature(est_pt)
        # Compute control
        u = controller.compute(robot_fpose, error, derror, curvature)
        # print(f"Planner s dot :{v_xy[0]} and d dot: {v_xy[1]}")
        # print(f"Controller v: {u[0]} and omega: {u[1]}")
        # Step the unicycle
        robot_p, robot_dp = robot.step(u, dsp.dt)
        # log data
        data_storage.set(SimData.robot_pose, robot_p, i)
        data_storage.set(SimData.robot_frenet_pose, robot_fpose, i)
        data_storage.set(SimData.control, u, i)
        data_storage.set(SimData.trajectory_2d, target_pos, i)
        data_storage.set(SimData.error, error, i)
        data_storage.set(SimData.derror, derror, i)
        data_storage.set(SimData.planner, target_pos, i)
        
    return data_storage


def test_planner_full(*args, **kwargs):
    plot_flag = False
    store_plot = None
    if 'plot' in kwargs:
        plot_flag = kwargs['plot']
    if 'store_plot' in kwargs:
        store_plot = kwargs['store_plot']
    
    plot_velocities = True
    sim_config = SimulationConfiguration(**kwargs)
    # Extract key objects from configuration object
    sim_config.controller = FrenetIOLController(.5, 0.0, 5, 0.0, 0.0)
    t_vect, robot, trajectory, transformer, controller, planner = sim_config.get_elements()
    # Configure SimulationDataStorage
    data_storage = SimulationDataStorage(t_vect)
    data_storage.add_argument(SimData.robot_pose)
    data_storage.add_argument(SimData.robot_frenet_pose)
    data_storage.add_argument(SimData.control)
    data_storage.add_argument(SimData.trajectory_2d)
    data_storage.add_argument(SimData.target_pos)
    data_storage.add_argument(SimData.error)
    data_storage.add_argument(SimData.derror)
    data_storage.add_argument(SimData.planner)

    data_storage = __simulate_experiment(sim_config, data_storage, trajectory,
                                        robot, transformer, controller, planner)

    def __plot_fn(store: str=None):
        fig, ax = generate_1_1_layout()
        # Plot trajectory
        ax.axis('equal')
        trajectory_line = plot_trajectory(ax, trajectory, t_vect)
        unicycle_poly   = plot_unicycle(ax, robot)
        # Extract data from data_storage
        rpose = data_storage.get(SimData.robot_pose)
        planner_path = data_storage.get(SimData.planner)
        planner_line, = ax.plot(planner_path[0, :], planner_path[1, :], 'g')
        # Animation callback
        def animate(i):
            # Center camera to robot
            ax.set_xlim(xmin=rpose[0, i] - 5, xmax=rpose[0, i] + 5)
            ax.set_ylim(ymin=rpose[1, i] - 5, ymax=rpose[1, i] + 5)
            ax.figure.canvas.draw()
            # Get new robot vertices
            unicycle_poly.set_xy(compute_unicycle_vertices(rpose[:, i]))
            # Plot next 30 (at most) planner points
            planner_line.set_xdata(planner_path[0, i:i+30])
            planner_line.set_ydata(planner_path[1, i:i+30])
            return [unicycle_poly, planner_line]
        ani = animation.FuncAnimation(fig, animate, frames=t_vect.shape[0], interval=30, blit=True, repeat = False)
        #ani.save("./planner_full.gif")
        if store is not None:
            # TODO (generate path inside images/<timeoftheday>/store:str)
            ani.save(store)
        plt.show()

    def __plot_vel(store: str=None):
        fig, ax = plt.subplots(1,1, figsize= (15,15))
        # Plot control outputs
        t = np.arange(0.1, 50, 0.1)
        u = data_storage.get(SimData.control)
        line, = ax.plot(t,u[0, :], color = "r", label='v')
        line2, = ax.plot(t,u[1, :], color = "g", label='$\omega$')
        # line, = ax.plot(u[0, :230], color = "r", label='v')
        # line2, = ax.plot(u[1, :230], color = "g", label='$\omega$')
        ax.legend([r"$v$", r"$\omega$"])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$u$")
        ax.grid(True)
        plt.savefig("./images/velocities/prova_vel.png")
        plt.show()

    if plot_flag:
        __plot_fn(store_plot)
    if plot_velocities:
        __plot_vel(store_plot)
    return data_storage
