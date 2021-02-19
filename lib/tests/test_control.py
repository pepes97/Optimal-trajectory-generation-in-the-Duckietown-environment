"""test_control.py
Contains tests for controllers
"""

import logging
import numpy as np

from .config import SimulationConfiguration
from .config import DefaultSimulationParameters as dsp

from ..logger import SimulationDataStorage, SimData
from ..trajectory import QuinticTrajectory2D, CircleTrajectory2D, SplineTrajectory2D
from ..transform import FrenetGNTransform
from ..controller import FrenetIOLController
from ..plotter import *

logger=logging.getLogger(__name__)

def test_trajectory_track_2D(*args, **kwargs):
    plot_flag = False
    store_plot = None
    if 'plot' in kwargs:
        plot_flag = kwargs['plot']
    if 'store_plot' in kwargs:
        store_plot = kwargs['store_plot']
    
    sim_config = SimulationConfiguration(**kwargs)
    # Extract key objects from configuration object
    t_vect, robot, trajectory, transformer, controller, planner = sim_config.get_elements()
    # Configure SimulationDataStorage
    data_storage = SimulationDataStorage(t_vect)
    data_storage.add_argument(SimData.robot_pose)
    data_storage.add_argument(SimData.robot_frenet_pose)
    data_storage.add_argument(SimData.control)
    data_storage.add_argument(SimData.trajectory_2d)
    data_storage.add_argument(SimData.target_pos)
    data_storage.add_argument('error', 2)
    data_storage.add_argument('derror', 2)

    data_storage = _simulate_experiment(sim_config, data_storage, trajectory,
                                        robot, transformer, controller, planner)

    def __plot_fn(store: str=None):
        fig = plot_2d_simulation(data_storage)
        if store is not None:
            # TODO (generate path inside images/<timeoftheday>/store:str)
            plt.savefig(store)
        plt.show()
    if plot_flag:
        __plot_fn(store_plot)
    return data_storage

def _simulate_experiment(sim_config, data_storage, trajectory, robot, transformer, controller, planner):
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
        data_storage.set('robot_frenet_pose', robot_fpose, i)
        data_storage.set('control', u, i)
        data_storage.set('trajectory', target_pos, i)
        data_storage.set('target_pos', target_fpos, i)
        data_storage.set('error', error, i)
        data_storage.set('derror', derror, i)
    return data_storage

# def _simulate_experiment(sim_config, data_storage, trajectory, robot, transformer, controller, planner):
#     # Simulation loop
#     t_vect = sim_config.get_time_vect()
#     robot_p = robot.p

#     robot_dp = np.zeros(3)
#     u = np.zeros(2)
      # Estimate frenet frame
#     est_pt = transformer.estimatePosition(trajectory, robot_p)
#     # Robot pose in frenet
#     robot_fpose = transformer.transform(robot_p)
#     # Robot velocity in frenet (need only p_dot and d_dot)
#     robot_fdp   = transformer.transform(robot_dp)[0:2]
#     # Robot aceleration in frenet (need only p_dot and d_dot)
#     robot_fddp   = transformer.transform(robot_dp)[0:2]
#     opt_path = planner.optimal_path(t_vect[0], (robot_fpose[1], robot_fdp[1], robot_fddp[1]), 
#                                      (robot_fpose[0], robot_fdp[0], robot_fddp[0]))
#     for i in range(sim_config.get_simulation_length()):
#         
#         index = round(t_vect[i] - opt_path.t[0]/0.05)
#         # Compute error and derror
#         target_fpos = np.array([opt_path.s[index], opt_path.d[index]])
#         target_fdpos = np.array([opt_path.dot_s[index], opt_path.dot_d[index]])
#         error = target_fpos - robot_fpose[0:2]
#         derror = target_fdpos - robot_fdp
#         # Get path curvature at estimate
#         curvature = trajectory.compute_curvature(est_pt)

#         # Compute control
#         u = controller.compute(robot_fpose, error, derror, curvature)

#         # Step the unicycle
#         robot_p, robot_dp = robot.step(u, dsp.dt)

#         # log data
#         data_storage.set('robot_pose', robot_p, i)
#         data_storage.set('robot_frenet_pose', robot_fpose, i)
#         data_storage.set('control', u, i)
#         # data_storage.set('trajectory', target_pos, i)
#         data_storage.set('target_pos', target_fpos, i)
#         data_storage.set('error', error, i)
#         data_storage.set('derror', derror, i)
#     return data_storage

