"""test_control.py
Contains tests for controllers
"""

import logging
import numpy as np

from .config import SimulationConfiguration
from .config import DefaultSimulationParameters as dsp

from ..logger import SimulationDataStorage, SimData, timeprofiledecorator
from ..trajectory import QuinticTrajectory2D, CircleTrajectory2D, SplineTrajectory2D
from ..transform import FrenetGNTransform
from ..controller import FrenetIOLController
from ..plotter import *

logger=logging.getLogger(__name__)

def test_planner(*args, **kwargs):
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
    robot_ddp = np.zeros(3)
    u = np.zeros(2)
    for i in range(sim_config.get_simulation_length()):
        # Estimate frenet frame
        est_pt = transformer.estimatePosition(trajectory, robot_p)
        # Robot pose in frenet
        robot_fpose = transformer.transform(robot_p)
        # Robot velocity in frenet (need only p_dot and d_dot)
        robot_fdp  = transformer.transform(robot_dp)[0:2]
        # Robot acceleration in frenet 
        robot_fddp  = transformer.transform(robot_ddp)[0:2]

        # Compute error and derror
        # target_pos = trajectory.compute_pt(t_vect[i])
        # target_fpos = transformer.transform(target_pos)
        # target_dpos = trajectory.compute_first_derivative(t_vect[i])
        # target_fdpos = transformer.transform(target_dpos)
        if i == 0:
            old_s = (0,0)
            old_d = (0,0)
            s0 = (robot_fpose[0],robot_fdp[0],robot_fddp[0])
            d0 = (robot_fpose[1],robot_fdp[1],robot_fddp[1])
            planner.initialize(t0 = t_vect[i], p0 = d0, s0 = s0)
            pos_s, pos_d = planner.s0, planner.p0
        else:
            pos_s, pos_d = planner.replanner(t_vect[i])
            
        target_pos = transformer.itransform(np.array([pos_s[0]-old_s[0], pos_d[0]-old_d[0]]))
        target_fpos = np.array([pos_s[0]-old_s[0], pos_d[0]-old_d[0]])
        target_fdpos = np.array([pos_s[1]-old_s[1], pos_d[1]-old_d[1]])
        error = target_fpos - robot_fpose[0:2]
        print(pos_s[0]-old_s[0], pos_d[0]-old_d[0])
        old_d = pos_d
        old_s = pos_s
        
        # Set error on s to 0 (TEST)
        # error[0] = 0.0
        derror = target_fdpos - robot_fdp
        #print(pos_s, pos_d)
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
