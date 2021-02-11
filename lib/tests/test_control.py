"""test_control.py
Contains tests for controllers
"""

import logging
import numpy as np

from .config import SimulationConfiguration
from .config import DefaultSimulationParameters as dsp

from ..logger import SimulationDataStorage, SimData
from ..trajectory import QuinticTrajectory2D
from ..transform import FrenetGNTransform
from ..controller import FrenetIOLController
from ..trajectory import CircleTrajectory2D
from ..trajectory import SplineTrajectory2D

logger=logging.getLogger(__name__)

def test_path_follower_2D() -> SimulationDataStorage:#TODO
    logger.error('Function not implemented')
    pass

def test_trajectory_track_2D() -> SimulationDataStorage:#TODO
    sim_config = SimulationConfiguration(dsp.t_start,
                                         dsp.t_end,
                                         dsp.dt,
                                         dsp.robot_pose)
    t_vect = sim_config.get_time_vect()
    robot  = sim_config.get_robot()
    # Configure SimulationDataStorage
    data_storage = SimulationDataStorage(t_vect)
    data_storage.add_argument(SimData.robot_pose)
    data_storage.add_argument(SimData.robot_frenet_pose)
    data_storage.add_argument(SimData.control)
    data_storage.add_argument(SimData.trajectory_2d)
    data_storage.add_argument(SimData.target_pos)
    data_storage.add_argument('error', 2)
    data_storage.add_argument('derror', 2)
    # Configure trajectory

    ## Quintic
    # trajectory = QuinticTrajectory2D(dsp.p_start, dsp.dp_start, dsp.ddp_start,
    #                                  dsp.p_end, dsp.dp_end, dsp.ddp_end,
    #                                  sim_config.t_start, dsp.t_end)

    ## Circle
    #trajectory = CircleTrajectory2D(8,5,2)

    ## Spline
    x = [0.0, 2.5, 5.0, 7.5, -3.0, 2.7]
    y = [0.7, -6, 5, -9.5, 0.0, 5]
    trajectory = SplineTrajectory2D(x,y)
    
    # Configure transformer
    transformer = FrenetGNTransform(trajectory)

    # Configure controller
    controller = FrenetIOLController(dsp.b, dsp.kp1, dsp.kp2, dsp.kd1, dsp.kd2)

    # Simulation loop
    robot_p = robot.p

    robot_dp = np.zeros(3)
    u = np.zeros(2)
    for i in range(sim_config.get_simulation_length()):
        # Estimate frenet frame
        est_pt = transformer.estimatePosition(robot_p)
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

        logger.debug(f'error: {error}, derror: {derror}')

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
