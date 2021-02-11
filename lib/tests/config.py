"""config.py
"""

import numpy as np
import logging
from ..platform import Unicycle
from ..trajectory import Trajectory, QuinticTrajectory2D
from ..transform import FrenetTransform, FrenetGNTransform
from ..controller import Controller, FrenetIOLController

logger = logging.getLogger(__name__)

class DefaultSimulationParameters:
    t_start = 0.0
    t_end  = 10.0
    dt     = 0.1
    robot_pose = np.array([0.0, 0.0, 0.0])
    p_start   = np.array([0.0, 0.0])
    dp_start  = np.array([2.0, -2.0])
    ddp_start = np.array([-1.0, 1.0])
    p_end   = np.array([10.0, 15.0])
    dp_end  = np.array([-2.0, 2.0])
    ddp_end = np.array([1.0, -1.0])
    # PD controller parameters
    kp1     = 10.0
    kp2     = 9.
    kd1     = 0.5
    kd2     = 0.5
    b       = 0.5

# Alias
dsp = DefaultSimulationParameters

class SimulationConfiguration:
    """ Configuration for the simulation. Metadata and config parameters can be stored in here
    """
    def __init__(self, **kwargs):
        # Initialize default arguments
        self.t_start = dsp.t_start
        self.t_end   = dsp.t_end
        self.dt      = dsp.dt
        self.robot_pose = dsp.robot_pose
        self.robot = None
        self.trajectory = None
        self.transformer = None
        self.controller = None
        # Apply changes if user provides them
        self.__dict__.update(kwargs)

    # def __init__(self, t_start: float, t_end: float, dt: float, robot_pose: np.array):
    #     self.t_start = t_start
    #     self.t_end = t_end
    #     self.dt = dt
    #     self.t = np.arange(t_start, t_end, dt)
    #     self.robot = Unicycle(robot_pose)
    #     # TODO
    
    def get_time_vect(self) -> np.array:
        return np.arange(self.t_start, self.t_end, self.dt)

    def get_simulation_length(self) -> np.array:
        return self.get_time_vect().shape[0]

    def get_robot(self) -> Unicycle:
        if self.robot is None:
            return Unicycle(self.robot_pose)
        else:
            return self.robot

    def get_trajectory(self) -> Trajectory:
        if self.trajectory is None:
            return QuinticTrajectory2D(dsp.p_start, dsp.dp_start, dsp.ddp_start,
                                       dsp.p_end, dsp.dp_end, dsp.ddp_end,
                                       dsp.t_start, dsp.t_end)
        else:
            return self.trajectory

    def get_transformer(self) -> FrenetTransform:
        if self.transformer is None:
            trajectory = self.get_trajectory()
            return FrenetGNTransform(trajectory)
        else:
            return self.transformer
        

    def get_controller(self) -> Controller:
        if self.controller is None:
            return FrenetIOLController(dsp.b, dsp.kp1, dsp.kp2, dsp.kd1, dsp.kd2)
        else:
            return self.controller

    def get_elements(self) -> (np.array, Unicycle, Trajectory, FrenetTransform, Controller):
        return (self.get_time_vect(), self.get_robot(), self.get_trajectory(), self.get_transformer(),
                self.get_controller())
