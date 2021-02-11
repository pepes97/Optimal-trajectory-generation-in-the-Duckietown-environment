"""config.py
"""

import numpy as np
import logging
from ..platform import Unicycle

logger = logging.getLogger(__name__)

class DefaultSimulationParameters:
    t_start = 0.0
    t_end  = 50
    dt     = 0.1
    robot_pose = np.array([0.0, 0.0, 0.0])
    p_start   = np.array([0.0, 0.0])
    dp_start  = np.array([2.0, -2.0])
    ddp_start = np.array([-1.0, 1.0])
    p_end   = np.array([10.0, 15.0])
    dp_end  = np.array([-2.0, 2.0])
    ddp_end = np.array([1.0, -1.0])
    # PD controller parameters
    kp1     = 2.0
    kp2     = 2.
    kd1     = 0.5
    kd2     = 0.5
    b       = 1.5


class SimulationConfiguration:
    """ Configuration for the simulation. Metadata and config parameters can be stored in here
    """
    def __init__(self, t_start: float, t_end: float, dt: float, robot_pose: np.array):
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.t = np.arange(t_start, t_end, dt)
        self.robot = Unicycle(robot_pose)
        # TODO

    def get_time_vect(self) -> np.array:
        return self.t

    def get_simulation_length(self) -> np.array:
        return self.t.shape[0]

    def get_robot(self) -> Unicycle:
        return self.robot
