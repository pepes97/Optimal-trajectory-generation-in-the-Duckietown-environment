"""config.py
"""

import numpy as np
import logging
from ..platform import Unicycle
from ..trajectory import Trajectory, QuinticTrajectory2D
from ..transform import FrenetTransform, FrenetGNTransform
from ..controller import Controller, FrenetIOLController
from ..planner import TrajectoryPlanner

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
    pf_k2   = 40
    pf_k3   = 1.5
    pf_v    = 0.5

    ## planner 

    GLOBAL_D_T = 0.05
    MAX_ROAD_WIDTH = 2.45 # maximum road width [m]
    D_ROAD_W = 0.6 # road width sampling length [m]
    T_0 = 0  # initial time [s]

    D_T = 0.8 # time tick [s]
    MAX_T = 5.0 # max prediction time [m]
    MIN_T = 0.9 # min prediction time [m]

    DES_SPEED = 5.0 # speed desired [m/s]
    D_D_S = 1  # target speed sampling length [m/s]
    N_S_SAMPLE = 3  # sampling number of target speed
    LOW_SPEED_THRESH = 2.0 # low speed switch [m/s]
    S_THRSH = 1.0

    # Cost weights
    K_J = 0.001
    K_T = 0.01
    K_D = 0.9
    K_S = 0.1
    K_DOT_S = 1.0
    K_LAT = 1.0
    K_LONG = 1.0

    TARGET_DIST = 1 #[m] distance from following target

    p0 = np.array([3,0.3,0]) 
    s0 = np.array([0,2,-0.5])
    t0 = 0 # instant in which the planner is initialized

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
        self.planner = None
        # Apply changes if user provides them
        self.__dict__.update(kwargs)
        # Update robot pose if robot object is passed
        if self.robot is not None:
            self.robot.set_initial_pose(self.robot_pose)
        logger.debug(f'Simulation configuration: {self.__dict__}')
    
    def get_time_vect(self) -> np.array:
        return np.arange(self.t_start, self.t_end, self.dt)

    def get_simulation_length(self) -> np.array:
        return self.get_time_vect().shape[0]

    def get_robot(self) -> Unicycle:
        if self.robot is None:
            robot = Unicycle()
            robot.set_initial_pose(self.robot_pose)
            return robot 
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
            return FrenetGNTransform()
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
                
    def get_planner(self) -> Planner:
        if self.planner is None:
            return TrajectoryPlanner(dsp.t0, dsp.p0, dsp.s0)
        else:
            return self.planner

class SimulationConfigurationData:
    def __init__(self, **kwargs):
        self.t_start = dsp.t_start
        self.t_end   = dsp.t_end
        self.dt      = dsp.dt
        self.robot_pose = dsp.robot_pose
        self.robot   = Unicycle()
        self.trajectory = QuinticTrajectory2D(dsp.p_start, dsp.dp_start, dsp.ddp_start,
                                              dsp.p_end, dsp.dp_end, dsp.ddp_end,
                                              dsp.t_start, dsp.t_end)
        self.transformer = FrenetGNTransform()
        self.controller  = FrenetIOLController(dsp.b, dsp.kp1, dsp.kp2, dsp.kd1, dsp.kd2)
        self.__dict__.update(kwargs)
