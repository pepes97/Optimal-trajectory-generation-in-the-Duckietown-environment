"""test_bot.py
"""

import logging
import numpy as np

from .config import SimulationConfiguration
from .config import DefaultSimulationParameters as dsp
from ..logger import SimulationDataStorage, SimData
from ..trajectory import CircleTrajectory2D, Trajectory
from ..transform import FrenetGNTransform
from ..controller import FrenetIOLController, FrenetPathFollowerController
from ..platform import Unicycle
from ..plotter import *


logger = logging.getLogger(__name__)

class UniBot:
    def __init__(self, trajectory: Trajectory, pos: np.array, initial_guess: float):
        self.trajectory = trajectory
        self.robot_pose = pos
        self.robot = Unicycle()
        self.robot.set_initial_pose(self.robot_pose)
        self.transformer = FrenetGNTransform()
        self.controller = FrenetPathFollowerController(dsp.pf_k2, dsp.pf_k3, dsp.pf_v)

    def step(self) -> np.array:
        self.transformer.estimatePosition(self.trajectory, self.robot_pose)
        robot_fpose = self.transformer.transform(self.robot_pose)
        u = self.controller.compute(robot_fpose)
        self.robot_pose, robot_vel = self.robot.step(u, dsp.dt)
        return self.robot_pose

class IdealBot:
    def __init__(self, trajectory: Trajectory, t_start: float, dt: float=0.1):
        self.trajectory = trajectory
        self.t_start = t_start
        self.dt = dt
        self.t = self.t_start

    def step(self) -> np.array:
        robot_pos  = self.trajectory.compute_pt(self.t)
        self.t += self.dt
        return robot_pos

def test_bot(**kwargs) -> SimulationDataStorage:
    plot_flag = False
    store_plot = None
    if 'plot' in kwargs:
        plot_flag = kwargs['plot']
    if 'store_plot' in kwargs:
        store_plot = kwargs['store_plot']
    sim_config = SimulationConfiguration(**kwargs)
    # Extract key objects from configuration object
    t_vect, robot, trajectory, transformer, controller = sim_config.get_elements()
    data_storage = SimulationDataStorage(t_vect)
    data_storage.add_argument(SimData.bot0_position)
    data_storage.add_argument(SimData.bot1_position)
    data_storage.add_argument(SimData.trajectory_2d)
    # Generate two bots
    #bot0 = UniBot(trajectory, np.append(trajectory.compute_pt(10), 0.1), 10)
    #bot1 = UniBot(trajectory, np.append(trajectory.compute_pt(0.1), 0.3), 0.1)
    bot0 = IdealBot(trajectory, 0.0)
    bot1 = IdealBot(trajectory, 10.0)
    bot_lst = [bot0, bot1]

    data_storage = _simulate_experiment(sim_config, data_storage, trajectory, robot,
                                        transformer, controller, bot_lst)

    def __plot_fn(store: str=None):
        fig = ...
        if store is not None:
            plt.savefig(store)
        plt.show()
    if plot_flag:
        __plot_fn(store_plot)
    return data_storage

def _simulate_experiment(sim_config, data_storage, trajectory, robot, transformer, controller, bot_lst) -> SimulationDataStorage:
    t_vect = sim_config.get_time_vect()

    for i in range(sim_config.get_simulation_length()):
        target_pos = trajectory.compute_pt(t_vect[i])
        for j, bot in enumerate(bot_lst):
            bpose = bot.step()
            data_storage.set(f'bot{j}_position', bpose, i)
        
        # log data
        data_storage.set('trajectory', target_pos, i)
    return data_storage
