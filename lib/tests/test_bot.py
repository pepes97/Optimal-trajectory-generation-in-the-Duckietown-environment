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

def test_bot(**kwargs) -> SimulationDataStorage:
    #kwargs['t_end'] = 60
    sim_config = SimulationConfiguration(**kwargs)
    # Extract key objects from configuration object
    t_vect, robot, trajectory, transformer, controller = sim_config.get_elements()
    data_storage = SimulationDataStorage(t_vect)
    data_storage.add_argument(SimData.bot0_pose)
    data_storage.add_argument(SimData.bot1_pose)
    data_storage.add_argument(SimData.trajectory_2d)
    # Generate two bots
    bot0 = UniBot(trajectory, np.append(trajectory.compute_pt(10), 0.1), 10)
    bot1 = UniBot(trajectory, np.append(trajectory.compute_pt(0.1), 0.3), 0.1)
    bot_lst = [bot0, bot1]

    data_storage = _simulate_experiment(sim_config, data_storage, trajectory, robot,
                                        transformer, controller, bot_lst)
    return data_storage

def _simulate_experiment(sim_config, data_storage, trajectory, robot, transformer, controller, bot_lst) -> SimulationDataStorage:
    t_vect = sim_config.get_time_vect()

    for i in range(sim_config.get_simulation_length()):
        target_pos = trajectory.compute_pt(t_vect[i])
        for j, bot in enumerate(bot_lst):
            bpose = bot.step()
            data_storage.set(f'bot{j}_pose', bpose, i)
        
        # log data
        data_storage.set('trajectory', target_pos, i)
    return data_storage
