"""test_serializer.py
"""

import logging
from .config import SimulationConfiguration
from .config import DefaultSimulationParameters as dsp
from ..trajectory import *
from ..utils import save_trajectory_configuration, load_trajectory_configuration

logger = logging.getLogger(__name__)

def test_serializer():
    STORE_PATH = 'temp.json'
    # Configure simulation
    sim_config = SimulationConfiguration(dsp.t_start,
                                         dsp.t_end,
                                         dsp.dt,
                                         dsp.robot_pose)
    # Generate Quintic Trajectory
    trajectory = QuinticTrajectory2D(dsp.p_start, dsp.dp_start, dsp.ddp_start,
                                     dsp.p_end, dsp.dp_end, dsp.ddp_end,
                                     sim_config.t_start, dsp.t_end)
    logger.info('Trajectory element:')
    logger.info(trajectory.__dict__)
    logger.info(f'Storing trajectory in {STORE_PATH}')
    save_trajectory_configuration(STORE_PATH, trajectory)
    logger.info(f'Recovering trajectory from {STORE_PATH}')
    new_trajectory = load_trajectory_configuration(STORE_PATH)
    logger.info('Recovered trajectory:')
    logger.info(new_trajectory.__dict__)
