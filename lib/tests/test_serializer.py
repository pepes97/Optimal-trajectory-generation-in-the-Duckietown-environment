"""test_serializer.py
"""

import logging
from .config import SimulationConfiguration
from .config import DefaultSimulationParameters as dsp
from ..trajectory import *
from ..serializer import Serializer

logger = logging.getLogger(__name__)

def test_serializer():
    serializer = Serializer('jsonpickle')
    STORE_PATH = 'temp.json'
    # Configure simulation
    sim_config = SimulationConfiguration()
    trajectory = sim_config.get_trajectory()
    logger.info('Trajectory element:')
    logger.info(trajectory.__dict__)
    logger.info(f'Storing trajectory in {STORE_PATH}')
    serialier.serialize(STORE_PATH, trajectory)
    logger.info(f'Recovering trajectory from {STORE_PATH}')
    new_trajectory = serializer.deserialize(STORE_PATH)
    logger.info('Recovered trajectory:')
    logger.info(new_trajectory.__dict__)
