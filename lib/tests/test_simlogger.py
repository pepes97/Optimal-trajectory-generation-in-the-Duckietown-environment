"""test_simlogger.py
"""

import numpy as np
import logging
from ..logger import SimulationDataStorage

logger = logging.getLogger(__name__)

def test_simlogger():
    t = np.arange(0.0, 10.0, 0.1)
    print(f'Simulation length: {t.shape[0]}')
    simlogger = SimulationDataStorage(t)
    print('Storing one dimensional variable y')
    simlogger.add_argument('y', 1)
    print('Storing 3D variable robot_pose')
    simlogger.add_argument('robot_pose', 3)
    for i in range(t.shape[0]):
        cval = np.sin(t[i] * 2 * np.pi / 10.0)
        simlogger.set('y', cval, i)
        simlogger.set('robot_pose', np.array([i, i-1, i-3]), i)
    print(f'stored y shape: {simlogger.get("y").shape}')
    print(f'stored robot_pose shape: {simlogger.get("robot_pose").shape}')

    print(f'Testing module criticies')
    try:
        simlogger.set('invalid_idx', 10, 0)
    except Exception as e:
        logger.debug(e)
