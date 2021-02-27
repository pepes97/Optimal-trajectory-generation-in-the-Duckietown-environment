"""test_sensor.py
"""

import logging
import numpy as np

from .config import SimulationConfiguration
from .config import DefaultSimulationParameters as dsp

from ..logger import SimulationDataStorage, SimData
from ..trajectory import QuinticTrajectory2D, CircleTrajectory2D, SplineTrajectory2D
from ..transform import FrenetGNTransform
from ..controller import FrenetIOLController

logger = logging.getLogger(__name__)

def test_proximity_sensor(*args, **kwargs) -> SimulationDataStorage:
    ...
