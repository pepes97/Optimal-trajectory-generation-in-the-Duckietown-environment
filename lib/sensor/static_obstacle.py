"""static_obstacle.py
"""

import numpy as np
import logging
from .obstacle import Obstacle

logger = logging.getLogger(__name__)

class StaticObstacle(Obstacle):
    """ Static Obstacle class
    """
    def __init__(self, p: np.array):
        self.position = p

    def position(self) -> np.array:
        return self.position
