"""planner.py
"""

from abc import ABC, abstractmethod
from .frenet import Frenet
import numpy as np
class Planner(ABC):

    @abstractmethod
    def step(t: float, dd: float = None, dsd: float = None, s_target: Frenet = None) -> np.array:
        """ Returns the frenet target position at time t
        """
        pass
