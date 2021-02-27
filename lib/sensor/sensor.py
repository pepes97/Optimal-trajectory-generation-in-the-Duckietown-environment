"""sensor.py
"""

from abc import ABC, abstractmethod
import numpy as np
from ..platform import Unicycle

class Sensor(ABC):
    """ Sensor abstract class
    """

    @abstractmethod
    def attach(self, robot: Unicycle):
        ...

    @abstractmethod
    def sense(self, *args, **kwargs):
        ...
