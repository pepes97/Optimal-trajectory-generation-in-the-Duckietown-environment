"""obstacle.py
"""

from abc import ABC, abstractmethod

class Obstacle(ABC):
    """ Obstacle abstract class
    """
    @abstractmethod
    def position(self):
        ...
