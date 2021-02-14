"""planner.py
"""

from abc import ABC, abstractmethod

class Planner(ABC):

    @abstractmethod
    def step(t: float) -> np.array:
        """ Returns the frenet target position at time t
        """
        pass
