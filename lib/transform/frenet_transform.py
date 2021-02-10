"""frenet_transform.py
"""

from abc import ABC, abstractmethod
import numpy as np

class FrenetTransform(ABC):

    @abstractmethod
    def estimatePosition(self, p: np.array):
        """ Estimate the position of the orthogonal projection of p onto the trajectory
        """
        pass
    
    @abstractmethod
    def transform(self, p: np.array) -> np.array:
        """ Transform a SE(2) pose or R2 point in the frenet frame 
        """
        pass

    @abstractmethod
    def itransform(self, pf: np.array) -> np.array:
        """ Transform a SE(2) pose or R2 point in the global frame
        """
        pass
