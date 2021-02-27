"""ctrl_frenet_pathf.py
Path follower controller
"""

import numpy as np
import logging
from .controller import Controller
from ..platform import coordinates_f

logger = logging.getLogger(__name__)

class FrenetPathFollowerController(Controller):
    """ Sanson controller class for path followers
    """
    def __init__(self, k2: float, k3: float, v: float=0.0):
        super().__init__()
        assert k2 > 0. and k3 > 0.
        self.k2 = k2
        self.k3 = k3
        self.v  = v

    def compute(self, robot_fpose: np.array, *args, **kwargs) -> np.array:
        """ Compute the path following control term.

        Parameters
        ----------
        robot_fpose : np.array
            Robot pose in Frenet frame (d, s, theta)
        
        Returns
        -------
        np.array
            control term to bring the robot on the actual path
        """
        u = np.zeros(2)
        s, d, t = coordinates_f(robot_fpose)
        u[0] = self.v
        u[1] = -self.k2 * d * u[0] * np.sin(t) / t - self.k3 * t
        return u
        





