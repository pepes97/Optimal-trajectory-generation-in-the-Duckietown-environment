"""ctrl_frenet_iol.py
"""

from .controller import Controller
from ..platform import Unicycle
import logging
import numpy as np
logger = logging.getLogger(__name__)


class FrenetIOLController(Controller):
    def __init__(self, b: float, k1: float, k2:float):
        super().__init__()
        self.b = b
        self.k1 = k1
        self.k2 = k2
        logger.debug(f'Instantiating controller with parameters: b={b}, k1={k1}, k2={k2}')
    def compute(self, rpose: np.array, k: float, p: np.array, dp: np.array) -> np.array:
        """ Compute the trajectory tracking control term.
        rpose: Robot pose in Frenet frame (d, s, theta)
        k: current path curvature
        p: Desired pose in Frenet frame (d_des, s_des, theta_des)
        dp: Desired speed in Frenet frame (dd_des, ds_des, dtheta_des)
        """
        u = np.zeros((2, ))
        # Build kinematic jacobian in Frenet frame
        b = self.b
        s = rpose[0]
        d = rpose[1]
        t = rpose[2]
        ct = np.cos(t)
        st = np.sin(t)
        A = np.array(
            [[ct * (1 + b * k * st) / (1 - k * d), -b * st],
             [st - (k * ct) / (1 - k * d), b * ct]])
        # Invert input/output jacobian
        A_inv = np.linalg.inv(A)
        # Apply Proportional term to ensure convergence.
        des_input = np.array([p[0] + b * ct, p[1] + b * st])
        error = des_input - np.array([s + b * ct, d + b * st])
        error *= np.array([self.k1, self.k2])
        # Compute control term
        u = np.matmul(A_inv, (dp[0:2] + error).T)
        return u
