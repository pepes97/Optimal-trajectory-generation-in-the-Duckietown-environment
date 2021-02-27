"""ctrl_frenet_iol.py
"""

from .controller import Controller
from ..platform import coordinates_f
import logging
import numpy as np

logger = logging.getLogger(__name__)

class FrenetIOLController(Controller):
    def __init__(self, b: float, kp1: float, kp2:float, kd1: float, kd2: float):
        super().__init__()
        self.b = b
        self.kp1 = kp1
        self.kp2 = kp2
        self.kd1 = kd1
        self.kd2 = kd2
        logger.debug(f'Instantiating controller with parameters: b={b}, kp1={kp1}, kp2={kp2}, kd1={kd1}, kd2={kd2}')
        self.kp = np.array([kp1, kp2])
        self.kd = np.array([kd1, kd2])

    def compute(self, robot_fpose: np.array,
                error: np.array, derror: np.array,
                k: float, *kwargs) -> np.array:
        """ Compute the trajectory tracking control term.
        
        Parameters
        ----------
        robot_fpose : np.array
            Robot pose in Frenet frame (d, s, theta)
        error : np.array
            Error between robot pose and target position in frenet frame
        derror : np.array
            Error between robot velocity and target velocity in frenet frame
        k : float
            Curvature of the path at the current step

        Returns
        -------
        np.array
            control term to bring the robot on the actual trajectory
        """
        u = np.zeros(2)
        b = self.b
        # Extract robot frenet coordinates
        s, d, t = coordinates_f(robot_fpose)
        # Precompute cos(t) and sin(t)
        ct = np.cos(t)
        st = np.sin(t)
        # Compute IO Jacobian inverse
        T = np.array(
        [[ct * (1 + b * k * st) / (1 - k * d), -b * st],
         [st - (k * ct) / (1 - k * d), b * ct]])
        T_inv = np.linalg.inv(T)
        # Compute PD input terms
        p_error = error * self.kp
        d_error = derror * self.kd
        # Compute control term
        u = np.matmul(T_inv, (p_error + d_error).T)
        return u
