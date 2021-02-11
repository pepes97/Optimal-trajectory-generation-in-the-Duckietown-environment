"""trajectory_2d.py
2-Dimensional Trajectory definitions
"""

import numpy as np
import logging
import json
from .defs import DifferentiableFunction, Trajectory
from .trajectory_1d import QuarticPolynomial, QuinticPolynomial

logger = logging.getLogger(__name__)

class QuinticTrajectory2D(Trajectory, DifferentiableFunction):
    """ Represents a 2D quintic polynomial trajectory.
    TODO: Ensure that time handling is correct
    """
    def __init__(self, p_start: np.array, dot_p_start: np.array, ddot_p_start: np.array,
                 p_end: np.array, dot_p_end: np.array, ddot_p_end: np.array,
                 t_start: float, t_end: float):
        # Check input dimensions
        assert p_start.shape[0] == 2
        assert p_end.shape[0] == 2
        assert dot_p_start.shape[0] == 2
        assert dot_p_end.shape[0] == 2
        assert ddot_p_start.shape[0] == 2
        assert ddot_p_end.shape[0] == 2
        # Check valid time interval
        assert t_start >= 0.
        assert t_end > t_start
        
        super().__init__()
        self.t_start = t_start
        self.t_end = t_end

        # Generate a polynomial for each dimension
        self.x_path = QuinticPolynomial(p_start[0], dot_p_start[0], ddot_p_start[0],
                                        p_end[0], dot_p_end[0], ddot_p_end[0], t_end-t_start)
        self.y_path = QuinticPolynomial(p_start[1], dot_p_start[1], ddot_p_start[1],
                                        p_end[1], dot_p_end[1], ddot_p_end[1], t_end-t_start)
        # TODO ?
    def compute_pt(self, t):
        """
        Compute pt given time t
        """
        # Check time validity (TODO)
        return np.array([self.x_path.compute_pt(t),
                         self.y_path.compute_pt(t)])
    
    def compute_first_derivative(self, t):
        """
        Compute first derivative given time t
        """
        # Check time validity (TODO)
        return np.array([self.x_path.compute_first_derivative(t),
                         self.y_path.compute_first_derivative(t)])

    def compute_second_derivative(self, t):
        """
        Compute second derivative given time t
        """
        # Check time validity (TODO)
        return np.array([self.x_path.compute_second_derivative(t),
                         self.y_path.compute_second_derivative(t)])

    def compute_third_derivative(self, t):
        """
        Compute second derivative given time t
        """
        # Check time validity (TODO)
        return np.array([self.x_path.compute_third_derivative(t),
                         self.y_path.compute_third_derivative(t)])

    def compute_curvature(self, t):
        acc_vect = self.compute_second_derivative(t)
        vel_vect = self.compute_first_derivative(t)
        numer = acc_vect[1] * vel_vect[0] - acc_vect[0] * vel_vect[1]
        denom = vel_vect[0]**2 + vel_vect[1]**2
        return numer / denom
        pass
