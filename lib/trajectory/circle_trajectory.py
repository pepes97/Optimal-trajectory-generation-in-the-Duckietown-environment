"""circle_trajectory.py
"""

import numpy as np
import logging
from .defs import DifferentiableFunction, Trajectory

logger = logging.getLogger(__name__)

class CircleTrajectory2D(Trajectory, DifferentiableFunction):
    def __init__(self, l: float, h: float, r: float):
        """
        Parameters
        ----------
        l : float
            TODO
        h : float
            TODO
        r : float
            TODO
        """
        self.l = l
        self.h = h
        self.r = r
        # TODO(Sveva)
        pass
    
    def compute_pt(self, t):
        # TODO(Sveva)
        return np.array([0.0, 0.0])

    def compute_first_derivative(self, t):
        # Possible approach (To Be Verified)
        h = 1e-5
        df = compute_pt(t+h/2) - compute_pt(t-h/2)
        logger.debug(f'Approx derivative: {df}')
        return df

    def compute_second_derivative(self, t):
        # Possible approach (To Be Verified)
        h = 1e-5
        ddf = compute_first_derivative(t+h/2) - compute_first_derivative(t-h/2)
        return ddf
        # TODO(Sveva) PS: Secondo me finch'è non ci serve, puoi lasciarla pure cosi
        logger.error('Function not yet implemented')
        return np.array([0.0, 0.0])

    def compute_third_derivative(self, t):
        # TODO(Sveva) PS: Secondo me finch'è non ci serve, puoi lasciarla pure cosi
        logger.error('Function not yet implemented')
        return np.array([0.0, 0.0])

    def compute_curvature(self, t):
        # TODO(Sveva)
        logger.error('Function not yet implemented')
        return 0.0
