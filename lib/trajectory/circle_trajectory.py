"""circle_trajectory.py
"""

import numpy as np
import json
import logging
import math
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
        self.dt = 0.1
        # TODO(Sveva)
        pass
    
    def compute_pt(self, t):
        t = float(t)
        while t>= 2*self.l + 2*self.r*math.pi + 2*self.h:
            # t= t- (2*self.l + 2*self.r*math.pi + 2*self.h)
            t = float(t) % (2*self.l + 2*self.r * np.pi + 2*self.h) 

        # first segment
        if t>=0 and t<self.l:
            return  np.array([t, 0.])
        # first 1/4 circle
        elif t>=self.l and t< self.l + 2*self.r*math.pi/4:
            alpha = (t- self.l)/self.r
            return  np.array([self.l + self.r*math.sin(alpha), self.r - self.r*math.cos(alpha)])
        # second segment
        elif t>= self.l + 2*self.r*math.pi/4 and t< self.l + 2*self.r*math.pi/4 + self.h:
            return  np.array([self.l + self.r , t- self.l - 2*self.r*math.pi/4 + self.r])
        # second 1/4 circle
        elif t>= self.l + 2*self.r*math.pi/4 + self.h and t< self.l + self.r*math.pi + self.h:
            alpha = (t- (self.l + 2*self.r*math.pi/4 + self.h))/self.r
            return  np.array([self.l + math.cos(alpha)*self.r, self.r + self.h + math.sin(alpha)*self.r])
        # third segment
        elif t>= self.l + self.r*math.pi + self.h and t< 2*self.l + self.r*math.pi + self.h:
            return  np.array([self.l - (t- self.l - self.r*math.pi - self.h), 2*self.r+self.h])
        # third 1/4 circle
        elif t>= 2*self.l + self.r*math.pi + self.h and t< 2*self.l + self.r*math.pi + self.h + 2*self.r*math.pi/4:
            alpha = (t- (2*self.l + self.r*math.pi + self.h))/self.r
            return  np.array([-math.sin(alpha)*self.r, self.r+ self.h+self.r*math.cos(alpha)])
        # fourth segment
        elif t>= 2*self.l + self.r*math.pi + self.h + 2*self.r*math.pi/4 and t< 2*self.l + self.r*math.pi + 2*self.h + 2*self.r*math.pi/4:
            return  np.array([-self.r, self.r+ self.h - (t-(2*self.l + self.r*math.pi + self.h + 2*self.r*math.pi/4))])
        # fourth 1/4 circle
        else:
            alpha = (t- (2*self.l + self.r*math.pi + 2*self.h + 2*self.r*math.pi/4))/self.r
            return  np.array([-self.r*math.cos(alpha), self.r- self.r*math.sin(alpha)])

    def compute_first_derivative(self, t):
        # # Possible approach (To Be Verified)
        # h = 1e-5
        # df = self.compute_pt(t+h/2) - self.compute_pt(t-h/2)
        # logger.debug(f'Approx derivative: {df}')
        df = (self.compute_pt(t+self.dt) - self.compute_pt(t))/self.dt
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
        t= float(t)
        while t>= 2*self.l + 2*self.r*math.pi + 2*self.h:
            # t= t- (2*self.l + 2*self.r*math.pi + 2*self.h)
            t = float(t) % (2*self.l + 2*self.r * np.pi + 2*self.h) 

        # first segment
        if t>=0 and t<self.l:
            return 0.
        # first 1/4 circle
        elif t>=self.l and t< self.l + 2*self.r*math.pi/4:
            return 1./self.r
        # second segment
        elif t>= self.l + 2*self.r*math.pi/4 and t< self.l + 2*self.r*math.pi/4 + self.h:
            return 0.
        # second 1/4 circle
        elif t>= self.l + 2*self.r*math.pi/4 + self.h and t< self.l + self.r*math.pi + self.h:
            return 1./self.r
        # third segment
        elif t>= self.l + self.r*math.pi + self.h and t< 2*self.l + self.r*math.pi + self.h:
            return 0.
        # third 1/4 circle
        elif t>= 2*self.l + self.r*math.pi + self.h and t< 2*self.l + self.r*math.pi + self.h + 2*self.r*math.pi/4:
            return 1./self.r
        # fourth segment
        elif t>= 2*self.l + self.r*math.pi + self.h + 2*self.r*math.pi/4 and t< 2*self.l + self.r*math.pi + 2*self.h + 2*self.r*math.pi/4:
            return 0.
        # fourth 1/4 circle
        else:
            return 1./self.r
        return 0.0
