"""defs.py
Trajectory classes definitions
"""

from abc import ABC, abstractmethod

class DifferentiableFunction(ABC):

    @abstractmethod
    def compute_pt(self, t):
        pass

    @abstractmethod
    def compute_first_derivative(self, t):
        pass

    @abstractmethod
    def compute_second_derivative(self, t):
        pass

    @abstractmethod
    def compute_third_derivative(self, t):
        pass

    def compute_curvature(self, t):
        pass

class Trajectory:
    """ Abstract Trajectory class TBD
    """
    def __init__(self):
        pass
