"""diff_function.py
Contains interface for differentiable functions
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
