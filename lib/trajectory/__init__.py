"""__init__.py
Trajectory package initializer
"""

from .defs import DifferentiableFunction, Trajectory
from .trajectory_1d import QuarticPolynomial, QuinticPolynomial
from .trajectory_2d import QuinticTrajectory2D
from .circle_trajectory import CircleTrajectory2D
from .spline_trajectory import SplineTrajectory2D
