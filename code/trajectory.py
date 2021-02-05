"""trajectory.py
"""

import numpy as np
from diff_function import DifferentiableFunction
from quintic_polynomial import QuinticPolynomial
from quartic_polynomial import QuarticPolynomial

class Trajectory:
    """ Abstract Trajectory class TBD
    """
    def __init__(self):
        pass

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

from plot import plot_trajectory2D

def __test_quintic_trajectory2d():
    p_start = np.array([0.0, 0.0])
    dp_start = np.array([0.0, 0.0])
    ddp_start = np.array([-3.0, 0.0])
    p_end = np.array([10.0, 3.0])
    dp_end = np.array([1.0, -1.0])
    ddp_end = np.array([0.0, 0.0])
    t_start = 0.0
    t_end = 10.0

    trajectory = QuinticTrajectory2D(p_start, dp_start, ddp_start,
                                     p_end, dp_end, ddp_end,
                                     t_start, t_end)

    t = np.arange(t_start, t_end, 0.1)
    path = np.zeros((2, t.shape[0]))
    for i in range(t.shape[0]):
        step = trajectory.compute_pt(t[i])
        path[:, i] = step

    plot_trajectory2D(path)
    return

    
if __name__ == '__main__':
    print('Trajectory main script')
    print('Launching trajectory test')
    __test_quintic_trajectory2d()
    
    exit(0)
