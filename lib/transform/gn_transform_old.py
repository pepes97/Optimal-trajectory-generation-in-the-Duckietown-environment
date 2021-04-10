"""gn_transform.py
"""

import numpy as np
import logging
from .frenet_transform import FrenetTransform
from ..trajectory import DifferentiableFunction

logger = logging.getLogger(__name__)

class FrenetGNTransformOld(FrenetTransform):

    def __init__(self, initial_guess: float=0.5, max_iter: int=10):
        self.estimate = initial_guess
        self.max_iter = max_iter
        self.robot_pose = None
    
    def estimatePosition(self, trajectory: DifferentiableFunction, p: np.array):
        """ Estimate the position of the orthogonal projection of p onto the trajectory
        """
        self.trajectory = trajectory
        def local_frame() -> (np.ndarray, np.array, float):
            """ Returns local frame parameters
            
            Returns
            ----------
            R : np.ndarray
                Rotation matrix associated with the frame
            origin : np.array
                Translation vector associated with the frame
            theta : float
                Frame orientation angle
            """
            # Frame position
            origin = self.trajectory.compute_pt(self.estimate)
            # Frame orientation
            t_grad = self.trajectory.compute_first_derivative(self.estimate)
            t_r = np.arctan2(t_grad[1], t_grad[0])
            # Precompute sin(t_r) and cos(t_r)
            ct = np.cos(t_r)
            st = np.sin(t_r)
            # Rotation matrix
            R = np.array([[ct, -st], [st, ct]])
            return R, origin, t_r

        # LEAST SQUARES BLOCK
        self.robot_pose = p
        estimate = self.estimate
        for it in range(self.max_iter):
            ds, chi = do_ls(self.trajectory, estimate, p[0:2])
            estimate += ds
        self.estimate = estimate
        # Once estimate is found, generate a local frame
        self.r_mat, self.t_vect, self.t_r = local_frame()
        return self.estimate

    def transform(self, p: np.array) -> np.array:
        """ Transform a SE(2) pose or R2 point in the frenet frame 
        """
        R, t = self.r_mat, self.t_vect
        if p.shape[0] == 3:
            # Handle SE(2) object
            p_pos = p[0:2]
            p_ang = p[2]
            # point in frenet
            p_pos_f = np.matmul(R.T, p_pos) - np.matmul(R.T, t)
            p_ang_f = p_ang - self.t_r
            return np.append(p_pos_f, p_ang_f)
        elif p.shape[0] == 2:
            # Handle point object
            return np.matmul(R.T, p) - np.matmul(R.T, t)
        else:
            logger.error('input array has invalid lenght')
            raise ValueError


    def itransform(self, p: np.array) -> np.array:
        """ Transform a SE(2) pose or R2 point in the global frame
        """
        R, t = self.r_mat, self.t_vect
        if p.shape[0] == 3:
            # Handle SE(2) object
            p_pos = p[0:2]
            p_ang = p[2]
            # point in frenet
            p_pos_f = np.matmul(R, p_pos) + t
            p_ang_f = p_ang + self.t_r
        elif p.shape[0] == 2:
            # Handle point object
            return np.matmul(R, p) + t
        else:
            logger.error('input array has invalid lenght')
            raise ValueError

def do_ls(trajectory: DifferentiableFunction, estimate: float,
          target: np.array, damping: float=0.01) -> (float, float):
    # Euclidean difference between current estimate and target
    cerror = trajectory.compute_pt(estimate) - target
    J = trajectory.compute_first_derivative(estimate)
    # Compute H matrix and b vector
    H = np.matmul(J.T, J) + damping
    b = np.matmul(J.T, cerror)
    # Compute new estimate and chi_squares
    ds = - b / H
    chi = np.matmul(cerror.T, cerror)
    return ds, chi