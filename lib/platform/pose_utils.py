"""pose_utils.py
"""

import numpy as np

def coordinates_f(robot_fpose: np.array) -> (float, float, float):
    """ Returns the frenet coordinates of robot_fpose as the tuple:
    (s, d, theta_~)
    """
    return robot_fpose[0], robot_fpose[1], robot_fpose[2]

def coordinates(robot_pose: np.array) -> (float, float, float):
    """ Returns the global coordinates of robot_pose as the tuple:
    (x, y, theta)
    """
    return coordinates_f(robot_pose)
