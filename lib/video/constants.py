"""constants.py
"""

import numpy as np

class DuckietownParameters:
    ASPECT_RATIO = 640/480
    FOVY = 75 / 180 * np.pi
    FOVX = FOVY * ASPECT_RATIO
    GAMMA = -19.15 / 180 * np.pi
    CAMERA_HEIGHT = 0.108 #m
    DIST_MIN = CAMERA_HEIGHT * np.tan(0.5 * np.pi + GAMMA - 0.5 * FOVY) # distance to the bottom pixel row
    DIST_MAX = 0.8 # We want to see up to DIST_MAX meters
    THETA_M = np.arctan(DIST_MAX / CAMERA_HEIGHT) - 0.5 * np.pi - GAMMA
    pM = np.tan(THETA_M) / np.tan(0.5 * FOVY)
    pM_corr = 0.5 * (1 - pM)
    lm = 2 * np.sqrt(CAMERA_HEIGHT**2 + DIST_MIN**2) * np.tan(0.5 * FOVX)
    lM = 2 * np.sqrt(CAMERA_HEIGHT**2 + DIST_MAX**2) * np.tan(0.5 * FOVX)
    X = lm / lM




