"""constants.py
"""

import numpy as np

class DuckietownParameters:
    ASPECT_RATIO = 640/480
    # Camera field of view angle in the Y direction
    FOVY = 75 / 180 * np.pi
    # Camera field of view angle in the X direction
    FOVX = FOVY * ASPECT_RATIO
    # Angle at which the camera is pitched downwards
    GAMMA = -19.15 / 180 * np.pi
    # Distance from camera to floor (10.8cm)
    CAMERA_HEIGHT = 0.108 #m
    # Sistance to the bottom pixel row
    DIST_MIN = CAMERA_HEIGHT * np.tan(0.5 * np.pi + GAMMA - 0.5 * FOVY) 
    # We want to see up to DIST_MAX meters
    DIST_MAX = 0.8 
    THETA_M = np.arctan(DIST_MAX / CAMERA_HEIGHT) - 0.5 * np.pi - GAMMA
    pM = np.tan(THETA_M) / np.tan(0.5 * FOVY)
    pM_corr = 0.5 * (1 - pM)
    lm = 2 * np.sqrt(CAMERA_HEIGHT**2 + DIST_MIN**2) * np.tan(0.5 * FOVX)
    lM = 2 * np.sqrt(CAMERA_HEIGHT**2 + DIST_MAX**2) * np.tan(0.5 * FOVX)
    X = lm / lM
    # Total robot length
    ROBOT_LENGTH = 0.18
    # Total robot width at wheel base, used for collision detection
    ROBOT_WIDTH = 0.13 + 0.02
    # Safety radius multiplier
    SAFETY_RAD_MULT = 1#1.8
    # Robot safety circle radius
    AGENT_SAFETY_RAD = (max(ROBOT_LENGTH, ROBOT_WIDTH) / 2) * SAFETY_RAD_MULT
    # Robot safety area
    AGENT_SAFETY_AREA = np.pi*AGENT_SAFETY_RAD**2