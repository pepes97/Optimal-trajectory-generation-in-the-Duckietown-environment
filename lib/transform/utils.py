"""utils.py
"""

import numpy as np
import logging

def homogeneous_transform(pose: np.array) -> (np.array, np.array):
    assert pose.shape == (3,)
    p = pose[:2]
    t = pose[2]
    ct = np.cos(t)
    st = np.sin(t)
    R = np.array([[ct, -st], [st, ct]])
    return R, p

def homogeneous_itransform(pose: np.array) -> (np.array, np.array):
    _R, _p = homogeneous_trasform(pose)
    R = np.linalg.inv(_R)
    p = - np.matmul(R, _p)
    return R, p
