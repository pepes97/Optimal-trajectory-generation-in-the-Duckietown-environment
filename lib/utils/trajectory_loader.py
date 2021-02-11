"""trajectory_loader.py
"""

import numpy as np
import jsonpickle
import logging
import os.path as path
from ..trajectory import *

logger = logging.getLogger(__name__)

def load_trajectory_configuration(path: str) -> Trajectory:
    """ Load a trajectory object from a json configuration
    """
    deserialized_obj = jsonpickle.decode(open(path, 'r').read())
    return deserialized_obj

def save_trajectory_configuration(path: str, trajectory: Trajectory):
    """ Save a trajectory object to a json configuration
    """
    data_json = jsonpickle.encode(trajectory)
    with open(path, 'w') as f:
        f.write(data_json)
