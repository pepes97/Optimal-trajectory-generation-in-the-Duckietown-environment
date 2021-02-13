"""test_generate_configurations.py
"""

import logging
import os.path as path
from ..trajectory import *
from ..platform import Unicycle
from ..controller import FrenetIOLController
from ..transform import FrenetGNTransform
from ..serializer import Serializer
from .config import SimulationConfigurationData

logger = logging.getLogger(__name__)


def test_generate_configurations(**kwargs):
    """ Generate multiple experiment configurations and consequently 
    dumps them in the config/ directory
    """
    if 'config_folder' in kwargs:
        config_folder = kwargs['config_folder']
    else:
        config_folder = './config'
    # Generate QuinticPolynomial configuration
    polynomial_config = SimulationConfigurationData()

    # Generate circle track configuration
    circle_config = SimulationConfigurationData(trajectory=CircleTrajectory2D(8, 5, 2))

    # Generate spline configuration
    x = [0.0, 2.5, 5.0, 7.5, -3.0, 2.7]
    y = [0.7, -6, 5, -9.5, 0.0, 5]
    spline_config = SimulationConfigurationData(trajectory=SplineTrajectory2D(x, y))

    # Generate serializer object
    serializer = Serializer('jsonpickle')
    serializer.serialize(path.join(config_folder, 'config_polynomial2D.json'), polynomial_config)
    serializer.serialize(path.join(config_folder, 'config_circle2D.json'), circle_config)
    serializer.serialize(path.join(config_folder, 'config_spline2D.json'), spline_config)
    
    
    
    
