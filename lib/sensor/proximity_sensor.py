"""range_sensor.py
"""

import logging
import numpy as np
from .sensor import Sensor
from .obstacle import Obstacle
from ..platform import Unicycle

logger = logging.getLogger(__name__)

class ProximitySensor(Sensor):
    """ ProximitySensor produces a sensor which can find obstacles inside 
    a user defined cone in front of the sensor
    """
    def __init__(self, range: float, aperture: float, *args, **kwargs):
        """ ProximitySensor produces a sensor which can find obstacles inside a user defined cone
        in front of the sensor.
        
        Parameters
        ----------
        range : float
            Lenght of the cone
        aperture : float
            Angle in radians which express the aperture of the cone
        """
        super().__init__()
        self.range = range
        self.aperture = aperture
        if len(args) > 0:
            if isinstance(args[0]) is Unicycle:
                self.robot = args[0]
        if kwargs is not None:
            self.__dict__.update(kwargs)

    def attach(self, robot: Unicycle):
        """ Attach the sensor to the robot
        """
        self.robot = robot

    def sense(self, *args, **kwargs) -> [Obstacle]:
        """ Sense the environment and returns the obstacles present inside the sensor's cone.
        """
        def get_angle(v1: np.array, v2: np.array):
            """ Returns the angle in radians between vector v1 and v2
            """
            cosang = np.dot(v1, v2)
            sinang = np.linalg.norm(np.cross(v1, v2))
            return np.arctan2(sinang, cosang)
        
        found_obstacles = []
        # Get cone tip
        if 'rpose' in kwargs:
            logger.debug(f'Using passed pose: {kwargs["rpose"]}')
            robot_pose = kwargs['rpose']
        else:
            robot_pose = self.robot.pose()
        p = robot_pose[:2]
        t = robot_pose[2]
        # Get cone direction
        dir = np.array([np.cos(t), np.sin(t)])
        h = self.range
        for obstacle in self.obstacles:
            diff_vect = obstacle.position - p
            cone_dist = np.dot(diff_vect, dir)
            # Reject objects not in range 
            if cone_dist < 0. or cone_dist >= self.range:
                continue
            # Check if obstacle is inside the cone
            obst_angle = get_angle(dir, diff_vect)
            if np.abs(obst_angle) > self.aperture:
                continue
            # If both checks are passed, obstacle is inside the cone
            found_obstacles.append(obstacle)
        return found_obstacles

    def set_obstacle(self, obs_lst: [Obstacle]):
        """ Set the obstacle list which the sensor uses to detect the latter
        """
        self.obstacles = obs_lst
