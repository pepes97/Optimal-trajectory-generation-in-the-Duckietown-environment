"""static_obstacle.py
"""

import numpy as np
import logging
from .obstacle import Obstacle
from ..trajectory import Trajectory, DifferentiableFunction
from ..transform import FrenetTransform, FrenetGNTransform


logger = logging.getLogger(__name__)

class MovingObstacle(Obstacle):
    """ Moving Obstacle class
    """
    def __init__(self, p: np.array, t: Trajectory):
        self.start_position = p
        self.trajectory = ObstacleTrajectory(o = np.random.randint(-1,1), \
            s = np.random.random(), t0 = np.random.randint(5,10), t = t)
        self.step(0)

    def position(self) -> np.array:
        return self.position
    
    def step(self, time: float):
        self.position = self.trajectory.compute_pt(time)
        self.orientation = self.trajectory.compute_orientation(time)
        self.pose = np.concatenate([self.position, self.orientation[None]],axis=0)

    def reset(self):
        self.position = self.start_position


class ObstacleTrajectory(Trajectory, DifferentiableFunction):
    
    def __init__(self, o: float, s: float, t0: float, t: Trajectory):
        self.offset = o
        self.speed = s
        self.time_offset = t0
        self.trajectory = t
        self.transformer = FrenetGNTransform()
    
    def time_law(self, t):
        return self.time_offset + t * self.speed % 50 # create a method to get max time of a trajectory
    
    def compute_ortogonal_vect(self, t: float) -> np.array:
        t_r = self.compute_orientation(t)
        return np.array([-np.sin(t_r), np.cos(t_r)])
    
    def compute_orientation(self, t: float) -> np.array:
        t_grad = self.trajectory.compute_first_derivative(self.time_law(t))
        t_r = np.arctan2(t_grad[1], t_grad[0])
        return t_r
    
    def compute_pt(self, t):
        return self.trajectory.compute_pt(self.time_law(t)) + self.compute_ortogonal_vect(t) * self.offset
    
    def compute_first_derivative(self, t):
        return self.trajectory.compute_first_derivative(t)
        
    def compute_second_derivative(self, t):
       return self.trajectory.compute_second_derivative(t)
       
    def compute_third_derivative(self, t):
        return self.trajectory.compute_third_derivative(t)
        
    def compute_curvature(self, t):
       return self.trajectory.compute_curvature(t)
   
    def compute_s(self,t):
        return self.transformer.transform(self.compute_pt(t))[0]
    
    def compute_ds(self,t):
        return self.transformer.transform(self.compute_first_derivative(t))[0]
    
    def compute_dds(self,t):
        return self.transformer.transform(self.compute_second_derivative(t))[0]
    
    def compute_ddds(self,t):
        return self.transformer.transform(self.compute_third_derivative(t))[0]

    def set_transformer(self, transformer: FrenetTransform):
        self.transformer = transformer