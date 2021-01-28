"""controller.py
"""

import numpy as np


class Controller:
    """Controller abstract class. TBD
    """
    def __init__(self):
        pass

class SansonController(Controller):
    """
    Sanson algorithm produces control terms that can keep the robot on a given trajectory
    assuming the difference between robot's heading and Frenet main axis is sufficently small
    """
    def __init__(self, k2, k3, v: float=None):
        super().__init__()
        assert k2 > 0. and k3 > 0.
        self.k2 = k2
        self.k3 = k3
        self.v = v

    def setVelocity(self, v: float):
        self.v = v

    def compute(self, p: np.array) -> np.array:
        r = p[0]
        l = p[1]
        t = p[2]
        u = np.zeros((2, ))
        u[0] = self.v
        u[1] = -self.k2 * l * u[0] * np.sin(t) / t - self.k3 * t
        return u

class AstolfiController(Controller):
    """Astolfi algorithm realizes point stabilization of non holonomic systems using
    discontinuous static feedback.
    TODO
    """
    def __init__(self):
        super().__init__()
        pass

class FrenetFullController(SansonController, AstolfiController):
    """Full Frenet controller which combines both Sanson and Astolfi controllers based on the
    current state of the system.
    TODO
    """
    def __init__(self):
        pass

from quintic_polynomial import QuinticPolynomial
from unicycle import Unicycle
from frenet_transform import FrenetTransform, FrenetTransform2D
from plot import plot_unicycle_evolution_animated, plot_unicycle_evolution_2D_animated
from trajectory import QuinticTrajectory2D

def __test_control():
    ROBOT_P0 = np.array([0.0, -1.0, np.pi/8])
    LEN_SIMULATION = 250
    T_end=10 # 1 second evolution
    robot = Unicycle(ROBOT_P0)
    # PATH CONSTRUCTION
    path = QuinticPolynomial(-1, 1, 0, 1, -1, 0, T_end)
    frenet_transform = FrenetTransform(path, initial_guess=0.0)
    ctrl_k2, ctrl_k3, ctrl_v = 30, 1.2, 0.5
    print(f'Controller parameters: k2: {ctrl_k2}\tk3: {ctrl_k3}')
    controller = SansonController(ctrl_k2, ctrl_k3, ctrl_v)
    
    
    gpose_vect = np.zeros((3, LEN_SIMULATION))
    gpose_vect[:, 0] = ROBOT_P0.T
    est_pose_vect = np.zeros(LEN_SIMULATION)
    fpose_vect = np.zeros((3, LEN_SIMULATION))
    p_robot = ROBOT_P0
    u = np.zeros((2,))
    for i in range(LEN_SIMULATION):
        est_pose, _, _ = frenet_transform.estimatePosition(p_robot[0:2])
        frenet_transform.generateLocalFrame()
        fpose = frenet_transform.transform(p_robot)
        u = controller.compute(fpose)
        #print(u)
        p_robot = robot.step(u)
        
        gpose_vect[:, i] = p_robot.T
        fpose_vect[:, i] = fpose.T
        est_pose_vect[i] = est_pose

    plot_unicycle_evolution_animated(gpose_vect, fpose_vect, est_pose_vect, path, T_end)    

def __test_control_2D():
    p_start = np.array([0.0, 0.0])
    dp_start = np.array([0.0, 0.0])
    ddp_start = np.array([-3.0, 0.0])
    p_end = np.array([10.0, 3.0])
    dp_end = np.array([1.0, -1.0])
    ddp_end = np.array([0.0, 0.0])
    t_start = 0.0
    t_end = 10.0
    t = np.arange(t_start, t_end, 0.05)
    LEN_SIMULATION = t.shape[0]

    trajectory = QuinticTrajectory2D(p_start, dp_start, ddp_start,
                                     p_end, dp_end, ddp_end,
                                     t_start, t_end)

    frenet_transform = FrenetTransform2D(trajectory, initial_guess=0.5)
    ctrl_k2, ctrl_k3, ctrl_v = 10, 2, 1
    print(f'Controller parameters: k2: {ctrl_k2}\tk3: {ctrl_k3}')
    controller = SansonController(ctrl_k2, ctrl_k3, ctrl_v)

    robot_position = np.array([0.0, 0.0, np.pi/2])
    robot = Unicycle(robot_position)

    gpose_vect = np.zeros((3, LEN_SIMULATION))
    gpose_vect[:, 0] = robot_position
    est_pose_vect = np.zeros(LEN_SIMULATION)
    fpose_vect = np.zeros((3, LEN_SIMULATION))
    p_robot = robot_position
    for i in range(LEN_SIMULATION):
        est_pose, _, _ = frenet_transform.estimatePosition(p_robot[0:2])
        frenet_transform.generateLocalFrame()
        fpose = frenet_transform.transform(p_robot)
        u = controller.compute(fpose)
        #print(u)
        p_robot = robot.step(u)
        
        gpose_vect[:, i] = p_robot.T
        fpose_vect[:, i] = fpose.T
        est_pose_vect[i] = est_pose
    plot_unicycle_evolution_2D_animated(gpose_vect, fpose_vect, est_pose_vect, trajectory, t)
    
if __name__ == '__main__':
    print('Controller main script')
    #print('Launching control test')
    #__test_control()
    print('Launching control2D test')
    __test_control_2D()
    exit(0)
