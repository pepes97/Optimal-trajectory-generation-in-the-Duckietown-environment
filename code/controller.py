"""controller.py
"""

import numpy as np
from unicycle import Unicycle


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

class FrenetInputOutputLinearizationController(Controller):
    """ Input Output Linearization controller in Frenet frame.
    TODO"""
    def __init__(self, b: float, k1: float, k2:float):
        super().__init__()
        self.b = b
        self.k1 = k1
        self.k2 = k2
        pass
    def compute(self, rpose: np.array, k: float, p: np.array, dp: np.array) -> np.array:
        """ Compute the trajectory tracking control term.
        rpose: Robot pose in Frenet frame (d, s, theta)
        k: current path curvature
        p: Desired pose in Frenet frame (d_des, s_des, theta_des)
        dp: Desired speed in Frenet frame (dd_des, ds_des, dtheta_des)
        """
        u = np.zeros((2, ))
        # Build kinematic matrix in Frenet frame
        b = self.b
        s = rpose[0]
        d = rpose[1]
        t = rpose[2]
        ct = np.cos(t)
        st = np.sin(t)
        A = np.array(
            [[ct * (1 + b * k * st) / (1 - k * d), -b * st],
             [st - (k * ct) / (1 - k * d), b * ct]])
        A_inv = np.linalg.inv(A)
        des_input = np.array([p[0] + b * ct, p[1] + b * st])
        error = des_input - np.array([s + b * ct, d + b * st])
        error[0] *= self.k1
        error[1] *= self.k2
        #error *= np.array([self.k1, self.k2])
        u = np.matmul(A_inv, (dp[0:2] + error).T)
        return u
        
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

from matplotlib import pyplot as plt
import matplotlib.animation as animation

def __test_control_2D_IOLin():
    p_start = np.array([0.0, 0.0])
    dp_start = np.array([5.0, 2.0])
    ddp_start = np.array([3.0, 1.0])
    p_end = np.array([10.0, 2.0])
    dp_end = np.array([-2.0, -2.0])
    ddp_end = np.array([0.0, 0.5])
    t_start = 0.0
    t_end = 10.0
    t = np.arange(t_start, t_end, 0.1)
    LEN_SIMULATION = t.shape[0]

    trajectory = QuinticTrajectory2D(p_start, dp_start, ddp_start,
                                     p_end, dp_end, ddp_end,
                                     t_start, t_end)

    frenet_transform = FrenetTransform2D(trajectory, initial_guess=0.5)
    # Controller components
    #ctrl_b, ctrl_k1, ctrl_k2 = 0.5, 0.7, 5.0
    ctrl_b, ctrl_k1, ctrl_k2 = 0.5, 12.0, 9.0
    print('Using FrenetInputOutputLinearizationController')
    print(f'Controller parameters: b: {ctrl_b}\tk1: {ctrl_k1}\tk2: {ctrl_k2}')
    controller = FrenetInputOutputLinearizationController(ctrl_b, ctrl_k1, ctrl_k2)
    # Robot components
    robot_position = np.array([0.0, 0.0, -np.pi])
    robot = Unicycle(robot_position)

    # Simulation loop
    gpose_vect = np.zeros((3, LEN_SIMULATION))
    gpose_vect[:, 0] = robot_position
    est_pose_vect = np.zeros(LEN_SIMULATION)
    fpose_vect = np.zeros((3, LEN_SIMULATION))
    tpose_vect = np.zeros(LEN_SIMULATION) # Target vector
    tfpose_vect = np.zeros((3, LEN_SIMULATION)) # Target in frenet frame
    errf_vect = np.zeros((3, LEN_SIMULATION)) # Error in frenet frame
    p_robot = robot_position
    for i in range(LEN_SIMULATION):
        est_pose, _, _ = frenet_transform.estimatePosition(p_robot[0:2])
        frenet_transform.generateLocalFrame()
        fpose = frenet_transform.transform(p_robot)
        curvature = frenet_transform.computeCurvature()
        
        target_pos = np.append(trajectory.compute_pt(t[i]), 0.0)
        target_pos = frenet_transform.transform(target_pos)
        dtarget_pos = np.append(trajectory.compute_first_derivative(t[i]), 0.0)
        dtarget_pos = 0 * frenet_transform.transform(dtarget_pos)
        u = controller.compute(fpose, curvature, target_pos, dtarget_pos)
        #print(f'robot: {p_robot}, target_pos: {target_pos}, curvature: {curvature}, u: {u}')
        #print(u)
        p_robot = robot.step(u)
        
        gpose_vect[:, i] = p_robot.T
        fpose_vect[:, i] = fpose.T
        est_pose_vect[i] = est_pose
        tfpose_vect[:, i] = target_pos
        errf_vect[:, i] = target_pos - fpose

    # Plot section
    fig, axs = plt.subplots(2)
    # Plot path
    path_vect = np.zeros((2, t.shape[0]))
    for i in range(t.shape[0]):
        path_i = trajectory.compute_pt(t[i])
        path_vect[:, i] = path_i.T
    # PLOT 0
    # Plot path
    axs[0].plot(path_vect[0, :], path_vect[1, :])
    # Plot robot pose
    rgline, = axs[0].plot(gpose_vect[0, :], gpose_vect[1, :])
    rpoint = axs[0].scatter(gpose_vect[0, 0], gpose_vect[1, 0], c='orange')
    rproj_pose = trajectory.compute_pt(est_pose_vect[0])
    rproj = axs[0].scatter(rproj_pose[0], rproj_pose[1], c='r')
    # Plot target position
    tgpoint = axs[0].scatter(path_vect[0, i], path_vect[1, i])
    # PLOT 1
    tfline, = axs[1].plot(errf_vect[0, :], errf_vect[1, :])
    def animate(i):
        # Update robot pose
        rgline.set_xdata(gpose_vect[0, :i])
        rgline.set_ydata(gpose_vect[1, :i])
        rproj_pose = trajectory.compute_pt(est_pose_vect[i])
        rproj.set_offsets([rproj_pose[0], rproj_pose[1]])
        rpoint.set_offsets([gpose_vect[0, i], gpose_vect[1, i]])
        # Update target position
        tgpnt = trajectory.compute_pt(t[i])
        tgpoint.set_offsets([tgpnt[0], tgpnt[1]])
        # Update target position in frenet
        tfline.set_xdata(errf_vect[0, :i])
        tfline.set_ydata(errf_vect[1, :i])
        return [rgline, rproj, tgpoint, tfline]

    ani = animation.FuncAnimation(
        fig, animate, frames=t.shape[0], interval=40, repeat=True)

    plt.tight_layout()
    plt.show()
    #ani.save('ctrl.gif')
    #plot_unicycle_evolution_2D_animated(gpose_vect, fpose_vect, est_pose_vect, trajectory, t)
    
if __name__ == '__main__':
    print('Controller main script')
    #print('Launching control test')
    #__test_control()
    #print('Launching control2D test')
    #__test_control_2D()
    print('Launching control2D IOLinearization test')
    __test_control_2D_IOLin()
    exit(0)
