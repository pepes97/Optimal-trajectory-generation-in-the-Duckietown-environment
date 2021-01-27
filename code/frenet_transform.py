"""frenet_transform.py
"""

import numpy as np
from diff_function import DifferentiableFunction


class FrenetTransform():
    def __init__(self, path: DifferentiableFunction=None, initial_guess: float=0.0, max_iter: int=10):
        self.path = path
        self.max_iter = max_iter
        self.proj_estimate = initial_guess
        self.robot_gpose = None
        pass

    def estimatePosition(self, p: np.array) -> (float, [float], [float]):
        """ Apply GN algorithm to find a good estimate for the orthogonal projection of p onto the
        path self.path. Also store the estimate in the class, to speed up convergence in the next
        iterations."""
        self.robot_gpose = p
        estimate = self.proj_estimate
        error_lst = []
        estimate_lst = []
        for iter in range(self.max_iter):
            # Distance between the point p and the path
            coordinate_err = np.array([estimate, self.path.compute_pt(estimate)]) - p[0:2]
            error = np.matmul(coordinate_err.T, coordinate_err)#np.linalg.norm(coordinate_err) ** 2
            # First derivative of path at the estimate point
            J = np.array([1, self.path.compute_first_derivative(estimate)])
            H = np.matmul(J.T, J)
            b = np.matmul(J.T, coordinate_err)
            # Perturbation vector
            Ds = - b / H
            estimate += Ds * 1
            
            error_lst.append(error)
            estimate_lst.append(estimate)
        # Final estimate is stored and returned
        self.proj_estimate = estimate
        return estimate, error_lst, estimate_lst

    def generateLocalFrame(self, radial_threshold:float=0.4):
        """ Generates a local Frenet frame at the current estimate. The principal axis is the tangential vector to the path at the estimate.
        The second vector is the vector pointing towards the robot. To be correctly designed, these two vectors should be orthogonal to each other.
        A warning message will inform the user if such condition is not met. 
        Also computes the curvature k at the current position estimate.
        """
        x = self.proj_estimate
        y = self.path.compute_pt(x)
        dy = self.path.compute_first_derivative(x)
        theta_r = np.arctan2(dy, x)
        # Rotation matrix between RF0 and FrenetFrame
        cR = np.cos(theta_r)
        sR = np.sin(theta_r)
        R = np.array([[cR, -sR], [sR, cR]])
        tangential_vector = np.matmul(R, np.array([1, 0]))
        radial_vector = np.matmul(R, np.array([0, 1]))
        # True distance vector from the estimate towards the robot position
        p_robot_vect = np.array([x, self.path.compute_pt(x)]) - self.robot_gpose[0:2]
        p_robot_distance = np.linalg.norm(p_robot_vect)
        p_robot_vect /= p_robot_distance
        if np.dot(radial_vector, p_robot_vect) > radial_threshold:
            print(f'Warning: Frenet radial vector is far from true distance vector')

        self.theta_r = theta_r
        self.R_transform = R
        self.tangential_vect = tangential_vector
        # CARE MUST BE TAKEN HERE (Next operations rely heavily on radial_vect. Using a different vector from p_robot_vect is dangerous
        # if it is too far from radial_vect
        self.radial_vect = radial_vector
        self.translation_vect = np.array([x, y])
        self.p_robot_vect = p_robot_vect
        self.p_robot_distance = p_robot_distance

    def getTransform(self) -> (np.array, np.array):
        """Returns rotation matrix and translation vector elements that characterize the homogeneous transform between
        global and local frames.
        """
        return self.R_transform, self.translation_vect

    def transform(self, p) -> np.array:
        """ Transform the unicycle pose in the frenet frame.
        """
        R, t = self.getTransform()
        r_t = p[2]
        r_p = p[0:2]
        # Point in frenet frame
        p_f = np.matmul(R.T, r_p) - np.matmul(R.T, t)
        t_f = r_t - self.theta_r
        return np.array([p_f[0], p_f[1], t_f])
        
        
    def transform_d(self, dp) -> np.array:
        ...
        
        


from quintic_polynomial import QuinticPolynomial
from matplotlib import pyplot as plt
from unicycle import Unicycle
from plot import plot_unicycle_evolution, plot_unicycle_evolution_animated

def __test_GN():
    T_end = 1
    ITERATIONS = 10
    path = QuinticPolynomial(0, -1, 0, 1, 0, 0, T_end)
    frenet_transform = FrenetTransform(path, initial_guess=0.8, max_iter=ITERATIONS)
    robot_position = np.array([0.4, 0., 0])
    robot_estimate_on_path, error_evolution, estimate_evolution= frenet_transform.estimatePosition(robot_position)
    x = np.arange(0, 0, 0.05)
    y = np.array([path.compute_pt(s) for s in x])
    fig, axs = plt.subplots(2)
    
    axs[0].plot(x, y)
    axs[0].scatter(robot_position[0], robot_position[1], c='r')
    axs[0].scatter(robot_estimate_on_path, path.compute_pt(robot_estimate_on_path), c='g')
    axs[0].plot([robot_position[0], robot_estimate_on_path], [robot_position[1], path.compute_pt(robot_estimate_on_path)], 'r-')

    est_x = estimate_evolution
    est_fx = [path.compute_pt(i) for i in est_x]
    axs[0].scatter(est_x, est_fx, c='g')
    
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Orthogonal projection of unicycle on path')
    axs[1].plot(np.arange(0, ITERATIONS, 1), np.array(error_evolution))
    axs[1].set_xlabel('iterations')
    axs[1].set_ylabel('chi_squares')
    plt.show()

def __test_transform():
    ROBOT_P0 = np.array([0.0, -1.0, 0.0])
    LEN_SIMULATION = 70
    T_end=1 # 1 second evolution
    robot = Unicycle(ROBOT_P0)
    u_vect = np.zeros((2, LEN_SIMULATION), dtype=float)
    # Forward movement
    u_vect[0, :] = 0.2;
    # Small curvilinear behaviour
    for i in range(LEN_SIMULATION):
        #rotation_step = i / (LEN_SIMULATION-1) * (np.pi / 2)
        rotation_step = 7 / (i + 1)
        u_vect[1, i] = rotation_step / 3
    # PATH CONSTRUCTION
    path = QuinticPolynomial(-1, 1, 0, 1, -1, 0, T_end)
    frenet_transform = FrenetTransform(path, initial_guess=0.0)
    
    
    gpose_vect = np.zeros((3, LEN_SIMULATION))
    gpose_vect[:, 0] = ROBOT_P0.T
    est_pose_vect = np.zeros(LEN_SIMULATION)
    fpose_vect = np.zeros((3, LEN_SIMULATION))
    for i in range(LEN_SIMULATION):
        p_robot = robot.step(u_vect[:, i])
        est_pose, _, _ = frenet_transform.estimatePosition(p_robot[0:2])
        frenet_transform.generateLocalFrame()
        gpose_vect[:, i] = p_robot.T
        fpose_vect[:, i] = frenet_transform.transform(p_robot).T
        est_pose_vect[i] = est_pose

    #plot_unicycle_evolution(gpose_vect, fpose_vect, path, T_end)
    plot_unicycle_evolution_animated(gpose_vect, fpose_vect, est_pose_vect, path, T_end)    

if __name__ == '__main__':
    print('Frenet Transform main script')
    #print('Launching GN test.')
    #__test_GN()
    print('Launching Transform test.')
    __test_transform()
    exit(0)
