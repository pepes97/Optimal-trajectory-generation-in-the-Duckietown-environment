"""frenet_transform.py
"""

import numpy as np
from diff_function import DifferentiableFunction


class FrenetTransform():
    def __init__(self, path: DifferentiableFunction=None, initial_guess: float=0.0, max_iter: int=10):
        self.path = path
        self.max_iter = max_iter
        self.proj_estimate = initial_guess
        pass

    def estimatePosition(self, p: np.array) -> (float, [float], [float]):
        """ Apply GN algorithm to find a good estimate for the orthogonal projection of p onto the
        path self.path. Also store the estimate in the class, to speed up convergence in the next
        iterations."""
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
        
        
    
    


from quintic_polynomial import QuinticPolynomial
from matplotlib import pyplot as plt

if __name__ == '__main__':
    print('Frenet transform test script.')
    T_end = 1
    ITERATIONS = 10
    path = QuinticPolynomial(0, -1, 0, 1, 0, 0, T_end)
    frenet_transform = FrenetTransform(path, initial_guess=0.8, max_iter=ITERATIONS)
    robot_position = np.array([0.4, 0., 0])
    robot_estimate_on_path, error_evolution, estimate_evolution= frenet_transform.estimatePosition(robot_position)
    print(f'Robot estimate position of path coordinates: {robot_estimate_on_path}')

    x = np.arange(0, 1, 0.05)
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
    
    
    
    
    
    
    exit(0)
