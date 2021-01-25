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

    def estimatePosition(self, p: np.array) -> (float, [float]):
        """ Apply GN algorithm to find a good estimate for the orthogonal projection of p onto the
        path self.path. Also store the estimate in the class, to speed up convergence in the next
        iterations."""
        estimate = self.proj_estimate
        error_lst = []
        for iter in range(self.max_iter):
            # Distance between the point p and the path
            coordinate_err = np.array([estimate, self.path.compute_pt(estimate)]) - p[0:2] 
            error = np.linalg.norm(coordinate_err)
            error_lst.append(error * error)
            # First derivative of path at the estimate point
            J = np.array([1, self.path.compute_first_derivative(estimate)])
            J *= coordinate_err
            J /= np.sqrt(error)

            # Perturbation vector
            Ds = -J.T * error / np.matmul(J.T, J)
            estimate += Ds[0]
        # Final estimate is stored and returned
        self.proj_estimate = estimate
        return estimate, error_lst


from quintic_polynomial import QuinticPolynomial
from matplotlib import pyplot as plt

if __name__ == '__main__':
    print('Frenet transform test script.')
    T_end = 10
    ITERATIONS = 50
    path = QuinticPolynomial(1, -2, 0, 5, 1, 0, T_end)
    frenet_transform = FrenetTransform(path, initial_guess=0, max_iter=ITERATIONS)
    robot_position = np.array([5, 1, 0])
    robot_estimate_on_path, error_evolution = frenet_transform.estimatePosition(robot_position)
    print(f'Robot estimate position of path coordinates: {robot_estimate_on_path}')

    x = np.arange(0, 10, 0.1)
    y = np.array([path.compute_pt(s) for s in x])
    fig, axs = plt.subplots(2)
    
    axs[0].plot(x, y)
    axs[0].scatter(robot_position[0], robot_position[1], c='r')
    axs[0].scatter(robot_estimate_on_path, path.compute_pt(robot_estimate_on_path), c='g')
    axs[0].plot([robot_position[0], robot_estimate_on_path], [robot_position[1], path.compute_pt(robot_estimate_on_path)], 'r-')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Orthogonal projection of unicycle on path')
    axs[1].plot(np.arange(0, ITERATIONS, 1), np.array(error_evolution))
    axs[1].set_xlabel('iterations')
    axs[1].set_ylabel('chi_squares')
    plt.show()
    
    
    
    
    
    
    exit(0)
