"""unicycle.py
"""

import numpy as np
from plot import plot_unicycle_path

class Unicycle:
    def __init__(self, p0: np.ndarray):
        """ Generates an unicycle at coordinates p0 wrt global frame RF0
        """
        assert p0.shape == (3,)
        self.p0 = p0
        self.p = self.p0

    def step(self, u: np.ndarray, dt:float = 0.1):
        """ Step the unicycle wrt global frame RF0 given the pair of control inputs
        u[0] : Tangential velocity control
        u[1] : Radial velocity control
        dt   : Delta of time for which control is applied (Default to 10 ms)
        """
        def kinematicModel():
            t = self.p[2]
            v = u[0]
            w = u[1]
            return np.array([v * np.cos(t), v * np.sin(t), w])

        assert u.shape == (2,)
        self.p += dt * kinematicModel()
        return self.p


if __name__ == '__main__':
    print('Unicycle test script.')
    P = np.array([0.0, 0.0, 0.0])
    LEN_SIMULATION = 1000
    print(f'Placing unicycle at position [{P[0]}, {P[1]}, {P[2]}]')
    print(f'Lenght of simulation: {LEN_SIMULATION} steps (assuming 0.1 s for each iteration)')
    print('Applying constant tangential velocity and slow sinusoidal radial velocity')
    robot = Unicycle(P)
    u_vect = np.zeros((2, LEN_SIMULATION), dtype=float)
    u_vect[0, :] = 0.5;
    for i in range(LEN_SIMULATION):
        rotation_step = i / (LEN_SIMULATION-1) * (2 * np.pi)
        u_vect[1, i] = np.sin(rotation_step) / 100
        

    pos_vect = np.zeros((3, LEN_SIMULATION))
    pos_vect[:, 0] = np.transpose(P)
    for i in range(LEN_SIMULATION):
        pos = robot.step(u_vect[:, i])
        pos_vect[:, i] = np.transpose(pos)

    plot_unicycle_path(pos_vect)

    

    

    

    
