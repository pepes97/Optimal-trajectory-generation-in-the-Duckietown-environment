"""unicycle.py
"""

import numpy as np

class Unicycle:
    def __init__(self, p0: np.array):
        """ Generates an unicycle at coordinates p0 wrt global frame RF0
        """
        assert p0.shape == (3,)
        self.p0 = p0
        self.p = self.p0

    def step(self, u: np.array, dt:float = 0.1) -> (np.array, np.array):
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
        vel = kinematicModel()
        self.p += dt * vel
        return self.p, vel
