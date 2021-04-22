"""unicycle.py
"""

import numpy as np

class Unicycle:
    def __init__(self):
        """ Generates an unicycle at coordinates p0 wrt global frame RF0
        """
        self.p = np.array([0.0, 0.0, 0.0])

    def set_initial_pose(self, p0: np.array):
        self.p = p0
        self.p0 = p0

    def step(self, u: np.array, dt:float = 0.1) -> (np.array, np.array):
        """ Step the unicycle wrt global frame RF0 given the pair of control inputs
        u[0] : Tangential velocity control
        u[1] : Radial velocity control
        dt   : Delta of time for which control is applied (Default to 10 ms)
        """
        def Sin(d_theta):
            return 1 - 1/6 * d_theta**2 + 1/120 * d_theta**4 - 1/5040 * d_theta**6
        
        def Cos(d_theta):
            return 1/2 * d_theta - 1/24 * d_theta**3 + 1/720 * d_theta**5 - 1/40320 * d_theta**7
        
        # computes the pose 2d pose vector v from an homogeneous transform A
        # A:[ R t ] 3x3 homogeneous transformation matrix, r translation vector
        # v: [x,y,theta]  2D pose vector
        def t2v(A):
            v = np.zeros((3,))
            v[:2]=A[:2,2]
            v[2]=np.arctan2(A[1,0],A[0,0])
            return v

        # computes the homogeneous transform matrix A of the pose vector v
        # A:[ R t ] 3x3 homogeneous transformation matrix, r translation vector
        # v: [x,y,theta]  2D pose vector
        def v2t(v):
            c=np.cos(v[2])
            s=np.sin(v[2])
            A=np.array([[c, -s, v[0]],
                        [s,  c, v[1]],
                        [0,  0, 1  ]])
            return A

        def exactIntegrationModel():
            # exact integration using Taylor expansion up to 7th order
            v = u[0]
            w = u[1]
            d_rho = dt * v
            d_theta = dt * w
            dx = d_rho * Sin(d_theta)
            dy = d_rho * Cos(d_theta)
            # since we start from different position wrt the origin
            # we have to express the increment wrt a different frame
            Dx = np.array([dx,dy,d_theta])
            dp = t2v(v2t(self.p0)@v2t(Dx)) - self.p
            theta = np.arctan2(np.sin(dp[2:]),np.cos(dp[2:])) # need normalization
            dp = np.concatenate([dp[:2],theta],axis=0)
            return dp/dt # velocity
        
        def rungeKuttaIntegrationModel():
            t = self.p[2]
            v = u[0]
            w = u[1]
            return np.array([v * np.cos(t+w*dt*0.5), v * np.sin(t+w*dt*0.5), w])
        
        def eulerIntegrationModel():
            t = self.p[2]
            v = u[0]
            w = u[1]
            return np.array([v * np.cos(t), v * np.sin(t), w])

        assert u.shape == (2,)
        
        vel = exactIntegrationModel()
        
        self.p += dt * vel
        return self.p, vel

    def pose(self):
        return self.p