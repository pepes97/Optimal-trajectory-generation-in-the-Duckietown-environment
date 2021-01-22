import numpy as np

class QuarticPolynomial:

    def __init__(self, s0, dot_s0, ddot_s0, dot_s1, ddot_s1, T): 

        """
            Start state S0 = [s0, dot_s0, ddot_s0]
            Final state S1 = [dot_s1, ddot_s1]
            T time 
        """

        self.a0 = s0
        self.a1 = dot_s0
        self.a2 = ddot_s0 / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])

        b = np.array([dot_s1 - self.a1 - 2 * self.a2 * T,
                      ddot_s1 - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def compute_st(self, t):
        """
            Compute st given time t
        """

        st = self.a0 + self.a1*t + self.a2*t**2 + self.a3 *t** 3 + self.a4*t**4

        return st

    def compute_first_derivative(self, t):
        """
            Compute first derivative given time t
        """
        dot_st = self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3

        return dot_st

    def compute_second_derivative(self, t):
        """
            Compute second derivative given time t
        """

        ddot_st = 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2

        return ddot_st

    def compute_third_derivative(self, t):
        """
            Compute third derivative given time t
        """
        
        dddot_st = 6*self.a3 + 24*self.a4*t

        return dddot_st