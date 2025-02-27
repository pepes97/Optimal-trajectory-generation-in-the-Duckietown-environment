import numpy as np
from diff_function import DifferentiableFunction

class QuinticPolynomial(DifferentiableFunction):

    def __init__(self, p0,dot_p0,ddot_p0,p1,dot_p1,ddot_p1, T):

      """
            Start state P0 = [p0, dot_p0, ddot_p0]
            Final state P1 = [p1, dot_p1, ddot_p1]
            T time 
      """
      super().__init__()

      self.a0 = p0
      self.a1 = dot_p0
      self.a2 = ddot_p0/2

      A = np.array([[T**3, T**4, T**5],
                    [3*T**2, 4*T**3, 5*T**4],
                    [6*T, 12*T**2, 20*T**3]])
    
      b = np.array([p1 - self.a0 - self.a1 *T - self.a2 *T**2,
                  dot_p1 - self.a1 - 2*self.a2*T,
                  ddot_p1-2*self.a2])
      
      x = np.linalg.solve(A,b)

      self.a3 = x[0]
      self.a4 = x[1]
      self.a5 = x[2]
 
    def compute_pt(self, t):
      """
            Compute pt given time t
      """
      pt = self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5
      return pt

    def compute_first_derivative(self, t):
      """
            Compute first derivative given time t
      """

      dot_pt = self.a1 + 2*self.a2*t + 3 * self.a3*t**2 + 4 * self.a4*t**3+ 5 * self.a5*t**4
      return dot_pt


    def compute_second_derivative(self, t):
      """
            Compute second derivative given time t
      """

      ddot_pt = 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3
      return ddot_pt

    def compute_third_derivative(self, t):
      """
            Compute third derivative given time t
      """
      dddot_pt = 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2
      return dddot_pt

