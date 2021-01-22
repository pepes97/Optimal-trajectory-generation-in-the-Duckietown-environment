from Frenet import Frenet
from quintic_polynomial import QuinticPolynomial
from quartic_polynomial import QuarticPolynomial
import numpy as np
import matplotlib.pyplot as plt
from operator import attrgetter

# constants K

KJ = 0.1
KT = 0.1
KD = 1.0
KDOT = 1.0
KLAT = 1.0
KLON = 1.0

# for lateral motion 

DI_INTERVAL = 7 # [m]
STEP = 0.5 # [m]

MIN_TIME = 1.0 # [s]
MAX_TIME = 10.0 # [s]
STEP_TIME = 0.5 # [s]

# for longitudinal motion

DELTA_SI = 1 # [m/s]
TARGET_SPEED = 4 # [m/s]


def lateral_motionLST_and_longitudinal_motionVK(p0, dp0, ddp0, s0, ds0):

    """
        Lateral Movement and Longitudinal movement with velocity keeping

        Start state = [p0, dp0, ddp0] using for coordinate d of Franet Frame
        Start state = [s0, ds0] using for coordinate s of Franet Frame

    """

    frenet_paths = []
    for di in np.arange(-DI_INTERVAL, DI_INTERVAL, STEP):

        # Lateral motion planning

        for Tj in np.arange(MIN_TIME,MAX_TIME, STEP_TIME):
            
            f = Frenet()

            latQP = QuinticPolynomial(p0, dp0, ddp0, di, 0.0, 0.0, Tj)

            f.t = [t for t in np.arange(0.0, Tj, STEP_TIME)]
            f.d = [latQP.compute_pt(t) for t in f.t]
            f.dot_d = [latQP.compute_first_derivative(t) for t in f.t]
            f.ddot_d = [latQP.compute_second_derivative(t) for t in f.t]
            f.dddot_d = [latQP.compute_third_derivative(t) for t in f.t]

            # square jerk

            Jlat = sum(np.power(f.dddot_d, 2)) 

            # control

            f.cd = KJ * Jlat + KT * Tj + KD * f.d[-1] ** 2

            # Longitudinal motion planning 

            for si in np.arange (0, TARGET_SPEED + DELTA_SI):
                lonQP = QuarticPolynomial(s0, ds0, 0.0, si, 0.0, Tj)
                f.s = [lonQP.compute_st(t) for t in f.t]
                f.dot_s = [lonQP.compute_first_derivative(t) for t in f.t]
                f.ddot_s = [lonQP.compute_second_derivative(t) for t in f.t]
                f.dddot_s = [lonQP.compute_third_derivative(t) for t in f.t]

                # square of jerk

                Jt = sum(np.power(f.dddot_s, 2))  

                # control

                f.cv = KJ*Jt + KT*Tj + KDOT*((f.dot_s[-1]-TARGET_SPEED)**2)

                f.tot = KLAT * f.cd + KLON * f.cv

            frenet_paths.append(f)

    for i in range(len(frenet_paths)):
        f = frenet_paths[i]
        plt.plot(f.t,f.d)
        plt.xlabel("t/s")
        plt.ylabel("d/m")
    plt.show()
        
    return frenet_paths


class LateralTrajectoryPlanner:
    def __init__(self, p0: (float, float, float), t_initial:float, kj: float, kt: float, kd: float,
                 di_interval: (float, float, float), t_interval: (float, float, float)):
        self.p0 = p0 # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.t_initial = t_initial
        self.kj = kj # Jerk cost parameter
        self.kt = kt # Temporal cost parameter
        self.kd = kd # Offset d cost parameter
        self.di_interval = di_interval # Interval expressed as tuple (T_min, T_max, delta_t)
        self.t_interval = t_interval # Interval expressed as tuple (D_min, D_max, delta_d)
        self.delta_t = 0.05
        

    def generate_range_polynomials(self) -> [Frenet]:
        """ Generates a range of possible polynomials paths, each with its associated cost """
        frenet_paths = []
        p0 = self.p0[0]
        dp0 = self.p0[1]
        ddp0 = self.p0[2]
        for di in np.arange(self.di_interval[0], self.di_interval[1], self.di_interval[2]):
            for tj in np.arange(self.t_interval[0], self.t_interval[1], self.t_interval[2]):
                f = Frenet()
                path = QuinticPolynomial(p0, dp0, ddp0, di, 0, 0, tj)
                # Fill Frenet class
                f.t = [t for t in np.arange(0, tj, self.delta_t)]#self.t_interval[2])]
                f.d = [path.compute_pt(t) for t in f.t]
                f.dot_d = [path.compute_first_derivative(t) for t in f.t]
                f.ddot_d = [path.compute_second_derivative(t) for t in f.t]
                f.dddot_d = [path.compute_third_derivative(t) for t in f.t]
                squared_jerk = sum(np.power(f.dddot_d, 2))
                f.cd = self.kj * squared_jerk + self.kt * tj + self.kd * di ** 2 # Compute longitudinal cost
                # Transform f.t into real time coordinates
                for i in range(len(f.t)):
                    f.t[i] += self.t_initial
                frenet_paths.append(f)
        return frenet_paths

    def forward_optimal(self) -> (float, float, float):
        frenet_paths = self.generate_range_polynomials()
        best_path = min(frenet_paths, key=attrgetter('cd'))
        sampling_t = -1
        return ((best_path.d[sampling_t], best_path.dot_d[sampling_t], best_path.ddot_d[sampling_t]),
                len(best_path.d) * self.delta_t)
    
        
        

    
        
