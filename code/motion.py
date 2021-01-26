from Frenet import Frenet
from quintic_polynomial import QuinticPolynomial
from quartic_polynomial import QuarticPolynomial
import numpy as np
import matplotlib.pyplot as plt
from operator import attrgetter
import copy
# constants K

# KJ = 0.1
# KT = 0.1
# KD = 1.0
# KDOT = 1.0
# KLAT = 1.0
# KLON = 1.0

# # for lateral motion 

# DI_INTERVAL = 7 # [m]
# STEP = 0.5 # [m]

# MIN_TIME = 1.0 # [s]
# MAX_TIME = 10.0 # [s]
# STEP_TIME = 0.5 # [s]

# # for longitudinal motion

# DELTA_SI = 1 # [m/s]
# TARGET_SPEED = 4 # [m/s]


# def lateral_motionLST_and_longitudinal_motionVK(p0, dp0, ddp0, s0, ds0):

#     """
#         Lateral Movement and Longitudinal movement with velocity keeping

#         Start state = [p0, dp0, ddp0] using for coordinate d of Franet Frame
#         Start state = [s0, ds0] using for coordinate s of Franet Frame

#     """

#     frenet_paths = []
#     for di in np.arange(-DI_INTERVAL, DI_INTERVAL, STEP):

#         # Lateral motion planning

#         for Tj in np.arange(MIN_TIME,MAX_TIME, STEP_TIME):
            
#             f = Frenet()

#             latQP = QuinticPolynomial(p0, dp0, ddp0, di, 0.0, 0.0, Tj)

#             f.t = [t for t in np.arange(0.0, Tj, STEP_TIME)]
#             f.d = [latQP.compute_pt(t) for t in f.t]
#             f.dot_d = [latQP.compute_first_derivative(t) for t in f.t]
#             f.ddot_d = [latQP.compute_second_derivative(t) for t in f.t]
#             f.dddot_d = [latQP.compute_third_derivative(t) for t in f.t]

#             # square jerk

#             Jlat = sum(np.power(f.dddot_d, 2)) 

#             # control

#             f.cd = KJ * Jlat + KT * Tj + KD * f.d[-1] ** 2

#             # Longitudinal motion planning 

#             for si in np.arange (0, TARGET_SPEED + DELTA_SI):
#                 lonQP = QuarticPolynomial(s0, ds0, 0.0, si, 0.0, Tj)
#                 f.s = [lonQP.compute_st(t) for t in f.t]
#                 f.dot_s = [lonQP.compute_first_derivative(t) for t in f.t]
#                 f.ddot_s = [lonQP.compute_second_derivative(t) for t in f.t]
#                 f.dddot_s = [lonQP.compute_third_derivative(t) for t in f.t]

#                 # square of jerk

#                 Jt = sum(np.power(f.dddot_s, 2))  

#                 # control

#                 f.cv = KJ*Jt + KT*Tj + KDOT*((f.dot_s[-1]-TARGET_SPEED)**2)

#                 f.tot = KLAT * f.cd + KLON * f.cv

#             frenet_paths.append(f)

#     for i in range(len(frenet_paths)):
#         f = frenet_paths[i]
#         plt.plot(f.t,f.d)
#         plt.xlabel("t/s")
#         plt.ylabel("d/m")
#     plt.show()
        
#     return frenet_paths

SPEED_THRESHOLD = 2 # [m/s]

class LateralTrajectoryPlanner:
    def __init__(self, p0: (float, float, float), t_initial:float, kj: float, kt: float, kd: float,
                 di_interval: (float, float, float), t_interval: (float, float, float),
                 s0: (float, float, float), si_interval: (float, float, float), sd:float, kdot_s:float,
                 k_lat: float, k_long:float):
        self.p0 = p0 # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.t_initial = t_initial
        self.kj = kj # Jerk cost parameter
        self.kt = kt # Temporal cost parameter
        self.kd = kd # Offset d cost parameter
        self.di_interval = di_interval # Interval expressed as tuple (D_min, D_max, delta_d)
        self.t_interval = t_interval # Interval expressed as tuple (T_min, T_max, delta_t)
        self.delta_t = 0.05
        self.s0 = s0
        self.si_interval = si_interval # Interval expressed as tuple (sd - delta_si, sd + delta_si, delta_s)
        self.sd = sd
        self.kdot_s = kdot_s
        self.k_long = k_long
        self.k_lat = k_lat
        self.paths = self.generate_range_polynomials(); # store current paths
        self.opt_path_d = min(self.paths, key=attrgetter('cd')); # store the best path for d
        self.opt_path_s = min(self.paths, key=attrgetter('cv')); # store the best path for s
        self.opt_path_tot = min(self.paths, key=attrgetter('ctot')); # store the best path for s
        

    def generate_range_polynomials(self) -> [Frenet]:
        """ Generates a range of possible polynomials paths, each with its associated cost """
        frenet_paths = []
        p0 = self.p0[0]
        dp0 = self.p0[1]
        ddp0 = self.p0[2]
        s0 = self.s0[0]
        ds0 = self.s0[1]
        # Lateral
        for si in np.arange(self.si_interval[0], self.si_interval[1], self.si_interval[2]):
            
            for tj in np.arange(self.t_interval[0], self.t_interval[1], self.t_interval[2]):
            
                path_long = QuarticPolynomial(s0, ds0, 0.0, si, 0.0, tj)
                # Fill Frenet class for s
                ft = Frenet()
                ft.t = [t for t in np.arange(0, tj, self.delta_t)]
                ft.s = [path_long.compute_pt(t) for t in ft.t]
                ft.dot_s = [path_long.compute_first_derivative(t) for t in ft.t]
                ft.ddot_s = [path_long.compute_second_derivative(t) for t in ft.t]
                ft.dddot_s = [path_long.compute_third_derivative(t) for t in ft.t]
                squared_jerk_long = sum(np.power(ft.dddot_s, 2))
                ft.cv = self.kj * squared_jerk_long + self.kt * tj + self.kdot_s * (self.sd - si) ** 2 
                S = ft.s[-1] - s0
                
                for di in np.arange(self.di_interval[0], self.di_interval[1], self.di_interval[2]):
                    
                    f = copy.deepcopy(ft)
                    # Fill Frenet class for d
                    if ds0 < SPEED_THRESHOLD: # low speed
                        path = QuinticPolynomial(p0, dp0, ddp0, di, 0, 0, S)
                        f.d = [path.compute_pt(s) for s in f.s]
                        f.dot_d = [path.compute_first_derivative(s) for s in f.s]
                        f.ddot_d = [path.compute_second_derivative(s) for s in f.s]
                        f.dddot_d = [path.compute_third_derivative(s) for s in f.s]
                        squared_jerk = sum(np.power(f.dddot_d, 2))
                        f.cd = self.kj * squared_jerk + self.kt * S + self.kd * di ** 2 # Compute longitudinal cost low speed
                    else: # high speed
                        path = QuinticPolynomial(p0, dp0, ddp0, di, 0, 0, tj)
                        f.d = [path.compute_pt(t) for t in f.t]
                        f.dot_d = [path.compute_first_derivative(t) for t in f.t]
                        f.ddot_d = [path.compute_second_derivative(t) for t in f.t]
                        f.dddot_d = [path.compute_third_derivative(t) for t in f.t]
                        squared_jerk = sum(np.power(f.dddot_d, 2))
                        f.cd = self.kj * squared_jerk + self.kt * tj + self.kd * di ** 2 # Compute longitudinal cost
                    f.ctot = self.k_lat * squared_jerk + self.k_long * squared_jerk_long
                    # Transform f.t into real time coordinates
                    for i in range(len(f.t)):
                        f.t[i] += self.t_initial
                    frenet_paths.append(f)
                    
        return frenet_paths

    # def forward_optimal(self) -> ((float, float, float), float):
    #     sampling_t = -1
    #     return ((self.opt_path.d[sampling_t], self.opt_path.dot_d[sampling_t], self.opt_path.ddot_d[sampling_t]),
    #             len(self.opt_path.d) * self.delta_t) #self.t_interval[2])#
    
    def optimal_at_time(self, time, opt_path, type_path) -> (float, float, float):
        if time <= opt_path.t[0]:
            index = 0
        elif time >= opt_path.t[-1]:
            index = -1
        else:
            index = round((time - opt_path.t[0])/self.delta_t)
        if type_path == "s":
            return (opt_path.s[index], opt_path.dot_s[index], opt_path.ddot_s[index])
        else:
            return (opt_path.d[index], opt_path.dot_d[index], opt_path.ddot_d[index])
    
    def replan(self, time):
        self.p0 = self.optimal_at_time(time, self.opt_path_d, "d") # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.s0 = self.optimal_at_time(time, self.opt_path_s, "s") # Initial step in frenet-frame as tuple (s0, ds0)
        self.t_initial = time
        self.paths = self.generate_range_polynomials()
        self.opt_path_d = min(self.paths, key=attrgetter('cd'))
        self.opt_path_s = min(self.paths, key=attrgetter('cv'))
        self.opt_path_tot = min(self.paths, key=attrgetter('ctot'))