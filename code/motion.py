from Frenet import Frenet
from quintic_polynomial import QuinticPolynomial
from quartic_polynomial import QuarticPolynomial
import numpy as np
import matplotlib.pyplot as plt
from operator import attrgetter
import copy
import config as cfg


class FrenetTrajectoryPlanner:
    def __init__(self, p0: (float, float, float), s0: (float, float, float), s_target: Frenet = None):
        self.delta_t = cfg.GLOBAL_D_T # planner sampling interval
        self.s0 = s0
        self.p0 = p0 # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.s_target = s_target
        self.kj = cfg.K_J # Jerk cost parameter
        self.kt = cfg.K_T # Temporal cost parameter
        self.kd = cfg.K_D # Offset d cost parameter
        self.ks = cfg.K_S
        self.kdots = cfg.K_DOT_S
        self.klong = cfg.K_LONG
        self.klat = cfg.K_LAT
        self.dsd = cfg.DES_SPEED
        self.t_initial = cfg.T_0
        self.di_interval = (-cfg.MAX_ROAD_WIDTH,cfg.MAX_ROAD_WIDTH,cfg.D_ROAD_W) # Interval expressed as tuple (D_min, D_max, delta_d)
        self.t_interval = (cfg.MIN_T,cfg.MAX_T,cfg.D_T) # Interval expressed as tuple (T_min, T_max, delta_t)
        self.si_interval = (-cfg.D_D_S*cfg.N_S_SAMPLE,cfg.D_D_S*cfg.N_S_SAMPLE+cfg.D_D_S,cfg.D_D_S)
        self.dsi_interval = (-cfg.D_D_S*cfg.N_S_SAMPLE,cfg.D_D_S*cfg.N_S_SAMPLE+cfg.D_D_S,cfg.D_D_S) # Interval expressed as tuple (dsd - delta_dsi, dsd + delta_dsi, delta_s)
        self.paths = self.generate_range_polynomials(); # store current paths
        self.opt_path_d = min(self.paths, key=attrgetter('cd')); # store the best path for d
        self.opt_path_s = min(self.paths, key=attrgetter('cv')); # store the best path for s
        self.opt_path_tot = min(self.paths, key=attrgetter('ctot')); # store the best path for s
        if self.s_target != None:
            self.opt_path_ct = min(self.paths, key=attrgetter('ct')); # store the best path for s

        self.opt_path_cd = min(self.paths, key=attrgetter('cd')); # store the best path for d
        self.opt_path_cv = min(self.paths, key=attrgetter('cv')); # store the best path for s
        

    def generate_range_polynomials(self) -> [Frenet]:
        """ Generates a range of posdsible polynomials paths, each with its associated cost """
        frenet_paths = []
        p0 = self.p0[0]
        dp0 = self.p0[1]
        ddp0 = self.p0[2]
        s0 = self.s0[0]
        ds0 = self.s0[1]
        dds0 = self.s0[2]

        self.si_interval = (round(-cfg.D_D_S*cfg.N_S_SAMPLE),round(+cfg.D_D_S*cfg.N_S_SAMPLE+cfg.D_D_S),cfg.D_D_S)
        self.dsi_interval = (round(ds0-cfg.D_D_S*cfg.N_S_SAMPLE),round(ds0+cfg.D_D_S*cfg.N_S_SAMPLE+cfg.D_D_S),cfg.D_D_S) # Interval expressed as tuple (dsd - delta_dsi, dsd + delta_dsi, delta_s)
 
        if self.s_target != None:
            long_interval = np.arange(self.si_interval[0], self.si_interval[1], self.si_interval[2])
        else:
            long_interval = np.arange(self.dsi_interval[0], self.dsi_interval[1], self.dsi_interval[2])
        # Lateral
        for si_dsi in long_interval:
            for tj in np.arange(self.t_interval[0], self.t_interval[1], self.t_interval[2]):
                # Fill Frenet class for s
                ft = Frenet()
                if self.s_target != None:
                    index = round((self.t_initial + tj - self.s_target.t[0])/self.delta_t)
                    st = self.s_target.s[index]
                    dst = self.s_target.dot_s[index]
                    ddst = self.s_target.ddot_s[index]
                    path_long = QuinticPolynomial(s0, ds0, dds0, st + si_dsi, dst, ddst, tj)
                    ft.t = [t for t in np.arange(0, tj+self.delta_t, self.delta_t)]
                    ft.s = [path_long.compute_pt(t) for t in ft.t]
                    ft.dot_s = [path_long.compute_first_derivative(t) for t in ft.t]
                    ft.ddot_s = [path_long.compute_second_derivative(t) for t in ft.t]
                    ft.dddot_s = [path_long.compute_third_derivative(t) for t in ft.t]
                    squared_jerklong = sum(np.power(ft.dddot_s, 2))
                    C_long = ft.ct = self.kj * squared_jerklong + self.kt * tj + self.ks * (ft.s[-1] - st) ** 2 
                else:
                    path_long = QuarticPolynomial(s0, ds0, dds0, si_dsi, 0.0, tj) #self.dsd +
                    ft.t = [t for t in np.arange(0, tj+self.delta_t, self.delta_t)]
                    ft.s = [path_long.compute_pt(t) for t in ft.t]
                    ft.dot_s = [path_long.compute_first_derivative(t) for t in ft.t]
                    ft.ddot_s = [path_long.compute_second_derivative(t) for t in ft.t]
                    ft.dddot_s = [path_long.compute_third_derivative(t) for t in ft.t]
                    squared_jerklong = sum(np.power(ft.dddot_s, 2))
                    C_long = ft.cv = self.kj * squared_jerklong + self.kt * tj + self.kdots * (ft.dot_s[-1] - self.dsd) ** 2 
                S = abs(ft.s[-1] - s0)
                for di in np.arange(self.di_interval[0], self.di_interval[1], self.di_interval[2]):
                    f = copy.deepcopy(ft)
                    # Fill Frenet class for d
                    if ds0 < cfg.LOW_SPEED_THRESH: # low speed
                        # if S>cfg.S_THRSH:
                        path = QuinticPolynomial(p0, dp0, ddp0, di, 0.0, 0.0, S)
                        f.d = [path.compute_pt(abs(s-s0)) for s in f.s]
                        f.dot_d = [path.compute_first_derivative(abs(s-s0)) for s in f.s]
                        f.ddot_d = [path.compute_second_derivative(abs(s-s0)) for s in f.s]
                        f.dddot_d = [path.compute_third_derivative(abs(s-s0)) for s in f.s]
                        squared_jerk = sum(np.power(f.dddot_d, 2))
                        C_lat = f.cd = self.kj * squared_jerk + self.kt * S + self.kd * di ** 2 # Compute longitudinal cost low speed
                        f.ctot = self.klat * C_lat + self.klong * C_long
                        # Transform f.t into real time coordinates
                        for i in range(len(f.t)):
                            f.t[i] += self.t_initial
                        frenet_paths.append(f)
                    else: # high speed
                        path = QuinticPolynomial(p0, dp0, ddp0, di, 0.0, 0.0, tj)
                        f.d = [path.compute_pt(t) for t in f.t]
                        f.dot_d = [path.compute_first_derivative(t) for t in f.t]
                        f.ddot_d = [path.compute_second_derivative(t) for t in f.t]
                        f.dddot_d = [path.compute_third_derivative(t) for t in f.t]
                        squared_jerk = sum(np.power(f.dddot_d, 2))
                        C_lat = f.cd = self.kj * squared_jerk + self.kt * tj + self.kd * di ** 2 # Compute longitudinal cost
                        f.ctot = self.klat * C_lat + self.klong * C_long
                        # Transform f.t into real time coordinates
                        for i in range(len(f.t)):
                            f.t[i] += self.t_initial
                        frenet_paths.append(f)
        return frenet_paths

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
    
    def replan_ct(self, time, replan_interval=None): # replan w.r.t opt_ct
        self.p0 = self.optimal_at_time(time, self.opt_path_ct, "d") # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.s0 = self.optimal_at_time(time, self.opt_path_ct, "s") # Initial step in frenet-frame as tuple (s0, ds0)
        self.t_initial = time
        self.paths = self.generate_range_polynomials()
        self.opt_path_ct = min(self.paths, key=attrgetter('ct'))

    def replan_ctot(self, time, replan_interval=None): # replan w.r.t opt_tot
        self.p0 = self.optimal_at_time(time, self.opt_path_tot, "d") # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.s0 = self.optimal_at_time(time, self.opt_path_tot, "s") # Initial step in frenet-frame as tuple (s0, ds0)
        # self.dsi_interval = (round(self.s0[1]-cfg.D_D_S*cfg.N_S_SAMPLE),round(self.s0[1]+cfg.D_D_S*cfg.N_S_SAMPLE)+cfg.D_D_S,cfg.D_D_S)
        self.t_initial = time
        self.paths = self.generate_range_polynomials()
        self.opt_path_tot = min(self.paths, key=attrgetter('ctot'))
        
    def replan_cd_cv(self, time, replan_interval=None): # replan w.r.t opt_d and opt_s
        self.p0 = self.optimal_at_time(time, self.opt_path_d, "d") # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.s0 = self.optimal_at_time(time, self.opt_path_s, "s") # Initial step in frenet-frame as tuple (s0, ds0)
        # self.dsi_interval = (round(self.s0[1]-cfg.D_D_S*cfg.N_S_SAMPLE),round(self.s0[1]+cfg.D_D_S*cfg.N_S_SAMPLE)+cfg.D_D_S,cfg.D_D_S)
        self.t_initial = time
        self.paths = self.generate_range_polynomials()
        self.opt_path_d = min(self.paths, key=attrgetter('cd'))
        self.opt_path_s = min(self.paths, key=attrgetter('cv'))

    def replan_cd(self, time, replan_interval=None): # replan w.r.t opt_d and opt_s
        self.p0 = self.optimal_at_time(time, self.opt_path_cd, "d") # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.s0 = self.optimal_at_time(time, self.opt_path_cd, "s") # Initial step in frenet-frame as tuple (s0, ds0)
        self.t_initial = time
        self.paths = self.generate_range_polynomials()
        self.opt_path_cd = min(self.paths, key=attrgetter('cd'))
  
    def replan_cv(self, time, replan_interval=None): # replan w.r.t opt_d and opt_s
        self.p0 = self.optimal_at_time(time, self.opt_path_cv, "d") # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.s0 = self.optimal_at_time(time, self.opt_path_cv, "s") # Initial step in frenet-frame as tuple (s0, ds0)
        self.t_initial = time
        self.paths = self.generate_range_polynomials()
        self.opt_path_cv = min(self.paths, key=attrgetter('cv'))
       