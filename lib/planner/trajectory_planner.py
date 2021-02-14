import numpy as np
from .planner import Planner 
from ..trajectory import QuinticPolynomial
from ..trajectory import QuarticPolynomial
from operator import attrgetter
import copy
from .frenet import Frenet

class ConfigurationParameterPlanner:
    ## planner 

    GLOBAL_D_T = 0.05
    MAX_ROAD_WIDTH = 2.45 # maximum road width [m]
    D_ROAD_W = 0.6 # road width sampling length [m]
    T_0 = 0  # initial time [s]

    D_T = 0.8 # time tick [s]
    MAX_T = 5.0 # max prediction time [m]
    MIN_T = 0.9 # min prediction time [m]

    DES_SPEED = 5.0 # speed desired [m/s]
    D_D_S = 1  # target speed sampling length [m/s]
    N_S_SAMPLE = 3  # sampling number of target speed
    LOW_SPEED_THRESH = 2.0 # low speed switch [m/s]
    S_THRSH = 1.0

    # Cost weights
    K_J = 0.001
    K_T = 0.01
    K_D = 0.9
    K_S = 0.1
    K_DOT_S = 1.0
    K_LAT = 1.0
    K_LONG = 1.0

    TARGET_DIST = 1 #[m] distance from following target 

dsp = ConfigurationParameterPlanner

class TrajectoryPlanner(Planner):
    def __init__(self, t0: (float), p0: np.array, s0: np.array, 
                        dd: float = 0, dsd: float = dsp.DES_SPEED, s_target: Frenet = None):
        
        assert abs(dd)<=dsp.MAX_ROAD_WIDTH, 'desired offset must be into road limits'
        self.delta_t = dsp.GLOBAL_D_T # planner sampling interval
        self.s0 = s0
        self.p0 = p0 # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.s_target = s_target
        self.kj = dsp.K_J # Jerk cost parameter
        self.kt = dsp.K_T # Temporal cost parameter
        self.kd = dsp.K_D # Offset d cost parameter
        self.ks = dsp.K_S
        self.kdots = dsp.K_DOT_S
        self.klong = dsp.K_LONG
        self.klat = dsp.K_LAT
        self.dd = dd
        self.dsd = dsp.DES_SPEED #desired speed
        self.t_initial = t0
        self.di_interval = (-dsp.MAX_ROAD_WIDTH,dsp.MAX_ROAD_WIDTH,dsp.D_ROAD_W) # Interval expressed as tuple (D_min, D_max, delta_d)
        self.t_interval = (dsp.MIN_T,dsp.MAX_T,dsp.D_T) # Interval expressed as tuple (T_min, T_max, delta_t)
        self.si_interval = (round(-dsp.D_D_S*dsp.N_S_SAMPLE),round(+dsp.D_D_S*dsp.N_S_SAMPLE+dsp.D_D_S),dsp.D_D_S)
        self.dsi_interval = (round(self.s0[1]-dsp.D_D_S*dsp.N_S_SAMPLE),round(self.s0[1]+dsp.D_D_S*dsp.N_S_SAMPLE+dsp.D_D_S),dsp.D_D_S) # Interval expressed as tuple (dsd - delta_dsi, dsd + delta_dsi, delta_s)
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

        self.si_interval = (round(-dsp.D_D_S*dsp.N_S_SAMPLE),round(+dsp.D_D_S*dsp.N_S_SAMPLE+dsp.D_D_S),dsp.D_D_S)
        self.dsi_interval = (round(ds0-dsp.D_D_S*dsp.N_S_SAMPLE),round(ds0+dsp.D_D_S*dsp.N_S_SAMPLE+dsp.D_D_S),dsp.D_D_S) # Interval expressed as tuple (dsd - delta_dsi, dsd + delta_dsi, delta_s)
 
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
                    if ds0 < dsp.LOW_SPEED_THRESH: # low speed
                        # if S>dsp.S_THRSH:
                        path = QuinticPolynomial(p0, dp0, ddp0, di, 0.0, 0.0, S)
                        f.d = [path.compute_pt(abs(s-s0)) for s in f.s]
                        f.dot_d = [path.compute_first_derivative(abs(s-s0)) for s in f.s]
                        f.ddot_d = [path.compute_second_derivative(abs(s-s0)) for s in f.s]
                        f.dddot_d = [path.compute_third_derivative(abs(s-s0)) for s in f.s]
                        squared_jerk = sum(np.power(f.dddot_d, 2))
                        C_lat = f.cd = self.kj * squared_jerk + self.kt * S + self.kd * (di-self.dd) ** 2 # Compute longitudinal cost low speed
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
                        C_lat = f.cd = self.kj * squared_jerk + self.kt * tj + self.kd * (di-self.dd) ** 2 # Compute longitudinal cost
                        f.ctot = self.klat * C_lat + self.klong * C_long
                        # Transform f.t into real time coordinates
                        for i in range(len(f.t)):
                            f.t[i] += self.t_initial
                        frenet_paths.append(f)
        return frenet_paths
    
    def optimal_at_time(self, time, opt_path, type_path) -> (float, float, float):
        if time <= opt_path.t[0]:
            print('Warning: requested time is out of optimal path time interval, increase the query time')
            index = 0
        elif time >= opt_path.t[-1]:
            print('Warning: requested time is out of optimal path time interval, reduce the query time')
            index = -1
        else:
            index = round((time - opt_path.t[0])/self.delta_t)
        if type_path == "s":
            return (opt_path.s[index], opt_path.dot_s[index], opt_path.ddot_s[index])
        else:
            return (opt_path.d[index], opt_path.dot_d[index], opt_path.ddot_d[index])

    def replan_ctot(self, time: float): # replan w.r.t opt_tot
        self.p0 = self.optimal_at_time(time, self.opt_path_tot, "d") # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.s0 = self.optimal_at_time(time, self.opt_path_tot, "s") # Initial step in frenet-frame as tuple (s0, ds0)
        # self.dsi_interval = (round(self.s0[1]-dsp.D_D_S*dsp.N_S_SAMPLE),round(self.s0[1]+dsp.D_D_S*dsp.N_S_SAMPLE)+dsp.D_D_S,dsp.D_D_S)
        self.t_initial = time
        self.paths = self.generate_range_polynomials()
        self.opt_path_tot = min(self.paths, key=attrgetter('ctot'))

    def step(self, time:float, dd: float = None, dsd: float = None, s_target: Frenet = None):
        # in order of priority
        if s_target != None: # follow
            self.s_target = s_target
        else:
            if dd != None: # collision
                self.dd = dd
            if dsd != None: # velocity keeping
                self.dsd = dsd
        self.replan_ctot(time=time)
        
        return np.array([self.p0[0], self.s0[0]])