import numpy as np
import logging
import copy
from operator import attrgetter
from .planner import Planner 
from ..trajectory import QuinticPolynomial, QuarticPolynomial
from .frenet import Frenet # Still needed ?
from ..logger import timeprofiledecorator # Profiling wrapper
from ..sensor import ObstacleTrajectory

logger = logging.getLogger(__name__)


class TrajectoryPlannerDefaultParamsDT:
    kj = 0.001
    ks = 0.01
    kd = 1.0
    kt = 0.01
    kdots = 1
    klong = 1
    klat  = 1
    delta_t = 1/30
    desired_speed = 1.0
    max_road_width = 0.1
    min_t = 20/30
    max_t = 1
    d_road_width = 0.1
    d_d_s = 1
    low_speed_threshold = 2
    s_threshold = 1
    target_distance = 1
    num_sample = 2    

tpdp = TrajectoryPlannerDefaultParamsDT

class TrajectoryPlannerParamsDT:
    """ Container for TrajectoryPlanner parameters
    """
    def __init__(self, *args, **kwargs):
        self.kj = tpdp.kj # Cost term for jerk
        self.ks = tpdp.ks # Cost term for longitudinal displacement
        self.kd = tpdp.kd # Cost term for lateral displacement
        self.klong = tpdp.klong # Cost term for longitudinal trajectory
        self.kdots = tpdp.kdots # Cost term for velocity keeping (longitudinal trajectory)
        self.klat  = tpdp.klat # Cost term for lateral trajectory
        self.kt = tpdp.kt 
        
        self.delta_t = tpdp.delta_t # Sampling time interval
        self.desired_speed = tpdp.desired_speed # Desired speed
        self.max_road_width = tpdp.max_road_width    # Maximum road width
        self.d_road_width = tpdp.d_road_width # step road width
        self.d_d_s = tpdp.d_d_s # step s longitunal trajectory
        self.min_t = tpdp.min_t # Minimum time displacement
        self.max_t = tpdp.max_t # Maximum time displacement       
        self.low_speed_threshold = tpdp.low_speed_threshold # threshold for low speed trajectory
        self.target_distance = tpdp.target_distance # distance for following, merging ad stopping trajectory
        self.num_sample = tpdp.num_sample # num sample si_interval (target) and di_interval (without target)
        self.s_target = None
        self.num_replan = 0
        
        self.__dict__.update(kwargs)
        ...
    def dict(self):
        return self.__dict__

class TrajectoryPlannerV1DT(Planner):
    def __init__(self, *args, **kwargs):
        # Load parameters if TrajectoryPlannerParams object is passed
        if args is not None and len(args) > 0:
            if isinstance(args[0], TrajectoryPlannerParamsDT):
                self.__dict__.update(args[0].__dict__)
        # Load parameters passed via arguments
        if kwargs is not None:
            self.__dict__.update(kwargs)

        #print(self.__dict__)
        
    def step(self, t: float, dd: float = None, dsd: float = None, s_target: Frenet = None) -> np.array:
        """ Returns the frenet target position at time t
        """
        pass
        
    def initialize(self, t0: (float), p0: (float, float, float), s0: (float, float, float), dd: float = 0, dsd: float = 5, s_target: ObstacleTrajectory = None):
        assert abs(dd)<=self.max_road_width, 'desired offset must be into road limits'
        self.s0 = s0
        self.p0 = p0 # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.s_target = s_target
        self.dd = dd
        self.t_initial = t0
        self.d_target = 2.
        self.di_interval = (-self.max_road_width,self.max_road_width,self.d_road_width) # Interval expressed as tuple (D_min, D_max, delta_d)
        self.t_interval = (self.min_t,self.max_t+self.delta_t,self.delta_t) # Interval expressed as tuple (T_min, T_max, delta_t)
        self.si_interval = (round(-self.d_d_s*self.num_sample),round(+self.d_d_s*self.num_sample+self.d_d_s),self.d_d_s)
        self.dsi_interval = (round(self.s0[1]-self.d_d_s*self.num_sample),round(self.s0[1]+self.d_d_s*self.num_sample+self.d_d_s),self.d_d_s) # Interval expressed as tuple (dsd - delta_dsi, dsd + delta_dsi, delta_s)
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
        #print(p0, dp0, ddp0, s0, ds0, dds0)

        self.si_interval = (round(-self.d_d_s*self.num_sample),round(+self.d_d_s*self.num_sample+self.d_d_s),self.d_d_s)
        self.dsi_interval = (round(ds0-self.d_d_s*self.num_sample),round(ds0+self.d_d_s*self.num_sample+self.d_d_s),self.d_d_s) # Interval expressed as tuple (dsd - delta_dsi, dsd + delta_dsi, delta_s)
        #self.dsi_interval = (round(ds0), round(ds0+self.d_d_s*self.num_sample), self.d_d_s)
 
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
                    time = self.t_initial + tj
                    st = s0 + self.s_target.compute_s(time) - self.d_target + self.delta_t * self.s_target.compute_ds(time)
                    dst = ds0 + self.s_target.compute_ds(time) - self.delta_t * self.s_target.compute_dds(time)
                    ddst = dds0 + self.s_target.compute_dds(time) #- self.delta_t * self.s_target.compute_ddds(time)
                    # print(st,dst,ddst)
                    path_long = QuinticPolynomial(s0, ds0, dds0, st + si_dsi, dst, ddst, tj)
                    ft.t = [t for t in np.arange(0, tj+self.delta_t, self.delta_t)]
                    ft.s = [path_long.compute_pt(t) for t in ft.t]
                    ft.dot_s = [path_long.compute_first_derivative(t) for t in ft.t]
                    ft.ddot_s = [path_long.compute_second_derivative(t) for t in ft.t]
                    ft.dddot_s = [path_long.compute_third_derivative(t) for t in ft.t]
                    squared_jerklong = sum(np.power(ft.dddot_s, 2))
                    C_long = ft.ct = self.kj * squared_jerklong + self.kt * tj + self.ks * (ft.s[-1] - st) ** 2 
                else:
                    path_long = QuarticPolynomial(s0, ds0, dds0, si_dsi, 0.0, tj)
                    ft.t = [t for t in np.arange(0, tj+self.delta_t, self.delta_t)]
                    ft.s = [path_long.compute_pt(t) for t in ft.t]
                    ft.dot_s = [path_long.compute_first_derivative(t) for t in ft.t]
                    ft.ddot_s = [path_long.compute_second_derivative(t) for t in ft.t]
                    ft.dddot_s = [path_long.compute_third_derivative(t) for t in ft.t]
                    squared_jerklong = sum(np.power(ft.dddot_s, 2))
                    C_long = ft.cv = self.kj * squared_jerklong + self.kt * tj + self.kdots * (ft.dot_s[-1] - self.desired_speed) ** 2 
                S = abs(ft.s[-1] - s0)
                for di in np.arange(self.di_interval[0], self.di_interval[1], self.di_interval[2]):
                    f = copy.deepcopy(ft)
                    # Fill Frenet class for d
                    if ds0 < self.low_speed_threshold and S>0 and False: # low speed
                        path = QuinticPolynomial(p0, dp0, ddp0, di, 0.0, 0.0, S)
                        f.d = [path.compute_pt(abs(s-s0)) for s in f.s]
                        f.dot_d = [path.compute_first_derivative(abs(s-s0)) for s in f.s]
                        f.ddot_d = [path.compute_second_derivative(abs(s-s0)) for s in f.s]
                        f.dddot_d = [path.compute_third_derivative(abs(s-s0)) for s in f.s]
                        squared_jerk = sum(np.power(f.dddot_d, 2))
                        C_lat = f.cd = self.kj * squared_jerk + self.kt * S + self.kd * (di-self.dd) ** 2 # Compute longitudinal cost low speed
                        f.ctot = self.klat * C_lat + self.klong * C_long
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
            print(f'Warning: requested time {time} is out of optimal path time interval, increase the query time')
            index = 0
        elif time >= opt_path.t[-1]:
            print(f'Warning: requested time {time} is out of optimal path time interval, reduce the query time')
            index = -1
        else:
            index = round((time - opt_path.t[0])/self.delta_t)
        if type_path == "s":
            return (opt_path.s[index], opt_path.dot_s[index], opt_path.ddot_s[index])
        else:
            return (opt_path.d[index], opt_path.dot_d[index], opt_path.ddot_d[index])

    def replan_ct(self, time: float, s_target: ObstacleTrajectory = None): # replan w.r.t opt_ct
        self.p0 = self.optimal_at_time(time, self.opt_path_ct, "d") # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.s0 = self.optimal_at_time(time, self.opt_path_ct, "s") # Initial step in frenet-frame as tuple (s0, ds0)
        self.t_initial = time
        if s_target != None:
            self.s_target = s_target
        self.paths = self.generate_range_polynomials()
        self.opt_path_ct = min(self.paths, key=attrgetter('ct'))

    def replan_ctot(self, time: float): # replan w.r.t opt_tot
        self.p0 = self.optimal_at_time(time, self.opt_path_tot, "d") # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.s0 = self.optimal_at_time(time, self.opt_path_tot, "s") # Initial step in frenet-frame as tuple (s0, ds0)
        # self.dsi_interval = (round(self.s0[1]-cfg.D_D_S*cfg.N_S_SAMPLE),round(self.s0[1]+cfg.D_D_S*cfg.N_S_SAMPLE)+cfg.D_D_S,cfg.D_D_S)
        self.t_initial = time
        self.paths = self.generate_range_polynomials()
        self.opt_path_tot = min(self.paths, key=attrgetter('ctot'))
        
    def replan_cd_cv(self, time: float): # replan w.r.t opt_d and opt_s
        self.p0 = self.optimal_at_time(time, self.opt_path_d, "d") # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.s0 = self.optimal_at_time(time, self.opt_path_s, "s") # Initial step in frenet-frame as tuple (s0, ds0)
        # self.dsi_interval = (round(self.s0[1]-cfg.D_D_S*cfg.N_S_SAMPLE),round(self.s0[1]+cfg.D_D_S*cfg.N_S_SAMPLE)+cfg.D_D_S,cfg.D_D_S)
        self.t_initial = time
        self.paths = self.generate_range_polynomials()
        self.opt_path_d = min(self.paths, key=attrgetter('cd'))
        self.opt_path_s = min(self.paths, key=attrgetter('cv'))

    def replanner(self, time: float, dd: float = None, dsd: float = None, s_target: ObstacleTrajectory = None):
        # in order of priority
        if s_target != None: # follow
            self.s_target = s_target
        else:
            if dd != None: # collision
                self.dd = dd
            if dsd != None: # velocity keeping
                self.desired_speed = dsd
        if time <= self.opt_path_tot.t[0] or time >= self.opt_path_tot.t[-2]:
            self.replan_ctot(time=time)
        else:
            self.p0 = self.optimal_at_time(time, self.opt_path_tot, "d") # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
            self.s0 = self.optimal_at_time(time, self.opt_path_tot, "s")
        return self.s0, self.p0 #, self.opt_path_tot
