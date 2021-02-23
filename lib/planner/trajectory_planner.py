import numpy as np
import logging
import copy
from operator import attrgetter
from .planner import Planner 
from ..trajectory import QuinticPolynomial, QuarticPolynomial
from .frenet import Frenet # Still needed ?
from ..logger import timeprofiledecorator # Profiling wrapper

logger = logging.getLogger(__name__)




class TrajectoryPlanner(Planner):
    def __init__(self, GLOBAL_D_T, K_J, K_T, K_D, K_S, K_DOT_S, K_LONG, K_LAT, DES_SPEED, MAX_ROAD_WIDTH, D_ROAD_W,
                        MIN_T,MAX_T, D_T, D_D_S, N_S_SAMPLE, LOW_SPEED_THRESH):
        
        #assert abs(dd)<=MAX_ROAD_WIDTH, 'desired offset must be into road limits'
        self.delta_t = GLOBAL_D_T # planner sampling interval
        self.kj = K_J # Jerk cost parameter
        self.kt = K_T # Temporal cost parameter
        self.kd = K_D # Offset d cost parameter
        self.ks = K_S
        self.kdots = K_DOT_S
        self.klong = K_LONG
        self.klat = K_LAT
        self.dsd = DES_SPEED #desired speed
        self.MAX_ROAD_WIDTH = MAX_ROAD_WIDTH
        self.D_ROAD_W = D_ROAD_W
        self.MIN_T = MIN_T
        self.MAX_T = MAX_T
        self.D_T = D_T
        self.DES_SPEED = DES_SPEED
        self.D_D_S = D_D_S
        self.LOW_SPEED_THRESH = LOW_SPEED_THRESH
        self.N_S_SAMPLE = N_S_SAMPLE
        self.di_interval = (-self.MAX_ROAD_WIDTH,self.MAX_ROAD_WIDTH,self.D_ROAD_W) # Interval expressed as tuple (D_min, D_max, delta_d)
        self.t_interval = (self.MIN_T,self.MAX_T,self.D_T) # Interval expressed as tuple (T_min, T_max, delta_t)
        self.si_interval = (round(-self.D_D_S*self.N_S_SAMPLE),round(+self.D_D_S*self.N_S_SAMPLE+self.D_D_S),self.D_D_S)

        # self.paths = self.generate_range_polynomials() # store current paths
        # self.opt_path_d = min(self.paths, key=attrgetter('cd')); # store the best path for d
        # self.opt_path_s = min(self.paths, key=attrgetter('cv')); # store the best path for s
        # self.opt_path_tot = min(self.paths, key=attrgetter('ctot')); # store the best path for s
        # if self.s_target != None:
        #     self.opt_path_ct = min(self.paths, key=attrgetter('ct')); # store the best path for s

        # self.opt_path_cd = min(self.paths, key=attrgetter('cd')); # store the best path for d
        # self.opt_path_cv = min(self.paths, key=attrgetter('cv')); # store the best path for s

    def optimal_path(self, t0: (float), p0: np.array, s0: np.array, 
                        dd: float = 0, s_target: Frenet = None):
        
        self.p0 = p0[0]
        self.dp0 = p0[1]
        self.ddp0 = p0[2]
        self.s0 = s0[0]
        self.ds0 = s0[1]
        self.dds0 = s0[2]
        self.dd = dd
        self.t_initial = t0
        self.s_target = s_target

        self.si_interval = (round(-self.D_D_S*self.N_S_SAMPLE),round(+self.D_D_S*self.N_S_SAMPLE+self.D_D_S),self.D_D_S)
        self.dsi_interval = (round(self.ds0-self.D_D_S*self.N_S_SAMPLE),round(self.ds0+self.D_D_S*self.N_S_SAMPLE+self.D_D_S),self.D_D_S) # Interval expressed as tuple (dsd - delta_dsi, dsd + delta_dsi, delta_s)

        self.paths = self.generate_range_polynomials() # store current paths
        self.opt_path_d = min(self.paths, key=attrgetter('cd')); # store the best path for d
        self.opt_path_s = min(self.paths, key=attrgetter('cv')); # store the best path for s
        self.opt_path_tot = min(self.paths, key=attrgetter('ctot')); # store the best path for s
        if self.s_target != None:
            self.opt_path_ct = min(self.paths, key=attrgetter('ct')); # store the best path for s

        self.opt_path_cd = min(self.paths, key=attrgetter('cd')); # store the best path for d
        self.opt_path_cv = min(self.paths, key=attrgetter('cv')); # store the best path for s

        return self.opt_path_tot
    
    def generate_range_polynomials(self) -> [Frenet]:
        """ Generates a range of posdsible polynomials paths, each with its associated cost """
        frenet_paths = []
    
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
                    path_long = QuinticPolynomial(self.s0, self.ds0, self.dds0, st + si_dsi, dst, ddst, tj)
                    ft.t = [t for t in np.arange(0, tj+self.delta_t, self.delta_t)]
                    ft.s = [path_long.compute_pt(t) for t in ft.t]
                    ft.dot_s = [path_long.compute_first_derivative(t) for t in ft.t]
                    ft.ddot_s = [path_long.compute_second_derivative(t) for t in ft.t]
                    ft.dddot_s = [path_long.compute_third_derivative(t) for t in ft.t]
                    squared_jerklong = sum(np.power(ft.dddot_s, 2))
                    C_long = ft.ct = self.kj * squared_jerklong + self.kt * tj + self.ks * (ft.s[-1] - st) ** 2 
                else:
                    path_long = QuarticPolynomial(self.s0, self.ds0, self.dds0, si_dsi, 0.0, tj) #self.dsd +
                    ft.t = [t for t in np.arange(0, tj+self.delta_t, self.delta_t)]
                    ft.s = [path_long.compute_pt(t) for t in ft.t]
                    ft.dot_s = [path_long.compute_first_derivative(t) for t in ft.t]
                    ft.ddot_s = [path_long.compute_second_derivative(t) for t in ft.t]
                    ft.dddot_s = [path_long.compute_third_derivative(t) for t in ft.t]
                    squared_jerklong = sum(np.power(ft.dddot_s, 2))
                    C_long = ft.cv = self.kj * squared_jerklong + self.kt * tj + self.kdots * (ft.dot_s[-1] - self.DES_SPEED) ** 2 
                S = abs(ft.s[-1] - self.s0)
                for di in np.arange(self.di_interval[0], self.di_interval[1], self.di_interval[2]):
                    f = copy.deepcopy(ft)
                    # Fill Frenet class for d
                    if self.ds0 < self.LOW_SPEED_THRESH: # low speed
                        # if S>dsp.S_THRSH:
                        path = QuinticPolynomial(self.p0, self.dp0, self.ddp0, di, 0.0, 0.0, S)
                        f.d = [path.compute_pt(abs(s-self.s0)) for s in f.s]
                        f.dot_d = [path.compute_first_derivative(abs(s-self.s0)) for s in f.s]
                        f.ddot_d = [path.compute_second_derivative(abs(s-self.s0)) for s in f.s]
                        f.dddot_d = [path.compute_third_derivative(abs(s-self.s0)) for s in f.s]
                        squared_jerk = sum(np.power(f.dddot_d, 2))
                        C_lat = f.cd = self.kj * squared_jerk + self.kt * S + self.kd * (di-self.dd) ** 2 # Compute longitudinal cost low speed
                        f.ctot = self.klat * C_lat + self.klong * C_long
                        # Transform f.t into real time coordinates
                        for i in range(len(f.t)):
                            f.t[i] += self.t_initial
                        frenet_paths.append(f)
                    else: # high speed
                        path = QuinticPolynomial(self.p0, self.dp0, self.ddp0, di, 0.0, 0.0, tj)
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
        
        return self.opt_path_tot

class TrajectoryPlannerDefaultParams:
    dt = 1
    kj = 0.001
    ks = 0.1
    kd = 0.9
    kt = .01
    kdots = 1
    klong = 1
    klat  = 1
    delta_t = 0.1
    desired_speed = 1.8
    max_road_width = 2.5
    min_t = 1
    max_t = 3
    d_road_width = 0.5
    d_d_s = 1
    low_speed_threshold = 2
    s_threshold = 1
    target_distance = 1
    num_sample = 2
    

tpdp = TrajectoryPlannerDefaultParams

class TrajectoryPlannerParams:
    """ Container for TrajectoryPlanner parameters
    """
    def __init__(self, *args, **kwargs):
        self.dt = tpdp.dt # Sampling time interval
        self.kj = tpdp.kj # Cost term for jerk
        self.ks = tpdp.ks # Cost term for longitudinal displacement
        self.kd = tpdp.kd # Cost term for lateral displacement
        self.klong = tpdp.klong # Cost term for longitudinal trajectory
        self.kdots = tpdp.kdots # Cost term for velocity keeping (longitudinal trajectory)
        self.klat  = tpdp.klat # Cost term for lateral trajectory
        self.kt = tpdp.kt 
        
        self.delta_t = tpdp.delta_t 
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

# Define inner path candidate
class PathCandidate():
    """ Object that represent a path candidate. 
    Contains costs term and trajectory"""
    def __init__(self, *args, **kwargs):
        self.longitudinal_trajectory = None
        self.lateral_trajectory = None
        self.cd = 0. # Longitudinal cost
        self.cv = 0. # Lateral cost
        self.ctot = 0 # Longitudinal + Lateral cost 

class TrajectoryPlannerV2(Planner):
    def __init__(self, *args, **kwargs):
        # Load parameters if TrajectoryPlannerParams object is passed
        if args is not None and len(args) > 0:
            if isinstance(args[0], TrajectoryPlannerParams):
                self.__dict__.update(args[0].__dict__)
        # Load parameters passed via arguments
        if kwargs is not None:
            self.__dict__.update(kwargs)
        print(self.__dict__)

    def step(self, t, robot_pose=None, robot_dpose=None, robot_ddpose=None, *args, **kwargs) -> np.array:
        """ Returns the frenet target position at time t
        """
        logger.error('Function not implemented yet')
        # Basic steps
        # take current time
        # if time exceedes local time boundaries, replan and get optimal trajectory
        # else take current optimal trajectory
        # Sample trajectory at the current time
        # return sample
        # TODO(Take requirements for obstacle avoidance into account)
        
        if t > self.num_replan*self.max_t-1:
            # TODO replanning
            if self.num_replan == 0:
                s0 = np.array([robot_pose[0],robot_dpose[0],robot_ddpose[0]])
                d0 = np.array([robot_pose[1],robot_dpose[1],robot_ddpose[1]])
                self.optimal_candidate= self.__generate_optimal_candidate(s0,d0, t)
                return s0,d0
            else:
                s,d = self.replan(t)
            self.num_replan +=1
        else:
            s,d = self.optimal_at_time(t, self.optimal_candidate)
            return s,d

    def replan(self, t_replan: float):
        """ Apply a replanning step to generate new possible trajectories
        """
        # TODO
        self.t_start = t_replan
        # Regenerate optimal path
        s,d = self.optimal_at_time(t_replan, self.optimal_candidate)
        self.optimal_candidate = self.__generate_optimal_candidate(s,d, t_replan)
        return self.s, self.d

    

    def __generate_lateral_candidates(self, p: np.array, *args, **kwargs) -> [PathCandidate]:
        """ Generates lateral PathCandidate(s)
        
        Parameters
        ----------
        p : np.array(3)
            Current robot position in frenet frame
        """
        candidate_lst = []
        d_sampling_interval = ... # Array containing three elements (d_min, d_max, delta_d)
        for di in np.arange(d_sampling_interval[0], d_sampling_interval[1], d_sampling_interval[2]):
            for tj in np.arange(self.min_t, self.max_t, self.dt):
                # Generate quintic polynomial trajectory
                trajectory = ...
                # Compute candidate attributes (cost, etc)
                # Build a PathCandidate object for the current configuration
                candidate = PathCandidate()
                candidate_lst.append(candidate)
        return candidate_lst

    def __generate_longitudinal_candidates(self, p: np.array, *args, **kwargs) -> [PathCandidate]:
        """ Generates longitudinal PathCandidate(s)
        
        Parameters
        ----------
        p : np.array(3)
            Current robot position in frenet frame
        """
        candidate_lst = []
        # TODO
        ...
        return candidate_lst

    @timeprofiledecorator
    def __generate_optimal_candidate(self, s:np.array, p:np.array, t: float, dd: float = 0)-> [PathCandidate]:
        """ Generate range polynomials for both longitudinal and lateral components, then search for
        the minimum cost one
        """
        self.s = s
        self.d = p
        self.dd = dd
        self.t_start = t

        candidates = self._generate_candidates()
        self.optimal_candidate = min(candidates, key=attrgetter('ctot')); # store the best path minimize total cost
        return self.optimal_candidate

    @timeprofiledecorator
    def _generate_candidates(self):
        """ Generate all possible candidates
        """
        self.di_interval = (-self.max_road_width,self.max_road_width,self.d_road_width) # Interval expressed as tuple (D_min, D_max, delta_d)
        self.t_interval = (self.min_t+self.t_start,self.max_t+self.t_start,self.dt) # Interval expressed as tuple (T_min, T_max, delta_t)
        self.si_interval = (round(-self.d_d_s*self.num_sample),round(+self.d_d_s*self.num_sample+self.d_d_s),self.d_d_s)
        self.dsi_interval = (round(self.s[1]-self.d_d_s*self.num_sample),round(self.s[1]+self.d_d_s*self.num_sample+self.d_d_s),self.d_d_s) # Interval expressed as tuple (dsd - delta_dsi, dsd + delta_dsi, delta_s)

        if self.s_target != None:
            long_interval = np.arange(self.si_interval[0], self.si_interval[1], self.si_interval[2])
        else:
            long_interval = np.arange(self.dsi_interval[0], self.dsi_interval[1], self.dsi_interval[2])
        
        candidate_lst = []
        for si_dsi in long_interval:
            ## Longitudinal trajectory
            for tj in np.arange(self.t_interval[0], self.t_interval[1], self.t_interval[2]):
                # Path candidate
                candidate = PathCandidate()
                # with target  
                if self.s_target != None:
                    # TODO  
                    ...
                else:
                    path_long = candidate.longitudinal_trajectory = QuarticPolynomial(self.s[0], self.s[1], self.s[2], si_dsi, 0.0, tj)
                    s1 = path_long.compute_pt(tj)
                    s0 = path_long.compute_pt(self.t_start)
                    dot_s1 = path_long.compute_first_derivative(tj)
                    ddot_s1 = np.true_divide(np.power(path_long.compute_second_derivative(tj), 3), 3)
                    ddot_s0 = np.true_divide(np.power(path_long.compute_second_derivative(self.t_start), 3), 3)
                    squared_jerklong = ddot_s1 - ddot_s0
                    # longitudinal cost (velocity keeping)
                    C_long = candidate.cv = self.kj * squared_jerklong + self.kt * tj + self.kdots * (dot_s1 - self.desired_speed) ** 2 
                S = abs(s1 - self.s[0])
                for di in np.arange(self.di_interval[0], self.di_interval[1], self.di_interval[2]):
                    candidate_d = copy.deepcopy(candidate)
                    # low speed
                    if self.s[1] < self.low_speed_threshold: 
                        # if S>dsp.S_THRSH:
                        path = candidate_d.lateral_trajectory = QuinticPolynomial(self.d[0], self.d[1], self.d[2], di, 0.0, 0.0, S)
                        ddot_d1 = np.true_divide(np.power(path.compute_second_derivative(abs(s1-self.s[0])), 3), 3)
                        ddot_d0 = np.true_divide(np.power(path.compute_second_derivative(abs(s0-self.s[0])), 3), 3) 
                        squared_jerk = ddot_d1-ddot_d0  
                        C_lat = candidate_d.cd = self.kj * squared_jerk + self.kt * S + self.kd * (di-self.dd) ** 2 # Compute longitudinal cost low speed
                        candidate_d.ctot = self.klat * C_lat + self.klong * C_long
                        candidate_lst.append(candidate_d)
                    # High speed
                    else:
                        path = candidate_d.lateral_trajectory = QuinticPolynomial(self.d[0], self.d[1], self.d[2], di, 0.0, 0.0, tj)
                        ddot_d1 = np.true_divide(np.power(path.compute_second_derivative(tj), 3), 3)
                        ddot_d0 = np.true_divide(np.power(path.compute_second_derivative(self.t_start), 3), 3) 
                        squared_jerk = ddot_d1-ddot_d0 
                        C_lat = candidate_d.cd = self.kj * squared_jerk + self.kt * tj + self.kd * (di-self.dd) ** 2 # Compute longitudinal cost
                        candidate_d.ctot = self.klat * C_lat + self.klong * C_long
                        candidate_lst.append(candidate_d)
            
        return candidate_lst
    
    def optimal_at_time(self, time : float, opt_path: PathCandidate):
        lateral_trajectory = opt_path.lateral_trajectory
        longitudinal_trajectory = opt_path.longitudinal_trajectory
        # s coordinate frenet
        s = lateral_trajectory.compute_pt(time)
        dot_s = lateral_trajectory.compute_first_derivative(time)
        ddot_s = lateral_trajectory.compute_second_derivative(time)
        # d coordinate frenet
        d = longitudinal_trajectory.compute_pt(time)
        dot_d = longitudinal_trajectory.compute_first_derivative(time)
        ddot_d = longitudinal_trajectory.compute_second_derivative(time)
        return np.array([s,dot_s, ddot_s]), np.array([d,dot_d,ddot_d])
        

class TrajectoryPlannerV1(Planner):
    def __init__(self, *args, **kwargs):
        # Load parameters if TrajectoryPlannerParams object is passed
        if args is not None and len(args) > 0:
            if isinstance(args[0], TrajectoryPlannerParams):
                self.__dict__.update(args[0].__dict__)
        # Load parameters passed via arguments
        if kwargs is not None:
            self.__dict__.update(kwargs)

        #print(self.__dict__)
        
    def step(self, t: float, dd: float = None, dsd: float = None, s_target: Frenet = None) -> np.array:
        """ Returns the frenet target position at time t
        """
        pass
        
    def initialize(self, t0: (float), p0: (float, float, float), s0: (float, float, float), dd: float = 0, dsd: float = 5, s_target: Frenet = None):
        assert abs(dd)<=self.max_road_width, 'desired offset must be into road limits'
        self.s0 = s0
        self.p0 = p0 # Initial step in frenet-frame as tuple (p0, dp0, ddp0)
        self.s_target = s_target
        self.dd = dd
        self.t_initial = t0
        self.di_interval = (-self.max_road_width,self.max_road_width,self.d_road_width) # Interval expressed as tuple (D_min, D_max, delta_d)
        self.t_interval = (self.min_t,self.max_t,self.dt) # Interval expressed as tuple (T_min, T_max, delta_t)
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
                    C_long = ft.cv = self.kj * squared_jerklong + self.kt * tj + self.kdots * (ft.dot_s[-1] - self.desired_speed) ** 2 
                S = abs(ft.s[-1] - s0)
                for di in np.arange(self.di_interval[0], self.di_interval[1], self.di_interval[2]):
                    f = copy.deepcopy(ft)
                    # Fill Frenet class for d
                    if ds0 < self.low_speed_threshold and S>0: # low speed
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

    def replan_ct(self, time: float, s_target: Frenet = None): # replan w.r.t opt_ct
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

    def replanner(self, time: float, dd: float = None, dsd: float = None, s_target: Frenet = None):
        # in order of priority
        if s_target != None: # follow
            self.s_target = s_target
        else:
            if dd != None: # collision
                self.dd = dd
            if dsd != None: # velocity keeping
                self.desired_speed = dsd
        self.replan_ctot(time=time)

        return self.s0, self.p0 #, self.opt_path_tot
