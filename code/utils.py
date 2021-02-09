from Frenet import Frenet
from quintic_polynomial import QuinticPolynomial
from quartic_polynomial import QuarticPolynomial
import config as cfg
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from spline_planner import Spline2D
from motion import FrenetTrajectoryPlanner
from plot import plot_xy_paths_lst_ctot, plot_longitudinal_paths_lst, plot_lateral_paths_lst, plot_lateral_paths_lst_ctot, plot_longitudinal_paths_lst_ctot, plot_following_paths_lst_ctot,plot_following_paths_lst_ct
import copy

class Target():
    def __init__(self, s0: (float,float,float) = (0.0,0.0,0.0), s1:(float,float,float) = (1.0,1.0,1.0), Ts: float = 1.0):
        self.s0 = s0
        self.s1 = s1
        self.Ts = Ts #must be constant for every Target
        self.DTs = cfg.GLOBAL_D_T #must be constant for every Target
        self.update()
        
    def update(self):
        self.path = QuinticPolynomial(self.s0[0], self.s0[1], self.s0[2], self.s1[0], self.s1[1], self.s1[2], self.Ts)
        self.frenet = Frenet()
        self.frenet.t = [t for t in np.arange(0, self.Ts+self.DTs, self.DTs)]
        self.frenet.s = [self.path.compute_pt(t) for t in self.frenet.t]
        self.frenet.dot_s = [self.path.compute_first_derivative(t) for t in self.frenet.t]
        self.frenet.ddot_s = [self.path.compute_second_derivative(t) for t in self.frenet.t]
        self.frenet.dddot_s = [self.path.compute_third_derivative(t) for t in self.frenet.t]
        
    def get_frenet(self) -> Frenet:
        return copy.deepcopy(self.frenet)

    def get_frenet_follow_target(self, D0: float = cfg.TARGET_DIST) -> Frenet:
        # assert s0[2]==s1[2], 'leading Target must have constant acceleration'
        target = self.get_frenet()
        target.s = [s_lv - D0 + self.DTs * target.dot_s[t_index] for t_index,s_lv in enumerate(target.s)] # recompute s considering the distance
        target.dot_s = [dot_s_lv - self.DTs * target.ddot_s[t_index] for t_index,dot_s_lv in enumerate(target.dot_s)]
        if not all(target.ddot_s[0]==target.ddot_s[t_index] for t_index,_ in enumerate(target.t)): # not clear on paper if we should consider it or not
            target.ddot_s = [ddot_s_lv - self.DTs * target.dddot_s[t_index] for t_index,ddot_s_lv in enumerate(target.ddot_s)] # formal
        return target

    def get_frenet_merge_target(self, t_b = None) -> Frenet: # merge to another target # t_b: Target
        # assert they moves at same speed needed?
        assert t_b != None, 'please give a Target'
        target_a = self.get_frenet()
        target_b = t_b.get_frenet()
        assert self.DTs == t_b.DTs, 'sampling times must be equal'
        assert self.Ts == t_b.Ts, 'time intervals must be equal'
        target_a.s = [0.5*(s_a + s_b) for s_a, s_b  in zip(target_a.s, target_b.s)] 
        return target_a

    def get_frenet_stop_target(self) -> Frenet:
        # assert they moves at same speed needed?
        target = self.get_frenet()
        assert target.dot_s[-1] == 0, 'stopping velocity must be equal zero'
        assert target.ddot_s[-1] == 0, 'stopping acceleration must be equal zero'
        return target


def frenet_coordinates_xy(frenet_paths: Frenet, spline: Spline2D) -> Frenet:
    max_interval_s = 0
    for list_frenet in frenet_paths:
        for f in list_frenet:
            for i in range(len(f.s)):
               if f.s[-1]-f.s[0] > max_interval_s:
                   max_interval_s = f.s[-1]-f.s[0]
    for list_frenet in frenet_paths:
        for f in list_frenet:
            for i in range(len(f.s)):
                s = f.s[i]/(max_interval_s+1e-6)*(spline.s[-1]-spline.s[0]) # 1e-6 to avoid take exactly 1.0, spline is not defined for 1
                ix, iy = spline.calc_position(s)
                if ix ==None:
                    break
                i_yaw = spline.calc_yaw(s)
                fx = ix + f.d[i] * math.cos(i_yaw + math.pi / 2.0)
                fy = iy + f.d[i] * math.sin(i_yaw + math.pi / 2.0)

                f.x.append(fx)
                f.y.append(fy)
    return frenet_paths


def frenet_target_ctot(p:(float, float, float), s:(float, float, float), Tn:float, s_target: Frenet = None)-> Frenet:

    # initialize the planner
    lateral_planner = FrenetTrajectoryPlanner(p0 = p, s0=s, s_target = s_target)# s_target = s1_t

    frenet_paths = []

    for i,t in enumerate(Tn):
        lateral_planner.replan_ctot(t)
        
        print('d0: ', lateral_planner.p0, lateral_planner.t_initial)
        print('s0: ', lateral_planner.s0, lateral_planner.t_initial)
        frenet_paths.append(lateral_planner.paths)
        
    if s_target == None:
        wx = [-1.0,  25.5, 30.0, 40]
        wy = [-3.0, -5.5, -4.0, -3]
        spline = Spline2D(wx, wy)

        frenet_paths = frenet_coordinates_xy(frenet_paths, spline)

    return frenet_paths

def frenet_cd_cv(p:(float, float, float), s:(float, float, float), Tn:float) -> Frenet:

    # initialize the planner
    lateral_planner = FrenetTrajectoryPlanner(p0 = p, s0=s)

    frenet_paths = []

    for i,t in enumerate(Tn):
        lateral_planner.replan_cd_cv(t)
        
        print('d0: ', lateral_planner.p0, lateral_planner.t_initial)
        print('s0: ', lateral_planner.s0, lateral_planner.t_initial)
        frenet_paths.append(lateral_planner.paths)

    return frenet_paths

def frenet_ct(p:(float, float, float), s:(float, float, float), Tn:float, s_target: Frenet) -> Frenet:
    lateral_planner = FrenetTrajectoryPlanner(p0 = p, s0=s, s_target = s_target)

    frenet_paths = []

    for i,t in enumerate(Tn):
        lateral_planner.replan_ct(t)
        
        print('d0: ', lateral_planner.p0, lateral_planner.t_initial)
        print('s0: ', lateral_planner.s0, lateral_planner.t_initial)
        frenet_paths.append(lateral_planner.paths)
    

    return frenet_paths


def frenet_xy(p:(float, float, float), s:(float, float, float)):
    # cfg.MIN_T = 3.0
    # cfg.MAX_T = 6.0
    # cfg.DT = 0.001
    lateral_planner = FrenetTrajectoryPlanner(p0 = p, s0=s)

    frenet_paths = []
    frenet_paths.append(lateral_planner.paths)
    wx = [-1.0,  25.5, 30.0, 40]
    wy = [-3.0, -5.5, -4.0, -3]
    spline = Spline2D(wx, wy)

    frenet_paths = frenet_coordinates_xy(frenet_paths, spline)
    return frenet_paths, spline

def save_images(frenet_paths: Frenet, path, type_i, s_target = None):
    if type_i == "cv":
        plot_lateral_paths_lst(frenet_paths, path+"lateral_trajectory",three=True,save=True) 
    elif type_i == "cd":
        plot_longitudinal_paths_lst(frenet_paths, path+"longitudial_trajectory",three=True,save=True) 
    elif type_i == "ctot":
        if s_target !=None:
            plot_following_paths_lst_ctot(frenet_paths, s_target, path+"following_trajectory_ctot",three=True,save=True)
        else:
            plot_xy_paths_lst_ctot(frenet_paths, path+"xy_coordinates", save= True)
    else:
        plot_following_paths_lst_ct(frenet_paths, s_target, path+"following_trajectory_ct", three=True, save= True)

        
