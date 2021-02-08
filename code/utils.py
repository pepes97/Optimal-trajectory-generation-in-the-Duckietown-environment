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

def frenet_follow_target(s0: (float,float,float), s1:(float,float,float), Ts: float) -> Frenet:
    DTs = cfg.GLOBAL_D_T
    path_target = QuinticPolynomial(s0[0], s0[1], s0[2], s1[0], s1[1], s1[2], Ts)
    target = Frenet()
    target.t = [t for t in np.arange(0, Ts+DTs, DTs)]
    target.s = [path_target.compute_pt(t) for t in target.t]
    target.dot_s = [path_target.compute_first_derivative(t) for t in target.t]
    target.ddot_s = [path_target.compute_second_derivative(t) for t in target.t]
    target.dddot_s = [path_target.compute_third_derivative(t) for t in target.t]
    return target

def frenet_coordinates_xy(frenet_paths: Frenet, spline: Spline2D) -> Frenet:
    for list_frenet in frenet_paths:
        for f in list_frenet:
            for i in range(len(f.s)):
                ix, iy = spline.calc_position(f.s[i])
                if ix ==None:
                    break
                i_yaw = spline.calc_yaw(f.s[i])
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
    wx = [-1.0,  5, 10, 12,15,]
    wy = [-3.0, -4.5,-5.8, -6.2, -6.6]
    spline = Spline2D(wx, wy)

    frenet_paths = frenet_coordinates_xy(frenet_paths, spline)
    return frenet_paths

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

        
