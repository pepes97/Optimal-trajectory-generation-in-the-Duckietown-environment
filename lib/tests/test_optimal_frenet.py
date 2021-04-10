import logging
import datetime
import numpy as np

from ..optimal_frenet_frame import OptimalParameters as op
from ..optimal_frenet_frame import Target
from ..planner import TrajectoryPlannerOptimal, TrajectoryPlannerParamsOptimal
from ..trajectory import SplineTrajectory2D
from ..planner import Frenet
from ..logger import timeprofiledecorator
from ..plotter import plot_4_paths_lst
import math
from matplotlib import pyplot as plt



def frenet_coordinates_xy(frenet_paths: Frenet, spline) -> Frenet:
    max_interval_s = 0
    for list_frenet in frenet_paths:
        for f in list_frenet:
            for i in range(len(f.s)):
               if f.s[-1]-f.s[0] > max_interval_s:
                   max_interval_s = f.s[-1]-f.s[0]
    for list_frenet in frenet_paths:
        for f in list_frenet:
            for i in range(len(f.s)):
                s = f.s[i]*2.7#/(max_interval_s+1e-6)*(spline.s[-1]-spline.s[0]) # 1e-6 to avoid take exactly 1.0, spline is not defined for 1
                ix, iy = spline.compute_pt(s)
                if ix ==None:
                    break
                i_yaw = spline.calc_yaw(s)
                fx = ix + f.d[i] * math.cos(i_yaw + math.pi / 2.0)
                fy = iy + f.d[i] * math.sin(i_yaw + math.pi / 2.0)

                f.x.append(fx)
                f.y.append(fy)
    return frenet_paths

@timeprofiledecorator
def test_optimal_frenet(*args, **kwargs):
    plot_flag = False
    store_plot = None
    stop = False
    merge = False
    follow = False
    if 'plot' in kwargs:
        plot_flag = kwargs['plot']
    if 'store_plot' in kwargs:
        store_plot = kwargs['store_plot']
    if 'stop' in kwargs:
        stop = kwargs['stop']
    if 'merge' in kwargs:
        merge = kwargs['merge']
    if 'follow' in kwargs:
        follow = kwargs['follow']

    if stop:
        s_target = Target(s0=op.s0_t, s1=op.s1_t, Ts=op.Ts).get_frenet_stop_target()
    
    elif merge:
        s_target = Target(s0=op.s0_t, s1=op.s1_t, Ts=op.Ts).get_frenet_merge_target(Target(s0=op.s0_tb, s1=op.s1_tb, Ts=op.Ts))

    elif follow:
        s_target = Target(s0=op.s0_t, s1=op.s1_t, Ts=op.Ts).get_frenet_follow_target(D0 = 2)
    
    else:
        s_target = None
    
    # Lateral and longitudinal paths
    frenet_paths_cd_cv = []
    planner = TrajectoryPlannerOptimal(TrajectoryPlannerParamsOptimal())
    planner.initialize(t0 = op.t0, p0=op.p, s0=op.s)
    
    for i,t in enumerate(op.Tn):
        planner.replan_cd_cv(t)
        frenet_paths_cd_cv.append(planner.paths)
    
    # Ctot = lat + long
    planner.initialize(t0 = op.t0, p0=op.p, s0=op.s, s_target=s_target)

    frenet_paths_ctot = []
    for i,t in enumerate(op.Tn):
        planner.replan_ctot(t)
        frenet_paths_ctot.append(planner.paths)
    
    planner.initialize(t0 = op.t0, p0 = op.p, s0=op.s, s_target = s_target)
    frenet_paths_ct = []

    for i,t in enumerate(op.Tn):
        planner.replan_ct(t)
        frenet_paths_ct.append(planner.paths)
    
    planner.initialize(t0 = op.t0, p0 = op.px, s0=op.sx)
    planner.klat = 0.3
    planner.t_interval = (planner.min_t,planner.max_t,planner.dt/2)
    planner.di_interval = (-1,2.45,0.2)
    planner.replanner(time=0)

    frenet_paths_xy = []
    frenet_paths_xy.append(planner.paths)
    wx = [-3,  25, 35, 42]
    wy = [-2, -9, -6, 2]
    spline = SplineTrajectory2D(wx, wy)

    frenet_xy_ctot = frenet_coordinates_xy(frenet_paths_xy, spline)

    @timeprofiledecorator
    def __plot_fn(store: str=None):
        plot_4_paths_lst(frenet_paths_cd_cv,frenet_paths_cd_cv, frenet_paths_ct, frenet_xy_ctot, spline, target=s_target)
        if store is not None:
            # TODO (generate path inside images/<timeoftheday>/store:str)
            ani.save(store)
            #ani.save(store, writer='ffmpeg')  
        plt.show()
    if plot_flag:
        __plot_fn(store_plot)
    