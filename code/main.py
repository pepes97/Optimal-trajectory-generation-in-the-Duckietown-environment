from motion import FrenetTrajectoryPlanner
from plot import plot_longitudinal_paths_lst, plot_lateral_paths_lst, plot_lateral_paths_lst_ctot, plot_longitudinal_paths_lst_ctot, plot_following_paths_lst_ctot,plot_4_paths_lst,plot_xy_paths_lst_ctot
from quintic_polynomial import QuinticPolynomial
from utils import Target, frenet_target_ctot, frenet_cd_cv, frenet_ct, save_images,frenet_xy
import numpy as np


def main():
    # initial state
    p = (3,0.3,0) 
    s = (0,2,-0.5)
    s0_t = (1,2,0)
    s1_t = (8,0,0)
    Ts = 10
    t0 = 0 # instant in which the planner is initialized
    # replanning instants
    Tn = [0, 2.5, 5]
    T = np.arange(0,5,0.5)

    s_target = Target(s0=s0_t, s1=s1_t, Ts=Ts).get_frenet_stop_target()
    
    # s0_tb = (3,2,0)
    # s1_tb = (10,0,0)
        
    # s_target = Target(s0=s0_t, s1=s1_t, Ts=Ts).get_frenet_merge_target(Target(s0=s0_tb, s1=s1_tb, Ts=Ts))

    # s_target = Target(s0=s0_t, s1=s1_t, Ts=Ts).get_frenet_follow_target(D0 = 2)
    
    # lateral e long
    frenet_paths_cd_cv = frenet_cd_cv(t0 = t0, p=p, s=s, Tn=Tn)

    # ctot with s_target
    frenet_paths_ctot_s = frenet_target_ctot(t0 = t0,p=p, s=s, Tn=Tn, s_target=s_target)

    # ctot xy replannig
    # frenet_paths_ctot = frenet_target_ctot(p=p, s=s, Tn=Tn)
    
    # ct with s_target
    frenet_paths_ct = frenet_ct(t0 = t0,p=p, s=s, Tn=Tn, s_target=s_target)

    frenet_xy_ctot, spline = frenet_xy(t0 = t0, p=p, s=s)

    plot_4_paths_lst(frenet_paths_cd_cv,frenet_paths_cd_cv, frenet_paths_ct, frenet_paths_ctot_s, target=s_target)
    plot_xy_paths_lst_ctot(frenet_xy_ctot, spline)

if __name__ == '__main__':
    main()

# # from spline_planner import Spline2D
# import matplotlib.pyplot as plt
# import numpy as np
# from quintic_polynomial import QuinticPolynomial

# # x = [-3,20,30, 42]
# # y = [-2,-8,-8, 2]

# # sp = Spline2D(x, y)
# # s = np.arange(0, sp.s[-1], 0.1)

# # rx, ry, ryaw, rk = [], [], [], []
# # for i_s in s:
# #     ix, iy = sp.calc_position(i_s)
# #     rx.append(ix)
# #     ry.append(iy)
# #     ryaw.append(sp.calc_yaw(i_s))
# #     rk.append(sp.calc_curvature(i_s))
# # xx = QuinticPolynomial(-3,0,0,42,0,0,6)
# yy = QuinticPolynomial(-2,-0.5,0,2,1,0,45)
# x = np.arange(0,45,0.05)
# y = [yy.compute_pt(i) for i in x]
# x -=3
# flg, ax = plt.subplots(1)
# plt.plot(x, y, "-r", label="input")
# plt.grid(True)
# x_ticks = np.arange(0, 42, 5)
# y_ticks = np.arange(-14, 4, 2)
# ax.set_xticks(x_ticks)
# ax.set_yticks(y_ticks)
# plt.axis("equal")
# plt.xlabel("x[m]")
# plt.ylabel("y[m]")
# plt.legend()
# plt.show()