from motion import FrenetTrajectoryPlanner
from plot import plot_longitudinal_paths_lst, plot_lateral_paths_lst, plot_lateral_paths_lst_ctot, plot_longitudinal_paths_lst_ctot, plot_following_paths_lst_ctot,plot_4_paths_lst,plot_xy_paths_lst_ctot
from quintic_polynomial import QuinticPolynomial
from utils import frenet_follow_target, frenet_target_ctot, frenet_cd_cv, frenet_ct, save_images,frenet_xy
import numpy as np


def main():
    # initial state
    p = (3,0.3,0) 
    s = (0,2,-0.5)
    s0_t = (1,2,0)
    s1_t = (8,0,0)
    Ts = 10

    # replanning instants
    Tn = [0, 2.5, 5]
    T = np.arange(0,5,0.5)

    s_target = frenet_follow_target(s0=s0_t, s1=s1_t, Ts=Ts)

    # lateral e long
    frenet_paths_cd_cv = frenet_cd_cv(p=p, s=s, Tn=Tn)

    # ctot with s_target
    frenet_paths_ctot_s = frenet_target_ctot(p=p, s=s, Tn=Tn, s_target=s_target)

    # ctot xy replannig
    frenet_paths_ctot = frenet_target_ctot(p=p, s=s, Tn=Tn)
    
    # ct with s_target
    frenet_paths_ct = frenet_ct(p=p, s=s, Tn=Tn, s_target=s_target)

    #frenet_xy_ctot = frenet_xy(p=p, s=s)
    
    plot_4_paths_lst(frenet_paths_cd_cv,frenet_paths_cd_cv, frenet_paths_ct, frenet_paths_ctot_s, target=s_target)
    #plot_xy_paths_lst_ctot(frenet_xy_ctot)

if __name__ == '__main__':
    main()