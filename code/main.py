
from motion import FrenetTrajectoryPlanner
from plot import plot_longitudinal_paths_lst, plot_lateral_paths_lst, plot_lateral_paths_lst_ctot, plot_longitudinal_paths_lst_ctot, plot_following_paths_lst_ctot, plot_following_paths_lst_ct
from quintic_polynomial import QuinticPolynomial
from utils import frenet_follow_target

# initial state
p = (3,0.3,0) 
s = (0,2,0)
s0_t = (1,1,0)
s1_t = (6,0,0)
Ts = 10
# initialize the planner
s_target = frenet_follow_target(s0=s0_t, s1=s1_t, Ts=Ts)
lateral_planner = FrenetTrajectoryPlanner(p0 = p, s0=s, s_target = s_target)

# replanning instants
Tn = [0, 2.5, 5]

frenet_paths = []

for i,t in enumerate(Tn):
    
    lateral_planner.replan_ctot(t)
    
    print('d0: ', lateral_planner.p0, lateral_planner.t_initial)
    print('s0: ', lateral_planner.s0, lateral_planner.t_initial)
    frenet_paths.append(lateral_planner.paths)

# plot_lateral_paths_lst(frenet_paths)
# plot_longitudinal_paths_lst(frenet_paths)
plot_lateral_paths_lst_ctot(frenet_paths)
plot_longitudinal_paths_lst_ctot(frenet_paths)
plot_following_paths_lst_ctot(frenet_paths, s_target)