
from motion import LateralTrajectoryPlanner
from plot import plot_longitudinal_paths_lst, plot_longitudinal_paths, plot_lateral_paths_lst, plot_longitudinal_paths

# initial state
p = (3,0.3,0) 
s = (0,2, -1.25)

# replanning instants
Tn = [0, 2.5, 5]

# target velocity and delta s
sd = 5
si_interval = [(0,5.5,1),(1,7,1), (2,9,1)]
# initialize the planner
lateral_planner = LateralTrajectoryPlanner(p, t_initial=0, kj=0.01, kt=0.4, kd=1.0,
                                            di_interval=(-2.2, 2.30, 0.55),
                                            t_interval=(1.0, 6.2, 0.8), 
                                            s0=s, si_interval= si_interval[0],
                                            sd=sd, kdot_s = 1.0, k_long=1.0, k_lat=1.0)

frenet_paths = []

for i,t in enumerate(Tn):
    
    if i > 0:
        lateral_planner.replan(t, si_interval[i])
    else:
        lateral_planner.replan(t)
    
    print('d0: ', lateral_planner.p0, lateral_planner.t_initial)
    print('s0: ', lateral_planner.s0, lateral_planner.t_initial)
    frenet_paths.append(lateral_planner.paths)
    #plot_longitudinal_paths(frenet_paths[i])

plot_lateral_paths_lst(frenet_paths)
plot_longitudinal_paths_lst(frenet_paths)