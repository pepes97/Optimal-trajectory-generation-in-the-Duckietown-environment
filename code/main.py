
from motion import LateralTrajectoryPlanner
from plot import plot_longitudinal_paths_lst, plot_longitudinal_paths, plot_lateral_paths_lst, plot_longitudinal_paths

# initial state
p = (3,0.5,0) 
s = (0,2, -0.1)

# replanning instants
Tn = [0, 2.5, 5]

# target velocity and delta s
sd = 5
delta_s = 3
num_samples = 3
# initialize the planner
lateral_planner = LateralTrajectoryPlanner(p, t_initial=0, kj=0.1, kt=1.0, kd=1.0,
                                            di_interval=(-2.0, 3.5, 1),
                                            t_interval=(1, 5.1, 0.5), 
                                            s0=s, si_interval= (sd-delta_s*num_samples, sd+delta_s*num_samples, delta_s),
                                            sd=sd, kdot_s = 2.0, k_long=1.0, k_lat=1.0)

frenet_paths = []

for i,t in enumerate(Tn):
        
    lateral_planner.replan(t)
    
    print('d0: ', lateral_planner.p0, lateral_planner.t_initial)
    print('s0: ', lateral_planner.s0, lateral_planner.t_initial)
    
    frenet_paths.append(lateral_planner.paths)
    #plot_longitudinal_paths(frenet_paths[i])

plot_lateral_paths_lst(frenet_paths)
plot_longitudinal_paths_lst(frenet_paths)