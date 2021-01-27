
from motion import LateralTrajectoryPlanner
from plot import plot_longitudinal_paths_lst, plot_longitudinal_paths, plot_lateral_paths_lst, plot_longitudinal_paths

# initial state
p=(3,0.5,0) 
s = (0,2)

# replanning instants
Tn = [0,2.5,5]

# target velocity and delta s
sd = 5



# initialize the planner
lateral_planner = LateralTrajectoryPlanner(p, t_initial=0, kj=0.1, kt=1.5, kd=1.0,
                                            di_interval=(-2.0, 3.5, 1),
                                            t_interval=(1, 5.1, 0.5), 
                                            s0=s, si_interval= (-2,8, 3.5),
                                            sd= sd, kdot_s = 1.5, k_long=1.0, k_lat=1.0)

frenet_paths = []

for i,t in enumerate(Tn):
        
    lateral_planner.replan(t)
    
    print(lateral_planner.p0, lateral_planner.t_initial)
    print(lateral_planner.s0, lateral_planner.t_initial)
    
    frenet_paths.append(lateral_planner.paths)
    #plot_longitudinal_paths(frenet_paths[i])

plot_lateral_paths_lst(frenet_paths)
plot_longitudinal_paths_lst(frenet_paths)