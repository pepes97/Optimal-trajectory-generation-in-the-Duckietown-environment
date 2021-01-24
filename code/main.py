
from motion import LateralTrajectoryPlanner
from plot import plot_longitudinal_paths_lst

# initial state
p=(3,-2,0) 

# replanning instants
Tn = [0,1.5,3]

# initialize the planner
lateral_planner = LateralTrajectoryPlanner(p, t_initial=0, kj=2, kt=0.5, kd=1,
                                            di_interval=(-2, 2.1, 0.5),
                                            t_interval=(1, 4, 1))

frenet_paths = []

for i,t in enumerate(Tn):
        
    lateral_planner.replan(t)
    
    print(lateral_planner.p0, lateral_planner.t_initial)
    
    frenet_paths.append(lateral_planner.paths)
    
plot_longitudinal_paths_lst(frenet_paths)