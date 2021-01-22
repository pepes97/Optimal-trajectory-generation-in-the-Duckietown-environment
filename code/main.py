from motion import lateral_motionLST_and_longitudinal_motionVK, LateralTrajectoryPlanner
from plot import plot_longitudinal_paths, plot_longitudinal_paths_lst
from operator import attrgetter
from quintic_polynomial import QuinticPolynomial

# initial state

p0 = 2.0  # current lateral position [m]
dp0 = 0.0  # current lateral speed [m/s]
ddp0 = 0.0  # current lateral acceleration [m/s]
s0 = 0.0  # current course position
ds0 = 10.0 / 3.6  # current speed [m/s]

#frenet_p = lateral_motionLST_and_longitudinal_motionVK(p0,dp0,ddp0,s0,ds0)
frenet_paths = []


p = (p0, dp0, ddp0)
p = (3.77, 3, 0)
t_initial = 0
delta_t = 0

for i in range(2):
    
    print(p, t_initial)
    lateral_planner = LateralTrajectoryPlanner(p, t_initial=t_initial, kj=2, kt=0.5, kd=1,
                                               di_interval=(-4, 2, 1),
                                               t_interval=(0.1, 5, 0.25))
    frenet_paths.append(lateral_planner.generate_range_polynomials())
    plot_longitudinal_paths(frenet_paths[i])
    p, delta_t = lateral_planner.forward_optimal()
    t_initial += delta_t
    
plot_longitudinal_paths_lst(frenet_paths)
#frenet_p = lateral_planner.generate_range_polynomials()
#plot_longitudinal_paths(frenet_p)
#best_path = min(frenet_p, key=attrgetter('cd'))


