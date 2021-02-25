import logging
import numpy as np

from .config import SimulationConfiguration
from .config import DefaultSimulationParameters as dsp

from ..logger import SimulationDataStorage, SimData, timeprofiledecorator
from ..trajectory import QuinticTrajectory2D, CircleTrajectory2D, SplineTrajectory2D
from ..transform import FrenetGNTransform
from ..controller import FrenetIOLController
from ..plotter import *
from ..sensor import StaticObstacle, ProximitySensor, MovingObstacle
from operator import attrgetter

robot_radius = 0.7

def frenet_to_glob(planner, trajectory, frenet_paths):
    for path in frenet_paths:
        for i in range(len(path.s)):
            target_pos = trajectory.compute_pt(path.s[i]) + compute_ortogonal_vect(trajectory, path.s[i]) * path.d[i]
            path.x.append(target_pos[0])
            path.y.append(target_pos[1])

    return frenet_paths

def check_paths(frenet_paths, sensor, rpose):
    measure_obst = sensor.sense(rpose)
    leading_obstacles = []
    if measure_obst:
        new_paths_idx = []
        for i in range(len(frenet_paths)):
            collision, obstacle = check_collisions(frenet_paths[i], measure_obst)                
            if not collision:
                leading_obstacles.append(obstacle) # obstacle is never None if collision is False
                continue
            new_paths_idx.append(i) 
        return [frenet_paths[i] for i in new_paths_idx], leading_obstacles
    else:
        return frenet_paths, leading_obstacles


def check_collisions(path,obstacles):

    for ob in obstacles:
        obx = ob.position[0]
        oby = ob.position[1]
            
        distance = [((ix - obx) ** 2 + (iy - oby) ** 2) for ix, iy in zip(path.x, path.y)]
        collision = any([di <= robot_radius**2 for di in distance])
        if collision:
            return False, ob
    return True, None        

def compute_ortogonal_vect(traj, s):
        t_grad = traj.compute_first_derivative(s)
        t_r = np.arctan2(t_grad[1], t_grad[0])
        return np.array([-np.sin(t_r), np.cos(t_r)])
    
@timeprofiledecorator
def __simulate_experiment(sim_config, data_storage, trajectory, robot, transformer, controller, planner, sensor):
    
    # Simulation loop
    t_vect = sim_config.get_time_vect()
    robot_p = robot.p

    robot_dp = np.zeros(3)
    robot_ddp = np.zeros(3)
    u = np.zeros(2)
    # Initialization
    est_pt = transformer.estimatePosition(trajectory, robot_p)
    # Robot pose in frenet
    robot_fpose = transformer.transform(robot_p)
    s0 = (robot_fpose[0], 0.0, 0.0)
    d0 = (robot_fpose[1], 0.0, 0.0)
    planner.initialize(t0=t_vect[0], p0=d0, s0=s0)
    for i in range(sim_config.get_simulation_length()):
        # Estimate frenet frame
        est_pt = transformer.estimatePosition(trajectory, robot_p)
        # Robot pose in frenet
        robot_fpose = transformer.transform(robot_p)
        # Robot velocity in frenet (need only p_dot and d_dot)
        robot_fdp  = transformer.transform(robot_dp)[0:2]
        # Robot acceleration in frenet 
        robot_fddp  = transformer.transform(robot_ddp)[0:2]
        # Get replanner step
        pos_s, pos_d = planner.replanner(t_vect[i])
        paths_planner = planner.paths
        paths_planner = frenet_to_glob(planner, trajectory, paths_planner)
        sensor.step_obstacle(t_vect[i])
        planner.paths, leading_obstacles = check_paths(paths_planner, sensor, robot_p)
        if len(planner.paths)==0: # no safe paths, follow nearest obstacle
            s_target = leading_obstacles[0].trajectory
            s_target.set_transformer(transformer)
            pos_s, pos_d = planner.replanner(t_vect[i], s_target = s_target)
            paths_planner = planner.paths
            paths_planner = frenet_to_glob(planner, trajectory, paths_planner)
        planner.opt_path_tot = min(planner.paths, key=attrgetter('ctot'))
        ts, td = pos_s[0], pos_d[0]
        # print(ts, td)
        #check_collisions(planner,t_vect[i],paths_planner)
        target_pos = trajectory.compute_pt(ts) + compute_ortogonal_vect(trajectory, ts) * td
        target_fpos = transformer.transform(target_pos)
        target_dpos = trajectory.compute_first_derivative(ts)
        target_fdpos = transformer.transform(target_dpos)
        #Compute error
        error = target_fpos - robot_fpose[0:2]
        derror = target_fdpos - robot_fdp
        # Get curvature
        curvature = trajectory.compute_curvature(est_pt)
        # Compute control
        u = controller.compute(robot_fpose, error, derror, curvature)
        # Step the unicycle
        robot_p, robot_dp = robot.step(u, dsp.dt)
        # log data
        data_storage.set(SimData.robot_pose, robot_p, i)
        data_storage.set(SimData.robot_frenet_pose, robot_fpose, i)
        data_storage.set(SimData.control, u, i)
        data_storage.set(SimData.trajectory_2d, target_pos, i)
        data_storage.set(SimData.error, error, i)
        data_storage.set(SimData.derror, derror, i)
        data_storage.set(SimData.planner, target_pos, i)
        
        
    return data_storage


def test_planner_obstacle(*args, **kwargs):
    plot_flag = False
    store_plot = None
    if 'plot' in kwargs:
        plot_flag = kwargs['plot']
    if 'store_plot' in kwargs:
        store_plot = kwargs['store_plot']

    sim_config = SimulationConfiguration(**kwargs)
    # Extract key objects from configuration object
    t_vect, robot, trajectory, transformer, controller, planner = sim_config.get_elements()
    # Configure SimulationDataStorage
    data_storage = SimulationDataStorage(t_vect)
    data_storage.add_argument(SimData.robot_pose)
    data_storage.add_argument(SimData.robot_frenet_pose)
    data_storage.add_argument(SimData.control)
    data_storage.add_argument(SimData.trajectory_2d)
    data_storage.add_argument(SimData.target_pos)
    data_storage.add_argument(SimData.error)
    data_storage.add_argument(SimData.derror)
    data_storage.add_argument(SimData.planner)

     # generate 20 obstacles randomly on the path
    np.random.seed(5)
    # Sample random time instants to get obstacles trajectory projections
    t_samples = np.random.choice(t_vect, 10)
    logger.debug(f't_samples: {t_samples}')
    obs_x_proj = []
    obs_y_proj = []
    for t in t_samples:
        pt = trajectory.compute_pt(t)
        obs_x_proj.append(pt[0])
        obs_y_proj.append(pt[1])
    obstacles_pos = np.array([obs_x_proj, obs_y_proj])
    # Generate noise on y axis to offset obtsacles from trajectory
    noise = np.random.normal(0, 2., obstacles_pos.shape[1])
    obstacles_pos[1, :] += noise
    # Generate static obstacles
    obstacle_lst = [MovingObstacle(obstacles_pos[:, i], trajectory) for i in range(obstacles_pos.shape[1])]
    for i in range(len(obstacle_lst)):
        logger.info(f'obstacle_{i}: {obstacle_lst[i].position}')

    sensor = ProximitySensor(3.5, np.pi/6)
    sensor.attach(robot)
    sensor.set_obstacle(obstacle_lst) 
      

    data_storage = __simulate_experiment(sim_config, data_storage, trajectory,
                                        robot, transformer, controller, planner, sensor)

    def __plot_fn(store: str=None):
        fig, ax = generate_1_1_layout()
        # Plot trajectory
        ax.axis('equal')
        trajectory_line = plot_trajectory(ax, trajectory, t_vect)
        unicycle_poly   = plot_unicycle(ax, robot)
        # Extract data from data_storage
        rpose = data_storage.get(SimData.robot_pose)
        planner_path = data_storage.get(SimData.planner)
        planner_line, = ax.plot(planner_path[0, :], planner_path[1, :], 'g')
        sensor_line   = plot_proximity_sensor(ax, sensor, robot)
        obstacles_poly = [plot_moving_obstacle(ax, obs) for obs in obstacle_lst]
        
        def get_obstacle_coordinates(obst_lst: [Obstacle]):
            x_lst = []
            y_lst = []
            for o in obst_lst:
                x_lst.append(o.position[0])
                y_lst.append(o.position[1])
            return np.array([x_lst, y_lst])


        measure_obst = sensor.sense(rpose=rpose[:, 0])
        measure_pts = get_obstacle_coordinates(measure_obst)
        measure_scat = ax.scatter(measure_pts[0, :], measure_pts[1, :], s=400, facecolors='none', edgecolors='r')
        # moving_obsacles = ax.scatter(measure_pts[0, :], measure_pts[1, :], c='k', marker='x')
        # Animation callback
        def animate(i):
            # Center camera to robot
            ax.set_xlim(xmin=rpose[0, i] - 5, xmax=rpose[0, i] + 5)
            ax.set_ylim(ymin=rpose[1, i] - 5, ymax=rpose[1, i] + 5)
            ax.figure.canvas.draw()

            # Sensor
            p1, p2 = compute_sensor_vertices(sensor, rpose[:, i])
            sensor_line.set_xdata([p1[0], rpose[0, i], p2[0]])
            sensor_line.set_ydata([p1[1], rpose[1, i], p2[1]])
            # Get new robot vertices
            unicycle_poly.set_xy(compute_unicycle_vertices(rpose[:, i]))
            # Plot measure 
            sensor.step_obstacle(t_vect[i])
            measure_obst = sensor.sense(rpose=rpose[:, i])
            logger.debug(f'time_{i}: sensor found {len(measure_obst)} obstacles')
            measure_pts = get_obstacle_coordinates(measure_obst)

            # moving_obsacles.set_offsets(np.c_[measure_pts[0, :], measure_pts[1, :]])
            for poly,obs in zip(obstacles_poly,obstacle_lst):
                poly.set_xy(compute_unicycle_vertices(obs.pose))
            measure_scat.set_offsets(np.c_[measure_pts[0, :], measure_pts[1, :]])
            # Plot next 30 (at most) planner points
            planner_line.set_xdata(planner_path[0, i:i+30])
            planner_line.set_ydata(planner_path[1, i:i+30])
            return [unicycle_poly, planner_line,sensor_line, measure_scat, obstacles_poly]
        ani = animation.FuncAnimation(fig, animate, frames=t_vect.shape[0], interval=30, blit=False)
        if store is not None:
            # TODO (generate path inside images/<timeoftheday>/store:str)
            ani.save(store)
        plt.show()
    if plot_flag:
        __plot_fn(store_plot)
    return data_storage