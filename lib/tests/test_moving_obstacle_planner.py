import logging
import numpy as np

from .config import SimulationConfiguration
from .config import DefaultSimulationParameters as dsp

from ..logger import SimulationDataStorage, SimData, timeprofiledecorator
from ..trajectory import QuinticTrajectory2D, CircleTrajectory2D, SplineTrajectory2D
from ..transform import FrenetGNTransformOld
from ..controller import FrenetIOLController
from ..plotter import *
from ..sensor import StaticObstacle, ProximitySensor, MovingObstacle
from operator import attrgetter
import datetime
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
         # Check paths that do not encounter obstacles
        planner.paths, leading_obstacles = check_paths(paths_planner, sensor, robot_p)
        if len(planner.paths)==0: # no safe paths, follow nearest obstacle
            s_target = leading_obstacles[0].trajectory
            s_target.set_transformer(transformer)
            pos_s, pos_d = planner.replanner(t_vect[i], s_target = s_target)
            paths_planner = planner.paths
            paths_planner = frenet_to_glob(planner, trajectory, paths_planner)
        planner.opt_path_tot = min(planner.paths, key=attrgetter('ctot'))
        # target pos for plot coordinates
        ts, td = pos_s[0], pos_d[0]
        target_pos = trajectory.compute_pt(ts) + compute_ortogonal_vect(trajectory, ts) * td
        #Compute error
        error = np.array([0, pos_d[0]]) - robot_fpose[0:2]
        derror = np.array([pos_s[1], pos_d[1]])
        # Print check
        logger.info(f'Planner s, d:{ts, td}')
        logger.info(f'Robot s,d :{robot_fpose[:2]}')
        logger.info(f'Error:{error}')
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


def test_planner_moving_obstacle(*args, **kwargs):
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
        fig, ani = plot_2d_planner_moving_obstacles_anim(data_storage, trajectory,robot, sensor, obstacle_lst, t_vect)
        #ani.save("./images/animated/planner_moving_obstacles.gif")
        if store is not None:
            # TODO (generate path inside images/<timeoftheday>/store:str)
            ani.save(store)
            #ani.save(store, writer="ffmpeg")
        plt.show()
    if plot_flag:
        __plot_fn(store_plot)
    return data_storage