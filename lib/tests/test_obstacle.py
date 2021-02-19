"""test_obstacle.py
"""

import logging
import numpy as np

from .config import SimulationConfiguration
from .config import DefaultSimulationParameters as dsp

from ..logger import SimulationDataStorage, SimData
from ..trajectory import QuinticTrajectory2D, CircleTrajectory2D, SplineTrajectory2D
from ..transform import FrenetGNTransform
from ..controller import FrenetIOLController
from ..plotter import *
from ..sensor import StaticObstacle, ProximitySensor


def test_obstacles(*args, **kwargs):
    plot_flag = False
    store_plot = None
    if 'plot' in kwargs:
        plot_flag = kwargs['plot']
    if 'store_plot' in kwargs:
        store_plot = kwargs['store_plot']
    
    sim_config = SimulationConfiguration(**kwargs)
    # Extract key objects from configuration object
    t_vect, robot, trajectory, transformer, controller, planner = sim_config.get_elements()

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
    obstacle_lst = [StaticObstacle(obstacles_pos[:, i]) for i in range(obstacles_pos.shape[1])]
    for i in range(len(obstacle_lst)):
        logger.info(f'obstacle_{i}: {obstacle_lst[i].position}')

    sensor = ProximitySensor(7, np.pi/6)
    sensor.attach(robot)
    sensor.set_obstacle(obstacle_lst)

    measure_lst = sensor.sense()
    print(f'Seen {len(measure_lst)} obstacles')

    def __plot_fn(store: str=None):
        fig, ax = generate_1_1_layout()
        ax.axis('equal')
        trajectory_line = plot_trajectory(ax, trajectory, t_vect)
        plot_obstacles(ax, obstacle_lst, marker='x', c='k')
        # Plot measured obstacles
        plot_obstacles(ax, measure_lst, facecolors='none', edgecolors='r')
        # Plot robot and sensor
        plot_unicycle(ax, robot)
        plot_proximity_sensor(ax, sensor, robot)
            
        if store is not None:
            plt.savefig(store)
        plt.show()

    if plot_flag:
        __plot_fn(store_plot)
    return

def test_obstacles_moving(*args, **kwargs):
    plot_flag = False
    store_plot = None
    if 'plot' in kwargs:
        plot_flag = kwargs['plot']
    if 'store_plot' in kwargs:
        store_plot = kwargs['store_plot']
    
    sim_config = SimulationConfiguration(**kwargs)
    # Extract key objects from configuration object
    t_vect, robot, trajectory, transformer, controller, planner = sim_config.get_elements()

    data_storage = SimulationDataStorage(t_vect)
    data_storage.add_argument(SimData.robot_pose)

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
    obstacle_lst = [StaticObstacle(obstacles_pos[:, i]) for i in range(obstacles_pos.shape[1])]
    for i in range(len(obstacle_lst)):
        logger.info(f'obstacle_{i}: {obstacle_lst[i].position}')

    sensor = ProximitySensor(7, np.pi/6)
    sensor.attach(robot)
    sensor.set_obstacle(obstacle_lst)
    # Run simulation
    data_storage = __simulate_experiment(sim_config, data_storage, trajectory, robot, transformer, controller, planner)

    def __plot_fn(store: str=None):
        fig, ax = generate_1_1_layout()
        ax.axis('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Moving sensor\nstatic objects')
        trajectory_line = plot_trajectory(ax, trajectory, t_vect)
        unicycle_poly = plot_unicycle(ax, robot)
        sensor_line   = plot_proximity_sensor(ax, sensor, robot)
        
        def get_obstacle_coordinates(obst_lst: [Obstacle]):
            x_lst = []
            y_lst = []
            for o in obst_lst:
                x_lst.append(o.position[0])
                y_lst.append(o.position[1])
            return np.array([x_lst, y_lst])

        # Extract data from data_storage
        rpose = data_storage.get(SimData.robot_pose)
        measure_obst = sensor.sense(rpose=rpose[:, 0])
        measure_pts = get_obstacle_coordinates(measure_obst)
        measure_scat = ax.scatter(measure_pts[0, :], measure_pts[1, :], facecolors='none', edgecolors='r')
        plot_obstacles(ax, obstacle_lst, c='k', marker='x')
        def animate(i):
            # Center camera to robot
            ax.set_xlim(xmin=rpose[0, i] - 10, xmax=rpose[0, i] + 10)
            ax.set_ylim(ymin=rpose[1, i] - 10, ymax=rpose[1, i] + 10)
            ax.figure.canvas.draw()
            
            # Get new sensor's points 
            p1, p2 = compute_sensor_vertices(sensor, rpose[:, i])
            sensor_line.set_xdata([p1[0], rpose[0, i], p2[0]])
            sensor_line.set_ydata([p1[1], rpose[1, i], p2[1]])
            # Get new robot vertices
            unicycle_poly.set_xy(compute_unicycle_vertices(rpose[:, i]))
            # Plot measured obstacles
            measure_obst = sensor.sense(rpose=rpose[:, i])
            logger.debug(f'time_{i}: sensor found {len(measure_obst)} obstacles')
            measure_pts = get_obstacle_coordinates(measure_obst)
            
            measure_scat.set_offsets(np.c_[measure_pts[0, :], measure_pts[1, :]])
            
            return [unicycle_poly, sensor_line, measure_scat]
        ani = animation.FuncAnimation(fig, animate, frames=t_vect.shape[0], interval=30, blit=False)
        if store is not None:
            ani.save(store)
        plt.show()
    if plot_flag:
        __plot_fn(store_plot)
    return data_storage




def __simulate_experiment(sim_config, data_storage, trajectory, robot, transformer, controller, planner) -> SimulationDataStorage:
    # Simulation loop
    t_vect = sim_config.get_time_vect()
    robot_p = robot.p

    robot_dp = np.zeros(3)
    u = np.zeros(2)
    for i in range(sim_config.get_simulation_length()):
         # Estimate frenet frame
        est_pt = transformer.estimatePosition(trajectory, robot_p)
        # Robot pose in frenet
        robot_fpose = transformer.transform(robot_p)
        # Robot velocity in frenet (need only p_dot and d_dot)
        robot_fdp   = transformer.transform(robot_dp)[0:2]

        # Compute error and derror
        target_pos = trajectory.compute_pt(t_vect[i])
        target_fpos = transformer.transform(target_pos)
        target_dpos = trajectory.compute_first_derivative(t_vect[i])
        target_fdpos = transformer.transform(target_dpos)
        error = target_fpos - robot_fpose[0:2]
        derror = target_fdpos - robot_fdp

        # Get path curvature at estimate
        curvature = trajectory.compute_curvature(est_pt)

        # Compute control
        u = controller.compute(robot_fpose, error, derror, curvature)

        # Step the unicycle
        robot_p, robot_dp = robot.step(u, dsp.dt)

        # log data
        data_storage.set('robot_pose', robot_p, i)
    return data_storage
