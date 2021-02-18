"""test_plot.py
"""

import logging
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from .config import SimulationConfiguration
from .config import DefaultSimulationParameters as dsp
from ..logger import SimulationDataStorage, SimData
from ..trajectory import QuinticTrajectory2D, CircleTrajectory2D, SplineTrajectory2D
from ..transform import FrenetGNTransform
from ..controller import FrenetIOLController
from ..plotter import *
from ..sensor import ProximitySensor

logger = logging.getLogger(__name__)

def test_plot_unicycle(*args, **kwargs):
    sim_config = SimulationConfiguration(**kwargs)
    # Extract key objects from configuration object
    t_vect, robot, trajectory, transformer, controller, planner = sim_config.get_elements()
    data_storage = SimulationDataStorage(t_vect)
    data_storage.add_argument(SimData.robot_pose)

    sensor = ProximitySensor(10, np.pi/8)
    sensor.attach(robot)

    data_storage = __simulate_experiment(sim_config, data_storage, trajectory, robot, transformer, controller, planner)

    def __plot_fn():
        fig, ax = generate_1_1_layout()
        ax.axis('equal')
        ax.use_sticky_edges = False
        trajectory_line = plot_trajectory(ax, trajectory, t_vect)
        unicycle_poly = plot_unicycle(ax, robot)
        sensor_line   = plot_proximity_sensor(ax, sensor, robot)

        # Extract data from data_storage
        rpose = data_storage.get(SimData.robot_pose)
        
        def animate(i):
            # Center camera to robot
            ax.set_xlim(xmin=rpose[0, i] - 10, xmax=rpose[0, i] + 10)
            ax.set_ylim(ymin=rpose[1, i] - 10, ymax=rpose[1, i] + 10)
            ax.figure.canvas.draw()
            
            unicycle_poly.set_xy(compute_unicycle_vertices(rpose[:, i]))
            # Get new sensor's points 
            p1, p2 = compute_sensor_vertices(sensor, rpose[:, i])
            sensor_line.set_xdata([p1[0], rpose[0, i], p2[0]])
            sensor_line.set_ydata([p1[1], rpose[1, i], p2[1]])
            return [unicycle_poly, sensor_line]
        ani = animation.FuncAnimation(fig, animate, frames=t_vect.shape[0], interval=30, blit=False)
        plt.show()
        
    __plot_fn()

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

