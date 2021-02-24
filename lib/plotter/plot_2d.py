"""plot_2d.py
"""

from matplotlib import pyplot as plt
import numpy as np
import logging

from .utils import *
from ..logger import SimulationDataStorage, SimData
from ..platform import Unicycle
from ..transform import homogeneous_transform
from ..sensor import ProximitySensor, Obstacle

logger = logging.getLogger(__name__)

def plot_2d_simulation(data: SimulationDataStorage) -> plt.Figure:
    fig, gs = generate_4_2_layout()
    t = data.t
    # Plot on the first row, the target and robot trajectories
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_title('Global view')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    target_traj = data.get(SimData.trajectory_2d)
    robot_traj  = data.get(SimData.robot_pose)
    ax0.plot(target_traj[0, :], target_traj[1, :], label='Path')
    ax0.plot(robot_traj[0, :], robot_traj[1, :], label='Robot path')

    """
    # Plot on the second row, frenet coordinates evolution
    robot_fpose = data.get(SimData.robot_frenet_pose)
    ax1_0 = fig.add_subplot(gs[1, 0])
    ax1_0.set_title('s(t)')
    ax1_0.set_xlabel('t')
    ax1_0.set_ylabel('s')
    ax1_0.plot(t, robot_fpose[0, :])
    ax1_1 = fig.add_subplot(gs[1, 1])
    ax1_1.set_title('d(t)')
    ax1_1.set_xlabel('t')
    ax1_1.set_ylabel('d')
    ax1_1.plot(t, robot_fpose[1, :])
    """
    # Plot on the second row, frenet errors
    error = data.get(SimData.error)
    ax2_0 = fig.add_subplot(gs[1, 0])
    ax2_0.set_title('error_s(t)')
    ax2_0.set_xlabel('t')
    ax2_0.set_ylabel('s')
    ax2_0.plot(t, error[0, :])
    ax2_1 = fig.add_subplot(gs[1, 1])
    ax2_1.set_title('error_d(t)')
    ax2_1.set_xlabel('t')
    ax2_1.set_ylabel('d')
    ax2_1.plot(t, error[1, :])

    """
    # Plot on third row, frenet errors
    error = data.get(SimData.error)
    ax2_0 = fig.add_subplot(gs[2, 0])
    ax2_0.set_title('error_s(t)')
    ax2_0.set_xlabel('t')
    ax2_0.set_ylabel('s')
    ax2_0.plot(t, error[0, :])
    ax2_1 = fig.add_subplot(gs[2, 1])
    ax2_1.set_title('error_d(t)')
    ax2_1.set_xlabel('t')
    ax2_1.set_ylabel('d')
    ax2_1.plot(t, error[1, :])
    """
    # Plot on third row, frenet derrors
    derror = data.get(SimData.derror)
    ax2_0 = fig.add_subplot(gs[2, 0])
    ax2_0.set_title('derror_s(t)')
    ax2_0.set_xlabel('t')
    ax2_0.set_ylabel('s')
    ax2_0.plot(t, derror[0, :])
    ax2_1 = fig.add_subplot(gs[2, 1])
    ax2_1.set_title('derror_d(t)')
    ax2_1.set_xlabel('t')
    ax2_1.set_ylabel('d')
    ax2_1.plot(t, derror[1, :])

    # Plot on fourth row, control terms
    u = data.get(SimData.control)
    ax3_0 = fig.add_subplot(gs[3, 0])
    ax3_0.set_title('v')
    ax3_0.set_xlabel('t')
    ax3_0.set_ylabel('u')
    ax3_0.plot(t, u[0, :])
    ax3_1 = fig.add_subplot(gs[3, 1])
    ax3_1.set_title('w')
    ax3_1.set_xlabel('t')
    ax3_1.set_ylabel('u')
    ax3_1.plot(t, u[1, :])
    return fig
    
    
    
def plot_2d_simulation_xy(data: SimulationDataStorage) -> plt.Figure:
    fig, ax0 = plt.subplots()
    t = data.t
    # Plot on the first row, the target and robot trajectories
    ax0.set_title('Global view')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    target_traj = data.get(SimData.trajectory_2d)
    robot_traj  = data.get(SimData.robot_pose)
    ax0.plot(target_traj[0, :], target_traj[1, :], label='Path')
    ax0.plot(robot_traj[0, :], robot_traj[1, :], label='Robot path')
    return fig

def plot_2d_simulation_bot_xy(data: SimulationDataStorage) -> plt.Figure:
    fig, ax0 = plt.subplots()
    t = data.t
    # Plot on the first row, the target and robot trajectories
    ax0.set_title('Global view')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    target_traj = data.get(SimData.trajectory_2d)
    bot0_traj   = data.get(SimData.bot0_pose)
    bot1_traj   =  data.get(SimData.bot1_pose)
    ax0.plot(target_traj[0, :], target_traj[1, :], label='Path')
    ax0.plot(bot0_traj[0, :], bot0_traj[1, :], label='bot0 path')
    ax0.plot(bot1_traj[0, :], bot1_traj[1, :], label='bot1 path')
    plt.legend()
    return fig

def compute_unicycle_vertices(rpose: np.array):
    """ Computes the vertices of the triangle representing the unicycle in global coordinates
    """
    # Generate local triangle polygon
    #local_poly_vertex = np.array([[-0.3, -0.6], [0., 0.5], [0.3, -0.6]])
    local_poly_vertex = np.array([[-0.3, -0.3], [0.3, 0], [-0.3, 0.3]])
    # Generate homogeneous transformation to rpose
    R, t = homogeneous_transform(rpose)
    def apply_transform(row):
        return np.matmul(R,row) + t
    # Apply transform to get global vertex positions
    global_poly_vertex = np.apply_along_axis(apply_transform, axis=1, arr=local_poly_vertex)
    return global_poly_vertex

def plot_unicycle(ax, robot):
    """ Plot on the axis ax a unicycle
    """
    if isinstance(robot, Unicycle):
        rpose = robot.pose()
    elif isinstance(robot, np.ndarray):
        rpose = robot
    else:
        raise ValueError('Robot is not an Unicycle object, nor a numpy array')
    global_poly_vertex = compute_unicycle_vertices(rpose)
    rpoly = plt.Polygon(global_poly_vertex, facecolor='black', edgecolor='black', lw=2)
    ax.add_patch(rpoly)
    return rpoly

def plot_trajectory(ax, trajectory, t_vect):
    traj_points = np.zeros((2, t_vect.shape[0]))
    for i, t in enumerate(t_vect):
        traj_points[:, i] = trajectory.compute_pt(t)
    line, = ax.plot(traj_points[0, :], traj_points[1, :], label='trajectory')
    return line

def compute_sensor_vertices(sensor: ProximitySensor, robot):
    if isinstance(robot, Unicycle):
        rpose = robot.pose()
    elif isinstance(robot, np.ndarray):
        rpose = robot
    else:
        raise ValueError('Robot is not an Unicycle object, nor a numpy array')
    base_point = np.array([sensor.range, 0.])
    # Duplicate segment and apply local robot's transform
    R, t = homogeneous_transform(np.array([0., 0., sensor.aperture]))
    p1 = np.matmul(R, base_point)
    R, t = homogeneous_transform(np.array([0., 0., -sensor.aperture]))
    p2 = np.matmul(R, base_point)
    # Apply robot's transform to get global vertices positions
    R, t = homogeneous_transform(rpose)
    p1 = np.matmul(R, p1) + t
    p2 = np.matmul(R, p2) + t
    return p1, p2

def plot_proximity_sensor(ax, sensor: ProximitySensor, robot):
    """ Plot on the axis ax the sensor's area
    """
    if isinstance(robot, Unicycle):
        rpose = robot.pose()
    elif isinstance(robot, np.ndarray):
        rpose = robot
    else:
        raise ValueError('Robot is not an Unicycle object, nor a numpy array')
    rposition = rpose[:2]
    p1, p2 = compute_sensor_vertices(sensor, robot)
    line, = plt.plot([p1[0], rposition[0], p2[0]],
                    [p1[1], rposition[1], p2[1]], 'r')
    return line

def plot_obstacles(ax, obstacle_lst: [Obstacle], *args, **kwargs):
    """ Plot on the axis ax the obstacles in obstacle_lst
    """
    for o in obstacle_lst:
        ax.scatter(o.position[0], o.position[1], **kwargs)
    return None
