from scipy.optimize import curve_fit
import gym
from gym_duckietown.envs import DuckietownEnv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import cv2
from pyglet.window import key
import pyglet
import sys
from numpy import arange
from ..video import *
from ..controller import FrenetIOLController
from ..planner import *
from ..transform import FrenetDKTransform
from ..platform import Unicycle
from ..mapper import *


def frenet_to_glob(trajectory, planner, projection):
    frenet_path = planner.opt_path_tot
    path = []
    for i in range(len(frenet_path.s)):
        s = projection + (frenet_path.s[i] - planner.s0[0])
        d = frenet_path.d[i] 
        target_pos = trajectory.compute_pt(s) + \
            compute_ortogonal_vect(trajectory, s) * d
        path.append(target_pos)
    path = np.array(path)
    return path

def compute_ortogonal_vect(trajectory, s):
    ds = 1/30
    s1 = s + ds
    t_grad = trajectory.compute_pt(s1) - trajectory.compute_pt(s)
    t_r = np.arctan2(t_grad[1], t_grad[0])
    return np.array([-np.sin(t_r), np.cos(t_r)])

def test_mapper_planner(*args, **kwargs):
    env = DuckietownEnv(seed=123,
                        map_name='loop_empty',
                        camera_rand=False)
    
    # Planner 
    planner = TrajectoryPlannerV1DT(TrajectoryPlannerParamsDT())
    # transformer 
    transformer = FrenetDKTransform()
    # Controller
    controller = FrenetIOLController(.5, 0.0, 27, 0.0, 0.0)
    # Mapper
    mapper = Mapper()
    # Env initialize
    env.reset()
    env.render()
    obs, reward, done, info = env.step(np.array([0.0, 0.0]))
    # Plot 
    fig, ax = plt.subplots(1, 2)
    im = ax[0].imshow(obs)
    im2 = ax[1].imshow(obs)
    curve_line, = ax[0].plot([], [], 'r')
    curve_unwarped_line, = ax[1].plot([], [], 'r')
    # Global variables 
    global u, robot_p, robot_dp, robot_ddp, pos_s, pos_d
    # Unicycle
    robot = Unicycle()
    robot.set_initial_pose(robot.p)
    robot_p = np.zeros(3)
    robot_dp = np.zeros(3)
    robot_ddp = np.zeros(3)
    u = np.zeros(2)
    # Initialization
    line_found, trajectory, observations = mapper.process(obs)
    est_pt = transformer.estimatePosition(trajectory, robot_p)
    # Robot pose in frenet
    robot_fpose = transformer.transform(robot_p)
    # Pose initizialize
    pos_s = s0 = (robot_fpose[0], 0.0, 0.0)
    pos_d = d0 = (robot_fpose[1], 0.0, 0.0)
    # Planner initialize
    planner.initialize(t0=0, p0=d0, s0=s0)
    dt = 1/60

    def animate(i):
        global u, robot_p, robot_dp, robot_ddp, pos_s, pos_d
        
        obs, reward, done, info = env.step(u)
        actual_u = np.array(info['Simulator']['action'])
        robot_p, robot_dp = robot.step(actual_u, dt)
        line_found, trajectory, observations = mapper.process(obs)

        if line_found:
            # Estimate frenet frame
            robot_p = np.array([0.1,0.0,0.0])
            est_pt = transformer.estimatePosition(trajectory,  robot_p)
            # Robot pose in frenet
            robot_fpose = transformer.transform(robot_p)
            # Get replanner step
            pos_s, pos_d = planner.replanner(time = i*dt)
            mapper.lat_filter.proj_planner = trajectory.compute_pt(est_pt)
            mapper.lat_filter.path_planner = frenet_to_glob(trajectory, planner, est_pt)
            #Compute error
            error = np.array([0, pos_d[0]]) - robot_fpose[0:2]
            derror = np.array([pos_s[1], pos_d[1]])
            # Get curvature
            curvature = trajectory.compute_curvature(est_pt)
            # Compute control
            u = controller.compute(robot_fpose, error, derror, curvature)       

        im.set_array(mapper.lat_filter.plot_image)
        im2.set_array(obs)
        env.render()
        return [im, im2, curve_line, curve_unwarped_line]
    ani = animation.FuncAnimation(fig, animate, frames=500, interval=50, blit=True)
    #ani.save("./prova.mp4", writer="ffmpeg")
    plt.show()
