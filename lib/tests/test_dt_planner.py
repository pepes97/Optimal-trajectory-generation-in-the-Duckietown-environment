"""test_dt.py
"""

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


class PerspectiveWarper:
    def __init__(self, dest_size=(640, 480),
                 src=np.float32([
                     (0.3796, 0.4438),
                     (0, 0.9396),
                     (1, 0.9396),
                     (0.6281, 0.4438)]),
                 dest=np.float32([(0.3, 0), (0.3, 1), (0.7, 1), (0.7, 0)])):
        self.dest_size = dest_size
        dest_size = np.float32(dest_size)
        self.src = src * dest_size
        self.dest = dest * dest_size
        self.M = cv2.getPerspectiveTransform(self.src, self.dest)
        self.iM = cv2.getPerspectiveTransform(self.dest, self.src)

    def warp(self, frame, draw=False):
        warped_frame = cv2.warpPerspective(frame, self.M, self.dest_size)
        if draw:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(frame)
            axs[0].plot(self.src[:, 0], self.src[:, 1], 'r')
            axs[1].imshow(warped_frame)
            axs[1].plot(self.dest[:, 0], self.dest[:, 1], 'r')
            plt.show()

        return warped_frame

    def iwarp(self, frame):
        return cv2.warpPerspective(frame, self.iM, self.dest_size)

u = np.array([1.0, 0.0])
pos_s = (0.0, 0.0, 0.0)
def test_duckietown_planner(*args, **kwargs):
    env = DuckietownEnv(seed=123,
                        map_name='loop_empty',
                        camera_rand=False)

    # Lines
    perspective_projector = PerspectiveWarper()
    line_filter = CenterLineFilter()
    lat_line_filter = LateralLineFilter()
    lat_line_tracker = SlidingWindowDoubleTracker(robust_factor=1)
    lateral_lane_filter = TrajectoryFilter(perspective_projector, line_filter, lat_line_filter, lat_line_tracker)
    # Controller
    controller = FrenetIOLController(.5, 0.0, 30, 0.0, 0.0)
    # Plot 
    fig, ax = plt.subplots(1, 2)
    env.reset()
    env.render()
    obs, reward, done, info = env.step(np.array([0.0, 0.0]))
    im = ax[0].imshow(obs)
    im2 = ax[1].imshow(obs)
    curve_line, = ax[0].plot([], [], 'r')
    curve_unwarped_line, = ax[1].plot([], [], 'r')
    # Planner 
    planner = TrajectoryPlannerV1DT(TrajectoryPlannerParamsDT())
    line_found, cpose, curvature, _ = lateral_lane_filter.process(obs)
    s0 = (0.0, 0.0, 0.0)
    d0 = (cpose[0], 0.0, 0.0)
    planner.initialize(t0=0, p0=d0, s0=s0)

    def animate(i):
        global u, pos_s
        obs, reward, done, info = env.step(u)
        #line_found, cpose, curv,a,b,c = lateral_lane_filter.process(obs)
        line_found, cpose, curvature, _ = lateral_lane_filter.process(obs)
        # f = lambda x: int(a*x**2 + b*x + c)
        # df = lambda x: int(2*a*x + b)
        if line_found:
            robot_fpose = np.array([0.0, cpose[0], cpose[1]])
            d0 = (cpose[0], 0.0, 0.0)
            planner.initialize(t0=i, p0=d0, s0=pos_s)
            pos_s, pos_d = planner.replanner(time = i+0.3)
            lateral_lane_filter.planner = planner
            # Compute error
            error = np.array([0, pos_d[0]]) - robot_fpose[0:2]
            derror = np.array([pos_s[1], pos_d[1]])
            # print(f'curvature={curvature}')
            u = controller.compute(robot_fpose, error, derror, curvature)
            if np.linalg.norm(u) !=0: 
                u = u / np.linalg.norm(u)
            # u[1] *= -1
            # print(f'fpose={robot_fpose}, u={u}')
    
        im.set_array(lateral_lane_filter.plot_image)
        im2.set_array(obs)
        #scat.set_offsets(target_pos)
        env.render()
        return [im, im2, curve_line, curve_unwarped_line]
        # return [im, im2, curve_line, curve_unwarped_line, scat]
    ani = animation.FuncAnimation(fig, animate, frames=500, interval=50, blit=True)
    plt.show()
    
def compute_ortogonal_vect(df, s):
    t_grad = np.array([df(s[0]), df(s[1])])
    t_r = np.arctan2(t_grad[1], t_grad[0])
    return np.array([-np.sin(t_r), np.cos(t_r)])