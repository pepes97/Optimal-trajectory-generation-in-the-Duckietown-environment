"""test_dt.py
"""

import gym
from gym_duckietown.envs import DuckietownEnv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import cv2
from pyglet.window import key
import pyglet
import sys
from ..video import *
from ..controller import FrenetIOLController

u = np.array([0.5, 0.0])

def test_duckietown(*args, **kwargs):
    env = DuckietownEnv(seed=123,
                        map_name='loop_empty',
                        camera_rand=False)
    perspective_projector = PerspectiveWarper()
    line_filter = CenterLineFilter()
    line_tracker = SlidingWindowTracker(robust_factor=1)
    middle_lane_filter = MiddleLineFilter(perspective_projector, line_filter, line_tracker)
    lat_line_filter = LateralLineFilter()
    lat_line_tracker = SlidingWindowDoubleTracker(robust_factor=1)
    lateral_lane_filter = TrajectoryFilter(perspective_projector, line_filter, lat_line_filter, lat_line_tracker)
    controller = FrenetIOLController(.5, 0.0, 30, 0.0, 0.0)
    fig, ax = plt.subplots(1, 2)
    env.reset()
    env.render()
    obs, reward, done, info = env.step(np.array([0.0, 0.0]))
    im = ax[0].imshow(obs)
    im2 = ax[1].imshow(obs)
    curve_line, = ax[0].plot([], [], 'r')
    curve_unwarped_line, = ax[1].plot([], [], 'r')
    
    def animate(i):
        global u
        obs, reward, done, info = env.step(u)

        line_found, cpose, curv = lateral_lane_filter.process(obs)

        if line_found:
            robot_fpose = np.array([0.0, cpose[0], cpose[1]])
            error = np.array([0, 0.0]) - robot_fpose[:2]
            t_fvel = np.array([1, 0.0])
            curvature = curv
            u = controller.compute(robot_fpose, error, t_fvel, curvature)
            u = u / np.linalg.norm(u)
            u[1] *= -1
            print(f'fpose={robot_fpose}, u={u}')
            
        """
        warped_frame = perspective_projector.warp(obs)
        thresholded_frame = line_filter.process(warped_frame)
        out_image, line_fit = line_tracker.search(thresholded_frame, draw_windows=True)
        if (line_fit != np.zeros(3)).all():
            # Line is found
            # plot line
            ploty = np.arange(0, warped_frame.shape[0]-1, 1)
            line_fit = line_fit[0] * ploty**2 + line_fit[1] * ploty + line_fit[2]
            curve_line.set_xdata(line_fit)
            curve_line.set_ydata(ploty)
            # Unwarp points of the line
            line_unwarped = np.expand_dims(np.vstack((line_fit, ploty)), axis=1)
            line_unwarped = np.transpose(line_unwarped, (2, 1, 0))
            line_unwarped = cv2.perspectiveTransform(line_unwarped, perspective_projector.iM)
            curve_unwarped_line.set_xdata(line_unwarped[:, 0, 0])
            curve_unwarped_line.set_ydata(line_unwarped[:, 0, 1])
        else:
            curve_line.set_xdata([])
            curve_line.set_ydata([])
            curve_unwarped_line.set_xdata([])
            curve_unwarped_line.set_ydata([])
        """
        
        im.set_array(lateral_lane_filter.plot_image)
        im2.set_array(obs)
        env.render()
        return [im, im2, curve_line, curve_unwarped_line]
    ani = animation.FuncAnimation(fig, animate, frames=500, interval=20, blit=True)
    plt.show()
        
    #for i in range(500):
    #    obs, reward, done, info = env.step(np.array([0.1, 0.0]))
    #    env.render()
    #plt.show()
