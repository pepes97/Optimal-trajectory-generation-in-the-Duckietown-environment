"""test_dt.py
"""

import gym
from gym_duckietown.envs import DuckietownEnv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import cv2
from ..video import *

def test_duckietown(*args, **kwargs):
    env = DuckietownEnv(seed=123,
                        map_name='loop_empty')
    perspective_projector = PerspectiveWarper()
    line_filter = CenterLineFilter()
    line_tracker = SlidingWindowTracker(robust_factor=1)
    fig, ax = plt.subplots(1, 2)
    env.reset()
    env.render()
    obs, reward, done, info = env.step(np.array([0.0, 0.0]))
    im = ax[0].imshow(obs)
    im2 = ax[1].imshow(obs)
    curve_line, = ax[0].plot([], [], 'r')
    curve_unwarped_line, = ax[1].plot([], [], 'r')
    def animate(i):
        obs, reward, done, info = env.step(np.array([0.3, .4]))
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
            
        im.set_array(out_image)
        im2.set_array(obs)
        env.render()
        return [im, im2,  curve_line, curve_unwarped_line]
    ani = animation.FuncAnimation(fig, animate, frames=500, interval=30, blit=True)
    plt.show()
        
    #for i in range(500):
    #    obs, reward, done, info = env.step(np.array([0.1, 0.0]))
    #    env.render()
    #plt.show()


