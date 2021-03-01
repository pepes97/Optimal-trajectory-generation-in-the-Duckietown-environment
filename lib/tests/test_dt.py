"""test_dt.py
"""

import gym
from gym_duckietown.envs import DuckietownEnv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from ..video import *

def test_duckietown(*args, **kwargs):
    env = DuckietownEnv(seed=123,
                        map_name='loop_empty')
    lane_filter = CenterLineFilter()
    perspective_projector = PerspectiveWarper()
    line_filter = CenterLineFilter()
    line_tracker = SlidingWindowTracker()
    fig, ax = plt.subplots(1, 1)
    env.reset()
    env.render()
    obs, reward, done, info = env.step(np.array([0.0, 0.0]))
    im = ax.imshow(obs)
    curve_line, = ax.plot([], [], 'y')
    def animate(i):
        obs, reward, done, info = env.step(np.array([0.2, -.2]))
        warped_image = perspective_projector.warp(obs)#lane_filter.process(obs)
        line_warped_image = line_filter.process(warped_image)
        line_warped_fit = line_tracker.search(line_warped_image)
        if (line_warped_fit != np.zeros(3)).all():
            # Line is found
            # plot line
            ploty = np.linspace(0, warped_image[0]-1, 10)
            line_fit = line_warped_fit[0] * ploty**2 + line_warped_fit[1] * ploty + line_warped_fit[2]
            curve_line.set_xdata(line_fit)
            curve_line.set_ydata(ploty)
            
        im.set_array(line_warped_image)
        env.render()
        return [im, curve_line]
    ani = animation.FuncAnimation(fig, animate, frames=500, interval=30, blit=True)
    plt.show()
        
    #for i in range(500):
    #    obs, reward, done, info = env.step(np.array([0.1, 0.0]))
    #    env.render()
    #plt.show()


