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

u = np.array([0.5, 0.0])

def test_duckietown_planner(*args, **kwargs):
    env = DuckietownEnv(seed=123,
                        map_name='loop_empty',
                        camera_rand=False)

    # Lines
    perspective_projector = PerspectiveWarper()
    line_filter = CenterLineFilter()
    line_tracker = SlidingWindowTracker(robust_factor=1)
    middle_lane_filter = MiddleLineFilter(perspective_projector, line_filter, line_tracker)
    lat_line_filter = LateralLineFilter()
    lat_line_tracker = SlidingWindowDoubleTracker(robust_factor=1)
    lateral_lane_filter = TrajectoryFilter(perspective_projector, line_filter, lat_line_filter, lat_line_tracker)
    # Controller
    controller = FrenetIOLController(.5, 0.0, 30, 0.0, 0.0)
    # Planner 
    planner = TrajectoryPlannerV1DT(TrajectoryPlannerParamsDT())
    s0 = (0.0, 0.0, 0.0)
    d0 = (0.0, 0.0, 0.0)
    planner.initialize(t0=0, p0=d0, s0=s0)
    # Plot 
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
        line_found, cpose, curv,a,b,c = lateral_lane_filter.process(obs)
        f = lambda x: int(a*x**2 + b*x + c)
        df = lambda x: int(2*a*x + b)
        if line_found:
            robot_fpose = np.array([0.0, cpose[0], cpose[1]])
            pos_s, pos_d = planner.replanner(i)
            ts, td = pos_s[0], pos_d[0]
            # target_pos = ts + compute_ortogonal_vect(df, np.array([f(ts), ts])) * td
            # print(f'target_pos{target_pos}')
            #Compute error
            error = np.array([0, pos_d[0]]) - robot_fpose[0:2]
            derror = np.array([pos_s[1], pos_d[1]])
            curvature = curv
            print(f'curvature{curvature}')
            u = controller.compute(robot_fpose, error, derror, curvature)
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
    ani = animation.FuncAnimation(fig, animate, frames=500, interval=50, blit=True)
    plt.show()
    
def compute_ortogonal_vect(df, s):
    t_grad = np.array([df(s[0]), df(s[1])])
    t_r = np.arctan2(t_grad[1], t_grad[0])
    return np.array([-np.sin(t_r), np.cos(t_r)])