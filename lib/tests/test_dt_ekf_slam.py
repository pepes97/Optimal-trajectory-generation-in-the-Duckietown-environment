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

from ..localization import EKF_SLAM
from ..localization import *



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


u = np.array([0.0, 0.0])
observations = np.array([]).reshape(0,2)
state = np.zeros((3,))

def transition(state: np.ndarray = np.zeros((ROBOT_DIM,)), \
                inputs: np.ndarray = np.zeros((CONTROL_DIM,),dtype=np.float32),\
                dt: float = 1/30):
    # predict the robot motion, the transition model f(x,u)
    # only affects the robot pose not the landmarks
    # odometry euler integration
    next_state = np.zeros((ROBOT_DIM,))
    x, y, th = state[0], state[1], state[2]
    v, w = inputs[0], inputs[1]
    next_state[0] = x + dt * v * np.cos(th)
    next_state[1] = y + dt * v * np.sin(th)
    th = th + dt * w
    next_state[2] = np.arctan2(np.sin(th),np.cos(th))
    return next_state

def test_duckietown_ekf_slam(*args, **kwargs):

    ekf = EKF_SLAM()

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
    fig, ax = plt.subplots(1, 3)
    env.reset()
    env.render()
    obs, reward, done, info = env.step(np.array([0.0, 0.0]))
    im = ax[0].imshow(obs)
    im2 = ax[1].imshow(obs)
    im1, = ax[2].plot([], [], 'b-')
    im3, = ax[2].plot([], [], 'bo')
    im4, = ax[2].plot([], [], 'rx')
    im5, = ax[2].plot([], [], 'g-')

    ax[2].set_xlim(-1,2)
    ax[2].set_ylim(-1,2)
    
    gt_states = []
    ekf_states = []

    def animate(i):
        
        global u

        global observations

        global state

        obs, reward, done, info = env.step(u)

        # inputs = np.array(info['Simulator']['action'],dtype=np.float32)
        
        inputs = u 

        ekf.step(inputs = inputs, observed = observations)

        state = transition(state = state, inputs = inputs, dt=DT)

        line_found, cpose, curv, observations = lateral_lane_filter.process(obs)

        if line_found:
            robot_fpose = np.array([0.0, cpose[0], cpose[1]])
            error = np.array([0, 0.0]) - robot_fpose[:2]
            t_fvel = np.array([1, 0.0])
            curvature = curv
            u = controller.compute(robot_fpose, error, t_fvel, curvature)
            u = u / np.linalg.norm(u)
            u[1] *= -1

        mu_robot = ekf.mu[:ROBOT_DIM-1].reshape(-1,2)
        mu_landmarks = ekf.mu[ROBOT_DIM:].reshape(-1,2)
        ekf_states.append(mu_robot)
        gt_states.append(state)
        gt_states_array = np.stack(gt_states)
        ekf_states_array = np.concatenate(ekf_states,axis=0)
        im.set_array(lateral_lane_filter.plot_image)
        im2.set_array(obs)
        im3.set_data(mu_robot[:,0],mu_robot[:,1])
        im4.set_data(mu_landmarks[:,0],mu_landmarks[:,1])
        im5.set_data(gt_states_array[:,0],gt_states_array[:,1])
        im1.set_data(ekf_states_array[:,0],ekf_states_array[:,1])
        env.render()

        return [im, im2, im5, im4, im1, im3]
    ani = animation.FuncAnimation(fig, animate, frames=500, interval=20, blit=True)
    plt.show()
