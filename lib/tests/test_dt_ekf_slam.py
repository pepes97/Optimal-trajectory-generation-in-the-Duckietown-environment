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
from ..planner import *
from ..localization import EKF_SLAM
from ..localization import *
from ..platform import Unicycle
from ..transform import FrenetDKTransform


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


def transition(state: np.ndarray = np.zeros((ROBOT_DIM,)), \
                controls: np.ndarray = np.zeros((CONTROL_DIM,),dtype=np.float32)):
    # predict the robot motion, the transition model f(x,u)
    # only affects the robot pose not the landmarks
    # odometry euler integration
    next_state = np.zeros((ROBOT_DIM,))
    x, y, th = state[0], state[1], state[2]
    # dp, dth are infinitesimal increment dp = DT * v , dth = DT * w
    dp, dth = controls[0], controls[1]
    next_state[0] = x + dp * np.cos(th)
    next_state[1] = y + dp * np.sin(th)
    th = th + dth
    next_state[2] = np.arctan2(np.sin(th),np.cos(th))
    return next_state


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

def test_duckietown_ekf_slam(*args, **kwargs):

    ekf = EKF_SLAM()

    env = DuckietownEnv(seed=123,
                        map_name='loop_empty',
                        camera_rand=False)
    perspective_projector = PerspectiveWarper()
    line_filter = CenterLineFilter()
    line_tracker = SlidingWindowTracker(robust_factor=1)
    lat_line_filter = LateralLineFilter()
    transformer = FrenetDKTransform()
    lat_line_tracker = SlidingWindowDoubleTracker(robust_factor=1)
    lateral_lane_filter = TrajectoryFilter(perspective_projector, line_filter, lat_line_filter, lat_line_tracker)
    controller = FrenetIOLController(.5, 0.0, 27, 0.0, 0.0)
    fig, ax = plt.subplots(1, 3)
    env.reset()
    env.render()
    obs, *_ = env.step(np.array([0.0, 0.0]))
    im = ax[0].imshow(obs)
    im2 = ax[1].imshow(obs)
    im1, = ax[2].plot([], [], 'b-') # estimated path
    im3, = ax[2].plot([], [], 'bo') # estimated current position
    im4, = ax[2].plot([], [], 'rx') # estimated landmarks
    im5, = ax[2].plot([], [], 'g-') # odometry only path

    ax[2].set_xlim(-1,2)
    ax[2].set_ylim(-1,2)
    
    gt_states = []
    ekf_states = []

    global u, state, observations

    observations = np.array([]).reshape(0,2)
    state = np.zeros((3,))        
 
    planner = TrajectoryPlannerV1DT(TrajectoryPlannerParamsDT())
    global u, robot_p, robot_dp, robot_ddp, pos_s, pos_d
    robot = Unicycle()
    robot.set_initial_pose(robot.p)
    robot_p = np.zeros(3)
    robot_dp = np.zeros(3)
    robot_ddp = np.zeros(3)
    u = np.zeros(2)
    # Initialization
    line_found, trajectory, observations = lateral_lane_filter.process(obs)
    est_pt = transformer.estimatePosition(trajectory, robot_p)
    # Robot pose in frenet
    robot_fpose = transformer.transform(robot_p)
    
    pos_s = s0 = (robot_fpose[0], 0.0, 0.0)
    pos_d = d0 = (robot_fpose[1], 0.0, 0.0)

    planner.initialize(t0=0, p0=d0, s0=s0)

    dt = 1/30

    def animate(i):
        global u, robot_dp, robot_ddp, pos_s, pos_d, state, observations

        obs, reward, done, info = env.step(u)

        # controls = np.array(info['Simulator']['action'])

        robot_p, robot_dp = robot.step(u=u, dt=dt)

        ekf.step(controls = u * dt, observations = observations)

        line_found, trajectory, observations = lateral_lane_filter.process(obs)

        if line_found:
            # Estimate frenet frame
            robot_b = np.array([0.1,0.0,0.0])
            est_pt = transformer.estimatePosition(trajectory,  robot_b)
            # Robot pose in frenet
            robot_fpose = transformer.transform(robot_b)
            # Get replanner step
    
            planner.p0=(robot_fpose[1],planner.p0[1],planner.p0[2]) # this is just to avoid blue trajectory to converge immediately
            pos_s, pos_d = planner.replanner(time = i*dt)

            lateral_lane_filter.proj_planner = trajectory.compute_pt(est_pt)
            lateral_lane_filter.path_planner = frenet_to_glob(trajectory, planner, est_pt)

            #Compute error
            error = np.array([0, pos_d[0]]) - robot_fpose[0:2]
            derror = np.array([pos_s[1], pos_d[1]])
            # Get curvature
            curvature = trajectory.compute_curvature(est_pt)
            # Compute control
            u = controller.compute(robot_fpose, error, derror, curvature)
            if np.linalg.norm(u) != 0: 
                u = u / np.linalg.norm(u)
            # Step the unicycle            
            
            # print(f'fpose={robot_fpose}, u={u}') 

        mu_robot = ekf.mu[:ROBOT_DIM-1].reshape(-1,2)
        mu_landmarks = ekf.mu[ROBOT_DIM:].reshape(-1,2)
        ekf_states.append(mu_robot)
        gt_states.append(robot_p[:2].copy())
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
