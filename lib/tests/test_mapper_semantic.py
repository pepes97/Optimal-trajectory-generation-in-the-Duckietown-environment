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
from ..logger import SimulationDataStorage, SimData, timeprofiledecorator

from numpy import arange
from ..video import *
from ..controller import FrenetIOLController
from ..planner import *
from ..transform import FrenetDKTransform
from ..platform import Unicycle
from ..mapper import *

IMAGE_PATH_LST = [f'./images/dt_samples/{i}.jpg' for i in range(0, 170)]


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

def test_mapper_semantic_planner(*args, **kwargs):
    env = DuckietownEnv(seed=0,
                        map_name='loop_empty',
                        domain_rand=False,
                        frame_skip=1,
                        distortion=False,
                        camera_rand=False,
                        dynamics_rand=False)
    
    # Planner 
    planner = TrajectoryPlannerV1DT(TrajectoryPlannerParamsDT())
    # transformer 
    transformer = FrenetDKTransform()
    # Controller
    controller = FrenetIOLController(.05, 0.0, 5, 0.0, 0.0)
    # Mapper
    mapper = MapperSemantic()
    # Env initialize
    env.reset()
    env.render()
    obs, reward, done, info = env.step(np.array([0.0, 0.0]))
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
    line_found, trajectory, obstacles = mapper.process(obs)
    est_pt = transformer.estimatePosition(trajectory, robot_p)
    # Plots
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    im1 = axs[0].imshow(obs)
    im2 = axs[1].imshow(obs)
    im3 = axs[2].imshow(obs)
    # Robot pose in frenet
    robot_fpose = transformer.transform(robot_p)
    # Pose initizialize
    pos_s = s0 = (robot_fpose[0], 0.0, 0.0)
    pos_d = d0 = (robot_fpose[1], 0.0, 0.0)
    # Planner initialize
    planner.initialize(t0=0, p0=d0, s0=s0)
    dt = 1/30
    all_v = []
    all_omega= []
    all_v.append(u[0])
    all_omega.append(u[1])
    wheel_v = []
    wheel_omega = []
    wheel_v.append(u[0])
    wheel_omega.append(u[1])

    def animate(i):
        global u, robot_p, robot_dp, robot_ddp, pos_s, pos_d
        obs, reward, done, info = env.step(u)
        #actual_u = np.array(info['Simulator']['wheel_velocities'])
        actual_u = np.array(info['Simulator']['action'])
        WHEEL_DIST = 0.102
        RADIUS = 0.0318
        u_real = np.array([RADIUS/2 * (actual_u[1]+actual_u[0]), RADIUS/WHEEL_DIST * (actual_u[1]-actual_u[0])])
        u_robot = u_real/dt
        u_robot = u_robot/np.r_[1.111, 1.111]
        #u_robot = u_robot/np.r_[0.6988,0.4455]
        wheel_v.append(u_robot[0])
        wheel_omega.append(u_robot[1])
        #u_robot = u_robot/np.r_[, -1.222]
        robot_p, robot_dp = robot.step(u_robot, dt)
        line_found, trajectory, obstacles = mapper.process(obs)
        
        if line_found:
            # Estimate frenet frame
            robot_p = np.array([0.07,0.0,0.0])
            est_pt = transformer.estimatePosition(trajectory,  robot_p)
            # Robot pose in frenet
            robot_fpose = transformer.transform(robot_p)
            # Get replanner step
            pos_s, pos_d = planner.replanner(time = i*dt)
            mapper.proj_planner = trajectory.compute_pt(est_pt)
            mapper.path_planner = frenet_to_glob(trajectory, planner, est_pt)
            #Compute error
            error = np.array([0, pos_d[0]]) - robot_fpose[0:2]
            derror = np.array([pos_s[1], pos_d[1]])
            # Get curvature
            curvature = trajectory.compute_curvature(est_pt)
            # Compute control
            print(f"Input u robot: {u_robot}")
            u = controller.compute(robot_fpose, error, derror, curvature)
            #u = u*np.r_[0.6988,0.4455]
            all_v.append(u[0])
            all_omega.append(u[1])
            print(f"Controll u: {u}")  
            print(f"Rapporto v: {u_robot[0]/u[0]} e omega :{u_robot[1]/u[1]}")
            
        im1.set_data(obs)
        im2.set_data(mapper.plot_image_w)
        im3.set_data(mapper.plot_image_p)
        env.render()
        return [im1, im2, im3]
    ani = animation.FuncAnimation(fig, animate, frames=1100, interval=50, blit=True)
    #ani.save("./prova.mp4", writer="ffmpeg")
    ani.save("./images/duckietown_video/planner_without_obstacles_b05.mp4")
    #plt.show()
    t = np.arange(0, 1102*(1/30), 1/30)
    fig, ax = plt.subplots(2,1, figsize= (15,5))
    #Plot control outputs
    #len_tot = trajectory.get_len()
    #u = data_storage.get(SimData.control)
    
    line, = ax[0].plot(t,all_v, color = "r", label='v',  linewidth=2)
    line2, = ax[0].plot(t,all_omega, color = "g", label='$\omega$',  linewidth=2)
    # line, = ax.plot(u[0, :230], color = "r", label='v')
    # line2, = ax.plot(u[1, :230], color = "g", label='$\omega$')
    ax[0].set_title("Velocities at controller output without obstacles")
    ax[0].legend([r"$v (m/s)$", r"$\omega (rad/s)$"])
    ax[0].set_xlabel(r"$t (s)$")
    ax[0].set_ylabel(r"$u$")
    #ax[0].set_xlim(-0.5,5)
    ax[0].set_ylim(-3,3)
    ax[0].set_aspect('equal', 'box')
    ax[0].grid(True)

    line3, = ax[1].plot(t,wheel_v, color = "r", label='v', linewidth=2)
    line4, = ax[1].plot(t,wheel_omega, color = "g", label='$\omega$', linewidth=2)
    # line, = ax.plot(u[0, :230], color = "r", label='v')
    # line2, = ax.plot(u[1, :230], color = "g", label='$\omega$')
    ax[1].set_title("Velocities at robot input without obstacles")
    ax[1].legend([r"$v (m/s)$", r"$\omega (rad/s)$"])
    ax[1].set_xlabel(r"$t (s)$")
    ax[1].set_ylabel(r"$u$")
    #ax[1].set_xlim(-0.5,5)
    ax[1].set_ylim(-3,3)
    ax[1].set_aspect('equal', 'box')
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig("./images/velocities/final_no_obs_2.png")
    plt.show()
    
    

