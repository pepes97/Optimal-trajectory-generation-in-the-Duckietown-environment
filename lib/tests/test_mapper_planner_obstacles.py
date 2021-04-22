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
from ..sensor import StaticObstacle, ProximitySensor
from operator import attrgetter
from ..video.constants import DuckietownParameters as dp


IMAGE_PATH_LST = [f'./images/dt_samples/{i}.jpg' for i in range(0, 170)]

def frenet_to_glob_planner(planner, trajectory, frenet_paths, projection):
    for path in frenet_paths:
        path.x=[]
        path.y=[]
        for i in range(len(path.s)):
            s = projection + (path.s[i] - planner.s0[0])
            d = path.d[i] 
            target_pos = trajectory.compute_pt(s) + \
            compute_ortogonal_vect(trajectory, s) * d
            path.x.append(target_pos[0])
            path.y.append(target_pos[1])

    return frenet_paths

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

def check_collisions(path,obstacles):

    for ob in obstacles:
        obx = ob[0]
        oby = ob[1]
        distance = [((ix - obx) ** 2 + (iy - oby) ** 2) for ix, iy in zip(path.x, path.y)]
        collision = any([di <= (dp.AGENT_SAFETY_RAD)**2 for di in distance])
        stop_before = True if path.x[-1]<obx else False
        if collision or stop_before:
            return False
    return True               

def check_offroad(path,params,side):
    h,k,a = params
    distance = [(y - k) - a * pow((x - h),2) for x, y in zip(path.x, path.y)]
    if side == 'right':
        collision = any([di<0 for di in distance])
    else:
        collision = any([di>0 for di in distance])
    return collision           

def obstacles_coordinates(obstacles,mapper):
    obstacles_list = []
    obstacles_list_c = []
    obstacles_list_r = []
    obstacles_list_lat = []
    obstacles_list_top = []
    for ob in obstacles:
        obstacles_list.append(ob["end_point"])
        obstacles_list_c.append(ob["center"])
        obstacles_list_r.append(ob["end_right"])
        obstacles_list_lat.append(ob["end_lat"])
        obstacles_list_top.append(ob["end_top"])
    obstacles_list = np.array(obstacles_list)
    obstacles_list_c = np.array(obstacles_list_c)
    obstacles_list_r = np.array(obstacles_list_r)
    obstacles_list_lat = np.array(obstacles_list_lat)
    obstacles_list_top = np.array(obstacles_list_top)

    if obstacles_list.shape[0] > 0:
        obs_rob = mapper.cam2rob(obstacles_list)
        obs_rob_c = mapper.cam2rob(obstacles_list_c)
        obs_rob_r = mapper.cam2rob(obstacles_list_r)
        obs_rob_lat = mapper.cam2rob(obstacles_list_lat)
        obs_rob_top = mapper.cam2rob(obstacles_list_top)
        return obs_rob,obs_rob_c, obs_rob_r,obs_rob_lat,obs_rob_top
    else:
        return [], [],[],[],[]

def get_vertex_and_focus_distance(fit: np.array):
    a, b, c = fit
    h = -b/(2*a)
    k = -(b**2-4*a*c)/(4*a)
    return h, k, a

def check_paths(frenet_paths, obstacles, rpose, mapper, rwfit, lwfit):

    measure_obst, measure_obst_c,measure_obst_r,measure_obst_lat,measure_obst_top = obstacles_coordinates(obstacles,mapper)
    new_paths_idx = []
    for i in range(len(frenet_paths)):
        if np.array(rwfit!=None).all():
            offroad = check_offroad(frenet_paths[i], get_vertex_and_focus_distance(rwfit),'right')
            if offroad:
                continue
        if np.array(lwfit!=None).all():
            offroad = check_offroad(frenet_paths[i], get_vertex_and_focus_distance(lwfit),'left')
            if offroad:
                continue
        if measure_obst!= []:
            collision_r = check_collisions(frenet_paths[i], measure_obst_r)
            if not collision_r:
                continue
            else:
                collision_lat = check_collisions(frenet_paths[i], measure_obst_lat)
                if not collision_lat:
                    continue
                else:
                    collision_top = check_collisions(frenet_paths[i], measure_obst_top)
                    if not collision_top:
                        continue
        new_paths_idx.append(i) 
    return [frenet_paths[i] for i in new_paths_idx]

def compute_ortogonal_vect(trajectory, s):
    ds = 1/30
    s1 = s + ds
    t_grad = trajectory.compute_pt(s1) - trajectory.compute_pt(s)
    t_r = np.arctan2(t_grad[1], t_grad[0])
    return np.array([-np.sin(t_r), np.cos(t_r)])

def test_mapper_semantic_planner_obstacles(*args, **kwargs):
    env = DuckietownEnv(seed=0,
                        map_name='loop_obstacles',
                        camera_rand=False,
                        frame_skip=1,
                        domain_rand=False,
                        dynamics_rand=False,
                        distortion=False)
    
    # Planner 
    planner = TrajectoryPlannerV1DTObstacles(TrajectoryPlannerParamsDTObstacles())
    # transformer 
    transformer = FrenetDKTransform()
    # Controller
    controller = FrenetIOLController(.5, 0.0, 27, 0.0, 0.0)
    # Mapper
    mapper = MapperSemanticObstacles()
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
    line_found, trajectory, obstacles, rwfit, lwfit, rw, lw = mapper.process(obs)

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
    switch = {'gate1':False,'gate2':False, 'tolerance':0}
    def animate(i):
        global u, robot_p, robot_dp, robot_ddp, pos_s, pos_d
        obs, reward, done, info = env.step(u)
        actual_u = np.array(info['Simulator']['action'])
        robot_p, robot_dp = robot.step(actual_u, dt)
        line_found, trajectory, obstacles, rwfit, lwfit, rw, lw = mapper.process(obs, verbose = 1)


        if line_found:
            # Estimate frenet frame
            robot_p = np.array([0.1,0.0,0.0])
            est_pt = transformer.estimatePosition(trajectory,  robot_p)
            # Robot pose in frenet
            robot_fpose = transformer.transform(robot_p)
            robot_fdp  = transformer.transform(robot_dp)[0:2]
            # Get replanner step
            pos_s, pos_d = planner.replanner(time = i*dt)
            paths_planner = planner.paths
            # mapper.obst_acle = obstacles_coordinates(obstacles,mapper)
            paths_planner = frenet_to_glob_planner(planner, trajectory, paths_planner, est_pt)
            # Check paths that do not encounter obstacles
            paths_planner = check_paths(paths_planner, obstacles, robot_p, mapper, rw, lw)
            if obstacles!=[]:
                planner.opt_path_tot = min(paths_planner, key=attrgetter('ctot'))
            # Planner in mapper
            mapper.proj_planner = trajectory.compute_pt(est_pt)
            mapper.path_planner = frenet_to_glob(trajectory, planner, est_pt)
            mapper.all_paths = paths_planner
            #Compute error
            error = np.array([0, pos_d[0]]) - robot_fpose[0:2]
            derror = np.array([pos_s[1], pos_d[1]])
            # Get curvature
            curvature = trajectory.compute_curvature(est_pt)
            # Compute control
            u = controller.compute(robot_fpose, error, derror, curvature)#/np.r_[0.6988, 0.4455]
            # linear_control = desired_linear_speed_control/0.6988
            # angular_control = desired_angular_speed_control/0.4455
        im1.set_data(obs)
        im2.set_data(mapper.plot_image_w)
        im3.set_data(mapper.plot_image_p)
        env.render()
        return [im1, im2, im3]
    ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50, blit=True)
    # ani.save("./images/duckietown_video/planner_with_obstacles_2.mp4", writer="ffmpeg")
    plt.show()
    

