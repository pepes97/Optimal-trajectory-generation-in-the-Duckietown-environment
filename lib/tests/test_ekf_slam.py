import numpy as np
from ..localization import EKF_SLAM
from ..localization import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
np.random.seed(41296)

def transition(state: np.ndarray = np.zeros((ROBOT_DIM,)), \
                inputs: np.ndarray = np.zeros((CONTROL_DIM,),dtype=np.float32), dt : float = DT):
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

def control():
    CONTROLS = [-1.,0.,1.,1.]
    out = np.random.choice(CONTROLS,2)
    out[1]*=np.pi/2
    return out

def sense(state: np.ndarray = np.zeros((ROBOT_DIM,)), \
            landmarks: np.ndarray = np.zeros((1,LANDMARK_DIM)), \
            sensor_radius: int = 5, noise: float = NOISE_Z):
    
    x, y, th = state[0], state[1], state[2]
    xl, yl = landmarks[:,0], landmarks[:,1]
    dist = np.sqrt((x-xl)**2+(y-yl)**2)
    sensed_indices = np.where(dist<sensor_radius)[0]
    sensed = np.take(landmarks,sensed_indices,axis=0)
    if sensed.shape[0] == 0 : return np.array([]).reshape(0,2)
    t = state[:2].reshape(2,1)
    l = sensed[:,:2].T
    c, s = np.cos(th), np.sin(th)
    RT = np.array([[c, s],[-s, c]], dtype = np.float32)
    dt = l - t
    sensed = RT @ dt
    sensed = sensed.T
    sensed += np.random.uniform(-noise,noise,sensed.shape)
    return sensed

def rand_map(steps: int = 100, map_size: int = 10, \
            num_landmarks: int = 100, sensor_radius: int = 3, \
            noise: float = NOISE_U, plot = False):

    states = np.zeros((steps,ROBOT_DIM))
    landmarks = np.random.uniform(-map_size//2,map_size//2,(num_landmarks,LANDMARK_DIM))
    controls = np.zeros((steps,CONTROL_DIM))
    observations = []
    state = np.zeros((ROBOT_DIM,))
    inputs = np.zeros((CONTROL_DIM,))
    for t in range(steps):
        next_state = np.array([map_size,map_size,0])
        while np.any(np.abs(next_state[:2]) > map_size//2):
            inputs = control()
            next_state = transition(state, inputs)
        state = next_state
        controls[t,:] = inputs + np.random.uniform(-noise,noise,inputs.shape)
        states[t,:] = state
        observations.append(sense(state, landmarks, sensor_radius).copy())

    if plot:
        plt.plot(states[0,0],states[0,1],'go')
        plt.plot(states[-1,0],states[-1,1],'gx')
        plt.plot(states[:,0],states[:,1],'g-')
        plt.plot(landmarks[:,0],landmarks[:,1],'bo')
        plt.show()
    
    return states, controls, observations, landmarks

def test_ekf_slam(*args, **kwargs):
    
    plot_flag = False
    store_plot = None
    if 'plot' in kwargs:
        plot_flag = kwargs['plot']
    if 'store_plot' in kwargs:
        store_plot = kwargs['store_plot']

    ekf = EKF_SLAM()

    map_size = 20
    steps = 100
    sensor_radius = 5
    num_landmarks = 100
    
    states, controls, observations, landmarks = rand_map(steps = steps,\
                num_landmarks = num_landmarks, map_size = map_size, \
                sensor_radius = sensor_radius, plot=True)

    fig, ax = plt.subplots()
    real_state_plt, = plt.plot([], [], 'g-')
    real_landmark_plt, = plt.plot([], [], 'bo')
    obs_landmark_plt, = plt.plot([], [], 'go')
    predicted_state_plt, = plt.plot([], [], 'r-')
    predicted_landmark_plt, = plt.plot([], [], 'r*')
    real_landmarks = []
    observed_landmarks = []
    real_states = []
    predicted_states = []

    def init():
        ax.set_xlim(-map_size//2-1, map_size//2+1)
        ax.set_ylim(-map_size//2-1, map_size//2+1)
        return real_state_plt, real_landmark_plt, predicted_state_plt, predicted_landmark_plt

    def update(t):
        
        inputs = controls[t,:].copy()

        state = states[t,:].copy().reshape(3,1)
        observed = observations[t]

        # HERE DO EKF
        ekf.step(inputs = inputs, observed = observed)

        t, th = state[0:2,0].reshape(2,1), state[2,0]
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s],[s, c]], dtype = np.float32)
        observed = R @ observed.T + t

        real_landmarks.append(observed.T)
        real_states.append(state.reshape(1,3))
        predicted_states.append(ekf.mu[:3].reshape(1,3).copy())

        real_landmark = np.concatenate(real_landmarks,axis=0)
        real_state = np.concatenate(real_states,axis=0)
        predicted_landmark = ekf.mu[ROBOT_DIM:].reshape(-1,2)
        predicted_state = np.concatenate(predicted_states,axis=0)

        real_landmark_plt.set_data(landmarks[:,0],landmarks[:,1])
        real_state_plt.set_data(real_state[:,0], real_state[:,1])
        obs_landmark_plt.set_data(real_landmark[:,0], real_landmark[:,1])
        predicted_state_plt.set_data(predicted_state[:,0], predicted_state[:,1])
        predicted_landmark_plt.set_data(predicted_landmark[:,0], predicted_landmark[:,1])
        
        return real_state_plt, real_landmark_plt, obs_landmark_plt, predicted_state_plt, predicted_landmark_plt

    ani = FuncAnimation(fig, update, frames=np.arange(0, steps),
                        init_func=init, blit=True, repeat=False)
    if plot_flag:
        if store_plot is not None:
            ani.save(store_plot)
    plt.show()

