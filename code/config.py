# Global parameters
GLOBAL_D_T = 0.05

# Parameters
MAX_ROAD_WIDTH = 2.45 # maximum road width [m]
D_ROAD_W = 0.6 # road width sampling length [m]
T_0 = 0  # initial time [s]

D_T = 0.8 # time tick [s]
MAX_T = 5.0 # max prediction time [m]
MIN_T = 0.9 # min prediction time [m]

# # for xy coordinates (figure 6)
# D_T = 0.5 # time tick [s]
# MAX_T = 6.0 # max prediction time [m]
# MIN_T = 3.0 # min prediction time [m]

DES_SPEED = 5.0 # speed desired [m/s]
D_D_S = 1  # target speed sampling length [m/s]
N_S_SAMPLE = 3  # sampling number of target speed
LOW_SPEED_THRESH = 2.0 # low speed switch [m/s]
S_THRSH = 1.0
#Cost weights
K_J = 0.001
K_T = 0.01
K_D = 0.9
K_S = 0.1
K_DOT_S = 1.0
K_LAT = 1.0
K_LONG = 1.0

TARGET_DIST = 1 #[m] distance from following target