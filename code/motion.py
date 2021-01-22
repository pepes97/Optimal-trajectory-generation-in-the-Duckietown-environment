from Frenet import Frenet
from quintic_polynomial import QuinticPolynomial
from quartic_polynomial import QuarticPolynomial
import numpy as np
import matplotlib.pyplot as plt

# constants K

KJ = 0.1
KT = 0.1
KD = 1.0
KDOT = 1.0
KLAT = 1.0
KLON = 1.0

# for lateral motion 

DI_INTERVAL = 7 # [m]
STEP = 0.5 # [m]

MIN_TIME = 1.0 # [s]
MAX_TIME = 10.0 # [s]
STEP_TIME = 0.5 # [s]

# for longitudinal motion

DELTA_SI = 1 # [m/s]
TARGET_SPEED = 4 # [m/s]


def lateral_motionLST_and_longitudinal_motionVK(p0, dp0, ddp0, s0, ds0):

    """
        Lateral Movement and Longitudinal movement with velocity keeping

        Start state = [p0, dp0, ddp0] using for coordinate d of Franet Frame
        Start state = [s0, ds0] using for coordinate s of Franet Frame

    """

    frenet_paths = []
    for di in np.arange(-DI_INTERVAL, DI_INTERVAL, STEP):

        # Lateral motion planning

        for Tj in np.arange(MIN_TIME,MAX_TIME, STEP_TIME):
            
            f = Frenet()

            latQP = QuinticPolynomial(p0, dp0, ddp0, di, 0.0, 0.0, Tj)

            f.t = [t for t in np.arange(0.0, Tj, STEP_TIME)]
            f.d = [latQP.compute_pt(t) for t in f.t]
            f.dot_d = [latQP.compute_first_derivative(t) for t in f.t]
            f.ddot_d = [latQP.compute_second_derivative(t) for t in f.t]
            f.dddot_d = [latQP.compute_third_derivative(t) for t in f.t]

            # square jerk

            Jlat = sum(np.power(f.dddot_d, 2)) 

            # control

            f.cd = KJ * Jlat + KT * Tj + KD * f.d[-1] ** 2

            # Longitudinal motion planning 

            for si in np.arange (0, TARGET_SPEED + DELTA_SI):
                lonQP = QuarticPolynomial(s0, ds0, 0.0, si, 0.0, Tj)
                f.s = [lonQP.compute_st(t) for t in f.t]
                f.dot_s = [lonQP.compute_first_derivative(t) for t in f.t]
                f.ddot_s = [lonQP.compute_second_derivative(t) for t in f.t]
                f.dddot_s = [lonQP.compute_third_derivative(t) for t in f.t]

                # square of jerk

                Jt = sum(np.power(f.dddot_s, 2))  

                # control

                f.cv = KJ*Jt + KT*Tj + KDOT*((f.dot_s[-1]-TARGET_SPEED)**2)

                f.tot = KLAT * f.cd + KLON * f.cv

            frenet_paths.append(f)

    for i in range(len(frenet_paths)):
        f = frenet_paths[i]
        plt.plot(f.t,f.d)
        plt.xlabel("t/s")
        plt.ylabel("d/m")
    plt.show()
        
    return frenet_paths