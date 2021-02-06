from Frenet import Frenet
from quintic_polynomial import QuinticPolynomial
from quartic_polynomial import QuarticPolynomial
import config as cfg
import numpy as np

def frenet_follow_target(s0: (float,float,float), s1:(float,float,float), Ts: float) -> Frenet:
    DTs = cfg.GLOBAL_D_T
    path_target = QuinticPolynomial(s0[0], s0[1], s0[2], s1[0], s1[1], s1[2], Ts)
    target = Frenet()
    target.t = [t for t in np.arange(0, Ts+DTs, DTs)]
    target.s = [path_target.compute_pt(t) for t in target.t]
    target.dot_s = [path_target.compute_first_derivative(t) for t in target.t]
    target.ddot_s = [path_target.compute_second_derivative(t) for t in target.t]
    target.dddot_s = [path_target.compute_third_derivative(t) for t in target.t]
    return target