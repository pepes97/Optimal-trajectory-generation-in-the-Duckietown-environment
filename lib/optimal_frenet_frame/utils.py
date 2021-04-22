import numpy as np
import logging

logger = logging.getLogger(__name__)


class OptimalParameters:
    # initial state
    p = (3,0.3,0) 
    s = (0,2,-0.5)
    s0_t = (1,2,0)
    s1_t = (8,0,0)
    Ts = 10
    t0 = 0 # instant in which the planner is initialized
    # replanning instants
    Tn = [0, 2.5, 5]
    T = np.arange(0,5,0.5)
    s0_tb = (3,2,0)
    s1_tb = (10,0,0)

    px= (3,0.2,0)
    sx = (0.8,2,-0.5)
op = OptimalParameters