
from ..trajectory import QuinticPolynomial,QuarticPolynomial
from ..planner import Frenet
from .utils import OptimalParameters as op
import numpy as np
import copy


class Target():
    def __init__(self, s0: (float,float,float) = (0.0,0.0,0.0), s1:(float,float,float) = (1.0,1.0,1.0), Ts: float = 1.0):
        self.s0 = s0
        self.s1 = s1
        self.Ts = Ts #must be constant for every Target
        self.DTs = 0.1 #must be constant for every Target
        self.update()
        
    def update(self):
        self.path = QuinticPolynomial(self.s0[0], self.s0[1], self.s0[2], self.s1[0], self.s1[1], self.s1[2], self.Ts)
        self.frenet = Frenet()
        self.frenet.t = [t for t in np.arange(0, self.Ts+self.DTs, self.DTs)]
        self.frenet.s = [self.path.compute_pt(t) for t in self.frenet.t]
        self.frenet.dot_s = [self.path.compute_first_derivative(t) for t in self.frenet.t]
        self.frenet.ddot_s = [self.path.compute_second_derivative(t) for t in self.frenet.t]
        self.frenet.dddot_s = [self.path.compute_third_derivative(t) for t in self.frenet.t]
        
    def get_frenet(self) -> Frenet:
        return copy.deepcopy(self.frenet)

    def get_frenet_follow_target(self, D0: float = 1) -> Frenet:
        # assert s0[2]==s1[2], 'leading Target must have constant acceleration'
        target = self.get_frenet()
        target.s = [s_lv - D0 + self.DTs * target.dot_s[t_index] for t_index,s_lv in enumerate(target.s)] # recompute s considering the distance
        target.dot_s = [dot_s_lv - self.DTs * target.ddot_s[t_index] for t_index,dot_s_lv in enumerate(target.dot_s)]
        if not all(target.ddot_s[0]==target.ddot_s[t_index] for t_index,_ in enumerate(target.t)): # not clear on paper if we should consider it or not
            target.ddot_s = [ddot_s_lv - self.DTs * target.dddot_s[t_index] for t_index,ddot_s_lv in enumerate(target.ddot_s)] # formal
        return target

    def get_frenet_merge_target(self, t_b = None) -> Frenet: # merge to another target # t_b: Target
        # assert they moves at same speed needed?
        assert t_b != None, 'please give a Target'
        target_a = self.get_frenet()
        target_b = t_b.get_frenet()
        assert self.DTs == t_b.DTs, 'sampling times must be equal'
        assert self.Ts == t_b.Ts, 'time intervals must be equal'
        target_a.s = [0.5*(s_a + s_b) for s_a, s_b  in zip(target_a.s, target_b.s)] 
        return target_a

    def get_frenet_stop_target(self) -> Frenet:
        # assert they moves at same speed needed?
        target = self.get_frenet()
        assert target.dot_s[-1] == 0, 'stopping velocity must be equal zero'
        assert target.ddot_s[-1] == 0, 'stopping acceleration must be equal zero'
        return target