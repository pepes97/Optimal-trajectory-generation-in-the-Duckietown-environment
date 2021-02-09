import math
import matplotlib.pyplot as plt
import numpy as np

class Circle:

    def __init__(self, l, h, r):
        self.l = l
        self.h = h
        self.r = r


    def path(self, s):
        s = float(s)
        while s >= 2*self.l + 2*self.r*math.pi + 2*self.h:
            s = s - (2*self.l + 2*self.r*math.pi + 2*self.h)

        # first segment
        if s>=0 and s<self.l:
            return (s, 0.)
        # first 1/4 circle
        elif s>=self.l and s< self.l + 2*self.r*math.pi/4:
            alpha = (s - self.l)/self.r
            return (self.l + self.r*math.sin(alpha), self.r - self.r*math.cos(alpha))
        # second segment
        elif s>= self.l + 2*self.r*math.pi/4 and s< self.l + 2*self.r*math.pi/4 + self.h:
            return (self.l + self.r , s - self.l - 2*self.r*math.pi/4 + self.r)
        # second 1/4 circle
        elif s>= self.l + 2*self.r*math.pi/4 + self.h and s < self.l + self.r*math.pi + self.h:
            alpha = (s - (self.l + 2*self.r*math.pi/4 + self.h))/self.r
            return (self.l + math.cos(alpha)*self.r, self.r + self.h + math.sin(alpha)*self.r)
        # third segment
        elif s>= self.l + self.r*math.pi + self.h and s < 2*self.l + self.r*math.pi + self.h:
            return (self.l - (s - self.l - self.r*math.pi - self.h), 2*self.r+self.h)
        # third 1/4 circle
        elif s>= 2*self.l + self.r*math.pi + self.h and s< 2*self.l + self.r*math.pi + self.h + 2*self.r*math.pi/4:
            alpha = (s - (2*self.l + self.r*math.pi + self.h))/self.r
            return (-math.sin(alpha)*self.r, self.r+ self.h+self.r*math.cos(alpha))
        # fourth segment
        elif s>= 2*self.l + self.r*math.pi + self.h + 2*self.r*math.pi/4 and s< 2*self.l + self.r*math.pi + 2*self.h + 2*self.r*math.pi/4:
            return (-self.r, self.r+ self.h - (s-(2*self.l + self.r*math.pi + self.h + 2*self.r*math.pi/4)))
        # fourth 1/4 circle
        else:
            alpha = (s - (2*self.l + self.r*math.pi + 2*self.h + 2*self.r*math.pi/4))/self.r
            return (-self.r*math.cos(alpha), self.r- self.r*math.sin(alpha))

    def get_curvature(self, s):
        s = float(s)
        while s >= 2*self.l + 2*self.r*math.pi + 2*self.h:
            s = s - (2*self.l + 2*self.r*math.pi + 2*self.h)

        # first segment
        if s>=0 and s<self.l:
            return 0.
        # first 1/4 circle
        elif s>=self.l and s< self.l + 2*self.r*math.pi/4:
            return 1./self.r
        # second segment
        elif s>= self.l + 2*self.r*math.pi/4 and s< self.l + 2*self.r*math.pi/4 + self.h:
            return 0.
        # second 1/4 circle
        elif s>= self.l + 2*self.r*math.pi/4 + self.h and s < self.l + self.r*math.pi + self.h:
            return 1./self.r
        # third segment
        elif s>= self.l + self.r*math.pi + self.h and s < 2*self.l + self.r*math.pi + self.h:
            return 0.
        # third 1/4 circle
        elif s>= 2*self.l + self.r*math.pi + self.h and s< 2*self.l + self.r*math.pi + self.h + 2*self.r*math.pi/4:
            return 1./self.r
        # fourth segment
        elif s>= 2*self.l + self.r*math.pi + self.h + 2*self.r*math.pi/4 and s< 2*self.l + self.r*math.pi + 2*self.h + 2*self.r*math.pi/4:
            return 0.
        # fourth 1/4 circle
        else:
            return 1./self.r

    def get_theta_r(self, s):
        s = float(s)
        while s >= 2*self.l + 2*self.r*math.pi + 2*self.h:
            s = s - (2*self.l + 2*self.r*math.pi + 2*self.h)

        # first segment
        if s>=0 and s<self.l:
            return 0.
        # first 1/4 circle
        elif s>=self.l and s< self.l + 2*self.r*math.pi/4:
            alpha = ( s - self.l)/self.r
            return alpha
        # second segment
        elif s>= self.l + 2*self.r*math.pi/4 and s< self.l + 2*self.r*math.pi/4 + self.h:
            return math.pi/2
        # second 1/4 circle
        elif s>= self.l + 2*self.r*math.pi/4 + self.h and s < self.l + self.r*math.pi + self.h:
            alpha = (s - (self.l + 2*self.r*math.pi/4 + self.h))/self.r
            return math.pi/2+alpha
        # third segment
        elif s>= self.l + self.r*math.pi + self.h and s < 2*self.l + self.r*math.pi + self.h:
            return math.pi
        # third 1/4 circle
        elif s>= 2*self.l + self.r*math.pi + self.h and s< 2*self.l + self.r*math.pi + self.h + 2*self.r*math.pi/4:
            alpha = (s - (2*self.l + self.r*math.pi + self.h))/self.r
            return math.pi+alpha
        # fourth segment
        elif s>= 2*self.l + self.r*math.pi + self.h + 2*self.r*math.pi/4 and s< 2*self.l + self.r*math.pi + 2*self.h + 2*self.r*math.pi/4:
            return (3/2)*math.pi
        # fourth 1/4 circle
        else: 
            alpha = (s - (2*self.l + self.r*math.pi + 2*self.h + 2*self.r*math.pi/4))/self.r
            return (3/2)*math.pi + alpha

    def get_len(self):
        return 2*self.l + 2*self.r*math.pi + 2*self.h


circle = Circle(8, 5, 2)
samples = np.arange(0., 70., 0.1)
coord = []
for s in samples:
    coord += [circle.path(s)]

x = [c[0] for c in coord]
y = [c[1] for c in coord]

plt.plot(x, y)
plt.show()