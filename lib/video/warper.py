"""warper.py
"""

import cv2
import numpy as np
import logging
import functools
from .utils import *
from .constants import DuckietownParameters as dp

class PerspectiveWarperOld:
    def __init__(self, dest_size=(640, 480),
                 src=np.float32([
                     (0.3796, 0.4438),
                     (0, 0.9396),
                     (1, 0.9396),
                     (0.6281, 0.4438)]),
                 dest=np.float32([(0.4, 0.5), (0.4, 1), (0.6, 1), (0.6, 0.5)])):
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

class PerspectiveWarper:
    def __init__(self, dest_size=(640, 480),
                 src=np.float32([(0, dp.pM_corr), (1, dp.pM_corr), (0, 1), (1, 1)]),
                 dest=np.float32([(0, 0), (1, 0), (0.5-dp.X/2, 1), (0.5+dp.X/2, 1)])):
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
        
