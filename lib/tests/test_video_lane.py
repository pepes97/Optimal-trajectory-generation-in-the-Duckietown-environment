"""test_video_lane.py
"""

import numpy as np
import cv2
import logging
from matplotlib import pyplot as plt
from ..plotter import *
from ..video import *

logger = logging.getLogger(__name__)

IMAGE_PATH_LST = ['./images/dt_samples/01.png',
                  './images/dt_samples/02.png',
                  './images/dt_samples/03.png',
                  './images/dt_samples/04.png',
                  './images/dt_samples/05.png',
                  './images/dt_samples/06.png',]


def test_video_lane(*args, **kwargs):
    #lane_filter = LaneFilter()
    line_filter = CenterLineFilter()
    line_filter.yellow_thresh = (23, 28)
    perspective_projector = PerspectiveWarper()
    line_tracker = SlidingWindowTracker()
    fig, axs = plt.subplots(2, len(IMAGE_PATH_LST), figsize=(10, 5))
    for i, impath in enumerate(IMAGE_PATH_LST):
        frame = cv2.imread(impath)[:,:,::-1]
        warped_frame = perspective_projector.warp(frame)
        thresholded_frame = line_filter.process(warped_frame)
        out_image, line_fit = line_tracker.search(thresholded_frame, draw_windows=True)
        if (line_fit != np.zeros(3)).all():
            # Line is found
            # plot line
            ploty = np.arange(0, warped_frame.shape[0]-1, 1)
            line_fit = line_fit[0] * ploty**2 + line_fit[1] * ploty + line_fit[2]
            line_unwarped = np.vstack((line_fit, ploty))
            line_unwarped = line_unwarped.reshape(-1, 1, 2)
            line_unwarped = cv2.perspectiveTransform(line_unwarped, perspective_projector.iM)
            axs[1, i].plot(line_fit, ploty, 'y', linewidth=5)
            # TODO FIX
            #axs[0, i].plot(line_unwarped[:, 0, 1], line_unwarped[:, 0, 0], 'y', 15)
        else:
            print('No line found.')
        axs[0, i].imshow(frame)
        axs[1, i].imshow(warped_frame)
    plt.tight_layout()
    plt.show()
    """
    fig, axs = plt.subplots(1, len(IMAGE_PATH_LST))
    lane_filter = CenterLineFilter()
    for i, impath in enumerate(IMAGE_PATH_LST):
        frame = cv2.imread(impath)
        proc_frame = lane_filter.process(frame)
        axs[i].imshow(proc_frame)
    plt.show()"""
    
