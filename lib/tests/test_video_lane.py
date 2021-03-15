"""test_video_lane.py
"""

import numpy as np
import cv2
import logging
from matplotlib import pyplot as plt
from ..plotter import *
from ..video import *

logger = logging.getLogger(__name__)

IMAGE_PATH_LST = ['./images/dt_samples/30.jpg',
                  #'./images/dt_samples/32.jpg',
                  #'./images/dt_samples/43.jpg',
                  './images/dt_samples/45.jpg',
                  './images/dt_samples/56.jpg',
                  #'./images/dt_samples/67.jpg',
]

IMAGE_PATH_LST = [f'./images/dt_samples/{i}.jpg' for i in range(1, 100)]


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
        line_fit, out_image, _ = line_tracker.search(thresholded_frame, draw_windows=True)
        if (line_fit != np.zeros(3)).all():
            # Line is found
            # plot line
            ploty = np.arange(0, warped_frame.shape[0]-1, 1)
            line_fit = line_fit[0] * ploty**2 + line_fit[1] * ploty + line_fit[2]
            line_unwarped = np.vstack((line_fit, ploty))
            line_unwarped = np.expand_dims(line_unwarped, axis=1)
            line_unwarped = np.transpose(line_unwarped, (2, 1, 0))
            line_unwarped = cv2.perspectiveTransform(line_unwarped, perspective_projector.iM)
            axs[1, i].plot(line_fit, ploty, 'b', linewidth=3)
            # TODO FIX
            axs[0, i].plot(line_unwarped[:, 0, 0], line_unwarped[:, 0, 1], 'b', linewidth=3)
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

def test_video_lane_obstacles(*args, **kwargs):
    yellow_filter = CenterLineFilter()
    yellow_filter.yellow_thresh = (20, 35)
    yellow_filter.s_thresh = (65, 190)
    yellow_filter.l_thresh = (30, 255)
    white_filter = LateralLineFilter()
    segmentator = Segmentator()
    perspective_projector = PerspectiveWarper()
    obstacle_finder = ObstacleDetector()
    line_tracker = SlidingWindowTracker()
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    axs[0].set_title('Frame')
    axs[1].set_title('Frame Warped')
    axs[2].set_title('Yellow segmented')
    im0 = im1 = im2 = None
    for i, impath in enumerate(IMAGE_PATH_LST):
        # Read frame
        frame = cv2.imread(impath)[:,:,::-1]
        # Warp via perspective_projector
        frame_warped = perspective_projector.warp(frame)
        #h, s, l = separate_hsl(cv2.blur(frame_warped, (1, 1)))
        # Separate color masks
        yellow_mask  = yellow_filter.process(frame_warped)
        white_mask   = white_filter.process(frame_warped)
        
        contours, evals_lst = segmentator.process(yellow_mask)
        obstacle_labels = obstacle_finder.process(evals_lst)
        ducks = []
        yellow_lane = []
        print(f'Image_{i}')
        for j, eval in enumerate(evals_lst):
            # Get centroid coordinates
            M = cv2.moments(contours[j])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            print(f'Box_{j}: center={(cx, cy)}, o_type={obstacle_labels[j][1]}')
            if obstacle_labels[j] == ObstacleType.duck:
                ducks.append(contours[j])
            elif obstacle_labels[j] == ObstacleType.yellow_lane:
                yellow_lane.append(contours[j])
        yellow_frame = cv2.bitwise_and(frame_warped, frame_warped, mask=yellow_mask)
        if len(ducks) > 0:
            cv2.drawContours(yellow_frame, ducks, -1, (255, 0, 0), 3)
        if len(yellow_lane) > 0:
            cv2.drawContours(yellow_frame, yellow_lane, -1, (0, 0, 255), 3)

        if im0 is None:
            im0 = axs[0].imshow(frame)
            im1 = axs[1].imshow(frame_warped)
            im2 = axs[2].imshow(yellow_frame)
        else:
            im0.set_data(frame)
            im1.set_data(frame_warped)
            im2.set_data(yellow_frame)
        plt.pause(5)
        plt.draw()
    
