"""test_obstacle_tracker.py
"""

import numpy as np
import cv2
import logging
from matplotlib import pyplot as plt
from ..plotter import *
from ..video import *
from .test_semantic_mapper import OBJ_COLOR_DICT

IMAGE_PATH_LST = [f'./lib/tests/test_images/{str(i).zfill(4)}.jpg' for i in range(15, 600)]
#IMAGE_PATH_LST = [f'./images/dt_samples/{i}.jpg' for i in range(0, 150)]

logger = logging.getLogger(__name__)

def test_obstacle_tracker(*args, **kwargs):
    # Define cv pipeline objects
    projector       = PerspectiveWarper()
    yellow_filter   = CenterLineFilter()
    white_filter    = LateralLineFilter()
    red_filter      = RedFilter()
    filter_dict     = {'white': white_filter, 'yellow': yellow_filter, 'red': red_filter}
    mask_dict       = {'white': None, 'yellow': None, 'red': None}
    segmentator     = Segmentator()
    segment_dict    = {'white': None, 'yellow': None, 'red': None}
    semantic_mapper = SemanticMapper()
    obstacle_tracker = ObstacleTracker()
    # Adjust filters properties
    yellow_filter.yellow_thresh = (20, 35)
    yellow_filter.s_thresh = (65, 190)
    yellow_filter.l_thresh = (30, 255)
    # Prepare the matplotlib container
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    im0 = im1 = im2 = None
    s0 = s1 = None
    for i, impath in enumerate(IMAGE_PATH_LST):
        frame = cv2.imread(impath)[:,:,::-1]
        wframe = projector.warp(frame)
        pframe = np.zeros((wframe.shape[0], wframe.shape[1], 3), dtype='uint8')
        # Compute mask dictionary
        for fkey in filter_dict.keys():
            mask_dict[fkey] = filter_dict[fkey].process(wframe)
        mask_t = cv2.bitwise_or(mask_dict['white'], mask_dict['yellow'])
        # Segmentize the masks
        segment_dict = segmentator.process(mask_dict)
        # Generate semantic dictionary
        print(f'Frame_{i}')
        object_dict, pfit, feat_dict  = semantic_mapper.process(segment_dict)
        for obj_lst in object_dict.values():
            for object in obj_lst:
                cv2.drawContours(wframe, object['contour'], -1, OBJ_COLOR_DICT[object['class']], 3)
        # Apply obstacle tracking
        obstacles, all_obstacles = obstacle_tracker.process(object_dict)
        
        for object in obstacles:
            print(object['contour'].shape)
            cv2.drawContours(pframe, object['contour'], -1, OBJ_COLOR_DICT[object['class']], 3)
        # Fill matplotlib container
        if im0 is None:
            im0 = axs[0].imshow(frame)
            im1 = axs[1].imshow(wframe)
            im2 = axs[2].imshow(pframe)
            s0 = axs[2].scatter([], [], c='r')
            s1 = axs[2].scatter([], [], c='g')
        else:
            im0.set_data(frame)
            im1.set_data(wframe)
            im2.set_data(pframe)
            all_obst_cntr = np.zeros((len(all_obstacles), 2))
            obst_cntr = np.zeros((len(obstacles), 2))
            for i, okey in enumerate(all_obstacles.keys()):
                all_obst_cntr[i, :] = all_obstacles[okey]['center']
            for i, obst in enumerate(obstacles):
                obst_cntr[i, :] = obst['center']
            s0.set_offsets(all_obst_cntr)
            s1.set_offsets(obst_cntr)
        plt.pause(.05)
        plt.draw()
