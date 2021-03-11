"""test_semantic_mapper.py
"""

import numpy as np
import cv2
import logging
from matplotlib import pyplot as plt
from ..plotter import *
from ..video import *

logger = logging.getLogger(__name__)

IMAGE_PATH_LST = [f'./images/dt_samples/{i}.jpg' for i in range(0, 150)]

OBJ_COLOR_DICT = {
    ObjectType.UNKNOWN: (0, 128, 128),
    ObjectType.YELLOW_LINE: (255, 255, 0),
    ObjectType.WHITE_LINE:  (255, 255, 255),
    ObjectType.DUCK: (0, 255, 255),
    ObjectType.CONE: (255, 0, 0),
    ObjectType.ROBOT: (0, 0, 128),
    ObjectType.WALL: (255, 0, 255)
}

def test_semantic_mapper(*args, **kwargs):
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
    # Adjust filters properties
    yellow_filter.yellow_thresh = (20, 35)
    yellow_filter.s_thresh = (65, 190)
    yellow_filter.l_thresh = (30, 255)
    # Prepare the matplotlib container
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    im0 = im1 = im2 = None
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
        print(type(segment_dict))
        # Generate semantic dictionary
        print(f'Frame_{i}')
        object_dict  = semantic_mapper.process(segment_dict)
        #object_dict = {}
        # Print object_dict
        for okey in object_dict.keys():
            print(f'{okey}: {len(object_dict[okey])}')
            if okey is ObjectType.WHITE_LINE:
                for ctr in object_dict[okey]:
                    print(f'area: {cv2.contourArea(ctr)}')
            # Write bounding boxes on output image
            cv2.drawContours(pframe, object_dict[okey], -1, OBJ_COLOR_DICT[okey], 3)
        # Fill matplotlib container
        if im0 is None:
            im0 = axs[0].imshow(frame)
            im1 = axs[1].imshow(wframe)
            im2 = axs[2].imshow(pframe)
        else:
            im0.set_data(frame)
            im1.set_data(wframe)
            im2.set_data(pframe)
        plt.pause(0.5)
        plt.draw()


def test_ransac(*args, **kwargs):
    IM_PATH = ['./images/dt_samples/8.jpg', './images/dt_samples/22.jpg']
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
    # Adjust filters properties
    yellow_filter.yellow_thresh = (20, 35)
    yellow_filter.s_thresh = (65, 190)
    yellow_filter.l_thresh = (30, 255)
    # Prepare the matplotlib container
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    im0 = im1 = im2 = None
    p1 = p2 = s1 = s2 = None
    axs[0].set_title('frame')
    axs[1].set_title('polyfit')
    axs[2].set_title('RANSAC fit')
    # Load the images
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
        object_dict  = semantic_mapper.process(segment_dict)
        for okey in object_dict.keys():
            #print(f'{okey}: {len(object_dict[okey])}')
            # Write bounding boxes on output image
            cv2.drawContours(pframe, object_dict[okey], -1, OBJ_COLOR_DICT[okey], 3)

        # Test both polyfit and RANSAC regressors
        yellow_midpts = np.zeros((2, len(object_dict[ObjectType.YELLOW_LINE])))
        for j, contour in enumerate(object_dict[ObjectType.YELLOW_LINE]):
            M = cv2.moments(contour)
            cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
            yellow_midpts[:, j] = np.array([cx, cy])
        


        # Use nonzero yellow pixels
        nonzero = mask_dict['yellow'].nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        pfit = np.polyfit(yellow_midpts[1, :], yellow_midpts[0, :], 2)
        rfit = ransac_polyfit(yellow_midpts[1, :], yellow_midpts[0, :], 2,
                              initial_estimate=[0, -1, 0])

        ploty = np.arange(0, wframe.shape[0]-1, 1)
        pline_fit = pfit[0] * ploty**2 + pfit[1] * ploty + pfit[2]
        rline_fit = rfit[0] * ploty**2 + rfit[1] * ploty + rfit[2]

        if im0 is None:
            im0 = axs[0].imshow(frame)
            im1 = axs[1].imshow(pframe)
            im2 = axs[2].imshow(pframe)
            p1,  = axs[1].plot(pline_fit, ploty, 'r', linewidth=2)
            p2,  = axs[2].plot(rline_fit, ploty, 'r', linewidth=2)
            s1   = axs[1].scatter(yellow_midpts[0, :], yellow_midpts[1, :])
            s2   = axs[2].scatter(yellow_midpts[0, :], yellow_midpts[1, :])
        else:
            im0.set_data(frame)
            im1.set_data(pframe)
            im2.set_data(pframe)
            p1.set_xdata(pline_fit)
            p2.set_xdata(rline_fit)
            s1.set_offsets(yellow_midpts.T)
            s2.set_offsets(yellow_midpts.T)
            
        """
        axs[0].imshow(frame)
        
        axs[1].imshow(pframe)
        axs[1].plot(pline_fit, ploty, 'r', linewidth=2)
        axs[1].scatter(yellow_midpts[0, :], yellow_midpts[1, :])
        axs[1].set_xlim(0, 640)
        
        axs[2].imshow(pframe)
        axs[2].plot(rline_fit, ploty, 'r', linewidth=2)
        axs[2].scatter(yellow_midpts[0, :], yellow_midpts[1, :])
        axs[2].set_xlim(0, 640)"""
        plt.pause(.1)
        plt.draw()
        
