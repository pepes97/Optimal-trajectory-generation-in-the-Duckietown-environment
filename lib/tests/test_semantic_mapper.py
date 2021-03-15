"""test_semantic_mapper.py
"""

import numpy as np
import cv2
import logging
from matplotlib import pyplot as plt
from ..plotter import *
from ..video import *

logger = logging.getLogger(__name__)

IMAGE_PATH_LST = [f'./images/dt_samples/{i}.jpg' for i in range(0, 170)]

OBJ_COLOR_DICT = {
    ObjectType.UNKNOWN: (0, 128, 128),
    ObjectType.YELLOW_LINE: (255, 255, 0),
    ObjectType.WHITE_LINE:  (255, 0, 0),
    ObjectType.DUCK: (0, 255, 255),
    ObjectType.CONE: (255, 0, 0),
    ObjectType.ROBOT: (0, 0, 128),
    ObjectType.WALL: (255, 0, 255),
    ObjectType.RIGHT_LINE: (255, 255, 255),
    ObjectType.LEFT_LINE:  (0, 0, 255)
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
    p0 = p1 = None
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
                cv2.drawContours(pframe, object['contour'], -1, OBJ_COLOR_DICT[object['class']], 3)
        if pfit is not None:
            ploty = np.arange(0, wframe.shape[0]-1, 1)
            #pline_fit = pfit[0] * ploty**2 + pfit[1] * ploty + pfit[2]
            pline_fit = np.polyval(pfit, ploty)
        # Fill matplotlib container
        if im0 is None:
            im0 = axs[0].imshow(frame)
            im1 = axs[1].imshow(wframe)
            im2 = axs[2].imshow(pframe)
            p0,  = axs[2].plot(pline_fit, ploty, 'r', linewidth=2)
        else:
            im0.set_data(frame)
            im1.set_data(wframe)
            im2.set_data(pframe)
            p0.set_xdata(pline_fit)
        plt.pause(0.01)
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
    p00 = p0 = p1 = p2 = s1 = s2 = None
    axs[0].set_title('frame')
    axs[1].set_title('polyfit')
    axs[2].set_title('RANSAC fit')
    axs[1].set_xlim(0, 640)
    axs[2].set_xlim(0, 640)
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
        if len(object_dict[ObjectType.YELLOW_LINE]) == 0:
            continue
        yellow_midpts = np.zeros((2, len(object_dict[ObjectType.YELLOW_LINE])))
        for j, contour in enumerate(object_dict[ObjectType.YELLOW_LINE]):
            M = cv2.moments(contour)
            cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
            yellow_midpts[:, j] = np.array([cx, cy])
        
        # Use nonzero yellow pixels
        nonzero = mask_dict['yellow'].nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        fit_weights = yellow_midpts[1, :]
        

        pfit = np.polyfit(yellow_midpts[1, :], yellow_midpts[0, :], 2, w=fit_weights)
        rfit = ransac_polyfit(yellow_midpts[1, :], yellow_midpts[0, :], 2,
                              n=int(yellow_midpts.shape[1]/2), w=fit_weights,
                              initial_estimate=[0, -1, 0])

        ploty = np.arange(0, wframe.shape[0]-1, 1)
        pline_fit = pfit[0] * ploty**2 + pfit[1] * ploty + pfit[2]
        rline_fit = rfit[0] * ploty**2 + rfit[1] * ploty + rfit[2]

        # Unwarp the RANSAC fit
        rline_unwarped = np.vstack((rline_fit, ploty))
        rline_unwarped = np.expand_dims(rline_unwarped, axis=1)
        rline_unwarped = np.transpose(rline_unwarped, (2, 1, 0))
        try:
            rline_unwarped = cv2.perspectiveTransform(rline_unwarped, projector.iM)
        except Exception as e:
            #print(rline_unwarped)
            print(e)

        pline_unwarped = np.vstack((pline_fit, ploty))
        pline_unwarped = np.expand_dims(pline_unwarped, axis=1)
        pline_unwarped = np.transpose(pline_unwarped, (2, 1, 0))
        pline_unwarped = cv2.perspectiveTransform(pline_unwarped, projector.iM)

        if im0 is None:
            im0 = axs[0].imshow(frame)
            im1 = axs[1].imshow(pframe)
            im2 = axs[2].imshow(pframe)
            p00, = axs[0].plot(pline_unwarped[:, 0, 0], pline_unwarped[:, 0, 1], 'b', linewidth=3)
            p0,  = axs[0].plot(rline_unwarped[:, 0, 0], rline_unwarped[:, 0, 1], 'r', linewidth=3)
            p1,  = axs[1].plot(pline_fit, ploty, 'b', linewidth=2)
            p2,  = axs[2].plot(rline_fit, ploty, 'r', linewidth=2)
            s1   = axs[1].scatter(yellow_midpts[0, :], yellow_midpts[1, :])
            s2   = axs[2].scatter(yellow_midpts[0, :], yellow_midpts[1, :])
        else:
            im0.set_data(frame)
            im1.set_data(pframe)
            im2.set_data(pframe)
            p00.set_xdata(pline_unwarped[:, 0, 0])
            p00.set_ydata(pline_unwarped[:, 0, 1])
            p0.set_xdata(rline_unwarped[:, 0, 0])
            p0.set_ydata(rline_unwarped[:, 0, 1])
            p1.set_xdata(pline_fit)
            p2.set_xdata(rline_fit)
            s1.set_offsets(yellow_midpts.T)
            s2.set_offsets(yellow_midpts.T)

        plt.pause(.01)
        plt.draw()
        
