"""semantic_mapper.py
"""

import cv2
import numpy as np
import logging
from enum import Enum
from .utils import *
from .binarize import *
from .obstacle_recognition import *
from ..transform import *

class ObjectType(Enum):
    UNKNOWN     = -1
    YELLOW_LINE = 0
    WHITE_LINE  = 1
    DUCK        = 2
    CONE        = 3
    ROBOT       = 4
    WALL        = 5
    LEFT_LINE   = 6
    RIGHT_LINE  = 7

class FeatureExtractor:
    def __init__(self):
        pass

    def process(self, contours, *args, **kwargs):
        output_lst = []
        for i, contour in enumerate(contours):
            M = cv2.moments(contour)
            cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
            u11 = (M['m11'] - cx * M['m01']) / M['m00']
            u20 = (M['m20'] - cx * M['m10']) / M['m00']
            u02 = (M['m02'] - cy * M['m01']) / M['m00']
            cov = np.array([[u20, u11], [u11, u02]])
            eigvals, _ = np.linalg.eig(cov)
            sort_idx = np.argsort(eigvals)[::-1]
            
            cdict = {}
            cdict['contour'] = contour
            cdict['center'] = np.float32([cx, cy])
            cdict['area'] = cv2.contourArea(contour)
            cdict['eigs'] = np.float32([eigvals[sort_idx[0]], eigvals[sort_idx[1]]])
            cdict['cov']  = cov
            output_lst.append(cdict)
        return output_lst

class SemanticMapper:
    """ The SemanticMapper takes as input a dictionary containing contours for each colored cluster
    and further returns a dictionary where each key represents a list of found objects in the frame.
    Also returns the best fit for yellow lane.
    """
    def __init__(self, *args, **kwargs):
        self.feat_ext = FeatureExtractor()
        pass

    def process(self, segment_dict):
        # Extract features for each color
        feat_dict = {'white': None, 'yellow': None, 'red': None}
        object_dict = {ObjectType.UNKNOWN: [],
                       ObjectType.YELLOW_LINE: [],
                       ObjectType.WHITE_LINE: [],
                       ObjectType.DUCK: [],
                       ObjectType.CONE: [],
                       ObjectType.ROBOT: [],
                       ObjectType.WALL: [],
                       ObjectType.RIGHT_LINE: [],
                       ObjectType.LEFT_LINE: []
        }
        for k in feat_dict.keys():
            feat_dict[k] = self.feat_ext.process(segment_dict[k])
        # Handle yellow elements
        for i, fdict in enumerate(feat_dict['yellow']):
            area = fdict['area']
            eigratio = fdict['eigs'][0] / fdict['eigs'][1]
            if area >= 1500:
                fdict['class'] = ObjectType.DUCK
            elif eigratio > 23. or area < 400 or eigratio < 2.:
                fdict['class'] = ObjectType.UNKNOWN
            else:
                fdict['class'] = ObjectType.YELLOW_LINE
            object_dict[fdict['class']].append(fdict)
            #center = fdict['center']
            #print(f'YELLOW_{i}: type={fdict["class"]}, area={area}, eigratio={eigratio:.3f}, center={center}')
        # Handle white elements
        for i, fdict in enumerate(feat_dict['white']):
            eigratio = fdict['eigs'][0] / fdict['eigs'][1]
            if eigratio >= 100. and eigratio <= 500. or fdict['area'] > 9000.:
                fdict['class'] = ObjectType.WHITE_LINE
            else:
                fdict['class'] = ObjectType.UNKNOWN
            object_dict[fdict['class']].append(fdict)
        # Handle red elements
        # TODO
        # Fit yellow line if possible
        if len(object_dict[ObjectType.YELLOW_LINE]) < 2:
            yellow_fit = None
        else:
            yellow_midpts = np.zeros((2, len(object_dict[ObjectType.YELLOW_LINE])))
            for i, yp in enumerate(object_dict[ObjectType.YELLOW_LINE]):
                yellow_midpts[:, i] = yp['center']
                # Weight based on y-value of points (Lower points are heavier)
                yellow_fit = np.polyfit(yellow_midpts[1, :], yellow_midpts[0, :], 1)
        
        # If fit is possible, get closest right and left lane
        if yellow_fit is not None and len(object_dict[ObjectType.WHITE_LINE]) > 0:
            # Project distances on fit frame of reference
            fit_t = -np.arctan(yellow_fit[0])
            # Take central midpoint as reference
            fit_p = np.array([np.polyval(yellow_fit, 240), 240.])
            R, p = homogeneous_itransform(np.append(fit_p, fit_t))
            # Compute the relative offset from the midpoint and all the white lines
            white_midpts_dist = np.array(
                [np.matmul(R, o['center']) + p for o in object_dict[ObjectType.WHITE_LINE]])
            # Handle singular cases (single point)
            if white_midpts_dist.shape[0] == 1:
                object = object_dict[ObjectType.WHITE_LINE][0]
                if white_midpts_dist[0, 0] > 0.:
                    object['class'] = ObjectType.RIGHT_LINE
                else:
                    object['class'] = ObjectType.LEFT_LINE
                object_dict[object['class']].append(object)
            # Handle single side elements (all blocks on right or all blocks on left)
            elif (white_midpts_dist[:, 0]>0.).all():
                # Can find only right line
                right_idx = np.where(white_midpts_dist[:, 0] > 0.,
                                     white_midpts_dist[:, 0], np.inf).argmin()
                right_obj = object_dict[ObjectType.WHITE_LINE][right_idx]
                right_obj['class'] = ObjectType.RIGHT_LINE
                object_dict[ObjectType.RIGHT_LINE].append(right_obj)
                # All other objects become walls
                remaining_idx = np.indices((white_midpts_dist.shape[0],))
                remaining_idx = remaining_idx[(remaining_idx != right_idx)]
                for idx in remaining_idx:
                    obj = object_dict[ObjectType.WHITE_LINE][idx]
                    obj['class'] = ObjectType.WALL
                    object_dict[ObjectType.WALL].append(obj)
                    
            elif (white_midpts_dist[:, 0]<0.).all():
                # Can find only left line
                left_idx = np.where(white_midpts_dist[:, 0] < 0.,
                                     white_midpts_dist[:, 0], -np.inf).argmax()
                left_obj = object_dict[ObjectType.WHITE_LINE][left_idx]
                left_obj['class'] = ObjectType.LEFT_LINE
                object_dict[ObjectType.LEFT_LINE].append(left_obj)
                # All other objects become walls
                remaining_idx = np.indices((white_midpts_dist.shape[0],))
                remaining_idx = remaining_idx[(remaining_idx != left_idx)]
                for idx in remaining_idx:
                    obj = object_dict[ObjectType.WHITE_LINE][idx]
                    obj['class'] = ObjectType.WALL
                    object_dict[ObjectType.WALL].append(obj)
            else:
                # Right index is the one whose x coordinates is positive and smallest
                right_idx = np.where(white_midpts_dist[:, 0] > 0., white_midpts_dist[:, 0], np.inf).argmin()
                # Left index is the one wshose x coordinates is negative and highest
                left_idx = np.where(white_midpts_dist[:, 0] < 0., white_midpts_dist[:, 0], -np.inf).argmax()
                right_obj = object_dict[ObjectType.WHITE_LINE][right_idx]
                right_obj['class'] = ObjectType.RIGHT_LINE
                left_obj = object_dict[ObjectType.WHITE_LINE][left_idx]
                left_obj['class'] = ObjectType.LEFT_LINE
                object_dict[ObjectType.RIGHT_LINE].append(right_obj)
                object_dict[ObjectType.LEFT_LINE].append(left_obj)
                # All remaining white objects are should be classified as walls
                remaining_idx = np.indices((white_midpts_dist.shape[0],))
                remaining_idx = remaining_idx[(remaining_idx != right_idx) & (remaining_idx != left_idx)]
                for idx in remaining_idx:
                    obj = object_dict[ObjectType.WHITE_LINE][idx]
                    obj['class'] = ObjectType.WALL
                    object_dict[ObjectType.WALL].append(obj)
        # If fit is not available, try to use previous last fit to approximate which line the camera is looking at
        
        return object_dict, yellow_fit, feat_dict