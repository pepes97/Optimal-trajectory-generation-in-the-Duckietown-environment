"""semantic_mapper.py
"""

import cv2
import numpy as np
import logging
from enum import Enum
from .utils import *
from .binarize import *
from .obstacle_recognition import *

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

class SemanticMapperWhite:
    """ SemanticMapperWhite separates white lanes from walls
    """
    def __init__(self, *args, **kwargs):
        self.lane_threshold = (100., 500.)
        pass

    def process(self, contours, moments):
        output_dict = {ObjectType.WHITE_LINE: [],
                       ObjectType.WALL: [],
                       ObjectType.UNKNOWN: []}
        for i, contour in enumerate(contours):
            object_type = ObjectType.UNKNOWN
            M = moments[i]
            cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
            u11 = (M['m11'] - cx * M['m01']) / M['m00']
            u20 = (M['m20'] - cx * M['m10']) / M['m00']
            u02 = (M['m02'] - cy * M['m01']) / M['m00']
            cov = np.array([[u20, u11], [u11, u02]])
            eigvals, _ = np.linalg.eig(cov)
            sort_idx = np.argsort(eigvals)[::-1]
            eigratio = eigvals[sort_idx[0]] / eigvals[sort_idx[1]]
            if eigratio >= self.lane_threshold[0] and eigratio < self.lane_threshold[1]:
                object_type = ObjectType.WHITE_LINE
            elif cv2.contourArea(contour) > 9000.:
                object_type = ObjectType.WHITE_LINE
            #print(f'White_{i}: center={(cx, cy)}, type={object_type}, eigratio={eigratio}')
            output_dict[object_type].append(contour)
        return output_dict

class SemanticMapperYellow:
    def __init__(self, *args, **kwargs):
        self.lane_threshold = [3., 10.]
        pass

    def process(self, contours, moments):
        output_dict = {ObjectType.YELLOW_LINE: [],
                       ObjectType.DUCK: [],
                       ObjectType.UNKNOWN: []}
        for i, contour in enumerate(contours):
            object_type = ObjectType.UNKNOWN
            M = moments[i]
            cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
            u11 = (M['m11'] - cx * M['m01']) / M['m00']
            u20 = (M['m20'] - cx * M['m10']) / M['m00']
            u02 = (M['m02'] - cy * M['m01']) / M['m00']
            cov = np.array([[u20, u11], [u11, u02]])
            eigvals, _ = np.linalg.eig(cov)
            sort_idx = np.argsort(eigvals)[::-1]
            eigratio = eigvals[sort_idx[0]] / eigvals[sort_idx[1]]
            if eigratio >= self.lane_threshold[0] and eigratio < self.lane_threshold[1]:
                object_type = ObjectType.YELLOW_LINE
            elif cv2.contourArea(contour) <= 1800:
                object_type = ObjectType.YELLOW_LINE
            else:
                object_type = ObjectType.DUCK
            output_dict[object_type].append(contour)
            #print(f'Yellow_{i}: center={(cx, cy)}, eigratio={eigratio:.3f}')
        return output_dict

class SemanticMapperRed:
    def __init__(self, *args, **kwargs):
        pass

    def process(self, contours, moments):
        output_dict = {ObjectType.CONE: [],
                       ObjectType.ROBOT: [],
                       ObjectType.UNKNOWN: []}
        for i, contour in enumerate(contours):
            M = moments[i]
            cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
            u11 = (M['m11'] - cx * M['m01']) / M['m00']
            u20 = (M['m20'] - cx * M['m10']) / M['m00']
            u02 = (M['m02'] - cy * M['m01']) / M['m00']
            cov = np.array([[u20, u11], [u11, u02]])
            eigvals, _ = np.linalg.eig(cov)
            sort_idx = np.argsort(eigvals)[::-1]
            eigratio = eigvals[sort_idx[0]] / eigvals[sort_idx[1]]
            print(f'Red_{i}: center={(cx, cy)}, eigratio={eigratio:.3f}')
        return output_dict

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
        self.mappers = {'white' : SemanticMapperWhite(),
                        'yellow': SemanticMapperYellow(),
                        'red'   : SemanticMapperRed()}
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
                       ObjectType.WALL: []
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
        if yellow_fit is not None:
            # Project distances on fit frame of reference
            fit_t = np.arctan2()
            ...
        return object_dict, yellow_fit, feat_dict
