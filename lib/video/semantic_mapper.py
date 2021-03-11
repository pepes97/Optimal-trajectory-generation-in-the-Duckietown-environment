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

    def process(self, contours):
        output_dict = {ObjectType.WHITE_LINE: [],
                       ObjectType.WALL: [],
                       ObjectType.UNKNOWN: []}
        for i, contour in enumerate(contours):
            object_type = ObjectType.UNKNOWN
            M = cv2.moments(contour)
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

    def process(self, contours):
        output_dict = {ObjectType.YELLOW_LINE: [],
                       ObjectType.DUCK: [],
                       ObjectType.UNKNOWN: []}
        for i, contour in enumerate(contours):
            object_type = ObjectType.UNKNOWN
            M = cv2.moments(contour)
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

    def process(self, contours):
        output_dict = {ObjectType.CONE: [],
                       ObjectType.ROBOT: [],
                       ObjectType.UNKNOWN: []}
        for i, contour in enumerate(contours):
            M = cv2.moments(contour)
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

class SemanticMapper:
    """ The SemanticMapper takes as input a dictionary containing contours for each colored cluster
    and further returns a dictionary where each key represents a list of found objects in the frame
    """
    def __init__(self, *args, **kwargs):
        self.mappers = {'white' : SemanticMapperWhite(),
                        'yellow': SemanticMapperYellow(),
                        'red'   : SemanticMapperRed()}
        pass

    def process(self, segment_dict):
        # Process each color differently
        # First generate an output space for each mask
        mapper_dict = {}
        output_dict = {}
        for key in segment_dict.keys():
            mapper_dict.update(self.mappers[key].process(contours=segment_dict[key]))
        # Apply higher reasoning to separate left, right middle lanes and different obstacles.
        # Also remove outliers
        return mapper_dict
