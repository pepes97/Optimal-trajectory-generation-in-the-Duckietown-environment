"""segmentation.py
"""

import cv2
import numpy as np
import logging
from .utils import *
from .binarize import *
"""
def raw_moment(data, i_order, j_order):
    nrows, ncols = data.shape
    y_indices, x_indices = np.mgrid[:nrows, :ncols]
    return (data * x_indices**i_order * y_indices**j_order).sum()

def moments_cov(data):
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_centroid = m10 / data_sum
    y_centroid = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return cov
"""

class SegmentatorYellow:
    def __init__(self, min_area=420):
        self.min_area = min_area

    def process(self, frame):
        eval_lst = []
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(filter(lambda x: cv2.contourArea(x) > self.min_area, contours))
        return contours#, eval_lst

class SegmentatorChannel:
    def __init__(self, min_area=100):
        self.min_area = min_area

    def process(self, frame):
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = list(filter(lambda x: cv2.contourArea(x) > self.min_area, contours))
        return contours

class Segmentator:
    """ Main Segmentator class.
    Given a dictionary of masks where each key represents the mask color, it returns a dictionary
    containing the contours of each cluster found for each mask.
    This class is intended to be used along the SemanticMapper.
    """
    def __init__(self, *args, **kwargs):
        self.segmentator_dict = {'white':  SegmentatorChannel(min_area=500),
                                 'yellow': SegmentatorChannel(min_area=200),
                                 'red':    SegmentatorChannel()}

    def process(self, mask_dict):
        output_dict = {'white': None, 'yellow': None, 'red': None}
        # Call segmentators for each type of mask
        for k in output_dict.keys():
            segmentator = self.segmentator_dict[k]
            output_dict[k] = segmentator.process(mask_dict[k])
        return output_dict
