"""binarize.py
"""

import cv2
import numpy as np
from .utils import separate_lab, separate_luv

def binary_threshold_lab_luv(image_rgb, b_thresh, l_thresh):
    """ Returns a binarized version of image_rgb computed by thresholding the
    blue-yellow channel (b) of LAB representation and the l channel of LUV representation
    """
    _, _, b = separate_lab(image_rgb)
    l, _, _ = separate_luv(image_rgb)
    binary = np.zeros_like(l)
    binary[
        ((b > bthresh[0]) & (b <= bthresh[1])) |
        ((l > lthresh[0]) & (l <= lthresh[1]))
    ] = 1
    return binary

def binary_threshold_grad(image_channel, threshold):
    """ Apply binarization by thresholding the gradient obtained with the Sobel operator.
    Filter works only on a single channel
    """
    ...

def binary_threshold_gray(image_rgb, threshold):
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    thresh, binarize = cv2.threshold(image_gray, threshold[0], threshold[1], cv2.THRESH_BINARY)
    return binarize
