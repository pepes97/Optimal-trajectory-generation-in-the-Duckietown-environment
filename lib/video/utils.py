"""utils.py
"""

import cv2
import numpy as np

def separate_lab(img):
    """ Returns the lab channels given an input image
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l = lab[:,:,0]
    a = lab[:,:,1]
    b = lab[:,:,2]
    return l, a, b

def separate_luv(img):
    """ Returns the luv channels given an input image
    """
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l = luv[:,:,0]
    u = luv[:,:,1]
    v = luv[:,:,2]
    return l, u, v

def separate_hsl(img):
    """ Returns the hls channels given an input image
    """
    hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h = hsl[:,:,0]
    s = hsl[:,:,1]
    l = hsl[:,:,2]
    return h, s, l

def get_hist(image):
    hist = np.sum(image[image.shape[0]//2:,:], axis=0)
    return hist
