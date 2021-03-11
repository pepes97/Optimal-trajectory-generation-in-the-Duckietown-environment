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

def ransac_polyfit(x, y, order=2, n=3, k=50, t=40, d=6, f=0.3, initial_estimate=None):
    """
    Thanks https://en.wikipedia.org/wiki/Random_sample_consensus
    n – minimum number of data points required to fit the model
    k – maximum number of iterations allowed in the algorithm
    t – threshold value to determine when a data point fits a model
    d – number of close data points required to assert that a model fits well to data
    f – fraction of close data points required
    """
    besterr = np.inf
    bestfit = initial_estimate
    for it in range(k):
        possible_inliers = np.random.randint(len(x), size=n)
        possible_model = np.polyfit(x[possible_inliers], y[possible_inliers], order)
        computed_inliers = np.abs(np.polyval(possible_model, x)-y) < t
        print(sum(computed_inliers), len(x) * f, d)
        if sum(computed_inliers) >= d and sum(computed_inliers) > len(x) * f:
            best_model = np.polyfit(x[computed_inliers], y[computed_inliers], order)
            current_err = np.sum(np.abs(np.polyval(best_model, x[computed_inliers])
                                        - y[computed_inliers]))
            if current_err < besterr:
                print('found bestfit')
                besterr = current_err
                bestfit = best_model
    return bestfit
