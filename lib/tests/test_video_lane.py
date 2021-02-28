"""test_video_lane.py
"""

import numpy as np
import cv2
import logging
from matplotlib import pyplot as plt
from ..plotter import *
from ..video import *

logger = logging.getLogger(__name__)

IMAGE_PATH_LST = ['./images/dt_samples/01.png',
                  './images/dt_samples/02.png',
                  './images/dt_samples/03.png',
                  './images/dt_samples/04.png',
                  './images/dt_samples/05.png']


def test_video_lane(*args, **kwargs):
    lane_filter = LaneFilter()
    for impath in IMAGE_PATH_LST:
        frame = cv2.imread(impath)
        lane_filter.process(frame)
