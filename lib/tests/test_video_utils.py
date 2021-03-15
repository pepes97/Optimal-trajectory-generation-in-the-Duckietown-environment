"""test_video_utils.py
"""

import cv2
import numpy as np
import sys
import glob, os


def test_jpg2mp4(*args, **kwargs):
    image_lst = []
    size = None
    for i in range(1200):
        file = f'./test_images/{i}.jpg'
        image = cv2.imread(file)
        h, w, l = image.shape
        size = (w, h)
        image_lst.append(image)
    out = cv2.VideoWriter('sample01.avi', 0, 30, size)
    for i in range(len(image_lst)):
        out.write(image_lst[i])
    out.release()


if __name__ == '__main__':
    test_jpg2mp4()
