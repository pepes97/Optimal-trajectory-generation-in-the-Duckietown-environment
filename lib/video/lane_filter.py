"""line_filter.py
"""

import cv2
import numpy as np
import logging
from matplotlib import pyplot as plt
import functools
from .utils import *
from .binarize import *

class PerspectiveWarper:
    def __init__(self, dest_size=(640, 480),
                 src=np.float32([
                     (0.3796, 0.4438),
                     (0, 0.9396),
                     (1, 0.9396),
                     (0.6281, 0.4438)]),
                 dest=np.float32([(0.3, 0), (0.3, 1), (0.7, 1), (0.7, 0)])):
        self.dest_size = dest_size
        dest_size = np.float32(dest_size)
        self.src = src * dest_size
        self.dest = dest * dest_size
        self.M = cv2.getPerspectiveTransform(self.src, self.dest)
        self.iM = cv2.getPerspectiveTransform(self.dest, self.src)

    def warp(self, frame, draw=False):
        warped_frame = cv2.warpPerspective(frame, self.M, self.dest_size)
        if draw:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(frame)
            axs[0].plot(self.src[:, 0], self.src[:, 1], 'r')
            axs[1].imshow(warped_frame)
            axs[1].plot(self.dest[:, 0], self.dest[:, 1], 'r')
            plt.show()

        return warped_frame

    def iwarp(self, frame):
        return cv2.warpPerspective(frame, self.iM, self.dest_size)

class CenterLineFilter:
    def __init__(self):
        # Yellow of line is between 14 and 22 in h channel
        self.yellow_thresh = (19, 24)
        self.s_thresh = (80, 150)
        pass
    def process(self, frame):
        def preproc(image):
            # Extract yellow info
            h, s, l = separate_hsl(cv2.blur(frame, (3, 3)))
            h_mask = cv2.inRange(h, self.yellow_thresh[0], self.yellow_thresh[1])
            s_mask = cv2.inRange(s, self.s_thresh[0], self.s_thresh[1])
            #preproc_frame = cv2.bitwise_and(frame, frame, mask=h_mask)
            mask = cv2.bitwise_and(h_mask, h_mask, mask=s_mask)
            # Apply morphological operation to remove imperfections
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (5, 5))

            return mask

        return preproc(frame)

class LateralLineFilter:
    def __init__(self):
        pass
    def process(self, frame):
        def preproc(image):
            # Extract yellow info
            r,g,b = np.split(cv2.blur(frame, (3, 3)),3,axis=-1)
            mask_r = cv2.inRange(r, 150, 255)
            mask_g = cv2.inRange(g, 128, 255)
            mask_b = cv2.inRange(b, 150, 255)
            mask = cv2.bitwise_and(cv2.bitwise_and(mask_r, mask_g), mask_b)
            # Apply morphological operation to remove imperfections
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5, 5))
            mask = cv2.dilate(mask,(3, 3), iterations = 10)
            return mask

        return preproc(frame)


class GrassFilter:
    def __init__(self):
        # Yellow of line is between 14 and 22 in h channel
        self.green_thresh = (40, 50)
        self.s_thresh = (250, 255)
        pass
    def process(self, frame):
        def preproc(image):
            # Extract yellow info
            h, s, l = separate_hsl(cv2.blur(frame, (3, 3)))
            h_mask = cv2.inRange(h, self.green_thresh[0], self.green_thresh[1])
            s_mask = cv2.inRange(s, self.s_thresh[0], self.s_thresh[1])
            #preproc_frame = cv2.bitwise_and(frame, frame, mask=h_mask)
            mask = cv2.bitwise_and(h_mask, h_mask, s_mask)
            # Apply morphological operation to remove imperfections
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5, 5))
            mask = cv2.dilate(mask,(50, 50), iterations = 100)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (5, 5))
            # mask = cv2.bilateralFilter(mask,15,75,75)
            mask = np.round(np.clip(mask,0,1))*255
            return mask

        return preproc(frame)

class SlidingWindowTracker:
    def __init__(self, robust_factor: int=1):
        self.coeff_a = []
        self.coeff_b = []
        self.coeff_c = []
        assert robust_factor >= 1
        self.robust_factor = robust_factor
        # Minimum mask pixels to initiate line search
        self.minimum_pixels = 1000
        self.contour_min_area = 200

    def search(self, image, no_windows: int=9, margin: int=50, min_pix: int=1, draw_windows=False):
        test_image = np.dstack((image, image, image))
        fit_params = np.empty(3)
        # Compute histogram
        histogram = get_hist(image)
        line_base = np.argmax(histogram)
        # Set window height
        window_height = np.int(image.shape[0] / no_windows)
        current_base = line_base
        # Get non zero pixels in the image
        if np.count_nonzero(image) < self.minimum_pixels:
            return np.zeros(3), test_image, np.zeros((2, 2))
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Find Contours approach
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(filter(lambda ctr: cv2.contourArea(ctr) > self.contour_min_area, contours))
        # Get midpoints for each contour
        contours_midpt = []
        for ctr in contours:
            ctr_moment = cv2.moments(ctr)
            px = int(ctr_moment['m10'] / ctr_moment['m00'])
            py = int(ctr_moment['m01'] / ctr_moment['m00'])
            contours_midpt.append([px, py])
        contours_midpt = np.array(contours_midpt)
        if draw_windows:
            cv2.drawContours(test_image, contours, -1, (0, 255, 0), 3)
            for i in range(len(contours)):
                cv2.circle(test_image, (contours_midpt[i, 0], contours_midpt[i, 1]), 2, (255, 0, 0), -1)
        """
        # REPLACE WITH FindingContours
        # lane pixels list
        lane_pixels_inds = []
        for window in range(no_windows):
            # Generate window boundaries
            window_y_low = image.shape[0] - (window+1) * window_height
            window_y_high = image.shape[0] - window * window_height
            window_x_low = current_base - margin
            window_x_high = current_base + margin
            if draw_windows:
                cv2.rectangle(test_image, (window_x_low, window_y_low), (window_x_high, window_y_high), 100, 3)
            # Get pixels in window
            good_pixels = ((nonzeroy >= window_y_low) &
                           (nonzeroy < window_y_high) &
                           (nonzerox >= window_x_low) &
                           (nonzerox < window_x_high)).nonzero()[0]
            lane_pixels_inds.append(good_pixels)
            if len(good_pixels) > min_pix:
                current_base = np.int(np.mean(nonzerox[good_pixels]))
        # Get all pixels indices in the lane
        lane_pixels_inds = np.concatenate(lane_pixels_inds)
        """
        lane_x = contours_midpt[:, 0]#nonzerox#[lane_pixels_inds]
        lane_y = contours_midpt[:, 1]#nonzeroy#[lane_pixels_inds]
        if lane_x.size == 0 or lane_y.size == 0:
            return test_image, np.zeros(3)
        lane_fit = np.polyfit(lane_y, lane_x, 2)
        self.coeff_a.append(lane_fit[0])
        self.coeff_b.append(lane_fit[1])
        self.coeff_c.append(lane_fit[2])
        fit_params[0] = np.mean(self.coeff_a[-self.robust_factor:])
        fit_params[1] = np.mean(self.coeff_b[-self.robust_factor:])
        fit_params[2] = np.mean(self.coeff_c[-self.robust_factor:])
        return lane_fit, test_image, contours_midpt


def linspace():
    xx = np.arange(0,640)
    xx = np.tile(xx[None],(480,1))[...,None]
    yy = np.arange(0,480)
    yy = np.tile(yy[...,None],(1,640))[...,None]
    linspace = np.concatenate([xx,yy],axis=-1)
    return linspace

class SlidingWindowDoubleTracker:
    def __init__(self, robust_factor: int=1):
        self.coeff_a = []
        self.coeff_b = []
        self.coeff_c = []
        assert robust_factor >= 1
        self.robust_factor = robust_factor
        # Minimum mask pixels to initiate line search
        self.minimum_pixels = 1000
        self.contour_min_area = 200
        self.last_seen_yellow_pos = [np.array([320,240])]

    def search(self, image_y, image_w, no_windows: int=9, margin: int=50, min_pix: int=1, draw_windows=False):
        test_image_y = np.dstack((image_y, image_y, image_y))
        test_image_w = np.dstack((image_w, image_w, image_w))
        # Get non zero pixels in the image
        if np.count_nonzero(image_y) < self.minimum_pixels and np.count_nonzero(image_w) < self.minimum_pixels:
            lane_fit = np.zeros(3)
            lane_fit[0] = np.mean(self.coeff_a[-self.robust_factor:])
            lane_fit[1] = np.mean(self.coeff_b[-self.robust_factor:])
            lane_fit[2] = np.mean(self.coeff_c[-self.robust_factor:])
        elif np.count_nonzero(image_y) < self.minimum_pixels:
            last_seen_yellow_pos = np.mean(self.last_seen_yellow_pos[-self.robust_factor:],axis=0) 
            # Find Contours white  
            contours_w, _ = cv2.findContours(image_w, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_w = list(filter(lambda ctr: cv2.contourArea(ctr) > self.contour_min_area, contours_w))
            # Get midpoints for each contour
            right = []
            left = []
            contour_right = []
            contour_left = []
            for ctr in contours_w:
                black_image = np.zeros_like(test_image_w)
                cv2.drawContours(black_image, ctr, -1, (255, 255, 255), thickness=cv2.FILLED)
                xy = linspace()
                xy = xy[black_image[...,1]>0]
                if np.mean(xy, axis=0)[0]>last_seen_yellow_pos[0]:
                    contour_right = ctr
                    right = xy
                if np.mean(xy, axis=0)[0]<last_seen_yellow_pos[0]:
                    contour_left = ctr
                    left = xy
            if len(right)==0 and len(left)==0:
                lane_fit = np.zeros(3)
                lane_fit[0] = np.mean(self.coeff_a[-self.robust_factor:])
                lane_fit[1] = np.mean(self.coeff_b[-self.robust_factor:])
                lane_fit[2] = np.mean(self.coeff_c[-self.robust_factor:])
            else:
                if len(right)>0:
                    points = right
                    contour = contour_right
                    offset = - 320
                else:
                    points = left
                    contour = contour_left
                    offset = 320
                if draw_windows:
                    cv2.drawContours(test_image_w, contour, -1, (0, 255, 0), 3)
                for i in range(len(contour)):
                    mdpt = np.array(np.mean(contour,axis=(0,1)),np.int32)
                    cv2.circle(test_image_w, (mdpt[0], mdpt[1]), 2, (255, 0, 0), -1)
                lane_x = points[:, 0]
                lane_y = points[:, 1]
                lane_fit = np.polyfit(lane_y, lane_x, 2)
                self.coeff_a.append(lane_fit[0])
                self.coeff_b.append(lane_fit[1])
                self.coeff_c.append(lane_fit[2] + offset) # don't apply offset
                lane_fit[0] = np.mean(self.coeff_a[-self.robust_factor:])
                lane_fit[1] = np.mean(self.coeff_b[-self.robust_factor:])
                lane_fit[2] = np.mean(self.coeff_c[-self.robust_factor:])
        else:
            # Find Contours yellow
            contours_y, _ = cv2.findContours(image_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_y = list(filter(lambda ctr: cv2.contourArea(ctr) > self.contour_min_area, contours_y))
            contours_midpt = []
            for ctr in contours_y:
                ctr_moment = cv2.moments(ctr)
                px = int(ctr_moment['m10'] / ctr_moment['m00'])
                py = int(ctr_moment['m01'] / ctr_moment['m00'])
                contours_midpt.append([px, py])
            contours_midpt = np.array(contours_midpt)
            if draw_windows:
                cv2.drawContours(test_image_y, contours_y, -1, (0, 255, 0), 3)
                for i in range(len(contours_y)):
                    cv2.circle(test_image_y, (contours_midpt[i, 0], contours_midpt[i, 1]), 2, (255, 0, 0), -1)
            if len(contours_midpt)==0:
                lane_fit = np.zeros(3)
                lane_fit[0] = np.mean(self.coeff_a[-self.robust_factor:])
                lane_fit[1] = np.mean(self.coeff_b[-self.robust_factor:])
                lane_fit[2] = np.mean(self.coeff_c[-self.robust_factor:])
            else:
                self.last_seen_yellow_pos.append(np.mean(contours_midpt,axis=0))
                lane_x = contours_midpt[:, 0]
                lane_y = contours_midpt[:, 1]
                lane_fit = np.polyfit(lane_y, lane_x, 2)
                self.coeff_a.append(lane_fit[0])
                self.coeff_b.append(lane_fit[1])
                self.coeff_c.append(lane_fit[2])
                lane_fit[0] = np.mean(self.coeff_a[-self.robust_factor:])
                lane_fit[1] = np.mean(self.coeff_b[-self.robust_factor:])
                lane_fit[2] = np.mean(self.coeff_c[-self.robust_factor:])
        test_image = test_image_y+test_image_w     
        return lane_fit, test_image, np.mean(self.last_seen_yellow_pos[-self.robust_factor:],axis=0) 
    
# class SlidingWindowDoubleTracker:
#     # TODO
#     def __init__(self):
#         pass

class MiddleLineFilter():
    """ Finds and tracks the middle dashed yellow line.
    If the line is found and verified, then it returns the best quadratic fit (in lane space), the
    camera offset (d) and inclination (theta~)
    """
    def __init__(self, projector, filter, tracker):
        self.projector = projector
        self.filter = filter
        self.tracker = tracker
        self.line_found = False
        self.plot_image = None

    def process(self, frame) -> (bool, np.array):
        d = 0.0
        t = 0.0
        # Generate warped frame
        warped_frame = self.projector.warp(frame)
        # Threshold warped frame to find yellow mid line
        thresh_frame = self.filter.process(warped_frame)
        # Try to fit a quadratic curve to the mid line
        line_fit, self.plot_image, contours_midpt = self.tracker.search(thresh_frame, draw_windows=True)
        if (line_fit != np.zeros(3)).all():
            # Line is found
            self.line_found = True
            # Compute d and t
            # Get the furthest midpoints (the one with minimum (x,y) and the one with max (x, y))
            def get_maxmin_idx(data_vect):
                max_idx = np.argmax(data_vect[:,1])
                min_idx = np.argmin(data_vect[:,1])
                return max_idx, min_idx
            max_idx, min_idx = get_maxmin_idx(contours_midpt)
            try:
                max_pt, min_pt = contours_midpt[max_idx], contours_midpt[min_idx]
            except Exception:
                max_pt = min_pt = np.zeros(2)
            # Draw on plot_image
            cv2.line(self.plot_image, (max_pt[0], max_pt[1]), (320, max_pt[1]), (255, 0, 0), 5)
            d = 320 - max_pt[0]
            diff_vect = max_pt - min_pt
            cv2.arrowedLine(self.plot_image, (max_pt[0], max_pt[1]), (min_pt[0], min_pt[1]), (0, 0, 255), 5)
            t = np.arctan2(diff_vect[0], diff_vect[1])
        else:
            self.line_found = False
        return self.line_found, np.float32([d, t])



class TrajectoryFilter():
    """ Finds and tracks the middle dashed yellow line.
    If the line is found and verified, then it returns the best quadratic fit (in lane space), the
    camera offset (d) and inclination (theta~)
    """
    def __init__(self, projector, filter_y, filter_w, tracker):
        self.projector = projector
        self.filter_y = filter_y
        self.filter_w = filter_w
        self.tracker = tracker
        self.line_found = False
        self.plot_image = None

    def process(self, frame) -> (bool, np.array):
        d = 0.0
        t = 0.0
        curv = 0.0
        # Generate warped frame
        warped_frame = self.projector.warp(frame)
        # Threshold warped frame to find yellow mid line
        thresh_frame_y = self.filter_y.process(warped_frame)
        thresh_frame_w = self.filter_w.process(warped_frame)
        # Try to fit a quadratic curve to the mid line
        line_fit, self.plot_image, _ = self.tracker.search(image_y=thresh_frame_y, image_w=thresh_frame_w, draw_windows=True)
        if (line_fit != np.zeros(3)).all():
            # Line is found
            self.line_found = True
            # Compute d and t
            a = line_fit[0]
            b = line_fit[1]
            c = line_fit[2]
            verbose = 0
            if verbose:
                if a>0:
                    print('turn right')
                else:
                    print('turn left')
            xy = []
            f = lambda x: int(a*x**2 + b*x + c)
            for y in range(0,480,20):
                point_on_line = np.array([f(y),y],np.int32)
                # t = np.arctan2(point_on_line[1],-point_on_line[0])
                # point_on_target = np.array(point_on_line - np.array([np.cos(t),np.sin(t)])*0,np.int32)
                xy.append(point_on_line)
                # cv2.circle(self.plot_image, tuple(np.array(point_on_target,np.int32)), 5, (0, 255, 0), -1)
                cv2.circle(self.plot_image, tuple(point_on_line), 10, (255, 0, 0), -1)
            
            xy = np.stack(xy)
            cv2.arrowedLine(self.plot_image, (xy[-1,0], xy[-1,1]),(xy[0,0], xy[0,1]), (0, 0, 255), 5)
            d = 320-xy[-1,0]
            t = np.array([xy[-1,0], xy[-1,1]]) - np.array([xy[0,0], xy[0,1]])
            t = np.arctan2(t[0], t[1])
        else:
            self.line_found = False
        return self.line_found, np.float32([d, t])

