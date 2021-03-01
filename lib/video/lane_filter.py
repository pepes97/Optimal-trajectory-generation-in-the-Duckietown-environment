"""line_filter.py
"""

import cv2
import numpy as np
import logging
from matplotlib import pyplot as plt
from .utils import *
from .binarize import *



class LaneFilter:
    """ Extracts from a video stream the relative offset (d) and angle (theta) of the camera with respect to the lane.
    
    Based on https://www.hackster.io/kemfic/curved-lane-detection-34f771
    Thanks to kemfic
    """
    def __init__(self):
        self.grass_h_thresh = (64, 81)
        self.binary_thresh = (100, 200)
        # Sliding window variables
        self.left_a, self.left_b, self.left_c = [], [], []
        self.right_a, self.right_b, self.right_c = [], [], []
        pass

    def process(self, frame):
        def preproc_image(image):
            # Filter grass
            h, s, l = separate_hsl(image)
            grass_mask = 255 - cv2.inRange(h, self.grass_h_thresh[0], self.grass_h_thresh[1])
            image = cv2.bitwise_and(image, image, mask=grass_mask)
            # Binary threshold
            image = binary_threshold_gray(image, self.binary_thresh)
            return image
        preproc_frame = preproc_image(frame)
        # Apply Perspective transform
        """
        np.float32([
            (0.2343, 0.4166),
            (0, 0.7292),
            (0.99, 0.7292),
            (0.7656, 0.4166)])
        """
        def perspective_warp(image,
                             dest_size=(640, 480),
                             src=np.float32([
                                 (0.3859, 0.4292),
                                 (0.0016, 0.9729),
                                 (0.9969, 0.9458),
                                 (0.6109, 0.4292)]),
                             dest=np.float32([(0.3, 0), (0.3, 1), (0.7, 1), (0.7, 0)])):
            image_size = np.float32([(image.shape[1], image.shape[0])])
            src = src * image_size
            dest = dest * np.float32(dest_size)
            M = cv2.getPerspectiveTransform(src, dest)
            warped = cv2.warpPerspective(image, M, dest_size)
            return warped
        def perspective_iwarp(image,
                              dest_size=(640, 480),
                              src=np.float32([(0.3, 0), (0.3, 1), (0.7, 1), (0.7, 0)]),
                              dest=np.float32([
                                 (0.3859, 0.4292),
                                 (0.0016, 0.9729),
                                 (0.9969, 0.9458),
                                 (0.6109, 0.4292)])):
            image_size = np.float32([(image.shape[1], image.shape[0])])
            src = src * image_size
            dest = dest * np.float32(dest_size)
            M = cv2.getPerspectiveTransform(src, dest)
            warped = cv2.warpPerspective(image, M, dest_size)
            return warped
        
        warped_frame = perspective_warp(preproc_frame)
        # Apply lane detection
        def sliding_window(image, nwindows=9, margin=80, minpix = 1, draw_windows=True):
            left_fit_ = np.empty(3)
            right_fit_ = np.empty(3)
            out_image = np.dstack((image, image, image)) * 255
            # Find peaks of left and right halves (Histogram)
            histogram = get_hist(image)
            midpoint = int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            # Set windows height
            window_height = np.int(image.shape[0]/nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = image.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current position to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            # Create empty lists to receive left and right lane pixels indices
            left_lane_inds = []
            right_lane_inds = []
            # Step through the windows one by one
            for window in range(nwindows):
                # Get window boundaries in x and y
                win_y_low = image.shape[0] - (window+1) * window_height
                win_y_high = image.shape[0] - window * window_height
                win_xleft_low  = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low  = rightx_current - margin
                win_xright_high = rightx_current + margin
                if draw_windows:
                    cv2.rectangle(out_image, (win_xleft_low, win_y_low),
                                  (win_xleft_high, win_y_high), (100, 255, 255), 3)
                    cv2.rectangle(out_image, (win_xright_low, win_y_low),
                                  (win_xright_high, win_y_high), (100, 255, 255), 3)
                # Identify nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) &
                                  (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) &
                                  (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) &
                                   (nonzeroy < win_y_high) &
                                   (nonzerox >= win_xright_low) &
                                   (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            # Wtf is going on here ?
            self.left_a.append(left_fit[0])
            self.left_b.append(left_fit[1])
            self.left_c.append(left_fit[2])
            self.right_a.append(right_fit[0])
            self.right_b.append(right_fit[1])
            self.right_c.append(right_fit[2])
            left_fit_[0] = np.mean(self.left_a[-10:])
            left_fit_[1] = np.mean(self.left_b[-10:])
            left_fit_[2] = np.mean(self.left_c[-10:])
            right_fit_[0] = np.mean(self.right_a[-10:])
            right_fit_[1] = np.mean(self.right_b[-10:])
            right_fit_[2] = np.mean(self.right_c[-10:])

            # Generate x and y values for plotting
            ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
            left_fitx = left_fit_[0] * ploty**2 + left_fit_[1] * ploty + left_fit_[2]
            right_fitx = right_fit_[0] * ploty**2 + right_fit_[1] * ploty + right_fit_[2]
            out_image[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
            out_image[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

            return out_image, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

        
        out_image, fit, fit_, ploty = sliding_window(warped_frame)
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(frame[:,:,::-1])
        # axs[0, 1].imshow(perspective_warp(frame[:,:,::-1]))
        # axs[1, 0].imshow(perspective_iwarp(out_image))
        # axs[1, 1].imshow(out_image)
        # plt.show()
        return perspective_iwarp(out_image)

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

class YellowLaneFilter:
    def __init__(self):
        # Yellow of line is between 14 and 22 in the H channel
        self.yellow_threshold = (14, 22)
        
    def process(self, frame):
        # Extract yellow component
        h, s, l = separate_hsl(cv2.blur(frame, (7, 7)))
        h_mask = cv2.inRange(h, self.yellow_threshold[0], self.yellow_threshold[1])
        frame = cv2.bitwise_and(frame, frame, mask=h_mask)
        return h_mask
        
        
        
        
class CenterLineFilter:
    def __init__(self):
        # Yellow of line is between 14 and 22 in h channel
        self.yellow_thresh = (14, 22)
        self.s_thresh = (100, 130)
        pass
    def process(self, frame):
        def preproc(image):
            # Extract yellow info
            h, s, l = separate_hsl(cv2.blur(frame, (7, 7)))
            h_mask = cv2.inRange(h, self.yellow_thresh[0], self.yellow_thresh[1])
            s_mask = cv2.inRange(s, self.s_thresh[0], self.s_thresh[1])
            preproc_frame = cv2.bitwise_and(frame, frame, mask=h_mask)
            return cv2.bitwise_and(h_mask, h_mask, mask=s_mask)
        
        return preproc(frame)

class SlidingWindowTracker:
    def __init__(self):
        self.coeff_a = []
        self.coeff_b = []
        self.coeff_c = []

    def search(self, image, no_windows: int=9, margin: int=50, min_pix: int=1, draw_windows=False):
        test_image = np.dstack((image, image, image)) * 255
        fit_params = np.empty(3)
        # Compute histogram
        histogram = get_hist(image)
        line_base = np.argmax(histogram)
        # Set window height
        window_height = np.int(image.shape[0] / no_windows)
        current_base = line_base
        # Get non zero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
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
        lane_x = nonzerox[lane_pixels_inds]
        lane_y = nonzeroy[lane_pixels_inds]
        if lane_x.size == 0 or lane_y.size == 0:
            return np.zeros(3)
        lane_fit = np.polyfit(lane_y, lane_x, 2)
        self.coeff_a.append(lane_fit[0])
        self.coeff_b.append(lane_fit[1])
        self.coeff_c.append(lane_fit[2])
        fit_params[0] = np.mean(self.coeff_a[-1:])
        fit_params[1] = np.mean(self.coeff_b[-1:])
        fit_params[2] = np.mean(self.coeff_c[-1:])
        return test_image, lane_fit

class SlidingWindowDoubleTracker:
    # TODO
    def __init__(self):
        pass
    
