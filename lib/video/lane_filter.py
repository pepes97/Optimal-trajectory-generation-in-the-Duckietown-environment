"""line_filter.py
"""

import cv2
import numpy as np
import logging
from matplotlib import pyplot as plt
import functools
from .utils import *
from .binarize import *
from ..trajectory import Trajectory

class CenterLineFilter:
    def __init__(self):
        # Yellow of line is between 14 and 22 in h channel
        self.yellow_thresh = (19, 24)
        self.s_thresh = (80, 150)
        self.l_thresh = (0, 255)
        pass
    def process(self, frame):
        def preproc(image):
            # Extract yellow info
            h, s, l = separate_hsl(frame)#cv2.blur(frame, (3, 3)))
            h_mask = cv2.inRange(h, self.yellow_thresh[0], self.yellow_thresh[1])
            s_mask = cv2.inRange(s, self.s_thresh[0], self.s_thresh[1])
            l_mask = cv2.inRange(l, self.l_thresh[0], self.l_thresh[1])
            #preproc_frame = cv2.bitwise_and(frame, frame, mask=h_mask)
            mask = cv2.bitwise_and(h_mask, h_mask, mask=s_mask)
            mask = cv2.bitwise_and(mask, mask, mask=l_mask)
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
        self.last_seen_yellow_pos = [np.array([320,480])]
        self.max_line_offset = 150
        self.line_offset_mean = []

    def search(self, image_y, image_w, no_windows: int=9, margin: int=50, min_pix: int=1, draw_windows=False):
        test_image_y = np.dstack((image_y, image_y, image_y))
        test_image_w = np.dstack((image_w, image_w, image_w))
        offset = 1
        contours_midpt = np.array([]).reshape(0,2)
        line_offset = self.max_line_offset

        if np.count_nonzero(image_y) < self.minimum_pixels and np.count_nonzero(image_w) < self.minimum_pixels:
            lane_fit = np.zeros(3)
        else:
            lane_fit_w, offset_w = self.white_line_fit(image_w, test_image_w, draw_windows)
            lane_fit_y, offset_y, contours_midpt = self.yellow_line_fit(image_y, test_image_y, draw_windows)
            if np.count_nonzero(image_y) < self.minimum_pixels:
                lane_fit = lane_fit_w
                offset = offset_w
            else:
                lane_fit = lane_fit_y
                offset = offset_y
            # we want to estimate the distance between the two parabola
            # by looking at it's coefficients
            if np.isfinite(lane_fit_y).all():
                a,b,c = lane_fit_y[0],lane_fit_y[1],lane_fit_y[2]
                f = lambda x: int(a*x**2 + b*x + c)
                p = 240 # query point
                dirr = np.array([f(p-1),p-1],dtype=np.float32) - np.array([f(p+1),p+1],dtype=np.float32)
                normal = dirr[::-1] * np.r_[1.,-1.]
                normal = normal / np.linalg.norm(normal)
                point = np.array([f(p),p],dtype=np.float32)
                points_to_line = lambda p1,p2: ((p1[1]-p2[1])/(p1[0]-p2[0]),(p1[0]*p2[1]-p2[0]*p1[1])/(p1[0]-p2[0]))
                m, q = points_to_line(point, point+normal)
                a,b,c = lane_fit_w[0],lane_fit_w[1],lane_fit_w[2]
                point_on_white = np.roots([c-q,b-m,a])
                line_offset = np.linalg.norm(point-point_on_white).astype(int)//2

            self.line_offset_mean.append(line_offset)
        
            self.coeff_a.append(lane_fit[0])
            self.coeff_b.append(lane_fit[1])
            self.coeff_c.append(lane_fit[2])
        
        def reject_outliers(data, m=10):
                return data[abs(data - np.mean(data)) < m * np.std(data)]

        line_offset = np.mean(self.line_offset_mean[-self.robust_factor:])
        lane_fit[0] = np.mean(self.coeff_a[-self.robust_factor:])
        lane_fit[1] = np.mean(self.coeff_b[-self.robust_factor:])
        lane_fit[2] = np.mean(self.coeff_c[-self.robust_factor:])
        
        test_image = test_image_y+test_image_w     
        # last_seen_yellow_pos = np.mean(self.last_seen_yellow_pos[-self.robust_factor:],axis=0) 
        
        return lane_fit, test_image, offset * line_offset, contours_midpt

    def yellow_line_fit(self, image_y, test_image_y, draw_windows):
        # Find Contours yellow
        contours_y, _ = cv2.findContours(image_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_y = list(filter(lambda ctr: cv2.contourArea(ctr) > self.contour_min_area, contours_y))
        contours_midpt = []
        offset = 1
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
            contours_midpt = np.array([]).reshape(0,2)
        else:
            self.last_seen_yellow_pos.append(np.mean(contours_midpt,axis=0))
            lane_x = contours_midpt[:, 0]
            lane_y = contours_midpt[:, 1]
            lane_fit = np.polyfit(lane_y, lane_x, 2)
            
        return lane_fit, offset, contours_midpt
    
    def white_line_fit(self,image_w,test_image_w,draw_windows):
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
            offset = 1
        else:
            if len(right)>0:
                points = right
                contour = contour_right
                offset = - 1
            else:
                points = left
                contour = contour_left
                offset = 1
            if draw_windows:
                cv2.drawContours(test_image_w, contour, -1, (0, 255, 0), 3)
            for i in range(len(contour)):
                mdpt = np.array(np.mean(contour,axis=(0,1)),np.int32)
                cv2.circle(test_image_w, (mdpt[0], mdpt[1]), 2, (255, 0, 0), -1)
            lane_x = points[:, 0]
            lane_y = points[:, 1]
            lane_fit = np.polyfit(lane_y, lane_x, 2)
        return lane_fit, offset
            

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

# class TrajectoryFilter():
#     """ Finds and tracks the middle dashed yellow line.
#     If the line is found and verified, then it returns the best quadratic fit (in lane space), the
#     camera offset (d) and inclination (theta~)
#     """
#     def __init__(self, projector, filter_y, filter_w, tracker, trasformer):
#         self.projector = projector
#         self.filter_y = filter_y
#         self.filter_w = filter_w
#         self.tracker = tracker
#         self.line_found = False
#         self.plot_image = None
#         self.trajectory_width = 0.21 #[m]
#         self.white_tape = 0.048 #[m]
#         self.yellow_tape = 0.024 #[m]
#         self.pixel_ratio = 0.00082 #[px/m]
#         self.planner = None
#         self.transformer = trasformer
#         self.K2R = np.array([[0, -self.pixel_ratio, 480*self.pixel_ratio],
#                             [-self.pixel_ratio, 0, 320*self.pixel_ratio],
#                             [0, 0, 1]],dtype=np.float64)
#         self.R2K = np.linalg.inv(self.K2R)
#         self.K2W = np.array([[self.pixel_ratio, 0, 0],
#                             [0, -self.pixel_ratio, 480*self.pixel_ratio],
#                             [0, 0, 1]],dtype=np.float64)
#         self.W2K = np.linalg.inv(self.K2W)
            
#     def rob2cam(self, robot_points: np.ndarray) -> np.ndarray: 
#         # homogeneous vector
#         robot_points = np.c_[robot_points, np.ones(robot_points.shape[0])] 
#         # homogeneous matrix
#         camera_points = (self.R2K @ robot_points.T).T
#         # euclidean vector
#         camera_points = camera_points[:,:-1]/camera_points[:,-1:]
#         # invert order to have min y in first position
#         camera_points = camera_points[::-1]
#         return camera_points.round().astype(int)

#     def cam2rob(self, camera_points: np.ndarray) -> np.ndarray: 
#         # homogeneous vector
#         camera_points = np.c_[camera_points, np.ones(camera_points.shape[0])] 
#         # homogeneous matrix
#         robot_points = (self.K2R @ camera_points.T).T
#         # euclidean vector
#         robot_points = robot_points[:,:-1]/robot_points[:,-1:]
#         # invert order to have min y in first position
#         robot_points = robot_points[::-1]
#         return robot_points
    
#     # def cam2world(self, camera_points: np.ndarray) -> np.ndarray: 
#     #     # homogeneous vector
#     #     camera_points = np.c_[camera_points, np.ones(camera_points.shape[0])] 
#     #     # homogeneous matrix
#     #     world_points = (self.K2W @ camera_points.T).T
#     #     # euclidean vector
#     #     world_points = world_points[:,:-1]
#     #     # invert order to have min y in first position
#     #     world_points = world_points[::-1]
#     #     return world_points
    
#     # def world2cam(self, world_points: np.ndarray) -> np.ndarray: 
#     #     # homogeneous vector
#     #     world_points = np.c_[world_points, np.ones(world_points.shape[0])] 
#     #     # homogeneous matrix
#     #     camera_points = (self.W2K @ world_points.T).T
#     #     # euclidean vector
#     #     camera_points = camera_points[:,:-1]
#     #     # invert order to have min y in first position
#     #     camera_points = camera_points[::-1]
#     #     return camera_points.round().astype(int)

#     def process_target(self, line_fit, lane_offset):
#         a, b, c = line_fit
#         target = []
#         xx = lambda y: int(a*y**2 + b*y + c)
#         yy = np.arange(0,480,20)
#         for i in range(yy.shape[0]-1):
#             point_on_line = np.array([xx(yy[i]),yy[i]],np.int32)
#             cv2.circle(self.plot_image, tuple(point_on_line), 5, (255, 0, 0), -1)
#             diff = point_on_line - np.array([xx(yy[i+1]),yy[i+1]],np.int32)
#             th = np.arctan2(diff[1],diff[0])
#             dirr = np.array([-np.sin(th),np.cos(th)])
#             point_on_target = np.array(point_on_line + dirr * (lane_offset - int(self.yellow_tape/2*1/self.pixel_ratio)),np.int32)
#             target.append(point_on_target)
#         target = np.stack(target)
#         # sample again to get the full trajectory
#         a, b, c = np.polyfit(target[:,1], target[:,0], 2)
#         xx = lambda y: int(a*y**2 + b*y + c)
#         yy = np.arange(0,480,20)
#         target = []
#         for y in yy:
#             point_on_target = np.array([xx(y),y],np.int32)
#             target.append(point_on_target)
#             cv2.circle(self.plot_image, tuple(point_on_target), 5, (0, 255, 0), -1)
#         return np.array(target)
 
#     def process(self, frame) -> (bool, np.array):
#         d = 0.0
#         t = 0.0
#         curv = 0.0
#         # Generate warped frame
#         warped_frame = self.projector.warp(frame)
#         # Threshold warped frame to find yellow mid line
#         thresh_frame_y = self.filter_y.process(warped_frame)
#         thresh_frame_w = self.filter_w.process(warped_frame)
#         # Try to fit a quadratic curve to the mid line
#         line_fit, self.plot_image, lane_offset, contours_midpt = self.tracker.search(image_y=thresh_frame_y, image_w=thresh_frame_w, draw_windows=True)
#         lane_offset = 150
#         lane_width = abs(lane_offset)*2
#         # self.pixel_ratio = (self.trajectory_width+self.yellow_tape/2+self.white_tape/2)/lane_width #[m/px]
#         observations = self.cam2rob(contours_midpt)
#         if (line_fit != np.zeros(3)).all() and np.isfinite(line_fit).all():
#             # Line is found
#             self.line_found = True
#             # Compute d and t
#             target = self.process_target(line_fit, lane_offset)
#             target_planner = []
#             xy0 = xy1 = xy2 = np.array([0.0,0.0])
#             s0 = s1 = s2 = 0
#             ds = 0.01
#             if self.planner is not None:
#                 rob_target = self.cam2rob(np.stack(target))
#                 a, b, c = np.polyfit(rob_target[:,0], rob_target[:,1], 2)
#                 trajectory = Trajectory()
#                 trajectory.compute_pt = lambda x: np.array([x, a*x**2 + b*x + c])
#                 trajectory.compute_first_derivative = lambda x: np.array([x, 2*a*x + b])
#                 robot_reference_p = np.array([0.0,0.0])
#                 s0 = self.transformer.estimatePosition(trajectory, robot_reference_p)
#                 xy0 = trajectory.compute_pt(s0)
#                 s1 = s0+ds
#                 xy1 = trajectory.compute_pt(s1)
#                 s2 = s1+ds
#                 xy2 = trajectory.compute_pt(s2)
#                 points_planner = frenet_to_glob(trajectory, self.planner, s0)
#                 points_planner = np.array(points_planner)
#                 a, b, c = np.polyfit(points_planner[:,0], points_planner[:,1], 2)
#                 yy = lambda x: a*x**2 + b*x + c
#                 xx = np.arange(0,480,20) * self.pixel_ratio
#                 for x in xx:
#                     point_planner = np.array([x,yy(x)],dtype=np.float32)
#                     target_planner.append(point_planner)
#                 target_planner = self.rob2cam(np.array(target_planner))
#                 for points in target_planner:
#                     cv2.circle(self.plot_image, tuple(points), 5, (0, 0, 255), -1)
#                 cv2.circle(self.plot_image, tuple(target_planner[-1]), 10, (0, 0, 255), -1) # first planner point
#             else:
#                 target_planner = target
#             # move x axis to the center and y axis at the bottom
#             # from camera frame to world frame
#             # transform 
#             # distance from origin (0,0) in robot frame
#             d = np.sign(xy0[1])*np.linalg.norm(xy0)
#             t =  np.arctan2((xy1-xy0)[1], (xy1-xy0)[0])
#             t1 =  np.arctan2((xy2-xy1)[1], (xy2-xy1)[0])
#             curv = (t1-t) / (np.linalg.norm(xy2-xy1))
 
#             xy0p = self.rob2cam(np.array([xy0]))[0] # projection point
#             xy1p = self.rob2cam(np.array([xy1]))[0] # first planner trajectory point
#             xy2p = self.rob2cam(np.array([xy2]))[0]
#             cv2.circle(self.plot_image, tuple(xy0p), 10, (255, 0, 0), -1)
#             cv2.circle(self.plot_image, tuple(xy1p), 10, (0, 255, 0), -1)
#             cv2.circle(self.plot_image, tuple(xy2p), 10, (0, 0, 255), -1)
            
#             target = self.cam2rob(np.stack(target_planner))
#             index = 0
#             # t = target[index+1] - target[index]
#             # t =  np.arctan2(t[1], t[0]) #- np.pi/2
#             # # compute curvature
#             # tt = target[index+2] - target[index+1]
#             # t1 = np.arctan2(tt[1], tt[0]) # - np.pi/2
            
#             points = self.rob2cam(target)[::-1]
#             cv2.arrowedLine(self.plot_image, (320, 480), (320+(points[index+2,0]-points[index,0]), 480+(points[index+2,1]-points[index,1])), (0, 255, 0), 5)
#             cv2.arrowedLine(self.plot_image, (320, 480), (xy0p[0], xy0p[1]), (255, 0, 0), 5) # distance to projection
#         else:
#             self.line_found = False
        
#         # self.plot_image = self.projector.iwarp(self.plot_image) # go back to street view
#         return self.line_found, np.float32([d, t]), curv, observations

class RedFilter:
    def __init__(self, *args, **kwargs):
        pass

    def process(self, frame):
        return np.zeros((frame.shape[0], frame.shape[1], 1), dtype='uint8')


def frenet_to_glob(trajectory, planner,projection):
    frenet_path = planner.opt_path_tot
    path = []
    for i in range(len(frenet_path.s)):
        s = projection + (frenet_path.s[i] - planner.s0[0]) *0.1
        d = frenet_path.d[i] 
        target_pos = trajectory.compute_pt(s) + \
            compute_ortogonal_vect(trajectory, s) * d
        path.append(target_pos)
    path = np.array(path)
    return path


def compute_ortogonal_vect(trajectory, s):
        t_grad = trajectory.compute_first_derivative(s)
        t_r = np.arctan2(t_grad[1], t_grad[0])
        return np.array([np.sin(t_r), -np.cos(t_r)])



class TrajectoryFilter():
    """ Finds and tracks the middle dashed yellow line.
    If the line is found and verified, then it returns the best quadratic fit (in lane space), the
    camera offset (d) and inclination (theta~)
    """
    def __init__(self, projector, filter_y, filter_w, tracker, trasformer):
        self.projector = projector
        self.filter_y = filter_y
        self.filter_w = filter_w
        self.tracker = tracker
        self.line_found = False
        self.plot_image = None
        self.trajectory_width = 0.21 #[m]
        self.white_tape = 0.048 #[m]
        self.yellow_tape = 0.024 #[m]
        self.pixel_ratio = 0.00082 #[px/m]
        self.proj_planner = None
        self.path_planner = None
        self.transformer = trasformer
        self.K2R = np.array([[0, -self.pixel_ratio, 480*self.pixel_ratio],
                            [-self.pixel_ratio, 0, 320*self.pixel_ratio],
                            [0, 0, 1]],dtype=np.float64)
        self.R2K = np.linalg.inv(self.K2R)
        self.K2W = np.array([[self.pixel_ratio, 0, 0],
                            [0, -self.pixel_ratio, 480*self.pixel_ratio],
                            [0, 0, 1]],dtype=np.float64)
        self.W2K = np.linalg.inv(self.K2W)
            
    def rob2cam(self, robot_points: np.ndarray, int_type = True) -> np.ndarray: 
        # homogeneous vector
        robot_points = np.c_[robot_points, np.ones(robot_points.shape[0])] 
        # homogeneous matrix
        camera_points = (self.R2K @ robot_points.T).T
        # euclidean vector
        camera_points = camera_points[:,:-1]/camera_points[:,-1:]
        # invert order to have min y in first position
        camera_points = camera_points[::-1]
        if int_type:
            camera_points = camera_points.round().astype(int)
        return camera_points

    def cam2rob(self, camera_points: np.ndarray) -> np.ndarray: 
        # homogeneous vector
        camera_points = np.c_[camera_points, np.ones(camera_points.shape[0])] 
        # homogeneous matrix
        robot_points = (self.K2R @ camera_points.T).T
        # euclidean vector
        robot_points = robot_points[:,:-1]/robot_points[:,-1:]
        # invert order to have min y in first position
        robot_points = robot_points[::-1]
        return robot_points
    

    def process_target(self, line_fit, lane_offset):
        a, b, c = line_fit
        target = []
        xx = lambda y: int(a*y**2 + b*y + c)
        yy = np.arange(0,480,20)
        for i in range(yy.shape[0]-1):
            point_on_line = np.array([xx(yy[i]),yy[i]],np.int32)
            cv2.circle(self.plot_image, tuple(point_on_line), 5, (255, 0, 0), -1)
            diff = point_on_line - np.array([xx(yy[i+1]),yy[i+1]],np.int32)
            th = np.arctan2(diff[1],diff[0])
            dirr = np.array([-np.sin(th),np.cos(th)])
            point_on_target = np.array(point_on_line + dirr * (lane_offset - int(self.yellow_tape/2*1/self.pixel_ratio)),np.int32)
            target.append(point_on_target)
        target = np.stack(target)
        # sample again to get the full trajectory
        a, b, c = np.polyfit(target[:,1], target[:,0], 2)
        xx = lambda y: int(a*y**2 + b*y + c)
        yy = np.arange(0,480,20)
        target = []
        for y in yy:
            point_on_target = np.array([xx(y),y],np.int32)
            target.append(point_on_target)
            cv2.circle(self.plot_image, tuple(point_on_target), 5, (0, 255, 0), -1)
        return np.array(target)
 
    def process(self, frame) -> (bool, np.array):
        # Generate warped frame
        warped_frame = self.projector.warp(frame)
        # Threshold warped frame to find yellow mid line
        thresh_frame_y = self.filter_y.process(warped_frame)
        thresh_frame_w = self.filter_w.process(warped_frame)
        # Try to fit a quadratic curve to the mid line
        line_fit, self.plot_image, lane_offset, contours_midpt = self.tracker.search(image_y=thresh_frame_y, image_w=thresh_frame_w, draw_windows=True)
        lane_offset = 150*np.sign(lane_offset)
        lane_width = abs(lane_offset)*2
        # self.pixel_ratio = (self.trajectory_width+self.yellow_tape/2+self.white_tape/2)/lane_width #[m/px]
        observations = self.cam2rob(contours_midpt)

        trajectory = Trajectory()

        if (line_fit != np.zeros(3)).all() and np.isfinite(line_fit).all():
            # Line is found
            self.line_found = True
            # Compute d and t
            target = self.process_target(line_fit, lane_offset)
            
            target = self.cam2rob(np.array(target))
            
            a, b, c = np.polyfit(target[:,0], target[:,1], 2)

            trajectory.compute_pt = lambda x: np.array([x, a*x**2 + b*x + c])

            trajectory.compute_first_derivative = lambda x: np.array([1, 2*a*x + b])
            
            trajectory.compute_second_derivative = lambda x: np.array([0, 2*a])

            def compute_curvature(trajectory, t):
                dt = 1/30
                x0, x1, x2 = t - dt, t, t + dt
                y0, y1, y2 = trajectory.compute_pt(x0)[1], trajectory.compute_pt(x1)[1], \
                    trajectory.compute_pt(x2)[1]
                t1 = np.arctan2(np.sin(y1-y0), np.cos(x1-x0))
                t2 = np.arctan2(np.sin(y2-y1), np.cos(x2-x1))
                k = np.abs(t2-t1)/np.linalg.norm(np.array([x1-x0,y1-y0]))
                return k
            
            trajectory.compute_curvature = lambda x: compute_curvature(trajectory, x)

            if np.array(self.proj_planner!=None).all():
                proj = self.rob2cam(self.proj_planner[None])[0]
                cv2.circle(self.plot_image, tuple(proj), 10, (255, 0, 0), -1)
                cv2.circle(self.plot_image, (320, 480), 10, (255, 0, 0), -1)
                cv2.arrowedLine(self.plot_image, (320, 480), (proj[0], proj[1]), (255, 0, 0), 5) # distance to projection
            
            if np.array(self.path_planner!=None).all():
                if (self.path_planner<3).all():
                    path = self.rob2cam(self.path_planner, int_type=False)
                    aa, bb, cc = np.polyfit(path[:,1], path[:,0], 2)
                    xxx = lambda y: int(aa*y**2 + bb*y + cc)
                    yyy = np.arange(0,480,20)
                    for y in yyy:
                        pts = np.array([xxx(y),y],np.int32)
                        cv2.circle(self.plot_image, tuple(pts), 5, (0, 0, 255), -1)

        else:
            self.line_found = False
        
        self.plot_image = self.projector.iwarp(self.plot_image) # go back to street view
        return self.line_found, trajectory, observations
