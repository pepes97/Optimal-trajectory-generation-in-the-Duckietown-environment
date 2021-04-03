from ..video import *
from ..transform import FrenetDKTransform
from ..controller import FrenetIOLController


class Mapper():
    def __init__(self):
        self.projector       = PerspectiveWarper()
        self.yellow_filter   = CenterLineFilter()
        self.white_filter    = LateralLineFilter()
        self.red_filter      = RedFilter()
        self.tracker         = SlidingWindowDoubleTracker(robust_factor=1)
        self.filter_dict     = {'white': self.white_filter, 'yellow': self.yellow_filter, 'red': self.red_filter}
        self.mask_dict       = {'white': None, 'yellow': None, 'red': None}
        self.segmentator     = Segmentator()
        self.segment_dict    = {'white': None, 'yellow': None, 'red': None}
        self.semantic_mapper = SemanticMapper()
        self.lat_filter      = TrajectoryFilter(self.projector, self.yellow_filter, self.white_filter, self.tracker)
        # Adjust filters properties
        self.yellow_filter.yellow_thresh = (20, 35)
        self.yellow_filter.s_thresh = (65, 190)
        self.yellow_filter.l_thresh = (30, 255)
        self.obstacle_tracker = ObstacleTracker()
    
    def process_obstacles(self, frame):
        # Generate warped frame
        wframe = self.projector.warp(frame)
        for fkey in self.filter_dict.keys():
            self.mask_dict[fkey] = self.filter_dict[fkey].process(wframe)
        mask_t = cv2.bitwise_or(self.mask_dict['white'], self.mask_dict['yellow'])
        # Segmentize the masks
        self.segment_dict = self.segmentator.process(self.mask_dict)
        # Generate semantic dictionary
        object_dict, pfit, feat_dict  = self.semantic_mapper.process(self.segment_dict)
        # Apply obstacle tracking
        obstacles, all_obstacles = self.obstacle_tracker.process(object_dict)

        return wframe, obstacles, object_dict
    
    def process(self, frame):  
        wframe,obstacles, object_dict = self.process_obstacles(frame)
        # Obstacle mask generation START
        obstacle_mask = np.zeros_like(wframe, dtype='uint8')
        obstacle_contour_lst = []
        for obstacle in obstacles:
            obstacle_contour_lst.append(obstacle['contour'])
        obstacle_mask = cv2.drawContours(obstacle_mask, obstacle_contour_lst, -1, (255, 255, 255),
                                         thickness=cv2.FILLED)
        obstacle_mask = cv2.bitwise_not(obstacle_mask)
        obstacle_mask = cv2.cvtColor(obstacle_mask, cv2.COLOR_BGR2GRAY)
        ret, obstacle_mask = cv2.threshold(obstacle_mask, 10, 255, cv2.THRESH_BINARY)
        wframe = cv2.bitwise_and(wframe, wframe, mask=obstacle_mask)
        # Obstacle mask generation END
        
        # Threshold warped frame to find yellow mid line
        thresh_frame_y = self.yellow_filter.process(wframe)
        thresh_frame_w = self.white_filter.process(wframe)
        # Try to fit a quadratic curve to the mid line
        line_fit, self.lat_filter.plot_image, offset, contours_midpt = self.tracker.search(image_y=thresh_frame_y, image_w=thresh_frame_w, draw_windows=True)
        #lane_offset = self.line_offset*offset
        lane_offset = offset//2

        observations = self.lat_filter.cam2rob(contours_midpt)
        self.lat_filter.line_found = True
        target = self.lat_filter.process_target(line_fit, lane_offset)            
        trajectory = self.lat_filter.build_trajectory(target)
        self.lat_filter.draw_path()
        # go back to street view
        self.lat_filter.plot_image = self.lat_filter.projector.iwarp(self.lat_filter.plot_image)
        return self.lat_filter.line_found, trajectory, observations

    
