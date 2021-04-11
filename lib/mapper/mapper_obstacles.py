from ..video import *
from ..video import *

COLORS_LIST_TUPLE = [tuple(np.random.randint(low = 0, high = 255, size=(3,)).tolist()) for i in range(500)]

OBJ_COLOR_DICT = {
    ObjectType.UNKNOWN: (0, 128, 128),      # medium dark turquoise
    ObjectType.YELLOW_LINE: (255, 255, 0),  # yellow
    ObjectType.WHITE_LINE:  (255, 255, 255),# white  
    ObjectType.DUCK: (0, 255, 255),         # light blue
    ObjectType.CONE: (255, 0, 0),           # red
    ObjectType.ROBOT: (0, 0, 128),          # dark blue
    ObjectType.WALL: (255, 0, 255),         # fuchsia
    ObjectType.RIGHT_LINE:(220, 139, 0),    # orange
    ObjectType.LEFT_LINE:  (122, 0, 174)    # violet
}

class MapperSemanticObstacles():
    def __init__(self):
        self.projector       = PerspectiveWarper()
        self.yellow_filter   = CenterLineFilter()
        self.yellow_filter.yellow_thresh = (20, 35)
        self.yellow_filter.s_thresh = (65, 190)
        self.yellow_filter.l_thresh = (30, 255)
        self.white_filter    = LateralLineFilter()
        self.red_filter      = RedFilter()
        self.filter_dict     = {'white': self.white_filter, 'yellow': self.yellow_filter, 'red': self.red_filter}
        self.mask_dict       = {'white': None, 'yellow': None, 'red': None}
        self.segmentator     = Segmentator()
        self.segment_dict    = {'white': None, 'yellow': None, 'red': None}
        self.semantic_mapper = SemanticMapper()
        self.obstacle_tracker = ObstacleTracker()
        self.minimum_pixels = 1500
        self.robust_factor = 1
        self.line_offset_mean = []
        self.plot_image_w = None
        self.plot_image_p = None 
        self.max_planner = None
        self.trajectory_width = 0.21 #[m]
        self.white_tape = 0.048 #[m]
        self.yellow_tape = 0.024 #[m]
        self.line_offset = 60 #[px]
        self.pixel_ratio = (self.trajectory_width+ \
            self.yellow_tape/2+self.white_tape/2)/(self.line_offset*2) #[m/px] = 0.00082
        self.proj_planner = None
        self.path_planner = None
        self.all_paths = None
        self.K2R = np.array([[0, -self.pixel_ratio, 480*self.pixel_ratio],
                            [-self.pixel_ratio, 0, 320*self.pixel_ratio],
                            [0, 0, 1]],dtype=np.float64)
        self.R2K = np.linalg.inv(self.K2R)
    
    def get_line_offset(self, lane_fit_y, lane_fit_w):
        x = np.arange(0,480,20)
        offset_mean=[]
        #for p in x:
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
        offset_mean.append(line_offset)
        return np.mean(line_offset).astype(int)
    
    def process_obstacles(self, frame):
        # Generate warped frame
        wframe = self.projector.warp(frame)
        for fkey in self.filter_dict.keys():
            self.mask_dict[fkey] = self.filter_dict[fkey].process(wframe)
        mask_t = cv2.bitwise_or(self.mask_dict['white'], self.mask_dict['yellow'])
        # Segmentize the masks
        self.segment_dict = self.segmentator.process(self.mask_dict)
        # Mask contours
        mask_cont = np.zeros_like(wframe, dtype='uint8')
        # Generate semantic dictionary
        object_dict, pfit,yellow_midpts,rwfit,lwfit, offset_y,offset_w = self.semantic_mapper.process(self.segment_dict, mask_cont)
        # Apply obstacle tracking
        obstacles, all_obstacles = self.obstacle_tracker.process(object_dict)
        return wframe, obstacles, object_dict,pfit,yellow_midpts,rwfit,lwfit,offset_y,offset_w
    
    def obstacle_mask_generation(self, obstacles):
        # Obstacle mask generation START
        obstacle_mask = np.zeros_like(self.plot_image_w, dtype='uint8')
        obstacle_contour_lst = []
        for obstacle in obstacles:
            obstacle_contour_lst.append(obstacle['contour'])
        obstacle_mask = cv2.drawContours(obstacle_mask, obstacle_contour_lst, -1, (255, 255, 255),
                                         thickness=cv2.FILLED)
        obstacle_mask = cv2.bitwise_not(obstacle_mask)
        obstacle_mask = cv2.cvtColor(obstacle_mask, cv2.COLOR_BGR2GRAY)
        ret, obstacle_mask = cv2.threshold(obstacle_mask, 10, 255, cv2.THRESH_BINARY)
        self.plot_image_w = cv2.bitwise_and(self.plot_image_w, self.plot_image_w, mask=obstacle_mask)

        return self.plot_image_w

    def draw_obstacles_contours(self, object_dict):
        for obj_lst in object_dict.values():
            for object in obj_lst:
                cv2.drawContours(self.plot_image_p, object['contour'], -1, OBJ_COLOR_DICT[object['class']], 3)

    def draw_obstacles_bbox(self,obstacles):
        for object in obstacles:
            cv2.drawContours(self.plot_image_p, object['contour'], -1, OBJ_COLOR_DICT[object['class']], 3)
            x, y, w, h = object['bbox']
            # Draw distance line
            cv2.rectangle(self.plot_image_p, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.arrowedLine(self.plot_image_p, (320, 480), (int(x+w/2), int(y+h)), (255, 0, 0), 3)
            cv2.arrowedLine(self.plot_image_p, (320, 480), (int(x), int(y+h)), (0, 0, 255), 3)

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
    
    def process_target(self, line_fit, lane_offset):
        a, b, c = line_fit
        target = []
        xx = lambda y: int(a*y**2 + b*y + c)
        yy = np.arange(0,480,20)
        for i in range(yy.shape[0]-1):
            point_on_line = np.array([xx(yy[i]),yy[i]],np.int32)
            cv2.circle(self.plot_image_p, tuple(point_on_line), 5, (255, 0, 0), -1)
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
            cv2.circle(self.plot_image_p, tuple(point_on_target), 5, (0, 255, 0), -1)
        return np.array(target)

    def build_trajectory(self, target):
        trajectory = Trajectory()
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
        return trajectory

    def draw_path(self, verbose = 0):
        if self.all_paths!=None and verbose:
            for i,path in enumerate(self.all_paths):
                points = self.rob2cam(np.c_[path.x,path.y], int_type=True)
                for p in points:
                    cv2.circle(self.plot_image_p, tuple(p), 2, COLORS_LIST_TUPLE[i], -1)
        if np.array(self.path_planner!=None).all():
            path = self.rob2cam(self.path_planner, int_type=True)
            for p in path:
                cv2.circle(self.plot_image_p, tuple(p), 5, (0, 0, 255), -1)
        if np.array(self.proj_planner!=None).all():
            proj = self.rob2cam(self.proj_planner[None])[0]
            robot_p = np.array([0.1,0.0])
            rob = self.rob2cam(robot_p[None])[0]
            cv2.circle(self.plot_image_p, (rob[0], rob[1]), 10, (255, 0, 0), -1)
            cv2.arrowedLine(self.plot_image_p, (rob[0], rob[1]), (proj[0], proj[1]), (255, 0, 0), 5) # distance to projection
        
        

    def search(self, pfit, rwfit, lwfit, offset_y, offset_w, yellow_midpts):
        if (np.count_nonzero(self.mask_dict["yellow"]) < self.minimum_pixels and np.count_nonzero(self.mask_dict["white"]) < self.minimum_pixels):
            pass
        else:
            with warnings.catch_warnings(record=True) as w:
                w = list(filter(lambda i: issubclass(i.category, np.RankWarning), w))
                if len(w): 
                    if rwfit is not None:
                        pfit = rwfit
                    elif lwfit is not None:
                        pfit = lwfit
                    offset_y = offset_w
            if rwfit is not None:
                line_offset = self.get_line_offset(pfit, rwfit)
            elif lwfit is not None:
                line_offset = self.get_line_offset(pfit, lwfit)
            else:
                line_offset = self.get_line_offset(pfit, pfit)
            if np.count_nonzero(self.mask_dict["yellow"]) < self.minimum_pixels:
                if rwfit is not None:
                    line_fit = rwfit
                else:
                    line_fit = lwfit
                offset = offset_w
            else:
                print(yellow_midpts)
                if yellow_midpts is None and rwfit is not None:
                    line_fit = rwfit
                    offset = offset_w
                else:
                    line_fit = pfit
                    offset = offset_y
            self.line_offset_mean.append(line_offset)
        line_offset = np.mean(self.line_offset_mean[-self.robust_factor:])
        return line_fit, offset

    def lateal_polyfit_cam2rob(self, trajectory, verbose = 0):
        rw = []
        lw = []
        offset_r = self.line_offset//4*self.pixel_ratio
        offset_l = -3*self.line_offset*self.pixel_ratio
        for i in np.arange(0.0,480*self.pixel_ratio,0.05):
            target_pos = trajectory.compute_pt(i) + \
                compute_ortogonal_vect(trajectory, i) * offset_r
            rw.append(target_pos)
            target_pos = trajectory.compute_pt(i) + \
                compute_ortogonal_vect(trajectory, i) * offset_l
            lw.append(target_pos)
        rw = np.array(rw)
        lw = np.array(lw)
        if verbose:
            rww = self.rob2cam(rw)
            lww = self.rob2cam(lw)
            for rp,lp in zip(rww,lww):
                cv2.circle(self.plot_image_p, tuple(rp), 5, (0, 0, 255), -1)
                cv2.circle(self.plot_image_p, tuple(lp), 5, (0, 0, 255), -1)
        rw = np.polyfit(rw[:,0],rw[:,1],2)
        lw = np.polyfit(lw[:,0],lw[:,1],2)
        return rw,lw

    def process(self, frame, verbose = 0):  
        self.plot_image_w, obstacles, object_dict,pfit,yellow_midpts,rwfit,lwfit,offset_y,offset_w = self.process_obstacles(frame)
        # Mask Generation for obstacles
        self.plot_image_w = self.obstacle_mask_generation(obstacles)
        # Frame projection
        self.plot_image_p = np.zeros((self.plot_image_w.shape[0], self.plot_image_w.shape[1], 3), dtype='uint8')
        # Draw contour of things that you detect ( white lines, yellow line, duck, etc.)
        self.draw_obstacles_contours(object_dict)
        # Draw bbox obstacles
        self.draw_obstacles_bbox(obstacles)
        # Line fit and offset
        line_fit, offset = self.search(pfit,rwfit,lwfit,offset_y,offset_w, yellow_midpts)
        # lane_offset = offset//2
        lane_offset = offset*self.line_offset
        # Set true line found
        self.line_found = True  
        # Take the target
        target = self.process_target(line_fit, lane_offset) 
        # Compute trajectory
        trajectory = self.build_trajectory(target)
        # Draw path
        self.draw_path(verbose = verbose)        
        rw, lw = self.lateal_polyfit_cam2rob(trajectory,verbose = verbose)

        return self.line_found, trajectory, obstacles, rwfit, lwfit, rw, lw

    
